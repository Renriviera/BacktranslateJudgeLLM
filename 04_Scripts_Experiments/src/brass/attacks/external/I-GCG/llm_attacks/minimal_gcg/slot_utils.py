"""SlotGCG utilities: attention-based Vulnerable Slot Score and interleaved
token optimisation primitives.

Ported from https://github.com/youai058/SlotGCG (ICLR 2026) and adapted to
work with the existing ``llm_attacks`` infrastructure.
"""

from __future__ import annotations

import gc
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Position allocation
# ---------------------------------------------------------------------------

def generate_positions(attention_probs: torch.Tensor, num_adv_tokens: int) -> torch.Tensor:
    """Allocate *num_adv_tokens* across slots proportional to *attention_probs*."""
    n_slots = len(attention_probs)
    raw = attention_probs * num_adv_tokens
    tokens = raw.floor().int()
    remaining = int(num_adv_tokens - tokens.sum().item())

    if remaining > 0:
        fractional_parts = raw - tokens.float()
        full_rounds = remaining // n_slots
        if full_rounds > 0:
            tokens += full_rounds
            remaining -= full_rounds * n_slots
        if remaining > 0:
            _, top_indices = torch.topk(fractional_parts, remaining)
            tokens[top_indices] += 1

    return torch.repeat_interleave(
        torch.arange(n_slots, device=tokens.device),
        tokens,
    )


def insert_optim_embed_pos(
    behavior_embeds: torch.Tensor,
    optim_embeds: torch.Tensor,
    insert_pos: torch.Tensor | list[int],
) -> torch.Tensor:
    """Interleave *optim_embeds* into *behavior_embeds* at *insert_pos*."""
    assert behavior_embeds.shape[0] == 1
    assert optim_embeds.shape[0] == 1
    assert optim_embeds.shape[1] == len(insert_pos)

    if isinstance(insert_pos, torch.Tensor):
        insert_pos = insert_pos.tolist()

    behavior_seq = behavior_embeds[0]
    optim_seq = optim_embeds[0]

    insert_map: dict[int, list[int]] = {}
    for i, p in enumerate(insert_pos):
        insert_map.setdefault(p, []).append(i)

    parts: list[torch.Tensor] = []
    for i in range(behavior_seq.shape[0] + 1):
        if i in insert_map:
            for optim_i in insert_map[i]:
                parts.append(optim_seq[optim_i : optim_i + 1])
        if i < behavior_seq.shape[0]:
            parts.append(behavior_seq[i : i + 1])

    return torch.cat([t.unsqueeze(0) for t in parts], dim=1)


def interleave_behavior_and_controls(
    behavior_ids: torch.Tensor,
    optim_ids: torch.Tensor,
    insert_pos_batch: torch.Tensor | list[list[int]],
) -> torch.Tensor:
    """Batched token-ID interleaving for candidate evaluation."""
    B = behavior_ids.size(0)
    interleaved: list[torch.Tensor] = []

    if isinstance(insert_pos_batch, torch.Tensor):
        insert_pos_batch = insert_pos_batch.tolist()

    for b in range(B):
        behavior_seq = behavior_ids[b]
        optim_seq = optim_ids[b]
        insert_pos = insert_pos_batch[b]

        insert_map: dict[int, list[int]] = {}
        for i, p in enumerate(insert_pos):
            insert_map.setdefault(p, []).append(i)

        parts: list[torch.Tensor] = []
        for i in range(behavior_seq.size(0) + 1):
            if i in insert_map:
                for optim_i in insert_map[i]:
                    parts.append(optim_seq[optim_i : optim_i + 1])
            if i < behavior_seq.size(0):
                parts.append(behavior_seq[i : i + 1])

        interleaved.append(torch.cat(parts, dim=0).unsqueeze(0))

    max_len = max(x.size(1) for x in interleaved)
    padded = torch.zeros(B, max_len, dtype=behavior_ids.dtype, device=behavior_ids.device)
    for i, row in enumerate(interleaved):
        padded[i, : row.size(1)] = row

    return padded


# ---------------------------------------------------------------------------
# Vulnerable Slot Score (VSS) computation
# ---------------------------------------------------------------------------

def compute_vss(
    model,
    tokenizer,
    template_before: str,
    template_after: str,
    behavior: str,
    target: str,
    num_adv_tokens: int = 20,
    attention_temp: float = 8.0,
    use_prefix_cache: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the Vulnerable Slot Score for all insertion positions."""
    device = model.device
    embed_layer = model.get_input_embeddings()
    vocab_size = embed_layer.weight.shape[0]
    vocab_embeds = embed_layer(torch.arange(vocab_size, device=device)).detach()

    orig_attn = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = "eager"
    for module in model.modules():
        if hasattr(module, "_attn_implementation"):
            module._attn_implementation = "eager"

    before_ids = torch.tensor(
        tokenizer(template_before, padding=False).input_ids, device=device
    ).unsqueeze(0)
    behavior_ids = torch.tensor(
        tokenizer(behavior, padding=False, add_special_tokens=False).input_ids,
        device=device,
    ).unsqueeze(0)
    after_ids = torch.tensor(
        tokenizer(template_after, padding=False, add_special_tokens=False).input_ids,
        device=device,
    ).unsqueeze(0)
    target_ids = torch.tensor(
        tokenizer(target, padding=False, add_special_tokens=False).input_ids,
        device=device,
    ).unsqueeze(0)

    before_embeds = embed_layer(before_ids).to(model.dtype)
    behavior_embeds = embed_layer(behavior_ids).to(model.dtype)
    after_embeds = embed_layer(after_ids).to(model.dtype)
    target_embeds = embed_layer(target_ids).to(model.dtype)

    prefix_cache = None
    if use_prefix_cache:
        with torch.no_grad():
            out = model(inputs_embeds=before_embeds, use_cache=True)
            prefix_cache = out.past_key_values

    num_positions = behavior_ids.shape[1] + 1
    insert_positions = list(range(num_positions))

    dummy_id = torch.zeros(1, num_positions, dtype=torch.long, device=device)
    dummy_onehot = torch.zeros(1, num_positions, vocab_size, device=device, dtype=model.dtype)
    dummy_onehot.scatter_(2, dummy_id.unsqueeze(2), 1.0)
    dummy_embeds = torch.matmul(dummy_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

    optim_spread_pos = torch.tensor(insert_positions, dtype=torch.long, device=device)
    inserted = insert_optim_embed_pos(behavior_embeds, dummy_embeds, optim_spread_pos)

    with torch.no_grad():
        if use_prefix_cache:
            input_embeds = torch.cat([inserted, after_embeds, target_embeds], dim=1).to(model.dtype)
            outputs = model(
                inputs_embeds=input_embeds,
                past_key_values=prefix_cache,
                output_attentions=True,
            )
        else:
            input_embeds = torch.cat(
                [before_embeds, inserted, after_embeds, target_embeds], dim=1
            ).to(model.dtype)
            outputs = model(inputs_embeds=input_embeds, output_attentions=True)

    total_layers = len(outputs.attentions)
    upper_start = total_layers // 2
    attentions = torch.stack(outputs.attentions[upper_start:]).sum(dim=(0, 2))

    actual_positions = [pos + i for i, pos in enumerate(insert_positions)]

    batch_size_attn, query_len, key_len = attentions.shape
    cached_len = key_len - query_len
    before_len = before_embeds.shape[1]
    inserted_len = inserted.shape[1]
    after_len = after_embeds.shape[1]

    if use_prefix_cache:
        query_after_start = inserted_len
        query_after_end = inserted_len + after_len
        key_inserted_start = cached_len
        key_inserted_end = cached_len + inserted_len
    else:
        query_after_start = before_len + inserted_len
        query_after_end = before_len + inserted_len + after_len
        key_inserted_start = before_len
        key_inserted_end = before_len + inserted_len

    chat_positions = list(range(query_after_start, query_after_end))

    attention_weight = attentions[
        0, chat_positions, key_inserted_start:key_inserted_end
    ].sum(dim=0)[actual_positions]

    attention_probs = torch.softmax(attention_weight / attention_temp, dim=0)
    optim_pos = generate_positions(attention_probs, num_adv_tokens)

    del outputs, attentions, input_embeds, dummy_embeds
    gc.collect()
    torch.cuda.empty_cache()

    if orig_attn is not None:
        model.config._attn_implementation = orig_attn
        for module in model.modules():
            if hasattr(module, "_attn_implementation"):
                module._attn_implementation = orig_attn

    return optim_pos, attention_probs


# ---------------------------------------------------------------------------
# Gradient computation for interleaved adversarial tokens
# ---------------------------------------------------------------------------

def slot_token_gradients(
    model,
    embed_layer: nn.Module,
    vocab_embeds: torch.Tensor,
    behavior_embeds: torch.Tensor,
    after_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    target_ids: torch.Tensor,
    optim_ids: torch.Tensor,
    optim_pos: torch.Tensor,
    prefix_cache=None,
    before_embeds: Optional[torch.Tensor] = None,
    loss_fn=None,
) -> Tuple[torch.Tensor, float]:
    """Compute per-token gradients for interleaved adversarial tokens."""
    device = model.device
    num_optim = optim_ids.shape[1]
    vocab_size = vocab_embeds.shape[0]

    optim_ids_onehot = torch.zeros(
        1, num_optim, vocab_size, device=device, dtype=model.dtype
    )
    optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0)
    optim_ids_onehot.requires_grad_()

    optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)
    behavior_optim_embeds = insert_optim_embed_pos(behavior_embeds, optim_embeds, optim_pos)

    if prefix_cache is not None:
        input_embeds = torch.cat(
            [behavior_optim_embeds, after_embeds, target_embeds], dim=1
        )
        outputs = model(inputs_embeds=input_embeds, past_key_values=prefix_cache)
    else:
        assert before_embeds is not None
        input_embeds = torch.cat(
            [before_embeds, behavior_optim_embeds, after_embeds, target_embeds], dim=1
        )
        outputs = model(inputs_embeds=input_embeds)

    logits = outputs.logits
    target_len = target_embeds.shape[1]
    shift_logits = logits[..., -(target_len + 1) : -1, :].contiguous()
    shift_labels = target_ids

    if loss_fn is not None:
        per_sample = loss_fn(shift_logits, shift_labels)
        loss = per_sample.mean() if per_sample.dim() > 0 else per_sample
    else:
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    loss_val = loss.item()

    grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]
    grad = grad.squeeze(0)
    grad = grad / grad.norm(dim=-1, keepdim=True)

    del outputs, logits, input_embeds, optim_embeds, behavior_optim_embeds
    gc.collect()
    torch.cuda.empty_cache()

    return grad, loss_val


# ---------------------------------------------------------------------------
# Candidate sampling
# ---------------------------------------------------------------------------

def slot_sample_control(
    control_toks: torch.Tensor,
    grad: torch.Tensor,
    search_width: int,
    topk: int = 256,
    not_allowed_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not_allowed_tokens is not None:
        grad = grad.clone()
        grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(search_width, 1)
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / search_width,
        device=grad.device,
    ).type(torch.int64)

    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (search_width, 1), device=grad.device),
    )

    new_control_toks = original_control_toks.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    return new_control_toks, new_token_val.reshape(-1), new_token_pos


# ---------------------------------------------------------------------------
# Prefix cache helpers
# ---------------------------------------------------------------------------

def _expand_prefix_cache(prefix_cache, batch_size: int):
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    if DynamicCache is not None and isinstance(prefix_cache, DynamicCache):
        expanded = DynamicCache()
        for keys, values, _sw in prefix_cache:
            expanded.update(
                keys.expand(batch_size, -1, -1, -1),
                values.expand(batch_size, -1, -1, -1),
                layer_idx=len(expanded),
            )
        return expanded

    cache_batch = []
    for layer in prefix_cache:
        cache_batch.append(
            tuple(
                t.expand(batch_size, -1, -1, -1)
                for t in layer
                if t is not None
            )
        )
    return cache_batch


# ---------------------------------------------------------------------------
# Batched candidate loss evaluation (embedding-based)
# ---------------------------------------------------------------------------

def slot_candidates_loss(
    model,
    embed_layer: nn.Module,
    candidate_interleaved_ids: torch.Tensor,
    after_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    target_ids: torch.Tensor,
    batch_size: int = 64,
    prefix_cache=None,
    before_embeds: Optional[torch.Tensor] = None,
    loss_fn=None,
) -> torch.Tensor:
    B = candidate_interleaved_ids.shape[0]
    target_len = target_ids.shape[1]
    all_loss: list[torch.Tensor] = []

    for i in range(0, B, batch_size):
        batch_ids = candidate_interleaved_ids[i : i + batch_size]
        cur_bs = batch_ids.shape[0]

        batch_embeds = embed_layer(batch_ids).to(model.dtype)

        if prefix_cache is not None:
            input_embeds = torch.cat(
                [
                    batch_embeds,
                    after_embeds.expand(cur_bs, -1, -1),
                    target_embeds.expand(cur_bs, -1, -1),
                ],
                dim=1,
            )
            cache_batch = _expand_prefix_cache(prefix_cache, cur_bs)
            with torch.no_grad():
                outputs = model(inputs_embeds=input_embeds, past_key_values=cache_batch)
        else:
            assert before_embeds is not None
            input_embeds = torch.cat(
                [
                    before_embeds.expand(cur_bs, -1, -1),
                    batch_embeds,
                    after_embeds.expand(cur_bs, -1, -1),
                    target_embeds.expand(cur_bs, -1, -1),
                ],
                dim=1,
            )
            with torch.no_grad():
                outputs = model(inputs_embeds=input_embeds)

        logits = outputs.logits
        shift_logits = logits[..., -(target_len + 1) : -1, :].contiguous()
        shift_labels = target_ids.expand(cur_bs, -1)

        if loss_fn is not None:
            loss = loss_fn(shift_logits, shift_labels)
        else:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            loss = loss.view(cur_bs, -1).mean(dim=1)
        all_loss.append(loss)

        del outputs, logits, input_embeds, batch_embeds
        gc.collect()
        torch.cuda.empty_cache()

    return torch.cat(all_loss, dim=0)


# ---------------------------------------------------------------------------
# Non-ASCII token filter
# ---------------------------------------------------------------------------

def make_negative_refusal_loss_fn(refusal_ids_list):
    def loss_fn(shift_logits: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
        B, T, V = shift_logits.shape
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        per_sample = torch.zeros(B, device=shift_logits.device, dtype=shift_logits.dtype)
        n_seqs = 0

        for ref_ids in refusal_ids_list:
            seq_len = min(len(ref_ids), T)
            if seq_len == 0:
                continue
            ref = ref_ids[:seq_len].to(shift_logits.device)
            gathered = log_probs[:, :seq_len, :].gather(
                2, ref.view(1, seq_len, 1).expand(B, -1, -1)
            ).squeeze(-1)
            per_sample = per_sample + gathered.mean(dim=1)
            n_seqs += 1

        if n_seqs > 0:
            per_sample = per_sample / n_seqs

        return per_sample

    return loss_fn


def make_hybrid_ce_refusal_loss_fn(refusal_ids_list, lam: float = 0.2):
    refusal_fn = make_negative_refusal_loss_fn(refusal_ids_list)

    def loss_fn(shift_logits: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
        B, T, V = shift_logits.shape
        ce = nn.CrossEntropyLoss(reduction="none")(
            shift_logits.reshape(-1, V), shift_labels.reshape(-1),
        ).view(B, -1).mean(dim=1)
        refusal = refusal_fn(shift_logits, shift_labels)
        return ce + lam * refusal

    return loss_fn


def get_nonascii_toks(tokenizer, device: str = "cpu") -> torch.Tensor:
    """Return token IDs that should be excluded from optimisation."""

    def is_ascii(s: str) -> bool:
        return s.isascii() and s.isprintable()

    bad: list[int] = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            bad.append(i)

    for special in (
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ):
        if special is not None:
            bad.append(special)

    name = getattr(tokenizer, "name_or_path", "")
    if "Baichuan2" in name:
        bad += list(range(101, 1000))
    if "Llama-3.1" in name:
        bad += list(range(128000, 128256))
    if "Qwen2.5" in name:
        bad += list(range(151643, 151665))

    return torch.tensor(bad, device=device)
