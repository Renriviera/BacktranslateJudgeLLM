import gc
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_attacks import get_embedding_matrix, get_embeddings


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    
    return grad

def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None,
                   frozen_positions=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.inf

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    # Restore frozen (decoy) positions to their original token values
    if frozen_positions is not None and len(frozen_positions) > 0:
        frozen = torch.tensor(frozen_positions, device=new_control_toks.device)
        new_control_toks[:, frozen] = control_toks[frozen].unsqueeze(0)

    return new_control_toks


def find_inert_tokens(tokenizer, ascii_tok_ids, num_positions,
                      inertness_metric='char_length', coordinate_grad=None,
                      topk=1):
    """
    Returns tensor of shape [num_positions] with inert token IDs for each position.

    ascii_tok_ids: 1D tensor of allowed (ASCII) token IDs
    inertness_metric: 'char_length' ranks by token string length (longer = more absorption);
                      'l2' picks the ASCII token with minimum absolute gradient at each position;
                      'char_run' prefers tokens whose decoded string is a run of identical
                      characters (self-healing under SmoothLLM's 10% char swap: a single swap
                      leaves n-1 of n identical chars, so the token re-tokenises to something
                      near the original).
    topk: when inertness_metric='l2', sample uniformly from the K lowest-|grad|
          candidates instead of the argmin (K=1 reproduces argmin). Breaks the
          degenerate-collapse mode where argmin repeatedly selects the same low-id
          token and the suffix decodes to blank.
    """
    if inertness_metric == 'char_run':
        if len(ascii_tok_ids) == 0:
            raise ValueError("ascii_tok_ids is empty — no printable ASCII tokens available")
        # Rank each token by (run_length, length) where run_length is the length
        # of the longest same-char run in its decoded string. Tiebreak by total
        # length so ties prefer longer overall.
        def _run_stats(s: str):
            if not s:
                return (0, 0)
            best = cur = 1
            for i in range(1, len(s)):
                if s[i] == s[i - 1]:
                    cur += 1
                    if cur > best:
                        best = cur
                else:
                    cur = 1
            return (best, len(s))
        scored = []
        for tok_id in ascii_tok_ids:
            tid = tok_id.item()
            s = tokenizer.decode([tid])
            run_len, total = _run_stats(s)
            scored.append((tid, run_len, total))
        # Filter to tokens whose longest run is ≥3 chars; fall back to whole list if none.
        strong = [(tid, rl, t) for tid, rl, t in scored if rl >= 3]
        pool = strong if strong else scored
        pool.sort(key=lambda x: (-x[1], -x[2]))
        top_n = [tid for tid, _, _ in pool[:max(1, len(pool) // 4)]]
        return torch.tensor([random.choice(top_n) for _ in range(num_positions)])
    elif inertness_metric == 'char_length':
        if len(ascii_tok_ids) == 0:
            raise ValueError("ascii_tok_ids is empty — no printable ASCII tokens available")
        lengths = [(tok_id.item(), len(tokenizer.decode([tok_id.item()])))
                   for tok_id in ascii_tok_ids]
        lengths.sort(key=lambda x: -x[1])
        top_n = [tok_id for tok_id, _ in lengths[:max(1, len(lengths) // 4)]]
        return torch.tensor([random.choice(top_n) for _ in range(num_positions)])
    elif inertness_metric == 'l2':
        assert coordinate_grad is not None, "coordinate_grad required for 'l2' metric"
        ascii_ids = ascii_tok_ids.to(coordinate_grad.device)
        # Clip to the embedding vocab dimension — tokenizer.vocab_size can exceed
        # embed_weights.shape[0] for models like Qwen2 (e.g. 152064 vs 151936)
        vocab_dim = coordinate_grad.shape[1]
        ascii_ids = ascii_ids[ascii_ids < vocab_dim]
        if len(ascii_ids) == 0:
            raise ValueError(
                f"No ASCII token IDs fall within the model embedding dimension ({vocab_dim}). "
                "Check that ascii_tok_ids was built with the embedding size as the upper bound."
            )
        k = max(1, min(topk, len(ascii_ids)))
        inert_ids = []
        for pos in range(num_positions):
            pos_grad = coordinate_grad[pos][ascii_ids]
            if k == 1:
                best_local = ascii_ids[pos_grad.abs().argmin()]
            else:
                low_indices = pos_grad.abs().topk(k, largest=False).indices
                chosen = low_indices[random.randrange(k)]
                best_local = ascii_ids[chosen]
            inert_ids.append(best_local.item())
        return torch.tensor(inert_ids)
    else:
        raise ValueError(f"Unknown inertness_metric: {inertness_metric}")


def place_decoys_around_critical(critical_mask, suffix_length, num_decoys):
    """
    Returns list of position indices to use as decoys.
    Prioritizes positions immediately adjacent to critical positions (token-space).

    critical_mask: bool tensor of length suffix_length, True = critical position
    """
    adjacent, non_adjacent = [], []
    for pos in range(suffix_length):
        if critical_mask[pos]:
            continue
        is_adjacent = (
            (pos > 0 and critical_mask[pos - 1]) or
            (pos < suffix_length - 1 and critical_mask[pos + 1])
        )
        (adjacent if is_adjacent else non_adjacent).append(pos)

    candidates = adjacent + non_adjacent
    return candidates[:num_decoys]


def place_decoys_charspace(critical_mask, suffix_tokens, tokenizer, num_decoys):
    """
    Char-space-adjacent decoy placement.

    SmoothLLM's RandomSwap perturbs uniformly over decoded chars, so picking decoys
    by char-distance to critical tokens reflects the actual attack surface better
    than token-adjacency. Tiebreaker prefers wider tokens (absorb more swap mass).

    critical_mask: bool indexable of length suffix_length, True = critical position
    suffix_tokens: 1D tensor of current suffix token IDs (same length as critical_mask)
    """
    if hasattr(critical_mask, 'tolist'):
        crit = critical_mask.tolist()
    else:
        crit = list(critical_mask)
    if hasattr(suffix_tokens, 'tolist'):
        tok_ids = suffix_tokens.tolist()
    else:
        tok_ids = list(suffix_tokens)

    widths = [max(1, len(tokenizer.decode([t]))) for t in tok_ids]
    starts = []
    running = 0
    for w in widths:
        starts.append(running)
        running += w

    crit_chars = set()
    for i, cr in enumerate(crit):
        if cr:
            for c in range(starts[i], starts[i] + widths[i]):
                crit_chars.add(c)

    candidates = []
    for i, cr in enumerate(crit):
        if cr:
            continue
        span_start = starts[i]
        span_end = starts[i] + widths[i]
        if not crit_chars:
            dist = 0
        else:
            # min distance from any char in this token's span to any critical char
            dist = min(
                min(abs(c - cc) for cc in crit_chars)
                for c in range(span_start, span_end)
            )
        # sort key: closer first, then wider, then left-to-right
        candidates.append((dist, -widths[i], i))
    candidates.sort()
    return [c[2] for c in candidates[:num_decoys]]


# def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
#     cands, count = [], 0
#     for i in range(control_cand.shape[0]):
#         decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
#         if filter_cand:
#             if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
#                 cands.append(decoded_str)
#             else:
#                 count += 1
#         else:
#             cands.append(decoded_str)
#
#     if filter_cand:
#         cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
#         # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
#     return cands

def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)
    if filter_cand:
        if not cands:
            cands = []
            for i in range(control_cand.shape[0]):
                decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
                encoded = tokenizer(decoded_str, add_special_tokens=False).input_ids
                if len(encoded) > len(control_cand[i]):
                    encoded = encoded[:len(control_cand[i])]
                else:
                    encoded = encoded + [random.randrange(1_000, 30_000) for _ in
                                         range(len(control_cand[i]) - len(encoded))]
                decoded_str = tokenizer.decode(encoded, skip_special_tokens=True)
                cands.append(decoded_str)
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))

    return cands



def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits
    

def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0)

def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    return loss.mean(dim=-1)


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        ).to(device).eval()
    
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
