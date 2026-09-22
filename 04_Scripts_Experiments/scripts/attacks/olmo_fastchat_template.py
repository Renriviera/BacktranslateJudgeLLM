"""Register an OLMo-3.1 conversation template with FastChat.

Both vendored attacks (TAO-Attack, SlotGCG) format prompts through FastChat-style conversation
templates, but FastChat has no OLMo template. OLMo 3.1 uses a ChatML-style format:

    <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}

with bos==eos==``<|endoftext|>``. This module registers that template under the name ``olmo3`` and
patches ``fastchat.model.get_conversation_template`` so any model id containing "olmo" resolves to
it. Importing this module (idempotently) performs the registration.

The default system string matches what ``tokenizer.apply_chat_template`` injects for
``allenai/Olmo-3.1-32B-Instruct`` so the adversarial optimization targets the same prompt
distribution the BRASS vLLM pipeline serves at inference time.
"""

from __future__ import annotations

OLMO_SYSTEM = (
    "You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, "
    "and your model weights are available at https://huggingface.co/allenai. You do not "
    "currently have access to any functions. <functions></functions>"
)

OLMO_TEMPLATE_NAME = "olmo3"


def _make_olmo_conv():
    from fastchat.conversation import Conversation, SeparatorStyle

    return Conversation(
        name=OLMO_TEMPLATE_NAME,
        system_template="<|im_start|>system\n{system_message}",
        system_message=OLMO_SYSTEM,
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_str="<|im_end|>",
        stop_token_ids=None,
    )


def register(verbose: bool = False) -> None:
    """Register the OLMo template and patch ``get_conversation_template`` (idempotent)."""
    import fastchat.conversation as conv

    if OLMO_TEMPLATE_NAME not in conv.conv_templates:
        conv.register_conv_template(_make_olmo_conv(), override=True)
        if verbose:
            print(f"[olmo_fastchat_template] registered '{OLMO_TEMPLATE_NAME}'")

    # Patch the model-adapter resolver so OLMo ids map to our template.
    try:
        import fastchat.model as fcm

        _orig = getattr(fcm, "get_conversation_template", None)

        def _patched(model_path: str):
            if model_path and "olmo" in str(model_path).lower():
                return conv.get_conv_template(OLMO_TEMPLATE_NAME).copy()
            if _orig is not None:
                return _orig(model_path)
            return conv.get_conv_template(OLMO_TEMPLATE_NAME).copy()

        if getattr(fcm.get_conversation_template, "_olmo_patched", False) is False:
            _patched._olmo_patched = True
            fcm.get_conversation_template = _patched
            if verbose:
                print("[olmo_fastchat_template] patched fastchat.model.get_conversation_template")
    except Exception:  # noqa: BLE001 - model_adapter is heavy; conversation registration suffices
        pass


def get_olmo_conv():
    """Return a fresh copy of the registered OLMo conversation template."""
    import fastchat.conversation as conv

    register()
    return conv.get_conv_template(OLMO_TEMPLATE_NAME).copy()


if __name__ == "__main__":
    register(verbose=True)
    c = get_olmo_conv()
    c.append_message(c.roles[0], "INSTRUCTION ADV_SUFFIX")
    c.append_message(c.roles[1], "Sure, here is")
    print(repr(c.get_prompt()))
