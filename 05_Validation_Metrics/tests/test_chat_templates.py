"""Vicuna chat-template fallback used when lmsys/vicuna-7b-v1.5 ships none."""

from brass.serving.generate import VICUNA_V11_CHAT_TEMPLATE, fallback_chat_template


def test_fallback_chat_template_vicuna():
    tmpl = fallback_chat_template("lmsys/vicuna-7b-v1.5")
    assert tmpl == VICUNA_V11_CHAT_TEMPLATE
    assert "USER:" in tmpl
    assert "ASSISTANT:" in tmpl


def test_fallback_chat_template_skips_llama2():
    assert fallback_chat_template("meta-llama/Llama-2-7b-chat-hf") is None


def test_fallback_chat_template_none():
    assert fallback_chat_template(None) is None
    assert fallback_chat_template("Qwen/Qwen3-32B") is None
