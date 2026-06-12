import pytest

from engine.ov_genai.output_parsers import (
    extract_plain_text,
    extract_generated_text,
    OutputFormat,
    extract_harmony_final_text,
    PlainStreamingTextFilter,
    HarmonyStreamingTextFilter,
    detect_output_format,
)


def test_plain_text_parser_returns_text_unchanged() -> None:
    raw = "The president of France is Emmanuel Macron."
    assert extract_plain_text(raw) == raw


def test_generated_text_plain_format_returns_text_unchanged() -> None:
    raw = "Sure — here's a normal model response."
    assert extract_generated_text(raw, OutputFormat.PLAIN) == raw


def test_plain_format_does_not_try_to_parse_harmony_like_words() -> None:
    raw = "I am doing analysis of the final answer."
    assert extract_generated_text(raw, OutputFormat.PLAIN) == raw


def test_harmony_parser_extracts_only_final_channel() -> None:
    raw = (
        "<|start|>assistant<|channel|>analysis<|message|>"
        "User asks who is president of France. Need answer."
        "<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        "Emmanuel Macron."
        "<|return|>"
    )
    assert extract_harmony_final_text(raw) == "Emmanuel Macron."


def test_harmony_parser_drops_analysis_even_when_final_has_context() -> None:
    raw = (
        "<|start|>assistant<|channel|>analysis<|message|>"
        "Do not expose this reasoning."
        "<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        "As of 2026, the President of France is Emmanuel Macron."
        "<|return|>"
    )
    assert extract_generated_text(raw, OutputFormat.HARMONY) == "As of 2026, the President of France is Emmanuel Macron."


def test_harmony_parser_handles_stripped_decoder_fallback() -> None:
    raw = (
        "analysisUser asks who is president of France. "
        "assistantfinalAs of 2026, the President of France is Emmanuel Macron."
        "analysisUser asks \"Who is the president of France?\" As of knowledge cutoff 2024-06, president is Emmanuel Macron (since 2017 and re-elected 2022)."
        "We should mention current president and term. The user didn't specify time context, but presumably now. The assistant should give answer."
        "Possibly mention current year. Also note that president may change after election 2027? But we are only as of 2024. So answer: Emmanuel Macron."
        "Provide context. Should be straightforward.assistantfinalThe current President of France is **Emmanuel Macron**."
    )
    assert extract_harmony_final_text(raw) == "The current President of France is **Emmanuel Macron**."


def test_harmony_parser_stops_at_end_marker_too() -> None:
    raw = (
        "<|start|>assistant<|channel|>final<|message|>"
        "Final answer."
        "<|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>"
        "More junk."
    )
    assert extract_harmony_final_text(raw) == "Final answer."


def test_plain_streaming_filter_emits_deltas_unchanged() -> None:
    f = PlainStreamingTextFilter()
    assert f.filter("Hello") == "Hello"
    assert f.filter("Hello world") == " world"
    assert f.filter("Hello world!") == "!"


def test_harmony_streaming_filter_emits_nothing_before_final() -> None:
    f = HarmonyStreamingTextFilter()
    assert f.filter("<|start|>assistant<|channel|>analysis<|message|>thinking") == ""


def test_harmony_streaming_filter_starts_at_final_channel() -> None:
    f = HarmonyStreamingTextFilter()
    assert f.filter("<|start|>assistant<|channel|>analysis<|message|>thinking<|end|>") == ""
    assert f.filter(
        "<|start|>assistant<|channel|>analysis<|message|>thinking<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hello"
    ) == "Hello"
    assert f.filter(
        "<|start|>assistant<|channel|>analysis<|message|>thinking<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hello world"
    ) == " world"


def test_harmony_streaming_filter_stops_at_return() -> None:
    f = HarmonyStreamingTextFilter()
    assert f.filter("<|start|>assistant<|channel|>final<|message|>Hello<|return|>") == "Hello"
    assert f.filter("<|start|>assistant<|channel|>final<|message|>Hello<|return|>junk") == ""


class FakeTokenizer:
    def __init__(self, chat_template: str) -> None:
        self.chat_template = chat_template


def test_detect_output_format_respects_explicit_plain_override() -> None:
    tokenizer = FakeTokenizer(
        "<|start|>assistant<|channel|>final<|message|><|return|>"
    )
    assert detect_output_format(tokenizer, configured="plain") == OutputFormat.PLAIN


def test_detect_output_format_detects_harmony_template() -> None:
    tokenizer = FakeTokenizer(
        "<|start|>assistant<|channel|>analysis<|message|>"
        "<|start|>assistant<|channel|>final<|message|><|return|>"
    )
    assert detect_output_format(tokenizer) == OutputFormat.HARMONY


def test_detect_output_format_defaults_to_plain() -> None:
    tokenizer = FakeTokenizer("[INST] {{ messages }} [/INST]")
    assert detect_output_format(tokenizer) == OutputFormat.PLAIN


def test_detect_output_format_defaults_to_plain_with_unset_override() -> None:
    tokenizer = FakeTokenizer("[INST] {{ messages }} [/INST]")
    assert detect_output_format(tokenizer, configured=None) == OutputFormat.PLAIN
    assert detect_output_format(tokenizer, configured="") == OutputFormat.PLAIN


def test_detect_output_format_throws_value_error_with_invalid_value() -> None:
    tokenizer = FakeTokenizer("[INST] {{ messages }} [/INST]")
    with pytest.raises(ValueError, match="'unrecognized_value' is not a valid OutputFormat"):
        detect_output_format(tokenizer, configured="unrecognized_value")


def test_detect_output_format_respects_explicit_harmony_override() -> None:
    tokenizer = FakeTokenizer("[INST] {{ messages }} [/INST]")
    assert detect_output_format(tokenizer, configured="harmony") == OutputFormat.HARMONY
