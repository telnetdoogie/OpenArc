from __future__ import annotations

from enum import Enum


class OutputFormat(str, Enum):
    PLAIN = "plain"
    HARMONY = "harmony"


HARMONY_FINAL_MARKER = "<|start|>assistant<|channel|>final<|message|>"
HARMONY_STOP_MARKERS = ("<|return|>", "<|call|>", "<|end|>")


def extract_plain_text(raw_text: str) -> str:
    return raw_text


def extract_harmony_final_text(raw_text: str) -> str:
    """
    Extract only the assistant final-channel text from Harmony output.

    Best case: raw_text still contains Harmony special tokens.
    Fallback case: some decoder stripped special tokens and left words like
    'analysis' / 'assistantfinal' behind.
    """
    if HARMONY_FINAL_MARKER in raw_text:
        text = raw_text.rsplit(HARMONY_FINAL_MARKER, 1)[1]
    elif "assistantfinal" in raw_text:
        # Ugly fallback for already-stripped output like:
        # analysis...assistantfinalActual answer
        text = raw_text.rsplit("assistantfinal", 1)[1]
    else:
        # I would return raw instead of blanking the answer,
        # but log this at the caller if format == harmony.
        text = raw_text

    for stop in HARMONY_STOP_MARKERS:
        if stop in text:
            text = text.split(stop, 1)[0]

    return text.strip()


def extract_generated_text(raw_text: str, output_format: OutputFormat) -> str:
    if output_format == OutputFormat.HARMONY:
        return extract_harmony_final_text(raw_text)

    return extract_plain_text(raw_text)


class StreamingTextFilter:
    def filter(self, cumulative_text: str) -> str:
        raise NotImplementedError


class PlainStreamingTextFilter(StreamingTextFilter):
    def __init__(self) -> None:
        self._last_len = 0

    def filter(self, cumulative_text: str) -> str:
        chunk = cumulative_text[self._last_len:]
        self._last_len = len(cumulative_text)
        return chunk


class HarmonyStreamingTextFilter(StreamingTextFilter):
    def __init__(self) -> None:
        self._seen_final = False
        self._last_final_len = 0
        self._stopped = False

    def filter(self, cumulative_text: str) -> str:
        if self._stopped:
            return ""

        if not self._seen_final:
            if HARMONY_FINAL_MARKER not in cumulative_text:
                return ""

            cumulative_text = cumulative_text.rsplit(HARMONY_FINAL_MARKER, 1)[1]
            self._seen_final = True
            self._last_final_len = 0
        else:
            # Once final has started, always work from final onward.
            cumulative_text = cumulative_text.rsplit(HARMONY_FINAL_MARKER, 1)[-1]

        for stop in HARMONY_STOP_MARKERS:
            if stop in cumulative_text:
                cumulative_text = cumulative_text.split(stop, 1)[0]
                self._stopped = True
                break

        if len(cumulative_text) <= self._last_final_len:
            return ""

        chunk = cumulative_text[self._last_final_len:]
        self._last_final_len = len(cumulative_text)
        return chunk


def make_streaming_filter(output_format: OutputFormat) -> StreamingTextFilter:
    if output_format == OutputFormat.HARMONY:
        return HarmonyStreamingTextFilter()
    return PlainStreamingTextFilter()


def detect_output_format(tokenizer, configured: str | None = None) -> OutputFormat:
    if configured:
        return OutputFormat(configured)
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    harmony_markers = (
        "<|channel|>analysis",
        "<|channel|>final",
        "<|return|>",
    )
    if all(marker in chat_template for marker in harmony_markers):
        return OutputFormat.HARMONY
    return OutputFormat.PLAIN