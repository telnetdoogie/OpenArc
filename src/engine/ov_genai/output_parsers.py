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

    FINAL_MARKERS = (
        "<|start|>assistant<|channel|>final<|message|>",
        "<|channel|>final<|message|>",
        "assistantfinal",
    )

    STOP_MARKERS = (
        "<|return|>",
        "<|call|>",
        "<|end|>",
    )

    def __init__(self) -> None:
        self._final_start: int | None = None
        self._emitted_until: int | None = None
        self._stopped = False

    def filter(self, cumulative_text: str) -> str:
        if self._stopped:
            return ""

        if self._final_start is None:
            final_start = self._find_final_start(cumulative_text)

            if final_start is None:
                return ""

            self._final_start = final_start
            self._emitted_until = final_start

        assert self._final_start is not None
        assert self._emitted_until is not None

        stop_at = len(cumulative_text)

        for stop_marker in self.STOP_MARKERS:
            stop_index = cumulative_text.find(stop_marker, self._final_start)
            if stop_index != -1:
                stop_at = min(stop_at, stop_index)
                self._stopped = True

        if stop_at <= self._emitted_until:
            return ""

        chunk = cumulative_text[self._emitted_until:stop_at]

        # Important: do not advance emitted position if decode is incomplete.
        if "\ufffd" in chunk:
            return ""

        self._emitted_until = stop_at
        return chunk

    def _find_final_start(self, cumulative_text: str) -> int | None:
        for marker in self.FINAL_MARKERS:
            index = cumulative_text.rfind(marker)
            if index != -1:
                return index + len(marker)

        return None

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