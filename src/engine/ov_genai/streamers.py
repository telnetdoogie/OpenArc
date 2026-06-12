from typing import List, Optional, Union
import openvino_genai
import asyncio

from openvino_genai import StreamerBase

from src.engine.ov_genai.output_parsers import PlainStreamingTextFilter
from src.server.models.ov_genai import OVGenAI_GenConfig


class ChunkStreamer(StreamerBase):
    """
    Streams decoded text in chunks of N tokens.
    - tokens_len == 1 → token-by-token streaming.
    - tokens_len  > 1 → emit after every N tokens.
    Uses cumulative decode + delta slicing to avoid subword boundary artifacts.
    """
    def __init__(self, decoder_tokenizer, gen_config: OVGenAI_GenConfig, text_filter=None):
        super().__init__()
        self.decoder_tokenizer = decoder_tokenizer
        self.tokens_len = max(1, gen_config.stream_chunk_tokens or 1)  # enforce at least 1
        self.tokens_cache: List[int] = []          # cumulative token buffer
        self.since_last_emit: int = 0              # tokens collected since last emit
        self.text_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        self._cancelled = asyncio.Event()          # cancellation flag for thread-safe signaling
        self.text_filter = text_filter or PlainStreamingTextFilter()

    def write(self, token: Union[int, List[int]]) -> openvino_genai.StreamingStatus:
        # Check for cancellation first
        if self._cancelled.is_set():
            # Signal completion to the queue so the consumer can exit
            self.text_queue.put_nowait(None)
            return openvino_genai.StreamingStatus.CANCEL

        # Normalize input to a list of ints
        if isinstance(token, list):
            self.tokens_cache.extend(token)
            self.since_last_emit += len(token)
        else:
            self.tokens_cache.append(token)
            self.since_last_emit += 1

        # Only emit when we've reached the chunk boundary
        if self.since_last_emit >= self.tokens_len:
            text = self.decoder_tokenizer.decode(self.tokens_cache)
            chunk = self.text_filter.filter(text)
            # Emit only the newly materialized portion

            if chr(65533) in chunk:
                self.since_last_emit -= 1
                return openvino_genai.StreamingStatus.RUNNING
            if chunk:
                self.text_queue.put_nowait(chunk)

            self.since_last_emit = 0

        return openvino_genai.StreamingStatus.RUNNING

    def cancel(self) -> None:
        """Signal cancellation of the streaming generation."""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """Check if cancellation has been signaled."""
        return self._cancelled.is_set()

    def end(self) -> None:
        # Flush any remaining tokens at the end
        text = self.decoder_tokenizer.decode(self.tokens_cache)
        chunk = self.text_filter.filter(text)
        if chunk:
            self.text_queue.put_nowait(chunk)
        # Signal completion
        self.text_queue.put_nowait(None)