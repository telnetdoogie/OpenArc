from unittest.mock import AsyncMock, MagicMock

import pytest  # type: ignore[import]

import src.engine.ov_genai.llm as llm_module
from src.engine.ov_genai.llm import OVGenAI_LLM
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType
from src.server.models.ov_genai import OVGenAI_GenConfig
from src.engine.ov_genai.output_parsers import (
    OutputFormat,
    HarmonyStreamingTextFilter,
    PlainStreamingTextFilter,
    extract_generated_text,
    extract_plain_text,
    extract_harmony_final_text,
    detect_output_format,
    )


MODEL_PATH ="some_fake_url/Qwen3-Reranker-0.6B-fp16-ov"


class DummyMeanValue:
    def __init__(self, mean: float) -> None:
        self.mean = mean


class DummyPerfMetrics:
    def __init__(self) -> None:
        self._input_tokens = 25
        self._generated_tokens = 10

    def get_load_time(self) -> int:
        return 5000

    def get_ttft(self) -> DummyMeanValue:
        return DummyMeanValue(250.0)

    def get_tpot(self) -> DummyMeanValue:
        return DummyMeanValue(7.5)

    def get_throughput(self) -> DummyMeanValue:
        return DummyMeanValue(12.34567)

    def get_generate_duration(self) -> DummyMeanValue:
        return DummyMeanValue(1000.0)

    def get_num_input_tokens(self) -> int:
        return self._input_tokens

    def get_num_generated_tokens(self) -> int:
        return self._generated_tokens


@pytest.fixture
def load_config() -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="test-model",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={"config": "value"},
    )


def test_prepare_inputs_passes_tools(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    llm = OVGenAI_LLM(load_config)
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"name": "tool", "description": "desc"}]

    apply_mock = MagicMock(return_value="np_payload")
    llm.encoder_tokenizer = MagicMock()
    llm.encoder_tokenizer.apply_chat_template = apply_mock

    class DummyTensor:
        def __init__(self, value):
            self.value = value

    monkeypatch.setattr(llm_module.ov, "Tensor", DummyTensor)

    result = llm.prepare_inputs(messages, tools)

    assert isinstance(result, DummyTensor)
    apply_mock.assert_called_once_with(
        messages,
        tools=tools,
        add_generation_prompt=True,
        skip_special_tokens=True,
        return_tensors="np",
    )


def test_prepare_inputs_without_tools_uses_none(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    llm = OVGenAI_LLM(load_config)
    messages = [{"role": "user", "content": "hi"}]

    apply_mock = MagicMock(return_value="np_payload")
    llm.encoder_tokenizer = MagicMock()
    llm.encoder_tokenizer.apply_chat_template = apply_mock

    class DummyTensor:
        def __init__(self, value):
            self.value = value

    monkeypatch.setattr(llm_module.ov, "Tensor", DummyTensor)

    llm.prepare_inputs(messages)

    apply_mock.assert_called_once_with(
        messages,
        tools=None,
        add_generation_prompt=True,
        skip_special_tokens=True,
        return_tensors="np",
    )


@pytest.mark.parametrize(
    ("stream", "target"),
    (
        (True, "generate_stream"),
        (False, "generate_text"),
    ),
)
def test_generate_type_respects_stream_flag(load_config: ModelLoadConfig, stream: bool, target: str) -> None:
    llm = OVGenAI_LLM(load_config)
    gen_config = OVGenAI_GenConfig(stream=stream)

    expected = object()
    setattr(llm, target, MagicMock(return_value=expected))

    result = llm.generate_type(gen_config)

    assert result is expected
    getattr(llm, target).assert_called_once_with(gen_config)


def test_collect_metrics_prefill_throughput(load_config: ModelLoadConfig) -> None:
    llm = OVGenAI_LLM(load_config)
    gen_config = OVGenAI_GenConfig(stream=True, stream_chunk_tokens=2)

    metrics = llm.collect_metrics(gen_config, DummyPerfMetrics())

    assert metrics["prefill_throughput (tokens/s)"] == 100.0
    assert metrics["stream"] is True
    assert metrics["stream_chunk_tokens"] == 2


def test_load_model_sets_pipeline_and_tokenizer(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    llm = OVGenAI_LLM(load_config)
    pipeline_instance = MagicMock()
    pipeline_factory = MagicMock(return_value=pipeline_instance)
    monkeypatch.setattr(llm_module, "LLMPipeline", pipeline_factory)

    tokenizer_instance = MagicMock()
    monkeypatch.setattr(
        llm_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=tokenizer_instance),
    )

    loader = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="loader-model",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={"hint": "value"},
    )

    llm.load_model(loader)

    pipeline_factory.assert_called_once_with(
        loader.model_path,
        loader.device,
        **loader.runtime_config,
    )
    llm_module.AutoTokenizer.from_pretrained.assert_called_once_with(loader.model_path)
    assert llm.model is pipeline_instance
    assert llm.encoder_tokenizer is tokenizer_instance


def test_load_model_forwards_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_factory = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(llm_module, "LLMPipeline", pipeline_factory)
    monkeypatch.setattr(
        llm_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=MagicMock()),
    )

    loader = ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="loader-model",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={"hint": "value"},
        cache_dir="/tmp/ov_cache",
    )

    OVGenAI_LLM(loader).load_model(loader)

    pipeline_factory.assert_called_once_with(
        loader.model_path,
        loader.device,
        hint="value",
        CACHE_DIR="/tmp/ov_cache",
    )


def _draft_loader(cache_dir):
    return ModelLoadConfig(
        model_path=str(MODEL_PATH),
        model_name="loader-model",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
        runtime_config={},
        cache_dir=cache_dir,
        draft_model_path="/models/draft",
        draft_device="CPU",
    )


def _patch_llm_load(monkeypatch):
    monkeypatch.setattr(llm_module, "LLMPipeline", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        llm_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=MagicMock()),
    )
    draft_factory = MagicMock(return_value=object())
    monkeypatch.setattr(llm_module.openvino_genai, "draft_model", draft_factory)
    return draft_factory


def test_load_model_forwards_cache_dir_to_draft_model(monkeypatch: pytest.MonkeyPatch) -> None:
    draft_factory = _patch_llm_load(monkeypatch)
    loader = _draft_loader("/tmp/ov_cache")

    OVGenAI_LLM(loader).load_model(loader)

    draft_factory.assert_called_once_with(
        "/models/draft",
        "CPU",
        CACHE_DIR="/tmp/ov_cache",
    )


def test_load_model_draft_model_without_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    draft_factory = _patch_llm_load(monkeypatch)
    loader = _draft_loader(None)

    OVGenAI_LLM(loader).load_model(loader)

    draft_factory.assert_called_once_with("/models/draft", "CPU")


@pytest.mark.asyncio
async def test_unload_model_resets_state(monkeypatch: pytest.MonkeyPatch, load_config: ModelLoadConfig) -> None:
    llm = OVGenAI_LLM(load_config)
    llm.model = object()
    llm.encoder_tokenizer = object()

    registry = MagicMock()
    registry.register_unload = AsyncMock(return_value=True)

    gc_mock = MagicMock()
    monkeypatch.setattr(llm_module.gc, "collect", gc_mock)

    result = await llm.unload_model(registry, "model-name")

    assert result is True
    assert llm.model is None
    assert llm.encoder_tokenizer is None
    registry.register_unload.assert_called_once_with("model-name")
    gc_mock.assert_called_once()


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
    )
    assert extract_harmony_final_text(raw) == "As of 2026, the President of France is Emmanuel Macron."


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