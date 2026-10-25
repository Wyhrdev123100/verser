from .registry import REGISTRY, ModelSpec
from .adapters import ToyGPTAdapter, HuggingFaceAdapter

REGISTRY.register_adapter("toy_gpt", ToyGPTAdapter)
REGISTRY.register_adapter("huggingface", HuggingFaceAdapter)

REGISTRY.register_model(ModelSpec(
    name="toy-gpt",
    family="causal_lm",
    adapter="toy_gpt",
    description="Tiny causal LM for smoke tests and offline vertical slices.",
))
REGISTRY.register_model(ModelSpec(
    name="hf-causal-lm",
    family="causal_lm",
    adapter="huggingface",
    description="Any Hugging Face causal language model.",
))
REGISTRY.register_model(ModelSpec(
    name="hf-seq2seq",
    family="seq2seq",
    adapter="huggingface",
    description="Any Hugging Face seq2seq model.",
))
REGISTRY.register_model(ModelSpec(
    name="hf-encoder",
    family="encoder",
    adapter="huggingface",
    description="Any Hugging Face masked LM / encoder model.",
))

__all__ = ["REGISTRY", "ModelSpec"]