from typing import Dict, Optional

import torch
from joblib import Memory
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from src.logger import logger

memory = Memory("./cache", verbose=0)

LOADED_MODELS: Dict[str, "ModelWrapper"] = {}


class ModelConfig:
    """
    Configuration for model loading, including quantization, compilation, and generation caching.
    """

    def __init__(
        self,
        enable_quantization: bool = False,
        quant_type: str = "nf4",
        compile_kwargs: Optional[Dict] = None,
        cache_implementation: Optional[str] = "static",
    ):
        self.enable_quantization = enable_quantization
        self.quant_type = quant_type

        self.compile_kwargs = compile_kwargs or {
            "fullgraph": False,
        }

        self.cache_implementation = cache_implementation

    @property
    def quant_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Create a BitsAndBytesConfig if quantization is enabled.
        """
        if not self.enable_quantization:
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=self.quant_type,
        )


class ModelWrapper:
    """
    Wrapper for a loaded model, its tokenizer, and the pipeline.
    """

    def __init__(
        self,
        model_id: str,
        config: ModelConfig,
        device_map: Optional[str] = "auto",
    ):
        logger.info(f"Initializing ModelWrapper for '{model_id}'...")
        self.model_id = model_id
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            quantization_config=config.quant_config,
            torch_dtype=torch.float16,
        )

        if config.compile_kwargs:
            self.model.forward = torch.compile(
                self.model.forward,
                **config.compile_kwargs,
            )
            logger.info(
                f"Compiled forward for '{model_id}' with {config.compile_kwargs}."
            )

        if config.cache_implementation:
            self.model.generation_config.cache_implementation = (
                config.cache_implementation
            )
            logger.info(
                f"Set generation cache implementation to '{config.cache_implementation}' for '{model_id}'."
            )

        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        logger.info(f"ModelWrapper for '{model_id}' initialized.")


def load_model(
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
    config: Optional[ModelConfig] = None,
) -> None:
    """
    Load and cache a model specified by model_id, using the given ModelConfig.
    If already loaded, this is a no-op.
    """
    if model_id in LOADED_MODELS:
        logger.info(f"Model '{model_id}' already loaded.")
        return

    config = config or ModelConfig()
    logger.info(
        f"Loading model '{model_id}' into cache with config: {config.__dict__}..."
    )
    wrapper = ModelWrapper(
        model_id=model_id,
        config=config,
        device_map="auto",
    )
    LOADED_MODELS[model_id] = wrapper
    logger.info(f"Model '{model_id}' loaded and cached.")


def get_model(
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
    config: Optional[ModelConfig] = None,
) -> ModelWrapper:
    """
    Retrieve a cached ModelWrapper. If not loaded yet, loads it first with the given config.

    Returns:
        ModelWrapper: An object containing model, tokenizer, and pipeline.
    """
    if model_id not in LOADED_MODELS:
        load_model(model_id=model_id, config=config)
    return LOADED_MODELS[model_id]


@memory.cache
def generate_using_transformers(
    messages: list[dict], model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
) -> str:
    """
    Generate text using the transformers pipeline.
    Args:
        messages (list[dict]): A list of message dictionaries to send to the model.
        model_id (str): The model to use for text generation.
    Returns:
        str: The generated text from the model.
    """
    warpper = get_model(model_id=model_id)
    transformers_pipe = warpper.pipeline
    result = transformers_pipe(
        messages,
        max_new_tokens=10000,
        do_sample=True,
        temperature=0.6,
    )
    return result[0]["generated_text"][-1]["content"]
