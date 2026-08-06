"""Qwen2-VL + LoRA inference wrapper for COT-IQA-Agent."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

from configs.config_loader import load_config
from models.parser import parse_cot_output
from tools.image_tools import validate_image


DEFAULT_COT_IQA_PROMPT = """Assess the perceptual quality of this image step by step.

1. Locate all visibly degraded regions.
2. Identify the degradation type and severity of each region.
3. Diagnose blur, noise, compression, color, and artifact intensity.
4. Suggest suitable restoration actions.
5. Determine the expert routing weights.
6. Predict the overall quality score from 1 (worst) to 5 (best)."""


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class COTIQAModel:
    """Lazy-loading inference wrapper for the CoT-IQA model."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        if config is None:
            project_config = load_config()
            config = project_config["cot_iqa"]

        self.config = dict(config)

        self.base_model_path = Path(
            self.config["base_model_path"]
        ).expanduser().resolve()

        self.adapter_path = Path(
            self.config["adapter_path"]
        ).expanduser().resolve()

        self.device = str(
            self.config.get("device", "auto")
        ).strip().lower()

        self.dtype_name = str(
            self.config.get("dtype", "bfloat16")
        ).strip().lower()

        self.max_new_tokens = int(
            self.config.get("max_new_tokens", 1024)
        )

        self.trust_remote_code = bool(
            self.config.get("trust_remote_code", True)
        )

        if self.dtype_name not in DTYPE_MAP:
            raise ValueError(
                "Unsupported dtype: "
                f"{self.dtype_name}. "
                f"Expected one of {sorted(DTYPE_MAP)}."
            )

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        self.torch_dtype = DTYPE_MAP[self.dtype_name]

        self.processor: Any | None = None
        self.model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Return whether both processor and model are loaded."""
        return self.processor is not None and self.model is not None

    def _validate_model_paths(self) -> None:
        """Validate local model and adapter directories."""

        if not self.base_model_path.is_dir():
            raise FileNotFoundError(
                f"Base model directory not found: "
                f"{self.base_model_path}"
            )

        if not self.adapter_path.is_dir():
            raise FileNotFoundError(
                f"Adapter directory not found: "
                f"{self.adapter_path}"
            )

        adapter_config = self.adapter_path / "adapter_config.json"

        if not adapter_config.is_file():
            raise FileNotFoundError(
                f"LoRA adapter_config.json not found: "
                f"{adapter_config}"
            )

    def _resolve_device_map(self) -> Any:
        """Convert project device configuration into a device map."""

        if self.device == "auto":
            return "auto"

        if self.device == "cpu":
            return {"": "cpu"}

        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but CUDA is unavailable."
                )

            return {"": "cuda:0"}

        if self.device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but CUDA is unavailable."
                )

            return {"": self.device}

        raise ValueError(
            "device must be auto, cpu, cuda, or cuda:<index>."
        )

    def _get_input_device(self) -> torch.device:
        """Return the device that should receive processor outputs."""

        if self.model is None:
            raise RuntimeError("The model is not loaded.")

        try:
            embedding_layer = self.model.get_input_embeddings()

            if embedding_layer is not None:
                return embedding_layer.weight.device

        except (AttributeError, RuntimeError):
            pass

        return next(self.model.parameters()).device

    def load(self) -> dict[str, Any]:
        """Load the processor, base model, and LoRA adapter."""

        if self.is_loaded:
            return self.get_model_info()

        self._validate_model_paths()

        device_map = self._resolve_device_map()

        processor = None
        base_model = None

        try:
            processor = AutoProcessor.from_pretrained(
                str(self.base_model_path),
                trust_remote_code=self.trust_remote_code,
                local_files_only=True,
            )

            base_model = (
                Qwen2VLForConditionalGeneration.from_pretrained(
                    str(self.base_model_path),
                    torch_dtype=self.torch_dtype,
                    device_map=device_map,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=True,
                )
            )

            model = PeftModel.from_pretrained(
                base_model,
                str(self.adapter_path),
                is_trainable=False,
            )

            model.eval()

            self.processor = processor
            self.model = model

        except Exception:
            self.processor = None
            self.model = None

            if base_model is not None:
                del base_model

            if processor is not None:
                del processor

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise

        return self.get_model_info()

    def get_model_info(self) -> dict[str, Any]:
        """Return JSON-serializable model runtime information."""

        input_device = None

        if self.is_loaded:
            input_device = str(self._get_input_device())

        return {
            "loaded": self.is_loaded,
            "base_model_path": str(self.base_model_path),
            "adapter_path": str(self.adapter_path),
            "configured_device": self.device,
            "input_device": input_device,
            "dtype": self.dtype_name,
            "max_new_tokens": self.max_new_tokens,
        }

    @staticmethod
    def build_prompt(prompt: str | None = None) -> str:
        """Return the default or caller-supplied diagnostic prompt."""

        if prompt is None:
            return DEFAULT_COT_IQA_PROMPT

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string or None.")

        cleaned_prompt = prompt.replace("<image>", "").strip()

        if not cleaned_prompt:
            raise ValueError("prompt must not be empty.")

        return cleaned_prompt

    def analyze(
        self,
        image_path: str | Path,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
        keep_raw_output: bool = True,
        auto_load: bool = True,
    ) -> dict[str, Any]:
        """Analyze one image and return raw and parsed CoT-IQA output."""

        validation = validate_image(image_path)

        if not validation["valid"]:
            return {
                "success": False,
                "image_path": validation["path"],
                "raw_output": "",
                "parsed_output": None,
                "generated_token_count": 0,
                "hit_max_new_tokens": False,
                "ended_with_eos": False,
                "inference_time_seconds": 0.0,
                "error": validation.get("error") or "Invalid image.",
            }

        if not self.is_loaded:
            if not auto_load:
                return {
                    "success": False,
                    "image_path": validation["path"],
                    "raw_output": "",
                    "parsed_output": None,
                    "generated_token_count": 0,
                    "hit_max_new_tokens": False,
                    "ended_with_eos": False,
                    "inference_time_seconds": 0.0,
                    "error": "The CoT-IQA model is not loaded.",
                }

            self.load()

        if self.model is None or self.processor is None:
            raise RuntimeError(
                "The processor or model is unavailable after loading."
            )

        generation_limit = (
            self.max_new_tokens
            if max_new_tokens is None
            else int(max_new_tokens)
        )

        if generation_limit <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        diagnostic_prompt = self.build_prompt(prompt)
        path = Path(validation["path"])

        try:
            with Image.open(path) as source_image:
                image = source_image.convert("RGB").copy()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": diagnostic_prompt,
                        },
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

            input_device = self._get_input_device()
            inputs = inputs.to(input_device)

            if input_device.type == "cuda":
                torch.cuda.synchronize(input_device)

            start_time = time.perf_counter()

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_limit,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )

            if input_device.type == "cuda":
                torch.cuda.synchronize(input_device)

            inference_time = time.perf_counter() - start_time

            input_token_count = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_token_count:]
            generated_token_count = int(generated_ids.numel())

            hit_max_new_tokens = (
                generated_token_count >= generation_limit
            )

            raw_eos = self.model.generation_config.eos_token_id

            if raw_eos is None:
                eos_token_ids: set[int] = set()

            elif isinstance(raw_eos, (list, tuple, set)):
                eos_token_ids = {
                    int(token_id)
                    for token_id in raw_eos
                }

            else:
                eos_token_ids = {int(raw_eos)}

            ended_with_eos = (
                generated_token_count > 0
                and int(generated_ids[-1]) in eos_token_ids
            )

            raw_output = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            parsed_output = parse_cot_output(
                raw_output,
                keep_raw_output=keep_raw_output,
            )

            return {
                "success": True,
                "image_path": str(path),
                "prompt": diagnostic_prompt,
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "generated_token_count": generated_token_count,
                "hit_max_new_tokens": hit_max_new_tokens,
                "ended_with_eos": ended_with_eos,
                "inference_time_seconds": round(
                    inference_time,
                    6,
                ),
                "error": None,
            }

        except Exception as exc:
            return {
                "success": False,
                "image_path": str(path),
                "raw_output": "",
                "parsed_output": None,
                "generated_token_count": 0,
                "hit_max_new_tokens": False,
                "ended_with_eos": False,
                "inference_time_seconds": 0.0,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    def analyze_batch(
        self,
        image_paths: Sequence[str | Path],
        prompt: str | None = None,
        max_new_tokens: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Analyze multiple images sequentially to limit GPU memory use."""

        return {
            str(image_path): self.analyze(
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            for image_path in image_paths
        }

    def unload(self) -> None:
        """Unload model resources and release CUDA cache."""

        model = self.model
        processor = self.processor

        self.model = None
        self.processor = None

        if model is not None:
            del model

        if processor is not None:
            del processor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
