"""Load CoT-IQA, run one smoke-test inference, and save the result."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cot_iqa_model import COTIQAModel
IMAGE_PATH = PROJECT_ROOT / "assets/test_images/real_test.png"
OUTPUT_PATH = PROJECT_ROOT / "outputs/reports/cot_iqa_smoke_test.json"


def main() -> int:
    print(f"[{datetime.now().isoformat()}] Starting CoT-IQA smoke test.")
    print(f"Image: {IMAGE_PATH}")

    model = COTIQAModel()

    try:
        print("Loading Qwen2-VL base model and LoRA adapter...")
        model_info = model.load()

        print("Model loaded successfully.")
        print(json.dumps(model_info, ensure_ascii=False, indent=2))

        print("Starting single-image inference...")

        result = model.analyze(
            image_path=IMAGE_PATH,
            max_new_tokens=768,
            keep_raw_output=False,
            auto_load=False,
        )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Inference success: {result['success']}")
        print(
            "Generated tokens:",
            result["generated_token_count"],
        )
        print(
            "Inference time:",
            result["inference_time_seconds"],
            "seconds",
        )
        print("Error:", result["error"])
        print(f"Result saved to: {OUTPUT_PATH}")

        if result["parsed_output"] is not None:
            parsed = result["parsed_output"]

            print("Six-step complete:", parsed["complete"])
            print("Warnings:", parsed["warnings"])
            print(
                "Predicted MOS:",
                parsed["quality_prediction"]["predicted_mos"],
            )

        return 0 if result["success"] else 1

    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    finally:
        print("Unloading model and releasing GPU memory...")
        model.unload()
        print(f"[{datetime.now().isoformat()}] Smoke test finished.")


if __name__ == "__main__":
    raise SystemExit(main())
