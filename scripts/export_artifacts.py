#!/usr/bin/env python3
"""
Package trained runs into runtime artifacts under artifacts/<model_version>/.

Layout:
artifacts/<model_version>/
  manifest.json
  baseline/
    model.joblib
  transformer/
    (HF model + tokenizer files)

The API loads using:
- ARTIFACT_DIR (default: ./artifacts/<MODEL_VERSION>)
- MODEL_PROVIDER: baseline | transformer | stub
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--provider", required=True, choices=["baseline", "transformer"])
    args = parser.parse_args()

    model_version = args.model_version
    provider = args.provider

    run_dir = Path(".runs") / model_version
    out_dir = Path("artifacts") / model_version
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_version": model_version,
        "provider": provider,
        "exported_from": str(run_dir),
    }

    if provider == "baseline":
        src = run_dir / "baseline.joblib"
        if not src.exists():
            raise FileNotFoundError(f"Missing baseline model at {src}")

        dst_dir = out_dir / "baseline"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "model.joblib")
        manifest["baseline"] = {"path": "baseline/model.joblib"}

    if provider == "transformer":
        src_dir = run_dir / "transformer"
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing transformer dir at {src_dir}")

        dst_dir = out_dir / "transformer"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        manifest["transformer"] = {"path": "transformer/"}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Exported artifacts to:", out_dir)


if __name__ == "__main__":
    main()
