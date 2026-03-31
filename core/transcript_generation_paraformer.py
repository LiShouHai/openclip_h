#!/usr/bin/env python3
"""
Chinese transcript generation via a local FunASR Paraformer checkout.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from core.config import PARAFORMER_DEVICE, PARAFORMER_PROJECT_DIR

logger = logging.getLogger(__name__)


class ParaformerTranscriptProcessor:
    """Bridge to the external funasr-paraformer project."""

    def __init__(
        self,
        project_dir: str = PARAFORMER_PROJECT_DIR,
        device: str = PARAFORMER_DEVICE,
    ):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.device = device
        self.python_bin = self.project_dir / ".venv" / "bin" / "python"
        self.transcribe_script = self.project_dir / "tools" / "transcribe_long_audio.py"
        self.json_to_srt_script = self.project_dir / "tools" / "funasr_json_to_srt.py"

    def availability_error(self) -> str | None:
        if not self.project_dir.exists():
            return f"Paraformer project dir not found: {self.project_dir}"
        if not self.python_bin.exists():
            return f"Paraformer Python not found: {self.python_bin}"
        if not self.transcribe_script.exists():
            return f"Paraformer transcription script not found: {self.transcribe_script}"
        if not self.json_to_srt_script.exists():
            return f"Paraformer JSON→SRT script not found: {self.json_to_srt_script}"
        return None

    def is_available(self) -> bool:
        return self.availability_error() is None

    def transcribe_chinese_to_srt(self, media_path: str, output_dir: str | Path) -> tuple[str, dict[str, Any]]:
        error = self.availability_error()
        if error:
            raise RuntimeError(error)

        media_path = Path(media_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_srt = output_dir / f"{media_path.stem}.srt"

        with tempfile.TemporaryDirectory(prefix="paraformer_", dir=str(output_dir)) as tmp_dir:
            tmp_output_dir = Path(tmp_dir)
            self._run(
                [
                    str(self.python_bin),
                    str(self.transcribe_script),
                    str(media_path),
                    "--lang",
                    "zh",
                    "--device",
                    self.device,
                    "--disable-pbar",
                    "--cleanup-normalized",
                    "-o",
                    str(tmp_output_dir),
                ],
                step=f"Paraformer transcription for {media_path.name}",
            )

            output_json = self._find_output_json(tmp_output_dir, media_path.stem)
            payload = json.loads(output_json.read_text(encoding="utf-8"))

            self._run(
                [
                    str(self.python_bin),
                    str(self.json_to_srt_script),
                    str(output_json),
                    str(output_srt),
                ],
                step=f"Paraformer SRT conversion for {media_path.name}",
            )

        return str(output_srt), payload

    def _find_output_json(self, output_dir: Path, stem: str) -> Path:
        direct_match = output_dir / f"{stem}.json"
        if direct_match.exists():
            return direct_match

        candidates = sorted(
            path
            for path in output_dir.glob("*.json")
            if path.name != "summary.jsonl"
        )
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(f"Could not locate Paraformer JSON output under {output_dir}")

    def _run(self, cmd: list[str], step: str) -> None:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0:
            return

        stderr_tail = (proc.stderr or "")[-1200:]
        stdout_tail = (proc.stdout or "")[-1200:]
        raise RuntimeError(
            f"{step} failed with exit code {proc.returncode}\n"
            f"stdout:\n{stdout_tail}\n"
            f"stderr:\n{stderr_tail}"
        )
