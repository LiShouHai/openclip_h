from pathlib import Path

import pytest

from core.transcript_generation_paraformer import ParaformerTranscriptProcessor


def test_availability_error_reports_missing_project_dir(tmp_path):
    processor = ParaformerTranscriptProcessor(project_dir=tmp_path / "missing-paraformer")

    assert "Paraformer project dir not found" in processor.availability_error()


def test_find_output_json_prefers_stem_matched_file(tmp_path):
    processor = ParaformerTranscriptProcessor(project_dir=tmp_path)
    expected = tmp_path / "clip.json"
    expected.write_text("{}", encoding="utf-8")
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")

    assert processor._find_output_json(tmp_path, "clip") == expected


def test_find_output_json_accepts_single_generated_candidate(tmp_path):
    processor = ParaformerTranscriptProcessor(project_dir=tmp_path)
    candidate = tmp_path / "generated.json"
    candidate.write_text("{}", encoding="utf-8")
    (tmp_path / "summary.jsonl").write_text("", encoding="utf-8")

    assert processor._find_output_json(tmp_path, "clip") == candidate


def test_find_output_json_rejects_ambiguous_candidates(tmp_path):
    processor = ParaformerTranscriptProcessor(project_dir=tmp_path)
    (tmp_path / "first.json").write_text("{}", encoding="utf-8")
    (tmp_path / "second.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Could not locate Paraformer JSON output"):
        processor._find_output_json(tmp_path, "clip")
