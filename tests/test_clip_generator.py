from pathlib import Path

from core.clip_generator import ClipGenerator


def test_generate_clips_prefers_whisper_source_for_boundary_normalization(tmp_path):
    generator = ClipGenerator(output_dir=str(tmp_path / "clips"), normalize_boundaries=True)
    analysis_file = tmp_path / "top_engaging_moments.json"
    video_dir = tmp_path / "videos"
    subtitle_dir = tmp_path / "subs"
    video_dir.mkdir()
    subtitle_dir.mkdir()

    (video_dir / "sample_part01.mp4").write_bytes(b"fake")
    original_srt = subtitle_dir / "sample_part01.srt"
    original_srt.write_text("", encoding="utf-8")
    whisper_srt = tmp_path / "candidate.absolute.srt"
    whisper_srt.write_text("", encoding="utf-8")

    analysis_file.write_text(
        """
{
  "top_engaging_moments": [
    {
      "rank": 1,
      "title": "Test Clip",
      "why_engaging": "Strong point.",
      "engagement_details": {"engagement_level": "high"},
      "timing": {
        "video_part": "part01",
        "start_time": "00:00:10",
        "end_time": "00:00:20",
        "duration": "10s"
      },
      "whisper_subtitle_source": "%s"
    }
  ],
  "analysis_summary": {}
}
"""
        % str(whisper_srt),
        encoding="utf-8",
    )

    parsed_paths = []

    def fake_parse(path):
        parsed_paths.append(path)
        return [{"start_time": "00:00:10,000", "end_time": "00:00:20,000", "text": "Hello."}]

    generator._parse_srt_file = fake_parse
    generator._normalize_clip_boundaries = lambda start, end, _: (start, end, {"start": "unchanged", "end": "unchanged"})
    generator._create_clip = lambda *args, **kwargs: True
    generator._extract_subtitle_for_clip = lambda *args, **kwargs: False
    generator._extract_subtitle_from_file = lambda *args, **kwargs: False
    generator._create_summary = lambda *args, **kwargs: None

    result = generator.generate_clips_from_analysis(
        str(analysis_file),
        str(video_dir),
        str(subtitle_dir),
    )

    assert result["success"] is True
    assert parsed_paths[0] == str(whisper_srt)


def test_extract_subtitle_from_explicit_file_supports_agentic_whisper_output(tmp_path):
    generator = ClipGenerator(output_dir=str(tmp_path))
    source_srt = tmp_path / "source.absolute.srt"
    source_srt.write_text(
        "\n".join(
            [
                "1",
                "00:00:10,000 --> 00:00:12,000",
                "First line.",
                "",
                "2",
                "00:00:12,000 --> 00:00:15,000",
                "Second line.",
                "",
                "3",
                "00:00:16,000 --> 00:00:18,000",
                "Outside clip.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output_srt = tmp_path / "clip.whisper.srt"

    generated = generator._extract_subtitle_from_file(
        str(source_srt),
        "00:00:11",
        "00:00:15",
        str(output_srt),
    )

    assert generated is True
    content = output_srt.read_text(encoding="utf-8")
    assert "First line." in content
    assert "Second line." in content
    assert "Outside clip." not in content


def test_create_summary_tolerates_partial_analysis_summary(tmp_path):
    generator = ClipGenerator(output_dir=str(tmp_path))
    clips_info = [
        {
            "rank": 1,
            "title": "Test Clip",
            "filename": "rank_01_test.mp4",
            "subtitle_filename": "rank_01_test.srt",
            "duration": 78,
            "time_range": "00:00:00 - 00:01:18",
            "engagement_level": "high",
            "why_engaging": "Clear standalone payoff.",
        }
    ]
    data = {
        "analysis_summary": {
            "verification_mode": "llm_standalone_first",
            "repair_pass_used": True,
            "selected_after_verification": 1,
        }
    }

    generator._create_summary(clips_info, data)

    summary_path = Path(tmp_path) / "engaging_moments_summary.md"
    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    assert "Generated Clips" in content
    assert "Test Clip" in content
