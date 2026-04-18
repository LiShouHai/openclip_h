import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from core.subtitle_burner import SubtitleBurner


def test_parse_srt_text_clips_overlapping_segments():
    burner = SubtitleBurner()
    text = (
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "[Host] Hello\n\n"
        "2\n"
        "00:00:01,500 --> 00:00:03,000\n"
        "World\n"
    )

    segments = burner._parse_srt_text(text)

    assert segments == [
        {
            "start": "00:00:00,000",
            "end": "00:00:01,500",
            "text": "[Host] Hello",
        },
        {
            "start": "00:00:01,500",
            "end": "00:00:03,000",
            "text": "World",
        },
    ]


def test_generate_ass_uses_cjk_font_and_strips_speaker_tags(monkeypatch):
    burner = SubtitleBurner()
    monkeypatch.setattr(
        SubtitleBurner,
        "_resolve_ass_font",
        classmethod(lambda cls, language: ("Noto Sans CJK SC", "/tmp/fonts")),
    )

    ass = burner._generate_ass(
        [
            {
                "start": "00:00:00,000",
                "end": "00:00:02,000",
                "text": "[Host] >> 你好\n第二行",
            }
        ]
    )

    assert "Style: Original,Noto Sans CJK SC,80,&H0000FFFF" in ass
    assert ",10,10,150,1" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:02.00,Original,,0,0,0,,你好 第二行" in ass


def test_generate_ass_keeps_dual_track_layout_when_translation_exists(monkeypatch):
    burner = SubtitleBurner()
    monkeypatch.setattr(
        SubtitleBurner,
        "_resolve_ass_font",
        classmethod(lambda cls, language: ("DejaVu Sans", None)),
    )

    ass = burner._generate_ass(
        [
            {
                "start": "00:00:00,000",
                "end": "00:00:02,000",
                "text": "Original line",
            }
        ],
        translated=[
            {
                "start": "00:00:00,000",
                "end": "00:00:02,000",
                "text": "Translated line",
            }
        ],
    )

    assert "Style: Original,DejaVu Sans,80,&H00FFFFFF" in ass
    assert "Style: Translation,DejaVu Sans,80,&H0000FFFF" in ass
    assert ",10,10,50,1" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:02.00,Translation,,0,0,0,,Translated line" in ass


def test_build_ass_filter_value_includes_fontsdir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SubtitleBurner,
        "_resolve_ass_font",
        classmethod(lambda cls, language: ("DejaVu Sans", "/tmp/fonts")),
    )

    filter_value = SubtitleBurner.build_ass_filter_value(tmp_path / "clip.ass", language="en")

    assert filter_value == f"ass={tmp_path / 'clip.ass'}:fontsdir=/tmp/fonts"


def test_build_ass_filter_value_omits_fontsdir_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SubtitleBurner,
        "_resolve_ass_font",
        classmethod(lambda cls, language: ("Arial", None)),
    )

    filter_value = SubtitleBurner.build_ass_filter_value(tmp_path / "clip.ass", language="en")

    assert filter_value == f"ass={tmp_path / 'clip.ass'}"


def test_build_ass_filter_value_escapes_filter_special_chars(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SubtitleBurner,
        "_resolve_ass_font",
        classmethod(lambda cls, language: ("DejaVu Sans", "C:/Windows/Fonts")),
    )

    filter_value = SubtitleBurner.build_ass_filter_value(tmp_path / "clip.ass", language="en")

    assert "fontsdir=C\\:/Windows/Fonts" in filter_value


def test_font_family_from_path_uses_primary_family_name(monkeypatch):
    class CompletedProcess:
        returncode = 0
        stdout = "Heiti TC,黑體-繁,黒体-繁\n"

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(),
    )

    family = SubtitleBurner._font_family_from_path("/System/Library/Fonts/STHeiti Light.ttc")

    assert family == "Heiti TC"


def test_generate_preview_image_renders_subtitles(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not installed")

    filters = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if " ass " not in filters.stdout:
        pytest.skip("ffmpeg was built without libass support")

    burner = SubtitleBurner()
    background_path = tmp_path / "background.png"
    baseline_video_path = tmp_path / "baseline.mp4"
    baseline_path = tmp_path / "baseline.png"
    preview_path = tmp_path / "preview.png"

    burner._create_preview_background(background_path)
    subprocess.run(
        [
            ffmpeg,
            "-loop", "1",
            "-i", str(background_path),
            "-t", "2",
            "-pix_fmt", "yuv420p",
            "-an",
            "-y", str(baseline_video_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-ss", "0.5",
            "-i", str(baseline_video_path),
            "-frames:v", "1",
            "-update", "1",
            "-y", str(baseline_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    ok = burner.generate_preview_image(
        preview_path,
        original_text="Preview subtitle",
    )

    assert ok is True
    assert baseline_path.exists()
    assert preview_path.exists()
    diff = ImageChops.difference(Image.open(baseline_path), Image.open(preview_path))
    assert diff.getbbox() is not None


def test_preferred_subtitle_path_for_clip_prefers_whisper_sidecar(tmp_path):
    mp4 = tmp_path / "rank_01_test.mp4"
    mp4.write_bytes(b"video")
    original_srt = tmp_path / "rank_01_test.srt"
    original_srt.write_text("original", encoding="utf-8")
    whisper_srt = tmp_path / "rank_01_test.whisper.srt"
    whisper_srt.write_text("whisper", encoding="utf-8")

    preferred = SubtitleBurner.preferred_subtitle_path_for_clip(mp4)

    assert preferred == whisper_srt


def test_burn_subtitles_for_clips_uses_whisper_sidecar_when_available(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    output_dir = tmp_path / "out"
    clips_dir.mkdir()
    mp4 = clips_dir / "rank_01_test.mp4"
    mp4.write_bytes(b"video")
    (clips_dir / "rank_01_test.srt").write_text("original", encoding="utf-8")
    whisper_srt = clips_dir / "rank_01_test.whisper.srt"
    whisper_srt.write_text("whisper", encoding="utf-8")

    burner = SubtitleBurner()
    used_subtitles = []

    def fake_process_clip(mp4_path, srt_path, output_path, subtitle_translation=None):
        used_subtitles.append(Path(srt_path))
        return True

    monkeypatch.setattr(burner, "_process_clip", fake_process_clip)

    result = burner.burn_subtitles_for_clips(str(clips_dir), str(output_dir))

    assert result["success"] is True
    assert used_subtitles == [whisper_srt]
