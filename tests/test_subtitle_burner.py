from pathlib import Path

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
