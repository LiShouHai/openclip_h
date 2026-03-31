import pytest

import core.cover_image_generator as cover_image_generator
import core.title_adder as title_adder


def test_artistic_text_renderer_requires_cjk_font(monkeypatch):
    monkeypatch.setattr(title_adder, "find_best_font", lambda **kwargs: None)

    renderer = title_adder.ArtisticTextRenderer(language="zh")

    with pytest.raises(RuntimeError, match="No suitable CJK font found"):
        renderer._get_font(40)


def test_cover_image_generator_requires_cjk_font(monkeypatch):
    monkeypatch.setattr(cover_image_generator, "find_best_font", lambda **kwargs: None)

    generator = cover_image_generator.CoverImageGenerator(language="zh")

    with pytest.raises(RuntimeError, match="No suitable CJK font found"):
        generator._load_font(48)
