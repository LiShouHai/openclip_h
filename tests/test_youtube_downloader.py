from pathlib import Path

from core.downloaders.youtube_downloader import COMMON_JS_RUNTIME_PATHS, YouTubeDownloader


def test_resolve_js_runtime_falls_back_to_common_node_install_paths(monkeypatch):
    downloader = YouTubeDownloader(js_runtime="auto")
    expected_path = str(COMMON_JS_RUNTIME_PATHS["node"][1])

    def fake_which(name):
        if name == "deno":
            return None
        if name == "node":
            return None
        return None

    monkeypatch.setattr("core.downloaders.youtube_downloader.shutil.which", fake_which)
    monkeypatch.setattr(
        "core.downloaders.youtube_downloader.Path.is_file",
        lambda self: str(self) == expected_path,
    )

    resolved = downloader._resolve_js_runtime()

    assert resolved == {
        "runtime": "node",
        "path": expected_path,
    }


def test_best_quality_format_selector_prefers_progressive_mp4_before_dash_pairs():
    downloader = YouTubeDownloader(quality="best")

    assert downloader._get_format_selector() == (
        "best[ext=mp4][vcodec^=avc1][acodec^=mp4a]/"
        "bestvideo[vcodec^=avc1][ext=mp4][height<=1080]+bestaudio[ext=m4a]/"
        "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/"
        "best[ext=mp4]/best"
    )
