from core.llm.qwen_api_client import QwenAPIClient


def test_qwen_client_uses_environment_overrides(monkeypatch):
    with monkeypatch.context() as env:
        env.setenv("QWEN_API_KEY", "test-key")
        env.setenv("QWEN_BASE_URL", "https://qwen.example/compatible-mode/v1/")
        env.setenv("QWEN_MODEL", "qwen-local-test")

        client = QwenAPIClient()

        assert client.base_url == "https://qwen.example/compatible-mode/v1/chat/completions"
        assert client.default_model == "qwen-local-test"


def test_qwen_client_preserves_explicit_full_endpoint(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "test-key")

    client = QwenAPIClient(
        base_url="https://qwen.example/custom/chat/completions",
    )

    assert client.base_url == "https://qwen.example/custom/chat/completions"
