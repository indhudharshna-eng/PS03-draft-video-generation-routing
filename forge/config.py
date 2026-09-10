from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings


class ForgeSettings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    deepseek_api_key: str = ""
    fal_api_key: str = ""
    kling_api_key: str = ""
    kling_api_secret: str = ""
    forge_workers: int = 4
    forge_video_backend: str = "mock"
    output_dir: str = "./output"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = ForgeSettings()


def load_forge_yaml(path: str | Path = "forge.yaml") -> dict[str, Any]:
    """Load forge.yaml if it exists."""
    p = Path(path)

    if not p.exists():
        return {}

    try:
        import yaml

        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    except ImportError:
        return {}


class ForgeConfig:
    """Merged runtime configuration.

    forge.yaml controls provider/model/routing choices.
    Environment variables provide API keys.
    """

    def __init__(self, yaml_path: str | Path = "forge.yaml"):
        self._raw = load_forge_yaml(yaml_path)
        self._env = settings

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------

    @property
    def llm_provider(self) -> str:
        return self._raw.get("llm", {}).get("provider", "openai")

    @property
    def llm_model(self) -> str | None:
        return self._raw.get("llm", {}).get("model", None)

    @property
    def llm_api_key(self) -> str:
        raw_key = self._raw.get("llm", {}).get("api_key", "")

        if raw_key:
            return raw_key

        if self.llm_provider == "anthropic":
            return (
                self._env.anthropic_api_key
                or os.environ.get("ANTHROPIC_API_KEY", "")
            )

        if self.llm_provider == "deepseek":
            return (
                self._env.deepseek_api_key
                or os.environ.get("DEEPSEEK_API_KEY", "")
            )

        return (
            self._env.openai_api_key
            or os.environ.get("OPENAI_API_KEY", "")
        )

    # ---------------------------------------------------------
    # Image Generation
    # ---------------------------------------------------------

    @property
    def imagegen_provider(self) -> str:
        return self._raw.get("imagegen", {}).get("provider", "mock")

    @property
    def imagegen_model(self) -> str | None:
        return self._raw.get("imagegen", {}).get("model", None)

    @property
    def imagegen_api_key(self) -> str:
        raw_key = self._raw.get("imagegen", {}).get("api_key", "")

        if raw_key:
            return raw_key

        if self.imagegen_provider == "flux":
            return (
                self._env.fal_api_key
                or os.environ.get("FAL_API_KEY", "")
            )

        return (
            self._env.openai_api_key
            or os.environ.get("OPENAI_API_KEY", "")
        )

    # ---------------------------------------------------------
    # VLM / Validator
    # ---------------------------------------------------------

    @property
    def vlm_provider(self) -> str:
        return self._raw.get("validator", {}).get("provider", "mock")

    @property
    def vlm_model(self) -> str | None:
        return self._raw.get("validator", {}).get("model", None)

    @property
    def vlm_api_key(self) -> str:
        raw_key = self._raw.get("validator", {}).get("api_key", "")

        if raw_key:
            return raw_key

        if self.vlm_provider == "anthropic":
            return (
                self._env.anthropic_api_key
                or os.environ.get("ANTHROPIC_API_KEY", "")
            )

        return (
            self._env.openai_api_key
            or os.environ.get("OPENAI_API_KEY", "")
        )

    # ---------------------------------------------------------
    # Routing
    # ---------------------------------------------------------

    @property
    def routing(self) -> dict[str, str]:
        defaults = {
            "dialogue": "kling_light",
            "action": "kling_heavy",
            "landscape": "cogvideo",
            "product": "kling_heavy",
            "transition": "cogvideo",
            "default": "mock",
        }

        defaults.update(self._raw.get("routing", {}))
        return defaults

    # ---------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------

    @property
    def workers(self) -> int:
        return self._raw.get(
            "scheduler",
            {}
        ).get(
            "workers",
            self._env.forge_workers,
        )

    @property
    def max_retries(self) -> int:
        return self._raw.get(
            "scheduler",
            {}
        ).get(
            "max_retries",
            2,
        )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    @property
    def output_dir(self) -> str:
        return self._raw.get(
            "output",
            {}
        ).get(
            "dir",
            self._env.output_dir,
        )

    # ---------------------------------------------------------
    # Video backend
    # ---------------------------------------------------------

    @property
    def video_backend(self) -> str:
        return self._env.forge_video_backend

    # ---------------------------------------------------------
    # Provider builders
    # ---------------------------------------------------------

    def build_llm_provider(self):
        """Instantiate the configured LLM provider."""

        from forge.providers.llm import (
            OpenAILLMProvider,
            AnthropicLLMProvider,
            DeepSeekLLMProvider,
            MockLLMProvider,
        )

        provider = self.llm_provider
        key = self.llm_api_key
        model = self.llm_model

        if provider == "mock":
            return MockLLMProvider(
                model=model or "mock-llm"
            )

        if provider == "anthropic":
            return AnthropicLLMProvider(
                api_key=key,
                model=model or "claude-opus-4-6",
            )

        if provider == "deepseek":
            return DeepSeekLLMProvider(
                api_key=key,
                model=model or "deepseek-chat",
            )

        return OpenAILLMProvider(
            api_key=key,
            model=model or "gpt-4o",
        )

    def build_imagegen_provider(self):
        """Instantiate the configured image generation provider."""

        from forge.providers.imagegen import (
            OpenAIImageGenProvider,
            FluxImageGenProvider,
            MockImageGenProvider,
        )

        provider = self.imagegen_provider
        key = self.imagegen_api_key
        model = self.imagegen_model

        if provider == "mock":
            return MockImageGenProvider()

        if provider == "flux":
            return FluxImageGenProvider(
                api_key=key,
                model=model or "fal-ai/flux/schnell",
            )

        return OpenAIImageGenProvider(
            api_key=key,
            model=model or "dall-e-3",
        )

    def build_vlm_provider(self):
        """Instantiate the configured VLM validator provider."""

        from forge.providers.llm import (
            OpenAILLMProvider,
            AnthropicLLMProvider,
            MockLLMProvider,
        )

        provider = self.vlm_provider
        key = self.vlm_api_key
        model = self.vlm_model

        if provider == "mock":
            return MockLLMProvider(
                model=model or "mock-vlm"
            )

        if provider == "anthropic":
            return AnthropicLLMProvider(
                api_key=key,
                model=model or "claude-opus-4-6",
            )

        return OpenAILLMProvider(
            api_key=key,
            model=model or "gpt-4o",
        )