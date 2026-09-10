"""Abstract LLM Provider layer.

Supports: OpenAI, Anthropic (Claude), DeepSeek.
All providers expose a common chat_completion interface.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base for LLM providers used by VisionCompiler."""

    @abstractmethod
    async def chat_completion(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        response_json: bool = True,
    ) -> str:
        """Send a chat completion request, return the text content."""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        ...


class OpenAILLMProvider(LLMProvider):
    """OpenAI ChatCompletion (GPT-4o, GPT-4-turbo, etc.)"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._model = model
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def default_model(self) -> str:
        return self._model

    async def chat_completion(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        response_json: bool = True,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class AnthropicLLMProvider(LLMProvider):
    """Anthropic Claude (claude-3-5-sonnet, claude-3-opus, etc.)"""

    def __init__(self, api_key: str, model: str = "claude-opus-4-6"):
        self._model = model
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def default_model(self) -> str:
        return self._model

    async def chat_completion(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        response_json: bool = True,
    ) -> str:
        # Claude uses system as a top-level param, not in messages
        response = await self._client.messages.create(
            model=model or self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        content = response.content[0].text
        if response_json:
            # Claude may wrap JSON in markdown — strip fences if present
            stripped = content.strip()
            if stripped.startswith("```"):
                lines = stripped.split("\n")
                stripped = "\n".join(lines[1:-1])
            # Validate it's parseable JSON
            json.loads(stripped)
            return stripped
        return content


class DeepSeekLLMProvider(LLMProvider):
    """DeepSeek API (OpenAI-compatible endpoint)."""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self._model = model
        import openai
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )

    @property
    def default_model(self) -> str:
        return self._model

    async def chat_completion(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        response_json: bool = True,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
class MockLLMProvider(LLMProvider):
    """Mock LLM provider for local testing without an API key."""

    def __init__(self, model: str = "mock-llm"):
        self._model = model

    @property
    def default_model(self) -> str:
        return self._model

    async def chat_completion(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        response_json: bool = True,
    ) -> str:

        if response_json:
            return json.dumps({
                "title": "Mock Generated Film",

                "scenes": [
                    {
                        "id": "S1",
                        "description": (
                            "A detective walks through a dark city street "
                            "at night while searching for clues."
                        ),
                        "complexity": 3,
                        "scene_type": "landscape",
                        "estimated_duration_sec": 5,
                        "dependencies": [],
                        "assets_required": [
                            "character_detective",
                            "location_city_street"
                        ]
                    },
                    {
                        "id": "S2",
                        "description": (
                            "The detective investigates a mysterious "
                            "location and searches for evidence."
                        ),
                        "complexity": 5,
                        "scene_type": "action",
                        "estimated_duration_sec": 5,
                        "dependencies": ["S1"],
                        "assets_required": [
                            "character_detective",
                            "location_mysterious_location"
                        ]
                    },
                    {
                        "id": "S3",
                        "description": (
                            "The detective discovers an important clue "
                            "and examines it carefully."
                        ),
                        "complexity": 3,
                        "scene_type": "dialogue",
                        "estimated_duration_sec": 5,
                        "dependencies": ["S2"],
                        "assets_required": [
                            "character_detective",
                            "prop_clue"
                        ]
                    }
                ],

                "assets": [
                    {
                        "id": "character_detective",
                        "type": "character",
                        "description": (
                            "A professional detective wearing a dark coat "
                            "and carrying a notebook."
                        )
                    },
                    {
                        "id": "location_city_street",
                        "type": "location",
                        "description": (
                            "A dark urban street at night with street "
                            "lights and wet pavement."
                        )
                    },
                    {
                        "id": "location_mysterious_location",
                        "type": "location",
                        "description": (
                            "A mysterious abandoned location with "
                            "dim lighting and old surroundings."
                        )
                    },
                    {
                        "id": "prop_clue",
                        "type": "prop",
                        "description": (
                            "A small mysterious piece of evidence "
                            "discovered by the detective."
                        )
                    }
                ],

                "dag": {
                    "S1": ["S2"],
                    "S2": ["S3"],
                    "S3": []
                }
            })

        return "Mock LLM response"