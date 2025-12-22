import asyncio
import json
from types import SimpleNamespace

import pytest

from backend.app.api.intelligent_orchestrator import IntelligentOrchestrator, QueryClassifier


class FakeCompletions:
    """Stands in for client.chat.completions and returns a canned reply."""

    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def classifier_with(content=None, error=None):
    completions = FakeCompletions(content, error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return QueryClassifier(client), completions


def run(coro):
    return asyncio.run(coro)
