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


def test_classify_returns_the_scores_from_the_reply():
    classifier, _ = classifier_with(json.dumps({"technical_analysis": 0.9, "education": 0.4}))
    assert run(classifier.classify_query("show me RSI")) == {"technical_analysis": 0.9, "education": 0.4}


def test_classify_clamps_scores_to_the_zero_one_range():
    classifier, _ = classifier_with(json.dumps({"technical_analysis": 3, "education": -2}))
    assert run(classifier.classify_query("x")) == {"technical_analysis": 1.0, "education": 0.0}


def test_classify_turns_string_scores_into_floats():
    classifier, _ = classifier_with(json.dumps({"options_flow": "0.75"}))
    assert run(classifier.classify_query("x")) == {"options_flow": 0.75}


def test_classify_returns_nothing_for_invalid_json():
    classifier, _ = classifier_with("this is not json")
    assert run(classifier.classify_query("x")) == {}


def test_classify_returns_nothing_when_the_api_fails():
    classifier, _ = classifier_with(error=RuntimeError("network down"))
    assert run(classifier.classify_query("x")) == {}


def test_classify_sends_the_query_and_the_agent_list():
    classifier, completions = classifier_with("{}")
    run(classifier.classify_query("analyze AAPL"))
    prompt = completions.calls[0]["messages"][1]["content"]
    assert "analyze AAPL" in prompt
    assert "technical_analysis" in prompt
    assert completions.calls[0]["temperature"] == 0.1


def test_extract_symbol_uppercases_the_reply():
    classifier, _ = classifier_with(" aapl \n")
    assert run(classifier.extract_stock_symbol("analyze apple")) == "AAPL"
