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


def test_extract_symbol_returns_none_for_none():
    classifier, _ = classifier_with("NONE")
    assert run(classifier.extract_stock_symbol("how is the market")) is None


@pytest.mark.parametrize("reply", ["A", "TOOLONG"])
def test_extract_symbol_rejects_implausible_lengths(reply):
    classifier, _ = classifier_with(reply)
    assert run(classifier.extract_stock_symbol("x")) is None


def test_extract_symbol_without_a_client_returns_none():
    assert run(QueryClassifier(None).extract_stock_symbol("analyze AAPL")) is None


def test_extract_symbol_returns_none_when_the_api_fails():
    classifier, _ = classifier_with(error=RuntimeError("boom"))
    assert run(classifier.extract_stock_symbol("x")) is None


def pick_agents(scores):
    # _determine_agents_to_trigger does not use self, so a bare call is enough.
    return IntelligentOrchestrator._determine_agents_to_trigger(None, scores)


def test_a_comprehensive_request_runs_the_four_core_agents():
    scores = {"technical_analysis": 0.9, "sentiment_analysis": 0.9, "options_flow": 0.9, "historical_analysis": 0.9}
    assert pick_agents(scores) == ["technical", "sentiment", "flow", "history"]


def test_a_targeted_request_runs_only_that_agent():
    assert pick_agents({"technical_analysis": 0.9, "sentiment_analysis": 0.1}) == ["technical"]


def test_scores_at_the_threshold_do_not_trigger_an_agent():
    assert pick_agents({"education": 0.3, "technical_analysis": 0.31}) == ["technical"]


def test_education_and_risk_are_added_on_top_of_analysis():
    scores = {"technical_analysis": 0.8, "education": 0.8, "risk_assessment": 0.8}
    assert pick_agents(scores) == ["technical", "education", "risk"]


def test_trade_execution_adds_the_buy_agent():
    assert "buy_agent" in pick_agents({"trade_execution": 0.9, "technical_analysis": 0.8})
