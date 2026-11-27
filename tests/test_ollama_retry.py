"""OllamaPolicy retries degenerate generations (HTTP 500) and charges the failed calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    import httpx2 as httpx
except ImportError:
    import httpx
from openai import InternalServerError

from agent.search.common import OllamaPolicy


def repeat_error() -> InternalServerError:
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    return InternalServerError(
        "prediction aborted, token repeat limit reached",
        response=httpx.Response(500, request=request),
        body=None,
    )


def ok_response(content: str = "done") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, logprobs=None)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
    )


class FakeCompletions:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise repeat_error()
        return ok_response()


def make_policy(failures: int) -> tuple[OllamaPolicy, FakeCompletions]:
    policy = OllamaPolicy(question="q")
    fake = FakeCompletions(failures)
    policy.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return policy, fake


MESSAGES = [{"role": "user", "content": "x" * 400}]


def test_retry_succeeds_and_charges_failed_call():
    policy, fake = make_policy(failures=1)
    cand = policy.sample(MESSAGES, temperature=0.0, seed=5)
    assert len(fake.calls) == 2
    assert fake.calls[1]["seed"] != fake.calls[0]["seed"]
    assert fake.calls[1]["temperature"] >= 0.7
    assert policy.n_failed_calls == 1
    assert cand.completion_tokens == 10 + policy.max_completion_tokens
    assert cand.prompt_tokens > 100
    again = policy.sample(MESSAGES, temperature=0.0, seed=6)
    assert (again.prompt_tokens, again.completion_tokens) == (100, 10)


def test_generate_text_retries_without_logprobs():
    policy, fake = make_policy(failures=1)
    text, p_tok, c_tok = policy.generate_text(MESSAGES, max_tokens=32, seed=0)
    assert text == "done"
    assert all("logprobs" not in c for c in fake.calls)
    assert c_tok == 10 + 32 and p_tok > 100


def test_persistent_failure_raises():
    policy, fake = make_policy(failures=10)
    with pytest.raises(InternalServerError):
        policy.sample(MESSAGES, temperature=0.7, seed=0)
    assert len(fake.calls) == policy.max_retries + 1
