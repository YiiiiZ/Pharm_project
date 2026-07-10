"""
Routing tests for the care plan query system.

Each test sends a natural-language query to TriageAgent and asserts
that it hands off to the correct specialist, then prints the final
agent name and response.

Run:
    OPENAI_API_KEY=sk-... python3.11 -m pytest care/test_query_routing.py -v -s
"""

import os
import pytest
from agents import Runner
from care.query_agents import triage_agent

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _print_result(query: str, agent_name: str, response: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"Query  : {query}")
    print(f"Agent  : {agent_name}")
    print(f"Reply  : {response}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Skip all tests if no API key is present
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def require_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------

async def test_progress_routing():
    """Progress query → should be answered by ProgressAgent."""
    query = "订单 12345 的 care plan 生成到哪了"

    result = await Runner.run(triage_agent, query)
    agent_name = result.last_agent.name

    _print_result(query, agent_name, result.final_output)

    assert agent_name == "ProgressAgent", (
        f"Expected ProgressAgent, got {agent_name}. "
        "Check Triage instructions for progress-related keywords."
    )


async def test_review_routing():
    """Verification query → should be answered by ReviewAgent."""
    query = "为什么 12345 的 care plan 标了 hallucination"

    result = await Runner.run(triage_agent, query)
    agent_name = result.last_agent.name

    _print_result(query, agent_name, result.final_output)

    assert agent_name == "ReviewAgent", (
        f"Expected ReviewAgent, got {agent_name}. "
        "Check Triage instructions for verification/audit keywords."
    )


async def test_resubmit_routing():
    """Regeneration request → should be answered by ResubmitAgent."""
    query = "12345 的 care plan 有问题，请重新生成"

    result = await Runner.run(triage_agent, query)
    agent_name = result.last_agent.name

    _print_result(query, agent_name, result.final_output)

    assert agent_name == "ResubmitAgent", (
        f"Expected ResubmitAgent, got {agent_name}. "
        "Check Triage instructions for resubmit/regeneration keywords."
    )
