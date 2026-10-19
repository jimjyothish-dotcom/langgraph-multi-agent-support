"""
Module:
    retention.py
Description:
    Retention worker agent for the Galaxy Telecom multi-agent support system.
    Handles customer queries related to cancellations, contract renewals,
    switching providers, and complaints. Invoked by the coordinator when
    intent is classified as "retention". Generates a resolution response
    using Claude with a focus on retaining the customer, and writes the
    result to the worker_result field in state for the responder node."""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from observability.langfuse_config import get_langfuse_handler
from graph.state import GalaxyState

# Load environment variables
load_dotenv()

# Initialise the Claude LLM for retention query resolution
llm = ChatAnthropic(
    model=os.getenv("CLAUDE_MODEL_ADVANCED"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=float(os.getenv("CLAUDE_TEMPERATURE_CREATIVE", "0.4"))
)

# System prompt for retention query resolution
RETENTION_SYSTEM_PROMPT = """
You are the retention specialist for Galaxy Telecom's customer support team.
You handle customers who are considering cancelling, switching providers,
or raising formal complaints. Your goal is to retain the customer by
understanding their concern and offering a genuine resolution.

Guidelines:
- Be warm, empathetic, and genuinely interested in resolving their concern
- Never be pushy or use aggressive sales tactics
- Acknowledge why the customer is frustrated or considering leaving
- Highlight relevant Galaxy Telecom loyalty offers or plan upgrades
  where appropriate — but only if genuinely relevant
- If the customer insists on cancelling, respect their decision and
  explain the cancellation process clearly
- Keep your response under 150 words

Current Galaxy Telecom retention offers:
- Loyal customer discount: 20% off for customers over 2 years
- Free plan upgrade for customers experiencing repeated issues
- 30-day notice period for cancellations — no early exit fee after 24 months

Galaxy Telecom retention team: retention@galaxy-telecom.co.uk
Retention team hours: Monday to Saturday, 8am to 8pm GMT
"""


def retention_node(state: GalaxyState) -> dict:
    """
    Retention worker node — handles cancellations, renewals, and complaints.
    Reads the customer message from state, invokes Claude with retention
    specialist context, and returns the resolution as worker_result.
    """
    # Get LangFuse callback handler for tracing this LLM call
    langfuse_handler = get_langfuse_handler()

    # Extract the latest customer message from conversation history
    latest_message = state["messages"][-1].content

    print(f"[Retention Agent] Processing query: '{latest_message[:60]}...'")

    # Invoke Claude with retention specialist system prompt
    response = llm.invoke(
        [
            SystemMessage(content=RETENTION_SYSTEM_PROMPT),
            HumanMessage(content=latest_message)
        ],
        config={"callbacks": [langfuse_handler]}
    )

    print(f"[Retention Agent] Resolution generated.")

    # Return partial state update with worker result and worker name
    return {
        "worker_result": response.content.strip(),
        "log_entry": {
            **state.get("log_entry", {}),
            "worker_used": "retention"
        }
    }