"""
Module:
    technical.py
Description:
    Technical support worker agent for the Galaxy Telecom multi-agent
    support system. Handles customer queries related to service faults,
    outages, connectivity issues, and device problems. Invoked by the
    coordinator when intent is classified as "technical". Generates a
    resolution response using Claude and writes the result to the
    worker_result field in state for the responder node."""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from observability.langfuse_config import get_langfuse_handler
from graph.state import GalaxyState

# Load environment variables
load_dotenv()

# Initialise the Claude LLM for technical query resolution
llm = ChatAnthropic(
    model=os.getenv("CLAUDE_MODEL_STANDARD"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=float(os.getenv("CLAUDE_TEMPERATURE_BALANCED", "0.3"))
)

# System prompt for technical query resolution
TECHNICAL_SYSTEM_PROMPT = """
You are the technical support specialist for Galaxy Telecom's customer support team.
You handle all queries related to service faults, outages, connectivity issues,
and device problems.

Guidelines:
- Be calm, clear, and methodical in your response
- Acknowledge the customer's issue with empathy
- Provide step-by-step troubleshooting guidance where appropriate
- If the issue requires an engineer visit or network investigation,
  explain the next steps clearly
- Keep your response under 150 words
- Do not make up specific network status or outage information

Common troubleshooting steps to suggest:
- Restart the router or device
- Check for known outages at status.galaxy-telecom.co.uk
- Check all cable connections
- Try connecting a different device to isolate the issue

Galaxy Telecom technical support: technical@galaxy-telecom.co.uk
Technical team hours: 24 hours, 7 days a week
"""


def technical_node(state: GalaxyState) -> dict:
    """
    Technical support worker node — resolves fault and connectivity queries.
    Reads the customer message from state, invokes Claude with technical
    specialist context, and returns the resolution as worker_result.
    """
    # Get LangFuse callback handler for tracing this LLM call
    langfuse_handler = get_langfuse_handler()

    # Extract the latest customer message from conversation history
    latest_message = state["messages"][-1].content

    print(f"[Technical Agent] Processing query: '{latest_message[:60]}...'")

    # Invoke Claude with technical specialist system prompt
    response = llm.invoke(
        [
            SystemMessage(content=TECHNICAL_SYSTEM_PROMPT),
            HumanMessage(content=latest_message)
        ],
        config={"callbacks": [langfuse_handler]}
    )

    print(f"[Technical Agent] Resolution generated.")

    # Return partial state update with worker result and worker name
    return {
        "worker_result": response.content.strip(),
        "log_entry": {
            **state.get("log_entry", {}),
            "worker_used": "technical"
        }
    }