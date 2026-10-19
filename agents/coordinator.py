"""
Module:
    coordinator.py
Description:
    Coordinator agent for the Galaxy Telecom multi-agent support system.
    Acts as the orchestrator — receives the customer message, classifies
    intent using Claude, and sets the routing decision in state.
    The coordinator never resolves requests itself — it delegates to the
    appropriate worker agent via conditional edges in the graph.
    Possible intent values: "billing", "technical", "retention", "kb", "unknown"
    "unknown" triggers the escalation handler."""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from observability.langfuse_config import get_langfuse_handler
from graph.state import GalaxyState

# Load environment variables
load_dotenv()

# Initialise the Claude LLM for intent classification
llm = ChatAnthropic(
    model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=float(os.getenv("CLAUDE_TEMPERATURE_PRECISE", "0"))
)

# System prompt for intent classification
COORDINATOR_SYSTEM_PROMPT = """
You are the coordinator for Galaxy Telecom's customer support system.
Your sole responsibility is to classify the customer's intent from their message.

Respond with exactly one word — the intent label. Nothing else.

Intent labels:
- billing     — questions about bills, charges, payments, invoices, refunds
- technical   — faults, outages, connectivity issues, device problems
- retention   — cancellations, contract renewals, switching providers, complaints
- kb          — general product questions, plan information, coverage queries
- unknown     — anything you cannot confidently classify into the above categories

Examples:
"My bill is higher than usual" -> billing
"My internet keeps dropping" -> technical
"I want to cancel my contract" -> retention
"What are your 5G coverage areas?" -> kb
"Purple elephant dancing" -> unknown

Respond with the intent label only. No explanation. No punctuation.
"""


def coordinator_node(state: GalaxyState) -> dict:
    """
    Coordinator node — classifies customer intent and sets routing decision.
    Reads the latest customer message from state, invokes Claude to classify
    intent, and returns the intent field for conditional edge routing.
    """
    # Get LangFuse callback handler for tracing this LLM call
    langfuse_handler = get_langfuse_handler()

    # Extract the latest customer message from conversation history
    latest_message = state["messages"][-1].content

    # Invoke Claude to classify intent
    response = llm.invoke(
        [
            SystemMessage(content=COORDINATOR_SYSTEM_PROMPT),
            HumanMessage(content=latest_message)
        ],
        config={"callbacks": [langfuse_handler]}
    )

    # Extract and sanitise the intent label from the response
    intent = response.content.strip().lower()

    # Validate against known intents — default to unknown if unexpected value
    valid_intents = {"billing", "technical", "retention", "kb", "unknown"}
    if intent not in valid_intents:
        intent = "unknown"

    print(f"[Coordinator] Intent classified as: '{intent}'")

    # Return partial state update — only the fields this node changes
    return {"intent": intent}