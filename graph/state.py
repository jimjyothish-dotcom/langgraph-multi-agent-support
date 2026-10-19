"""
Module:
    state.py
Description:
    Defines the shared state schema for the Galaxy Telecom multi-agent
    support system. The GalaxyState TypedDict is passed through every node
    in the LangGraph graph, accumulating data as each agent executes.
    All nodes read from and write partial updates back to this single
    state object — LangGraph merges updates automatically after each node."""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class GalaxyState(TypedDict):
    # Conversation history — append-only via add_messages reducer.
    # Stores HumanMessage, AIMessage, and ToolMessage objects in sequence.
    messages: Annotated[list, add_messages]

    # Intent classification result set by the coordinator node.
    # Drives conditional edge routing to the appropriate worker agent.
    # Possible values: "billing", "technical", "retention", "kb", "unknown"
    intent: str

    # Structured result returned by whichever worker agent was invoked.
    # Read by the responder node to compose the final customer response.
    worker_result: str

    # Final response string assembled by the responder node.
    # This is what gets returned to the customer.
    final_response: str

    # Escalation flag — set to True when coordinator cannot classify intent
    # with sufficient confidence and request must be routed to a human agent.
    escalated: bool

    # Human-readable explanation of why escalation was triggered.
    # Included in the Slack alert and audit log entry.
    escalation_reason: str

    # Structured audit log entry populated at the end of every graph run.
    # Written to SQLite by the logger regardless of resolution or escalation.
    log_entry: dict