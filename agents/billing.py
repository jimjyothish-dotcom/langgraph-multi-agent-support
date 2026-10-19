"""
Module:
    billing.py
Description:
    Billing worker agent for the Galaxy Telecom multi-agent support system.
    Handles customer queries related to bills, charges, payments, invoices,
    and refunds. Invoked by the coordinator when intent is classified as
    "billing". Generates a resolution response using Claude and writes
    the result to the worker_result field in state for the responder node."""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from observability.langfuse_config import get_langfuse_handler
from graph.state import GalaxyState

# Load environment variables
load_dotenv()

# Initialise the Claude LLM for billing query resolution
llm = ChatAnthropic(
    model=os.getenv("CLAUDE_MODEL_STANDARD"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=float(os.getenv("CLAUDE_TEMPERATURE_BALANCED", "0.3"))
)

# System prompt for billing query resolution
BILLING_SYSTEM_PROMPT = """
You are the billing specialist for Galaxy Telecom's customer support team.
You handle all queries related to bills, charges, payments, invoices, and refunds.

Guidelines:
- Be empathetic and professional
- Acknowledge the customer's concern clearly
- Provide a clear, concise resolution or next step
- If you need account-specific details you don't have, explain what the
  customer should do next (e.g. call billing team, check online account)
- Keep your response under 150 words
- Do not make up specific account details or balances

Galaxy Telecom billing contact: billing@galaxy-telecom.co.uk
Billing team hours: Monday to Friday, 9am to 6pm GMT
"""


def billing_node(state: GalaxyState) -> dict:
    """
    Billing worker node — resolves billing and payment queries.
    Reads the customer message from state, invokes Claude with billing
    specialist context, and returns the resolution as worker_result.
    """
    # Get LangFuse callback handler for tracing this LLM call
    langfuse_handler = get_langfuse_handler()

    # Extract the latest customer message from conversation history
    latest_message = state["messages"][-1].content

    print(f"[Billing Agent] Processing query: '{latest_message[:60]}...'")

    # Invoke Claude with billing specialist system prompt
    response = llm.invoke(
        [
            SystemMessage(content=BILLING_SYSTEM_PROMPT),
            HumanMessage(content=latest_message)
        ],
        config={"callbacks": [langfuse_handler]}
    )

    print(f"[Billing Agent] Resolution generated.")

    # Return partial state update with worker result and worker name
    return {
        "worker_result": response.content.strip(),
        "log_entry": {
            **state.get("log_entry", {}),
            "worker_used": "billing"
        }
    }