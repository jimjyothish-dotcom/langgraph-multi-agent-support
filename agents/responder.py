"""
Module:
    responder.py
Description:
    Response assembler for the Galaxy Telecom multi-agent support system.
    Invoked after a worker agent has generated a resolution. Takes the
    worker_result from state, applies a consistent Galaxy Telecom brand
    tone and structure, assembles the final customer response, and writes
    the completed interaction to the SQLite audit log.
    This node is the final step in the graph before END for all
    successfully resolved requests."""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from observability.langfuse_config import get_langfuse_handler
from audit.logger import write_log
from graph.state import GalaxyState

# Load environment variables
load_dotenv()

# Initialise the Claude LLM for response assembly
llm = ChatAnthropic(
    model=os.getenv("CLAUDE_MODEL_STANDARD"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=float(os.getenv("CLAUDE_TEMPERATURE_BALANCED", "0.3"))
)

# System prompt for final response assembly
RESPONDER_SYSTEM_PROMPT = """
You are the response assembler for Galaxy Telecom's customer support system.
You receive a draft resolution from a specialist agent and your job is to
polish it into a final, consistently branded customer response.

Guidelines:
- Open with a warm, professional greeting
- Present the resolution clearly and concisely
- Close with an offer to help further
- Maintain a friendly, professional tone throughout
- Keep the response under 200 words
- Do not add information beyond what is in the draft resolution

Sign off every response as:
Galaxy Telecom Customer Support Team
"""


def responder_node(state: GalaxyState) -> dict:
    """
    Responder node — assembles the final branded customer response.
    Takes the worker_result from state, polishes it using Claude,
    writes the completed interaction to the audit log, and returns
    the final_response field to conclude the graph run.
    """
    # Get LangFuse callback handler for tracing this LLM call
    langfuse_handler = get_langfuse_handler()

    # Extract the worker result and customer message from state
    worker_result = state.get("worker_result", "")
    latest_message = state["messages"][-1].content
    intent = state.get("intent", "")

    print(f"[Responder] Assembling final response for intent: '{intent}'")

    # Build the prompt for response assembly
    assembly_prompt = f"""
Customer message: {latest_message}

Draft resolution from specialist agent:
{worker_result}

Please polish this into a final, consistently branded customer response.
"""

    # Invoke Claude to assemble the final response
    response = llm.invoke(
        [
            SystemMessage(content=RESPONDER_SYSTEM_PROMPT),
            HumanMessage(content=assembly_prompt)
        ],
        config={"callbacks": [langfuse_handler]}
    )

    final_response = response.content.strip()

    print(f"[Responder] Final response assembled.")

    # Build the complete audit log entry for this interaction
    log_entry = {
        **state.get("log_entry", {}),
        "customer_message": latest_message,
        "intent": intent,
        "worker_result": worker_result,
        "final_response": final_response,
        "escalated": False,
        "escalation_reason": ""
    }

    # Write the completed interaction to the SQLite audit log
    write_log(log_entry)

    print(f"[Responder] Audit log entry written.")

    # Return partial state update with the final assembled response
    return {
        "final_response": final_response,
        "log_entry": log_entry
    }