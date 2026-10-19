"""
Module:
    main.py
Description:
    Entry point for the Galaxy Telecom multi-agent support system.
    Provides a command-line interface for testing the LangGraph agent
    graph interactively. Accepts customer messages from the terminal,
    invokes the compiled graph, and displays the final response.
    Each run is fully traced in LangFuse and logged to the SQLite
    audit database regardless of resolution or escalation outcome."""

from langchain_core.messages import HumanMessage
from graph.workflow import galaxy_support_app


def run_support_agent(customer_message: str) -> str:
    """
    Invoke the Galaxy Telecom multi-agent support graph with a customer message.
    Initialises the graph state with the customer message and runs the full
    agent pipeline — classification, routing, resolution, and logging.

    Args:
        customer_message: The customer's support query as a plain text string.

    Returns:
        str: The final assembled response to return to the customer.
    """
    print("\n" + "=" * 60)
    print("GALAXY TELECOM — MULTI-AGENT SUPPORT SYSTEM")
    print("=" * 60)
    print(f"Customer: {customer_message}")
    print("-" * 60)

    # Initialise the graph state with the customer message
    # All other state fields start empty and are populated by nodes
    initial_state = {
        "messages": [HumanMessage(content=customer_message)],
        "intent": "",
        "worker_result": "",
        "final_response": "",
        "escalated": False,
        "escalation_reason": "",
        "log_entry": {}
    }

    # Invoke the compiled LangGraph Runnable with the initial state
    result = galaxy_support_app.invoke(initial_state)

    # Extract and display the final response
    final_response = result.get("final_response", "No response generated.")

    print("-" * 60)
    print(f"Response:\n{final_response}")
    print("=" * 60 + "\n")

    return final_response


if __name__ == "__main__":
    # Test the system with a set of sample customer messages
    # covering each intent category including escalation
    test_messages = [
        "My bill this month is much higher than usual, can you help?",
        "My broadband has been down since yesterday morning.",
        "I am thinking about cancelling my contract with Galaxy Telecom.",
        "What 5G plans do you offer and what is the coverage like?",
        "Purple elephant dancing in the moonlight."
    ]

    for message in test_messages:
        run_support_agent(message)
        print("\n")