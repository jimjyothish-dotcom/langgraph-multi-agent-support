"""
Module:
    workflow.py
Description:
    LangGraph workflow definition for the Galaxy Telecom multi-agent
    support system. Constructs the StateGraph by registering all agent
    nodes and defining the edges between them. The coordinator node
    uses a conditional edge with a router function to dynamically route
    each customer request to the appropriate worker agent at runtime.
    The compiled graph is exported as a reusable Runnable for invocation
    from main.py or a FastAPI endpoint."""

from langgraph.graph import StateGraph, END
from graph.state import GalaxyState
from agents.coordinator import coordinator_node
from agents.billing import billing_node
from agents.technical import technical_node
from agents.retention import retention_node
from agents.kb import kb_node
from agents.escalation import escalation_node
from agents.responder import responder_node


def route_from_coordinator(state: GalaxyState) -> str:
    """
    Router function for the conditional edge from the coordinator node.
    Reads the intent field set by the coordinator and returns the name
    of the next node to execute. This function is evaluated at runtime
    after every coordinator node execution — enabling dynamic control flow.

    Returns:
        str: Name of the next node to route to.
    """
    intent = state.get("intent", "unknown")

    routing_map = {
        "billing":   "billing_node",
        "technical": "technical_node",
        "retention": "retention_node",
        "kb":        "kb_node",
        "unknown":   "escalation_node"
    }

    next_node = routing_map.get(intent, "escalation_node")
    print(f"[Router] Routing intent '{intent}' to '{next_node}'")
    return next_node


def build_graph() -> StateGraph:
    """
    Construct and compile the Galaxy Telecom multi-agent support graph.
    Registers all nodes, sets the entry point, defines normal and
    conditional edges, and compiles into an executable Runnable.

    Returns:
        Compiled LangGraph Runnable ready for invocation.
    """
    # Initialise the StateGraph with the shared GalaxyState schema
    graph = StateGraph(GalaxyState)

    # -------------------------------------------------------------------------
    # Register all agent nodes
    # Each node is a Python function that reads state and returns a partial update
    # -------------------------------------------------------------------------
    graph.add_node("coordinator_node", coordinator_node)
    graph.add_node("billing_node",     billing_node)
    graph.add_node("technical_node",   technical_node)
    graph.add_node("retention_node",   retention_node)
    graph.add_node("kb_node",          kb_node)
    graph.add_node("escalation_node",  escalation_node)
    graph.add_node("responder_node",   responder_node)

    # -------------------------------------------------------------------------
    # Define the entry point — coordinator always runs first
    # -------------------------------------------------------------------------
    graph.set_entry_point("coordinator_node")

    # -------------------------------------------------------------------------
    # Conditional edge from coordinator to worker agents
    # The router function evaluates state at runtime to determine next node
    # -------------------------------------------------------------------------
    graph.add_conditional_edges(
        "coordinator_node",
        route_from_coordinator,
        {
            "billing_node":   "billing_node",
            "technical_node": "technical_node",
            "retention_node": "retention_node",
            "kb_node":        "kb_node",
            "escalation_node":"escalation_node"
        }
    )

    # -------------------------------------------------------------------------
    # Normal edges from worker agents to responder
    # All successfully resolved requests flow through the responder node
    # -------------------------------------------------------------------------
    graph.add_edge("billing_node",   "responder_node")
    graph.add_edge("technical_node", "responder_node")
    graph.add_edge("retention_node", "responder_node")
    graph.add_edge("kb_node",        "responder_node")

    # -------------------------------------------------------------------------
    # Terminal edges — graph execution ends after responder or escalation
    # -------------------------------------------------------------------------
    graph.add_edge("responder_node",  END)
    graph.add_edge("escalation_node", END)

    # Compile the graph definition into an executable Runnable
    app = graph.compile()

    print("[Workflow] Galaxy Telecom support graph compiled successfully.")
    return app


# Compile the graph on module import — ready for use in main.py
galaxy_support_app = build_graph()