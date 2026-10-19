"""
Module:
    escalation.py
Description:
    Escalation handler for the Galaxy Telecom multi-agent support system.
    Invoked by the coordinator when intent cannot be classified with
    sufficient confidence (intent == "unknown"). Sends a Slack alert to
    the human agent team, writes a structured escalation record to the
    SQLite audit log, and sets the escalated flag in state.
    This node represents a graceful degradation strategy — the system
    fails safely by escalating rather than producing an unreliable response."""

import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from audit.logger import write_log
from graph.state import GalaxyState

# Load environment variables
load_dotenv()

# Slack webhook URL for human agent escalation alerts
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def send_slack_alert(customer_message: str, escalation_reason: str):
    """
    Send a Slack notification to the human agent team channel.
    Posts a structured message containing the customer query and
    the reason the AI agent could not classify it.
    """
    # Build the Slack message payload
    payload = {
        "text": ":warning: *Galaxy Telecom — Human Escalation Required*",
        "attachments": [
            {
                "color": "#FF0000",
                "fields": [
                    {
                        "title": "Customer Message",
                        "value": customer_message,
                        "short": False
                    },
                    {
                        "title": "Escalation Reason",
                        "value": escalation_reason,
                        "short": False
                    },
                    {
                        "title": "Timestamp",
                        "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "short": True
                    },
                    {
                        "title": "Priority",
                        "value": "High — requires immediate human response",
                        "short": True
                    }
                ]
            }
        ]
    }

    # Post the alert to the Slack webhook
    try:
        response = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            print(f"[Escalation Handler] Slack alert sent successfully.")
        else:
            print(f"[Escalation Handler] Slack alert failed: {response.status_code}")
    except Exception as e:
        print(f"[Escalation Handler] Slack alert error: {e}")


def escalation_node(state: GalaxyState) -> dict:
    """
    Escalation handler node — triggers human escalation for unclassifiable requests.
    Sends a Slack alert to the human agent team, writes an escalation record
    to the audit log, and returns updated state with escalation flags set.
    """
    # Extract the latest customer message from conversation history
    latest_message = state["messages"][-1].content

    # Define the escalation reason
    escalation_reason = (
        "AI coordinator could not classify customer intent with sufficient "
        "confidence. Manual review and response required."
    )

    print(f"[Escalation Handler] Escalating request to human agent team.")

    # Send Slack alert to human agent team
    send_slack_alert(latest_message, escalation_reason)

    # Build the final response to return to the customer
    final_response = (
        "Thank you for contacting Galaxy Telecom. We want to make sure your "
        "query receives the attention it deserves. One of our specialist agents "
        "will review your message and get back to you within 2 hours. "
        "We apologise for any inconvenience."
    )

    # Build the audit log entry for this escalation
    log_entry = {
        "customer_message": latest_message,
        "intent": "unknown",
        "worker_used": "escalation_handler",
        "worker_result": "Escalated to human agent",
        "final_response": final_response,
        "escalated": True,
        "escalation_reason": escalation_reason
    }

    # Write the escalation record to the SQLite audit log
    write_log(log_entry)

    print(f"[Escalation Handler] Audit log entry written.")

    # Return partial state update with escalation flags and final response
    return {
        "escalated": True,
        "escalation_reason": escalation_reason,
        "final_response": final_response,
        "log_entry": log_entry
    }