"""
Module:
    langfuse_config.py
Description:
    LangFuse observability configuration for the Galaxy Telecom
    multi-agent support system. Initialises the LangFuse callback
    handler that automatically traces every LLM call, tool invocation,
    and agent step through the graph. Credentials are read automatically
    from environment variables by the LangFuse SDK.
    LangFuse provides LLM-level observability — token usage, latency,
    and full execution traces — complementing the SQLite audit log
    which captures business-level outcomes."""

import os
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

# Load environment variables from .env file
load_dotenv()

# Set LangFuse credentials as environment variables
# The SDK reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST
# automatically when CallbackHandler is initialised
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


def get_langfuse_handler():
    """
    Initialise and return a LangFuse CallbackHandler instance.
    Pass this handler into every LangGraph node that invokes an LLM
    to enable automatic tracing of the full agent execution.
    Credentials are picked up automatically from environment variables.
    """
    handler = CallbackHandler()
    return handler