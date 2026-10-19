"""
Module:
    kb.py
Description:
    Knowledge base worker agent for the Galaxy Telecom multi-agent support
    system. Handles general product and policy queries by performing a
    semantic search against the Galaxy Telecom product knowledge base
    stored in Qdrant cloud. Retrieves the most relevant chunks and passes
    them to Claude to generate a grounded, accurate response.
    Invoked by the coordinator when intent is classified as "kb"."""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from observability.langfuse_config import get_langfuse_handler
from graph.state import GalaxyState

# Load environment variables
load_dotenv()

# Initialise the Claude LLM for response generation
llm = ChatAnthropic(
    model=os.getenv("CLAUDE_MODEL_STANDARD"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=float(os.getenv("CLAUDE_TEMPERATURE_BALANCED", "0.3"))
)
# Initialise Qdrant client using existing galaxy-telecom-kb cluster
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Initialise the same embedding model used during ingestion
# Must match the model used to build the Qdrant collection
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Qdrant collection name for Galaxy Telecom product knowledge base
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "galaxy-telecom-products")

# Number of top results to retrieve from Qdrant
TOP_K = 3

# System prompt for KB-grounded response generation
KB_SYSTEM_PROMPT = """
You are the product knowledge specialist for Galaxy Telecom's customer support team.
You answer general questions about Galaxy Telecom products, plans, coverage, and policies.

You will be given relevant excerpts from the Galaxy Telecom knowledge base.
Use only the information provided in those excerpts to answer the customer's question.
If the excerpts do not contain enough information to answer, say so honestly and
direct the customer to contact Galaxy Telecom directly.

Guidelines:
- Be clear, accurate, and helpful
- Base your answer strictly on the provided knowledge base excerpts
- Do not make up product details, prices, or policies
- Keep your response under 200 words

Galaxy Telecom general enquiries: info@galaxy-telecom.co.uk
"""


def search_knowledge_base(query: str) -> str:
    """
    Perform semantic search against the Qdrant knowledge base.
    Embeds the query using the same model used during ingestion,
    retrieves the top K most relevant chunks, and returns them
    as a single concatenated context string.
    """
    # Generate query embedding using the same model as ingestion
    query_vector = embedding_model.encode(query).tolist()

    # Search Qdrant for the most semantically similar chunks
    # query_points is the current method in qdrant-client v1.x
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K
    ).points

    # Extract text payload from each result and join into context block
    chunks = [result.payload.get("text", "") for result in results]
    context = "\n\n".join(chunks)

    print(f"[KB Agent] Retrieved {len(chunks)} chunks from Qdrant.")
    return context


def kb_node(state: GalaxyState) -> dict:
    """
    Knowledge base worker node — answers product and policy queries using RAG.
    Performs semantic search against Qdrant, passes retrieved context to
    Claude for grounded response generation, and returns worker_result.
    """
    # Get LangFuse callback handler for tracing this LLM call
    langfuse_handler = get_langfuse_handler()

    # Extract the latest customer message from conversation history
    latest_message = state["messages"][-1].content

    print(f"[KB Agent] Processing query: '{latest_message[:60]}...'")

    # Retrieve relevant knowledge base chunks from Qdrant
    context = search_knowledge_base(latest_message)

    # Build the augmented prompt with retrieved context
    augmented_prompt = f"""
Customer question: {latest_message}

Relevant knowledge base excerpts:
{context}

Please answer the customer's question based on the excerpts above.
"""

    # Invoke Claude with KB context for grounded response generation
    response = llm.invoke(
        [
            SystemMessage(content=KB_SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ],
        config={"callbacks": [langfuse_handler]}
    )

    print(f"[KB Agent] Resolution generated.")

    # Return partial state update with worker result and worker name
    return {
        "worker_result": response.content.strip(),
        "log_entry": {
            **state.get("log_entry", {}),
            "worker_used": "kb"
        }
    }