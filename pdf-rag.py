"""
1. Run: `./cookbook/run_pgvector.sh` to start a postgres container with pgvector
2. Run: `pip install openai sqlalchemy 'psycopg[binary]' pgvector phidata` to install the dependencies
3. Run: `python cookbook/rag/02_agentic_rag_pgvector.py` to run the agent
"""
import os
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.ollama import OllamaEmbedder
from phi.playground import Playground, serve_playground_app

from phi.knowledge.pdf import PDFKnowledgeBase,PDFReader
from phi.vectordb.pgvector import PgVector, SearchType
from phi.vectordb.qdrant import Qdrant

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = "realestatekb"

vector_db = Qdrant(
    collection=collection_name,
    url=qdrant_url,
    api_key=api_key,
)


# Create a knowledge base of PDFs from URLs
knowledge_base = PDFKnowledgeBase(
    path="./pdfs",
    optimize_on=5,
    vector_db=vector_db,
    # vector_db=PgVector(
    #     table_name="realestatekb",
    #     db_url=db_url,
    #     search_type=SearchType.hybrid,
    #     embedder=OllamaEmbedder(),
    # ),
    reader=PDFReader(chunk=True)
)
# # Load the knowledge base: Comment after first run as the knowledge base is already loaded
# knowledge_base.load(upsert=True)

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    role="Urban planner in canada",
    description="""
    You are an  urban planner in canada. Your role is to answer questions from users based on available knowledgebase.
    """,
    instructions=["Analyse the question from user before searching knowledgebase",
                  "Always provide factual information from knowledgebase, do not assume",
                  "answer concisely in one line and show references",
                  "Analyse the response from knowledgebase and final response should answer the question from user."],
    knowledge=knowledge_base,
    # Add a tool to search the knowledge base which enables agentic RAG.
    # This is enabled by default when `knowledge` is provided to the Agent.
    search_knowledge=True,
    show_tool_calls=False,
    add_context=True,
    add_history_to_messages=True,
    read_chat_history=True,
    markdown=True,
)
# agent.print_response("can i further divide OSFPD? answer concisely in one line and show references", stream=True)
# agent.print_response("Can i build 70 senior housing units in my project", stream=True)
# agent.print_response(
#     "Hi, i want to make a 3 course meal. Can you recommend some recipes. "
#     "I'd like to start with a soup, then im thinking a thai curry for the main course and finish with a dessert",
#     stream=True,
# )

app = Playground(agents=[agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("agentic-rag:app", reload=True)
