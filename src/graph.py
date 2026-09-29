import re
from typing import List, TypedDict

from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph

from src import config


class AgentState(TypedDict):
    question: str
    context: List[str]
    pages: List[int]
    top_similarity: float
    answer: str
    grounded: float
    score: float
    attempts: int


GENERATE_PROMPT = """You are a strict assistant answering questions about the Agentic AI eBook.
Answer using ONLY the context below. Do not use outside knowledge.
If the context does not contain the answer, reply exactly: {refusal}

Context:
{context}

Question: {question}

Answer:"""

GRADE_PROMPT = """You are a fact-checking grader. Rate how well the ANSWER is supported by the CONTEXT.
1.0 = every claim is directly supported, 0.0 = unsupported or contradicted.
Reply with only a number between 0 and 1.

CONTEXT:
{context}

ANSWER:
{answer}"""


def build_rag_graph(index_name=config.INDEX_NAME):
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=config.get_embeddings())
    llm = config.get_llm()

    def retrieve_node(state: AgentState):
        results = vectorstore.similarity_search_with_score(state["question"], k=config.TOP_K)
        return {
            "context": [doc.page_content for doc, _ in results],
            "pages": [int(doc.metadata.get("page", 0)) for doc, _ in results],
            "top_similarity": max((float(s) for _, s in results), default=0.0),
        }

    def generate_node(state: AgentState):
        prompt = GENERATE_PROMPT.format(
            refusal=config.REFUSAL_TEXT,
            context="\n\n---\n\n".join(state["context"]),
            question=state["question"],
        )
        if state["attempts"] > 0:
            prompt += "\n\nBe extra careful: include only statements directly supported by the context."
        answer = llm.invoke(prompt).content.strip()
        return {"answer": answer, "attempts": state["attempts"] + 1}

    def grade_node(state: AgentState):
        prompt = GRADE_PROMPT.format(context="\n\n---\n\n".join(state["context"]), answer=state["answer"])
        match = re.search(r"\d*\.?\d+", llm.invoke(prompt).content)
        grounded = min(1.0, max(0.0, float(match.group()))) if match else 0.0
        retrieval = min(1.0, state["top_similarity"] / 0.6)
        return {"grounded": grounded, "score": round(0.7 * grounded + 0.3 * retrieval, 2)}

    def refuse_node(state: AgentState):
        return {"answer": config.REFUSAL_TEXT, "grounded": 0.0, "score": 0.0}

    def route_after_retrieve(state: AgentState):
        return "generate" if state["context"] and state["top_similarity"] >= config.RELEVANCE_THRESHOLD else "refuse"

    def route_after_generate(state: AgentState):
        return "refuse" if state["answer"].startswith(config.REFUSAL_TEXT) else "grade"

    def route_after_grade(state: AgentState):
        return "generate" if state["grounded"] < config.GROUNDED_THRESHOLD and state["attempts"] < config.MAX_ATTEMPTS else END

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("refuse", refuse_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges("retrieve", route_after_retrieve, {"generate": "generate", "refuse": "refuse"})
    workflow.add_conditional_edges("generate", route_after_generate, {"grade": "grade", "refuse": "refuse"})
    workflow.add_conditional_edges("grade", route_after_grade, {"generate": "generate", END: END})
    workflow.add_edge("refuse", END)
    return workflow.compile()


def run(graph, query):
    initial = {
        "question": query, "context": [], "pages": [], "top_similarity": 0.0,
        "answer": "", "grounded": 0.0, "score": 0.0, "attempts": 0,
    }
    return graph.invoke(initial)


def to_payload(query, state):
    return {
        "query": query,
        "final_answer": state["answer"],
        "retrieved_context_chunks": state["context"],
        "confidence_score": state["score"],
    }


def ask(graph, query):
    return to_payload(query, run(graph, query))
