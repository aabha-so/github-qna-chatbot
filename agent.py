from embeddings import embed_query
from retriever import search_index
from llm import generate_answer


def classify_question(question):
    """
    Simple rule-based classifier that decides what kind of repository
    search this question needs. This is our basic 'agentic' decision step
    — the agent deciding which action fits the question.
    """
    q = question.lower()

    if any(word in q for word in ["overview", "about", "purpose", "what is this", "architecture", "structure"]):
        return "broad"       # needs a wide view across many files
    if any(word in q for word in ["where", "which file", "located", "implement"]):
        return "specific"    # needs a few precise, targeted chunks
    return "general"


def choose_top_k(question_type):
    """
    Decide how many chunks to retrieve based on the question type.
    This is the 'tool selection' part of the agent — broad questions
    get more context, specific ones get fewer, more focused chunks.
    """
    if question_type == "broad":
        return 6
    elif question_type == "specific":
        return 3
    else:
        return 4


def answer_with_agent(question, index, chunks):
    """
    The main agent/controller:
    1. Classify the question
    2. Choose a retrieval strategy based on that classification
    3. Retrieve relevant chunks
    4. If the first retrieval looks too thin, retrieve again with more
       chunks (a simple, bounded 'second attempt' rather than giving up)
    5. Generate the final grounded answer
    """
    question_type = classify_question(question)
    top_k = choose_top_k(question_type)

    query_vector = embed_query(question)
    results = search_index(index, chunks, query_vector, top_k=top_k)

    # If the retrieved content is very short, it's probably not enough
    # context — try once more with a wider net before answering.
    total_context_length = sum(len(r["text"]) for r in results)
    if total_context_length < 200 and top_k < 8:
        results = search_index(index, chunks, query_vector, top_k=top_k + 4)

    answer = generate_answer(question, results)

    return {
        "answer": answer,
        "sources": [r["file_path"] for r in results],
        "question_type": question_type,
        "chunks_used": len(results)
    }


if __name__ == "__main__":
    from github_loader import clone_repository, get_repo_name
    from processor import load_repository_files, chunk_repository_files
    from embeddings import embed_chunks
    from retriever import build_faiss_index, load_faiss_index

    test_url = input("Enter a public GitHub repository URL: ").strip()
    repo_path = clone_repository(test_url)
    repo_name = get_repo_name(test_url)

    files = load_repository_files(repo_path)
    chunks = chunk_repository_files(files)
    print(f"Embedding {len(chunks)} chunks...")
    chunks = embed_chunks(chunks)
    build_faiss_index(chunks, repo_name)

    index, loaded_chunks = load_faiss_index(repo_name)

    question = input("\nAsk a question about this repo: ").strip()
    result = answer_with_agent(question, index, loaded_chunks)

    print(f"\n[Agent classified this as a '{result['question_type']}' question, used {result['chunks_used']} chunks]\n")
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f" - {s}")