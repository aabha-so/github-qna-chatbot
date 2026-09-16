import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Using Flash — fast, cheap, and well within the free tier for a mini-project
MODEL_NAME = "gemini-2.5-flash"


def build_context(retrieved_chunks):
    """
    Turn the retrieved chunks into a single text block the LLM can read,
    labeling each piece with its source file so the model can cite it.
    """
    context_parts = []
    for chunk in retrieved_chunks:
        header = f"--- From file: {chunk['file_path']} ---"
        context_parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(context_parts)


def generate_answer(question, retrieved_chunks):
    """
    Generate a grounded answer to the user's question using only the
    retrieved repository content. Returns the answer text.
    """
    context = build_context(retrieved_chunks)

    prompt = f"""You are a helpful assistant that answers questions about a GitHub repository.

Use ONLY the repository content below to answer the question. Do not use
any outside knowledge or make assumptions beyond what is shown here.

If the answer is not clearly present in the content below, say:
"I couldn't find this in the repository content I have access to."

Mention the relevant file name(s) in your answer when possible.

Repository content:
{context}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


if __name__ == "__main__":
    # Quick manual test — reuses the full pipeline so far
    from github_loader import clone_repository, get_repo_name
    from processor import load_repository_files, chunk_repository_files
    from embeddings import embed_chunks, embed_query
    from retriever import build_faiss_index, load_faiss_index, search_index

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
    query_vector = embed_query(question)
    results = search_index(index, loaded_chunks, query_vector, top_k=3)

    print("\nGenerating answer...\n")
    answer = generate_answer(question, results)

    print("ANSWER:")
    print(answer)

    print("\nSOURCES:")
    for r in results:
        print(f" - {r['file_path']}")