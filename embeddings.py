import os
from dotenv import load_dotenv
from google import genai

# Load the API key from .env (never hardcode keys in code)
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Google's embedding model — free tier, no cost for a mini-project's usage
MODEL_NAME = "gemini-embedding-001"


def embed_chunks(chunks):
    """
    Take the list of chunks from processor.py (each with a "text" field)
    and add an "embedding" field to each one, using Gemini's embedding API.
    """
    for i, chunk in enumerate(chunks):
        result = client.models.embed_content(
            model=MODEL_NAME,
            contents=chunk["text"]
        )
        chunk["embedding"] = result.embeddings[0].values

        # Simple progress indicator so you can see it's working on big repos
        if (i + 1) % 50 == 0:
            print(f"Embedded {i + 1}/{len(chunks)} chunks...")

    return chunks


def embed_query(query_text):
    """
    Turn a single user question into an embedding, the same way chunks
    are embedded, so we can compare the question to the chunks later.
    """
    result = client.models.embed_content(
        model=MODEL_NAME,
        contents=query_text
    )
    return result.embeddings[0].values


if __name__ == "__main__":
    # Quick manual test
    from github_loader import clone_repository
    from processor import load_repository_files, chunk_repository_files

    test_url = input("Enter a public GitHub repository URL: ").strip()
    repo_path = clone_repository(test_url)

    files = load_repository_files(repo_path)
    chunks = chunk_repository_files(files)
    print(f"Embedding {len(chunks)} chunks... (this calls the API, may take a bit)")

    chunks = embed_chunks(chunks)

    print(f"\nDone. Each chunk now has an embedding.")
    print(f"Example: chunk 0 from '{chunks[0]['file_path']}'")
    print(f"Embedding length: {len(chunks[0]['embedding'])} numbers")
    print(f"First 5 values: {chunks[0]['embedding'][:5]}")