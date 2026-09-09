import faiss
import numpy as np
import pickle
import os

INDEX_DIR = "data"


def build_faiss_index(chunks, repo_name):
    """
    Build a FAISS index from embedded chunks and save it to disk,
    along with the chunk metadata (file_path, text, etc.) needed
    to interpret search results later.
    """
    # Stack all embeddings into one big array — FAISS needs this format
    vectors = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")

    dimension = vectors.shape[1]  # length of each embedding vector

    # A simple, exact-search index — no approximation, good for
    # a single-repo, mini-project scale (hundreds to a few thousand chunks)
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Save the index itself
    index_path = os.path.join(INDEX_DIR, f"{repo_name}.faiss")
    faiss.write_index(index, index_path)

    # Save the chunk metadata separately (FAISS only stores vectors,
    # not the text/file_path — we need this to make sense of results)
    metadata_path = os.path.join(INDEX_DIR, f"{repo_name}_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index ({len(chunks)} vectors) to {index_path}")
    return index_path, metadata_path


def load_faiss_index(repo_name):
    """
    Load a previously saved FAISS index and its metadata back into memory.
    """
    index_path = os.path.join(INDEX_DIR, f"{repo_name}.faiss")
    metadata_path = os.path.join(INDEX_DIR, f"{repo_name}_metadata.pkl")

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def search_index(index, chunks, query_embedding, top_k=5):
    """
    Search the FAISS index for the chunks most similar to a query embedding.
    Returns the top_k matching chunks (with their file_path, text, etc.)
    """
    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results


if __name__ == "__main__":
    from github_loader import clone_repository, get_repo_name
    from processor import load_repository_files, chunk_repository_files
    from embeddings import embed_chunks, embed_query

    test_url = input("Enter a public GitHub repository URL: ").strip()
    repo_path = clone_repository(test_url)
    repo_name = get_repo_name(test_url)

    files = load_repository_files(repo_path)
    chunks = chunk_repository_files(files)
    print(f"Embedding {len(chunks)} chunks...")
    chunks = embed_chunks(chunks)

    build_faiss_index(chunks, repo_name)

    # Try a test search
    index, loaded_chunks = load_faiss_index(repo_name)
    test_question = input("\nAsk a test question about this repo: ").strip()
    query_vector = embed_query(test_question)

    results = search_index(index, loaded_chunks, query_vector, top_k=3)

    print("\nTop matching chunks:")
    for r in results:
        print(f" - {r['file_path']} [chunk {r['chunk_index']}]: {r['text'][:100]}...")