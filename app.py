from flask import Flask, render_template, request, jsonify
import os

from github_loader import clone_repository, get_repo_name
from processor import load_repository_files, chunk_repository_files
from embeddings import embed_chunks
from retriever import build_faiss_index, load_faiss_index
from agent import answer_with_agent

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/load_repo", methods=["POST"])
def load_repo():
    """Clone + index a repository. Returns repo_name to use in later questions."""
    data = request.get_json()
    repo_url = data.get("repo_url", "").strip()

    if not repo_url:
        return jsonify({"error": "Please enter a repository URL."}), 400

    try:
        repo_path = clone_repository(repo_url)
        repo_name = get_repo_name(repo_url)

        index_path = os.path.join("data", f"{repo_name}.faiss")

        if os.path.exists(index_path):
            index, chunks = load_faiss_index(repo_name)
        else:
            files = load_repository_files(repo_path)
            if len(files) == 0:
                return jsonify({"error": "No supported files were found in this repository."}), 400

            chunks = chunk_repository_files(files)
            chunks = embed_chunks(chunks)
            build_faiss_index(chunks, repo_name)
            index, chunks = load_faiss_index(repo_name)

        return jsonify({"repo_name": repo_name, "chunk_count": len(chunks)})

    except ValueError as e:
        return jsonify({"error": f"Invalid repository URL: {e}"}), 400
    except RuntimeError as e:
        return jsonify({"error": f"Could not clone repository: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {e}"}), 500


@app.route("/api/ask", methods=["POST"])
def ask():
    """Answer a question about an already-loaded repository."""
    data = request.get_json()
    repo_name = data.get("repo_name", "").strip()
    question = data.get("question", "").strip()

    if not repo_name or not question:
        return jsonify({"error": "Missing repository or question."}), 400

    try:
        # Reload from disk each time — simple and stateless, no server-side
        # session needed, since retriever.py already persists the index.
        index, chunks = load_faiss_index(repo_name)
        result = answer_with_agent(question, index, chunks)

        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "question_type": result["question_type"]
        })
    except Exception as e:
        return jsonify({"error": f"Something went wrong while answering: {e}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)