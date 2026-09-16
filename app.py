import streamlit as st
from github_loader import clone_repository, get_repo_name
from processor import load_repository_files, chunk_repository_files
from embeddings import embed_chunks, embed_query
from retriever import build_faiss_index, load_faiss_index, search_index
from llm import generate_answer
import os

st.set_page_config(page_title="GitHub Repo Q&A Chatbot", page_icon="🤖")
st.title("🤖 GitHub Repository Q&A Chatbot")
st.write("Enter a public GitHub repository URL, then ask questions about it.")

# session_state keeps values between reruns (Streamlit reruns the whole
# script on every interaction, so we store the index/chunks/chat here
# instead of recomputing them every time)
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.repo_name = None
    st.session_state.chat_history = []


repo_url = st.text_input("Public GitHub repository URL")

if st.button("Load Repository"):
    if not repo_url.strip():
        st.error("Please enter a repository URL.")
    else:
        with st.spinner("Cloning repository..."):
            repo_path = clone_repository(repo_url)
            repo_name = get_repo_name(repo_url)

        index_path = os.path.join("data", f"{repo_name}.faiss")

        if os.path.exists(index_path):
            # Already indexed before — just load it, no need to redo everything
            with st.spinner("Loading existing index..."):
                index, chunks = load_faiss_index(repo_name)
        else:
            with st.spinner("Reading and filtering files..."):
                files = load_repository_files(repo_path)

            with st.spinner("Splitting files into chunks..."):
                chunks = chunk_repository_files(files)

            with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
                chunks = embed_chunks(chunks)

            with st.spinner("Building FAISS index..."):
                build_faiss_index(chunks, repo_name)
                index, chunks = load_faiss_index(repo_name)

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.repo_name = repo_name
        st.session_state.chat_history = []

        st.success(f"Repository indexed! ({len(chunks)} chunks ready)")


# Only show the chat interface once a repo has been loaded
if st.session_state.index is not None:
    st.subheader(f"Ask about: {st.session_state.repo_name}")

    # Show past messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about this repository...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                query_vector = embed_query(question)
                results = search_index(
                    st.session_state.index,
                    st.session_state.chunks,
                    query_vector,
                    top_k=3
                )
                answer = generate_answer(question, results)

                st.write(answer)

                with st.expander("Sources"):
                    for r in results:
                        st.write(f"- `{r['file_path']}`")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})