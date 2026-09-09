import os

# Folders we never want to look inside
SKIP_FOLDERS = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist"}

# File extensions we care about for this project
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".java", ".cpp", ".c",
    ".html", ".css", ".md", ".txt",
    ".json", ".yaml", ".yml"
}

# Skip files bigger than this (avoids huge generated/data files)
MAX_FILE_SIZE_BYTES = 300_000  # about 300 KB


def is_supported_file(filename):
    """Check if a file's extension is one we want to process."""
    _, ext = os.path.splitext(filename)
    return ext.lower() in SUPPORTED_EXTENSIONS


def read_file_safely(file_path):
    """
    Read a file's text content, handling encoding issues gracefully.
    Returns None if the file can't be read as text.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def load_repository_files(repo_path):
    """
    Walk the cloned repository folder and collect useful files.
    Returns a list of dicts: {"file_path": ..., "content": ...}
    file_path is relative to the repo root (easier to show to the user later).
    """
    collected_files = []

    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in-place so os.walk skips these folders entirely
        dirs[:] = [d for d in dirs if d not in SKIP_FOLDERS and not d.startswith(".")]

        for filename in files:
            if not is_supported_file(filename):
                continue

            full_path = os.path.join(root, filename)

            # Skip very large files
            if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
                continue

            content = read_file_safely(full_path)
            if content is None or content.strip() == "":
                continue

            # Store the path relative to the repo root, e.g. "src/auth.py"
            relative_path = os.path.relpath(full_path, repo_path)

            collected_files.append({
                "file_path": relative_path,
                "content": content
            })

    return collected_files

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # characters repeated between chunks, so context isn't cut off


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split a piece of text into overlapping chunks.
    Overlap means the end of one chunk repeats a bit at the start of
    the next, so we don't lose context right at a chunk boundary.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward, but overlap a little

    return chunks


def chunk_repository_files(files):
    """
    Take the list of {file_path, content} from load_repository_files()
    and turn it into a list of chunks with metadata attached.

    Returns a list of dicts:
    {
        "file_path": "src/auth.py",
        "chunk_index": 0,
        "text": "...chunk content..."
    }
    """
    all_chunks = []

    for file in files:
        text_chunks = chunk_text(file["content"])

        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "file_path": file["file_path"],
                "chunk_index": i,
                "text": chunk
            })

    return all_chunks

if __name__ == "__main__":
    from github_loader import clone_repository

    test_url = input("Enter a public GitHub repository URL: ").strip()
    repo_path = clone_repository(test_url)

    files = load_repository_files(repo_path)
    print(f"\nFound {len(files)} usable files.")

    chunks = chunk_repository_files(files)
    print(f"Split into {len(chunks)} chunks.\n")

    for c in chunks[:5]:  # preview the first 5 chunks
        print(f" - {c['file_path']} [chunk {c['chunk_index']}]: {c['text'][:80]}...")