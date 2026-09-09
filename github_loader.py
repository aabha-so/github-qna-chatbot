import os
import re
import subprocess

DATA_DIR = "data"


def validate_github_url(url):
    """
    Check if the URL looks like a valid public GitHub repository URL.
    Example of a valid URL: https://github.com/username/repo-name
    """
    pattern = r"^https://github\.com/[\w.-]+/[\w.-]+/?$"
    return re.match(pattern, url.strip()) is not None


def get_repo_name(url):
    """
    Turn a GitHub URL into a safe local folder name.
    Example: https://github.com/psf/requests -> psf_requests
    """
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    owner, repo = url.split("/")[-2:]
    return f"{owner}_{repo}"


def clone_repository(url):
    """
    Clone a public GitHub repository into the data/ folder.
    Returns the local path to the cloned repository.
    Skips cloning again if the repo is already there.
    """
    if not validate_github_url(url):
        raise ValueError(f"Invalid GitHub repository URL: {url}")

    os.makedirs(DATA_DIR, exist_ok=True)

    repo_name = get_repo_name(url)
    local_path = os.path.join(DATA_DIR, repo_name)

    if os.path.exists(local_path):
        print(f"Repository already exists locally at: {local_path}")
        return local_path

    print(f"Cloning {url} into {local_path} ...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, local_path],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone repository: {e.stderr.strip()}")

    print("Clone successful.")
    return local_path


if __name__ == "__main__":
    # Quick manual test — run this file directly to try it out
    test_url = input("Enter a public GitHub repository URL: ").strip()
    try:
        path = clone_repository(test_url)
        print(f"Repository is available at: {path}")
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")