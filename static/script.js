const loadBtn = document.getElementById("load-btn");
const repoUrlInput = document.getElementById("repo-url");
const loadStatus = document.getElementById("load-status");
const chatSection = document.getElementById("chat-section");
const repoLabel = document.getElementById("repo-label");
const chatWindow = document.getElementById("chat-window");
const questionInput = document.getElementById("question-input");
const askBtn = document.getElementById("ask-btn");

let currentRepoName = null;

loadBtn.addEventListener("click", async () => {
  const repoUrl = repoUrlInput.value.trim();
  if (!repoUrl) {
    loadStatus.textContent = "Please enter a repository URL.";
    return;
  }

  loadBtn.disabled = true;
  loadStatus.textContent = "Loading and indexing repository... this may take a moment.";

  try {
    const res = await fetch("/api/load_repo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_url: repoUrl })
    });
    const data = await res.json();

    if (!res.ok) {
      loadStatus.textContent = data.error || "Something went wrong.";
      return;
    }

    currentRepoName = data.repo_name;
    loadStatus.textContent = `Ready! (${data.chunk_count} chunks indexed)`;
    repoLabel.textContent = `Chatting with: ${data.repo_name}`;
    chatWindow.innerHTML = "";
    chatSection.classList.remove("hidden");

  } catch (err) {
    loadStatus.textContent = "Network error — is the server running?";
  } finally {
    loadBtn.disabled = false;
  }
});

askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question || !currentRepoName) return;

  addMessage(question, "user");
  questionInput.value = "";
  askBtn.disabled = true;

  const thinkingEl = addMessage("Thinking...", "bot");

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_name: currentRepoName, question })
    });
    const data = await res.json();

    if (!res.ok) {
      thinkingEl.textContent = data.error || "Something went wrong.";
      return;
    }

    thinkingEl.textContent = data.answer;

    if (data.sources && data.sources.length > 0) {
      const sourcesEl = document.createElement("div");
      sourcesEl.className = "sources";
      sourcesEl.innerHTML = "Sources: " + data.sources.map(s => `<code>${s}</code>`).join(" ");
      thinkingEl.appendChild(sourcesEl);
    }

  } catch (err) {
    thinkingEl.textContent = "Network error — please try again.";
  } finally {
    askBtn.disabled = false;
  }
}

function addMessage(text, role) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = text;
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return el;
}