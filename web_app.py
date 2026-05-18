"""Small browser UI for the RAG chatbot.

Run:
    python web_app.py --host 127.0.0.1 --port 7860
"""

from __future__ import annotations

import argparse
import html
import json
import threading
import traceback
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from chatbot import build_chatbot
from config import RAGConfig


DEFAULT_KB_PATH = "data/knowledge_base/processed/chunks.jsonl"
DEFAULT_EMBEDDER_KIND = "sentence-transformer"
DEFAULT_RETRIEVER_MODEL = "BAAI/bge-m3"
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen3-1.7B"


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Chatbot</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #d8dee9;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --danger: #b42318;
      --bubble: #e8f4f2;
      --shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .app {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: 100vh;
    }

    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      overflow-y: auto;
    }

    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
      min-height: 100vh;
    }

    header {
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.78);
      backdrop-filter: blur(12px);
    }

    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }

    .sub {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }

    .section {
      padding: 14px 0;
      border-bottom: 1px solid var(--line);
    }

    .section:first-child { padding-top: 0; }
    .section:last-child { border-bottom: 0; }

    .section-title {
      margin: 0 0 10px;
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      color: #344054;
    }

    label {
      display: block;
      margin: 10px 0 6px;
      color: #344054;
      font-size: 13px;
      font-weight: 600;
    }

    input, select, textarea, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      font: inherit;
    }

    input, select, textarea {
      min-height: 38px;
      padding: 8px 10px;
      background: #fff;
      color: var(--text);
    }

    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      min-height: 16px;
      margin: 0;
      accent-color: var(--accent);
    }

    .inline {
      display: flex;
      align-items: center;
      gap: 9px;
      margin-top: 10px;
      color: #344054;
      font-weight: 600;
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .hint {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }

    .chat {
      padding: 22px;
      overflow-y: auto;
    }

    .empty {
      max-width: 720px;
      margin: 8vh auto 0;
      color: var(--muted);
      text-align: center;
    }

    .message {
      max-width: 860px;
      margin: 0 auto 14px;
      display: grid;
      gap: 6px;
    }

    .role {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .bubble {
      padding: 12px 14px;
      border-radius: 8px;
      box-shadow: var(--shadow);
      white-space: pre-wrap;
      word-break: break-word;
    }

    .user .bubble {
      background: #172033;
      color: #fff;
    }

    .assistant .bubble {
      background: var(--panel);
    }

    .sources {
      display: grid;
      gap: 8px;
      margin-top: 8px;
    }

    details {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px 10px;
    }

    summary {
      cursor: pointer;
      color: #344054;
      font-weight: 600;
    }

    .source-meta {
      color: var(--muted);
      font-size: 12px;
      margin: 5px 0;
    }

    .composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 120px;
      gap: 10px;
      padding: 16px 22px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }

    textarea {
      resize: vertical;
      min-height: 54px;
      max-height: 180px;
    }

    button {
      cursor: pointer;
      min-height: 42px;
      padding: 9px 12px;
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      font-weight: 700;
    }

    button:hover { background: var(--accent-strong); }
    button:disabled { cursor: wait; opacity: 0.65; }

    .secondary {
      background: #fff;
      border-color: var(--line);
      color: #344054;
    }

    .secondary:hover { background: #f8fafc; }

    .status {
      min-height: 20px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }

    .status.error { color: var(--danger); }

    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      aside {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .composer { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="section">
        <p class="section-title">Knowledge Base</p>
        <label for="kbPath">KB path</label>
        <input id="kbPath" value="data/knowledge_base/processed/chunks.jsonl" />
        <label for="embedder">Embedder</label>
        <select id="embedder">
          <option value="tfidf">TF-IDF fallback</option>
          <option value="sentence-transformer" selected>SentenceTransformer</option>
        </select>
        <label for="retrieverModel">Retriever model</label>
        <input id="retrieverModel" value="BAAI/bge-m3" />
        <div class="grid-2">
          <div>
            <label for="topK">Retriever top-k</label>
            <input id="topK" type="number" min="1" max="30" value="5" />
          </div>
          <div>
            <label for="showSources">Sources</label>
            <select id="showSources">
              <option value="yes" selected>Show</option>
              <option value="no">Hide</option>
            </select>
          </div>
        </div>
      </div>

      <div class="section">
        <p class="section-title">Generator</p>
        <label for="generatorModel">Generator model</label>
        <input id="generatorModel" value="Qwen/Qwen3-1.7B" />
        <label class="inline">
          <input id="useLlm" type="checkbox" checked />
          Use HuggingFace LLM
        </label>
        <div class="hint">Turn off to use extractive fallback only.</div>
        <label for="maxNewTokens">Max new tokens</label>
        <input id="maxNewTokens" type="number" min="1" placeholder="Unlocked until EOS/context" />
      </div>

      <div class="section">
        <p class="section-title">Sampling</p>
        <label class="inline">
          <input id="sample" type="checkbox" />
          Enable sampling
        </label>
        <div class="grid-2">
          <div>
            <label for="temperature">Temperature</label>
            <input id="temperature" type="number" step="0.05" min="0" value="0.2" />
          </div>
          <div>
            <label for="topP">Top-p</label>
            <input id="topP" type="number" step="0.05" min="0" max="1" value="0.9" />
          </div>
        </div>
        <label for="llmTopK">LLM top-k</label>
        <input id="llmTopK" type="number" min="1" value="20" />
      </div>

      <div class="section">
        <button class="secondary" id="clearBtn" type="button">Clear chat</button>
        <div id="status" class="status">Models load on first message and stay cached while this server runs.</div>
      </div>
    </aside>

    <main>
      <header>
        <h1>RAG Chatbot</h1>
        <div class="sub">Chat over the local knowledge base with retrieval and generation controls.</div>
      </header>
      <section id="chat" class="chat">
        <div class="empty">Ask a question to start. First response may take a while while models load.</div>
      </section>
      <form id="composer" class="composer">
        <textarea id="question" placeholder="Ask about VNU, UET admissions, regulations..." required></textarea>
        <button id="sendBtn" type="submit">Send</button>
      </form>
    </main>
  </div>

  <script>
    const chat = document.getElementById("chat");
    const statusEl = document.getElementById("status");
    const form = document.getElementById("composer");
    const questionEl = document.getElementById("question");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");
    let hasMessages = false;

    function value(id) {
      return document.getElementById(id).value.trim();
    }

    function numberOrNull(id) {
      const raw = value(id);
      if (!raw) return null;
      const n = Number(raw);
      return Number.isFinite(n) ? n : null;
    }

    function settings() {
      const useLlm = document.getElementById("useLlm").checked;
      const sample = document.getElementById("sample").checked;
      return {
        kb_path: value("kbPath"),
        embedder_kind: value("embedder"),
        retriever_model: value("retrieverModel") || null,
        generator_model: useLlm ? value("generatorModel") : null,
        top_k: numberOrNull("topK") || 5,
        max_new_tokens: numberOrNull("maxNewTokens"),
        temperature: sample ? numberOrNull("temperature") : null,
        top_p: sample ? numberOrNull("topP") : null,
        llm_top_k: sample ? numberOrNull("llmTopK") : null,
      };
    }

    function escapeHtml(text) {
      return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function ensureChat() {
      if (!hasMessages) {
        chat.innerHTML = "";
        hasMessages = true;
      }
    }

    function addMessage(role, text, sources = []) {
      ensureChat();
      const wrapper = document.createElement("div");
      wrapper.className = `message ${role}`;
      const safeText = escapeHtml(text);
      const label = role === "user" ? "You" : "Assistant";
      let sourceHtml = "";
      if (role === "assistant" && value("showSources") === "yes" && sources.length) {
        sourceHtml = `<div class="sources">${sources.map((source, index) => `
          <details>
            <summary>Source ${index + 1}: score ${source.score.toFixed(3)}</summary>
            <div class="source-meta">${escapeHtml(source.title || "")}</div>
            <div class="source-meta">${escapeHtml(source.source_url || source.source || "")}</div>
            <div>${escapeHtml(source.preview || "")}</div>
          </details>
        `).join("")}</div>`;
      }
      wrapper.innerHTML = `
        <div class="role">${label}</div>
        <div class="bubble">${safeText}${sourceHtml}</div>
      `;
      chat.appendChild(wrapper);
      chat.scrollTop = chat.scrollHeight;
    }

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.classList.toggle("error", isError);
    }

    clearBtn.addEventListener("click", () => {
      hasMessages = false;
      chat.innerHTML = '<div class="empty">Ask a question to start. First response may take a while while models load.</div>';
      setStatus("Chat cleared. Cached models remain loaded on the server.");
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = questionEl.value.trim();
      if (!question) return;

      addMessage("user", question);
      questionEl.value = "";
      sendBtn.disabled = true;
      setStatus("Thinking...");

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question, settings: settings()}),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Request failed");
        addMessage("assistant", payload.answer, payload.sources || []);
        setStatus(payload.cache_hit ? "Answered using cached bot." : "Loaded settings and answered.");
      } catch (error) {
        addMessage("assistant", `Error: ${error.message}`);
        setStatus(error.message, true);
      } finally {
        sendBtn.disabled = false;
        questionEl.focus();
      }
    });
  </script>
</body>
</html>
"""


class BotCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple[Any, ...] | None = None
        self._bot = None

    def get(self, config: RAGConfig):
        key = tuple(sorted(asdict(config).items()))
        with self._lock:
            if self._bot is not None and self._key == key:
                return self._bot, True
            bot = build_chatbot(config)
            self._bot = bot
            self._key = key
            return bot, False

    def preload(self, config: RAGConfig) -> None:
        bot, _ = self.get(config)
        if bot is None:
            raise RuntimeError("Failed to preload chatbot.")


def _optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value):
    if value in (None, ""):
        return None
    return float(value)


def config_from_payload(settings: dict[str, Any]) -> RAGConfig:
    generator_model = (
        settings["generator_model"]
        if "generator_model" in settings
        else DEFAULT_GENERATOR_MODEL
    )
    retriever_model = (
        settings["retriever_model"]
        if "retriever_model" in settings
        else DEFAULT_RETRIEVER_MODEL
    )
    return RAGConfig(
        kb_path=settings.get("kb_path") or DEFAULT_KB_PATH,
        embedder_kind=settings.get("embedder_kind") or DEFAULT_EMBEDDER_KIND,
        retriever_model=retriever_model,
        generator_model=generator_model,
        top_k=int(settings.get("top_k") or 5),
        max_new_tokens=_optional_int(settings.get("max_new_tokens")),
        temperature=_optional_float(settings.get("temperature")),
        top_p=_optional_float(settings.get("top_p")),
        llm_top_k=_optional_int(settings.get("llm_top_k")),
    )


def default_config() -> RAGConfig:
    return RAGConfig(
        kb_path=DEFAULT_KB_PATH,
        embedder_kind=DEFAULT_EMBEDDER_KIND,
        retriever_model=DEFAULT_RETRIEVER_MODEL,
        generator_model=DEFAULT_GENERATOR_MODEL,
        top_k=5,
    )


def source_payload(result) -> dict[str, Any]:
    document = result.document
    metadata = document.metadata or {}
    return {
        "id": document.id,
        "score": result.score,
        "source": document.source,
        "title": metadata.get("title") or metadata.get("parent_id") or document.id,
        "source_url": metadata.get("source_url") or metadata.get("url") or "",
        "preview": document.text[:700],
    }


class ChatHandler(BaseHTTPRequestHandler):
    cache = BotCache()

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_html(HTML_PAGE)
            return
        if self.path == "/api/health":
            self._send_json({"ok": True})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404)
            return

        try:
            payload = self._read_json()
            question = str(payload.get("question") or "").strip()
            if not question:
                raise ValueError("Question is required.")

            config = config_from_payload(payload.get("settings") or {})
            bot, cache_hit = self.cache.get(config)
            answer, results = bot.answer(question, return_context=True)
            self._send_json(
                {
                    "answer": answer,
                    "sources": [source_payload(result) for result in results],
                    "cache_hit": cache_hit,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": str(exc)}, status=500)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the browser UI for the RAG chatbot.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_config()
    print("Loading default retriever and generator models...")
    ChatHandler.cache.preload(config)
    print("Default models loaded and cached.")
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"RAG chatbot UI running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
