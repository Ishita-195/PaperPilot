// Thin API client for the FastAPI backend.
const BASE = "/api";

export async function getHealth() {
  const r = await fetch(`${BASE}/health`);
  return r.json();
}

export async function getTopics() {
  const r = await fetch(`${BASE}/topics`);
  return r.json();
}

export async function sendChat(question, threadId) {
  const r = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: threadId }),
  });
  if (!r.ok) throw new Error(`Chat failed: ${r.status}`);
  return r.json();
}

export async function uploadPdfs(files) {
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f));
  const r = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
  return r.json();
}

export async function runEvaluation() {
  const r = await fetch(`${BASE}/evaluate`, { method: "POST" });
  if (!r.ok) throw new Error(`Evaluation failed: ${r.status}`);
  return r.json();
}

export async function getMetrics() {
  const r = await fetch(`${BASE}/metrics`);
  return r.json();
}
