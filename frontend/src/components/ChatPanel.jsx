import { useRef, useState } from "react";
import { sendChat, uploadPdfs } from "../api.js";

function FaithBadge({ score }) {
  if (score == null) return null;
  const good = score >= 0.7;
  return (
    <span
      className={`ml-2 rounded-full px-2 py-0.5 text-[11px] font-medium ${
        good ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"
      }`}
      title="LLM-judged faithfulness to retrieved context"
    >
      faithfulness {score.toFixed(2)}
    </span>
  );
}

export default function ChatPanel() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [threadId, setThreadId] = useState(null);
  const [uploadMsg, setUploadMsg] = useState("");
  const fileRef = useRef();

  async function submit(e) {
    e.preventDefault();
    const q = input.trim();
    if (!q || busy) return;
    setMessages((m) => [...m, { role: "user", content: q }]);
    setInput("");
    setBusy(true);
    try {
      const res = await sendChat(q, threadId);
      setThreadId(res.thread_id);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: res.answer,
          sources: res.sources,
          faithfulness: res.faithfulness,
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `⚠️ ${err.message}` },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function onUpload(e) {
    const files = e.target.files;
    if (!files?.length) return;
    setUploadMsg("Uploading & indexing…");
    try {
      const res = await uploadPdfs(files);
      setUploadMsg(`Added: ${res.added.join(", ") || "none (no extractable text)"}`);
    } catch {
      setUploadMsg("Upload failed");
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b bg-white px-6 py-2 text-sm">
        <button
          onClick={() => {
            setMessages([]);
            setThreadId(null);
          }}
          className="rounded-md border px-3 py-1 text-slate-600 hover:bg-slate-50"
        >
          + New conversation
        </button>
        <div className="flex items-center gap-2">
          {uploadMsg && <span className="text-xs text-slate-500">{uploadMsg}</span>}
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf"
            multiple
            hidden
            onChange={onUpload}
          />
          <button
            onClick={() => fileRef.current?.click()}
            className="rounded-md border px-3 py-1 text-slate-600 hover:bg-slate-50"
          >
            📎 Upload papers (PDF)
          </button>
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-6 py-6">
        {messages.length === 0 && (
          <div className="mx-auto mt-10 max-w-md text-center text-slate-400">
            <p className="text-lg font-medium text-slate-500">Ask an ML question</p>
            <p className="mt-1 text-sm">
              e.g. “How do SHAP and LIME differ?” or “When should I use Random Forest vs SVM?”
            </p>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-2xl rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
                m.role === "user"
                  ? "bg-brand-600 text-white"
                  : "bg-white text-slate-800"
              }`}
            >
              <div className="whitespace-pre-wrap">{m.content}</div>

              {m.role === "assistant" && (m.sources?.length || m.faithfulness != null) && (
                <div className="mt-3 border-t pt-2 text-xs text-slate-500">
                  {m.sources?.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1">
                      <span className="font-medium">Sources:</span>
                      {m.sources.map((s) => (
                        <span
                          key={s}
                          className="rounded bg-slate-100 px-1.5 py-0.5 text-slate-600"
                        >
                          {s}
                        </span>
                      ))}
                    </div>
                  )}
                  <FaithBadge score={m.faithfulness} />
                </div>
              )}
            </div>
          </div>
        ))}

        {busy && (
          <div className="flex justify-start">
            <div className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-400 shadow-sm">
              Thinking… (retrieving · reranking · grounding)
            </div>
          </div>
        )}
      </div>

      <form onSubmit={submit} className="border-t bg-white px-6 py-3">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask your ML question…"
            className="flex-1 rounded-lg border px-4 py-2 text-sm outline-none focus:border-brand-500"
          />
          <button
            type="submit"
            disabled={busy}
            className="rounded-lg bg-brand-600 px-5 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
