import { useEffect, useState } from "react";
import { getTopics } from "../api.js";

export default function Sidebar({ health }) {
  const [topics, setTopics] = useState({});

  useEffect(() => {
    getTopics().then(setTopics).catch(() => setTopics({}));
  }, []);

  const ok = health?.status === "ok";

  return (
    <aside className="hidden w-72 flex-col border-r bg-slate-900 text-slate-100 md:flex">
      <div className="border-b border-slate-700 px-5 py-4">
        <h2 className="text-base font-semibold">🤖 ML Research Assistant</h2>
        <p className="mt-1 text-xs text-slate-400">
          Grounded answers from a curated ML knowledge base.
        </p>
      </div>

      <div className="px-5 py-3 text-xs">
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 ${
            ok ? "bg-emerald-500/20 text-emerald-300" : "bg-rose-500/20 text-rose-300"
          }`}
        >
          <span className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-400" : "bg-rose-400"}`} />
          {ok ? "Backend online" : "Backend offline"}
        </span>
        {ok && (
          <p className="mt-2 text-slate-400">
            {health.model} · {health.kb_documents} docs · threshold{" "}
            {health.faithfulness_threshold}
          </p>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-2">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Knowledge base
        </p>
        {Object.entries(topics).map(([cat, items]) => (
          <div key={cat} className="mb-3">
            <p className="text-xs font-medium text-brand-500">{cat}</p>
            <p className="text-xs leading-relaxed text-slate-400">{items.join(", ")}</p>
          </div>
        ))}
      </div>

      <div className="border-t border-slate-700 px-5 py-3 text-[11px] text-slate-500">
        Built with FastAPI · LangGraph · ChromaDB · React
      </div>
    </aside>
  );
}
