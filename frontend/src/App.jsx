import { useEffect, useState } from "react";
import ChatPanel from "./components/ChatPanel.jsx";
import MetricsDashboard from "./components/MetricsDashboard.jsx";
import Sidebar from "./components/Sidebar.jsx";
import { getHealth } from "./api.js";

export default function App() {
  const [tab, setTab] = useState("chat");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "down" }));
  }, []);

  return (
    <div className="flex h-screen text-slate-800">
      <Sidebar health={health} />

      <main className="flex flex-1 flex-col">
        <header className="flex items-center justify-between border-b bg-white px-6 py-3">
          <div>
            <h1 className="text-lg font-semibold">ML Research Assistant</h1>
            <p className="text-xs text-slate-500">
              Retrieval-Augmented Generation · LangGraph · faithfulness-gated answers
            </p>
          </div>
          <nav className="flex gap-1 rounded-lg bg-slate-100 p-1">
            {[
              ["chat", "Chat"],
              ["metrics", "Evaluation Dashboard"],
            ].map(([key, label]) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`rounded-md px-4 py-1.5 text-sm font-medium transition ${
                  tab === key
                    ? "bg-white text-brand-600 shadow"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </header>

        <section className="flex-1 overflow-hidden">
          {tab === "chat" ? <ChatPanel /> : <MetricsDashboard />}
        </section>
      </main>
    </div>
  );
}
