import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getMetrics, runEvaluation } from "../api.js";

const METRIC_META = [
  ["faithfulness", "Faithfulness", "Answer grounded in retrieved context"],
  ["answer_relevancy", "Answer Relevancy", "Answer addresses the question"],
  ["context_precision", "Context Precision", "Retrieved context is on-topic"],
];

function ScoreCard({ label, value, hint }) {
  const pct = value != null ? Math.round(value * 100) : null;
  const color =
    value == null ? "text-slate-400" : value >= 0.8 ? "text-emerald-600" : value >= 0.7 ? "text-amber-600" : "text-rose-600";
  return (
    <div className="rounded-xl border bg-white p-5 shadow-sm">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p className={`mt-1 text-3xl font-bold ${color}`}>{pct == null ? "—" : `${pct}%`}</p>
      <p className="mt-1 text-xs text-slate-500">{hint}</p>
    </div>
  );
}

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    getMetrics().then((m) => m.faithfulness != null && setMetrics(m)).catch(() => {});
  }, []);

  async function evaluate() {
    setRunning(true);
    try {
      setMetrics(await runEvaluation());
    } finally {
      setRunning(false);
    }
  }

  const chartData = metrics
    ? METRIC_META.map(([key, label]) => ({ name: label, value: metrics[key] }))
    : [];

  const improvement = metrics?.context_precision_improvement_pct;

  return (
    <div className="h-full overflow-y-auto px-8 py-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Evaluation Dashboard</h2>
          <p className="text-sm text-slate-500">
            RAGAS-style LLM-as-judge metrics over a golden question set
            {metrics ? ` · ${metrics.num_questions} questions` : ""}.
          </p>
        </div>
        <button
          onClick={evaluate}
          disabled={running}
          className="rounded-lg bg-brand-600 px-5 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50"
        >
          {running ? "Running evaluation…" : "▶ Run evaluation"}
        </button>
      </div>

      {!metrics && !running && (
        <div className="rounded-xl border border-dashed bg-white p-10 text-center text-slate-400">
          No metrics yet. Click <span className="font-medium">Run evaluation</span> to
          score the assistant live.
        </div>
      )}

      {metrics && (
        <>
          {improvement != null && (
            <div className="mb-6 rounded-xl bg-gradient-to-r from-emerald-500 to-brand-600 p-6 text-white shadow">
              <p className="text-sm opacity-90">Headline result</p>
              <p className="mt-1 text-3xl font-bold">
                +{improvement}% context precision
              </p>
              <p className="mt-1 text-sm opacity-90">
                Cross-encoder reranking improved retrieval precision from{" "}
                {Math.round(metrics.baseline_context_precision * 100)}% to{" "}
                {Math.round(metrics.context_precision * 100)}% vs. the vector-only baseline.
              </p>
            </div>
          )}

          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
            {METRIC_META.map(([key, label, hint]) => (
              <ScoreCard key={key} label={label} value={metrics[key]} hint={hint} />
            ))}
          </div>

          <div className="mb-6 rounded-xl border bg-white p-5 shadow-sm">
            <p className="mb-3 text-sm font-medium text-slate-600">Metric overview</p>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" fontSize={12} />
                <YAxis domain={[0, 1]} fontSize={12} />
                <Tooltip formatter={(v) => `${Math.round(v * 100)}%`} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {chartData.map((d, i) => (
                    <Cell key={i} fill={d.value >= 0.8 ? "#059669" : d.value >= 0.7 ? "#d97706" : "#e11d48"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {metrics.per_question?.length > 0 && (
            <div className="rounded-xl border bg-white p-5 shadow-sm">
              <p className="mb-3 text-sm font-medium text-slate-600">Per-question breakdown</p>
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase text-slate-400">
                  <tr>
                    <th className="py-2">Question</th>
                    <th className="py-2 text-right">Faith.</th>
                    <th className="py-2 text-right">Relev.</th>
                    <th className="py-2 text-right">Precision</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.per_question.map((q, i) => (
                    <tr key={i} className="border-t">
                      <td className="py-2 pr-4 text-slate-700">{q.question}</td>
                      <td className="py-2 text-right">{q.faithfulness.toFixed(2)}</td>
                      <td className="py-2 text-right">{q.answer_relevancy.toFixed(2)}</td>
                      <td className="py-2 text-right">{q.context_precision.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
