import io
import re
import traceback
import uuid
from collections import defaultdict

import requests
import streamlit as st
from pypdf import PdfReader

from agent import FAITHFULNESS_THRESHOLD, MAX_EVAL_RETRIES
from backend.kb import DOCUMENTS
from views.common import load_agent

APP_NAME = "PaperPilot"
CHUNK_WORDS = 200
CHUNK_OVERLAP = 40
MAX_CHUNKS_PER_PDF = 60

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


# ================= HELPERS =================
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for start in range(0, len(words), size - overlap):
        chunk = " ".join(words[start : start + size])
        if chunk.strip():
            chunks.append(chunk)
        if len(chunks) >= MAX_CHUNKS_PER_PDF:
            break
    return chunks


def fetch_arxiv_paper(raw):
    """Resolve an arXiv ID or URL to (id, title, text chunks)."""
    match = ARXIV_ID_RE.search(raw)
    if not match:
        raise ValueError("no arXiv ID found — expected something like 2106.09685")
    aid = match.group(1)

    title = f"arXiv:{aid}"
    try:
        meta = requests.get(
            "https://export.arxiv.org/api/query", params={"id_list": aid}, timeout=15
        ).text
        titles = re.findall(r"<title>(.*?)</title>", meta, re.S)
        if len(titles) > 1:
            title = re.sub(r"\s+", " ", titles[1]).strip()
    except Exception:
        pass  # title is cosmetic; the PDF fetch below is what matters

    resp = requests.get(f"https://arxiv.org/pdf/{aid}", timeout=60)
    resp.raise_for_status()
    text = extract_text_from_pdf(io.BytesIO(resp.content))
    return aid, title, chunk_text(text)


def transcript_markdown():
    lines = [f"# {APP_NAME} conversation\n"]
    for msg in st.session_state.messages:
        who = "You" if msg["role"] == "user" else APP_NAME
        lines.append(f"**{who}:** {msg['content']}\n")
        if msg.get("sources"):
            lines.append(f"*Sources: {', '.join(msg['sources'])}*\n")
        if msg.get("faithfulness") is not None:
            lines.append(f"*Faithfulness: {msg['faithfulness']:.2f}*\n")
    return "\n".join(lines)


# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
st.session_state.setdefault("arxiv_docs", [])
st.session_state.setdefault("arxiv_meta", [])
st.session_state.setdefault("stats", {"questions": 0, "scores": [], "retries": 0})

# ================= SIDEBAR =================
with st.sidebar:
    st.title(f"📄 {APP_NAME}")
    st.markdown(
        "An ML research assistant that answers only from its knowledge base "
        "and **verifies every answer for faithfulness** before showing it."
    )
    metrics_slot = st.container()  # filled at the end of the script
    st.markdown("---")

    with st.expander(f"Browse knowledge base ({len(DOCUMENTS)} topics)"):
        by_category = defaultdict(list)
        for d in DOCUMENTS:
            by_category[d.get("category", "General")].append(d["topic"])
        for category, topics in by_category.items():
            st.markdown(f"**{category}**")
            st.caption(" · ".join(topics))

    st.markdown("---")
    st.markdown("**Add papers from arXiv**")
    arxiv_input = st.text_input(
        "arXiv ID or URL",
        placeholder="2106.09685 or arxiv.org/abs/...",
        label_visibility="collapsed",
    )
    if st.button("Fetch and index", use_container_width=True) and arxiv_input.strip():
        already_indexed = {m["id"] for m in st.session_state.arxiv_meta}
        try:
            with st.spinner("Fetching from arXiv..."):
                aid, title, chunks = fetch_arxiv_paper(arxiv_input)
            if aid in already_indexed:
                st.info("That paper is already indexed.")
            elif not chunks:
                st.warning("No extractable text in that PDF.")
            else:
                for i, chunk in enumerate(chunks):
                    st.session_state.arxiv_docs.append(
                        {
                            "id": f"arxiv_{aid}_{i}",
                            "topic": f"{title} (part {i + 1})",
                            "text": chunk,
                        }
                    )
                st.session_state.arxiv_meta.append(
                    {"id": aid, "title": title, "chunks": len(chunks)}
                )
                st.rerun()
        except Exception as e:
            st.error(f"Couldn't fetch that paper: {e}")

    for meta in st.session_state.arxiv_meta:
        left, right = st.columns([5, 1])
        short = meta["title"] if len(meta["title"]) <= 45 else meta["title"][:45] + "…"
        left.caption(f"{short} — {meta['chunks']} chunks")
        if right.button("✕", key=f"rm_{meta['id']}", help="Remove from index"):
            prefix = f"arxiv_{meta['id']}_"
            st.session_state.arxiv_docs = [
                d for d in st.session_state.arxiv_docs if not d["id"].startswith(prefix)
            ]
            st.session_state.arxiv_meta = [
                m for m in st.session_state.arxiv_meta if m["id"] != meta["id"]
            ]
            st.rerun()

    st.markdown("---")
    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

# ================= EXTRACT & CHUNK PDFs =================
pdf_docs = []
pdf_counts = {}
if uploaded_files:
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        chunks = chunk_text(text)
        pdf_counts[file.name] = len(chunks)
        for i, chunk in enumerate(chunks):
            pdf_docs.append(
                {
                    "id": f"{file.name}_{i}",
                    "topic": f"{file.name} (part {i + 1})",
                    "text": chunk,
                }
            )

with st.sidebar:
    for name, count in pdf_counts.items():
        st.caption(f"{name} — {count} chunks indexed")
    st.markdown("---")
    st.download_button(
        "Download chat (.md)",
        transcript_markdown(),
        file_name="paperpilot_chat.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# ================= LOAD AGENT =================
extra_docs = pdf_docs + st.session_state.arxiv_docs
with st.spinner("Loading agent... (first load takes ~30 seconds)"):
    app, embedder, collection = load_agent(extra_docs)

# ================= MAIN UI =================
st.title(APP_NAME)
st.caption("Grounded answers about ML — with a faithfulness check on every response.")

EXAMPLE_QUESTIONS = [
    "How does XGBoost differ from Random Forest?",
    "When should I use SHAP instead of LIME?",
    "What is the bias-variance tradeoff?",
]

examples_slot = st.empty()
if not st.session_state.messages:
    with examples_slot.container():
        st.markdown("**Try asking:**")
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, q in zip(cols, EXAMPLE_QUESTIONS):
            if col.button(q, use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

# chat history (sources and scores are stored per message)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("**Sources used:** " + " · ".join(msg["sources"]))
        if msg.get("faithfulness") is not None:
            f = msg["faithfulness"]
            color = "green" if f >= FAITHFULNESS_THRESHOLD else "orange"
            st.caption(f":{color}[Faithfulness score: {f:.2f}]")

prompt = st.chat_input("Ask your ML question...")
if not prompt:
    prompt = st.session_state.pop("pending_question", None)

if prompt:
    examples_slot.empty()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = {}
        answers_seen = 0
        try:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            # NOTE: "messages" is intentionally not passed — the checkpointer
            # keeps conversation history per thread_id.
            with st.status("Retrieving context...", expanded=False) as status:
                for update in app.stream(
                    {"question": prompt, "eval_retries": 0},
                    config,
                    stream_mode="updates",
                ):
                    for node, delta in update.items():
                        if delta:
                            result.update(delta)
                        if node == "retrieve":
                            status.update(label="Generating answer...")
                        elif node == "answer":
                            answers_seen += 1
                            status.update(label="Fact-checking answer...")
                        elif node == "eval":
                            f = result.get("faithfulness")
                            if (
                                f is not None
                                and f < FAITHFULNESS_THRESHOLD
                                and result.get("eval_retries", 0) <= MAX_EVAL_RETRIES
                            ):
                                status.update(
                                    label="Low faithfulness — retrying with stricter grounding..."
                                )
                retries = max(0, answers_seen - 1)
                f = result.get("faithfulness")
                if f is None or f >= FAITHFULNESS_THRESHOLD:
                    done = "Answer verified"
                else:
                    done = "Best-effort answer (low faithfulness)"
                if retries:
                    done += f" after {retries} " + ("retry" if retries == 1 else "retries")
                status.update(label=done, state="complete")

            answer = result.get("answer", "Sorry, I could not generate a response.")
            sources = list(dict.fromkeys(result.get("sources", [])))
            faithfulness = result.get("faithfulness")
        except Exception:
            traceback.print_exc()  # full details go to the server logs only
            answer = (
                "Sorry — something went wrong while generating the answer. "
                "Please try again in a moment."
            )
            sources = []
            faithfulness = None
            retries = 0

        st.markdown(answer)
        if sources:
            st.markdown("**Sources used:** " + " · ".join(sources))
        if faithfulness is not None:
            color = "green" if faithfulness >= FAITHFULNESS_THRESHOLD else "orange"
            st.caption(f":{color}[Faithfulness score: {faithfulness:.2f}]")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "faithfulness": faithfulness,
        }
    )
    stats = st.session_state.stats
    stats["questions"] += 1
    stats["retries"] += retries
    if faithfulness is not None:
        stats["scores"].append(faithfulness)

# ================= SIDEBAR METRICS (rendered last so they're fresh) =================
with metrics_slot:
    stats = st.session_state.stats
    scores = stats["scores"]
    avg = f"{sum(scores) / len(scores):.2f}" if scores else "–"
    c1, c2, c3 = st.columns(3)
    c1.metric("Asked", stats["questions"])
    c2.metric("Faithful", avg)
    c3.metric("Retries", stats["retries"])
