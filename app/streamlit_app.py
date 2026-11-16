"""Demo UI.   streamlit run app/streamlit_app.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agrisight.agent.agent import LANGUAGES  # noqa: E402
from agrisight.config import load_config  # noqa: E402
from agrisight.factory import build_agent  # noqa: E402

st.set_page_config(page_title="AgriSight", page_icon="🌱", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def load():
    cfg = load_config()
    return cfg, build_agent(cfg)


@st.cache_resource(show_spinner="Loading speech model...")
def load_asr(model_name: str):
    from agrisight.speech import ASR

    return ASR(model_name)


cfg, agent = load()

with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Answer language", list(LANGUAGES), format_func=lambda c: LANGUAGES[c])
    use_rag = st.toggle("Use official guidelines (RAG)", value=True)
    st.caption("Location (optional, for spray-timing weather)")
    lat = st.number_input("Latitude", value=None, format="%.4f")
    lon = st.number_input("Longitude", value=None, format="%.4f")
    st.divider()
    st.write("Components loaded")
    st.write({"classifier": agent.predictor is not None, "retriever": agent.retriever is not None, "vlm": agent.vlm is not None})

st.title("🌱 AgriSight")
st.caption("Photo of the affected leaf + your question, by text or voice.")

col_in, col_out = st.columns([1, 1.4])
with col_in:
    source = st.radio("Photo", ["Upload", "Camera"], horizontal=True)
    file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp"]) if source == "Upload" else st.camera_input("Take a photo")
    image = Image.open(file).convert("RGB") if file else None
    if image:
        st.image(image, use_container_width=True)
    question = st.text_area("Your question", placeholder="What is this disease and what should I do?")
    audio = st.file_uploader("...or a voice question", type=["wav", "mp3", "m4a", "ogg"])
    go = st.button("Ask", type="primary", use_container_width=True)

if go:
    if audio is not None:
        with tempfile.NamedTemporaryFile(suffix=Path(audio.name).suffix, delete=False) as tmp:
            tmp.write(audio.getvalue())
        with st.spinner("Transcribing..."):
            transcript = load_asr(cfg["asr"]["model"]).transcribe(tmp.name, language)
        question = f"{question} {transcript}".strip()
        col_in.info(f"Heard: {transcript}")
    if not question and image is None:
        st.warning("Add a photo or a question.")
        st.stop()
    with st.spinner("Thinking..."):
        result = agent.ask(question, image=image, language=language, lat=lat, lon=lon, use_rag=use_rag)

    with col_out:
        d = result["diagnosis"]
        if d:
            st.subheader(d["display"])
            st.progress(d["confidence"], text=f"Confidence {d['confidence']:.0%}")
            if d["abstain"]:
                st.warning("The model is not sure. Take a closer, well-lit photo of one affected leaf.")
            st.caption("Other possibilities: " + ", ".join(f"{t['display']} ({t['prob']:.0%})" for t in d["top_k"][1:]))
        st.markdown("### Advice")
        st.write(result["answer"])
        if result["unsupported_numbers"]:
            st.error("Check with your local KVK: the answer contains numbers not found in the sources: "
                     + ", ".join(result["unsupported_numbers"]))
        if result["vlm_description"]:
            with st.expander("What the vision-language model sees"):
                st.write(result["vlm_description"])
        if result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])})"):
                for s in result["sources"]:
                    page = f", p.{s['page']}" if s["page"] else ""
                    st.markdown(f"**[{s['n']}] {s['source']}{page}**")
                    st.caption(s["text"][:600])
        if result["weather"]:
            with st.expander("Weather (next 3 days)"):
                st.write(result["weather"]["summary"])
        with st.expander("Pipeline trace"):
            st.json(result["trace"])

st.divider()
st.caption("AgriSight is a decision-support prototype, not a replacement for an agriculture officer. "
           "Always follow the pesticide label and local Krishi Vigyan Kendra (KVK) advice.")
