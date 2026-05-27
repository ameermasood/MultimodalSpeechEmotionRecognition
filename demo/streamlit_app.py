"""Local Streamlit demo for speech emotion recognition."""

from __future__ import annotations

from importlib.util import find_spec
import os
import tempfile
from pathlib import Path

import streamlit as st

from mer.config import RuntimeConfig
from mer.inference import DemoEmotionPredictor


DEFAULT_BASE_MODEL = "mistralai/Voxtral-Mini-3B-2507"
DEFAULT_ADAPTER_PATH = "checkpoints/final_adapter_dora"
EMOTION_COLORS = {
    "Angry": "#c2410c",
    "Happy": "#047857",
    "Sad": "#2563eb",
    "Neutral": "#525252",
    "Unknown": "#7c2d12",
}


def main() -> None:
    st.set_page_config(
        page_title="Speech Emotion Recognition",
        page_icon=None,
        layout="wide",
    )

    _inject_styles()

    with st.sidebar:
        st.markdown("## Runtime")
        bitsandbytes_available = find_spec("bitsandbytes") is not None
        base_model_id = st.text_input(
            "Base model",
            value=os.getenv("BASE_MODEL_ID", DEFAULT_BASE_MODEL),
        )
        adapter_path = st.text_input(
            "Adapter path",
            value=os.getenv("ADAPTER_PATH", DEFAULT_ADAPTER_PATH),
        )
        load_in_4bit = st.toggle(
            "Load in 4-bit",
            value=_env_bool("LOAD_IN_4BIT", False) and bitsandbytes_available,
            disabled=not bitsandbytes_available,
            help="Requires bitsandbytes and compatible GPU support.",
        )
        if not bitsandbytes_available:
            st.caption("4-bit loading is disabled because bitsandbytes is not installed.")
        device = st.selectbox(
            "Device",
            options=["auto", "mps", "cuda", "cuda:0", "cpu"],
            index=0,
        )
        max_new_tokens = st.number_input(
            "Max new tokens",
            min_value=1,
            max_value=16,
            value=int(os.getenv("MAX_NEW_TOKENS", "8")),
            step=1,
        )

    config = RuntimeConfig(
        base_model_id=base_model_id,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
        device=device,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        temperature=0.2,
        top_p=0.95,
    )

    _render_header()

    left, right = st.columns([1.35, 0.9], gap="large")
    with left:
        st.markdown('<section class="panel">', unsafe_allow_html=True)
        st.markdown("### Input")
        uploaded_audio = st.file_uploader(
            "Audio file",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            label_visibility="collapsed",
        )
        if uploaded_audio is not None:
            st.audio(uploaded_audio)
            st.caption(f"{uploaded_audio.name} · {_format_bytes(uploaded_audio.size)}")

        transcript = st.text_area(
            "Transcript",
            placeholder="Optional transcript for audio+text prediction.",
            height=140,
        )
        run_button = st.button(
            "Predict emotion",
            type="primary",
            use_container_width=True,
            disabled=uploaded_audio is None,
        )
        st.markdown("</section>", unsafe_allow_html=True)

    with right:
        _render_runtime_summary(config)

    if run_button:
        prediction = _run_prediction(uploaded_audio, transcript, config)
        if prediction is not None:
            _render_prediction(prediction)


@st.cache_resource(show_spinner=False)
def _get_predictor(config: RuntimeConfig) -> DemoEmotionPredictor:
    """Cache one loaded model per runtime configuration."""
    return DemoEmotionPredictor(config)


def _persist_upload(uploaded_file) -> str:
    """Persist Streamlit's upload object to a temporary file for the processor."""
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return handle.name


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _run_prediction(uploaded_audio, transcript: str, config: RuntimeConfig):
    """Persist upload and run one prediction."""
    if uploaded_audio is None:
        st.warning("Upload an audio file first.")
        return None

    try:
        with st.spinner("Loading model and running inference..."):
            audio_path = _persist_upload(uploaded_audio)
            predictor = _get_predictor(config)
            return predictor.predict(audio_path, transcript=transcript)
    except Exception as exc:
        st.error("Prediction failed.")
        st.exception(exc)
        return None


def _render_header() -> None:
    """Render the app header."""
    st.markdown(
        """
        <div class="app-header">
            <div>
                <p class="eyebrow">Multimodal speech emotion recognition</p>
                <h1>Voxtral Emotion Demo</h1>
                <p class="subtitle">
                    Upload speech audio, optionally add a transcript, and run a PEFT-adapted
                    Voxtral model over four emotion labels.
                </p>
            </div>
            <div class="label-set">
                <span>Angry</span>
                <span>Happy</span>
                <span>Sad</span>
                <span>Neutral</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_runtime_summary(config: RuntimeConfig) -> None:
    """Render compact runtime metadata."""
    adapter = Path(config.adapter_path).name
    quantization = "4-bit" if config.load_in_4bit else "float16"
    st.markdown('<section class="panel side-panel">', unsafe_allow_html=True)
    st.markdown("### Model")
    st.markdown(
        f"""
        <dl class="metadata">
            <dt>Base</dt><dd>{config.base_model_id}</dd>
            <dt>Adapter</dt><dd>{adapter}</dd>
            <dt>Device</dt><dd>{config.device}</dd>
            <dt>Precision</dt><dd>{quantization}</dd>
        </dl>
        """,
        unsafe_allow_html=True,
    )
    st.info("First prediction may take time because the base model and adapter are loaded once.")
    st.markdown("</section>", unsafe_allow_html=True)


def _render_prediction(prediction) -> None:
    """Render the prediction result."""
    color = EMOTION_COLORS.get(prediction.label, EMOTION_COLORS["Unknown"])
    confidence = "Not available" if prediction.confidence is None else f"{prediction.confidence:.2%}"
    transcript_mode = "Audio + transcript" if prediction.transcript_used else "Audio only"
    st.markdown(
        f"""
        <section class="result-panel" style="border-left-color: {color};">
            <p class="eyebrow">Prediction</p>
            <div class="result-row">
                <div>
                    <h2 style="color: {color};">{prediction.label}</h2>
                    <p class="subtitle">{transcript_mode}</p>
                </div>
                <div class="metric-block">
                    <span>Label confidence</span>
                    <strong>{confidence}</strong>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Prediction details"):
        if prediction.label_scores:
            st.bar_chart(prediction.label_scores)
            st.caption("Label confidence is computed by scoring each allowed emotion label.")
        st.json(prediction.to_dict())


def _format_bytes(size: int) -> str:
    """Format upload size for display."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _inject_styles() -> None:
    """Inject Streamlit styling for the local demo."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #607086;
            --line: #d8dee8;
            --panel: #ffffff;
            --soft: #f5f7fb;
            --accent: #0f766e;
            --accent-dark: #115e59;
        }

        .stApp {
            background:
                linear-gradient(180deg, #eef3f8 0%, #f8fafc 38%, #ffffff 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        .app-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 2rem;
            padding: 1.35rem 0 1.5rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.4rem;
        }

        .eyebrow {
            color: var(--accent-dark);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0;
            margin: 0 0 0.35rem;
            text-transform: uppercase;
        }

        .app-header h1 {
            color: var(--ink);
            font-size: 2.45rem;
            line-height: 1.04;
            margin: 0;
            letter-spacing: 0;
        }

        .subtitle {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
            margin: 0.55rem 0 0;
            max-width: 720px;
        }

        .label-set {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.45rem;
            max-width: 320px;
        }

        .label-set span {
            border: 1px solid var(--line);
            background: #ffffff;
            border-radius: 999px;
            color: #2f3b4c;
            font-size: 0.82rem;
            padding: 0.35rem 0.65rem;
            white-space: nowrap;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.15rem 1.15rem 1.25rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }

        .side-panel {
            background: #fbfcfe;
        }

        .panel h3 {
            color: var(--ink);
            font-size: 1.05rem;
            margin: 0 0 0.9rem;
            letter-spacing: 0;
        }

        .metadata {
            display: grid;
            grid-template-columns: 86px minmax(0, 1fr);
            row-gap: 0.8rem;
            column-gap: 0.8rem;
            margin: 0.2rem 0 1rem;
        }

        .metadata dt {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
        }

        .metadata dd {
            color: var(--ink);
            font-size: 0.9rem;
            margin: 0;
            overflow-wrap: anywhere;
        }

        .result-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 6px solid var(--accent);
            border-radius: 8px;
            margin-top: 1.25rem;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        }

        .result-row {
            align-items: center;
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
        }

        .result-row h2 {
            font-size: 2.1rem;
            line-height: 1;
            margin: 0;
            letter-spacing: 0;
        }

        .metric-block {
            background: var(--soft);
            border: 1px solid var(--line);
            border-radius: 8px;
            min-width: 160px;
            padding: 0.8rem 0.95rem;
            text-align: right;
        }

        .metric-block span {
            color: var(--muted);
            display: block;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            text-transform: uppercase;
        }

        .metric-block strong {
            color: var(--ink);
            font-size: 1rem;
        }

        div[data-testid="stFileUploader"] {
            border: 1px dashed #a8b5c7;
            border-radius: 8px;
            background: #f8fafc;
            padding: 0.6rem;
        }

        .stButton > button {
            border-radius: 8px;
            font-weight: 700;
            min-height: 2.8rem;
        }

        .stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
        }

        @media (max-width: 760px) {
            .app-header {
                align-items: flex-start;
                flex-direction: column;
            }

            .label-set {
                justify-content: flex-start;
            }

            .result-row {
                align-items: flex-start;
                flex-direction: column;
            }

            .metric-block {
                text-align: left;
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
