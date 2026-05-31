"""Local Gradio demo for speech emotion recognition."""

from __future__ import annotations

from functools import lru_cache
from html import escape
import os

import gradio as gr

from mer.config import RuntimeConfig
from mer.inference import DemoEmotionPredictor


DEFAULT_BASE_MODEL = "mistralai/Voxtral-Mini-3B-2507"
DEFAULT_ADAPTER_PATH = "checkpoints/final_adapter_dora"
EMOTION_COLORS = {
    "Angry": "#c2410c",
    "Happy": "#047857",
    "Sad": "#2563eb",
    "Neutral": "#525252",
    "Unknown": "#525252",
}


def main() -> None:
    """Launch the local Gradio app."""
    app = build_app()
    app.launch()


def build_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Speech Emotion Recognition",
        css=_custom_css(),
        theme=gr.themes.Soft(
            primary_hue="teal",
            neutral_hue="slate",
            radius_size="sm",
        ),
    ) as app:
        gr.HTML(_header_html())

        with gr.Group(elem_classes="recognition-card"):
            gr.HTML(_card_intro_html())
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=340):
                    gr.HTML('<p class="field-label">Upload or record speech</p>')
                    audio = gr.Audio(
                        label="Upload or record speech",
                        show_label=False,
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    gr.HTML('<p class="field-label">Optional transcript</p>')
                    transcript = gr.Textbox(
                        label="Optional transcript",
                        show_label=False,
                        placeholder="Paste the spoken sentence here if you want audio + transcript prediction.",
                        lines=4,
                    )
                    predict_button = gr.Button("Predict emotion", variant="primary", size="lg")

                with gr.Column(scale=1, min_width=340):
                    result_card = gr.HTML(_empty_result_html())
                    label_scores = gr.HTML(_empty_scores_html())

        predict_button.click(
            fn=_predict,
            inputs=[audio, transcript],
            outputs=[result_card, label_scores],
            api_name=False,
        )

    return app


@lru_cache(maxsize=4)
def _get_predictor(config: RuntimeConfig) -> DemoEmotionPredictor:
    """Cache loaded predictors by runtime configuration."""
    return DemoEmotionPredictor(config)


def _predict(
    audio_path: str | None,
    transcript: str,
) -> tuple[str, str]:
    """Run one prediction from Gradio inputs."""
    if not audio_path:
        raise gr.Error("Upload or record an audio file first.")

    config = _default_runtime_config()

    try:
        predictor = _get_predictor(config)
        prediction = predictor.predict(audio_path, transcript=transcript or "")
    except Exception as exc:
        raise gr.Error(f"Prediction failed: {exc}") from exc

    return (
        _result_html(prediction),
        _scores_html(prediction.label_scores or {}),
    )


def _default_runtime_config() -> RuntimeConfig:
    """Build demo runtime config from environment variables and defaults."""
    return RuntimeConfig(
        base_model_id=os.getenv("BASE_MODEL_ID", DEFAULT_BASE_MODEL).strip(),
        adapter_path=os.getenv("ADAPTER_PATH", DEFAULT_ADAPTER_PATH).strip(),
        load_in_4bit=_env_bool("LOAD_IN_4BIT", False),
        device=os.getenv("DEVICE", "auto").strip(),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "8")),
        do_sample=False,
        temperature=0.2,
        top_p=0.95,
    )


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _header_html() -> str:
    """Render the app header."""
    return """
    <header class="hero">
        <div class="emotion-bands" aria-hidden="true">
            <span class="emotion-band band-sad"></span>
            <span class="emotion-band band-happy"></span>
            <span class="emotion-band band-angry"></span>
            <span class="emotion-band band-neutral"></span>
        </div>
        <div class="hero-copy">
            <h1>Speech Emotion Recognition</h1>
            <p class="subtitle">AI Demo</p>
            <p class="subtitle">
                Upload or record a speech sample. This demo uses a fine-tuned Voxtral
                model to estimate one of four emotions in your speech: Angry, Happy, Sad, and Neutral.
            </p>
        </div>
    </header>
    """


def _card_intro_html() -> str:
    """Render the title area inside the app card."""
    return """
    <div class="card-intro">
        <div>
            <p class="eyebrow">Interactive demo</p>
            <h2>Classify one speech sample</h2>
        </div>
        <p>
            Works with audio alone or audio plus transcript. The label confidence
            compares the four allowed emotion labels for this one prediction.
        </p>
    </div>
    """


def _empty_result_html() -> str:
    """Render the initial prediction card."""
    return """
    <section class="result-card empty-result">
        <p class="eyebrow">Prediction</p>
        <h2>Waiting for audio</h2>
        <p class="subtitle">
            Your emotion label, confidence score, and label distribution will appear here.
        </p>
    </section>
    """


def _empty_scores_html() -> str:
    """Render the empty label distribution card."""
    return """
    <section class="score-card">
        <p class="eyebrow">Label distribution</p>
        <p class="muted-text">Run a prediction to compare the four emotion labels.</p>
    </section>
    """


def _result_html(prediction) -> str:
    """Render one prediction card."""
    color = EMOTION_COLORS.get(prediction.label, EMOTION_COLORS["Unknown"])
    confidence = "Not available" if prediction.confidence is None else f"{prediction.confidence:.2%}"
    transcript_mode = "Audio + transcript" if prediction.transcript_used else "Audio only"
    return f"""
    <section class="result-card" style="border-left-color: {color};">
        <p class="eyebrow">Prediction</p>
        <div class="result-row">
            <div>
                <h2 style="color: {color};">{escape(prediction.label)}</h2>
                <p class="subtitle">{transcript_mode}</p>
            </div>
            <div class="metric-block">
                <span>Label confidence</span>
                <strong>{confidence}</strong>
            </div>
        </div>
    </section>
    """


def _scores_html(label_scores: dict[str, float]) -> str:
    """Render label scores as horizontal bars."""
    if not label_scores:
        return _empty_scores_html()

    rows = []
    for label, score in sorted(label_scores.items(), key=lambda item: item[1], reverse=True):
        color = EMOTION_COLORS.get(label, EMOTION_COLORS["Unknown"])
        rows.append(
            f"""
            <div class="score-row">
                <div class="score-label">
                    <span>{escape(label)}</span>
                    <strong>{score:.2%}</strong>
                </div>
                <div class="score-track">
                    <div class="score-fill" style="width: {score * 100:.2f}%; background: {color};"></div>
                </div>
            </div>
            """
        )

    return f"""
    <section class="score-card">
        <p class="eyebrow">Label distribution</p>
        {''.join(rows)}
    </section>
    """


def _custom_css() -> str:
    """Return Gradio CSS for the local demo."""
    return """
    body,
    .gradio-container {
        background: #eef0f2 !important;
    }

    .gradio-container {
        max-width: 1120px !important;
        margin: 0 auto !important;
        padding: 22px 20px 34px !important;
    }

    .hero {
        align-items: center;
        display: flex;
        justify-content: center;
        min-height: 330px;
        overflow: hidden;
        padding: 1.6rem 0 1.85rem;
        position: relative;
    }

    .hero-copy {
        margin: 0 auto;
        max-width: 760px;
        position: relative;
        text-align: center;
        z-index: 3;
    }

    .eyebrow {
        color: #146c63;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0;
        margin: 0 0 0.35rem;
        text-transform: uppercase;
    }

    .subtitle {
        color: #646b78;
        font-size: 1rem;
        line-height: 1.55;
        margin: 0.65rem auto 0;
        max-width: 720px;
    }

    .label-set {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin-top: 1.15rem;
    }

    .label-set span {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(174, 181, 191, 0.65);
        border-radius: 999px;
        color: #303743;
        font-size: 0.82rem;
        padding: 0.35rem 0.65rem;
        white-space: nowrap;
    }

    .emotion-bands {
        height: 100%;
        inset: 0;
        pointer-events: none;
        position: absolute;
        width: 100%;
        z-index: 1;
    }

    .emotion-band {
        border-radius: 999px;
        filter: blur(26px);
        height: 58px;
        opacity: 0.22;
        position: absolute;
        transform: rotate(-9deg);
        width: 42%;
    }

    .band-sad {
        background: #2563eb;
        left: 5%;
        top: 22%;
    }

    .band-happy {
        background: #facc15;
        left: 26%;
        top: 12%;
        transform: rotate(6deg);
    }

    .band-angry {
        background: #dc2626;
        right: 6%;
        top: 32%;
        transform: rotate(-12deg);
    }

    .band-neutral {
        background: #71717a;
        bottom: 18%;
        left: 31%;
        transform: rotate(3deg);
        width: 36%;
    }

    .recognition-card {
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.78) 0%, rgba(248, 250, 252, 0.7) 100%) !important;
        border: 1px solid rgba(214, 219, 226, 0.92) !important;
        border-radius: 8px !important;
        box-shadow: 0 20px 52px rgba(30, 41, 59, 0.1) !important;
        margin-bottom: 1rem;
        padding: 1.1rem !important;
    }

    .recognition-card,
    .recognition-card * {
        color: #22242a;
    }

    .recognition-card > div,
    .recognition-card .form,
    .recognition-card .wrap,
    .recognition-card .block,
    .recognition-card .panel,
    .recognition-card .contain,
    .recognition-card [data-testid="block-info"] {
        background: transparent !important;
    }

    .recognition-card .block {
        border-color: rgba(211, 218, 228, 0.95) !important;
        box-shadow: none !important;
    }

    .card-intro {
        align-items: flex-end;
        display: flex;
        gap: 1.5rem;
        justify-content: space-between;
        margin-bottom: 0.9rem;
    }

    .card-intro h2 {
        color: #22242a;
        font-size: 1.28rem;
        line-height: 1.2;
        margin: 0;
    }

    .card-intro p:last-child {
        color: #646b78;
        font-size: 0.92rem;
        line-height: 1.45;
        margin: 0;
        max-width: 460px;
    }

    .field-label {
        color: #146c63;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0;
        margin: 0 0 0.35rem;
        text-transform: uppercase;
    }

    .recognition-card textarea,
    .recognition-card input {
        background: rgba(255, 255, 255, 0.86) !important;
        border-color: rgba(211, 218, 228, 0.95) !important;
        color: #22242a !important;
    }

    .recognition-card textarea::placeholder,
    .recognition-card input::placeholder {
        color: #7b8492 !important;
        opacity: 1 !important;
    }

    .recognition-card .upload-container,
    .recognition-card .dropzone,
    .recognition-card .input-container {
        background: rgba(255, 255, 255, 0.72) !important;
        border-color: rgba(196, 205, 218, 0.9) !important;
        color: #303743 !important;
    }

    .recognition-card button {
        border-radius: 8px !important;
        font-weight: 750 !important;
    }

    .recognition-card button.primary,
    .recognition-card button[variant="primary"] {
        background: linear-gradient(135deg, #168f82 0%, #0f766e 100%) !important;
        border-color: #0f766e !important;
        color: #ffffff !important;
    }

    .result-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #dde3eb;
        border-left: 6px solid #0f766e;
        border-radius: 8px;
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06);
        min-height: 230px;
        padding: 1.35rem 1.45rem;
    }

    .empty-result {
        align-content: center;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.94) 0%, rgba(248, 250, 252, 0.86) 100%);
        border-left-color: #94a3b8;
    }

    .result-row {
        align-items: center;
        display: flex;
        gap: 1.5rem;
        justify-content: space-between;
    }

    .result-row h2,
    .empty-result h2 {
        color: #22242a;
        font-size: 2.7rem;
        line-height: 1;
        margin: 0;
    }

    .metric-block {
        background: #f5f7f9;
        border: 1px solid #dde3eb;
        border-radius: 8px;
        min-width: 160px;
        padding: 0.8rem 0.95rem;
        text-align: right;
    }

    .metric-block span {
        color: #607086;
        display: block;
        font-size: 0.76rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
    }

    .metric-block strong {
        color: #172033;
        font-size: 1.15rem;
    }

    .score-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #dde3eb;
        border-radius: 8px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
        margin-top: 0.85rem;
        padding: 1rem 1.05rem 0.9rem;
    }

    .muted-text {
        color: #607086;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0.35rem 0 0;
    }

    .score-row {
        margin-top: 0.75rem;
    }

    .score-label {
        align-items: center;
        color: #172033;
        display: flex;
        font-size: 0.9rem;
        font-weight: 650;
        justify-content: space-between;
        margin-bottom: 0.32rem;
    }

    .score-label strong {
        color: #607086;
        font-size: 0.84rem;
    }

    .score-track {
        background: #e8edf4;
        border-radius: 999px;
        height: 0.56rem;
        overflow: hidden;
    }

    .score-fill {
        border-radius: 999px;
        height: 100%;
    }

    @media (max-width: 760px) {
        .hero,
        .card-intro,
        .result-row {
            align-items: flex-start;
            flex-direction: column;
        }

        .hero {
            min-height: 360px;
        }

        .hero-copy {
            text-align: center;
        }

        .hero h1,
        .hero h2 {
            font-size: 2.2rem;
            white-space: normal;
        }

        .emotion-band {
            height: 46px;
            opacity: 0.18;
            width: 62%;
        }

        .band-sad {
            left: -16%;
            top: 24%;
        }

        .band-happy {
            left: 28%;
            top: 10%;
        }

        .band-angry {
            right: -22%;
            top: 38%;
        }

        .band-neutral {
            bottom: 22%;
            left: 20%;
            width: 62%;
        }

        .metric-block {
            text-align: left;
            width: 100%;
        }

    }
    """


if __name__ == "__main__":
    main()
