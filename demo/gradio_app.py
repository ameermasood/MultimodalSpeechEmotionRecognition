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
        <div class="emotion-stage" aria-hidden="true">
            <div class="emotion-orb orb-sad">
                <span class="brow left"></span>
                <span class="brow right"></span>
                <span class="eye left"></span>
                <span class="eye right"></span>
                <span class="mouth"></span>
            </div>
            <div class="emotion-orb orb-happy">
                <span class="brow left"></span>
                <span class="brow right"></span>
                <span class="eye left"></span>
                <span class="eye right"></span>
                <span class="mouth"></span>
            </div>
            <div class="emotion-orb orb-angry">
                <span class="brow left"></span>
                <span class="brow right"></span>
                <span class="eye left"></span>
                <span class="eye right"></span>
                <span class="mouth"></span>
            </div>
            <div class="emotion-orb orb-neutral">
                <span class="brow left"></span>
                <span class="brow right"></span>
                <span class="eye left"></span>
                <span class="eye right"></span>
                <span class="mouth"></span>
            </div>
            <div class="emotion-orb orb-calm">
                <span class="brow left"></span>
                <span class="brow right"></span>
                <span class="eye left"></span>
                <span class="eye right"></span>
                <span class="mouth"></span>
            </div>
            <div class="emotion-orb orb-surprised">
                <span class="brow left"></span>
                <span class="brow right"></span>
                <span class="eye left"></span>
                <span class="eye right"></span>
                <span class="mouth"></span>
            </div>
            <div class="emotion-orb orb-tense">
                <span class="brow left"></span>
                <span class="brow right"></span>
                <span class="eye left"></span>
                <span class="eye right"></span>
                <span class="mouth"></span>
            </div>
        </div>
        <div class="hero-copy">
            <p class="eyebrow">Multimodal speech emotion recognition</p>
            <h1>How does this voice feel?</h1>
            <p class="subtitle">
                Upload or record speech, optionally add the transcript, and let a
                PEFT-adapted Voxtral model estimate the speaker's emotional tone.
            </p>
            <div class="label-set">
                <span>Angry</span>
                <span>Happy</span>
                <span>Sad</span>
                <span>Neutral</span>
            </div>
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

    .hero h1 {
        color: #22242a;
        font-size: 3.65rem;
        line-height: 0.98;
        margin: 0;
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

    .emotion-stage {
        height: 100%;
        inset: 0;
        min-width: 0;
        opacity: 0.34;
        pointer-events: none;
        position: relative;
        position: absolute;
        width: 100%;
        z-index: 1;
    }

    .emotion-orb {
        border-radius: 50%;
        box-shadow:
            inset -18px -26px 42px rgba(0, 0, 0, 0.17),
            inset 16px 16px 34px rgba(255, 255, 255, 0.22),
            0 18px 34px rgba(31, 41, 55, 0.16);
        position: absolute;
    }

    .emotion-orb::after {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 50%;
        content: "";
        height: 34%;
        left: 18%;
        position: absolute;
        top: 13%;
        width: 38%;
    }

    .orb-happy {
        background: linear-gradient(145deg, #ffe86d 0%, #f5c51b 62%, #dca20a 100%);
        height: 150px;
        left: 43%;
        top: 18px;
        width: 150px;
        z-index: 4;
    }

    .orb-sad {
        background: linear-gradient(145deg, #5d7df7 0%, #3344d2 64%, #242b92 100%);
        height: 116px;
        left: 14%;
        top: 72px;
        width: 116px;
        z-index: 2;
    }

    .orb-angry {
        background: linear-gradient(145deg, #f87171 0%, #dc2626 58%, #991b1b 100%);
        height: 128px;
        right: 14%;
        top: 80px;
        width: 128px;
        z-index: 3;
    }

    .orb-neutral {
        background: linear-gradient(145deg, #a5abb1 0%, #62686e 58%, #34383d 100%);
        height: 96px;
        right: 31%;
        bottom: 14px;
        width: 96px;
        z-index: 1;
    }

    .orb-calm {
        background: linear-gradient(145deg, #7dd3fc 0%, #06b6d4 58%, #0e7490 100%);
        height: 102px;
        left: 29%;
        bottom: 22px;
        width: 102px;
        z-index: 2;
    }

    .orb-surprised {
        background: linear-gradient(145deg, #d8b4fe 0%, #8b5cf6 58%, #6d28d9 100%);
        height: 112px;
        right: 4%;
        bottom: 42px;
        width: 112px;
        z-index: 1;
    }

    .orb-tense {
        background: linear-gradient(145deg, #fca5a5 0%, #f97316 56%, #c2410c 100%);
        height: 90px;
        left: 5%;
        bottom: 38px;
        width: 90px;
        z-index: 1;
    }

    .emotion-orb .eye {
        background: #fbfbfb;
        border-radius: 50%;
        box-shadow: 0 5px 12px rgba(0, 0, 0, 0.2);
        height: 31%;
        position: absolute;
        top: 34%;
        width: 31%;
        z-index: 2;
    }

    .emotion-orb .eye::after {
        background: #2d2d32;
        border-radius: 50%;
        content: "";
        height: 70%;
        left: 15%;
        position: absolute;
        top: 15%;
        width: 70%;
    }

    .emotion-orb .eye.left {
        left: 22%;
    }

    .emotion-orb .eye.right {
        right: 22%;
    }

    .emotion-orb .brow {
        background: rgba(46, 40, 47, 0.46);
        border-radius: 999px;
        height: 6%;
        position: absolute;
        top: 26%;
        width: 24%;
        z-index: 3;
    }

    .emotion-orb .brow.left {
        left: 23%;
    }

    .emotion-orb .brow.right {
        right: 23%;
    }

    .emotion-orb .mouth {
        border-radius: 999px;
        position: absolute;
        z-index: 3;
    }

    .orb-happy .mouth {
        border-bottom: 7px solid rgba(42, 74, 39, 0.54);
        bottom: 22%;
        height: 18%;
        left: 34%;
        width: 32%;
    }

    .orb-happy .brow.left,
    .orb-happy .brow.right {
        transform: rotate(0deg);
    }

    .orb-sad .mouth {
        border-top: 6px solid rgba(20, 25, 68, 0.5);
        bottom: 19%;
        height: 16%;
        left: 32%;
        width: 36%;
    }

    .orb-sad .brow.left {
        transform: rotate(-12deg);
    }

    .orb-sad .brow.right {
        transform: rotate(12deg);
    }

    .orb-angry .brow.left {
        transform: rotate(18deg);
    }

    .orb-angry .brow.right {
        transform: rotate(-18deg);
    }

    .orb-angry .mouth {
        background: rgba(70, 27, 39, 0.42);
        bottom: 22%;
        height: 5%;
        left: 34%;
        transform: rotate(2deg);
        width: 34%;
    }

    .orb-neutral .mouth {
        background: rgba(28, 31, 35, 0.5);
        bottom: 28%;
        height: 5%;
        left: 37%;
        width: 26%;
    }

    .orb-neutral .brow {
        top: 28%;
    }

    .orb-calm .mouth {
        border-bottom: 5px solid rgba(19, 78, 74, 0.48);
        bottom: 25%;
        height: 14%;
        left: 36%;
        width: 28%;
    }

    .orb-surprised .mouth {
        background: rgba(58, 28, 91, 0.55);
        border-radius: 50%;
        bottom: 19%;
        height: 24%;
        left: 39%;
        width: 22%;
    }

    .orb-surprised .brow.left {
        transform: rotate(-12deg);
    }

    .orb-surprised .brow.right {
        transform: rotate(12deg);
    }

    .orb-tense .brow.left {
        transform: rotate(20deg);
    }

    .orb-tense .brow.right {
        transform: rotate(-20deg);
    }

    .orb-tense .mouth {
        background: rgba(90, 36, 19, 0.5);
        bottom: 24%;
        height: 5%;
        left: 35%;
        width: 30%;
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

        .hero h1 {
            font-size: 2.45rem;
        }

        .emotion-stage {
            opacity: 0.22;
        }

        .orb-happy {
            left: 31%;
            top: 16px;
        }

        .orb-sad {
            left: -6%;
            top: 92px;
        }

        .orb-angry {
            right: -8%;
            top: 112px;
        }

        .orb-calm {
            left: 8%;
            bottom: 26px;
        }

        .orb-neutral {
            right: 12%;
            bottom: 16px;
        }

        .orb-surprised,
        .orb-tense {
            display: none;
        }

        .metric-block {
            text-align: left;
            width: 100%;
        }

    }
    """


if __name__ == "__main__":
    main()
