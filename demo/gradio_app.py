"""Local Gradio demo for speech emotion recognition."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from html import escape
import base64
import os
from pathlib import Path
import time
from typing import Any

import gradio as gr

from mer.config import RuntimeConfig
from mer.inference import DemoEmotionPredictor


DEFAULT_BASE_MODEL = "mistralai/Voxtral-Mini-3B-2507"
DEFAULT_ADAPTER_PATH = "checkpoints/final_adapter_dora"
GITHUB_URL = "https://github.com/ameermasood/MultimodalSpeechEmotionRecognition"
POLITO_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "polito_logo.png"
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
        title="Multimodal Speech Emotion Recognition",
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
            with gr.Group(elem_classes="input-panel"):
                gr.HTML('<p class="field-label">Upload or record speech</p>')
                audio = gr.Audio(
                    label="Upload or record speech",
                    show_label=False,
                    type="filepath",
                    sources=["upload", "microphone"],
                    elem_classes="speech-audio",
                )
                gr.HTML('<p class="field-label">Optional transcript</p>')
                transcript = gr.Textbox(
                    label="Optional transcript",
                    show_label=False,
                    placeholder="Add a transcript to enable multimodal prediction, or leave it blank for audio-only prediction.",
                    lines=4,
                )
                predict_button = gr.Button(
                    "Predict Emotion",
                    variant="primary",
                    size="lg",
                    elem_classes="predict-button",
                )
            processing_status = gr.HTML("")
            prediction_panel = gr.HTML(_prediction_panel_html(_empty_result_html(), _empty_scores_html()))

        gr.HTML(_footer_html())

        predict_button.click(
            fn=_predict,
            inputs=[audio, transcript],
            outputs=[processing_status, prediction_panel],
            api_name=False,
            preprocess=False,
            show_progress="hidden",
        )

    return app


@lru_cache(maxsize=4)
def _get_predictor(config: RuntimeConfig) -> DemoEmotionPredictor:
    """Cache loaded predictors by runtime configuration."""
    return DemoEmotionPredictor(config)


def _predict(
    audio_input: Any,
    transcript: str,
) -> Iterator[tuple[str, str]]:
    """Run one prediction from Gradio inputs."""
    audio_path = _audio_path_from_input(audio_input)
    if not audio_path:
        yield (
            "",
            _prediction_panel_html(
                _message_result_html(
                    title="Upload audio first",
                    message="Add an audio file or record a short speech sample, then run prediction.",
                ),
                _empty_scores_html(),
            ),
        )
        return

    config = _default_runtime_config()

    try:
        predictor = _get_predictor(config)
        yield (
            _processing_status_html(
                percent=6,
                title="Preparing audio",
                detail="Reading the uploaded speech sample.",
            ),
            _prediction_panel_html(_empty_result_html(), _empty_scores_html()),
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(predictor.predict, audio_path, transcript or "")
            percent = 10
            while not future.done():
                yield (
                    _processing_status_html(
                        percent=percent,
                        title=_progress_title(percent),
                        detail=_progress_detail(percent),
                    ),
                    _prediction_panel_html(_empty_result_html(), _empty_scores_html()),
                )
                time.sleep(0.45)
                percent = _next_progress_percent(percent)

            prediction = future.result()
    except Exception as exc:
        yield (
            _processing_status_html(
                percent=100,
                title="Prediction failed",
                detail="Check the audio file and local model artifacts.",
                state="error",
            ),
            _prediction_panel_html(
                _message_result_html(
                    title="Prediction failed",
                    message=str(exc),
                ),
                _empty_scores_html(),
            ),
        )
        return

    yield (
        "",
        _prediction_panel_html(
            _result_html(prediction),
            _scores_html(prediction.label_scores or {}),
            accent_color=EMOTION_COLORS.get(prediction.label, EMOTION_COLORS["Unknown"]),
        ),
    )


def _next_progress_percent(percent: int) -> int:
    """Advance estimated progress without reaching completion before inference ends."""
    if percent < 35:
        return percent + 5
    if percent < 70:
        return percent + 3
    if percent < 90:
        return percent + 2
    return min(96, percent + 1)


def _progress_title(percent: int) -> str:
    """Return a stage title for estimated inference progress."""
    if percent < 35:
        return "Loading model"
    if percent < 70:
        return "Analyzing speech"
    if percent < 90:
        return "Scoring labels"
    return "Finalizing"


def _progress_detail(percent: int) -> str:
    """Return a short explanation for estimated inference progress."""
    if percent < 35:
        return "The first run can take longer while Voxtral and the adapter load."
    if percent < 70:
        return "Processing the speech sample with the fine-tuned model."
    if percent < 90:
        return "Comparing Angry, Happy, Sad, and Neutral."
    return "Waiting for the model to finish cleanly."


def _audio_path_from_input(audio_input: Any) -> str | None:
    """Extract a file path from Gradio audio input without audio decoding."""
    if audio_input is None:
        return None
    if isinstance(audio_input, str):
        return audio_input
    if isinstance(audio_input, dict):
        path = audio_input.get("path") or audio_input.get("name")
        return str(path) if path else None
    path = getattr(audio_input, "path", None) or getattr(audio_input, "name", None)
    return str(path) if path else None


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
    logo = _polito_logo_data_uri()
    logo_html = f'<img class="polito-logo" src="{logo}" alt="Politecnico di Torino logo">' if logo else ""
    return f"""
    <header class="hero">
        <div class="hero-brand">
            {logo_html}
        </div>
        <div class="hero-copy">
            <h1>Multimodal Speech Emotion Recognition</h1>
            <p class="subtitle">
                Upload or record speech, add a transcript if available, and predict the speaker’s emotion.
            </p>
        </div>
    </header>
    """


@lru_cache(maxsize=1)
def _polito_logo_data_uri() -> str:
    """Return the PoliTo logo as a data URI if the local asset exists."""
    if not POLITO_LOGO_PATH.is_file():
        return ""
    encoded = base64.b64encode(POLITO_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _card_intro_html() -> str:
    """Render the title area inside the app card."""
    return """
    <div class="card-intro">
        <div>
            <h2>DEMO</h2>
        </div>
    </div>
    """


def _footer_html() -> str:
    """Render the project reference footer."""
    return f"""
    <footer class="project-footer">
        <div class="footer-main">
            <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">GitHub</a> | Demo Developed by <span>Amir Masoud Almasi</span> | Supervised by <span>Politecnico di Torino</span> and <span>LINKS Foundation</span>
        </div>
        <div class="footer-meta">
            <span aria-hidden="true"></span>
        </div>
    </footer>
    """


def _empty_result_html() -> str:
    """Render the initial prediction card."""
    return """
    <div class="result-card empty-result">
        <p class="eyebrow">Prediction</p>
        <h2>Pending</h2>
    </div>
    """


def _prediction_panel_html(result_html: str, scores_html: str, accent_color: str | None = None) -> str:
    """Render prediction and label distribution as one visual panel."""
    accent_style = f' style="border-color: {accent_color};"' if accent_color else ""
    return f"""
    <section class="prediction-panel"{accent_style}>
        {result_html}
        {scores_html}
    </section>
    """


def _empty_scores_html() -> str:
    """Render the empty label distribution card."""
    return """
    <div class="score-card">
    </div>
    """


def _processing_status_html(percent: int, title: str, detail: str, state: str = "active") -> str:
    """Render a compact processing progress bar."""
    percent = max(0, min(100, percent))
    return f"""
    <section class="processing-status processing-{state}">
        <div class="processing-row">
            <div>
                <p class="eyebrow">Processing</p>
                <strong>{escape(title)}</strong>
                <span>{escape(detail)}</span>
            </div>
            <b>{percent}%</b>
        </div>
        <div class="processing-track">
            <div class="processing-fill" style="width: {percent}%;"></div>
        </div>
    </section>
    """


def _message_result_html(title: str, message: str) -> str:
    """Render a friendly message in the prediction card."""
    return f"""
    <div class="result-card message-result">
        <p class="eyebrow">Prediction</p>
        <h2>{escape(title)}</h2>
        <p class="subtitle">{escape(message)}</p>
    </div>
    """


def _result_html(prediction) -> str:
    """Render one prediction card."""
    color = EMOTION_COLORS.get(prediction.label, EMOTION_COLORS["Unknown"])
    confidence = "Not available" if prediction.confidence is None else f"{prediction.confidence:.2%}"
    transcript_mode = "Audio + transcript" if prediction.transcript_used else "Audio only"
    return f"""
    <div class="result-card">
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
    </div>
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
    <div class="score-card">
        <p class="eyebrow">Label distribution</p>
        {''.join(rows)}
    </div>
    """


def _custom_css() -> str:
    """Return Gradio CSS for the local demo."""
    return """
    html,
    body,
    #root,
    .app,
    main,
    .gradio-container {
        background: #ffffff !important;
    }

    .gradio-container {
        margin: 0 !important;
        max-width: none !important;
        min-height: 100vh !important;
        padding: 10px 32px 28px !important;
        width: 100% !important;
    }

    .hero {
        align-items: center;
        display: flex;
        justify-content: center;
        min-height: 170px;
        overflow: visible;
        padding: 0.85rem 1rem 0.95rem;
        position: relative;
    }

    .hero-brand {
    position: absolute;
    left: 50%;
    top: 1rem;
    transform: translateX(-50%);
    width: 150px;
}

    .polito-logo {
        display: block;
        height: auto;
        max-height: 250px;
        max-width: 250px;
        object-fit: contain;
        opacity: 0.2;
        width: 100%;
    }

    .hero-copy {
        background: transparent;
        border-radius: 8px;
        margin: 0 auto;
        max-width: 100%;
        padding: 0.65rem 1rem;
        position: relative;
        text-align: center;
        z-index: 2;
    }

    .eyebrow {
        color: #003576;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0;
        margin: 0 0 0.35rem;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #111827;
        font-size: clamp(2.15rem, 3.35vw, 3.2rem);
        line-height: 1.05;
        margin: 0;
        max-width: 100%;
        white-space: nowrap;
        text-wrap: balance;
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

    .recognition-card {
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.78) 0%, rgba(248, 250, 252, 0.7) 100%) !important;
        border: 1px solid rgba(214, 219, 226, 0.92) !important;
        border-radius: 8px !important;
        box-shadow: 0 20px 52px rgba(30, 41, 59, 0.1) !important;
        margin-bottom: 1rem;
        padding: 1rem !important;
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
        align-items: center;
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        margin-bottom: 0.65rem;
        text-align: center;
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

    .input-panel {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #dde3eb !important;
        border-radius: 8px !important;
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06) !important;
        margin-top: 0.85rem;
        overflow: hidden;
        padding: 1.15rem 1.25rem !important;
    }

    .input-panel,
    .input-panel > div,
    .input-panel .form,
    .input-panel .wrap,
    .input-panel .block,
    .input-panel .panel,
    .input-panel .contain {
        background: rgba(255, 255, 255, 0.9) !important;
    }

    .field-label {
        color: #003576;
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

    .speech-audio .upload-container,
    .speech-audio .dropzone,
    .speech-audio .input-container {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
        color: #111827 !important;
        min-height: 160px !important;
    }

    .speech-audio .wrap,
    .speech-audio .container,
    .speech-audio .block,
    .speech-audio .panel,
    .speech-audio .contain,
    .speech-audio .form {
        background: transparent !important;
        background-color: transparent !important;
        color: #111827 !important;
        min-height: 160px !important;
    }

    .speech-audio,
    .speech-audio > div {
        background: transparent !important;
        background-color: transparent !important;
        color: #111827 !important;
    }

    .recognition-card button {
        border-radius: 8px !important;
        font-weight: 750 !important;
    }

    .recognition-card .predict-button,
    .recognition-card .predict-button button {
        background: #000000 !important;
        background-color: #000000 !important;
        border-color: #000000 !important;
        color: #ffffff !important;
    }

    .recognition-card .predict-button:hover,
    .recognition-card .predict-button button:hover {
        background: #000000 !important;
        background-color: #000000 !important;
        border-color: #000000 !important;
        color: #ffffff !important;
    }

    .speech-audio button,
    .speech-audio select {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
        color: #111827 !important;
    }

    .speech-audio button:hover,
    .speech-audio select:hover {
        background: #f8fafc !important;
        background-color: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #003576 !important;
    }

    .speech-audio button *,
    .speech-audio select * {
        color: inherit !important;
    }

    .processing-status {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        margin-top: 0.9rem;
        padding: 0.95rem 1rem;
    }

    .processing-row {
        align-items: flex-start;
        display: flex;
        gap: 1rem;
        justify-content: space-between;
    }

    .processing-row .eyebrow {
        color: #003576;
        margin-bottom: 0.2rem;
    }

    .processing-row strong {
        color: #111827;
        display: block;
        font-size: 1rem;
        line-height: 1.25;
    }

    .processing-row span {
        color: #607086;
        display: block;
        font-size: 0.86rem;
        line-height: 1.45;
        margin-top: 0.2rem;
    }

    .processing-row b {
        color: #111827;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.95rem;
        white-space: nowrap;
    }

    .processing-track {
        background: #e5e7eb;
        border-radius: 999px;
        height: 0.55rem;
        margin-top: 0.8rem;
        overflow: hidden;
    }

    .processing-fill {
        background: #111827;
        border-radius: 999px;
        height: 100%;
        transition: width 240ms ease;
    }

    .processing-complete .processing-fill {
        background: #111827;
    }

    .processing-error .processing-fill {
        background: #ef4444;
    }

    .prediction-panel {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #dde3eb;
        border-radius: 8px;
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06);
        margin-top: 1rem;
        overflow: hidden;
    }

    .result-card {
        background: transparent;
        border: 0;
        border-left: 0 solid transparent;
        border-radius: 0;
        box-shadow: none;
        margin-top: 0;
        min-height: 190px;
        padding: 1.35rem 1.45rem;
    }

    .result-card .eyebrow {
        color: #003576;
    }

    .empty-result {
        align-content: center;
        background: transparent;
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
        background: rgba(248, 250, 252, 0.72);
        border-top: 1px solid #dde3eb;
        margin-top: 0;
        padding: 1rem 1.45rem 1.05rem;
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

    .project-footer {
        align-items: center;
        border-top: 1px solid #e5e7eb;
        color: #64748b;
        display: flex;
        flex-direction: column;
        font-size: 0.88rem;
        gap: 0.25rem;
        justify-content: center;
        margin: 1.4rem 0 0;
        padding: 1rem 0 0;
        text-align: center;
    }

    .footer-main {
        color: #111827;
        font-weight: 500;
    }

    .footer-main strong {
        color: #003576;
        font-weight: 750;
    }

    .footer-main span {
        color: #003576;
        font-weight: 650;
    }

    .footer-meta {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        justify-content: center;
    }

    .project-footer a {
        color: #003576;
        font-weight: 750;
        text-decoration: none;
    }

    .project-footer a:hover {
        text-decoration: underline;
    }

    footer:not(.project-footer),
    .footer,
    .built-with,
    a[href*="gradio.app"],
    a[href*="/api"] {
        display: none !important;
        visibility: hidden !important;
    }

    @media (max-width: 760px) {
        .hero,
        .card-intro,
        .result-row {
            align-items: flex-start;
            flex-direction: column;
        }

        .hero {
            min-height: 220px;
            padding-inline: 0;
        }

        .hero-brand {
            justify-content: center;
            position: static;
            width: 130px;
            margin: 0 auto 0.85rem;
        }

        .hero-copy {
            padding: 0;
            text-align: center;
        }

        .hero h1 {
            font-size: 2.2rem;
            white-space: normal;
        }

        .gradio-container {
            padding: 10px 16px 26px !important;
        }

        .metric-block {
            text-align: left;
            width: 100%;
        }

    }
    """


if __name__ == "__main__":
    main()
