"""Local Gradio demo for speech emotion recognition."""

from __future__ import annotations

from functools import lru_cache
from html import escape
from importlib.util import find_spec
import json
import os
from pathlib import Path
from typing import Any

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
    bitsandbytes_available = find_spec("bitsandbytes") is not None

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

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### Speech sample")
                audio = gr.Audio(
                    label="Upload audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                transcript = gr.Textbox(
                    label="Optional transcript",
                    placeholder="Paste the spoken sentence here if you want audio + transcript prediction.",
                    lines=4,
                )
                predict_button = gr.Button("Predict emotion", variant="primary", size="lg")

            with gr.Column(scale=1, min_width=340):
                result_card = gr.HTML(_empty_result_html())
                label_scores = gr.HTML(_empty_scores_html())

        with gr.Accordion("Advanced settings", open=False):
            gr.Markdown("These settings are useful for local experiments. Most demo runs can keep the defaults.")
            with gr.Row():
                with gr.Column():
                    base_model = gr.Textbox(
                        label="Base model",
                        value=os.getenv("BASE_MODEL_ID", DEFAULT_BASE_MODEL),
                    )
                    adapter_path = gr.Textbox(
                        label="Adapter path",
                        value=os.getenv("ADAPTER_PATH", DEFAULT_ADAPTER_PATH),
                    )
                with gr.Column():
                    device = gr.Dropdown(
                        label="Device",
                        choices=["auto", "mps", "cuda", "cuda:0", "cpu"],
                        value=os.getenv("DEVICE", "auto"),
                    )
                    max_new_tokens = gr.Slider(
                        label="Max new tokens",
                        minimum=1,
                        maximum=16,
                        step=1,
                        value=int(os.getenv("MAX_NEW_TOKENS", "8")),
                    )
                    load_in_4bit = gr.Checkbox(
                        label="Load in 4-bit",
                        value=_env_bool("LOAD_IN_4BIT", False) and bitsandbytes_available,
                        interactive=bitsandbytes_available,
                        info="Requires bitsandbytes and compatible GPU support.",
                    )
                    if not bitsandbytes_available:
                        gr.Markdown("4-bit loading is unavailable because bitsandbytes is not installed.")

            runtime_summary = gr.HTML()

        with gr.Accordion("Technical details", open=False):
            technical_details = gr.HTML(_technical_details_html({}))

        advanced_inputs = [base_model, adapter_path, device, max_new_tokens, load_in_4bit]
        for control in advanced_inputs:
            control.change(
                fn=_runtime_summary_html,
                inputs=advanced_inputs,
                outputs=runtime_summary,
            )
        app.load(
            fn=_runtime_summary_html,
            inputs=advanced_inputs,
            outputs=runtime_summary,
        )

        predict_button.click(
            fn=_predict,
            inputs=[audio, transcript, *advanced_inputs],
            outputs=[result_card, label_scores, technical_details],
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
    base_model_id: str,
    adapter_path: str,
    device: str,
    max_new_tokens: int,
    load_in_4bit: bool,
) -> tuple[str, str, str]:
    """Run one prediction from Gradio inputs."""
    if not audio_path:
        raise gr.Error("Upload or record an audio file first.")

    config = RuntimeConfig(
        base_model_id=base_model_id.strip(),
        adapter_path=adapter_path.strip(),
        load_in_4bit=bool(load_in_4bit),
        device=device,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        temperature=0.2,
        top_p=0.95,
    )

    try:
        predictor = _get_predictor(config)
        prediction = predictor.predict(audio_path, transcript=transcript or "")
    except Exception as exc:
        raise gr.Error(f"Prediction failed: {exc}") from exc

    return (
        _result_html(prediction),
        _scores_html(prediction.label_scores or {}),
        _technical_details_html(prediction.to_dict()),
    )


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _runtime_summary_html(
    base_model_id: str,
    adapter_path: str,
    device: str,
    max_new_tokens: int,
    load_in_4bit: bool,
) -> str:
    """Render compact runtime metadata."""
    adapter = Path(adapter_path).name or adapter_path
    precision = "4-bit" if load_in_4bit else "float16"
    return f"""
    <div class="runtime-strip">
        <span><strong>Base</strong>{escape(base_model_id)}</span>
        <span><strong>Adapter</strong>{escape(adapter)}</span>
        <span><strong>Device</strong>{escape(device)}</span>
        <span><strong>Precision</strong>{precision}</span>
        <span><strong>Max tokens</strong>{int(max_new_tokens)}</span>
    </div>
    """


def _header_html() -> str:
    """Render the app header."""
    return """
    <header class="app-header">
        <div>
            <p class="eyebrow">Multimodal speech emotion recognition</p>
            <h1>Speech Emotion Recognition</h1>
            <p class="subtitle">
                Upload a speech clip and classify the speaker's emotional tone across
                four labels using a fine-tuned Voxtral model.
            </p>
        </div>
        <div class="label-set">
            <span>Angry</span>
            <span>Happy</span>
            <span>Sad</span>
            <span>Neutral</span>
        </div>
    </header>
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


def _technical_details_html(payload: dict[str, Any]) -> str:
    """Render technical payload as escaped JSON inside HTML."""
    formatted = json.dumps(payload, indent=2, ensure_ascii=False)
    return f"""
    <pre class="technical-json">{escape(formatted)}</pre>
    """


def _custom_css() -> str:
    """Return Gradio CSS for the local demo."""
    return """
    .gradio-container {
        max-width: 1120px !important;
        margin: 0 auto !important;
    }

    .app-header {
        align-items: flex-end;
        border-bottom: 1px solid #d8dee8;
        display: flex;
        gap: 2rem;
        justify-content: space-between;
        margin-bottom: 1.25rem;
        padding: 1rem 0 1.35rem;
    }

    .eyebrow {
        color: #115e59;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0;
        margin: 0 0 0.35rem;
        text-transform: uppercase;
    }

    .app-header h1 {
        color: #172033;
        font-size: 2.55rem;
        line-height: 1.04;
        margin: 0;
    }

    .subtitle {
        color: #607086;
        font-size: 1rem;
        line-height: 1.55;
        margin: 0.55rem 0 0;
        max-width: 720px;
    }

    .label-set {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        justify-content: flex-end;
        max-width: 320px;
    }

    .label-set span {
        background: #ffffff;
        border: 1px solid #d8dee8;
        border-radius: 999px;
        color: #2f3b4c;
        font-size: 0.82rem;
        padding: 0.35rem 0.65rem;
        white-space: nowrap;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #d8dee8;
        border-left: 6px solid #0f766e;
        border-radius: 8px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        min-height: 230px;
        padding: 1.35rem 1.45rem;
    }

    .empty-result {
        align-content: center;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
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
        color: #172033;
        font-size: 2.7rem;
        line-height: 1;
        margin: 0;
    }

    .metric-block {
        background: #f5f7fb;
        border: 1px solid #d8dee8;
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
        background: #ffffff;
        border: 1px solid #d8dee8;
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
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

    .technical-json {
        background: #0f172a;
        border-radius: 8px;
        color: #e5edf7;
        font-size: 0.82rem;
        line-height: 1.5;
        margin: 0;
        overflow-x: auto;
        padding: 1rem;
        white-space: pre-wrap;
    }

    .runtime-strip {
        display: grid;
        gap: 0.7rem;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        margin-top: 0.65rem;
    }

    .runtime-strip span {
        background: #f8fafc;
        border: 1px solid #d8dee8;
        border-radius: 8px;
        color: #172033;
        display: block;
        font-size: 0.82rem;
        min-width: 0;
        overflow-wrap: anywhere;
        padding: 0.7rem 0.75rem;
    }

    .runtime-strip strong {
        color: #607086;
        display: block;
        font-size: 0.7rem;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
    }

    @media (max-width: 760px) {
        .app-header,
        .result-row {
            align-items: flex-start;
            flex-direction: column;
        }

        .label-set {
            justify-content: flex-start;
        }

        .metric-block {
            text-align: left;
            width: 100%;
        }

        .runtime-strip {
            grid-template-columns: 1fr;
        }
    }
    """


if __name__ == "__main__":
    main()
