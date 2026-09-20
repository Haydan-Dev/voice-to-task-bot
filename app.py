import hashlib
import json
import os
import time

import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL = "openai/whisper-large-v3-turbo"
WHISPER_FALLBACK_MODEL = "openai/whisper-tiny.en"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

HF_INFERENCE_BASE = "https://router.huggingface.co/hf-inference/models"
CHAT_API_URL = "https://router.huggingface.co/v1/chat/completions"

SYSTEM_PROMPT = """You are a task-extraction assistant. Your ONLY job is to read spoken text and output a formatted bulleted To-Do list.

Rules:
- Output ONLY a markdown bulleted list (each item starts with "- ").
- Extract actionable tasks from the spoken text.
- Do not add introductions, explanations, summaries, or closing remarks.
- Do not repeat the original transcript.
- If no clear tasks are found, output exactly: "- No actionable tasks detected."
- Keep each task concise and action-oriented."""

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 10
REQUEST_TIMEOUT_SECONDS = 120


def get_api_key() -> str | None:
    return os.getenv("HUGGINGFACE_API_KEY")


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_api_key()}"}


def _request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    last_response: requests.Response | None = None

    for attempt in range(MAX_RETRIES):
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
        last_response = response

        if response.status_code == 200:
            return response

        if response.status_code == 503:
            wait_time = RETRY_WAIT_SECONDS * (attempt + 1)
            time.sleep(wait_time)
            continue

        return response

    return last_response  # type: ignore[return-value]


def _parse_hf_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return response.text or f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        if "error" in payload:
            error = payload["error"]
            if isinstance(error, dict):
                return error.get("message", str(error))
            return str(error)
        if "message" in payload:
            return str(payload["message"])

    return str(payload)


def _transcribe_with_model(audio_bytes: bytes, model_id: str) -> requests.Response:
    headers = {
        **_auth_headers(),
        "Content-Type": "audio/wav",
        "Accept": "application/json",
    }
    url = f"{HF_INFERENCE_BASE}/{model_id}"
    return _request_with_retry("POST", url, headers=headers, data=audio_bytes)


def transcribe_audio(audio_bytes: bytes) -> tuple[str, str]:
    """Return (transcription, model_used)."""
    models_to_try = [WHISPER_MODEL]
    if WHISPER_FALLBACK_MODEL not in models_to_try:
        models_to_try.append(WHISPER_FALLBACK_MODEL)

    last_error = "Unknown transcription error."
    for model_id in models_to_try:
        response = _transcribe_with_model(audio_bytes, model_id)

        if response.status_code == 200:
            payload = response.json()
            if isinstance(payload, dict):
                text = payload.get("text", "").strip()
                if text:
                    return text, model_id
            last_error = "Whisper returned an empty transcription."
            continue

        last_error = _parse_hf_error(response)
        model_unsupported = (
            response.status_code == 400
            and "not supported" in last_error.lower()
        )
        if model_unsupported and model_id != models_to_try[-1]:
            continue

        raise RuntimeError(
            f"Whisper transcription failed ({response.status_code}): {last_error}"
        )

    raise RuntimeError(last_error)


def extract_tasks(transcription: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Convert the following spoken text into a bulleted To-Do list:\n\n"
                    f"{transcription}"
                ),
            },
        ],
        "max_tokens": 512,
        "temperature": 0.2,
    }

    response = _request_with_retry(
        "POST",
        CHAT_API_URL,
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json=payload,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Task extraction failed ({response.status_code}): {_parse_hf_error(response)}"
        )

    payload = response.json()
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected response format from Llama API.") from exc

    content = content.strip()
    if not content:
        raise RuntimeError("Llama returned an empty To-Do list.")

    return content


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
            .main-header {
                font-size: 2.4rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }
            .sub-header {
                color: #6b7280;
                font-size: 1.05rem;
                margin-bottom: 1.5rem;
            }
            .section-card {
                background: #f8fafc;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 1rem;
                color: #0f172a;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "processed_audio_hash": None,
        "transcription": None,
        "todo_list": None,
        "whisper_model_used": None,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_recording(audio_bytes: bytes) -> None:
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    if audio_hash == st.session_state.processed_audio_hash:
        return

    st.session_state.processed_audio_hash = audio_hash
    st.session_state.transcription = None
    st.session_state.whisper_model_used = None
    st.session_state.last_error = None

    try:
        with st.spinner("Transcribing..."):
            transcription, model_used = transcribe_audio(audio_bytes)
        st.session_state.transcription = transcription
        st.session_state.whisper_model_used = model_used

        with st.spinner("Extracting Tasks..."):
            new_tasks = extract_tasks(transcription)
        
        if new_tasks != "- No actionable tasks detected.":
            if st.session_state.todo_list and st.session_state.todo_list != "- No actionable tasks detected.":
                st.session_state.todo_list += "\n" + new_tasks
            else:
                st.session_state.todo_list = new_tasks
        elif not st.session_state.todo_list:
            st.session_state.todo_list = new_tasks
    except requests.RequestException as exc:
        st.session_state.last_error = f"Network error: {exc}"
    except RuntimeError as exc:
        st.session_state.last_error = str(exc)


def main() -> None:
    st.set_page_config(
        page_title="Voice-to-Task Bot",
        page_icon="🎙️",
        layout="centered",
    )
    apply_custom_styles()
    init_session_state()

    st.markdown('<p class="main-header">🎙️ Voice-to-Task Bot</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Record your voice, get a clean To-Do list — powered by '
        "Hugging Face Serverless Inference.</p>",
        unsafe_allow_html=True,
    )

    api_key = get_api_key()
    if not api_key:
        st.error(
            "Missing `HUGGINGFACE_API_KEY` in your `.env` file. "
            "Create a token at https://huggingface.co/settings/tokens with "
            "**Inference Providers** permission, then restart the app."
        )
        st.stop()

    with st.sidebar:
        st.header("How it works")
        st.markdown(
            "1. Click the mic and speak your tasks.\n"
            "2. Whisper transcribes your audio.\n"
            "3. Llama 3 extracts a bulleted To-Do list."
        )
        st.divider()
        st.caption(f"Whisper (primary): `{WHISPER_MODEL}`")
        st.caption(f"Whisper (fallback): `{WHISPER_FALLBACK_MODEL}`")
        st.caption(f"LLM: `{LLM_MODEL}`")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Record")
    st.caption("Click the microphone, speak your tasks, then click again to stop.")

    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#ef4444",
        neutral_color="#64748b",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=2.0,
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        process_recording(audio_bytes)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if st.session_state.transcription:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Transcription")
        if (
            st.session_state.whisper_model_used
            and st.session_state.whisper_model_used != WHISPER_MODEL
        ):
            st.info(
                f"`{WHISPER_MODEL}` is not available on HF Serverless Inference. "
                f"Used `{st.session_state.whisper_model_used}` instead."
            )
        st.write(st.session_state.transcription)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.todo_list:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Your To-Do List")
        st.markdown(st.session_state.todo_list)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
