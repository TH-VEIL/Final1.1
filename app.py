"""VEIL Final — privacy-first prompt redaction and Gemini dashboard."""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from veil_guard import GuardReview, OllamaVeilGuard, VeilGuard, ollama_generate


@dataclass(frozen=True)
class Finding:
    start: int
    end: int
    kind: str


# Presidio provides the general PII recognizers. These local patterns cover
# common Indian/student identifiers and provide a useful fallback if the
# optional spaCy model is not installed.
PATTERNS: list[tuple[str, str]] = [
    ("EMAIL_ADDRESS", r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    ("CREDIT_CARD", r"\b(?:\d[ -]*?){13,19}\b"),
    ("PHONE_NUMBER", r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)"),
    ("IP_ADDRESS", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("ORDER_ID", r"(?i)#[A-Z]-\d{3,6}\b"),
    ("SRN_ID", r"(?i)\bPES\d[A-Z]{2}\d{2}[A-Z]{2}\d{3}\b"),
    ("DATE_TIME", r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{0,4})\b"),
    ("TIME", r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[AP]M)?\b|\b(?:1[0-2]|0?[1-9])\s?[AP]M\b"),
]

# These are deliberately context-based to reduce false positives for ordinary
# words. Presidio's NER recognizer handles names and locations when the spaCy
# model is installed; these patterns keep obvious user-labelled values covered
# in lightweight installations.
CONTEXT_PATTERNS: list[tuple[str, str]] = [
    ("PERSON", r"(?i)\b(?:my name is|i am|this is|dear)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"),
    ("ADDRESS", r"(?i)\b(?:address is|live at|located at|ship to|deliver to)\s+([^.!?\n]+)"),
    ("LOCATION", r"(?i)\b(?:in|at|near|from)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})"),
]


@st.cache_resource(show_spinner=False)
def _load_analyzer():
    """Load Presidio once per Streamlit process, including custom recognizers."""
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

        # Prefer the smaller model documented in README. If no spaCy model is
        # installed, do not attempt Presidio's default large-model discovery.
        # The local regex/context recognizers remain active without it.
        import spacy
        if not spacy.util.is_package("en_core_web_sm"):
            return None
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        )
        analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=["en"],
        )
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="TIME",
                patterns=[Pattern("time", PATTERNS[-1][1], 0.85)],
            )
        )
        return analyzer
    except Exception:
        return None


def _presidio_findings(text: str) -> list[Finding]:
    analyzer = _load_analyzer()
    if analyzer is None:
        return []
    try:
        results = analyzer.analyze(
            text=text,
            entities=[
                "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "DATE_TIME",
                "ORGANIZATION", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "URL", "NRP", "TIME",
            ],
            language="en",
            score_threshold=0.45,
        )
        return [Finding(r.start, r.end, r.entity_type) for r in results]
    except Exception:
        return []


def find_sensitive_data(text: str) -> list[Finding]:
    findings = _presidio_findings(text)
    for kind, pattern in PATTERNS:
        findings.extend(
            Finding(match.start(), match.end(), kind)
            for match in re.finditer(pattern, text)
        )
    for kind, pattern in CONTEXT_PATTERNS:
        findings.extend(
            Finding(match.start(1), match.end(1), kind)
            for match in re.finditer(pattern, text)
        )

    # Prefer the longest finding at an overlapping range, then remove repeats.
    accepted: list[Finding] = []
    for candidate in sorted(findings, key=lambda item: (item.start, -(item.end - item.start))):
        if any(candidate.start < item.end and candidate.end > item.start for item in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: item.start)


def redact(text: str, session_vault: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    findings = find_sensitive_data(text)
    counters: defaultdict[str, int] = defaultdict(int)
    for item in session_vault:
        counters[item["kind"]] = max(counters[item["kind"]], int(item["placeholder"].rsplit("_", 1)[1][:-1]))
    new_items: list[dict[str, str]] = []
    chunks: list[str] = []
    cursor = 0

    for finding in findings:
        original = text[finding.start:finding.end]
        existing = next((
            item for item in session_vault
            if item["kind"] == finding.kind and item["value"] == original
        ), None)
        if existing:
            placeholder = existing["placeholder"]
        else:
            counters[finding.kind] += 1
            placeholder = f"<{finding.kind}_{counters[finding.kind]}>"
            new_items.append({"placeholder": placeholder, "value": original, "kind": finding.kind})
        chunks.extend((text[cursor:finding.start], placeholder))
        cursor = finding.end

    chunks.append(text[cursor:])
    session_vault.extend(new_items)
    return "".join(chunks), new_items


def available_gemini_models(client) -> list[str]:
    """List models that this API key can actually use for text generation."""
    names = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or getattr(model, "supported_methods", [])
        if "generateContent" in actions:
            names.append(getattr(model, "name", "").removeprefix("models/"))
    return names


def session_gemini_models(client, api_key: str) -> list[str]:
    """Avoid a network model-list call on every Streamlit rerun."""
    import hashlib

    cache = st.session_state.setdefault("gemini_model_cache", {})
    key_id = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    if key_id not in cache:
        cache[key_id] = available_gemini_models(client)
    return cache[key_id]


def choose_model(requested: str, available: list[str]) -> str:
    if requested and requested.lower() not in {"auto", "default"}:
        if requested in available:
            return requested
        raise RuntimeError(
            f"'{requested}' is not available for this API key. "
            f"Choose one of: {', '.join(available[:12]) or 'no generateContent models'}"
        )

    # Prefer a Flash model, while accepting whatever model names Google has
    # currently enabled for this key.
    preferred = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for candidate in preferred:
        if candidate in available:
            return candidate
    flash = [name for name in available if "flash" in name.lower()]
    return (flash or available)[0] if available else ""


def is_temporary_gemini_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return any(marker in message for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "408"))


def is_model_not_found_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return "404" in message and ("MODEL" in message or "NOT_FOUND" in message)


def run_with_progress(work, progress_box, stage: str, timeout_seconds: int = 75):
    """Run a blocking SDK call while keeping the Streamlit page informative."""
    started = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(work)
    try:
        while not future.done():
            elapsed = int(time.monotonic() - started)
            minutes, seconds = divmod(elapsed, 60)
            minute_note = f" · {minutes} minute(s) elapsed" if minutes else ""
            progress_box.info(
                f"**Stage:** {stage}  \n"
                f"Waiting for the LLM… `{minutes:02d}:{seconds:02d}`{minute_note}"
            )
            if elapsed >= timeout_seconds:
                future.cancel()
                raise TimeoutError(f"{stage} exceeded {timeout_seconds} seconds.")
            time.sleep(1)
        return future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def gemini_response(redacted_prompt: str, api_key: str, requested_model: str) -> tuple[str, str]:
    """Discover a valid model and send only the redacted prompt to Gemini."""
    from google import genai
    from google.genai import types

    # 60 seconds is enough for a normal Flash response and prevents the UI
    # from appearing hung if the network or endpoint is unhealthy.
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=45_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    available = session_gemini_models(client, api_key)
    selected_model = choose_model(requested_model.strip(), available)
    if not selected_model:
        raise RuntimeError("This API key has no model that supports generateContent.")
    preferred_fallbacks = [
        "gemini-3.8-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    candidates = []
    for candidate in [selected_model] + preferred_fallbacks + available:
        if candidate not in candidates and ("flash" in candidate.lower() or candidate == selected_model):
            candidates.append(candidate)
    last_error = None
    for candidate in candidates[:4]:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=redacted_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a helpful assistant behind a privacy firewall. "
                        "Answer the user's request directly. Preserve every token like "
                        "<PERSON_1> or <EMAIL_ADDRESS_1> exactly as written; never try "
                        "to guess or reconstruct the hidden values."
                    ),
                    temperature=0.4,
                    max_output_tokens=512,
                ),
            )
            answer = getattr(response, "text", None)
            if not answer:
                raise RuntimeError("Gemini returned an empty response.")
            return answer.strip(), candidate
        except Exception as exc:
            last_error = exc
            if not (is_temporary_gemini_error(exc) or is_model_not_found_error(exc)):
                raise
    raise RuntimeError(f"Gemini models are temporarily unavailable. Last error: {last_error}")


def gemini_client(api_key: str):
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=45_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def generate_with_model(client, model_name: str, prompt: str) -> str:
    from google.genai import types

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a helpful assistant behind a privacy firewall. "
                "Answer directly. Preserve every placeholder token exactly. "
                "Do not add unrelated personal information."
            ),
            temperature=0.4,
            max_output_tokens=512,
        ),
    )
    answer = getattr(response, "text", None)
    if not answer:
        raise RuntimeError("Gemini returned an empty response.")
    return answer.strip()


def restore_placeholders(text: str, mappings: list[dict[str, str]]) -> str:
    text = canonicalize_placeholders(text, mappings)
    for item in mappings:
        text = text.replace(item["placeholder"], item["value"])
    return text


def canonicalize_placeholders(text: str, mappings: list[dict[str, str]]) -> str:
    """Accept model variants such as [DATE_TIME_1] as well as <DATE_TIME_1>."""
    for item in mappings:
        token = item["placeholder"]
        square_token = f"[{token[1:-1]}]"
        text = text.replace(square_token, token)
    return text


def mask_revealed_values(text: str, mappings: list[dict[str, str]]) -> str:
    """Normalize a retry answer before guard review and final restoration."""
    text = canonicalize_placeholders(text, mappings)
    for item in mappings:
        text = text.replace(item["value"], item["placeholder"])
    return text


def review_with_fallback(
    client,
    api_key: str,
    model_name: str,
    redacted_prompt: str,
    answer: str,
    placeholders: list[str],
    progress_box,
    stage: str,
) -> tuple[GuardReview, str]:
    available = session_gemini_models(client, api_key)
    candidates = []
    for candidate in [model_name, "gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"] + available:
        if candidate not in candidates and "flash" in candidate.lower():
            candidates.append(candidate)
    last_error = None
    for candidate in candidates[:3]:
        try:
            guard = VeilGuard(client, candidate)
            review = run_with_progress(
                lambda: guard.review(redacted_prompt, answer, placeholders),
                progress_box,
                stage,
            )
            return review, candidate
        except Exception as exc:
            last_error = exc
            if not (is_temporary_gemini_error(exc) or is_model_not_found_error(exc)):
                raise
    raise RuntimeError(f"VEIL GUARD models are temporarily unavailable. Last error: {last_error}")


def review_with_provider(
    provider: str,
    client,
    api_key: str,
    model_name: str,
    redacted_prompt: str,
    answer: str,
    placeholders: list[str],
    progress_box,
    stage: str,
) -> tuple[GuardReview, str]:
    if provider == "LLM (local)":
        guard = OllamaVeilGuard(model_name)
        return run_with_progress(
            lambda: guard.review(redacted_prompt, answer, placeholders),
            progress_box,
            stage,
        ), model_name
    return review_with_fallback(
        client, api_key, model_name, redacted_prompt, answer, placeholders, progress_box, stage
    )


def demo_response(redacted_prompt: str) -> str:
    return (
        "Demo mode is active because no GEMINI_API_KEY was provided.\n\n"
        f"I received your privacy-safe prompt:\n\n{redacted_prompt}"
    )


def render_copy_box(label: str, value: str, key: str) -> None:
    st.markdown(f"**{label}**")
    st.code(value, language=None)
    st.caption(f"{len(value)} characters · {key}")


st.set_page_config(page_title="VEIL · Secure AI Gateway", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    .hero { padding: 1.4rem 1.6rem; border: 1px solid #293548; border-radius: 18px;
            background: linear-gradient(120deg, #111827, #152c39); margin-bottom: 1.4rem; }
    .hero h1 { margin: 0; color: #e9f5f2; letter-spacing: -0.04em; }
    .hero p { color: #a6c4c0; margin: .45rem 0 0; }
    [data-testid="stMetricValue"] { color: #6ee7b7; }
    </style>
    <div class="hero"><h1>🛡️ VEIL</h1><p>Private prompts. Useful answers. A transparent gateway between you and your selected local or cloud model.</p></div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Gateway settings")
provider = st.sidebar.radio("LLM provider", ("LLM (local)", "LLM (cloud)"), index=0)
if provider == "LLM (local)":
    api_key = ""
    model_name = st.sidebar.text_input(
        "Local LLM model",
        value=os.getenv("OLLAMA_MODEL", "granite4:latest"),
        help="Use the exact name shown by `ollama list`, for example granite4:latest.",
    )
    st.sidebar.success("Local LLM mode: prompts stay on this machine.")
else:
    api_key = st.sidebar.text_input(
        "Cloud LLM API key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Used only for this request and never shown in the dashboard.",
    )
    model_name = st.sidebar.text_input("Cloud LLM model", value=os.getenv("GEMINI_MODEL", "auto"), help="Use auto to discover a model supported by your cloud LLM API key.")
demo_mode = st.sidebar.checkbox("Allow demo mode without an API key", value=True)
if "session_vault" not in st.session_state:
    st.session_state.session_vault = []
if st.sidebar.button("Wipe session vault", use_container_width=True):
    st.session_state.session_vault.clear()
    st.rerun()
st.sidebar.metric("Protected values in this session", len(st.session_state.session_vault))
st.sidebar.info("Redaction happens locally. The temporary vault belongs to this Streamlit session and is wiped when it ends.")

left, right = st.columns([1.12, 0.88], gap="large")
with left:
    st.subheader("1 · Enter a prompt")
    prompt = st.text_area(
        "Your prompt",
        height=210,
        placeholder="Try: Draft a polite reply to Priya at priya@example.com about order #A-234.",
        label_visibility="collapsed",
    )
    submitted = st.button("Redact & ask LLM  →", type="primary", use_container_width=True)

with right:
    st.subheader("How VEIL works")
    st.markdown("""
    **Local scan**

    Sensitive values are found and replaced before any network call.

    **Safe hand-off**

    The LLM sees typed placeholders, never the original values.

    **Visible result**

    You can inspect exactly what was sent and the response that came back.
    """)

if submitted:
    if not prompt.strip():
        st.warning("Enter a prompt first.")
        st.stop()

    safe_prompt, vault = redact(prompt, st.session_state.session_vault)
    st.divider()
    st.subheader("2 · Redacted prompt")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("New sensitive values", len(vault))
    metric_b.metric("Original length", len(prompt))
    metric_c.metric("Provider", provider.split(" ")[0])
    render_copy_box("This is the only prompt that leaves VEIL", safe_prompt, "local redaction complete")

    with st.expander("View redaction details", expanded=bool(vault)):
        if vault:
            st.dataframe(
                [{"Type": item["kind"], "Placeholder": item["placeholder"], "Original": "••••••••"} for item in vault],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write("No new sensitive values were detected. Existing session mappings are reused when applicable.")

    st.divider()
    st.subheader("3 · LLM response")
    progress_box = st.empty()
    llm_solution = ""
    with st.spinner("The LLM is answering the redacted prompt…"):
        try:
            if provider == "LLM (local)":
                selected_model = model_name.strip()
                if not selected_model:
                    raise RuntimeError("Enter a local LLM model name, such as granite4:latest.")
                answer = run_with_progress(
                    lambda: ollama_generate(safe_prompt, selected_model),
                    progress_box,
                    f"Generating local LLM answer with {selected_model}",
                )
                client = None
            if api_key or provider == "LLM (local)":
                if provider != "LLM (local)":
                    answer, selected_model = run_with_progress(
                        lambda: gemini_response(safe_prompt, api_key.strip(), model_name),
                        progress_box,
                        "Generating first LLM answer",
                    )
                    client = gemini_client(api_key.strip())
                request_mappings = [
                    item for item in st.session_state.session_vault
                    if item["placeholder"] in safe_prompt
                ]
                normalized_answer = mask_revealed_values(answer, request_mappings)
                review, guard_model = review_with_provider(
                    provider,
                    client,
                    api_key.strip(),
                    selected_model,
                    safe_prompt,
                    normalized_answer,
                    [item["placeholder"] for item in request_mappings],
                    progress_box,
                    "VEIL GUARD reviewing the first answer",
                )

                if review.verdict == "APPROVE":
                    st.success(
                        f"Approved by VEIL GUARD · score {review.score}/100 · "
                        f"{provider} · `{selected_model}`"
                    )
                    st.caption(f"Guard reason: {review.feedback}")
                    llm_solution = normalized_answer
                    answer = restore_placeholders(normalized_answer, request_mappings)
                elif review.verdict == "RETRY_WITH_REVEAL":
                    st.warning(
                        f"VEIL GUARD requested a targeted retry · score {review.score}/100"
                    )
                    selected = [
                        item for item in request_mappings
                        if item["placeholder"] in review.reveal_placeholders
                    ]
                    retry_prompt = safe_prompt
                    for item in selected:
                        retry_prompt = retry_prompt.replace(item["placeholder"], item["value"])
                    retry_prompt += (
                        "\n\nVEIL GUARD retry instruction: "
                        f"{review.retry_instruction or review.feedback}\n"
                        "Use the revealed value only as needed. Preserve every "
                        "unrevealed placeholder exactly."
                    )
                    with st.spinner("VEIL is running one targeted retry…"):
                        if provider == "LLM (local)":
                            retry_fn = lambda: ollama_generate(retry_prompt, selected_model)
                        else:
                            retry_fn = lambda: generate_with_model(client, selected_model, retry_prompt)
                        retry_answer = run_with_progress(
                            retry_fn,
                            progress_box,
                            "Generating targeted model retry",
                        )
                    retry_normalized = mask_revealed_values(retry_answer, request_mappings)
                    second_review, _ = review_with_provider(
                        provider,
                        client,
                        api_key.strip(),
                        selected_model,
                        safe_prompt,
                        retry_normalized,
                        [item["placeholder"] for item in request_mappings],
                        progress_box,
                        "VEIL GUARD reviewing the retry",
                    )
                    if second_review.verdict == "APPROVE":
                        st.success(
                            f"Approved by VEIL GUARD after retry · score {second_review.score}/100"
                        )
                        st.caption(f"Guard reason: {second_review.feedback}")
                        llm_solution = retry_normalized
                        answer = restore_placeholders(retry_normalized, request_mappings)
                    else:
                        st.error("solution rejected")
                        st.warning("bad prompt")
                        st.caption(f"VEIL GUARD: {second_review.feedback}")
                        st.stop()
                else:
                    st.error("solution rejected")
                    st.warning("bad prompt")
                    st.caption(f"VEIL GUARD score: {review.score}/100 · {review.feedback}")
                    st.stop()
            elif demo_mode:
                answer = demo_response(safe_prompt)
                llm_solution = canonicalize_placeholders(answer, st.session_state.session_vault)
                answer = restore_placeholders(answer, st.session_state.session_vault)
                st.success("Approved by VEIL GUARD · demo review")
                st.info("Demo mode: add a live LLM provider in the sidebar to make a real request.")
            else:
                st.error("Configure an LLM provider or enable demo mode in the sidebar.")
                st.stop()
        except Exception as exc:
            st.error(f"LLM request failed: {exc}")
            st.stop()
    progress_box.success("Pipeline complete.")
    st.subheader("LLM solution")
    st.caption("Protected response returned by the LLM, before VEIL restores sensitive values.")
    st.markdown(llm_solution)
    st.divider()
    st.subheader("Final solution")
    st.markdown(answer)
    st.caption("Final response after VEIL restored the approved placeholders locally.")
else:
    st.divider()
    st.subheader("Ready when you are")
    st.caption("Your redacted prompt and LLM response will appear here after you submit.")
