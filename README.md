# VEIL Final

A polished Streamlit dashboard for the VEIL privacy pipeline:

1. The user enters a prompt.
2. Sensitive data is redacted locally into typed placeholders.
3. Only the redacted prompt is sent to Gemini.
4. Gemini's answer is sent to VEIL GUARD for a 0–100 quality/privacy review.
5. Approved answers have placeholders restored locally and are shown to the user.
6. If a hidden value is genuinely required, VEIL GUARD can request one targeted retry. Otherwise the answer is rejected as a bad prompt.

## Run

```bash
cd final
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
streamlit run app.py
```

The API key may also be entered in the sidebar. If no key is available, demo mode shows the exact redacted payload without making a network request.

The app uses Google's current `google-genai` SDK and automatically discovers a model that supports `generateContent` for the supplied API key. Set `GEMINI_MODEL=auto` to use this behavior, or enter an exact model name in the sidebar.

## Redaction and temporary memory

The redaction layer uses [Microsoft Presidio](https://microsoft.github.io/presidio/) for local PII detection. It supports names, locations, dates/times, phone numbers, emails, cards, URLs, IP addresses, and other common identifiers, plus VEIL-specific order IDs and SRNs. Context-aware fallback recognizers cover labelled values such as `my name is ...` and `address is ...` when the spaCy NER model is unavailable.

The placeholder vault is stored only in `st.session_state` for the current Streamlit session. It remembers mappings across prompts (for example, the same email gets the same placeholder), never renders the original values, and can be wiped immediately with the sidebar button. Install the recommended English NER model for stronger name and location detection:

```bash
python -m spacy download en_core_web_sm
```

No detector guarantees perfect recall. Treat this as a privacy filter and test it against the data formats used by your application.

## VEIL GUARD

`final/veil_guard.py` is the working adapter for the second-stage guard. It sees only the redacted prompt, the model answer, and placeholder names—not the sensitive values. It enforces unknown-placeholder rejection independently of Gemini. At most one retry is allowed, and only the exact placeholders requested by the guard are revealed to Gemini for that retry; the retry output is masked back to placeholders before review and local restoration.

The dashboard also shows the active pipeline stage and elapsed time every second, including an explicit minute counter. Temporary 503/429 capacity failures are handled by disabling long SDK retry backoffs and trying up to three available Flash models.

If an older configured model returns a 404 because it is no longer available to new users, the app automatically retries with `gemini-3.8-flash`. You can also set `GEMINI_MODEL=auto`.

## Local Ollama mode

The sidebar defaults to `Ollama (local)`. Install and start Ollama, then pull a local DeepSeek model:

```bash
ollama serve
ollama pull granite4:latest
streamlit run app.py
```

Choose the exact model name shown by `ollama list`. The default is `granite4:latest`. In this mode, both the answer model and VEIL GUARD run through Ollama at `http://127.0.0.1:11434`; the prompt and response are not sent to Gemini or another cloud service.

Presidio is used when its language model is available. The built-in regex fallback still catches common email, phone, credit-card, IP, order-ID, and SRN patterns.
