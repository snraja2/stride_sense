# CrashCart

## Setup and run

CrashCart uses mock mode by default in this example configuration and needs no API key to develop the local workflow. Copy `.env.example` to `.env` to customize settings.

For live calls, set `LLM_BASE_URL` to your provider's OpenAI-compatible API base URL and `LLM_API_KEY` to its key. The defaults target Nebius Token Factory; `NEBIUS_API_KEY` remains supported as a fallback when `LLM_API_KEY` is empty. Set the five `LLM_MODEL_*` values to model IDs offered by your provider. Never commit `.env`.

Run locally with `LLM_MODE=mock uvicorn app.main:app --reload` after the API scaffold is in place.
