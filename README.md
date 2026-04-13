# JobFinderAgent (CrewAI)

Autonomous (and beginner-friendly) job hunting system using CrewAI:

- Parse resume (PDF or text)
- Extract skills/roles/preferences
- Search jobs (mock, Serper, or Tavily)
- Match + rank jobs
- Basic company risk check
- Generate personalized cold emails
- Send emails (simulate by default; Gmail optional)

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Run (mock mode, no keys required)

1. Put your resume at `resume.txt` or `resume.pdf` in the project root (or set `RESUME_PATH` in `.env`)
2. Run:

```bash
python main.py
```

## Optional: Real job search

Set in `.env`:

- `SEARCH_PROVIDER=serper` and `SERPER_API_KEY=...`, or
- `SEARCH_PROVIDER=tavily` and `TAVILY_API_KEY=...`

## Optional: Send via Gmail API

This project supports Gmail sending using OAuth files.

1. Create OAuth credentials (Desktop app) in Google Cloud console
2. Download the client secrets JSON into project root (ex: `client_secret.json`)
3. In `.env` set:

- `OUTREACH_MODE=gmail`
- `GMAIL_SENDER=you@gmail.com`
- `GMAIL_OAUTH_CLIENT_SECRETS=client_secret.json`
- `GMAIL_OAUTH_TOKEN=token.json` (will be created on first auth)

Then run `python main.py` and follow the browser OAuth prompt.

## Notes

- If no LLM key is configured, the system will still run end-to-end using **safe fallbacks** (template-based extraction and email writing).
- CrewAI will use your configured provider if available; otherwise you still get a working demo pipeline.
