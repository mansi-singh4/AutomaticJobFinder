from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from agents import agents_bundle
from tasks import run_pipeline_fallback
from tools import extract_profile_from_resume, read_resume_text, send_email


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    logging.basicConfig(level=getattr(logging, level, logging.INFO))


def load_inputs() -> Tuple[str, str, str, List[str], str, str]:
    return (
        os.getenv("RESUME_PATH", "resume.txt"),
        os.getenv("TARGET_ROLE", "Machine Learning Engineer"),
        os.getenv("TARGET_LOCATION", "Remote"),
        os.getenv("TARGET_KEYWORDS", "Python,LLM,Agents").split(","),
        os.getenv("SEARCH_PROVIDER", "mock"),
        os.getenv("OUTREACH_MODE", "simulate"),
    )


def print_results(results: List[Dict[str, Any]]) -> None:
    for r in results:
        print("\n==============================")
        print(f"{r['job_title']} @ {r['company']}")
        print(f"Match: {r['match_score']} | Risk: {r['risk_score']}")
        print(r["generated_email"])


def maybe_send_emails(results: List[Dict[str, Any]], outreach_mode: str):
    to_email = os.getenv("TEST_RECIPIENT_EMAIL", "").strip()

    for r in results:
        send_email(
            to_email=to_email,
            subject=f"Interest in {r['job_title']}",
            body=r["generated_email"],
            mode=outreach_mode,
            gmail_sender=os.getenv("GMAIL_SENDER"),
            client_secrets_path=os.getenv("GMAIL_OAUTH_CLIENT_SECRETS"),
            token_path=os.getenv("GMAIL_OAUTH_TOKEN"),
        )


def main():
    load_dotenv()
    setup_logging()

    agents_bundle()

    resume_path, role, location, keywords, provider, outreach = load_inputs()

    resume_text = read_resume_text(resume_path)

    profile = extract_profile_from_resume(
        resume_text,
        target_role=role,
        target_location=location,
        target_keywords=keywords,
    )

    query = f"{role} jobs in {location}"

    results = run_pipeline_fallback(
        profile=profile,
        search_provider=provider,
        query=query,
    )

    print_results(results)

    maybe_send_emails(results, outreach)


if __name__ == "__main__":
    main()