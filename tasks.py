from __future__ import annotations

import logging
from typing import Any, Dict, List

try:
    # CrewAI API can vary across versions / environments (e.g. streamlit launcher).
    # The Streamlit app only needs `run_pipeline_fallback`, so we keep this optional.
    from crewai import Task  # type: ignore
except Exception:  # pragma: no cover
    Task = object  # type: ignore

from tools import JobListing, match_score, risk_score, search_jobs, write_cold_email

logger = logging.getLogger(__name__)


# NOTE:
# We create CrewAI Task objects for "agent structure", but we also keep a direct,
# beginner-friendly Python pipeline so the project works even without an LLM key.


def task_find_jobs(agent, *, query: str) -> Task:
    return Task(
        description=(
            "Find a list of relevant job listings for the user.\n"
            f"Search query: {query}\n\n"
            "Return a compact list of jobs with: title, company, location, url, and a short description."
        ),
        expected_output="JSON list of job listings (title, company, location, url, description).",
        agent=agent,
    )


def task_match_jobs(agent, *, profile: Dict[str, Any], jobs: List[JobListing]) -> Task:
    return Task(
        description=(
            "Score how well the user's profile matches each job.\n\n"
            f"User profile (summary): target_role={profile.get('target_role')}, "
            f"keywords={profile.get('preferences', {}).get('keywords', [])}\n\n"
            f"Jobs: {len(jobs)} listings. Return match scores 0..100 for each job."
        ),
        expected_output="JSON list of job urls with match_score 0..100 and a one-line reason.",
        agent=agent,
    )


def task_risk_analysis(agent, *, jobs: List[JobListing]) -> Task:
    return Task(
        description=(
            "Assess basic risk/legitimacy for each company behind the job listing.\n"
            "Use signals: website presence, suspicious domains, and any basic reputation hints.\n\n"
            f"Jobs: {len(jobs)} listings. Return risk_score 0..100 (higher = riskier) for each."
        ),
        expected_output="JSON list of job urls with risk_score 0..100 and key signals.",
        agent=agent,
    )


def task_write_emails(agent, *, profile: Dict[str, Any], jobs: List[JobListing]) -> Task:
    return Task(
        description=(
            "Write short, personalized cold emails for each job.\n"
            "Make it specific to the job title and the user's top skills.\n"
            "Keep it professional and concise, include a clear CTA.\n"
            "Be very humble and polite .\n"
            f"Jobs: {len(jobs)}"
        ),
        expected_output="JSON list of job urls with email subject and email body.",
        agent=agent,
    )


def task_outreach(agent, *, emails: List[Dict[str, Any]]) -> Task:
    return Task(
        description=(
            "Send the prepared emails safely using the configured method. "
            "If sending isn't enabled, simulate sends and return message ids.\n\n"
            f"Emails to send: {len(emails)}"
        ),
        expected_output="JSON list of sends with message_id and status.",
        agent=agent,
    )


# -----------------------------
# Beginner-friendly pipeline
# -----------------------------
def run_pipeline_fallback(
    *,
    profile: Dict[str, Any],
    search_provider: str,
    query: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Runs the entire job->match->risk->email flow without requiring an LLM.
    Returns structured results ready to print.
    """
    jobs = search_jobs(query, provider=search_provider, limit=limit)
    results: List[Dict[str, Any]] = []
    for job in jobs:
        m = match_score(profile, job)
        r = risk_score(job.company, job.url)  # using job.url as a proxy for website presence
        email = write_cold_email(profile=profile, job=job, match=m, risk=r)
        logger.info("job title=%s | company=%s | match score=%s", job.title, job.company, m)
        results.append(
            {
                "job_title": job.title,
                "company": job.company,
                "match_score": m,
                "risk_score": r,
                "job_url": job.url,
                "generated_email": email,
            }
        )

    # Rank: highest match first, then lowest risk
    results.sort(key=lambda x: (-x["match_score"], x["risk_score"]))
    return results
