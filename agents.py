from __future__ import annotations

import logging
import os
from typing import Dict

from crewai import Agent

logger = logging.getLogger(__name__)


def _llm_available() -> bool:
    """
    We still check for LLM presence (LM Studio / OpenAI),
    but we DO NOT pass it into CrewAI directly.
    """
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


# -----------------------------
# Agents (NO LLM INSIDE)
# -----------------------------
def create_job_finder_agent() -> Agent:
    return Agent(
        role="Job Finder Agent",
        goal="Find relevant job listings based on user profile",
        backstory=(
            "You are an expert job researcher who can find high-quality job listings "
            "across multiple platforms."
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_job_matcher_agent() -> Agent:
    return Agent(
        role="Job Matcher Agent",
        goal="Match jobs with the user's skills and preferences",
        backstory=(
            "You analyze job descriptions and compare them with candidate profiles "
            "to determine the best fit."
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_risk_analysis_agent() -> Agent:
    return Agent(
        role="Risk Analysis Agent",
        goal="Detect fraudulent or risky job postings",
        backstory=(
            "You evaluate companies and job listings to identify potential scams "
            "or suspicious opportunities."
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_email_writer_agent() -> Agent:
    return Agent(
        role="Email Writer Agent",
        goal="Write personalized and effective cold emails",
        backstory=(
            "You craft concise, human-like outreach emails that maximize response rates "
            "and highlight candidate strengths."
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_outreach_agent() -> Agent:
    return Agent(
        role="Outreach Agent",
        goal="Send emails safely and reliably",
        backstory=(
            "You ensure emails are correctly formatted and delivered using configured methods "
            "like Gmail API."
        ),
        verbose=True,
        allow_delegation=False,
    )


# -----------------------------
# Bundle
# -----------------------------
def agents_bundle() -> Dict[str, Agent]:
    """
    Creates all agents.

    NOTE:
    - LLM is used in tools.py (not here)
    - This avoids CrewAI compatibility issues
    """
    if not _llm_available():
        logger.warning("⚠️ No LLM detected. Using fallback or local model.")

    return {
        "job_finder": create_job_finder_agent(),
        "job_matcher": create_job_matcher_agent(),
        "risk": create_risk_analysis_agent(),
        "email_writer": create_email_writer_agent(),
        "outreach": create_outreach_agent(),
    }