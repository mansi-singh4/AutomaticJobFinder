from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover
    st_autorefresh = None  # type: ignore

from tools import (
    JobListing,
    extract_profile_from_resume,
    find_company_email,
    match_score,
    read_resume_text,
    risk_score,
    search_jobs,
    send_email,
    write_cold_email,
)

logger = logging.getLogger(__name__)
PROCESSED_PATH = "processed_jobs.json"
STATE_LOCK = threading.Lock()


def _setup() -> None:
    load_dotenv()
    level = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    logging.basicConfig(level=getattr(logging, level, logging.INFO))
    os.makedirs(".tmp", exist_ok=True)
    st.session_state["_tmp_dir"] = ".tmp"


def _read_uploaded_resume(upload) -> Tuple[str, Optional[bytes], str]:
    if upload is None:
        return "", None, ""
    filename = upload.name
    data = upload.getvalue()
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="ignore"), data, filename
    if ext == ".pdf":
        tmp_path = os.path.join(st.session_state.get("_tmp_dir", "."), f"_upload_{filename}")
        with open(tmp_path, "wb") as f:
            f.write(data)
        return read_resume_text(tmp_path), data, filename
    return data.decode("utf-8", errors="ignore"), data, filename


def _parse_subject_and_body(email_text: str) -> Tuple[str, str]:
    lines = (email_text or "").splitlines()
    if lines and lines[0].lower().startswith("subject:"):
        subject = lines[0].split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).lstrip("\n")
        return subject, body
    return "", email_text


def _load_processed_jobs() -> Dict[str, float]:
    if not os.path.exists(PROCESSED_PATH):
        return {}
    try:
        with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
        # Backward compatibility with old list format.
        if isinstance(data, list):
            now = time.time()
            return {str(url): now for url in data}
        return {}
    except Exception:
        return {}


def _save_processed_jobs(urls: Dict[str, float]) -> None:
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(urls, f, indent=2)


def _build_search_queries(*, role: str, location: str, profile: Dict[str, Any]) -> List[str]:
    skills = [s for s in (profile.get("skills") or []) if isinstance(s, str)]
    top_skills = " ".join(skills[:4])
    return [
        f"{role} jobs {location}",
        f"AI Engineer hiring {location}",
        f"Python ML jobs {location}",
        f"LLM Engineer jobs {location}",
        f"Entry level {role} jobs {location}",
        f"{role} {top_skills} jobs {location}",
        f"{role} posted today {location}",
        f"{role} recent jobs {location}",
        f"{role} new openings {location}",
    ]


def _collect_jobs_for_cycle(*, provider: str, queries: List[str], limit_per_query: int) -> List[JobListing]:
    merged: List[JobListing] = []
    seen_urls: set[str] = set()
    for q in queries:
        jobs = search_jobs(q, provider=provider, limit=limit_per_query)
        for job in jobs:
            if not job.url:
                continue
            if job.url in seen_urls:
                continue
            seen_urls.add(job.url)
            merged.append(job)
    random.shuffle(merged)
    return merged


def process_job(
    *,
    job: JobListing,
    profile: Dict[str, Any],
    resume_bytes: Optional[bytes],
    resume_filename: str,
    outreach_mode: str,
    gmail_sender: str,
    client_secrets: str,
    token_path: str,
    processed_urls: Dict[str, float],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "job_title": job.title,
        "company": job.company,
        "job_url": job.url,
        "match_score": 0.0,
        "risk_score": 0.0,
        "generated_email": "",
        "company_email": "",
        "send_status": "",
        "send_error": "",
        "send_message_id": "",
        "is_new_job": False,
    }

    if job.url in processed_urls:
        result["send_status"] = "Already Applied 🔁"
        return result
    result["is_new_job"] = True

    try:
        result["match_score"] = float(match_score(profile, job))
        result["risk_score"] = float(risk_score(job.company, job.url))
        logger.info("job title=%s | match score=%s", job.title, result["match_score"])

        if result["match_score"] < 30:
            result["send_status"] = f"Skipped ⚠️ (match {result['match_score']} < 30)"
        elif result["risk_score"] > 40:
            result["send_status"] = f"Skipped ⚠️ (risk {result['risk_score']} > 40)"
        else:
            result["generated_email"] = write_cold_email(
                profile=profile, job=job, match=result["match_score"], risk=result["risk_score"]
            )
            result["company_email"] = (find_company_email(job.url, job.company, result["match_score"], result["risk_score"]) or "").strip()
            logger.info("job title=%s | email target=%s", job.title, result["company_email"] or "none")

            if not result["company_email"]:
                result["send_status"] = "Skipped ⚠️ (no email target)"
            else:
                subject, body = _parse_subject_and_body(result["generated_email"])
                subject = subject or f"Application for {job.title} at {job.company}"
                send_res = send_email(
                    to_email=result["company_email"],
                    subject=subject,
                    body=body,
                    mode=outreach_mode,
                    gmail_sender=gmail_sender or None,
                    client_secrets_path=client_secrets or None,
                    token_path=token_path or None,
                    attachment_bytes=resume_bytes,
                    attachment_filename=resume_filename or "resume.pdf",
                    attachment_mime="application/pdf" if resume_filename.lower().endswith(".pdf") else "text/plain",
                )
                result["send_status"] = "Email Sent ✅"
                result["send_message_id"] = str(send_res.get("message_id", ""))

        processed_urls[job.url] = time.time()
        _save_processed_jobs(processed_urls)
        return result
    except Exception as e:
        result["send_status"] = "Failed ❌"
        result["send_error"] = str(e)
        logger.exception("Failed processing job %s: %s", job.url, e)
        processed_urls[job.url] = time.time()
        _save_processed_jobs(processed_urls)
        return result


def job_monitor_loop(config: Dict[str, Any]) -> None:
    processed_urls = _load_processed_jobs()
    logger.info("Starting autonomous monitor loop.")
    while not config["stop_event"].is_set():
        try:
            logger.info("Starting job search cycle.")
            queries = _build_search_queries(
                role=config["role"],
                location=config["location"],
                profile=config["profile"],
            )
            jobs = _collect_jobs_for_cycle(
                provider=config["search_provider"],
                queries=queries,
                limit_per_query=config["max_results"],
            )
            total_fetched = len(jobs)
            new_jobs = [j for j in jobs if j.url not in processed_urls]
            skipped_existing = total_fetched - len(new_jobs)
            logger.info(
                "Cycle fetched=%s | new=%s | skipped_existing=%s",
                total_fetched,
                len(new_jobs),
                skipped_existing,
            )
            with STATE_LOCK:
                config["state"]["last_cycle_jobs"] = total_fetched
                config["state"]["last_cycle_new_jobs"] = len(new_jobs)
                config["state"]["last_cycle_skipped"] = skipped_existing
            for idx, job in enumerate(new_jobs):
                if config["stop_event"].is_set():
                    break
                result = process_job(
                    job=job,
                    profile=config["profile"],
                    resume_bytes=config["resume_bytes"],
                    resume_filename=config["resume_filename"],
                    outreach_mode=config["outreach_mode"],
                    gmail_sender=config["gmail_sender"],
                    client_secrets=config["client_secrets"],
                    token_path=config["token_path"],
                    processed_urls=processed_urls,
                )
                with STATE_LOCK:
                    config["state"]["results"] = [result] + config["state"]["results"][:49]
                    config["state"]["last_update_ts"] = time.time()
                if idx < len(new_jobs) - 1:
                    time.sleep(random.uniform(2.0, 3.0))
        except Exception as e:
            logger.exception("Search cycle failed: %s", e)

        wait_seconds = random.randint(300, 600)
        logger.info("Sleeping %s seconds until next cycle.", wait_seconds)
        for _ in range(wait_seconds):
            if config["stop_event"].is_set():
                break
            time.sleep(1)
    logger.info("Automation loop stopped.")


def _start_automation(
    *,
    resume_text: str,
    resume_bytes: Optional[bytes],
    resume_filename: str,
    role: str,
    location: str,
    keywords: List[str],
    search_provider: str,
    outreach_mode: str,
    max_results: int,
    gmail_sender: str,
    client_secrets: str,
    token_path: str,
) -> None:
    profile = extract_profile_from_resume(
        resume_text,
        target_role=role,
        target_location=location,
        target_keywords=keywords,
    )
    stop_event = threading.Event()
    state = {
        "results": [],
        "last_update_ts": time.time(),
        "last_cycle_jobs": 0,
        "last_cycle_new_jobs": 0,
        "last_cycle_skipped": 0,
    }
    config = {
        "profile": profile,
        "role": role,
        "location": location,
        "resume_bytes": resume_bytes,
        "resume_filename": resume_filename,
        "search_provider": search_provider,
        "outreach_mode": outreach_mode,
        "max_results": max_results,
        "gmail_sender": gmail_sender,
        "client_secrets": client_secrets,
        "token_path": token_path,
        "stop_event": stop_event,
        "state": state,
    }
    t = threading.Thread(target=job_monitor_loop, args=(config,), daemon=True)
    t.start()
    st.session_state["automation"] = {
        "thread": t,
        "stop_event": stop_event,
        "config": config,
        "resume_hash": hashlib.sha256((resume_text + resume_filename).encode("utf-8")).hexdigest(),
    }


def main() -> None:
    _setup()
    st.set_page_config(page_title="JobFinderAgent", page_icon="💼", layout="wide")
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=5000, key="refresh_loop")
    elif st_autorefresh is not None:
        st_autorefresh(interval=5000, key="refresh_loop")

    st.title("JobFinderAgent - Autonomous Mode")
    st.caption("Upload once. The system continuously discovers jobs and applies automatically.")

    with st.sidebar:
        st.subheader("Automation Settings")
        search_provider = st.selectbox("Job search provider", ["serper", "mock"], index=0)
        outreach_default = os.getenv("OUTREACH_MODE", "gmail").strip().lower()
        outreach_index = 1 if outreach_default == "gmail" else 0
        outreach_mode = st.selectbox("Outreach mode", ["simulate", "gmail"], index=outreach_index)
        max_results = st.slider("Max jobs per cycle", min_value=1, max_value=10, value=5)
        role = st.text_input("Target role", value=os.getenv("TARGET_ROLE", "Machine Learning Engineer"))
        location = st.text_input("Location", value=os.getenv("TARGET_LOCATION", "Remote"))
        keywords_csv = st.text_input("Keywords (comma-separated)", value=os.getenv("TARGET_KEYWORDS", "Python,LLM,Agents,CrewAI"))
        st.divider()
        st.subheader("Gmail")
        gmail_sender = st.text_input("Sender email", value=os.getenv("GMAIL_SENDER", ""))
        client_secrets = st.text_input("OAuth client secrets path", value=os.getenv("GMAIL_OAUTH_CLIENT_SECRETS", "client_secret.json"))
        token_path = st.text_input("OAuth token path", value=os.getenv("GMAIL_OAUTH_TOKEN", "token.json"))

    upload = st.file_uploader("Upload Resume (starts automatically)", type=["pdf", "txt", "md"])
    stop_clicked = st.button("Stop Automation")

    automation = st.session_state.get("automation")
    if stop_clicked and automation:
        automation["stop_event"].set()
        st.warning("Automation stop requested.")

    if upload is None:
        st.info("Upload your resume to start continuous autonomous mode.")
        return

    if outreach_mode != "gmail":
        st.warning("Real sending is OFF. Set 'Outreach mode' to 'gmail' to send actual emails.")
        return
    if not gmail_sender.strip():
        st.error("GMAIL sender is required for real email sending.")
        return

    try:
        resume_text, resume_bytes, resume_filename = _read_uploaded_resume(upload)
        if not resume_text.strip():
            st.error("Resume appears empty.")
            return
    except Exception as e:
        st.error(f"Resume parsing failed: {e}")
        return

    upload_hash = hashlib.sha256((resume_text + resume_filename).encode("utf-8")).hexdigest()
    keywords = [k.strip() for k in keywords_csv.split(",") if k.strip()]
    should_start = (
        automation is None
        or not automation["thread"].is_alive()
        or automation.get("resume_hash") != upload_hash
    )
    if should_start:
        _start_automation(
            resume_text=resume_text,
            resume_bytes=resume_bytes,
            resume_filename=resume_filename,
            role=role,
            location=location,
            keywords=keywords,
            search_provider=search_provider,
            outreach_mode=outreach_mode,
            max_results=max_results,
            gmail_sender=gmail_sender,
            client_secrets=client_secrets,
            token_path=token_path,
        )
        st.success("Automation started.")
        automation = st.session_state.get("automation")

    if not automation:
        st.error("Automation is not running.")
        return

    profile = automation["config"]["profile"]
    st.subheader("Extracted Profile")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Name", profile.get("name", "") or "-")
    c2.metric("Email", profile.get("email", "") or "-")
    c3.metric("Phone", profile.get("phone", "") or "-")
    c4.metric("LinkedIn", profile.get("linkedin", "") or "-")

    with STATE_LOCK:
        state = automation["config"]["state"]
        results = list(state.get("results", []))
        last_cycle_jobs = int(state.get("last_cycle_jobs", 0))
        last_cycle_new_jobs = int(state.get("last_cycle_new_jobs", 0))
        last_cycle_skipped = int(state.get("last_cycle_skipped", 0))

    st.subheader("Live Job Dashboard")
    st.caption(
        f"Last cycle: fetched={last_cycle_jobs} | new={last_cycle_new_jobs} | skipped={last_cycle_skipped}"
    )

    if not results:
        st.info("Monitoring is running. Waiting for first cycle results...")
        return

    for r in results:
        with st.container(border=True):
            st.markdown(f"**{r.get('job_title', 'Job')}**")
            left, right = st.columns([0.65, 0.35])
            left.write(f"**Company:** {r.get('company', '')}")
            left.markdown(f"**URL:** [{r.get('job_url', '')}]({r.get('job_url', '')})")
            right.write(f"**Match:** {r.get('match_score', 0)}")
            right.write(f"**Risk:** {r.get('risk_score', 0)}")
            if r.get("is_new_job"):
                right.info("New Job 🆕")
            st.markdown("**Generated email**")
            st.text_area("Email", value=r.get("generated_email", ""), height=170, key=f"email_{r.get('job_url', '')}")

            status = r.get("send_status", "")
            if "Email Sent" in status:
                st.success(status)
            elif "Already Applied" in status:
                st.info(status)
            elif "Failed" in status:
                st.error(status)
            else:
                st.warning(status)

            if r.get("company_email"):
                st.caption(f"Target: {r['company_email']}")
            if r.get("send_message_id"):
                st.caption(f"Message ID: {r['send_message_id']}")
            if r.get("send_error"):
                st.caption(f"Error: {r['send_error']}")


if __name__ == "__main__":
    main()

