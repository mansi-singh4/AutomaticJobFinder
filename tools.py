from __future__ import annotations

import base64
import logging
import os
import re
import time
from dataclasses import dataclass
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\s\-\(\)]{8,16}\d")
LINKEDIN_RE = re.compile(r"(https?://)?(www\.)?linkedin\.com/[^\s]+", re.IGNORECASE)


@dataclass
class JobListing:
    title: str
    company: str
    location: str
    url: str
    description: str


def retry(func, retries: int = 2, base_delay_s: float = 0.75):
    attempt = 0
    while True:
        try:
            return func()
        except Exception:
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(base_delay_s * (2 ** (attempt - 1)))


def get_llm_client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install dependencies (pip install -r requirements.txt).")
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
    )


def extract_company(title: str, url: str) -> str:
    title = (title or "").strip()
    url = (url or "").strip()

    if "-" in title:
        part = title.split("-")[-1].strip()
        part = re.sub(r"\b(jobs?|careers?)\b", "", part, flags=re.IGNORECASE).strip(" -|")
        if part:
            return _to_company_case(part)

    if "|" in title:
        part = title.split("|")[-1].strip()
        part = re.sub(r"\b(jobs?|careers?)\b", "", part, flags=re.IGNORECASE).strip(" -|")
        if part:
            return _to_company_case(part)

    domain_guess = _company_from_domain(url)
    if domain_guess:
        return _to_company_case(domain_guess)
    return "Company"


def _company_from_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if not netloc:
        return ""
    netloc = netloc.replace("www.", "")
    parts = [p for p in netloc.split(".") if p]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else ""


def _to_company_case(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 &\-]", " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "Company"
    return " ".join([w.capitalize() for w in cleaned.split(" ")])


def read_resume_text(resume_path: str) -> str:
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume file not found: {resume_path}")

    ext = os.path.splitext(resume_path)[1].lower()
    if ext in (".txt", ".md"):
        with open(resume_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".pdf":
        # Preferred parser
        if pdfplumber is not None:
            try:
                pages: List[str] = []
                with pdfplumber.open(resume_path) as pdf:
                    for page in pdf.pages:
                        pages.append(page.extract_text() or "")
                text = "\n".join(pages).strip()
                if text:
                    return text
            except Exception as e:
                logger.warning("pdfplumber failed parsing %s: %s", resume_path, e)

        # Backup parser
        if PdfReader is not None:
            reader = PdfReader(resume_path)
            return "\n".join([(p.extract_text() or "") for p in reader.pages]).strip()

        raise RuntimeError("PDF parsing is unavailable. Install pdfplumber or pypdf.")

    raise ValueError("Unsupported resume format. Use .pdf or .txt.")


def extract_profile_from_resume(
    resume_text: str,
    *,
    target_role: str = "",
    target_location: str = "",
    target_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    target_keywords = target_keywords or []

    email_match = EMAIL_RE.search(resume_text or "")
    phone_match = PHONE_RE.search(resume_text or "")
    linkedin_match = LINKEDIN_RE.search(resume_text or "")

    name = _extract_name_heuristic(resume_text)
    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0) if phone_match else ""
    linkedin = linkedin_match.group(0) if linkedin_match else ""

    skill_pool = set(target_keywords)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{2,}", resume_text):
        if token.lower() in {"experience", "summary", "skills", "project", "projects", "work"}:
            continue
        if token[0].isupper() or any(c in token for c in "+#.") or token.lower() in {
            "python", "docker", "kubernetes", "sql", "numpy", "pandas", "fastapi", "crewai", "openai"
        }:
            skill_pool.add(token)

    profile = {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "target_role": target_role.strip(),
        "target_location": target_location.strip(),
        "skills": sorted(list(skill_pool))[:40],
        "preferences": {"keywords": [k.strip() for k in target_keywords if k.strip()]},
    }
    return profile


def _extract_name_heuristic(resume_text: str) -> str:
    lines = [ln.strip() for ln in (resume_text or "").splitlines() if ln.strip()]
    for ln in lines[:8]:
        if "@" in ln or "linkedin.com" in ln.lower() or any(ch.isdigit() for ch in ln):
            continue
        words = re.findall(r"[A-Za-z]+", ln)
        if 2 <= len(words) <= 4:
            return " ".join([w.capitalize() for w in words])
    return ""


def search_jobs(query: str, *, provider: str = "mock", limit: int = 5) -> List[JobListing]:
    provider = (provider or "mock").lower().strip()
    logger.info("Using provider: %s", provider)

    if provider == "serper":
        api_key = os.getenv("SERPER_API_KEY", "").strip()
        if not api_key:
            logger.error("Serper selected but SERPER_API_KEY is missing.")
            return _mock_jobs(query, limit)
        try:
            response = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": min(limit, 10)},
                timeout=30,
            )
            logger.info("Serper API response status: %s", response.status_code)
            response.raise_for_status()
            data = response.json().get("organic", [])[:limit]
            jobs: List[JobListing] = []
            for d in data:
                title = d.get("title", "Job Opportunity")
                url = d.get("link", "")
                company = extract_company(title, url)
                logger.info("Job title: %s | extracted company: %s", title, company)
                jobs.append(
                    JobListing(
                        title=title,
                        company=company,
                        location="",
                        url=url,
                        description=d.get("snippet", ""),
                    )
                )
            return jobs
        except Exception as e:
            logger.exception("Serper search failed. Falling back to mock. Error: %s", e)
            return _mock_jobs(query, limit)

    return _mock_jobs(query, limit)


def _mock_jobs(query: str, limit: int) -> List[JobListing]:
    role = query.split(" jobs")[0].strip() if " jobs" in query.lower() else "ML Engineer"
    jobs = [
        JobListing(
            title=f"{role} - Northwind AI",
            company=extract_company(f"{role} - Northwind AI", "https://careers.northwind.ai"),
            location="Remote",
            url="https://careers.northwind.ai/jobs/ml-engineer",
            description="Build LLM applications, Python services, and reliable deployment workflows.",
        ),
        JobListing(
            title=f"Jobs | Contoso Careers",
            company=extract_company("Jobs | Contoso Careers", "https://careers.contoso.com/openings"),
            location="Remote",
            url="https://careers.contoso.com/openings",
            description="Work on backend systems, APIs, and cloud infrastructure.",
        ),
    ]
    return jobs[:limit]


def match_score(profile: Dict[str, Any], job: JobListing) -> float:
    prompt = f"""
Score this candidate-job fit from 0 to 100 and return only one number.
Job title: {job.title}
Job description: {job.description}
Candidate skills: {profile.get("skills")}
Target role: {profile.get("target_role")}
"""
    try:
        client = get_llm_client()
        res = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "google/gemma-3-4b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = (res.choices[0].message.content or "").strip()
        m = re.search(r"(\d{1,3}(?:\.\d+)?)", text)
        if not m:
            return 50.0
        value = float(m.group(1))
        return max(0.0, min(100.0, value))
    except Exception:
        return 50.0


def risk_score(company: str, url: str) -> float:
    risk = 20.0
    if not url or not url.startswith("https://"):
        risk += 20.0
    if "gmail.com" in url or "yahoo.com" in url or "outlook.com" in url:
        risk += 30.0
    if company.strip().lower() in {"unknown", "test company", "company"}:
        risk += 10.0
    return min(100.0, max(0.0, risk))


def write_cold_email(profile=None, job=None, match=None, risk=None, **kwargs):
    if profile is None:
        profile = kwargs.get("profile")
    if job is None:
        job = kwargs.get("job")
    if match is None:
        match = kwargs.get("match")
    if risk is None:
        risk = kwargs.get("risk")
    return _write_cold_email_impl(profile=profile or {}, job=job, match=match, risk=risk)


def _write_cold_email_impl(profile: Dict[str, Any], job: JobListing, match, risk):
    name = (profile.get("name") or "").strip() or "Applicant"
    email = (profile.get("email") or "").strip()
    phone = (profile.get("phone") or "").strip()
    linkedin = (profile.get("linkedin") or "").strip()

    prompt = f"""
You are a JOB APPLICANT writing a cold outreach email for a role.
Do not act like a recruiter.
Do NOT use placeholders such as [Your Name] or [Candidate Name].
Do NOT include lines like "check jobs here" or "browse listings".

Candidate data:
Name: {name}
Email: {email}
Phone: {phone}
LinkedIn: {linkedin}
Skills: {profile.get("skills")}

Job:
Title: {job.title}
Company: {job.company}
Description: {job.description}

Write:
1) Subject line
2) Professional concise body showing interest, relevant skills, and asking for next steps.
3) Signature:
Best regards,
{name}
{email}
{phone}
{linkedin}

Format exactly:
Subject: ...
Body: ...
"""
    try:
        client = get_llm_client()
        res = client.chat.completions.create(
            model=os.getenv("LOCAL_LLM_MODEL") or os.getenv("OPENAI_MODEL_NAME", "google/gemma-3-4b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return (res.choices[0].message.content or "").strip()
    except Exception:
        skills = ", ".join((profile.get("skills") or [])[:6]) or "Python and AI development"
        signature_lines = ["Best regards,", name]
        if email:
            signature_lines.append(email)
        if phone:
            signature_lines.append(phone)
        if linkedin:
            signature_lines.append(linkedin)
        signature = "\n".join(signature_lines)
        return (
            f"Subject: Application for {job.title} at {job.company}\n\n"
            f"Body: Hi {job.company} team,\n\n"
            f"My name is {name}, and I am interested in the {job.title} role. "
            f"My background includes {skills}, and I would love to contribute to your team.\n\n"
            f"Could we schedule a short call to discuss next steps?\n\n"
            f"{signature}"
        )


def find_company_email(job_url: str) -> Optional[str]:
    if not job_url:
        return None

    parsed = urlparse(job_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    domain = parsed.netloc.replace("www.", "")

    # Basic domain-based first guess, then verify via pages.
    guessed = [f"careers@{domain}", f"jobs@{domain}", f"hr@{domain}"]

    base = f"{parsed.scheme}://{parsed.netloc}"
    candidates = [job_url, urljoin(base, "/contact"), urljoin(base, "/contact-us")]
    for url in candidates:
        html = _fetch_html(url)
        if not html:
            continue
        emails = _extract_emails(html)
        pick = _pick_best_email(emails)
        if pick:
            return pick

    # Return best-effort domain-based candidate
    return guessed[0]


def _fetch_html(url: str) -> str:
    try:
        r = requests.get(url, timeout=15, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception:
        return ""


def _extract_emails(html: str) -> List[str]:
    emails = set(m.group(0) for m in EMAIL_RE.finditer(html or ""))
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.select('a[href^="mailto:"]'):
                href = a.get("href", "")
                mail = href.split("mailto:", 1)[-1].split("?")[0].strip()
                if mail and EMAIL_RE.fullmatch(mail):
                    emails.add(mail)
        except Exception:
            pass
    return sorted(emails)


def _pick_best_email(emails: List[str]) -> Optional[str]:
    if not emails:
        return None
    preferred = ["careers@", "jobs@", "hr@", "talent@", "recruiting@", "people@"]
    for p in preferred:
        for e in emails:
            if e.lower().startswith(p):
                return e
    return emails[0]


def send_email(
    *,
    to_email: str,
    subject: str,
    body: str,
    mode: str = "simulate",
    gmail_sender: Optional[str] = None,
    client_secrets_path: Optional[str] = None,
    token_path: Optional[str] = None,
    attachment_bytes: Optional[bytes] = None,
    attachment_filename: Optional[str] = None,
    attachment_mime: Optional[str] = None,
):
    mode = (mode or "simulate").lower().strip()
    if mode != "gmail":
        return {
            "mode": "simulate",
            "message_id": f"sim-{int(time.time())}",
            "to": to_email,
            "subject": subject,
            "attached": bool(attachment_bytes),
        }

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    if not gmail_sender:
        raise ValueError("gmail_sender is required for gmail mode.")
    if not client_secrets_path:
        raise ValueError("client_secrets_path is required for gmail mode.")
    if not token_path:
        raise ValueError("token_path is required for gmail mode.")

    scopes = ["https://www.googleapis.com/auth/gmail.send"]
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, scopes)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    raw = _build_raw_message(
        sender=gmail_sender,
        to_email=to_email,
        subject=subject,
        body=body,
        attachment_bytes=attachment_bytes,
        attachment_filename=attachment_filename,
        attachment_mime=attachment_mime,
    )
    result = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return {"mode": "gmail", "message_id": result.get("id"), "to": to_email, "subject": subject}


def _build_raw_message(
    *,
    sender: str,
    to_email: str,
    subject: str,
    body: str,
    attachment_bytes: Optional[bytes],
    attachment_filename: Optional[str],
    attachment_mime: Optional[str],
) -> str:
    if not attachment_bytes:
        msg = f"From: {sender}\nTo: {to_email}\nSubject: {subject}\n\n{body}"
        return base64.urlsafe_b64encode(msg.encode("utf-8")).decode("utf-8")

    outer = MIMEMultipart()
    outer["To"] = to_email
    outer["From"] = sender
    outer["Subject"] = subject
    outer.attach(MIMEText(body, "plain", "utf-8"))
    filename = attachment_filename or "resume.pdf"
    subtype = (attachment_mime or "application/octet-stream").split("/")[-1]
    part = MIMEApplication(attachment_bytes, _subtype=subtype)
    part.add_header("Content-Disposition", "attachment", filename=filename)
    outer.attach(part)
    return base64.urlsafe_b64encode(outer.as_bytes()).decode("utf-8")