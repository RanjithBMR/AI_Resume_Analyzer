"""
AI Resume Analyzer — Flask Backend
====================================
This module provides the web server for the AI Resume Analyzer application.
It handles PDF uploads, extracts text using PyPDF2, sends the resume content
(and optional job description) to the Groq API for analysis, and renders the
structured results in the browser.

Key Components:
    - PDF text extraction  (extract_text_from_pdf)
    - Groq API integration (analyse_resume)
    - Result parsing       (parse_analysis_sections)
    - Flask routes         (GET /, POST /analyse)
"""

import os
import re
import tempfile
from io import BytesIO

from flask import Flask, render_template, request, redirect, url_for
from PyPDF2 import PdfReader
from groq import Groq

# ---------------------------------------------------------------------------
# Flask App Configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Limit upload size to 16 MB to prevent abuse
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Allowed file extensions — only PDF is permitted
ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename: str) -> bool:
    """Check that the uploaded file has a .pdf extension.

    Args:
        filename: The original filename from the upload.

    Returns:
        True if the file extension is in the allowed set.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# PDF Text Extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all readable text from a PDF file.

    Uses PyPDF2's PdfReader to iterate over every page in the document and
    concatenate the text content.  Empty pages are skipped gracefully.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        A single string containing all extracted text, with pages separated
        by newlines.

    Raises:
        ValueError: If no text could be extracted from the PDF.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    pages_text = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())

    full_text = "\n\n".join(pages_text)

    if not full_text.strip():
        raise ValueError(
            "Could not extract any text from the PDF. "
            "The file may be image-based or encrypted."
        )

    return full_text


# ---------------------------------------------------------------------------
# Groq API Integration
# ---------------------------------------------------------------------------
def analyse_resume(resume_text: str, job_description: str = "") -> str:
    """Send the resume text (and optional job description) to the Groq API
    for AI-powered analysis.

    The function constructs a detailed system prompt that instructs the LLM
    to return structured output with clear section headings.  If a job
    description is provided, the model is also asked to calculate a match
    percentage.

    Args:
        resume_text: Plain text extracted from the uploaded PDF resume.
        job_description: Optional job description for match analysis.

    Returns:
        The raw analysis text returned by the LLM.

    Raises:
        RuntimeError: If the GROQ_API_KEY environment variable is not set.
        Exception:    Propagates any API / network errors.
    """
    # --- Validate API key ---------------------------------------------------
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it before running the application."
        )

    # --- Build the analysis prompt ------------------------------------------
    # The system prompt tells the model *how* to behave; the user message
    # supplies the actual resume content and (optionally) the job description.
    system_prompt = (
        "You are an expert career coach, ATS (Applicant Tracking System) specialist, "
        "and professional resume reviewer.  Analyse the resume provided and return "
        "a detailed, actionable report using the EXACT section headings below.\n\n"
        "## Key Strengths\n"
        "List the candidate's most compelling strengths, achievements, and skills.\n\n"
        "## Skill Gaps\n"
        "Identify missing or weak skills relative to modern industry expectations"
    )

    # Conditionally add job-description-specific instructions
    if job_description.strip():
        system_prompt += " and the provided job description"

    system_prompt += (
        ".\n\n"
        "## ATS Optimisation Suggestions\n"
        "Provide concrete tips to improve ATS compatibility — keyword usage, "
        "formatting, section ordering, and quantifiable achievements.\n\n"
    )

    if job_description.strip():
        system_prompt += (
            "## Match Percentage\n"
            "Estimate how well the resume matches the job description as a "
            "percentage (0-100%).  Briefly justify the score.\n\n"
        )

    system_prompt += (
        "## Overall Summary\n"
        "Wrap up with a concise overall assessment and top three recommended "
        "next steps for the candidate.\n\n"
        "IMPORTANT: Use the exact section headings listed above (prefixed with ##). "
        "Use bullet points for lists.  Be specific, not generic."
    )

    # --- Compose user message -----------------------------------------------
    user_message = f"### RESUME\n\n{resume_text}"
    if job_description.strip():
        user_message += f"\n\n### JOB DESCRIPTION\n\n{job_description}"

    # --- Call the Groq API --------------------------------------------------
    # Using the llama-3.1-8b-instant model (successor to llama3-8b-8192)
    client = Groq(api_key=api_key)

    chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.4,       # Lower temperature for more focused analysis
        max_tokens=2048,       # Enough room for a thorough report
    )

    return chat_completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Result Parsing — Turn raw LLM output into structured sections
# ---------------------------------------------------------------------------

# Map section titles to display icons for the frontend cards
SECTION_ICONS = {
    "Key Strengths":              "💪",
    "Skill Gaps":                 "🔍",
    "ATS Optimisation Suggestions": "🎯",
    "ATS Optimization Suggestions": "🎯",   # handle US spelling too
    "Match Percentage":           "📊",
    "Overall Summary":            "📝",
}


def parse_analysis_sections(raw_text: str) -> list[dict]:
    """Parse the LLM's markdown-like output into a list of section dicts.

    The function looks for lines starting with "##" and splits the text
    into sections.  Each section becomes a dict with keys: title, icon, body.

    If the output doesn't contain recognisable headings, the entire text
    is returned as a single "Analysis" section so the user always sees
    something meaningful.

    Args:
        raw_text: The raw string returned by the LLM.

    Returns:
        A list of dicts, each containing 'title', 'icon', and 'body'.
    """
    sections: list[dict] = []
    current_title = None
    current_body_lines: list[str] = []

    for line in raw_text.split("\n"):
        # Detect section headings (## Heading or **Heading**)
        heading_match = re.match(r"^##\s+(.+)", line.strip())
        if heading_match:
            # Save the previous section (if any)
            if current_title is not None:
                sections.append(_build_section(current_title, current_body_lines))
                current_body_lines = []
            current_title = heading_match.group(1).strip().rstrip("#").strip()
        else:
            current_body_lines.append(line)

    # Don't forget the last section
    if current_title is not None:
        sections.append(_build_section(current_title, current_body_lines))

    # Fallback: if no headings were detected, wrap everything in one card
    if not sections:
        sections.append({
            "title": "Analysis",
            "icon": "📋",
            "body": raw_text.strip(),
        })

    return sections


def _build_section(title: str, body_lines: list[str]) -> dict:
    """Create a section dict from a title and a list of body lines.

    Looks up the appropriate icon from SECTION_ICONS and joins the body
    lines, stripping leading/trailing whitespace.
    """
    # Try to match the icon (case-insensitive partial match)
    icon = "📌"
    for key, value in SECTION_ICONS.items():
        if key.lower() in title.lower():
            icon = value
            break

    return {
        "title": title,
        "icon": icon,
        "body": "\n".join(body_lines).strip(),
    }


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the upload form (home page).

    Simply serves the Jinja2 template with no context variables, which
    causes the template to display the upload form.
    """
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    """Handle the resume analysis form submission.

    Workflow:
        1. Validate that a file was uploaded and is a PDF.
        2. Extract text from the PDF using PyPDF2.
        3. Send the text (and optional job description) to the Groq API.
        4. Parse the structured response into display-ready sections.
        5. Render the results page — or an error message if anything fails.
    """
    try:
        # --- Step 1: File validation ----------------------------------------
        if "resume" not in request.files:
            return render_template("index.html", error="No file was uploaded. Please select a PDF resume.")

        file = request.files["resume"]

        if file.filename == "":
            return render_template("index.html", error="No file selected. Please choose a PDF file.")

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                error="Invalid file type. Only PDF files are accepted.",
            )

        # --- Step 2: Extract text from the PDF ------------------------------
        pdf_bytes = file.read()
        resume_text = extract_text_from_pdf(pdf_bytes)

        # --- Step 3: Grab the optional job description ----------------------
        job_description = request.form.get("job_description", "").strip()

        # --- Step 4: Call the Groq API for analysis -------------------------
        raw_analysis = analyse_resume(resume_text, job_description)

        # --- Step 5: Parse into structured sections -------------------------
        sections = parse_analysis_sections(raw_analysis)

        return render_template(
            "index.html",
            analysis=True,
            sections=sections,
        )

    except ValueError as ve:
        # PDF extraction issues (e.g. image-based PDF, empty file)
        return render_template("index.html", error=str(ve))

    except RuntimeError as re_err:
        # Missing API key or configuration errors
        return render_template("index.html", error=str(re_err))

    except Exception as exc:
        # Catch-all for unexpected errors (API failures, network issues, etc.)
        # In production, you'd log the full traceback here.
        app.logger.exception("Unexpected error during analysis")
        return render_template(
            "index.html",
            error=f"An unexpected error occurred: {str(exc)}. Please try again.",
        )


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Port is configurable via the PORT environment variable (default: 5000)
    port = int(os.environ.get("PORT", 5000))

    # Bind to 0.0.0.0 so the app is accessible on all network interfaces
    # (required for container/cloud deployments)
    app.run(host="0.0.0.0", port=port, debug=True)
