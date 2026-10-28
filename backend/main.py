
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader


load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )

client = Groq(api_key=API_KEY)

model = "openai/gpt-oss-120b"
app = FastAPI()


# Parse resume
class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None

    total_experience_years: float | None = None

    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


resume_schema = Resume.model_json_schema()


class ChatRequest(BaseModel):
    question: str


# Chat with candidate profile
def ask_candidate(question: str, resume: Resume):

    system_prompt = f"""
You are an AI assistant representing a job candidate.

Below is everything you know about the candidate.

{resume.model_dump_json(indent=2)}

Rules:

1. Answer only using this information.
2. Never hallucinate.
3. If information is unavailable, say:
   "I don't have enough information to answer that."
4. Be professional.
5. Answer as if HR is interviewing this candidate.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content


# Resume parsing
def parse_resume(resume_text: str) -> Resume:

    system_prompt = f"""
You are an expert resume parser.

Extract information from the resume based on its meaning,
not only based on exact section headings.

Different resumes may use different headings.

For example:
- Experience
- Professional Experience
- Work History
- Employment
- Internships

These may all contain relevant experience.

Skills may also appear in the skills section, work experience,
internships or projects.

Return ONLY valid JSON matching this schema:

{resume_schema}

Important rules:

1. Do not invent information.
2. If a value is not available, return null.
3. If a list has no information, return an empty list.
4. Include internships inside experiences.
5. Extract skills mentioned across the entire resume.
"""

    user_prompt = f"""
Parse the following resume:

{resume_text}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response_format = {
        "type": "json_object"
    }

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format
    )

    raw_output = response.choices[0].message.content

    if not raw_output:
        raise ValueError("Groq returned an empty response.")

    data = json.loads(raw_output)
    resume = Resume(**data)

    return resume


# PDF extraction
def read_pdf(file_path: Path) -> str:

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    if not text.strip():
        raise ValueError("No readable text found in the PDF.")

    return text


# Home route
@app.get("/")
def home():
    return {
        "message": "AI Resume Assistant is running"
    }


# Chat route
@app.post("/chat")
def chat(request: ChatRequest):

    resume_path = Path(__file__).resolve().parent / "Arpan_Banik_Resume (3).pdf"

    if not resume_path.exists():
        return {
            "error": "my_resume.pdf not found in the backend folder."
        }

    resume_text = read_pdf(resume_path)
    resume = parse_resume(resume_text)

    answer = ask_candidate(request.question, resume)

    return {
        "answer": answer
    }
