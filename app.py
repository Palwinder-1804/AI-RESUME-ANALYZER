import streamlit as st
from openai import OpenAI
import json
import PyPDF2
import docx
import pandas as pd
import plotly.express as px


# -------------------- 🔧 Utility: Read Resume --------------------
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for p in doc.paragraphs:
        text += p.text + "\n"
    return text


def load_resume(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return read_docx(uploaded_file)
    else:
        st.error("Unsupported file type — please upload PDF or DOCX.")
        return ""


# -------------------- 🧹 Safe JSON Parser --------------------
def parse_json_response(raw_text):
    if not raw_text:
        return {}

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        st.error("⚠️ Invalid JSON response from API. Please retry.")
        st.text_area("Raw output for debugging:", cleaned)
        return {}


# -------------------- ⚙️ Job Analyzer --------------------
class JobAnalyzer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def analyze_job(self, job_description):
        prompt = f"""
Analyze this job description and return ONLY valid JSON:
{{
"required_skills": ["..."],
"soft_skills": ["..."],
"experience_required": "...",
"education": "...",
"key_responsibilities": ["..."],
"industry": "...",
"job_level": "...",
"technologies": ["..."]
}}

Job Description:
{job_description}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return parse_json_response(response.choices[0].message.content)


# -------------------- 🧾 Resume Analyzer --------------------
class ResumeAnalyzer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def analyze_resume(self, resume_text):
        prompt = f"""
Analyze this resume and return ONLY valid JSON:
{{
"technical_skills": ["..."],
"soft_skills": ["..."],
"experience": "...",
"education": "...",
"projects": ["..."],
"certifications": ["..."],
"achievements": ["..."]
}}

Resume:
{resume_text}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return parse_json_response(response.choices[0].message.content)


# -------------------- 🎯 Match + ATS Analyzer --------------------
class MatchAnalyzer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def analyze_match(self, job_data, resume_data):
        prompt = f"""
Compare the following Job Description and Resume.
Return ONLY valid JSON with this format:
{{
"overall_match_percentage": "85%",
"ats_score": "88%",
"matching_skills": ["..."],
"missing_skills": ["..."],
"ats_optimization_suggestions": [
    {{
        "section": "Skills",
        "suggested_change": "Add more cloud technologies like AWS, GCP",
        "reason": "These are mentioned in the JD but missing from the resume"
    }}
],
"recommendations": [
    "Quantify achievements using metrics.",
    "Add certifications like AWS Certified Developer for stronger alignment."
]
}}

Job Description:
{json.dumps(job_data, indent=2)}

Resume:
{json.dumps(resume_data, indent=2)}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return parse_json_response(response.choices[0].message.content)


# -------------------- 💌 Cover Letter Generator --------------------
class CoverLetterGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_cover_letter(self, job_data, resume_data, tone="professional"):
        prompt = f"""
Write a {tone} cover letter based on:
Job Details:
{json.dumps(job_data, indent=2)}

Candidate Resume:
{json.dumps(resume_data, indent=2)}

Make it concise, ATS-friendly, and highlight relevant skills.
Return ONLY plain text (no JSON).
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()


# -------------------- 🌟 Streamlit App --------------------
def main():
    st.set_page_config(page_title="AI Resume Match & ATS Optimizer", layout="wide")
    st.title("🚀 AI Resume Match & ATS Optimizer")

    # API key input
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API key to continue.")
        return

    job_analyzer = JobAnalyzer(api_key)
    resume_analyzer = ResumeAnalyzer(api_key)
    match_analyzer = MatchAnalyzer(api_key)
    cover_gen = CoverLetterGenerator(api_key)

    # Upload and input section
    st.sidebar.header("📋 Inputs")
    job_description = st.sidebar.text_area("Paste Job Description", height=250)
    resume_file = st.sidebar.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

    if st.sidebar.button("Analyze Match"):
        if not job_description or not resume_file:
            st.error("Please upload your resume and paste the job description.")
            return

        with st.spinner("Extracting resume text..."):
            resume_text = load_resume(resume_file)

        with st.spinner("Analyzing job description..."):
            job_data = job_analyzer.analyze_job(job_description)

        with st.spinner("Analyzing resume..."):
            resume_data = resume_analyzer.analyze_resume(resume_text)

        with st.spinner("Comparing match and ATS..."):
            match_data = match_analyzer.analyze_match(job_data, resume_data)

        if not match_data:
            st.error("No valid analysis data received.")
            return

        # Save data in session state for later (cover letter)
        st.session_state.job_data = job_data
        st.session_state.resume_data = resume_data
        st.session_state.match_data = match_data

        # -------------------- Results Display --------------------
        st.header("📊 Match & ATS Analysis")

        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Match", match_data.get("overall_match_percentage", "N/A"))
        col2.metric("ATS Score", match_data.get("ats_score", "N/A"))
        col3.metric("Missing Skills", len(match_data.get("missing_skills", [])))

        # Skills analysis chart
        skills_data = pd.DataFrame({
            "Type": ["Matching Skills", "Missing Skills"],
            "Count": [len(match_data.get("matching_skills", [])),
                      len(match_data.get("missing_skills", []))]
        })

        fig = px.bar(skills_data, x="Type", y="Count", color="Type",
                     color_discrete_sequence=["#4CAF50", "#E74C3C"],
                     title="Skills Comparison")
        st.plotly_chart(fig)

        # Recommendations
        st.subheader("💡 Recommendations")
        for rec in match_data.get("recommendations", []):
            st.info(f"✅ {rec}")

        # ATS Suggestions
        st.subheader("🤖 ATS Optimization Suggestions")
        for sug in match_data.get("ats_optimization_suggestions", []):
            st.warning(f"**Section:** {sug.get('section')} — {sug.get('suggested_change')}")
            st.write(f"Reason: {sug.get('reason')}")

    # -------------------- Cover Letter (works after analysis) --------------------
    if "job_data" in st.session_state and "resume_data" in st.session_state:
        st.subheader("📝 Generate Custom Cover Letter")
        tone = st.selectbox("Select tone", ["Professional", "Friendly", "Confident"])
        if st.button("Generate Cover Letter"):
            with st.spinner("Generating cover letter..."):
                cover = cover_gen.generate_cover_letter(
                    st.session_state.job_data,
                    st.session_state.resume_data,
                    tone.lower()
                )
                if cover:
                    st.text_area("Generated Cover Letter", cover, height=350)
                    st.download_button("📥 Download Cover Letter", cover, "cover_letter.txt", "text/plain")
                else:
                    st.error("Failed to generate cover letter. Please try again.")


if __name__ == "__main__":
    main()
