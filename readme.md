🧠 AI Resume Match & ATS Optimizer

🚀 A powerful AI-powered career assistant that analyzes your resume against any job description, calculates ATS score, highlights missing skills, and even generates a personalized cover letter — all in one sleek Streamlit app.

🌟 Key Features

✅ Resume Upload (PDF/DOCX) – Upload your resume directly and let the system extract the text automatically.
✅ Job Description Analysis – Understand required skills, roles, and technologies from the job post.
✅ ATS Score Calculation – Get a real-time Applicant Tracking System (ATS) compatibility score.
✅ Skill Gap Insights – Instantly identify missing technical and soft skills.
✅ Improvement Suggestions – AI-generated recommendations to boost your job alignment.
✅ Interactive Visuals – Beautiful skill comparison charts powered by Plotly.
✅ Smart Cover Letter Generator – Generate an ATS-friendly, custom cover letter tailored to the role.
✅ Download Options – Download your optimized cover letter directly from the app.


🧩 Tech Stack

Python 3.9+

Streamlit – Interactive web UI

OpenAI GPT-4o-mini API – Core AI brain for NLP and text generation

PyPDF2 / python-docx – Resume text extraction

Plotly – Data visualization for skill match

Pandas – Data organization and tabular display

⚙️ Installation & Setup

Clone the Repository

git clone https://github.com/yourusername/AI-Resume-Match-ATS.git
cd AI-Resume-Match-ATS


Create a Virtual Environment (Optional)

python -m venv myenv
source myenv/bin/activate   # On Windows: myenv\Scripts\activate


Install Dependencies

pip install -r requirements.txt


(If you don’t have a requirements.txt, here’s a quick one you can make)

streamlit
openai
PyPDF2
python-docx
plotly
pandas


Set Your OpenAI API Key

Go to OpenAI API Keys

Copy your key

Paste it in the app sidebar when prompted

Run the Application

streamlit run app.py

🧮 How It Works

Upload your Resume (PDF/DOCX).

Paste the Job Description of your target role.

Click “Analyze Match” — the system:

Extracts and analyzes your resume

Extracts job requirements

Computes ATS score, overall match, and missing skills

View interactive insights and AI suggestions for improvement.

Optionally, Generate a personalized cover letter instantly!

💡 Future Enhancements (Roadmap)

🧾 AI Resume Rewriter – Automatically rewrite and format your resume for higher ATS scores

☁️ Cloud Deployment (AWS / Streamlit Cloud)

📊 Detailed Candidate Analytics Dashboard

📬 Auto Email Integration – Send your optimized resume directly to recruiters

🧑‍💻 Author

👤 Palwinder Singh
💼 AI Developer & ML Enthusiast
📫 LinkedIn
 | GitHub

⭐ Support the Project

If you find this project helpful, please give it a ⭐ on GitHub!
Your support motivates further improvements and new features.

🛡️ License

This project is licensed under the MIT License — free to use, modify, and distribute.