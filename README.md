Resume Analyzer – Quick Instruction Manual
A Python tool that extracts, scores, and analyzes resumes using AI + ML, with optional Gemini 2.0 Flash support.

1. How It Works
The script automatically:
Reads resumes (PDF/DOCX/TXT) from the resumes/ folder
Extracts text
Scores the resume (action verbs, skills, experience, metrics, sections, etc.)
Generates improvement recommendations
Matches resumes with jobs (optional)
Saves results to CSV and JSON
Stores training data and auto-trains ML models over time

2. Folder Setup
resume_analysis.py
resumes/                 # Add your resume files here
jobs.csv                 # Optional job list (job_role, required_skills)
scoring_criteria.csv     # Scoring rules
models/                  # Auto-saved ML models
training_resumes.csv     # Auto-generated training data

3. Install Requirements
pip install -r requirements.txt
python -m spacy download en_core_web_sm

4. Enable Gemini (Optional)
Set API key:
export GEMINI_API_KEY="your_key"
If not provided, the script uses fallback extraction.

5. Run the Analyzer
python resume_analysis.py
The script will:
Analyze all resumes in resumes/
Save results to:
resume_analysis_results.csv
resume_analysis_results_detailed.json

6. Output Highlights
Each resume includes:
Score, grade, quality
Experience, skills, metrics, action verbs
Recommendations
Top job match (if jobs.csv is present)

7. Customize Scoring
Edit scoring_criteria.csv to change:
Word count ranges
Skill weight
Experience weight
Action verb list
Skills list
Section keywords
Grade thresholds
No code changes needed.

8. Auto-Training
As more resumes are processed, training data grows.
Once 10+ samples exist, ML models auto-train and improve scoring accuracy.

For More Details
Refer to **resume_analysis.pdf** for complete documentation and deeper technical explanations.
