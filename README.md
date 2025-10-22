Prerequisites:
Before running the script, ensure you have:
Python 3.8+
The following Python libraries installed:
pip install pandas numpy sentence-transformers rapidfuzz PyPDF2 python-docx spacy google-generativeai

Download the English NLP model for spaCy:
python -m spacy download en_core_web_sm

Environment Setup
1. Directory Structure
Your project folder should look like this:

📁 project_root/
 ├── skill_rec_cons.py
 ├── jobs.csv                # Job dataset with columns: job_role, required_skills
 ├── resumes/                # Folder containing resumes to process
 │    ├── resume1.pdf
 │    ├── resume2.docx
 │    └── ...
 └── consultant_job_skill_results.csv  # Output (auto-generated)

2. API Key (Optional)
To enable Gemini API-based recommendations, set your API key as an environment variable:
Linux/Mac
export GENAI_API_KEY="your_api_key_here"

Windows (PowerShell)
setx GENAI_API_KEY "your_api_key_here"

If the API key is missing, the script will still run using embedding-based fallback logic.

How to Run:
Place all your resumes inside the resumes/ directory.
Ensure your job dataset (jobs.csv) is properly formatted.

Run the script:
python skill_rec_cons.py
Once processing completes, results will be saved in:
consultant_job_skill_results.csv
