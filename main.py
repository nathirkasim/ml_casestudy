import os, re, time, random
import pandas as pd
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz
from PyPDF2 import PdfReader
from docx import Document
import spacy
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
# NOTE: The API key must be dynamically provided in the runtime environment.
GENAI_API_KEY = "YOUR_API_KEY"
RESUME_DIR = "resumes" # Assuming this directory and files are accessible
JOBS_FILE = "jobs.csv" # Assuming this file is accessible
OUTPUT_FILE = "consultant_job_skill_results.csv"
API_DELAY = 0
MAX_RETRIES = 5

# --- TUNING ADJUSTMENTS FOR RECALL ---
# Reverting to slightly lower thresholds to increase skill recall
FUZZY_SKILL_THRESHOLD = 75  # 0-100 fuzzy match threshold
EMBED_SIM_THRESHOLD = 0.62 # cosine threshold for semantic skill match
JOB_TITLE_THRESHOLD = 0.70 # required combined score to accept a match
TITLE_WEIGHT = 0.6
SKILL_WEIGHT = 0.4

# ----------------- SETUP -----------------
# Gemini config and initialization
GEMINI_AVAILABLE = False
try:
    if GENAI_API_KEY:
        genai.configure(api_key=GENAI_API_KEY)
        GEMINI_AVAILABLE = True
except Exception as e:
    pass
if not GEMINI_AVAILABLE:
    print("⚠️ Gemini not available or API key missing — will use embedding fallback for normalization.")

# NLP & embedding models
nlp = spacy.load("en_core_web_sm")
# Load model, setting device='cpu' to prevent potential GPU-related issues
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

# Load job DB
if not os.path.exists(JOBS_FILE):
    print(f"FATAL ERROR: {JOBS_FILE} not found. Please ensure it is present.")
    exit(1)

jobs_df = pd.read_csv(JOBS_FILE).dropna(subset=["job_role", "required_skills"])
jobs_df["job_role"] = jobs_df["job_role"].astype(str).str.strip()

# Build canonical skills vocabulary from jobs.csv (preserve original casing mapping)
skill_to_canonical = {}    # lower -> original (first occurrence)
skills_set = set()
for _, row in jobs_df.iterrows():
    raw = str(row["required_skills"])
    for s in raw.split(","):
        s_clean = s.strip()
        if not s_clean:
            continue
        key = s_clean.lower()
        if key not in skill_to_canonical:
            skill_to_canonical[key] = s_clean
        skills_set.add(key)
skills_list = sorted(list(skills_set))

print(f"Loaded {len(jobs_df)} jobs and {len(skills_list)} unique skills from jobs.csv")

# Precompute embeddings for job titles and skills
job_titles = jobs_df["job_role"].tolist()
job_title_embs = embed_model.encode(job_titles, convert_to_tensor=True, show_progress_bar=False)
skill_embs = embed_model.encode(skills_list, convert_to_tensor=True, show_progress_bar=False)

# ---------------- UTILS -------------------

def gemini_api_call_with_backoff(prompt: str, max_retries: int = MAX_RETRIES):
    """Handles Gemini API calls with exponential backoff and retries."""
    if not GEMINI_AVAILABLE:
        return None

    model = genai.GenerativeModel("gemini-2.5-flash")
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt)
            # Return only the first line/most succinct part
            return resp.text.strip().split("\n")[0].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + (random.random() * 0.1)
                time.sleep(delay)
            else:
                return None
    return None

def clean_text(text: str) -> str:
    """Basic resume text cleaning to remove bullets, odd characters, and compress whitespace."""
    if not text: return ""
    text = re.sub(r"[•▪\u2022\u2023\u25E6\u2043\u2219\u00A0]", " ", text)
    text = re.sub(r"\b\d+[%+]\b", " ", text)
    text = re.sub(r"[-–—]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_resume(path: str) -> str:
    """Extracts text from PDF or DOCX files."""
    t = ""
    try:
        if path.lower().endswith(".pdf"):
            r = PdfReader(path)
            for p in r.pages:
                txt = p.extract_text()
                if txt: t += txt + " "
        elif path.lower().endswith(".docx"):
            doc = Document(path)
            for p in doc.paragraphs:
                t += p.text + " "
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                t = f.read()
    except Exception as e:
        print(f" - Error reading {path}: {e}")
        return ""
    return clean_text(t)

def normalize_title_with_gemini(text_hint: str) -> str:
    """Ask Gemini for a concise job title."""
    if not GEMINI_AVAILABLE: return text_hint

    prompt = (
        "You are a succinct job title normalizer. Given a resume text hint (often a summary or "
        "a job title), return ONLY a single standard job title (e.g., 'Senior Software Engineer' or "
        "'Financial Analyst'). Do not add any conversational text or prefixes.\n\n"
        f"Input hint: {text_hint}\nOutput:"
    )

    title = gemini_api_call_with_backoff(prompt)

    if not title or len(title.split()) > 7:
        return text_hint
    return title

def gemini_recommend_skills(job_role: str, extracted_skills: list, dataset_recs: list, max_api_rec: int = 3):
    """
    Uses Gemini to recommend additional, high-impact skills based on the job role.
    It considers existing skills and dataset recommendations to avoid repetition.
    """
    if not GEMINI_AVAILABLE: return []

    existing_skills_str = ", ".join(extracted_skills)
    # Exclude skills already present or already recommended from the dataset
    excluded_skills = set([s.lower() for s in extracted_skills + dataset_recs])

    prompt = (
        f"Act as a career consultant. A candidate is applying for the role: '{job_role}'.\n"
        f"Their current primary skills include: {existing_skills_str}.\n"
        f"Based on modern industry needs for a '{job_role}', list {max_api_rec} critical, high-impact technical or domain skills "
        f"they should consider adding to their resume. "
        f"Exclude any skills already listed: {', '.join(excluded_skills) or 'None'}.\n"
        "Return ONLY a comma-separated list of the skill names (e.g., 'Kubernetes, Cloud Security, Agile Methodology')."
    )

    recommended_str = gemini_api_call_with_backoff(prompt)

    if not recommended_str: return []

    # Parse and clean the Gemini output
    api_recs = []
    for skill in recommended_str.split(','):
        clean_skill = skill.strip()
        if clean_skill and clean_skill.lower() not in excluded_skills:
            api_recs.append(clean_skill)
            excluded_skills.add(clean_skill.lower())

    return api_recs[:max_api_rec]


def semantic_find_skills_in_text(text: str):
    """Hybrid fuzzy/semantic skill extraction."""
    found = set()
    text_low = text.lower()
    tokens = [t for t in re.split(r"\W+", text_low) if t and len(t) > 1]
    max_ngram = 4
    
    # Build n-grams (1-4 words)
    ngrams = set()
    for n in range(1, max_ngram+1):
        for i in range(0, len(tokens)-n+1):
            ng = " ".join(tokens[i:i+n])
            if len(ng) >= 2: ngrams.add(ng)
    
    ngrams = sorted(ngrams, key=lambda x: -len(x))

    # 1. Fuzzy Pre-Check + Semantic Verification
    for ng in ngrams:
        candidate = process.extractOne(ng, skills_list, scorer=fuzz.WRatio)
        if candidate is None: continue
        
        skill_match, score = candidate[0], candidate[1]
        
        if score >= FUZZY_SKILL_THRESHOLD:
            try:
                # Semantic verification for high confidence
                ng_emb = embed_model.encode(ng, convert_to_tensor=True)
                sims = util.cos_sim(ng_emb, skill_embs).cpu().numpy().squeeze()
                best_idx = int(np.argmax(sims))
                sim_val = float(sims[best_idx])
                
                if sim_val >= EMBED_SIM_THRESHOLD:
                    found.add(skill_to_canonical[skills_list[best_idx]])
            except Exception:
                pass # Skip if embedding fails
        
        if len(found) >= 50: break

    # 2. Fallback: Direct Substring Match
    if len(found) < 50:
        for sl in skills_list:
            if sl in text_low:
                found.add(skill_to_canonical[sl])
    
    return sorted(list(found))

def resume_embedding(text: str):
    return embed_model.encode(text, convert_to_tensor=True)

def compute_skill_overlap_score(resume_skills_lower, job_required_lower):
    """Compute the ratio of matched required skills."""
    if len(job_required_lower) == 0: return 0.0
    overlap = len(set(resume_skills_lower).intersection(set(job_required_lower)))
    ratio = overlap / len(job_required_lower)
    return float(ratio)

def find_best_job_match(title_hint: str, extracted_skills_canonical: list):
    """Finds best match using weighted hybrid score (Title Sim + Skill Overlap)."""
    # 1) Title similarity: Use the embedding of the high-quality title hint
    title_emb = resume_embedding(title_hint)
    sims = util.cos_sim(title_emb, job_title_embs).cpu().numpy().squeeze()
    if sims.ndim == 0: sims = np.array([sims.item()])
    per_job_title_sims = sims

    # 2) Skill overlap score
    job_scores = []
    resume_lower = [s.strip().lower() for s in extracted_skills_canonical]
    for _, row in jobs_df.iterrows():
        req = str(row["required_skills"])
        req_norm = [s.strip().lower() for s in req.split(",") if s.strip()]
        skill_overlap = compute_skill_overlap_score(resume_lower, req_norm)
        job_scores.append(skill_overlap)
    job_scores = np.array(job_scores)

    # 3) Combined hybrid score
    combined = TITLE_WEIGHT * per_job_title_sims + SKILL_WEIGHT * job_scores

    best_idx = int(np.argmax(combined))
    matched_job = jobs_df.iloc[best_idx]["job_role"]
    combined_score = float(combined[best_idx])
    title_sim_for_best = float(per_job_title_sims[best_idx])
    skill_overlap_for_best = float(job_scores[best_idx])

    return matched_job, combined_score, title_sim_for_best, skill_overlap_for_best

def recommend_skills_for_job(resume_skills_canonical, matched_job, max_rec=None):
    """Returns missing skills (canonical casing) from the matched job based on the dataset."""
    row = jobs_df[jobs_df["job_role"] == matched_job]
    if row.empty: return []

    req_raw = str(row.iloc[0]["required_skills"])
    reqs = [s.strip() for s in req_raw.split(",") if s.strip()]
    reqs_lower = [s.lower() for s in reqs]
    resume_lower = [s.lower() for s in resume_skills_canonical]

    missing = []
    for req_original, req_low in zip(reqs, reqs_lower):
        if req_low not in resume_lower:
            missing.append(req_original)

    return missing

# ---------------- MAIN LOOP ----------------
results = []
if not os.path.isdir(RESUME_DIR):
    print(f"FATAL ERROR: Resume directory '{RESUME_DIR}' not found. Cannot process resumes.")
    # Setup mock files for demonstration if not found
    os.makedirs(RESUME_DIR, exist_ok=True)
    with open(os.path.join(RESUME_DIR, "sample_resume_1.txt"), "w") as f:
        f.write("I am a Senior Software Engineer specializing in Python, JavaScript, and AWS. I have experience in data analysis and cloud computing.")
    with open(os.path.join(RESUME_DIR, "sample_resume_2.txt"), "w") as f:
        f.write("A highly skilled Financial Analyst with experience in financial forecasting and Tableau. I also have deep knowledge of Excel.")
    print(f"Created sample resumes in '{RESUME_DIR}' for demonstration.")

for fname in sorted(os.listdir(RESUME_DIR)):
    if not fname.lower().endswith((".pdf", ".docx", ".txt")): continue
    path = os.path.join(RESUME_DIR, fname)
    print("Processing:", fname)

    text = extract_text_from_resume(path)
    if len(text.strip()) == 0:
        print(" - empty or unparseable, skipping")
        continue

    # 2. Extract Candidate Skills
    extracted_skills = semantic_find_skills_in_text(text)

    # 3. ***TUNING: ENHANCED TITLE HINT EXTRACTION USING SPACY***
    short_hint = ""
    doc = nlp(text[:4000]) # Process initial part of the resume

    # Attempt 1: Look for relevant Named Entities (Job Titles/Orgs)
    job_title_candidates = []
    for ent in doc.ents:
        if ent.label_ in ["JOB_TITLE", "ORG", "PERSON"] and len(ent.text.split()) >= 2 and len(ent.text.split()) <= 6:
            # Simple heuristic to prioritize roles over names/companies
            if any(token.text.lower() in ["senior", "engineer", "manager", "analyst", "developer", "consultant", "director"] for token in ent):
                job_title_candidates.append(ent.text)

    if job_title_candidates:
        # Use the two most promising candidates joined
        short_hint = " - ".join(job_title_candidates[:2])
    else:
        # Fallback to Summary or first 30 words (original logic)
        m = re.search(r"(summary[:\-]?|professional summary|profile|about me)[:\s\-]*([\w\s\,\.\-]{20,300})", text[:3000], re.I)
        if m:
            short_hint = m.group(2)[:200]
        else:
            short_hint = " ".join(text.split()[:30])

    # 4. Normalize Title (using Gemini or fallback)
    normalized_title = normalize_title_with_gemini(short_hint)
    if not normalized_title: normalized_title = short_hint

    # 5. Hybrid Job Matching
    matched_job, combined_score, title_sim, skill_overlap = find_best_job_match(normalized_title, extracted_skills)

    # 6. Accept Match and Recommend Skills
    accepted_job = matched_job if combined_score >= JOB_TITLE_THRESHOLD else "No close match found"
    
    recommended_skills = []

    if accepted_job != "No close match found":
        # 1. Dataset-based recommendations (Missing required skills from JOBS_FILE)
        dataset_missing_skills = recommend_skills_for_job(extracted_skills, matched_job)
        
        # Filter 1: Ensure missing skills aren't already extracted
        lower_extracted = [s.lower() for s in extracted_skills]
        dataset_recs_filtered = [r for r in dataset_missing_skills if r.lower() not in lower_extracted]

        # 2. Gemini-based recommendations (Broader/Modern skills)
        # Pass the extracted skills and the dataset recs to Gemini to avoid overlap
        gemini_recs = gemini_recommend_skills(accepted_job, extracted_skills, dataset_recs_filtered)

        # 3. Combine and limit output
        recommended_skills = dataset_recs_filtered + gemini_recs

    results.append({
        "Resume": fname,
        "Normalized Title Hint": normalized_title,
        "Matched Job Role": accepted_job,
        "Combined Score": round(combined_score, 3),
        "Title Sim": round(title_sim, 3),
        "Skill Overlap": round(skill_overlap, 3),
        "Extracted Skills": "; ".join(extracted_skills[:40]) if extracted_skills else "None found",
        "Recommended Skills": "; ".join(recommended_skills[:5]) if recommended_skills else (
            "Perfect Skill Match" if accepted_job != "No close match found" and skill_overlap == 1.0 else "No new skills suggested"
        )
    })

# Save results
df_out = pd.DataFrame(results)
if not df_out.empty:
    df_out.to_csv(OUTPUT_FILE, index=False)
    print("✅ Results saved to", OUTPUT_FILE)

    # --- Empirical Calculations (ML Context) ---
    
    total_resumes = len(df_out)
    
    # 1. Match Rate (Acceptance Rate)
    num_matched = len(df_out[df_out["Combined Score"] >= JOB_TITLE_THRESHOLD])
    percent_matched = (num_matched / total_resumes) * 100 if total_resumes > 0 else 0

    print("\n--- Empirical Model Analysis ---")
    print(f"Total Resumes Processed: {total_resumes}")
    print(f"Match Acceptance Rate (Score >= {JOB_TITLE_THRESHOLD:.2f}): {num_matched} / {total_resumes} ({percent_matched:.2f}%)")
    
    # 2. Score Distribution (Median)
    median_combined_score = df_out["Combined Score"].median()
    median_title_sim = df_out["Title Sim"].median()
    median_skill_overlap = df_out["Skill Overlap"].median()

    print("\nScore Distribution (Median):")
    print(f"  Median Combined Score: {median_combined_score:.3f}")
    print(f"  Median Title Similarity: {median_title_sim:.3f}")
    print(f"  Median Skill Overlap: {median_skill_overlap:.3f}")
    
    # 3. Feature Contribution Analysis
    avg_title_contrib = (df_out["Title Sim"] * TITLE_WEIGHT).mean()
    avg_skill_contrib = (df_out["Skill Overlap"] * SKILL_WEIGHT).mean()
    
    print(f"\nModel Feature Weights: Title={TITLE_WEIGHT} | Skill={SKILL_WEIGHT}")
    print("Average Weighted Feature Contribution:")
    print(f"  Title Sim Contribution: {avg_title_contrib:.3f}")
    print(f"  Skill Overlap Contribution: {avg_skill_contrib:.3f}")
    print("--------------------------------------")
    
    # Print detailed summary table
    print("\n--- Detailed Summary of Results ---")
    print(df_out.to_string(index=False))
else:
    print("No resumes were successfully processed.")

