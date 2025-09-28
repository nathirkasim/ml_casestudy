import pandas as pd
import numpy as np
import re

# --- 1. Load Data ---
# Using the newly provided file 'simplified_job_roles.csv' for job data
try:
    jobs_df = pd.read_csv('simplified_job_roles.csv')
    # Assuming 'sample_user_data.csv' is available
    user_df = pd.read_csv('sample_user_data.csv') 
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure 'simplified_job_roles.csv' and 'sample_user_data.csv' are uploaded and accessible.")
    raise

# --- 2. Data Preprocessing and Skill/Title Standardization Functions ---

def preprocess_skill(skill_str):
    """Cleans and tokenizes a single skill string."""
    if pd.isna(skill_str):
        return set()
    # Lowercase, strip surrounding spaces, split by common delimiters
    skills = [s.strip().lower() for s in re.split(r'[,|]', skill_str) if s.strip()]
    
    cleaned_skills = set()
    for s in skills:
        s = s.replace('"', '').replace('.', '').replace('(', '').replace(')', '').strip()
        if s and s not in ('not disclosed by recruiter', 'nan'):
            cleaned_skills.add(s)
            
    return cleaned_skills

def standardize_job_title(title):
    """Standardizes job titles for robust matching (removes non-alphanumeric and spaces)."""
    if pd.isna(title):
        return ""
    # Lowercase, remove all non-alphanumeric characters (including spaces, periods, commas, etc.)
    title = re.sub(r'[\W_]+', '', str(title).lower().strip())
    return title

# --- 3. Build the Robust Job Skill Map ---

job_skill_map = {}
for job_title in jobs_df['Job Title'].unique():
    standard_title = standardize_job_title(job_title)
    if not standard_title:
        continue
        
    job_rows = jobs_df[jobs_df['Job Title'] == job_title]
    all_skills = set()
    for skills_str in job_rows['Key Skills'].dropna():
        all_skills.update(preprocess_skill(skills_str))
        
    # Aggregate skills for the standardized title.
    if standard_title in job_skill_map:
        job_skill_map[standard_title]['required_skills'].update(all_skills)
        job_skill_map[standard_title]['original_titles'].append(job_title)
    else:
        job_skill_map[standard_title] = {
            'required_skills': all_skills,
            'original_titles': [job_title]
        }

# Preprocess the user's current skills and standardize their target job
user_df['current_skills_set'] = user_df['current_skills'].apply(preprocess_skill)
user_df['standard_target_job'] = user_df['target_job_title'].apply(standardize_job_title)


# --- 4. The Core Recommendation Model Logic ---

def recommend_skills_and_job(user_skills, standard_target_job, original_target_job_title, job_skill_map, score_threshold=50):
    """
    Performs skill gap analysis, scoring, and alternative job suggestion.
    """
    results = {
        'target_job': original_target_job_title,
        'current_skills_match_score': 0.0,
        'suggested_skills': [],
        'alternative_job_suggestion': 'N/A',
        'alternative_job_score': 'N/A'
    }

    # A. Target Job Analysis
    job_data = job_skill_map.get(standard_target_job)
    
    if not job_data or not job_data['required_skills']:
        results['suggested_skills'] = ["Cannot analyze: Target job not found/lacks data in training set."]
        results['current_skills_match_score'] = "N/A"
        return results
    
    required_skills = job_data['required_skills']

    # Calculate match
    matching_skills = user_skills.intersection(required_skills)
    
    # Calculate score
    total_required = len(required_skills)
    match_count = len(matching_skills)
    
    score = (match_count / total_required) * 100 if total_required > 0 else 0.0
    
    results['current_skills_match_score'] = f"{score:.2f}%"

    # Identify missing skills (Feature 1)
    missing_skills = list(required_skills.difference(user_skills))
    results['suggested_skills'] = missing_skills[:3]
    
    
    # B. Alternative Job Recommendation (Feature 2 & 3)
    if score < score_threshold:
        
        best_alt_job = None
        best_alt_score = -1
        
        for standard_alt_job, alt_data in job_skill_map.items():
            if standard_alt_job == standard_target_job or not alt_data['required_skills']:
                continue
                
            required_alt_skills = alt_data['required_skills']

            alt_match_count = len(user_skills.intersection(required_alt_skills))
            alt_total_required = len(required_alt_skills)
            
            alt_score = (alt_match_count / alt_total_required) * 100 if alt_total_required > 0 else 0.0
            
            # Find the best alternative job
            if alt_score > best_alt_score:
                best_alt_score = alt_score
                # Use the first original title for display purposes
                best_alt_job = alt_data['original_titles'][0]
        
        # Suggest the alternative if it's better than the current target score
        if best_alt_job and best_alt_score > score:
            results['alternative_job_suggestion'] = best_alt_job
            results['alternative_job_score'] = f"{best_alt_score:.2f}%"
        else:
             results['alternative_job_suggestion'] = 'N/A (No better alternative found)'
             results['alternative_job_score'] = 'N/A'

    return results

# --- 5. Apply Model to User Data (Testing) and Merge Current Skills ---

recommendations = []
for _, row in user_df.iterrows():
    user_id = row['user_id']
    user_skills = row['current_skills_set']
    original_current_skills = row['current_skills'] # Capture original skills for output
    standard_target_job = row['standard_target_job']
    original_target_job = row['target_job_title']
    
    result = recommend_skills_and_job(
        user_skills=user_skills,
        standard_target_job=standard_target_job,
        original_target_job_title=original_target_job,
        job_skill_map=job_skill_map,
        score_threshold=50
    )
    
    result['user_id'] = user_id
    result['original_current_skills'] = original_current_skills # Add original skills to results
    recommendations.append(result)

# --- 6. Final Presentation ---

results_df = pd.DataFrame(recommendations)

# Clean up and format columns
results_df['Suggested Skills (Top 3 Missing)'] = results_df['suggested_skills'].apply(lambda x: ', '.join([s.title() for s in x]) if isinstance(x, list) else x)

def format_alt_job(row):
    alt_job = row['alternative_job_suggestion']
    alt_score = row['alternative_job_score']
    
    if alt_job.startswith('N/A'):
        return 'N/A'
    
    return f"Job: {alt_job}, Score: {alt_score}"

results_df['Alternative Job Suggestion'] = results_df.apply(format_alt_job, axis=1)

# Define the final column order, INCLUDING the new 'original_current_skills' column
final_df = results_df[[
    'user_id', 
    'target_job', 
    'original_current_skills', # ADDED COLUMN FOR DISPLAY
    'current_skills_match_score', 
    'Suggested Skills (Top 3 Missing)', 
    'Alternative Job Suggestion'
]]

final_df.columns = [
    'User ID', 
    'Target Job', 
    'Current Skills', # New Column Title
    'Match Score (%)', 
    'Suggested Skills (Top 3 Missing)', 
    'Alternative Job Suggestion'
]

# Write the final DataFrame to a CSV file
final_df.to_csv("skill_recommendation_results_final_with_skills.csv", index=False)

# Display the final results table
print("Final results table using simplified job data, including Current Skills column, saved to skill_recommendation_results_final_with_skills.csv:")
print(final_df.to_markdown(index=False))
