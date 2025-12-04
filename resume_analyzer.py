import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
from typing import Dict, List, Optional
import logging

# Document processing
import PyPDF2
from docx import Document

# NLP and ML
import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Gemini API
import google.generativeai as genai

warnings.filterwarnings('ignore')

# Configuration
RESUMES_FOLDER = "resumes"
JOBS_CSV = "jobs.csv"
TRAINING_DATA_CSV = "training_resumes.csv"
OUTPUT_CSV = "resume_analysis_results.csv"
MODEL_DIR = "models"
SCORING_CRITERIA_CSV = "scoring_criteria.csv"
GEMINI_API_KEY = "YOUR_API_KEY" #insert your gemini api key

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resume_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScoringConfig:
    """Manages scoring criteria loaded from CSV"""
    
    def __init__(self, config_path: str = SCORING_CRITERIA_CSV):
        self.config_path = config_path
        self.criteria = {}
        self.grade_thresholds = {}
        self.quality_thresholds = {}
        self.load_config()
    
    def create_default_config(self):
        """Create default scoring_criteria.csv if it doesn't exist"""
        default_criteria = {
            'criterion': [
                # Word count scoring
                'word_count_min_optimal',
                'word_count_max_optimal',
                'word_count_max_weight',
                'word_count_penalty_per_excess',
                
                # Weights for different features
                'contact_score_weight',
                'section_completeness_weight',
                'experience_weight_per_year',
                'experience_max_points',
                'action_verb_weight',
                'action_verb_max_points',
                'skill_count_weight',
                'skill_count_max_points',
                'number_count_weight',
                'number_count_max_points',
                'has_metrics_bonus',
                
                # Grade thresholds
                'grade_A_plus_threshold',
                'grade_A_threshold',
                'grade_B_threshold',
                'grade_C_threshold',
                
                # Quality thresholds
                'quality_high_threshold',
                'quality_medium_threshold',
                
                # Semantic matching weights
                'semantic_similarity_weight',
                'skill_match_weight',
                
                # Action verbs list (comma-separated)
                'action_verbs',
                
                # Degree keywords (comma-separated)
                'degree_keywords',
                
                # Common skills (comma-separated)
                'common_technical_skills',
                
                # Section keywords (JSON format)
                'section_keywords_json'
            ],
            'value': [
                # Word count
                300,  # min optimal
                800,  # max optimal
                30,   # max weight for word count
                50,   # penalty divisor per excess word
                
                # Feature weights
                10,   # contact score
                10,   # section completeness
                2,    # experience per year
                15,   # experience max
                0.5,  # action verb
                5,    # action verb max
                0.3,  # skill count
                15,   # skill count max
                1.5,  # number count
                10,   # number count max
                5,    # metrics bonus
                
                # Grades
                90,   # A+
                80,   # A
                70,   # B
                60,   # C
                
                # Quality
                80,   # high
                60,   # medium
                
                # Matching
                0.7,  # semantic weight
                0.3,  # skill weight
                
                # Lists
                'led,managed,developed,created,implemented,designed,built,achieved,improved,launched,optimized,delivered',
                'bachelor,master,phd,mba,b.tech,m.tech',
                'python,java,javascript,c++,c#,ruby,php,swift,kotlin,react,angular,vue,node.js,django,flask,spring,html,css,sql,mongodb,postgresql,mysql,aws,azure,gcp,docker,kubernetes,git,machine learning,deep learning,data analysis,nlp,agile,scrum,jira,leadership,communication,tensorflow,pytorch,scikit-learn,pandas,numpy',
                '{"summary":["summary","objective","profile"],"experience":["experience","employment","work history"],"education":["education","academic"],"skills":["skills","technical"],"projects":["projects","portfolio"],"certifications":["certifications","licenses"]}'
            ],
            'description': [
                'Minimum optimal word count for resume',
                'Maximum optimal word count for resume',
                'Maximum points awarded for optimal word count',
                'Penalty divisor for words exceeding maximum',
                'Weight for contact information completeness',
                'Weight for section completeness',
                'Points per year of experience',
                'Maximum points for experience',
                'Weight per action verb occurrence',
                'Maximum points for action verbs',
                'Weight per skill mentioned',
                'Maximum points for skills',
                'Weight per number/metric mentioned',
                'Maximum points for numbers/metrics',
                'Bonus points for having any metrics',
                'Minimum score for A+ grade',
                'Minimum score for A grade',
                'Minimum score for B grade',
                'Minimum score for C grade',
                'Minimum score for high quality',
                'Minimum score for medium quality',
                'Weight for semantic similarity in job matching',
                'Weight for skill match in job matching',
                'List of action verbs to detect',
                'List of degree keywords to detect',
                'Common technical skills to look for',
                'Section keywords mapping (JSON format)'
            ]
        }
        
        df = pd.DataFrame(default_criteria)
        df.to_csv(self.config_path, index=False)
        logger.info(f"Created default scoring criteria: {self.config_path}")
        return df
    
    def load_config(self):
        """Load scoring criteria from CSV"""
        if not os.path.exists(self.config_path):
            logger.warning(f"{self.config_path} not found. Creating default...")
            df = self.create_default_config()
        else:
            df = pd.read_csv(self.config_path)
        
        # Parse criteria into dictionary
        for _, row in df.iterrows():
            criterion = row['criterion']
            value = row['value']
            
            # Convert string values to appropriate types
            if criterion.endswith('_json'):
                try:
                    self.criteria[criterion] = json.loads(value)
                except:
                    logger.warning(f"Failed to parse JSON for {criterion}")
                    self.criteria[criterion] = {}
            elif ',' in str(value) and not criterion.endswith('_weight'):
                # Comma-separated list
                self.criteria[criterion] = [item.strip() for item in str(value).split(',')]
            else:
                # Numeric value
                try:
                    self.criteria[criterion] = float(value) if '.' in str(value) else int(value)
                except:
                    self.criteria[criterion] = value
        
        # Build quick access dictionaries
        self._build_grade_thresholds()
        self._build_quality_thresholds()
        
        logger.info(f"Loaded scoring criteria from {self.config_path}")
    
    def _build_grade_thresholds(self):
        """Build grade threshold dictionary"""
        self.grade_thresholds = {
            'A+': self.criteria.get('grade_A_plus_threshold', 90),
            'A': self.criteria.get('grade_A_threshold', 80),
            'B': self.criteria.get('grade_B_threshold', 70),
            'C': self.criteria.get('grade_C_threshold', 60)
        }
    
    def _build_quality_thresholds(self):
        """Build quality threshold dictionary"""
        self.quality_thresholds = {
            'high': self.criteria.get('quality_high_threshold', 80),
            'medium': self.criteria.get('quality_medium_threshold', 60)
        }
    
    def get(self, key: str, default=None):
        """Get criterion value"""
        return self.criteria.get(key, default)
    
    def reload(self):
        """Reload configuration from CSV"""
        self.load_config()
        logger.info("Scoring criteria reloaded")


class SelfLearningResumeAnalyzer:
    """
    Self-learning resume analyzer with Gemini 2.0 Flash integration and CSV-based scoring
    """
    
    def __init__(self, gemini_api_key: str = None, scoring_config: ScoringConfig = None):
        logger.info("Initializing Self-Learning Resume Analyzer with Gemini...")
        
        # Load scoring configuration
        self.scoring_config = scoring_config or ScoringConfig()
        
        # Configure Gemini
        self.use_gemini = False
        self.gemini_model = None
        if gemini_api_key or GEMINI_API_KEY:
            try:
                genai.configure(api_key=gemini_api_key or GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                self.use_gemini = True
                logger.info("✓ Gemini 2.0 Flash enabled")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}. Using fallback methods.")
        else:
            logger.info("Gemini API key not provided. Using traditional extraction.")
        
        # Load spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Semantic model
        logger.info("Loading sentence transformer...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # ML Models
        self.score_predictor = None
        self.quality_classifier = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
        # Model metadata
        self.model_metadata = {
            'trained': False,
            'training_samples': 0,
            'validation_score': 0.0,
            'last_trained': None
        }
        
        # Create directories
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(RESUMES_FOLDER, exist_ok=True)
        
        # Load existing models
        self._load_models()
        
        logger.info("Initialization complete\n")
    
    def _load_models(self):
        """Load pre-trained models if available"""
        model_path = f"{MODEL_DIR}/score_predictor.pkl"
        if os.path.exists(model_path):
            try:
                self.score_predictor = joblib.load(f"{MODEL_DIR}/score_predictor.pkl")
                self.quality_classifier = joblib.load(f"{MODEL_DIR}/quality_classifier.pkl")
                self.anomaly_detector = joblib.load(f"{MODEL_DIR}/anomaly_detector.pkl")
                self.scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
                self.label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
                
                with open(f"{MODEL_DIR}/model_metadata.json", 'r') as f:
                    self.model_metadata = json.load(f)
                
                self.feature_names = self.model_metadata.get('feature_names', [])
                logger.info(f"Loaded trained models ({self.model_metadata['training_samples']} samples)")
            except Exception as e:
                logger.warning(f"Could not load models: {e}")
    
    def save_models(self):
        """Save trained models"""
        joblib.dump(self.score_predictor, f"{MODEL_DIR}/score_predictor.pkl")
        joblib.dump(self.quality_classifier, f"{MODEL_DIR}/quality_classifier.pkl")
        joblib.dump(self.anomaly_detector, f"{MODEL_DIR}/anomaly_detector.pkl")
        joblib.dump(self.scaler, f"{MODEL_DIR}/scaler.pkl")
        joblib.dump(self.label_encoder, f"{MODEL_DIR}/label_encoder.pkl")
        
        with open(f"{MODEL_DIR}/model_metadata.json", 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        logger.info("Models saved")
    
    # ==================== TEXT EXTRACTION ====================
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF, DOCX, or TXT"""
        ext = Path(file_path).suffix.lower()
        text = ""
        
        try:
            if ext == '.pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
            
            elif ext == '.docx':
                doc = Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
        
        return text
    
    # ==================== GEMINI-ENHANCED EXTRACTION ====================
    
    def gemini_extract_structured_data(self, text: str) -> Dict:
        """Use Gemini to extract structured data from resume"""
        if not self.use_gemini:
            return {}
        
        try:
            prompt = f"""Analyze this resume and extract structured information in JSON format.

Resume Text:
{text[:4000]}

Extract the following in valid JSON format:
{{
    "name": "candidate full name",
    "email": "email address",
    "phone": "phone number",
    "location": "city, country",
    "linkedin": "linkedin url if present",
    "github": "github url if present",
    "summary": "professional summary or objective",
    "experience": [
        {{
            "title": "job title",
            "company": "company name",
            "duration": "time period",
            "responsibilities": ["list", "of", "key", "achievements"]
        }}
    ],
    "education": [
        {{
            "degree": "degree name",
            "institution": "university/college",
            "year": "graduation year",
            "gpa": "gpa if mentioned"
        }}
    ],
    "skills": {{
        "technical": ["skill1", "skill2"],
        "soft": ["skill1", "skill2"],
        "tools": ["tool1", "tool2"]
    }},
    "certifications": ["cert1", "cert2"],
    "projects": [
        {{
            "name": "project name",
            "description": "brief description",
            "technologies": ["tech1", "tech2"]
        }}
    ],
    "languages": ["language1", "language2"],
    "total_experience_years": estimated_years
}}

Return ONLY valid JSON, no explanations."""

            response = self.gemini_model.generate_content(prompt)
            extracted = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            logger.info("✓ Gemini extracted structured data")
            return extracted
        except Exception as e:
            logger.warning(f"Gemini extraction failed: {e}")
            return {}
    
    def gemini_generate_structured_recommendations(self, text: str, features: Dict, scores: Dict) -> Dict:
        """Use Gemini to generate structured recommendations"""
        if not self.use_gemini:
            return self._fallback_structured_recommendations(features, scores)
        
        try:
            prompt = f"""You are an expert resume consultant. Analyze this resume and provide detailed, structured recommendations.

Resume Text:
{text[:3000]}

Current Analysis:
- Score: {scores['predicted_score']:.1f}/100 ({scores['grade']})
- Quality: {scores['quality_label']}
- Word Count: {features['word_count']}
- Experience: {features['experience_years']} years
- Skills Mentioned: {features['skill_count']}

Provide recommendations in this EXACT JSON format:

{{
    "skills_to_add": [
        "skill1 (reason why it's relevant)",
        "skill2 (reason why it's relevant)",
        "skill3 (reason why it's relevant)"
    ],
    "required_skills_missing": [
        "critical skill 1 for this role",
        "critical skill 2 for this role"
    ],
    "areas_to_improve": [
        "Project portfolio: Add 2-3 detailed projects with metrics",
        "Resume structure: Reorganize experience section with bullet points",
        "Quantifiable achievements: Add metrics to 80% of accomplishments",
        "Technical depth: Expand on technologies used in each project"
    ],
    "critical_issues": [
        "Missing contact information (phone/email)",
        "No measurable achievements or metrics"
    ],
    "formatting_suggestions": [
        "Use consistent bullet points throughout",
        "Add clear section headers"
    ]
}}

Be specific and actionable. Return ONLY valid JSON."""

            response = self.gemini_model.generate_content(prompt)
            recommendations = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            logger.info("✓ Gemini generated structured recommendations")
            return recommendations
        except Exception as e:
            logger.warning(f"Gemini recommendations failed: {e}. Using fallback.")
            return self._fallback_structured_recommendations(features, scores)
    
    def _fallback_structured_recommendations(self, features: Dict, scores: Dict) -> Dict:
        """Fallback structured recommendations using config"""
        recommendations = {
            "skills_to_add": [],
            "required_skills_missing": [],
            "areas_to_improve": [],
            "critical_issues": [],
            "formatting_suggestions": []
        }
        
        word_count_min = self.scoring_config.get('word_count_min_optimal', 300)
        
        # Critical issues
        if features['contact_score'] < 0.5:
            missing = []
            if not features['has_email']: missing.append('email')
            if not features['has_phone']: missing.append('phone')
            if missing:
                recommendations['critical_issues'].append(f"Add missing contact: {', '.join(missing)}")
        
        if features['word_count'] < word_count_min:
            recommendations['critical_issues'].append(
                f"Resume too short ({features['word_count']} words). Expand to {word_count_min}-{self.scoring_config.get('word_count_max_optimal', 800)} words"
            )
        
        # Areas to improve
        if not features.get('has_projects', 0):
            recommendations['areas_to_improve'].append("Add project portfolio with 2-3 detailed projects showing technical skills")
        
        if features['action_verb_count'] < 5:
            action_verbs = self.scoring_config.get('action_verbs', [])
            verb_examples = ', '.join(action_verbs[:4]) if action_verbs else 'Led, Developed, Managed, Implemented'
            recommendations['areas_to_improve'].append(f"Use more action verbs ({verb_examples})")
        
        if features['number_count'] < 5:
            recommendations['areas_to_improve'].append("Add quantifiable metrics (increased by X%, reduced by Y hours, managed Z projects)")
        
        if not features.get('has_summary'):
            recommendations['areas_to_improve'].append("Add professional summary highlighting key achievements and career objectives")
        
        # Skills suggestions
        if features['skill_count'] < 10:
            recommendations['skills_to_add'].extend([
                "Industry-relevant technical skills (based on target role)",
                "Modern frameworks and tools",
                "Cloud platforms (AWS/Azure/GCP)"
            ])
        
        recommendations['required_skills_missing'].extend([
            "Domain-specific technical skills",
            "Leadership or team collaboration skills"
        ])
        
        # Formatting
        recommendations['formatting_suggestions'].extend([
            "Use consistent bullet point format",
            "Add clear section headers",
            "Ensure proper spacing between sections"
        ])
        
        return recommendations
    
    # ==================== SKILLS EXTRACTION ====================
    
    def extract_current_skills(self, text: str, gemini_data: Dict = None) -> List[str]:
        """Extract current skills from resume using config"""
        if gemini_data and gemini_data.get('skills'):
            skills_dict = gemini_data['skills']
            all_skills = []
            all_skills.extend(skills_dict.get('technical', []))
            all_skills.extend(skills_dict.get('tools', []))
            all_skills.extend(skills_dict.get('soft', []))
            return all_skills
        
        # Fallback: Extract using config-based skill list
        common_skills = self.scoring_config.get('common_technical_skills', [])
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in common_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill.title())
        
        return list(set(found_skills))
    
    # ==================== FEATURE ENGINEERING ====================
    
    def extract_features(self, text: str, gemini_data: Dict = None) -> Dict:
        """Extract comprehensive features (enhanced with Gemini data and config)"""
        doc = self.nlp(text)
        text_lower = text.lower()
        words = text.split()
        
        features = {}
        
        # Text statistics
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        features['sentence_count'] = len(list(doc.sents))
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['lexical_diversity'] = len(set(words)) / max(len(words), 1)
        
        # Contact info (enhanced with Gemini)
        if gemini_data:
            features['has_email'] = int(bool(gemini_data.get('email')))
            features['has_phone'] = int(bool(gemini_data.get('phone')))
            features['has_linkedin'] = int(bool(gemini_data.get('linkedin')))
            features['has_github'] = int(bool(gemini_data.get('github')))
        else:
            features['has_email'] = int(bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)))
            features['has_phone'] = int(bool(re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)))
            features['has_linkedin'] = int('linkedin.com' in text_lower)
            features['has_github'] = int('github.com' in text_lower)
        
        features['contact_score'] = sum([features['has_email'], features['has_phone'], 
                                             features['has_linkedin'], features['has_github']]) / 4
        
        # Sections (from config)
        sections = self.scoring_config.get('section_keywords_json', {})
        for section, keywords in sections.items():
            features[f'has_{section}'] = int(any(kw in text_lower for kw in keywords))
        features['section_completeness'] = sum([features[f'has_{s}'] for s in sections]) / max(len(sections), 1)
        
        # Action verbs (from config)
        action_verbs = self.scoring_config.get('action_verbs', [])
        features['action_verb_count'] = sum(text_lower.count(verb.lower()) for verb in action_verbs)
        features['action_verb_density'] = features['action_verb_count'] / max(features['word_count'], 1) * 1000
        
        # Metrics
        features['number_count'] = len(re.findall(r'\b\d+[\.,]?\d*\s*[%$KkMm]?\b', text))
        features['has_metrics'] = int(features['number_count'] > 0)
        
        # Experience (enhanced with Gemini)
        if gemini_data and gemini_data.get('total_experience_years'):
            features['experience_years'] = gemini_data['total_experience_years']
        else:
            years = [int(y) for y in re.findall(r'\b(19|20)\d{2}\b', text) 
                         if 1970 < int(y) <= datetime.now().year]
            features['experience_years'] = max(years) - min(years) if len(years) >= 2 else 0
        
        # Skills (enhanced with Gemini)
        if gemini_data and gemini_data.get('skills'):
            skills = gemini_data['skills']
            total_skills = len(skills.get('technical', [])) + len(skills.get('soft', [])) + len(skills.get('tools', []))
            features['skill_count'] = total_skills
        else:
            features['skill_count'] = text_lower.count('skill') * 3
        
        # Education (from config)
        degree_keywords = self.scoring_config.get('degree_keywords', [])
        features['degree_count'] = sum(kw.lower() in text_lower for kw in degree_keywords)
        features['has_gpa'] = int(bool(re.search(r'\b[0-9]\.[0-9]{1,2}\b', text)))
        
        # NLP features
        features['noun_count'] = sum(1 for token in doc if token.pos_ == 'NOUN')
        features['verb_count'] = sum(1 for token in doc if token.pos_ == 'VERB')
        features['entity_count'] = len(doc.ents)
        
        # Quality indicators
        features['uppercase_ratio'] = sum(c.isupper() for c in text) / max(len(text), 1)
        features['unique_word_ratio'] = len(set(words)) / max(len(words), 1)
        
        return features
    
    def prepare_feature_vector(self, features: Dict) -> np.ndarray:
        """Convert features to numpy array"""
        if not self.feature_names:
            self.feature_names = sorted(features.keys())
        vector = np.array([features.get(name, 0) for name in self.feature_names])
        return vector.reshape(1, -1)
    
    # ==================== SCORING (CONFIG-DRIVEN) ====================
    
    def predict_score(self, features: Dict) -> Dict:
        """Predict resume score"""
        if self.model_metadata['trained'] and self.score_predictor and self.quality_classifier and self.anomaly_detector:
            try:
                X = self.prepare_feature_vector(features)
                X_scaled = self.scaler.transform(X)
                pred = self.score_predictor.predict(X_scaled)[0]
                pred = np.clip(pred, 0, 100)
                
                # Check if label_encoder is fitted
                quality_pred = self.quality_classifier.predict(X_scaled)
                if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
                    quality = self.label_encoder.inverse_transform(quality_pred)[0]
                else:
                    quality = self._score_to_quality(pred)
                
                anomaly = self.anomaly_detector.predict(X_scaled)[0] == -1
                return {
                    'predicted_score': round(float(pred), 2),
                    'quality_label': quality,
                    'grade': self._score_to_grade(pred),
                    'is_anomaly': bool(anomaly),
                    'model_based': True
                }
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}. Using rule-based.")
        
        return self._rule_based_scoring(features)
    
    def _rule_based_scoring(self, features: Dict) -> Dict:
        """Rule-based scoring fallback using config criteria"""
        score = 0
        
        # Word count scoring (from config)
        wc = features['word_count']
        wc_min = self.scoring_config.get('word_count_min_optimal', 300)
        wc_max = self.scoring_config.get('word_count_max_optimal', 800)
        wc_weight = self.scoring_config.get('word_count_max_weight', 30)
        wc_penalty = self.scoring_config.get('word_count_penalty_per_excess', 50)
        
        if wc_min <= wc <= wc_max:
            score += wc_weight
        elif wc < wc_min:
            score += (wc / wc_min) * wc_weight
        else:
            score += max(0, wc_weight - ((wc - wc_max) / wc_penalty))
        
        # Contact score
        score += features['contact_score'] * self.scoring_config.get('contact_score_weight', 10)
        
        # Section completeness
        score += features['section_completeness'] * self.scoring_config.get('section_completeness_weight', 10)
        
        # Experience
        exp_weight = self.scoring_config.get('experience_weight_per_year', 2)
        exp_max = self.scoring_config.get('experience_max_points', 15)
        score += min(features['experience_years'] * exp_weight, exp_max)
        
        # Action verbs
        av_weight = self.scoring_config.get('action_verb_weight', 0.5)
        av_max = self.scoring_config.get('action_verb_max_points', 5)
        score += min(features['action_verb_count'] * av_weight, av_max)
        
        # Skills
        skill_weight = self.scoring_config.get('skill_count_weight', 0.3)
        skill_max = self.scoring_config.get('skill_count_max_points', 15)
        score += min(features['skill_count'] * skill_weight, skill_max)
        
        # Numbers/metrics
        num_weight = self.scoring_config.get('number_count_weight', 1.5)
        num_max = self.scoring_config.get('number_count_max_points', 10)
        score += min(features['number_count'] * num_weight, num_max)
        
        # Metrics bonus
        score += features['has_metrics'] * self.scoring_config.get('has_metrics_bonus', 5)
        
        score = min(100, max(0, score))
        return {
            'predicted_score': round(score, 2),
            'quality_label': self._score_to_quality(score),
            'grade': self._score_to_grade(score),
            'is_anomaly': False,
            'model_based': False
        }
    
    def _score_to_quality(self, score: float) -> str:
        """Convert score to quality label using config thresholds"""
        high_threshold = self.scoring_config.quality_thresholds.get('high', 80)
        medium_threshold = self.scoring_config.quality_thresholds.get('medium', 60)
        
        if score >= high_threshold:
            return 'high'
        elif score >= medium_threshold:
            return 'medium'
        else:
            return 'low'
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to grade using config thresholds"""
        thresholds = self.scoring_config.grade_thresholds
        
        if score >= thresholds.get('A+', 90):
            return 'A+ (Exceptional)'
        elif score >= thresholds.get('A', 80):
            return 'A (Excellent)'
        elif score >= thresholds.get('B', 70):
            return 'B (Good)'
        elif score >= thresholds.get('C', 60):
            return 'C (Average)'
        else:
            return 'D (Needs Improvement)'
    
    # ==================== JOB MATCHING ====================
    
    def match_with_jobs(self, text: str, features: Dict, jobs_df: pd.DataFrame) -> List[Dict]:
        """Match resume with jobs using config-based weights"""
        if jobs_df is None or len(jobs_df) == 0:
            return []
        
        resume_embedding = self.semantic_model.encode(text[:2000], convert_to_tensor=True)
        matches = []
        
        # Get weights from config
        semantic_weight = self.scoring_config.get('semantic_similarity_weight', 0.7)
        skill_weight = self.scoring_config.get('skill_match_weight', 0.3)
        
        for _, job in jobs_df.iterrows():
            try:
                job_text = f"{job['job_role']} {job.get('required_skills', '')}"
                job_embedding = self.semantic_model.encode(job_text[:2000], convert_to_tensor=True)
                semantic_sim = util.cos_sim(resume_embedding, job_embedding).item()
                
                job_skills = {s.strip().lower() for s in str(job.get('required_skills','')).split(',') if s.strip()}
                resume_words = {w.lower() for w in text.split() if len(w)>3}
                skill_match = len(job_skills & resume_words) / len(job_skills) if job_skills else 0
                
                final = (semantic_sim * semantic_weight + skill_match * skill_weight) * 100
                
                matches.append({
                    'job_role': str(job.get('job_role','Unknown')),
                    'match_score': round(final, 2),
                    'semantic_similarity': round(semantic_sim*100, 2),
                    'skill_match_pct': round(skill_match*100, 2)
                })
            except:
                continue
        
        return sorted(matches, key=lambda x: x['match_score'], reverse=True)
    
    # ==================== PROCESSING ====================
    
    def process_all_resumes(self):
        """Process all resumes in folder - fully automated"""
        logger.info("\n" + "="*70)
        logger.info("SELF-LEARNING RESUME ANALYZER WITH GEMINI")
        logger.info("="*70)
        
        logger.info(f"Gemini: {'ENABLED ✓' if self.use_gemini else 'DISABLED'}")
        logger.info(f"ML Models: {'TRAINED' if self.model_metadata['trained'] else 'NOT TRAINED (using rules)'}")
        logger.info(f"Scoring Config: {self.scoring_config.config_path}")
        
        # Load jobs
        jobs_df = None
        if os.path.exists(JOBS_CSV):
            try:
                jobs_df = pd.read_csv(JOBS_CSV)
                if 'job_role' in jobs_df.columns and 'required_skills' in jobs_df.columns:
                    jobs_df = jobs_df.dropna(subset=['job_role', 'required_skills'])
                    logger.info(f"Jobs loaded: {len(jobs_df)}")
                else:
                    logger.warning(f"{JOBS_CSV} missing required columns")
                    jobs_df = None
            except Exception as e:
                logger.warning(f"Error loading {JOBS_CSV}: {e}")
                jobs_df = None
        else:
            logger.info(f"{JOBS_CSV} not found - job matching disabled")
        
        # Find resumes
        if not os.path.exists(RESUMES_FOLDER):
            os.makedirs(RESUMES_FOLDER)
            logger.error(f"'{RESUMES_FOLDER}' folder is empty!")
            logger.info("    Add resume files (PDF, DOCX, TXT) and run again")
            return
        
        resume_files = []
        for ext in ['*.pdf', '*.docx', '*.txt']:
            resume_files.extend(Path(RESUMES_FOLDER).glob(ext))
        
        if not resume_files:
            logger.error(f"No resumes found in '{RESUMES_FOLDER}'")
            return
        
        logger.info(f"Found {len(resume_files)} resumes\n")
        
        # Process each resume
        results = []
        for i, resume_path in enumerate(resume_files, 1):
            logger.info(f"[{i}/{len(resume_files)}]")
            try:
                result = self.analyze_resume(str(resume_path), jobs_df)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error during analysis: {e}")
                continue
        
        if not results:
            logger.error("\nNo resumes successfully analyzed")
            return
        
        # Save results
        self._save_results(results)
        
        # Save as training data (auto mode)
        self.save_as_training_data(results)
        
        # Auto-retrain if threshold met
        self.auto_retrain_if_ready()
        
        # Print summary
        self._print_summary(results)
    
    def _save_results(self, results: List[Dict]):
        """Save analysis results to CSV and JSON with structured columns"""
        logger.info(f"\n{'='*70}")
        logger.info("SAVING RESULTS")
        logger.info(f"{'='*70}")
        
        output_data = []
        for result in results:
            features = result['features']
            scores = result['scores']
            matches = result['matches']
            gemini_data = result.get('gemini_data', {})
            recommendations = result.get('recommendations', {})
            current_skills = result.get('current_skills', [])
            
            row = {
                'filename': result['filename'],
                'score': scores['predicted_score'],
                'grade': scores['grade'],
                'quality': scores['quality_label'],
                'is_anomaly': scores.get('is_anomaly', False),
                'model_based': scores['model_based'],
                'gemini_enhanced': bool(gemini_data),
                
                # Current resume metrics
                'word_count': features['word_count'],
                'experience_years': features['experience_years'],
                'skill_count': features['skill_count'],
                'action_verbs': features['action_verb_count'],
                'metrics': features['number_count'],
                'contact_score': features['contact_score'],
                'sections_complete': features['section_completeness'],
                
                # Extracted current skills
                'current_skills': ', '.join(current_skills) if current_skills else 'N/A',
                
                # Structured recommendations
                'skills_to_add': ', '.join(recommendations.get('skills_to_add', [])) if isinstance(recommendations, dict) else 'N/A',
                'required_skills_missing': ', '.join(recommendations.get('required_skills_missing', [])) if isinstance(recommendations, dict) else 'N/A',
                'areas_to_improve': ' | '.join(recommendations.get('areas_to_improve', [])) if isinstance(recommendations, dict) else 'N/A',
                'critical_issues': ', '.join(recommendations.get('critical_issues', [])) if isinstance(recommendations, dict) else 'N/A',
                'formatting_suggestions': ', '.join(recommendations.get('formatting_suggestions', [])) if isinstance(recommendations, dict) else 'N/A'
            }
            
            # Add Gemini-extracted data
            if gemini_data:
                row['name'] = gemini_data.get('name', 'N/A')
                row['email'] = gemini_data.get('email', 'N/A')
                row['phone'] = gemini_data.get('phone', 'N/A')
                row['location'] = gemini_data.get('location', 'N/A')
            
            # Job matching
            if matches:
                row['top_match'] = matches[0]['job_role']
                row['match_score'] = matches[0]['match_score']
            else:
                row['top_match'] = 'N/A'
                row['match_score'] = 0
            
            output_data.append(row)
        
        df = pd.DataFrame(output_data)
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Results: {OUTPUT_CSV}")
        
        json_output = OUTPUT_CSV.replace('.csv', '_detailed.json')
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Detailed: {json_output}")
    
    def _print_summary(self, results: List[Dict]):
        """Print analysis summary"""
        logger.info(f"\n{'='*70}")
        logger.info("ANALYSIS SUMMARY")
        logger.info(f"{'='*70}\n")
        
        for result in results:
            scores = result['scores']
            features = result['features']
            matches = result['matches']
            gemini_data = result.get('gemini_data', {})
            current_skills = result.get('current_skills', [])
            recommendations = result.get('recommendations', {})
            
            print(f"{result['filename']}")
            print(f"    Score: {scores['predicted_score']:.1f}/100 - {scores['grade']}")
            print(f"    Quality: {scores['quality_label']} | Method: {'ML' if scores['model_based'] else 'Rules'}")
            
            if gemini_data:
                print(f"    ✓ Gemini Enhanced")
                if gemini_data.get('name'):
                    print(f"    Candidate: {gemini_data['name']}")
            
            if scores.get('is_anomaly'):
                print(f"    ⚠ ANOMALY DETECTED")
            
            # Current skills
            if current_skills:
                print(f"\n    Current Skills ({len(current_skills)}): {', '.join(current_skills[:5])}")
                if len(current_skills) > 5:
                    print(f"                          {', '.join(current_skills[5:10])}")
            
            # Top recommendations
            if isinstance(recommendations, dict):
                if recommendations.get('critical_issues'):
                    print(f"\n    Critical Issues:")
                    for issue in recommendations['critical_issues'][:2]:
                        print(f"      - {issue}")
                
                if recommendations.get('skills_to_add'):
                    print(f"\n    Skills to Add:")
                    for skill in recommendations['skills_to_add'][:3]:
                        print(f"      - {skill}")
            
            if matches:
                print(f"\n    Top Match: {matches[0]['job_role']} ({matches[0]['match_score']:.1f}%)")
            
            print(f"\n    Metrics:")
            print(f"      Experience: {features['experience_years']} years | Skills: {features['skill_count']}")
            print(f"      Action Verbs: {features['action_verb_count']} | Numbers: {features['number_count']}")
            print(f"\n{'-'*70}\n")
        
        avg_score = np.mean([r['scores']['predicted_score'] for r in results])
        ml_count = sum(1 for r in results if r['scores']['model_based'])
        gemini_count = sum(1 for r in results if r.get('gemini_data'))
        print(f"OVERALL STATISTICS")
        print(f"    Total Resumes: {len(results)}")
        print(f"    Average Score: {avg_score:.1f}/100")
        print(f"    ML Predictions: {ml_count}/{len(results)}")
        print(f"    Gemini Enhanced: {gemini_count}/{len(results)}")
        print(f"\n{'='*70}\n")
    
    # ==================== TRAINING DATA MANAGEMENT ====================
    
    def save_as_training_data(self, results: List[Dict]):
        """Save results as training data automatically"""
        logger.info("\n" + "="*70)
        logger.info("SAVING AS TRAINING DATA (AUTO MODE)")
        logger.info("="*70)
        
        existing_df = pd.read_csv(TRAINING_DATA_CSV) if os.path.exists(TRAINING_DATA_CSV) else pd.DataFrame()
        
        new_data = []
        for r in results:
            new_data.append({
                'filename': r['filename'],
                'text': r['text'][:5000],
                'expert_score': r['scores']['predicted_score'],
                'quality_label': r['scores']['quality_label'],
                'is_hired': 0,
                'word_count': r['features']['word_count'],
                'experience_years': r['features']['experience_years'],
                'skill_count': r['features']['skill_count'],
                'date_added': datetime.now().isoformat(),
                'needs_review': False  # Auto-mark as reviewed
            })
        
        new_df = pd.DataFrame(new_data)
        
        if len(existing_df):
            existing_df = existing_df[~existing_df['filename'].isin(new_df['filename'])]
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
        
        combined.to_csv(TRAINING_DATA_CSV, index=False)
        logger.info(f"\nTraining data saved: {TRAINING_DATA_CSV}")
        logger.info(f"    Total samples: {len(combined)}")
        logger.info(f"    New samples: {len(new_df)}")
        
        return combined
    
    # ==================== MODEL TRAINING ====================
    
    def train_models(self, training_data_path: str = TRAINING_DATA_CSV):
        """Train ML models"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING ML MODELS")
        logger.info("="*70)
        
        if not os.path.exists(training_data_path):
            logger.error(f"Training data not found: {training_data_path}")
            return False
        
        df = pd.read_csv(training_data_path)
        logger.info(f"Training samples: {len(df)}")
        
        if len(df) < 10:
            logger.error("Need at least 10 samples")
            return False
        
        X_list, y_scores, y_quality = [], [], []
        for _, row in df.iterrows():
            try:
                text = row.get('text','')
                if not text or len(text)<50:
                    continue
                feats = self.extract_features(text)
                X_list.append(self.prepare_feature_vector(feats)[0])
                y_scores.append(row['expert_score'])
                y_quality.append(row['quality_label'])
            except Exception as e:
                logger.warning(f"Row error: {e}")
                continue
        
        if len(X_list) < 10:
            logger.error("Insufficient valid samples")
            return False
        
        X = np.array(X_list)
        y_scores = np.array(y_scores)
        y_quality = self.label_encoder.fit_transform(y_quality)
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scores, test_size=0.2, random_state=42)
        
        logger.info("Training score predictor...")
        self.score_predictor = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        self.score_predictor.fit(X_train, y_train)
        
        y_pred = self.score_predictor.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"    R²: {r2:.4f} | MAE: {mae:.2f}")
        
        Xq_train, Xq_test, yq_train, yq_test = train_test_split(X_scaled, y_quality, test_size=0.2, random_state=42)
        
        logger.info("Training quality classifier...")
        self.quality_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.quality_classifier.fit(Xq_train, yq_train)
        
        logger.info("Training anomaly detector...")
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_detector.fit(X_scaled)
        
        self.model_metadata = {
            'trained': True,
            'training_samples': len(X),
            'validation_score': float(r2),
            'mae': float(mae),
            'last_trained': datetime.now().isoformat(),
            'feature_names': self.feature_names
        }
        
        self.save_models()
        logger.info("✓ Model training complete!\n")
        return True
    
    def auto_retrain_if_ready(self):
        """Automatically retrain if threshold is met"""
        if not os.path.exists(TRAINING_DATA_CSV):
            return False
        
        df = pd.read_csv(TRAINING_DATA_CSV)
        
        # Check if we have enough samples to train
        if len(df) >= 10:
            # If never trained, train now
            if not self.model_metadata['trained']:
                logger.info(f"\n✓ Training threshold met ({len(df)} samples). Auto-training models...")
                return self.train_models()
            
            # If trained, check for new samples since last training
            last = self.model_metadata.get('last_trained')
            if last:
                new_samples = df[df['date_added'] > last]
                if len(new_samples) >= 10:
                    logger.info(f"\n✓ {len(new_samples)} new samples detected. Auto-retraining models...")
                    return self.train_models()
        
        return False
    
    # ==================== MAIN ANALYSIS ====================
    
    def analyze_resume(self, resume_path: str, jobs_df: pd.DataFrame = None) -> Dict:
        """Analyze a single resume"""
        filename = Path(resume_path).name
        logger.info(f"\nAnalyzing: {filename}")
        
        text = self.extract_text(resume_path)
        if not text or len(text) < 50:
            logger.error("Failed to extract text")
            return None
        
        # Gemini-enhanced extraction
        gemini_data = {}
        if self.use_gemini:
            gemini_data = self.gemini_extract_structured_data(text)
        
        # Extract features
        features = self.extract_features(text, gemini_data)
        
        # Predict score
        scores = self.predict_score(features)
        logger.info(f"    Score: {scores['predicted_score']:.1f}/100 ({scores['grade']})")
        
        # Extract current skills
        current_skills = self.extract_current_skills(text, gemini_data)
        logger.info(f"    Skills Found: {len(current_skills)}")
        
        # Job matching
        matches = self.match_with_jobs(text, features, jobs_df) if jobs_df is not None else []
        
        # Gemini-enhanced structured recommendations
        if self.use_gemini:
            recommendations = self.gemini_generate_structured_recommendations(text, features, scores)
        else:
            recommendations = self._fallback_structured_recommendations(features, scores)
        
        return {
            'filename': filename,
            'text': text[:5000],
            'features': features,
            'scores': scores,
            'matches': matches,
            'recommendations': recommendations,
            'gemini_data': gemini_data,
            'current_skills': current_skills
        }


def main():
    print("\n" + "="*70)
    print("SELF-LEARNING RESUME ANALYZER WITH GEMINI 2.0 FLASH")
    print("="*70)
    print("Features:")
    print("  - CSV-based scoring configuration (scoring_criteria.csv)")
    print("  - Gemini 2.0 Flash API integration for enhanced extraction")
    print("  - Structured recommendations (skills, improvements, issues)")
    print("  - Current skills extraction from resume")
    print("  - AI-powered suggestions")
    print("  - Automatic ML training (no user prompts)")
    print("  - Semantic job matching")
    print("="*70 + "\n")
    
    # Get Gemini API key
    api_key = GEMINI_API_KEY
    if not api_key:
        print("Gemini API Key Setup:")
        print("  Option 1: Set environment variable GEMINI_API_KEY")
        print("  Option 2: Enter key below (or press Enter to skip)")
        api_key = input("\nEnter Gemini API key (or Enter to continue without): ").strip()
        if api_key:
            print("✓ Gemini enabled")
        else:
            print("→ Continuing without Gemini (using traditional methods)")
    
    analyzer = SelfLearningResumeAnalyzer(gemini_api_key=api_key if api_key else None)
    
    print(f"\nScoring criteria loaded from: {SCORING_CRITERIA_CSV}")
    print("  → Edit this CSV file to customize scoring weights and thresholds!")
    
    if os.path.exists(TRAINING_DATA_CSV):
        df = pd.read_csv(TRAINING_DATA_CSV)
        print(f"\nCurrent training data: {len(df)} samples")
        if len(df) >= 10:
            print(f"  ✓ Auto-training enabled ({len(df)} samples available)")
    else:
        print("\nNo training data yet. Will start collecting automatically!")
    
    print("\n" + "="*70)
    analyzer.process_all_resumes()
    
    print("\nAnalysis complete!")
    print(f"    Results: {OUTPUT_CSV}")
    print(f"    Training Data: {TRAINING_DATA_CSV}")
    print(f"    Scoring Config: {SCORING_CRITERIA_CSV}")
    print(f"    Logs: resume_analyzer.log")


if __name__ == "__main__":
    main()
