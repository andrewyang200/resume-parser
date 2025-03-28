# src/query_processor.py (Enhanced for Professional Networking Platform)
# ----- START OF COMPLETE UPDATED CODE BLOCK -----

import os
import json
import logging
import argparse
import time
from typing import Optional, Tuple

try:
    # Import the updated models from src/models.py
    from .models import StructuredQuery, QueryFilters, ProfessionalConceptExpansions, ConfidenceMetrics
except ImportError:
    from models import StructuredQuery, QueryFilters, ProfessionalConceptExpansions, ConfidenceMetrics

try:
    from dotenv import load_dotenv
    from openai import OpenAI, APIError, RateLimitError
    from pydantic import ValidationError
except ImportError as e:
    print(f"Error: Required library not found: {e.name}. Run 'pip install -r requirements.txt'")
    exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Constants ---
# USE THE BEST AVAILABLE MODEL FOR THIS TASK
DEFAULT_QUERY_MODEL = "gpt-4o"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# --- ENHANCED SCHEMA DEFINITION ---
QUERY_SCHEMA_DEFINITION = """
{
  "original_query": "string (The original user query, verbatim)",
  "semantic_query": "string | null (Core conceptual meaning, extracted or slightly refined)",
  "refined_semantic_query": "string | null (Optional: A more concrete rephrasing likely to match resume text)",
  "filters": {
    "location": ["string"] | null,
    "company_name": ["string"] | null,
    "job_title": ["string"] | null,
    "industry": ["string"] | null,
    "skills": ["string"] | null,
    "role_seniority": ["string"] | null (e.g., 'senior', 'VP', 'junior'),
    "min_experience_years": integer | null,
    "max_experience_years": integer | null,
    "keywords": ["string"] | null,
    "currently_working_at": "string | null",
    "previously_worked_at": "string | null",
    "founded_company": boolean | null,
    "open_to_consulting": boolean | null,
    "network_relation": "string | null",
    "career_stage": ["string"] | null,
    "project_types": ["string"] | null,
    "educational_background": ["string"] | null,
    "interests": ["string"] | null,
    "availability": "string | null"
  },
  "expanded_keywords": ["string"] | null,
  "professional_concept_expansions": {
    "skills": ["string"] | null,
    "roles": ["string"] | null,
    "industries": ["string"] | null,
    "company_types": ["string"] | null,
    "technologies": ["string"] | null,
    "achievements": ["string"] | null,
    "certifications": ["string"] | null
  },
  "implicit_needs": ["string"] | null,
  "query_type": "string | null",
  "confidence": {
    "overall": float | null,
    "ambiguity_level": string | null,
    "ambiguity_reason": string | null
  }
}
"""
# --- End of Schema Update ---

# --- Helper Functions ---
def load_openai_key() -> Optional[str]:
    load_dotenv(); api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: logging.error("OpenAI API key not found in .env"); return None
    return api_key

def process_query(
    user_query: str,
    client: OpenAI,
    model: str = DEFAULT_QUERY_MODEL
) -> Optional[StructuredQuery]:
    """ Uses LLM for advanced query understanding, expansion, and refinement. """
    logging.info(f"Processing query using model {model}: '{user_query}'")
    start_time = time.time()

    # --- ENHANCED SYSTEM PROMPT ---
    system_prompt = f"""
        You are an expert professional profile search system for a cutting-edge networking platform (similar to LinkedIn + Perplexity). Your task is to accurately parse natural language queries about professional connections and convert them into structured JSON for precise profile matching.

        ### YOUR CORE MISSION
        Transform any professional networking query into a structured representation that will match the RIGHT profiles, even when skills, experiences, or needs are implicitly stated. ACCURACY IS CRITICAL - users will see "no results found" if your parsing misses important concepts or fails to expand properly.

        ### QUERY UNDERSTANDING WORKFLOW

        1. **CORE UNDERSTANDING**
           - Preserve the original query verbatim
           - Extract the primary search intent
           - If the query is vague, create a concrete refinement using resume-like language

        2. **EXTRACT EXPLICIT ATTRIBUTES**
           - Identify all concrete filters (company names, titles, years of experience, etc.)
           - Map qualitative terms to quantitative when possible (e.g., "senior" -> 5+ years)
           - Detect temporal qualifiers ("currently", "previously", "used to")

        3. **PROFESSIONAL CONTEXT ENRICHMENT (MOST IMPORTANT)**
           - For EVERY key concept, generate comprehensive professional expansions:
             * If user mentions "startup experience" → include "entrepreneurship", "early-stage company", "founder", "co-founder", "seed funding", "product-market fit", "MVP development"
             * If user mentions "healthcare technology" → include "health tech", "medical devices", "patient data", "HIPAA compliance", "electronic health records", "telehealth"
             * If user mentions "growth marketing" → include "user acquisition", "retention strategies", "conversion optimization", "CAC", "LTV", "viral loops", "referral programs"

           - Expand abbreviations and industry jargon:
             * "PM" → "Product Manager", "Project Manager", "Program Manager"
             * "ML" → "Machine Learning", "Deep Learning", "AI", "Neural Networks", "Data Science"
             * "UI/UX" → "User Interface", "User Experience", "Design Thinking", "Wireframing", "Prototyping", "Usability Testing"

           - Generate synonyms and variations for roles and skills:
             * "Software Engineer" → "Developer", "Programmer", "SWE", "Coder", "Software Developer"
             * "Marketing" → "Growth", "Digital Marketing", "Content Strategy", "Brand Development"

           - Map abstract concepts to concrete implementations:
             * "Analytics expertise" → "SQL", "Tableau", "Power BI", "Data Visualization", "Looker", "Data Analysis"
             * "Cloud experience" → "AWS", "Azure", "GCP", "Cloud Architecture", "Serverless", "Kubernetes"

        4. **IMPLICIT NEED DETECTION**
           - Identify unstated but implied requirements
           - Infer career stages relevant to the query
           - Detect mentorship/guidance needs

        5. **AMBIGUITY HANDLING**
           - Assess confidence in your interpretation
           - Identify potential ambiguous terms (like "PM" or "design")
           - When ambiguous, expand ALL possible interpretations

        ### PROFESSIONAL CATEGORY EXPANSIONS (CRITICAL)

        **ALWAYS expand categories into specific examples:**

        - **Tech Companies:** FAANG → "Facebook/Meta", "Apple", "Amazon", "Netflix", "Google"; Include others like "Microsoft", "Twitter/X", "LinkedIn", "Salesforce", "Adobe", "Oracle"
          
        - **Industries:** Fintech → "Digital Banking", "Payment Processing", "Lending Platforms", "Wealth Management", "Blockchain", "Cryptocurrency", "InsurTech"

        - **Technical Domains:** AI/ML → "Natural Language Processing", "Computer Vision", "Predictive Analytics", "Recommendation Systems", "Reinforcement Learning", "LLMs", "Transformers"

        - **Job Functions:** Marketing → "Content Marketing", "SEO", "Social Media", "Email Marketing", "Brand Strategy", "Performance Marketing", "Marketing Analytics"

        - **Funding Stages:** Early Stage → "Pre-seed", "Seed", "Series A"; Late Stage → "Series B", "Series C", "Series D", "Pre-IPO"

        **WHEN IN DOUBT, EXPAND MORE BROADLY** - It's better to capture more potentially relevant terms than to miss important matches.

        ### OUTPUT REQUIREMENTS
        - Return a complete, valid JSON object matching the schema
        - Use snake_case for all keys
        - Fill all relevant fields, using null for truly inapplicable ones
        - DON'T INCLUDE any COMMENTARY or EXPLANATION outside the JSON

        **Target JSON Schema:**
        ```json
        {QUERY_SCHEMA_DEFINITION}
        ```
    """
    
    user_prompt = f"""
        Analyze the following user query based strictly on the detailed instructions and schema in the system prompt. Perform decomposition, refinement, and expansion as requested. Ensure all keys are snake_case.

        User Query: "{user_query}"
    """

    # --- LLM Call and Validation Logic ---
    raw_json_output = None
    for attempt in range(MAX_RETRIES):
        logging.info(f"LLM API call attempt {attempt + 1}/{MAX_RETRIES}")
        try:
            api_start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0, response_format={"type": "json_object"}
            )
            api_end_time = time.time()
            logging.info(f"LLM API call successful (attempt {attempt+1}). API time: {api_end_time - api_start_time:.2f}s")
            raw_json_output = response.choices[0].message.content
            if not raw_json_output:
                logging.error(f"LLM empty response (attempt {attempt+1}).")
                if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY_SECONDS); continue
                else: return None
            try:
                validated_query = StructuredQuery.model_validate(json.loads(raw_json_output))
                processing_time = time.time() - start_time
                logging.info(f"Query processing and validation successful in {processing_time:.2f}s.")
                return validated_query
            except (json.JSONDecodeError, ValidationError) as err:
                logging.error(f"LLM output validation/decode failed (attempt {attempt+1}): {err}")
                logging.error(f"LLM Raw Output Snippet: {raw_json_output[:500]}...")
                if isinstance(err, ValidationError): return None # Don't retry Pydantic errors
                if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY_SECONDS); continue
                else: return None
        except RateLimitError as e:
            logging.warning(f"Rate limit exceeded (attempt {attempt+1}). Retrying...")
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
            else: logging.error("Max retries reached for rate limiting."); return None
        except APIError as e:
            logging.error(f"OpenAI API error (attempt {attempt+1}): {e}")
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY_SECONDS)
            else: logging.error("Non-retryable API error or max retries."); return None
        except Exception as e:
            logging.error(f"Unexpected error during query processing (attempt {attempt+1}): {e}")
            logging.exception("Detailed traceback:")
            return None
    logging.error("Failed to process query after all retries.")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a natural language query into a structured format using an LLM.")
    parser.add_argument("query", help="The natural language query string to process.")
    parser.add_argument(
        "--model", 
        default=DEFAULT_QUERY_MODEL, 
        help=f"OpenAI model to use (default: {DEFAULT_QUERY_MODEL})."
    )
    args = parser.parse_args()

    # Load API Key
    api_key = load_openai_key()
    if not api_key:
        exit(1)

    # Initialize OpenAI Client
    try:
        client = OpenAI(api_key=api_key, timeout=60.0) # Adjust timeout if needed
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        exit(1)

    # Process the query
    structured_query_result = process_query(args.query, client, model=args.model)

    # Print the result
    if structured_query_result:
        print("\n--- Structured Query Result ---")
        # Use model_dump_json for clean Pydantic output
        print(structured_query_result.model_dump_json(indent=2))
        print("-----------------------------\n")
    else:
        print("\n--- Query processing failed. ---")
        exit(1)

# ----- END OF COMPLETE UPDATED CODE BLOCK -----