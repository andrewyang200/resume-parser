# src/query_processor.py (Updated schema definition and system prompt)
# ----- START OF COMPLETE UPDATED CODE BLOCK -----

import os
import json
import logging
import argparse
import time
from typing import Optional, Tuple

try:
    # Import the updated models from src/models.py
    from .models import StructuredQuery, QueryFilters
except ImportError:
    from models import StructuredQuery, QueryFilters

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
DEFAULT_QUERY_MODEL = "gpt-4o"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# --- UPDATED SCHEMA DEFINITION (with role_seniority) ---
QUERY_SCHEMA_DEFINITION = """
{
  "semantic_query": "string | null (Core conceptual meaning for vector search. E.g., 'experience scaling SaaS startup', 'mobile UX design fintech', 'career advice finance to tech transition')",
  "filters": {
    "location": ["string"] | null (Cities, states, countries, 'local'),
    "company_name": ["string"] | null (Company names),
    "job_title": ["string"] | null (Job titles),
    "industry": ["string"] | null (e.g., 'fintech', 'healthcare AI', 'SaaS'),
    "skills": ["string"] | null (Specific skills mentioned),
    "role_seniority": ["string"] | null (Seniority levels like 'senior', 'VP', 'junior', 'lead'),
    "min_experience_years": integer | null,
    "max_experience_years": integer | null,
    "keywords": ["string"] | null (Use for concepts not in other filters: scaling factors, project types like 'enterprise', transition elements like 'finance'/'tech' when switching, specific technologies mentioned loosely, etc.),
    "currently_working_at": "string | null",
    "previously_worked_at": "string | null",
    "founded_company": boolean | null,
    "open_to_consulting": boolean | null,
    "network_relation": "string | null (e.g., 'my_network', 'alumni_network')"
  },
  "query_type": "string | null (Intent: 'seeking_expertise', 'exploring_paths', 'company_search', 'role_search', 'networking', 'collaboration', 'other')"
}
"""
# --- End of Schema Update ---

# --- Helper Functions --- (load_openai_key remains the same)
def load_openai_key() -> Optional[str]:
    load_dotenv(); api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: logging.error("OpenAI API key not found in .env"); return None
    return api_key

def process_query(
    user_query: str,
    client: OpenAI,
    model: str = DEFAULT_QUERY_MODEL
) -> Optional[StructuredQuery]:
    """ Processes query using LLM, validates against Pydantic models. """
    logging.info(f"Processing query using model {model}: '{user_query}'")
    start_time = time.time()

    # --- UPDATED SYSTEM PROMPT ---
    system_prompt = f"""
        You are an intelligent assistant analyzing user queries for a professional networking platform.
        Convert the user's query into a structured JSON object according to the schema below.

        **Instructions:**
        1.  **Semantic Core (`semantic_query`):** Extract the central theme/concept for vector search. Focus on the *what* and *why*.
        2.  **Filters (`filters`):** Extract specific constraints using the exact `snake_case` keys provided. Use `null` or omit keys if not present.
            *   **Use `keywords` flexibly:** Place concepts here if they don't fit specific filters. Examples: scaling factors ("scaled team 10 to 50"), project types ("enterprise projects"), technologies mentioned generally, named methodologies ("Agile").
            *   **Career Transitions:** If the query is about switching careers (e.g., "moving from X to Y"), extract both X and Y and put them into the `keywords` list.
            *   **Seniority:** Extract terms like "senior", "junior", "VP", "lead", "principal" into the `role_seniority` list.
            *   Interpret context: "used to work at Google" -> `previously_worked_at: "Google"`. "founder" -> `founded_company: true`.
        3.  **Query Type (`query_type`):** Classify the user's likely intent.
        4.  **Output Format:** Return ONLY the valid JSON object matching the schema. Use `snake_case` keys exactly. Adhere strictly to types.

        **Target JSON Schema:**
        ```json
        {QUERY_SCHEMA_DEFINITION}
    """
    
    user_prompt = f"""
        Analyze the following user query and generate the structured JSON output according to the instructions and schema provided in the system prompt. Ensure all keys in the output JSON use snake_case.

        User Query: "{user_query}"
    """

    # --- LLM Call and Validation Logic (Remains the same as previous version) ---
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