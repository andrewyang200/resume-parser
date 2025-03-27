import argparse
import logging
import json
import time
import datetime
from pathlib import Path
from typing import Optional

# Import local modules
import resume_parser
import database
from models import ParsedResume # Pydantic model for validation

try:
    from dotenv import load_dotenv
    from openai import OpenAI
    from pydantic import ValidationError
except ImportError as e:
    print(f"Error: Required library not found: {e.name}. Run 'pip install -r requirements.txt'")
    exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

def run_pipeline(file_path: Path, model: str = resume_parser.DEFAULT_LLM_MODEL):
    """Orchestrates the resume parsing, validation, and database storage."""
    logging.info(f"--- Starting Ingestion Pipeline for: {file_path.name} ---")
    overall_start_time = time.time()

    # --- 1. Load Config & Initialize Clients ---
    logging.info("Loading configuration and initializing clients...")
    load_dotenv()
    api_key = resume_parser.load_config()
    if not api_key:
        logging.error("OpenAI API key missing. Aborting.")
        return False

    try:
        openai_client = OpenAI(api_key=api_key, timeout=90.0)
        db_conn = database.get_db_connection() # Ensure DB connection details are in .env
        if not db_conn:
             raise ConnectionError("Failed to establish database connection.")
        database.create_tables(db_conn) # Ensure tables exist
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        return False

    validated_data: Optional[ParsedResume] = None
    raw_text: Optional[str] = None
    success = False

    try:
        # --- 2. Extract Text ---
        logging.info("Step 1: Extracting text from resume...")
        extract_start = time.time()
        raw_text = resume_parser.get_resume_text(file_path)
        extract_time = time.time() - extract_start
        if not raw_text:
            logging.error(f"Failed to extract text from {file_path.name}.")
            raise ValueError("Text extraction failed")
        logging.info(f"Text extraction completed in {extract_time:.2f}s. Length: {len(raw_text)} chars.")

        # --- 3. Parse with LLM ---
        logging.info("Step 2: Parsing text with LLM...")
        raw_json_str, llm_time = resume_parser.parse_resume_with_llm_raw(raw_text, openai_client, model)
        if not raw_json_str:
            logging.error("LLM parsing failed.")
            raise ValueError("LLM Parsing returned no data")
        logging.info(f"LLM parsing completed in {llm_time:.2f}s.")

        # --- 4. Validate and Structure Data ---
        logging.info("Step 3: Validating and structuring data...")
        validate_start = time.time()
        try:
            # Add metadata before validation
            llm_output_dict = json.loads(raw_json_str)

            # Ensure parsing_metadata exists and add details
            metadata = llm_output_dict.setdefault('parsing_metadata', {})
            metadata['model_used'] = model
            metadata['timestamp_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
            metadata['processing_time_seconds'] = round(llm_time, 2) # Approximate time

            # Add raw text preview
            llm_output_dict['raw_text_preview'] = raw_text[:500] + ('...' if len(raw_text) > 500 else '')

            # Validate using Pydantic
            validated_data = ParsedResume.model_validate(llm_output_dict) # Use model_validate for dict
            validate_time = time.time() - validate_start
            logging.info(f"Data validation successful in {validate_time:.2f}s.")

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from LLM: {e}")
            logging.error(f"LLM Raw Output Snippet: {raw_json_str[:500]}...")
            raise
        except ValidationError as e:
            logging.error(f"Pydantic validation failed: {e}")
            # Log details about validation errors
            # logging.error(e.errors()) # Provides detailed error breakdown
            raise
        
        
        # Add after validation
        logging.info("--- Data Content Diagnostics ---")
        if validated_data.contact_information:
            logging.info(f"Contact Name: {validated_data.contact_information.name}")
            logging.info(f"Contact Email: {validated_data.contact_information.email}")
        else:
            logging.info("No contact information found")

        logging.info(f"Work Experience Count: {len(validated_data.work_experience)}")
        if validated_data.work_experience:
            logging.info(f"First Work Title: {validated_data.work_experience[0].title}")



        # --- 5. Store in Database ---
        logging.info("Step 4: Storing data in database...")
        store_start = time.time()
        user_id = database.insert_resume_data(db_conn, validated_data, raw_text, file_path.name)
        store_time = time.time() - store_start
        if user_id:
            logging.info(f"Database storage successful for user_id {user_id} in {store_time:.2f}s.")
            success = True
        else:
            logging.error("Database storage failed.")
            # Error is logged within insert_resume_data

    except Exception as e:
        logging.error(f"Pipeline failed for {file_path.name}: {e}")
        logging.exception("Detailed traceback:") # Log full traceback for unexpected errors
        success = False # Ensure failure state
    finally:
        if db_conn:
            db_conn.close()
            logging.info("Database connection closed.")

    overall_time = time.time() - overall_start_time
    logging.info(f"--- Ingestion Pipeline {'Completed' if success else 'Failed'} for: {file_path.name} in {overall_time:.2f}s ---")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the resume ingestion pipeline: Parse -> Validate -> Store.")
    parser.add_argument("input_file", help="Path to the input resume file (PDF or DOCX).")
    parser.add_argument("--model", default=resume_parser.DEFAULT_LLM_MODEL, help=f"OpenAI model to use for parsing (default: {resume_parser.DEFAULT_LLM_MODEL}).")
    # Add option to process a directory later if needed
    # parser.add_argument("-d", "--directory", help="Path to a directory containing resume files to process.")

    args = parser.parse_args()

    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)

    # Run the pipeline for the single file
    run_pipeline(input_path, model=args.model)