# src/database.py (FINAL COMPLETE - Step 1 + Step 2 + Step 4 FTS)
# ----- START OF COMPLETE FINAL CODE BLOCK -----

import psycopg2
import psycopg2.extras # For dictionary cursor
import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Iterable

# Import Pydantic model if used (adjust path if needed)
try:
    from .models import ParsedResume
except ImportError:
    try:
        from models import ParsedResume
    except ImportError:
        # If ParsedResume isn't strictly needed in this file after all,
        # you could remove the import and type hint below.
        # For now, assume it might be used indirectly or kept for consistency.
        ParsedResume = Any # Define as Any if not found, to avoid crashing import
        logging.warning("ParsedResume model not found. Type hints may be incorrect.")


# Import pgvector types if installed and numpy
try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None
    logging.warning("pgvector Python client not found. Install with 'pip install pgvector'. Vector operations might require manual casting.")

try:
    import numpy as np
except ImportError:
    np = None
    # Raise error here as numpy is critical for the current vector insertion logic
    logging.error("numpy not found. Install with 'pip install numpy'. Required for vector operations.")
    raise ImportError("numpy not found, which is required for vector operations.")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Database Connection ---
def get_db_connection():
    """Establishes connection and registers pgvector."""
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        logging.info("Database connection established successfully.")
        if register_vector:
            register_vector(conn)
            logging.info("Registered pgvector type adapter.")
        # Check for numpy again, as it's essential
        if np is None:
             logging.error("numpy was not imported successfully. Vector operations will likely fail.")
             # Optionally raise an error here to prevent proceeding without numpy
             # raise RuntimeError("numpy is required but not available.")
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"Database connection failed: {e}")
        if conn: conn.close()
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during database connection: {e}")
        if conn: conn.close()
        raise

# --- Table Creation for Step 1 ---
def create_tables(conn):
    """Creates the necessary BASE tables (users, experiences, etc.) if they don't exist."""
    commands = (
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255) UNIQUE,
            phone VARCHAR(50),
            location TEXT,
            linkedin_url TEXT,
            portfolio_url TEXT,
            github_url TEXT,
            other_url TEXT,
            summary TEXT,
            skills JSONB,
            raw_resume_text TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            source_filename VARCHAR(512),
            llm_metadata JSONB
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS work_experiences (
            experience_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            company VARCHAR(255),
            title VARCHAR(255),
            location TEXT,
            start_date VARCHAR(50),
            end_date VARCHAR(50),
            duration VARCHAR(100),
            description TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS education (
            education_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            institution VARCHAR(255),
            degree VARCHAR(255),
            field_of_study VARCHAR(255),
            location TEXT,
            start_date VARCHAR(50),
            end_date VARCHAR(50),
            graduation_date VARCHAR(50),
            description TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            name VARCHAR(255),
            description TEXT,
            technologies_used TEXT[],
            url TEXT,
            associated_experience VARCHAR(255)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS awards_recognition (
            award_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            title VARCHAR(255),
            issuer VARCHAR(255),
            award_date VARCHAR(50),
            description TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS publications (
            publication_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            title TEXT,
            authors TEXT[],
            journal_or_conference TEXT,
            publication_date VARCHAR(50),
            url_or_doi TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS volunteer_experiences (
            volunteer_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            organization VARCHAR(255),
            role VARCHAR(255),
            start_date VARCHAR(50),
            end_date VARCHAR(50),
            description TEXT
        );
        """,
        """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
           NEW.updated_at = NOW();
           RETURN NEW;
        END;
        $$ language 'plpgsql';
        """,
        """
        DO $$ BEGIN
          IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_users_updated_at') THEN
            CREATE TRIGGER update_users_updated_at
            BEFORE UPDATE ON users
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
          END IF;
        END $$;
        """
    )
    cur = None
    try:
        cur = conn.cursor()
        logging.info("Checking/Creating base data tables (users, experiences, etc.)...")
        for command in commands:
            cur.execute(command)
        conn.commit()
        logging.info("Base data tables checked/created successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error creating base tables: {error}")
        if conn: conn.rollback()
        raise
    finally:
        if cur: cur.close()


# --- Data Insertion for Step 1 ---
def insert_resume_data(conn, parsed_data: ParsedResume, raw_text: str, source_filename: str) -> Optional[int]:
    """Inserts validated and parsed resume data into the BASE tables using a transaction."""
    user_id = None
    cur = None

    # Prepare data using model_dump for Pydantic v2 if available
    contact = {}
    skills_json = '{}'
    metadata_json = '{}'
    if hasattr(parsed_data, 'contact_information') and parsed_data.contact_information:
         # Use .model_dump() for Pydantic v2, .dict() for v1
         if hasattr(parsed_data.contact_information, 'model_dump'):
             contact = parsed_data.contact_information.model_dump()
         elif hasattr(parsed_data.contact_information, 'dict'):
              contact = parsed_data.contact_information.dict()
    if hasattr(parsed_data, 'skills') and parsed_data.skills:
         if hasattr(parsed_data.skills, 'model_dump_json'):
             skills_json = parsed_data.skills.model_dump_json()
         elif hasattr(parsed_data.skills, 'json'):
              skills_json = parsed_data.skills.json()
    if hasattr(parsed_data, 'parsing_metadata') and parsed_data.parsing_metadata:
         if hasattr(parsed_data.parsing_metadata, 'model_dump_json'):
              metadata_json = parsed_data.parsing_metadata.model_dump_json()
         elif hasattr(parsed_data.parsing_metadata, 'json'):
              metadata_json = parsed_data.parsing_metadata.json()


    try:
        cur = conn.cursor()
        cur.execute("BEGIN;") # Start transaction

        # 1. Insert into users table
        user_sql = """
            INSERT INTO users (
                name, email, phone, location, linkedin_url, portfolio_url, github_url, other_url,
                summary, skills, raw_resume_text, source_filename, llm_metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
            RETURNING user_id;
        """
        cur.execute(user_sql, (
            contact.get('name'), contact.get('email'), contact.get('phone'), contact.get('location'),
            contact.get('linkedin_url'), contact.get('portfolio_url'), contact.get('github_url'), contact.get('other_url'),
            parsed_data.summary if hasattr(parsed_data, 'summary') else None,
            skills_json,
            raw_text,
            source_filename,
            metadata_json
        ))
        user_id = cur.fetchone()[0]
        logging.info(f"Inserted user with ID: {user_id}")

        # 2. Insert Work Experiences
        if hasattr(parsed_data, 'work_experience') and parsed_data.work_experience:
            exp_sql = """
                INSERT INTO work_experiences (
                    user_id, company, title, location, start_date, end_date, duration, description
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """
            exp_values = [
                (user_id, exp.company, exp.title, exp.location, exp.start_date, exp.end_date, exp.duration, exp.description)
                for exp in parsed_data.work_experience
            ]
            if exp_values:
                psycopg2.extras.execute_batch(cur, exp_sql, exp_values)
                logging.info(f"Inserted {len(exp_values)} work experience(s).")

        # 3. Insert Education
        if hasattr(parsed_data, 'education') and parsed_data.education:
            edu_sql = """
                INSERT INTO education (
                    user_id, institution, degree, field_of_study, location, start_date, end_date, graduation_date, description
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            edu_values = [
                (user_id, edu.institution, edu.degree, edu.field_of_study, edu.location, edu.start_date, edu.end_date, edu.graduation_date, edu.description)
                for edu in parsed_data.education
            ]
            if edu_values:
                psycopg2.extras.execute_batch(cur, edu_sql, edu_values)
                logging.info(f"Inserted {len(edu_values)} education record(s).")

        # 4. Insert Projects
        if hasattr(parsed_data, 'projects') and parsed_data.projects:
            proj_sql = """
                INSERT INTO projects (
                    user_id, name, description, technologies_used, url, associated_experience
                ) VALUES (%s, %s, %s, %s, %s, %s);
            """
            proj_values = [
                (user_id, proj.name, proj.description, proj.technologies_used, proj.url, proj.associated_experience)
                for proj in parsed_data.projects
            ]
            if proj_values:
                psycopg2.extras.execute_batch(cur, proj_sql, proj_values)
                logging.info(f"Inserted {len(proj_values)} project(s).")

        # 5. Insert Awards
        if hasattr(parsed_data, 'awards_and_recognition') and parsed_data.awards_and_recognition:
            award_sql = """
                INSERT INTO awards_recognition (
                    user_id, title, issuer, award_date, description
                ) VALUES (%s, %s, %s, %s, %s);
            """
            award_values = [
                (user_id, award.title, award.issuer, award.date, award.description)
                for award in parsed_data.awards_and_recognition
            ]
            if award_values:
                psycopg2.extras.execute_batch(cur, award_sql, award_values)
                logging.info(f"Inserted {len(award_values)} award(s)/recognition(s).")

        # 6. Insert Publications
        if hasattr(parsed_data, 'publications') and parsed_data.publications:
            pub_sql = """
                INSERT INTO publications (
                    user_id, title, authors, journal_or_conference, publication_date, url_or_doi
                ) VALUES (%s, %s, %s, %s, %s, %s);
            """
            pub_values = [
                (user_id, pub.title, pub.authors, pub.journal_or_conference, pub.date, pub.url_or_doi)
                for pub in parsed_data.publications
            ]
            if pub_values:
                psycopg2.extras.execute_batch(cur, pub_sql, pub_values)
                logging.info(f"Inserted {len(pub_values)} publication(s).")

        # 7. Insert Volunteer Experiences
        if hasattr(parsed_data, 'volunteer_experience') and parsed_data.volunteer_experience:
            vol_sql = """
                INSERT INTO volunteer_experiences (
                    user_id, organization, role, start_date, end_date, description
                ) VALUES (%s, %s, %s, %s, %s, %s);
            """
            vol_values = [
                (user_id, vol.organization, vol.role, vol.start_date, vol.end_date, vol.description)
                for vol in parsed_data.volunteer_experience
            ]
            if vol_values:
                psycopg2.extras.execute_batch(cur, vol_sql, vol_values)
                logging.info(f"Inserted {len(vol_values)} volunteer experience(s).")

        conn.commit() # Commit transaction
        logging.info(f"Successfully committed all data for user_id: {user_id}")
        return user_id

    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Database error during insertion for {source_filename}: {error}")
        if conn and not conn.closed:
            try: conn.rollback()
            except psycopg2.Error as rb_err: logging.error(f"Error during rollback: {rb_err}")
        logging.error("Transaction rolled back.")
        return None # Indicate failure
    finally:
        if cur: cur.close()


# --- Indexing & FTS Setup Functions (Step 2 & 4) ---
# Place this corrected function inside src/database.py, replacing the previous version

def setup_fts_columns_and_triggers(conn):
    """Adds tsvector columns and triggers for automatic updates. CORRECTED VERSION."""
    fts_config = [
        # Ensure text_cols match actual column names in your tables
        {'table': 'users', 'text_cols': ['name', 'location', 'summary', 'raw_resume_text'], 'vector_col': 'fts_doc'},
        {'table': 'work_experiences', 'text_cols': ['company', 'title', 'location', 'description'], 'vector_col': 'fts_doc'},
        {'table': 'education', 'text_cols': ['institution', 'degree', 'field_of_study', 'location', 'description'], 'vector_col': 'fts_doc'},
        {'table': 'projects', 'text_cols': ['name', 'description'], 'vector_col': 'fts_doc'},
    ]
    fts_language = 'english' # Ensure this matches your data's primary language

    with conn.cursor() as cur:
        logging.info("Setting up FTS tsvector columns and triggers...")
        for config in fts_config:
            table = config['table']
            vector_col = config['vector_col']
            text_cols = config['text_cols'] # Columns specific to this table

            # 1. Add tsvector column (Keep the correct DO $$ block from previous fix)
            logging.debug(f"Checking/Adding tsvector column '{vector_col}' to table '{table}'")
            cur.execute(f"""
                DO $$ BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = '{table}' AND column_name = '{vector_col}')
                    THEN ALTER TABLE {table} ADD COLUMN {vector_col} tsvector; RAISE NOTICE 'Column {vector_col} added to {table}.';
                    ELSE RAISE NOTICE 'Column {vector_col} already exists in {table}.'; END IF;
                END $$;
            """)

            # --- CORRECTED LOGIC: Build weighted_cols specific to *this* table's text_cols ---
            weighted_cols_trigger = []
            # Define weights (Adjust as needed)
            if 'name' in text_cols: weighted_cols_trigger.append(f"setweight(to_tsvector('{fts_language}', COALESCE(NEW.name,'')), 'A')")
            if 'title' in text_cols: weighted_cols_trigger.append(f"setweight(to_tsvector('{fts_language}', COALESCE(NEW.title,'')), 'A')")
            if 'institution' in text_cols: weighted_cols_trigger.append(f"setweight(to_tsvector('{fts_language}', COALESCE(NEW.institution,'')), 'A')")

            if 'company' in text_cols: weighted_cols_trigger.append(f"setweight(to_tsvector('{fts_language}', COALESCE(NEW.company,'')), 'B')")
            if 'degree' in text_cols: weighted_cols_trigger.append(f"setweight(to_tsvector('{fts_language}', COALESCE(NEW.degree,'')), 'B')")
            if 'field_of_study' in text_cols: weighted_cols_trigger.append(f"setweight(to_tsvector('{fts_language}', COALESCE(NEW.field_of_study,'')), 'B')")

            if 'summary' in text_cols: weighted_cols_trigger.append(f"setweight(to_tsvector('{fts_language}', COALESCE(NEW.summary,'')), 'C')")
            if 'description' in text_cols: weighted_cols_trigger.append(f"setweight(to_tsvector('{fts_language}', COALESCE(NEW.description,'')), 'C')")

            if 'location' in text_cols: weighted_cols_trigger.append(f"setweight(to_tsvector('{fts_language}', COALESCE(NEW.location,'')), 'D')")
            if 'raw_resume_text' in text_cols: weighted_cols_trigger.append(f"setweight(to_tsvector('{fts_language}', COALESCE(NEW.raw_resume_text,'')), 'D')")
            # Add checks for other columns if you expand fts_config

            if not weighted_cols_trigger:
                logging.warning(f"No text columns configured in fts_config matched actual columns for trigger on table {table}. Skipping trigger creation.")
                continue # Skip trigger creation if no relevant columns found for this table
            tsvector_expression_trigger = " || ' ' || ".join(weighted_cols_trigger)
            # --- End Corrected Logic ---

            # 2. Create/Replace trigger function using the correctly built expression
            trigger_func_name = f"{table}_fts_update"
            logging.debug(f"Creating/Replacing trigger function '{trigger_func_name}' for '{table}'")
            cur.execute(f"""
            CREATE OR REPLACE FUNCTION {trigger_func_name}() RETURNS trigger AS $$
            BEGIN NEW.{vector_col} := {tsvector_expression_trigger}; RETURN NEW; END $$ LANGUAGE plpgsql;
            """)

            # 3. Create/Replace trigger
            trigger_name = f"trigger_{table}_fts_update"
            logging.debug(f"Creating/Replacing trigger '{trigger_name}' on table '{table}'")
            cur.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON {table};")
            cur.execute(f""" CREATE TRIGGER {trigger_name} BEFORE INSERT OR UPDATE ON {table} FOR EACH ROW EXECUTE FUNCTION {trigger_func_name}(); """)

        conn.commit() # Commit column add & trigger creation/replacement
        logging.info("FTS tsvector columns and triggers checked/created/replaced.")

        # 4. Populate tsvector columns for existing data (Also needs corrected expression)
        logging.info("Populating/Re-populating tsvector columns for existing data...")
        for config in fts_config:
            table = config['table']; vector_col = config['vector_col']; text_cols = config['text_cols']

            # --- CORRECTED LOGIC: Rebuild weighted_cols specific to *this* table for UPDATE ---
            update_weighted_cols = []
            # Reference columns directly (without NEW.)
            if 'name' in text_cols: update_weighted_cols.append(f"setweight(to_tsvector('{fts_language}', COALESCE(name,'')), 'A')")
            if 'title' in text_cols: update_weighted_cols.append(f"setweight(to_tsvector('{fts_language}', COALESCE(title,'')), 'A')")
            if 'institution' in text_cols: update_weighted_cols.append(f"setweight(to_tsvector('{fts_language}', COALESCE(institution,'')), 'A')")
            if 'company' in text_cols: update_weighted_cols.append(f"setweight(to_tsvector('{fts_language}', COALESCE(company,'')), 'B')")
            if 'degree' in text_cols: update_weighted_cols.append(f"setweight(to_tsvector('{fts_language}', COALESCE(degree,'')), 'B')")
            if 'field_of_study' in text_cols: update_weighted_cols.append(f"setweight(to_tsvector('{fts_language}', COALESCE(field_of_study,'')), 'B')")
            if 'summary' in text_cols: update_weighted_cols.append(f"setweight(to_tsvector('{fts_language}', COALESCE(summary,'')), 'C')")
            if 'description' in text_cols: update_weighted_cols.append(f"setweight(to_tsvector('{fts_language}', COALESCE(description,'')), 'C')")
            if 'location' in text_cols: update_weighted_cols.append(f"setweight(to_tsvector('{fts_language}', COALESCE(location,'')), 'D')")
            if 'raw_resume_text' in text_cols: update_weighted_cols.append(f"setweight(to_tsvector('{fts_language}', COALESCE(raw_resume_text,'')), 'D')")

            if not update_weighted_cols: continue # Skip if no columns configured for this table
            tsvector_update_expr = " || ' ' || ".join(update_weighted_cols)
            # --- End Corrected Logic ---

            logging.debug(f"Updating existing rows for {table}.{vector_col}...")
            try:
                # Update ALL rows this time to fix previous potentially empty/incorrect values
                cur.execute(f"UPDATE {table} SET {vector_col} = {tsvector_update_expr};")
                conn.commit() # Commit after each table update
                logging.info(f"Updated FTS vector for {cur.rowcount} existing rows in {table}.")
            except Exception as update_err:
                logging.error(f"Error updating FTS data for {table}: {update_err}"); conn.rollback()
        logging.info("Finished populating tsvector columns.")


def setup_indexing(conn, vector_dimension: int, rebuild_fts: bool = False):
    """Creates vector/metadata/FTS tables and indexes."""
    # --- Part 1: Ensure Extension and Create Chunk Table ---
    with conn.cursor() as cur:
        logging.info("Checking for pgvector extension...")
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except psycopg2.Error as ext_err:
            logging.warning(f"Vector extension check failed: {ext_err}"); conn.rollback()
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
            if not cur.fetchone(): raise RuntimeError("pgvector extension missing") from ext_err
        logging.info("pgvector extension checked/ensured.")

        logging.info(f"Creating/Checking text_chunks table (dim: {vector_dimension})...")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS text_chunks (
            chunk_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            source_table VARCHAR(100) NOT NULL,
            source_column VARCHAR(100) NOT NULL,
            source_pk INTEGER,
            chunk_text TEXT NOT NULL,
            embedding VECTOR({vector_dimension}),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        logging.info("text_chunks table checked/created.")
        conn.commit()
        logging.info("Committed extension check and table creation.")

    # --- Part 2: Create Vector Index ---
    hnsw_index_name = "idx_text_chunks_embedding_hnsw"
    logging.info(f"Checking/Creating HNSW index '{hnsw_index_name}' CONCURRENTLY...")
    temp_conn = None
    try:
        index_exists = False
        with conn.cursor() as cur:
             cur.execute(f"SELECT 1 FROM pg_class WHERE relname = '{hnsw_index_name}' AND relkind = 'i';")
             if cur.fetchone(): index_exists = True; logging.info(f"HNSW index '{hnsw_index_name}' exists.")
        if not index_exists:
            logging.info(f"Attempting HNSW CONCURRENTLY creation...")
            temp_conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
            )
            temp_conn.autocommit = True
            with temp_conn.cursor() as temp_cur:
                try:
                    # Use L2 distance for HNSW with models like all-mpnet-base-v2
                    temp_cur.execute(f""" CREATE INDEX CONCURRENTLY {hnsw_index_name} ON text_chunks USING hnsw (embedding vector_l2_ops); """)
                    logging.info(f"HNSW index '{hnsw_index_name}' creation initiated.")
                except psycopg2.Error as conc_err:
                    if "already exists" in str(conc_err): logging.info(f"HNSW index '{hnsw_index_name}' exists (concurrent check).")
                    else: logging.warning(f"CONCURRENTLY index creation failed: {conc_err}.")
                    # Consider adding non-concurrent fallback here if needed
    except Exception as e: logging.error(f"Error during HNSW index phase: {e}")
    finally:
        if temp_conn: temp_conn.close(); logging.debug("Temp HNSW conn closed.")

    # --- Part 3: Setup FTS Columns and Triggers ---
    if rebuild_fts:
        try:
            setup_fts_columns_and_triggers(conn)
        except Exception as fts_setup_err:
            logging.error(f"Failed during FTS column/trigger setup: {fts_setup_err}")
            raise # Propagate FTS setup errors
    else:
        logging.info("Skipping FTS column/trigger setup (use --rebuild_fts to force).")

    # --- Part 4: Create Metadata & FTS Indexes ---
    logging.info("Checking/Creating metadata and FTS indexes...")
    indexes_to_create = {
        # Metadata Indexes
        "idx_users_email": "CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);",
        "idx_users_location": "CREATE INDEX IF NOT EXISTS idx_users_location ON users (location);",
        "idx_users_skills": "CREATE INDEX IF NOT EXISTS idx_users_skills ON users USING gin(skills);",
        "idx_work_experiences_user_id": "CREATE INDEX IF NOT EXISTS idx_work_experiences_user_id ON work_experiences (user_id);",
        "idx_work_experiences_company": "CREATE INDEX IF NOT EXISTS idx_work_experiences_company ON work_experiences (company);",
        "idx_work_experiences_title": "CREATE INDEX IF NOT EXISTS idx_work_experiences_title ON work_experiences (title);",
        "idx_education_user_id": "CREATE INDEX IF NOT EXISTS idx_education_user_id ON education (user_id);",
        "idx_education_institution": "CREATE INDEX IF NOT EXISTS idx_education_institution ON education (institution);",
        "idx_education_degree": "CREATE INDEX IF NOT EXISTS idx_education_degree ON education (degree);",
        "idx_projects_user_id": "CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects (user_id);",
        "idx_projects_name": "CREATE INDEX IF NOT EXISTS idx_projects_name ON projects (name);",
        "idx_projects_technologies": "CREATE INDEX IF NOT EXISTS idx_projects_technologies ON projects USING gin(technologies_used);",
        # FTS Indexes (on tsvector columns created by setup_fts_columns_and_triggers)
        "idx_users_fts_doc": "CREATE INDEX IF NOT EXISTS idx_users_fts_doc ON users USING gin(fts_doc);",
        "idx_work_experiences_fts_doc": "CREATE INDEX IF NOT EXISTS idx_work_experiences_fts_doc ON work_experiences USING gin(fts_doc);",
        "idx_education_fts_doc": "CREATE INDEX IF NOT EXISTS idx_education_fts_doc ON education USING gin(fts_doc);",
        "idx_projects_fts_doc": "CREATE INDEX IF NOT EXISTS idx_projects_fts_doc ON projects USING gin(fts_doc);",
    }
    try:
        if conn.autocommit: conn.autocommit = False
        with conn.cursor() as cur:
            for index_name, index_sql in indexes_to_create.items():
                logging.debug(f"Executing: {index_sql}")
                cur.execute(index_sql)
        conn.commit()
        logging.info("Metadata and FTS indexes checked/created.")
    except psycopg2.Error as meta_err:
        logging.error(f"Error during metadata/FTS index creation: {meta_err}")
        conn.rollback()
        raise meta_err
    logging.info("Indexing setup completed.")


# --- Vector Indexing Helper Functions ---
def get_text_data_to_index(conn) -> List[Dict[str, Any]]:
    """Fetches relevant text data from various tables that needs embedding."""
    data_to_index = []
    queries = {
        "user_summary": "SELECT user_id, summary as text, 'users' as source_table, 'summary' as source_column, NULL as source_pk FROM users WHERE summary IS NOT NULL AND summary != ''",
        "work_desc": "SELECT experience_id as source_pk, user_id, description as text, 'work_experiences' as source_table, 'description' as source_column FROM work_experiences WHERE description IS NOT NULL AND description != ''",
        "project_desc": "SELECT project_id as source_pk, user_id, description as text, 'projects' as source_table, 'description' as source_column FROM projects WHERE description IS NOT NULL AND description != ''",
        "education_desc": "SELECT education_id as source_pk, user_id, description as text, 'education' as source_table, 'description' as source_column FROM education WHERE description IS NOT NULL AND description != ''"
    }
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
             for query_name, sql in queries.items():
                 logging.debug(f"Fetching data for indexing: {query_name}")
                 cur.execute(sql); rows = cur.fetchall()
                 if rows: data_to_index.extend([dict(row) for row in rows])
                 logging.debug(f"Fetched {len(rows)} items from {query_name}")
        logging.info(f"Fetched {len(data_to_index)} total text items to potentially index.")
        return data_to_index
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error fetching data to index: {error}"); raise
    # No finally block needed for cursor due to 'with' statement

def clear_existing_chunks(conn, user_id: Optional[int] = None):
    """Deletes existing chunks, optionally for a specific user."""
    try:
        with conn.cursor() as cur:
             if user_id:
                 logging.warning(f"Deleting chunks for user_id: {user_id}")
                 cur.execute("DELETE FROM text_chunks WHERE user_id = %s;", (user_id,));
                 logging.info(f"Deleted {cur.rowcount} chunks for user {user_id}.")
             else:
                 logging.warning("Deleting ALL chunks using TRUNCATE.")
                 cur.execute("TRUNCATE TABLE text_chunks RESTART IDENTITY;")
                 logging.info("Truncated text_chunks table.")
             conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error clearing chunks: {error}"); conn.rollback(); raise

def insert_text_chunks_batch(conn, chunks_data: List[Dict[str, Any]]):
    """Inserts a batch of text chunks with their embeddings."""
    if not chunks_data: return 0
    if np is None: raise ImportError("numpy not found, cannot process embeddings.")
    sql = """
        INSERT INTO text_chunks (user_id, source_table, source_column, source_pk, chunk_text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s);
    """
    batch_values = []
    conversion_errors = 0
    for chunk in chunks_data:
        try:
            embedding_input = chunk['embedding']
            if isinstance(embedding_input, (list, tuple)): embedding_np = np.array(embedding_input, dtype=np.float32)
            elif isinstance(embedding_input, np.ndarray): embedding_np = embedding_input.astype(np.float32, copy=False)
            else: raise TypeError(f"Unsupported embedding type: {type(embedding_input)}")
            batch_values.append((
                chunk['user_id'], chunk['source_table'], chunk['source_column'],
                chunk['source_pk'], chunk['chunk_text'], embedding_np
            ))
        except Exception as conversion_err:
            conversion_errors += 1; logging.error(f"Embedding conversion error: {conversion_err}")
    if conversion_errors > 0: logging.warning(f"Skipped {conversion_errors} chunks due to conversion errors.")
    if not batch_values: logging.warning("Batch empty after errors."); return 0
    try:
        with conn.cursor() as cur:
             psycopg2.extras.execute_batch(cur, sql, batch_values, page_size=100); conn.commit()
             logging.info(f"Inserted batch of {len(batch_values)} chunks.")
             return len(batch_values)
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error inserting chunk batch: {error}"); conn.rollback(); raise

# ----- END OF COMPLETE FINAL CODE BLOCK -----