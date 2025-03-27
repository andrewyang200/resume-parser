# src/database.py (Updated for Step 1 & Step 2)
# ----- START OF COMPLETE UPDATED CODE BLOCK -----

import psycopg2
import psycopg2.extras # For dictionary cursor
import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Iterable

# Import the Pydantic model used in insert_resume_data
# If models.py is in the same directory, this should work. Adjust if needed.
try:
    from .models import ParsedResume
except ImportError:
    # Fallback if running script directly or structure issues
    from models import ParsedResume

# NEW: Import pgvector types if installed and numpy
try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None # Define as None if pgvector client not installed
    logging.warning("pgvector Python client not found. Install with 'pip install pgvector'. Vector fetching might require manual casting.")

try:
    import numpy as np
except ImportError:
    np = None
    logging.warning("numpy not found. Install with 'pip install numpy'. Required for pgvector client embedding conversion.")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Database Connection (Updated for pgvector) ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database and registers pgvector type."""
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

        # NEW: Register the vector type adapter globally for this connection
        # This allows fetching/inserting vector columns using numpy arrays.
        if register_vector:
            register_vector(conn)
            logging.info("Registered pgvector type adapter for psycopg2.")
        elif np is None:
             logging.error("numpy is required for pgvector integration but not found. Please install numpy.")


        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"Database connection failed: {e}")
        if conn: conn.close() # Ensure connection is closed on failure
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during database connection: {e}")
        if conn: conn.close()
        raise


# --- Table Creation for Step 1 (Your Existing Code - Preserved) ---
def create_tables(conn):
    """Creates the necessary BASE tables (users, experiences, etc.) if they don't exist."""
    # --- This is your existing, working code for Step 1 tables ---
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
        if conn:
            conn.rollback() # Rollback changes on error
        raise
    finally:
        if cur:
            cur.close()


# --- Data Insertion for Step 1 (Your Existing Code - Preserved) ---
def insert_resume_data(conn, parsed_data: ParsedResume, raw_text: str, source_filename: str) -> Optional[int]:
    """Inserts validated and parsed resume data into the BASE tables using a transaction."""
    # --- This is your existing, working code for Step 1 insertion ---
    user_id = None
    cur = None

    # Prepare data, handling potential None values from Pydantic models
    contact = parsed_data.contact_information.model_dump() if parsed_data.contact_information else {}
    skills_json = parsed_data.skills.model_dump_json() if parsed_data.skills else '{}'
    metadata_json = parsed_data.parsing_metadata.model_dump_json() if parsed_data.parsing_metadata else '{}'

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
            parsed_data.summary,
            skills_json,
            raw_text,
            source_filename,
            metadata_json
        ))
        user_id = cur.fetchone()[0]
        logging.info(f"Inserted user with ID: {user_id}")

        # 2. Insert Work Experiences
        if parsed_data.work_experience:
            exp_sql = """
                INSERT INTO work_experiences (
                    user_id, company, title, location, start_date, end_date, duration, description
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """
            exp_values = [
                (user_id, exp.company, exp.title, exp.location, exp.start_date, exp.end_date, exp.duration, exp.description)
                for exp in parsed_data.work_experience
            ]
            psycopg2.extras.execute_batch(cur, exp_sql, exp_values)
            logging.info(f"Inserted {len(exp_values)} work experience(s).")

        # 3. Insert Education
        if parsed_data.education:
            edu_sql = """
                INSERT INTO education (
                    user_id, institution, degree, field_of_study, location, start_date, end_date, graduation_date, description
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            edu_values = [
                (user_id, edu.institution, edu.degree, edu.field_of_study, edu.location, edu.start_date, edu.end_date, edu.graduation_date, edu.description)
                for edu in parsed_data.education
            ]
            psycopg2.extras.execute_batch(cur, edu_sql, edu_values)
            logging.info(f"Inserted {len(edu_values)} education record(s).")

        # 4. Insert Projects
        if parsed_data.projects:
            proj_sql = """
                INSERT INTO projects (
                    user_id, name, description, technologies_used, url, associated_experience
                ) VALUES (%s, %s, %s, %s, %s, %s);
            """
            proj_values = [
                (user_id, proj.name, proj.description, proj.technologies_used, proj.url, proj.associated_experience)
                for proj in parsed_data.projects
            ]
            psycopg2.extras.execute_batch(cur, proj_sql, proj_values)
            logging.info(f"Inserted {len(proj_values)} project(s).")

        # 5. Insert Awards
        if parsed_data.awards_and_recognition:
            award_sql = """
                INSERT INTO awards_recognition (
                    user_id, title, issuer, award_date, description
                ) VALUES (%s, %s, %s, %s, %s);
            """
            award_values = [
                (user_id, award.title, award.issuer, award.date, award.description)
                for award in parsed_data.awards_and_recognition
            ]
            psycopg2.extras.execute_batch(cur, award_sql, award_values)
            logging.info(f"Inserted {len(award_values)} award(s)/recognition(s).")

        # 6. Insert Publications
        if parsed_data.publications:
            pub_sql = """
                INSERT INTO publications (
                    user_id, title, authors, journal_or_conference, publication_date, url_or_doi
                ) VALUES (%s, %s, %s, %s, %s, %s);
            """
            pub_values = [
                (user_id, pub.title, pub.authors, pub.journal_or_conference, pub.date, pub.url_or_doi)
                for pub in parsed_data.publications
            ]
            psycopg2.extras.execute_batch(cur, pub_sql, pub_values)
            logging.info(f"Inserted {len(pub_values)} publication(s).")

        # 7. Insert Volunteer Experiences
        if parsed_data.volunteer_experience:
            vol_sql = """
                INSERT INTO volunteer_experiences (
                    user_id, organization, role, start_date, end_date, description
                ) VALUES (%s, %s, %s, %s, %s, %s);
            """
            vol_values = [
                (user_id, vol.organization, vol.role, vol.start_date, vol.end_date, vol.description)
                for vol in parsed_data.volunteer_experience
            ]
            psycopg2.extras.execute_batch(cur, vol_sql, vol_values)
            logging.info(f"Inserted {len(vol_values)} volunteer experience(s).")

        conn.commit() # Commit transaction
        logging.info(f"Successfully committed all data for user_id: {user_id}")
        return user_id

    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Database error during insertion for {source_filename}: {error}")
        if conn:
            conn.rollback() # Rollback on error
        logging.error("Transaction rolled back.")
        return None # Indicate failure
    finally:
        if cur:
            cur.close()


# --- NEW: Indexing Related Functions (Step 2) ---

def setup_indexing(conn, vector_dimension: int):
    """Creates the text_chunks table and necessary vector/metadata indexes."""
    # --- Part 1: Ensure Extension and Create Table ---
    # Use the main connection with its default transaction behavior
    with conn.cursor() as cur:
        logging.info("Checking for pgvector extension...")
        try:
            # Run check/create within the default transaction
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logging.info("pgvector extension checked/ensured.")
        except psycopg2.Error as ext_err:
             logging.warning(f"Could not ensure 'vector' extension is enabled (may require manual setup or permissions): {ext_err}")
             conn.rollback() # Rollback the current transaction
             # Check if it exists anyway
             cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
             if not cur.fetchone():
                 logging.error("FATAL: 'vector' extension is not installed or enabled.")
                 raise RuntimeError("pgvector extension missing") from ext_err
             # If it exists despite error, continue cautiously

        logging.info(f"Creating/Checking text_chunks table...")
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
        # Commit this part using the main connection
        conn.commit()
        logging.info("Committed extension check and table creation.")

    # --- Part 2: Create Indexes ---

    # A. Create HNSW Vector Index using a SEPARATE, AUTOCOMMIT connection
    hnsw_index_name = "idx_text_chunks_embedding_hnsw"
    logging.info(f"Checking/Creating HNSW index '{hnsw_index_name}' CONCURRENTLY...")
    temp_conn = None
    try:
        # Check if index exists using the main connection first
        index_exists = False
        with conn.cursor() as cur:
             cur.execute(f"SELECT 1 FROM pg_class WHERE relname = '{hnsw_index_name}' AND relkind = 'i';")
             if cur.fetchone():
                 index_exists = True
                 logging.info(f"HNSW index '{hnsw_index_name}' already exists.")

        if not index_exists:
            logging.info(f"Attempting CONCURRENTLY creation using a temporary autocommit connection...")
            # Establish a new connection specifically for the concurrent index
            temp_conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
            )
            temp_conn.autocommit = True # IMPORTANT: Enable autocommit for this connection
            with temp_conn.cursor() as temp_cur:
                try:
                    temp_cur.execute(f"""
                        CREATE INDEX CONCURRENTLY {hnsw_index_name}
                        ON text_chunks USING hnsw (embedding vector_l2_ops);
                    """)
                    logging.info(f"HNSW index '{hnsw_index_name}' creation initiated successfully.")
                except psycopg2.Error as conc_err:
                    # Handle specific errors if needed, e.g., if it somehow exists now
                    if "already exists" in str(conc_err):
                         logging.info(f"HNSW index '{hnsw_index_name}' already exists (detected during concurrent attempt).")
                    else:
                         logging.warning(f"CONCURRENTLY index creation failed: {conc_err}. Index may need manual creation or a non-concurrent attempt.")
                         # Optionally, add a fallback here to try non-concurrently on the main connection
                         # try:
                         #     with conn.cursor() as cur:
                         #         logging.info("Falling back to non-concurrent index creation...")
                         #         cur.execute(f"CREATE INDEX {hnsw_index_name} ON text_chunks USING hnsw (embedding vector_l2_ops);")
                         #         conn.commit()
                         #         logging.info(f"HNSW index '{hnsw_index_name}' created non-concurrently.")
                         # except psycopg2.Error as fallback_err:
                         #      logging.error(f"Non-concurrent fallback also failed: {fallback_err}")
                         #      conn.rollback()
    except Exception as e:
        logging.error(f"Error during HNSW index check/creation phase: {e}")
        # Don't necessarily stop metadata index creation, but log the error
    finally:
        if temp_conn:
            temp_conn.close()
            logging.debug("Temporary connection for HNSW index closed.")


    # B. Create Metadata Indexes (using IF NOT EXISTS on the main connection)
    logging.info("Checking/Creating metadata indexes...")
    metadata_indexes = {
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
        "idx_projects_technologies": "CREATE INDEX IF NOT EXISTS idx_projects_technologies ON projects USING gin(technologies_used);"
    }

    try:
        # Use the main connection, ensure it's not in autocommit
        if conn.autocommit:
            logging.warning("Main connection was in autocommit mode before metadata indexing. Setting to False.")
            conn.autocommit = False

        with conn.cursor() as cur:
            for index_name, index_sql in metadata_indexes.items():
                logging.debug(f"Executing: {index_sql}")
                cur.execute(index_sql)
        conn.commit() # Commit all metadata index creations
        logging.info("Metadata indexes checked/created.")
    except psycopg2.Error as meta_err:
        logging.error(f"Error during metadata index creation: {meta_err}")
        conn.rollback() # Rollback this batch of indexes
        raise meta_err # Reraise for now

    logging.info("Indexing setup completed.")

    
def get_text_data_to_index(conn) -> List[Dict[str, Any]]:
    """Fetches relevant text data from various tables that needs embedding."""
    data_to_index = []
    cur = None
    queries = {
        "user_summary": "SELECT user_id, summary as text, 'users' as source_table, 'summary' as source_column, NULL as source_pk FROM users WHERE summary IS NOT NULL AND summary != ''",
        "work_desc": "SELECT experience_id as source_pk, user_id, description as text, 'work_experiences' as source_table, 'description' as source_column FROM work_experiences WHERE description IS NOT NULL AND description != ''",
        "project_desc": "SELECT project_id as source_pk, user_id, description as text, 'projects' as source_table, 'description' as source_column FROM projects WHERE description IS NOT NULL AND description != ''",
        "education_desc": "SELECT education_id as source_pk, user_id, description as text, 'education' as source_table, 'description' as source_column FROM education WHERE description IS NOT NULL AND description != ''"
        # Add others like volunteer_desc if desired
    }
    try:
        # Use DictCursor to get rows as dictionaries
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        for query_name, sql in queries.items():
            logging.debug(f"Executing query to fetch data for indexing: {query_name}")
            cur.execute(sql)
            rows = cur.fetchall()
            if rows:
                # Convert DictRow objects to standard dicts
                data_to_index.extend([dict(row) for row in rows])
            logging.debug(f"Fetched {len(rows)} items from {query_name}")

        logging.info(f"Fetched {len(data_to_index)} total text items to potentially index.")
        return data_to_index

    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error fetching data to index: {error}")
        raise
    finally:
        if cur: cur.close()


def clear_existing_chunks(conn, user_id: Optional[int] = None):
    """Deletes existing chunks, optionally for a specific user."""
    cur = None
    try:
        cur = conn.cursor()
        if user_id:
            logging.warning(f"Deleting existing text chunks for user_id: {user_id}")
            cur.execute("DELETE FROM text_chunks WHERE user_id = %s;", (user_id,))
            deleted_count = cur.rowcount
            logging.info(f"Deleted {deleted_count} chunks for user_id {user_id}.")
        else:
            logging.warning("Deleting ALL existing text chunks from the table using TRUNCATE.")
            cur.execute("TRUNCATE TABLE text_chunks RESTART IDENTITY;") # Reset sequence too
            logging.info("Truncated text_chunks table.")

        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error deleting existing chunks: {error}")
        if conn: conn.rollback()
        raise
    finally:
        if cur: cur.close()

def insert_text_chunks_batch(conn, chunks_data: List[Dict[str, Any]]):
    """Inserts a batch of text chunks with their embeddings."""
    if not chunks_data:
        logging.debug("No chunks provided in batch, skipping insertion.")
        return 0
    if np is None:
        logging.error("numpy is required for pgvector integration but was not imported. Cannot insert vectors.")
        raise ImportError("numpy not found, cannot process embeddings for pgvector.")

    sql = """
        INSERT INTO text_chunks (
            user_id, source_table, source_column, source_pk, chunk_text, embedding
        ) VALUES (%s, %s, %s, %s, %s, %s);
    """
    # Prepare data for execute_batch, ensuring embedding is a numpy array
    batch_values = []
    conversion_errors = 0
    for chunk in chunks_data:
        embedding_input = chunk['embedding']
        try:
            # Ensure embedding is a numpy array of float32, required by pgvector adapter
            if isinstance(embedding_input, (list, tuple)):
                embedding_np = np.array(embedding_input, dtype=np.float32)
            elif isinstance(embedding_input, np.ndarray):
                # Ensure correct dtype if it's already numpy
                embedding_np = embedding_input.astype(np.float32, copy=False)
            else:
                raise TypeError(f"Unsupported embedding type: {type(embedding_input)}")

            batch_values.append((
                chunk['user_id'],
                chunk['source_table'],
                chunk['source_column'],
                chunk['source_pk'],
                chunk['chunk_text'],
                embedding_np # Use the numpy array
            ))
        except Exception as conversion_err:
            conversion_errors += 1
            logging.error(f"Error converting embedding to numpy array for chunk (User ID: {chunk.get('user_id')}, Text: '{chunk.get('chunk_text', '')[:50]}...'): {conversion_err}")

    if conversion_errors > 0:
        logging.warning(f"Skipped {conversion_errors} chunks due to embedding conversion errors.")

    if not batch_values:
        logging.warning("Batch is empty after handling conversion errors, skipping insertion.")
        return 0

    cur = None
    try:
        cur = conn.cursor()
        # Use execute_batch for efficiency
        psycopg2.extras.execute_batch(cur, sql, batch_values, page_size=100) # Adjust page_size based on performance/memory
        conn.commit()
        logging.info(f"Successfully inserted batch of {len(batch_values)} text chunks.")
        return len(batch_values)
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error inserting chunk batch: {error}")
        # Log first few chunk texts for debugging if possible
        if batch_values:
            logging.error(f"Sample chunk text from failed batch: '{batch_values[0][4][:100]}...'")
        if conn: conn.rollback()
        raise
    finally:
        if cur: cur.close()


# ----- END OF COMPLETE UPDATED CODE BLOCK -----