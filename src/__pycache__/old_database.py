# src/database.py

import psycopg2
import psycopg2.extras # For dictionary cursor
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from models import ParsedResume # Import the Pydantic model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        logging.info("Database connection established successfully.")
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"Database connection failed: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during database connection: {e}")
        raise

def create_tables(conn):
    """Creates the necessary tables in the database if they don't exist."""
    commands = (
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255) UNIQUE, -- Consider making email unique constraint
            phone VARCHAR(50),
            location TEXT,
            linkedin_url TEXT,
            portfolio_url TEXT,
            github_url TEXT,
            other_url TEXT,
            summary TEXT, -- Storing summary directly here
            skills JSONB, -- Store skills structure as JSONB
            raw_resume_text TEXT, -- Store the full raw text extracted
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            source_filename VARCHAR(512), -- Track the original filename
            llm_metadata JSONB -- Store parsing metadata (model, time etc)
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
            description TEXT -- Store the full description for indexing
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
            description TEXT -- Store description if available
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            name VARCHAR(255),
            description TEXT, -- Store the full description for indexing
            technologies_used TEXT[], -- Store as array of text
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
            award_date VARCHAR(50), -- Changed 'date' to 'award_date' to avoid keyword conflict
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
            publication_date VARCHAR(50), -- Changed 'date' to 'publication_date'
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
        # Trigger to automatically update 'updated_at' timestamp
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
        for command in commands:
            cur.execute(command)
        conn.commit()
        logging.info("Tables checked/created successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error creating tables: {error}")
        if conn:
            conn.rollback() # Rollback changes on error
        raise
    finally:
        if cur:
            cur.close()


def insert_resume_data(conn, parsed_data: ParsedResume, raw_text: str, source_filename: str) -> Optional[int]:
    """Inserts validated and parsed resume data into the database using a transaction."""
    user_id = None
    cur = None

    # Prepare data, handling potential None values from Pydantic models
    contact = parsed_data.contact_information.model_dump() if parsed_data.contact_information else {}
    skills_json = parsed_data.skills.model_dump_json() if parsed_data.skills else '{}' # Convert skills Pydantic model to JSON string
    metadata_json = parsed_data.parsing_metadata.model_dump_json() if parsed_data.parsing_metadata else '{}'

    try:
        cur = conn.cursor()

        # --- Start Transaction ---
        cur.execute("BEGIN;")

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
            raw_text, # Store the full raw text
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
            # Convert list of tech strings to PostgreSQL array format if needed, psycopg2 usually handles lists automatically
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

        # --- Commit Transaction ---
        conn.commit()
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


