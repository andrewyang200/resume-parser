# src/indexer.py (Corrected)

import argparse
import logging
import time
from typing import List, Dict, Any, Optional

# Import local modules
import database # Our database interaction functions

try:
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError as e:
    print(f"Error: Required library not found: {e.name}. Run 'pip install -r requirements.txt'")
    exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# Choose a sentence transformer model
# 'all-mpnet-base-v2' is a strong baseline. Dimension = 768
# 'all-MiniLM-L6-v2' is faster, smaller, slightly less accurate. Dimension = 384
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
EMBEDDING_DIMENSION = 768 # MUST match the model

# Chunking parameters
MIN_CHUNK_LENGTH = 30 # Minimum characters for a chunk to be considered meaningful

# --- Function Definition (Remains the same) ---
def chunk_text(text: str, min_length: int = MIN_CHUNK_LENGTH) -> List[str]:
    """Simple chunking strategy: split by double newlines (paragraphs)."""
    if not text:
        return []
    # Split by paragraphs, remove leading/trailing whitespace from each
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    # Filter out very short chunks
    meaningful_chunks = [chunk for chunk in chunks if len(chunk) >= min_length]
    return meaningful_chunks

# --- Main Indexing Function (Loop variable renamed) ---
def run_indexing(rebuild_indexes: bool = False, clear_all: bool = False, rebuild_fts: bool = False):
    """Fetches data, generates embeddings, and stores them in PostgreSQL."""
    logging.info("--- Starting Indexing Process ---")
    start_time = time.time()

    # --- 1. Load Config & Initialize ---
    logging.info("Loading configuration and initializing...")
    load_dotenv()
    db_conn = None
    model = None

    try:
        db_conn = database.get_db_connection()
        if not db_conn:
            raise ConnectionError("Failed to get DB connection.")

        # Optionally set up tables and base indexes (usually done once)
        if rebuild_indexes or rebuild_fts: # Check either flag
            logging.info("Setting up database tables and indexes (incl. FTS if requested)...")
            # Pass the rebuild_fts flag to setup_indexing
            database.setup_indexing(db_conn, EMBEDDING_DIMENSION, rebuild_fts=rebuild_fts)
        else:
            logging.info("Skipping index/table setup (use --rebuild to force).")

        # Load the Sentence Transformer model
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        model_load_start = time.time()
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info(f"Model loaded in {time.time() - model_load_start:.2f}s.")

        # --- 2. Clear Existing Chunks (Optional) ---
        if clear_all:
            database.clear_existing_chunks(db_conn)

        # --- 3. Fetch Data ---
        logging.info("Fetching text data to index from database...")
        items_to_process = database.get_text_data_to_index(db_conn)
        if not items_to_process:
            logging.info("No text data found to index.")
            return

        # --- 4. Chunk, Embed, and Store ---
        logging.info("Starting chunking, embedding, and storing...")
        total_chunks_inserted = 0
        chunks_batch = [] # Accumulate chunks for batch insertion
        batch_size = 128 # Adjust based on memory/performance

        for i, item in enumerate(items_to_process):
            user_id = item['user_id']
            text = item['text']
            source_table = item['source_table']
            source_column = item['source_column']
            source_pk = item['source_pk']

            if i % 100 == 0 and i > 0: # Log progress periodically
                 logging.info(f"Processing item {i+1}/{len(items_to_process)} (User ID: {user_id}, Source: {source_table}.{source_column})")

            # --- FIX IS HERE: Calling the function chunk_text ---
            chunks = chunk_text(text, MIN_CHUNK_LENGTH)
            # --- End Fix ---

            if not chunks:
                continue

            # Generate embeddings for the chunks of the current item IN BATCH
            try:
                 # Ensure model uses MPS device on Mac if available and suitable
                 # encode method handles device placement automatically based on model initialization
                 chunk_embeddings = model.encode(chunks, show_progress_bar=False).tolist() # Convert numpy arrays to lists for storage prep
            except Exception as e:
                 logging.error(f"Error generating embeddings for user {user_id}, source {source_table}.{source_column}: {e}")
                 continue # Skip this item

            if len(chunks) != len(chunk_embeddings):
                 logging.error(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(chunk_embeddings)}) for user {user_id}. Skipping.")
                 continue

            # Prepare chunk data for insertion
            # --- FIX IS HERE: Renamed loop variable ---
            for current_chunk_text, embedding in zip(chunks, chunk_embeddings):
                 # --- End Fix ---
                 chunks_batch.append({
                     'user_id': user_id,
                     'source_table': source_table,
                     'source_column': source_column,
                     'source_pk': source_pk,
                      # --- FIX IS HERE: Using the renamed variable ---
                     'chunk_text': current_chunk_text,
                      # --- End Fix ---
                     'embedding': embedding # Store as list or numpy array (handled by insert func)
                 })

            # Insert when batch is full
            if len(chunks_batch) >= batch_size:
                 inserted_count = database.insert_text_chunks_batch(db_conn, chunks_batch)
                 total_chunks_inserted += inserted_count
                 chunks_batch = [] # Reset batch

        # Insert any remaining chunks
        if chunks_batch:
             inserted_count = database.insert_text_chunks_batch(db_conn, chunks_batch)
             total_chunks_inserted += inserted_count

        logging.info(f"Finished processing {len(items_to_process)} items.")
        logging.info(f"Total chunks inserted: {total_chunks_inserted}")

    except Exception as e:
        logging.error(f"Indexing pipeline failed: {e}")
        logging.exception("Detailed traceback:")
    finally:
        if db_conn:
            db_conn.close()
            logging.info("Database connection closed.")

    end_time = time.time()
    logging.info(f"--- Indexing Process Completed in {end_time - start_time:.2f}s ---")

# --- Main Execution Logic for Testing ---
if __name__ == "__main__":
    # --- FIX: Replace ... with actual arguments ---
    parser = argparse.ArgumentParser(description="Generate and store text embeddings for resume data.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force creation/recreation of vector tables and indexes (use with caution)." # Clarified help text
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing chunks before indexing (use with caution)."
    )
    # --- End Fix ---

    # --- ADD THIS (from previous step, KEEP THIS) ---
    parser.add_argument(
        "--rebuild_fts",
        action="store_true",
        help="Force setup/re-setup of FTS columns, triggers, and indexes (Run once or if schema changes)."
    )
    # --- END ADD ---
    args = parser.parse_args()

    # Modify the call to run_indexing (KEEP THIS)
    run_indexing(rebuild_indexes=args.rebuild, clear_all=args.clear, rebuild_fts=args.rebuild_fts)