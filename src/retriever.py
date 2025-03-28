# src/retriever.py (Updated for Enhanced Query & FTS)
# ----- START OF COMPLETE UPDATED CODE BLOCK -----

import argparse
import logging
import time
import os
from typing import List, Dict, Optional, Tuple, Any, Set

import database
try:
    # Import updated models
    from .models import StructuredQuery, QueryFilters
    from .query_processor import process_query as get_structured_query
    from .query_processor import load_openai_key
    from openai import OpenAI
except ImportError:
    from models import StructuredQuery, QueryFilters
    from query_processor import process_query as get_structured_query
    from query_processor import load_openai_key
    from openai import OpenAI

try:
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import psycopg2
    import psycopg2.extras
except ImportError as e:
    print(f"Error: Required library not found: {e.name}. Run 'pip install -r requirements.txt'")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Constants ---
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
EMBEDDING_DIMENSION = 768
DEFAULT_VECTOR_SEARCH_TOP_K = 200
DEFAULT_FINAL_RESULTS_LIMIT = 20
FTS_LANGUAGE = 'english' # Match language used in database.py triggers

# --- Global Embedding Model ---
embedding_model: Optional[SentenceTransformer] = None
# ... (Keep load_embedding_model function as before) ...
def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    global embedding_model
    if embedding_model is None:
        logging.info(f"Loading embedding model: {model_name}...")
        model_load_start = time.time()
        try: embedding_model = SentenceTransformer(model_name)
        except Exception as e: logging.error(f"Failed to load model: {e}"); raise
        logging.info(f"Model loaded in {time.time() - model_load_start:.2f}s.")
    return embedding_model

# --- Embed Query (Use Refined if available) ---
def embed_query(query_text: str) -> Optional[np.ndarray]:
    """Generates embedding, ensuring correct type and dimension."""
    try:
        model = load_embedding_model()
        if not query_text: logging.warning("Empty query text for embedding."); return None
        embedding = model.encode(query_text, show_progress_bar=False)
        if isinstance(embedding, np.ndarray) and embedding.shape == (EMBEDDING_DIMENSION,):
             return embedding.astype(np.float32)
        else: logging.error(f"Embedding has unexpected shape/type"); return None
    except Exception as e: logging.error(f"Error embedding query: {e}"); return None

# Replace the existing _execute_vector_search function with this fixed version
def _execute_vector_search(conn, query_embedding: np.ndarray, top_k: int) -> Dict[int, float]:
    """Performs vector similarity search on text_chunks."""
    if query_embedding is None: return {}
    logging.info(f"Executing vector search for top {top_k} chunks...")
    vector_search_start = time.time()
    distance_operator = '<=>'  # Cosine distance operator for pgvector
    
    sql = f"""
        WITH RankedChunks AS (
            SELECT user_id, chunk_id, embedding {distance_operator} %s as distance
            FROM text_chunks
            ORDER BY embedding {distance_operator} %s
            LIMIT %s
        )
        SELECT user_id, MIN(distance) as best_distance 
        FROM RankedChunks 
        GROUP BY user_id 
        ORDER BY best_distance ASC;
    """
    
    params = (query_embedding, query_embedding, top_k)
    candidate_scores: Dict[int, float] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            results = cur.fetchall()
            for user_id, distance in results:
                candidate_scores[user_id] = float(distance)
            logging.info(f"Vector search found {len(candidate_scores)} candidates in {time.time() - vector_search_start:.3f}s.")
            return candidate_scores
    except Exception as e:
        logging.error(f"Vector search error: {e}")
        conn.rollback()
        return {}
    
# --- UPDATED: Build Metadata & FTS Filter ---
def _build_metadata_fts_sql_filter(filters: QueryFilters, expanded_keywords: List[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Builds SQL WHERE clause using specific filters and FTS for combined keywords.
    Determines required JOINs based on filters and keyword search scope.
    """
    clauses: List[str] = []
    params: Dict[str, Any] = {}
    required_joins: Set[str] = set() # Track needed joins ('we', 'e', 'p')
    param_count = 0

    def get_param_name(base: str) -> str: nonlocal param_count; param_count += 1; return f"{base}_{param_count}"

    # Combine explicit and expanded keywords
    all_keywords = list(set((filters.keywords or []) + (expanded_keywords or [])))

    # --- Specific Filters (adding required joins) ---
    if filters.location:
        loc_param = get_param_name("loc")
        clauses.append(f"u.location ILIKE ANY(%({loc_param})s)")
        params[loc_param] = [f"%{loc}%" for loc in filters.location]
        # No join needed for users.location

    if filters.company_name:
        comp_param = get_param_name("comp")
        clauses.append(f"we.company = ANY(%({comp_param})s)")
        params[comp_param] = filters.company_name
        required_joins.add('we')

    if filters.job_title:
        title_clauses = []
        for title in filters.job_title:
             p_name = get_param_name("title"); title_clauses.append(f"we.title ILIKE %({p_name})s"); params[p_name] = f"%{title}%"
        if title_clauses: clauses.append(f"({' OR '.join(title_clauses)})")
        required_joins.add('we')

    if filters.role_seniority:
        seniority_clauses = []
        for seniority in filters.role_seniority:
             p_name = get_param_name("sen"); seniority_clauses.append(f"we.title ILIKE %({p_name})s"); params[p_name] = f"%{seniority}%"
        if seniority_clauses: clauses.append(f"({' OR '.join(seniority_clauses)})")
        required_joins.add('we')

    if filters.currently_working_at:
        curr_comp_param = get_param_name("curr_comp")
        clauses.append(f"(we.company ILIKE %({curr_comp_param})s AND (we.end_date IS NULL OR we.end_date ILIKE 'Present'))")
        params[curr_comp_param] = f"%{filters.currently_working_at}%" # Use ILIKE
        required_joins.add('we')

    if filters.previously_worked_at:
        prev_comp_param = get_param_name("prev_comp")
        clauses.append(f"(we.company ILIKE %({prev_comp_param})s AND we.end_date IS NOT NULL AND we.end_date NOT ILIKE 'Present')")
        params[prev_comp_param] = f"%{filters.previously_worked_at}%" # Use ILIKE
        required_joins.add('we')

    # Note: min/max experience, founded_company, open_to_consulting, industry, network_relation
    # filters are not fully implemented here for MVP filter logic, but could be added.
    if filters.founded_company is not None:
         logging.warning("Filter 'founded_company' not implemented in SQL yet.")
         # Requires checking titles or descriptions in `we` join
    # ... add checks for other filters and required_joins ...

    # --- FTS Keyword Filter (using combined keywords) ---
    if all_keywords:
        keyword_fts_clauses = []
        fts_query_param = get_param_name("fts_query")

        # Determine which FTS columns to search based on potential keyword relevance
        fts_columns_to_search = ["u.fts_doc"] # Always search user doc
        required_joins.add('we') # Assume keywords might relate to experience
        fts_columns_to_search.append("we.fts_doc")
        required_joins.add('p') # Assume keywords might relate to projects
        fts_columns_to_search.append("p.fts_doc")
        # Add e.fts_doc if keywords might be school names, etc.
        # For simplicity, let's search education too if keywords present
        required_joins.add('e')
        fts_columns_to_search.append("e.fts_doc")


        # Build the FTS query string (e.g., 'finance & tech' or 'harvard | yale | princeton')
        # Use websearch_to_tsquery for more flexible user input style parsing
        # Or plainto_tsquery for simple ANDing
        # Use to_tsquery with OR logic between terms
        # First, properly format each keyword for to_tsquery
        formatted_keywords = []
        for keyword in all_keywords:
            keyword_param = get_param_name("kw")
            formatted_keywords.append(f"to_tsquery('{FTS_LANGUAGE}', %({keyword_param})s)")
            params[keyword_param] = keyword.replace(' ', ' & ')  # Handle multi-word terms

        if formatted_keywords:
            # Join all keyword queries with OR operator
            fts_query_expr = " || ".join(formatted_keywords)
            
            # Create OR condition across relevant FTS columns
            fts_match_clauses = []
            for col in fts_columns_to_search:
                fts_match_clauses.append(f"{col} @@ ({fts_query_expr})")
            
            fts_match_expr = " OR ".join(fts_match_clauses)
            clauses.append(f"({fts_match_expr})")

    # --- Construct Final SQL Query Parts ---
    base_query = "SELECT DISTINCT u.user_id FROM users u"
    join_clauses: List[str] = []
    # Add joins ONLY if required by specific filters or FTS keyword search
    if 'we' in required_joins: join_clauses.append("LEFT JOIN work_experiences we ON u.user_id = we.user_id")
    if 'e' in required_joins: join_clauses.append("LEFT JOIN education e ON u.user_id = e.user_id")
    if 'p' in required_joins: join_clauses.append("LEFT JOIN projects p ON u.user_id = p.user_id")
    # Add other joins...

    query_with_joins = f"{base_query} {' '.join(join_clauses)}"
    sql_where_clause = ""
    if clauses:
        sql_where_clause = "WHERE " + " AND ".join(f"({c})" for c in clauses)

    final_sql_base = f"{query_with_joins} {sql_where_clause}"

    logging.debug(f"Constructed Base Filter SQL: {final_sql_base}")
    logging.debug(f"Parameters: {params}")

    return final_sql_base, params


# --- Execute Metadata Filter (Keep previous version) ---
def _execute_metadata_filter(conn, candidate_user_ids: List[int], base_filter_sql: str, filter_params: Dict[str, Any]) -> List[int]:
    """Executes the metadata/FTS filter query against the candidate users."""
    # ... (Keep the existing logic that adds "AND u.user_id = ANY(...) " to base_filter_sql) ...
    if not candidate_user_ids: return []
    logging.info(f"Executing metadata/FTS filter on {len(candidate_user_ids)} candidates...")
    metadata_filter_start = time.time()
    if "WHERE" in base_filter_sql.upper(): candidate_filter_clause = "AND u.user_id = ANY(%(candidate_ids)s)"
    else: candidate_filter_clause = "WHERE u.user_id = ANY(%(candidate_ids)s)"
    final_sql = f"{base_filter_sql} {candidate_filter_clause}"
    params = filter_params.copy(); params["candidate_ids"] = candidate_user_ids
    logging.debug(f"Final Filter SQL: {final_sql}"); logging.debug(f"Final Params: {params}")
    filtered_user_ids: List[int] = []
    try:
        with conn.cursor() as cur:
            cur.execute(final_sql, params); results = cur.fetchall()
            filtered_user_ids = [row[0] for row in results]
        logging.info(f"Metadata/FTS filter yielded {len(filtered_user_ids)} users in {time.time() - metadata_filter_start:.3f}s.")
        return filtered_user_ids
    except Exception as e: logging.error(f"Metadata/FTS filter query error: {e}"); conn.rollback(); return []

# --- Main Retrieval Function (Updated) ---
def hybrid_retrieval(
    structured_query: StructuredQuery,
    top_k_vector: int = DEFAULT_VECTOR_SEARCH_TOP_K,
    limit: int = DEFAULT_FINAL_RESULTS_LIMIT
) -> List[Tuple[int, float]]:
    """ Performs hybrid retrieval using enhanced query and FTS filtering. """
    retrieval_start_time = time.time()
    conn = None
    final_results: List[Tuple[int, float]] = []

    if not structured_query:
        logging.error("Received None for structured_query.")
        return []

    try:
        conn = database.get_db_connection()
        if not conn: raise ConnectionError("Failed DB connection")

        # --- Use Refined Semantic Query if available ---
        query_text_to_embed = structured_query.refined_semantic_query or structured_query.semantic_query
        logging.info(f"Using query for embedding: '{query_text_to_embed}'")

        query_embedding = None
        if query_text_to_embed:
            query_embedding = embed_query(query_text_to_embed)
        else:
            logging.warning("No semantic query or refined query available for embedding.")
            # Decide fallback: Only metadata search? Return empty?
            # For now, require semantic component.
            return []

        if query_embedding is None:
            logging.error("Failed query embedding. Aborting.")
            return []

        # --- Vector Search ---
        candidate_scores = _execute_vector_search(conn, query_embedding, top_k_vector)
        if not candidate_scores:
            logging.info("Vector search returned no candidates.")
            return []
        candidate_user_ids = list(candidate_scores.keys())

        # --- Build and Execute Metadata/FTS Filter ---
        filtered_user_ids = candidate_user_ids
        # Combine keywords
        all_keywords = list(set((structured_query.filters.keywords or []) + (structured_query.expanded_keywords or [])))
        logging.info(f"Combined Keywords for FTS/Filtering: {all_keywords}")

        # Check if any filters or combined keywords exist
        has_filters = any(getattr(structured_query.filters, f) for f in structured_query.filters.model_fields_set if getattr(structured_query.filters, f) not in [None, []])
        has_keywords = bool(all_keywords)

        if has_filters or has_keywords:
             base_filter_sql, filter_params = _build_metadata_fts_sql_filter(structured_query.filters, all_keywords)
             filtered_user_ids = _execute_metadata_filter(conn, candidate_user_ids, base_filter_sql, filter_params)
        else:
             logging.info("No metadata filters or keywords specified, using all vector search candidates.")

        if not filtered_user_ids:
            logging.info("No users matched combined vector search and filters.")
            return []

        # --- Combine Scores & Rank ---
        for user_id in filtered_user_ids:
            score = candidate_scores.get(user_id)
            if score is not None:
                 final_results.append((user_id, score))
            else:
                 logging.warning(f"User {user_id} passed filtering but missing score.")
        final_results.sort(key=lambda item: item[1]) # Sort by distance ASC

        logging.info(f"Hybrid retrieval completed in {time.time() - retrieval_start_time:.3f}s. Found {len(final_results)} final results.")
        return final_results[:limit]

    except Exception as e:
        logging.error(f"Hybrid retrieval failed: {e}"); logging.exception("Traceback:")
        return []
    finally:
        if conn: conn.close(); logging.debug("DB connection closed.")


# --- Main Execution Logic for Testing (Remains the same) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform hybrid retrieval with query expansion/FTS.")
    parser.add_argument("query", help="Natural language query.")
    parser.add_argument("--limit", type=int, default=DEFAULT_FINAL_RESULTS_LIMIT)
    parser.add_argument("--vector_k", type=int, default=DEFAULT_VECTOR_SEARCH_TOP_K)
    parser.add_argument("--query_model", default="gpt-4o", help="LLM for query processing.") # Use best model
    args = parser.parse_args()

    # --- Get Structured Query ---
    openai_client = None; structured_query = None; api_key = load_openai_key()
    if not api_key: exit(1)
    try:
        openai_client = OpenAI(api_key=api_key, timeout=90.0)
        # Use the updated query processor function
        structured_query = get_structured_query(args.query, openai_client, model=args.query_model)
    except Exception as e: logging.error(f"Query processing init failed: {e}"); exit(1)
    if not structured_query: print("\n--- Failed structured query gen. ---"); exit(1)
    print("\n--- Generated Structured Query ---"); print(structured_query.model_dump_json(indent=2)); print("---")

    # --- Perform Retrieval ---
    print(f"\n--- Performing Hybrid Retrieval (Limit: {args.limit}, Vector K: {args.vector_k}) ---")
    ranked_results = hybrid_retrieval(
        structured_query=structured_query, top_k_vector=args.vector_k, limit=args.limit
    )

    # --- Print Results ---
    if ranked_results:
        print(f"\nFound {len(ranked_results)} results:")
        for i, (user_id, score) in enumerate(ranked_results): print(f"{i+1}. User ID: {user_id:<5} (Score: {score:.4f})")
        print("\nNOTE: Score = distance (lower is better).")
    else: print("\n--- No matching users found. ---")

# ----- END OF COMPLETE UPDATED CODE BLOCK -----


