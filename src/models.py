# src/models.py (Updated for Step 1 & Step 3)
# ----- START OF COMPLETE UPDATED CODE BLOCK -----

from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Optional, Dict, Any, Union # Ensure all needed types are imported
import datetime
import re # Needed for new QueryFilters validator

# --- Reusable Date Handling (Your version) ---
def normalize_date_string(date_str: Optional[str]) -> Optional[str]:
    if not date_str or date_str.lower() == 'present':
        return date_str
    try:
        datetime.datetime.strptime(date_str, '%Y-%m')
        return date_str
    except ValueError:
        pass
    try:
        datetime.datetime.strptime(date_str, '%Y')
        return date_str
    except ValueError:
        pass
    try:
        dt = datetime.datetime.strptime(date_str, '%B %Y')
        return dt.strftime('%Y-%m')
    except ValueError:
         pass
    try:
        dt = datetime.datetime.strptime(date_str, '%b %Y')
        return dt.strftime('%Y-%m')
    except ValueError:
         pass
    return date_str

# --- Nested Models for Resume Parsing (Your version) ---

class ContactInformation(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    github_url: Optional[str] = None
    other_url: Optional[str] = None

class WorkExperience(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None

    @field_validator('start_date', 'end_date', mode='before')
    @classmethod
    def validate_and_normalize_dates(cls, value):
        if isinstance(value, str):
            return normalize_date_string(value.strip())
        return value

class Education(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None # Your version (no alias)
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    graduation_date: Optional[str] = None
    description: Optional[str] = None

    @field_validator('start_date', 'end_date', 'graduation_date', mode='before')
    @classmethod
    def validate_and_normalize_dates(cls, value):
        if isinstance(value, str):
            return normalize_date_string(value.strip())
        return value

class Skills(BaseModel):
    technical: List[str] = Field(default_factory=list)
    soft: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    other: List[str] = Field(default_factory=list)

class Project(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    technologies_used: List[str] = Field(default_factory=list) # Your version (no alias)
    url: Optional[str] = None
    associated_experience: Optional[str] = None # Your version (no alias)

class AwardRecognition(BaseModel):
    title: Optional[str] = None
    issuer: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None

    @field_validator('date', mode='before')
    @classmethod
    def validate_and_normalize_dates(cls, value):
        if isinstance(value, str):
            return normalize_date_string(value.strip())
        return value

class Publication(BaseModel):
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    journal_or_conference: Optional[str] = None # Your version (no alias)
    date: Optional[str] = None
    url_or_doi: Optional[str] = None # Your version (no alias)

    @field_validator('date', mode='before')
    @classmethod
    def validate_and_normalize_dates(cls, value):
        if isinstance(value, str):
            return normalize_date_string(value.strip())
        return value

class VolunteerExperience(BaseModel):
    organization: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

    @field_validator('start_date', 'end_date', mode='before')
    @classmethod
    def validate_and_normalize_dates(cls, value):
        if isinstance(value, str):
            return normalize_date_string(value.strip())
        return value

class ParsingMetadata(BaseModel):
    model_used: Optional[str] = None # Your version (no alias)
    timestamp_utc: Optional[str] = None # Your version (no alias)
    processing_time_seconds: Optional[float] = None # Your version (no alias)

# --- Main Resume Model (Your version) ---

class ParsedResume(BaseModel):
    schema_version: Optional[str] = None
    parsing_metadata: Optional[ParsingMetadata] = None
    contact_information: Optional[ContactInformation] = None
    summary: Optional[str] = None
    work_experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: Optional[Skills] = Field(default_factory=Skills)
    projects: List[Project] = Field(default_factory=list)
    awards_and_recognition: List[AwardRecognition] = Field(default_factory=list)
    publications: List[Publication] = Field(default_factory=list)
    volunteer_experience: List[VolunteerExperience] = Field(default_factory=list)
    raw_text_preview: Optional[str] = None

    @field_validator('*', mode='before')
    @classmethod
    def strip_strings(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

    class Config:
        extra = 'allow'

# --- NEW: Pydantic Models for Structured Query Output (Step 3) ---
# --- Using snake_case consistently ---

class QueryFilters(BaseModel):
    """Defines the structure for extracted filters from user queries."""
    location: Optional[List[str]] = Field(default_factory=list)
    company_name: Optional[List[str]] = Field(default_factory=list)
    job_title: Optional[List[str]] = Field(default_factory=list)
    industry: Optional[List[str]] = Field(default_factory=list)
    skills: Optional[List[str]] = Field(default_factory=list)
    min_experience_years: Optional[int] = None
    max_experience_years: Optional[int] = None
    keywords: Optional[List[str]] = Field(default_factory=list, description="Catch-all for non-specific concepts, transition elements, project types, etc.")
    currently_working_at: Optional[str] = None
    previously_worked_at: Optional[str] = None
    founded_company: Optional[bool] = None
    open_to_consulting: Optional[bool] = None
    network_relation: Optional[str] = Field(None, description="e.g., 'my_network', 'alumni'")
    role_seniority: Optional[List[str]] = Field(default_factory=list, description="e.g., 'senior', 'junior', 'lead', 'VP', 'principal', 'entry-level'") # <-- NEW FIELD

    class Config:
        extra = 'allow'

    @field_validator('skills', 'keywords', 'location', 'company_name', 'job_title', 'industry', 'role_seniority', mode='before') # Added role_seniority
    @classmethod
    def clean_string_list(cls, value: Any) -> List[str]:
        """Cleans list fields: ensures strings, strips whitespace, handles single string input."""
        if isinstance(value, list):
            return [str(item).strip() for item in value if isinstance(item, (str, int, float)) and str(item).strip()]
        elif isinstance(value, str) and value.strip():
            items = re.split(r'[;,]', value)
            return [item.strip() for item in items if item.strip()]
        return []

class StructuredQuery(BaseModel):
    """The final structured representation of the user's query after LLM processing."""
    semantic_query: Optional[str] = Field(None, description="Core meaning for vector search.")
    filters: QueryFilters = Field(default_factory=QueryFilters, description="Structured filters extracted.")
    query_type: Optional[str] = Field(None, description="Classification of query intent.")

    @field_validator('semantic_query', mode='before')
    @classmethod
    def clean_semantic_query(cls, value: Any) -> Optional[str]:
        if isinstance(value, str):
            cleaned = value.strip(); return cleaned if cleaned else None
        return None

# ----- END OF COMPLETE UPDATED CODE BLOCK -----