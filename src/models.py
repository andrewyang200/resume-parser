# src/models.py (Enhanced for Professional Networking Query Processing)
# ----- START OF COMPLETE UPDATED CODE BLOCK -----

from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Optional, Dict, Any, Union, cast # Ensure all needed types are imported
import datetime
import re # Needed for new QueryFilters validator

# --- Reusable Date Handling ---
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

# --- Nested Models for Resume Parsing ---

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
    field_of_study: Optional[str] = None
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
    technologies_used: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    associated_experience: Optional[str] = None

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
    journal_or_conference: Optional[str] = None
    date: Optional[str] = None
    url_or_doi: Optional[str] = None

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
    model_used: Optional[str] = None
    timestamp_utc: Optional[str] = None
    processing_time_seconds: Optional[float] = None

# --- Main Resume Model ---

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

# --- Enhanced Models for Structured Query Output ---

class ProfessionalConceptExpansions(BaseModel):
    """Detailed expansions of professional concepts by category."""
    skills: Optional[List[str]] = Field(default_factory=list)
    roles: Optional[List[str]] = Field(default_factory=list)
    industries: Optional[List[str]] = Field(default_factory=list)
    company_types: Optional[List[str]] = Field(default_factory=list)
    technologies: Optional[List[str]] = Field(default_factory=list)
    achievements: Optional[List[str]] = Field(default_factory=list)
    certifications: Optional[List[str]] = Field(default_factory=list)

    class Config: extra = 'allow'

    @field_validator('*', mode='before')
    @classmethod
    def clean_string_list(cls, value: Any) -> List[str]:
        if isinstance(value, list): 
            return [str(item).strip() for item in value if isinstance(item, (str, int, float)) and str(item).strip()]
        elif isinstance(value, str) and value.strip(): 
            return [item.strip() for item in re.split(r'[;,]', value) if item.strip()]
        return []

class ConfidenceMetrics(BaseModel):
    """Confidence metrics for query understanding."""
    overall: Optional[float] = None
    ambiguity_level: Optional[str] = None
    ambiguity_reason: Optional[str] = None
    
    @field_validator('overall', mode='before')
    @classmethod
    def validate_overall(cls, value):
        if isinstance(value, list) and not value:
            return None
        return value
        
    @field_validator('ambiguity_level', 'ambiguity_reason', mode='before')
    @classmethod
    def validate_string_fields(cls, value):
        if isinstance(value, list) and not value:
            return None
        if isinstance(value, str):
            return value.strip() or None
        return value

class QueryFilters(BaseModel):
    """Extracted structured filters."""
    location: Optional[List[str]] = Field(default_factory=list)
    company_name: Optional[List[str]] = Field(default_factory=list)
    job_title: Optional[List[str]] = Field(default_factory=list)
    industry: Optional[List[str]] = Field(default_factory=list)
    skills: Optional[List[str]] = Field(default_factory=list)
    min_experience_years: Optional[int] = None
    max_experience_years: Optional[int] = None
    keywords: Optional[List[str]] = Field(default_factory=list, description="Keywords explicitly mentioned or inferred from context.")
    currently_working_at: Optional[str] = None
    previously_worked_at: Optional[str] = None
    founded_company: Optional[bool] = None
    open_to_consulting: Optional[bool] = None
    network_relation: Optional[str] = None
    role_seniority: Optional[List[str]] = Field(default_factory=list)
    # New filter fields
    career_stage: Optional[List[str]] = Field(default_factory=list)
    project_types: Optional[List[str]] = Field(default_factory=list)
    educational_background: Optional[List[str]] = Field(default_factory=list)
    interests: Optional[List[str]] = Field(default_factory=list)
    availability: Optional[str] = None

    class Config: extra = 'allow'

    # Validators for list fields
    @field_validator('location', 'company_name', 'job_title', 'industry', 'skills', 
                    'keywords', 'role_seniority', 'career_stage', 'project_types', 
                    'educational_background', 'interests', mode='before')
    @classmethod
    def clean_string_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list): 
            return [str(item).strip() for item in value if isinstance(item, (str, int, float)) and str(item).strip()]
        elif isinstance(value, str) and value.strip(): 
            return [item.strip() for item in re.split(r'[;,]', value) if item.strip()]
        return []
    
    # Validators for scalar fields that might receive empty lists
    @field_validator('min_experience_years', 'max_experience_years', mode='before')
    @classmethod
    def clean_integer_field(cls, value: Any) -> Optional[int]:
        if isinstance(value, list) and not value:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return int(value.strip())
            except ValueError:
                return None
        return None
    
    @field_validator('currently_working_at', 'previously_worked_at', 'network_relation', 'availability', mode='before')
    @classmethod
    def clean_string_field(cls, value: Any) -> Optional[str]:
        if isinstance(value, list) and not value:
            return None
        if isinstance(value, str):
            return value.strip() or None
        return None
    
    @field_validator('founded_company', 'open_to_consulting', mode='before')
    @classmethod
    def clean_boolean_field(cls, value: Any) -> Optional[bool]:
        if isinstance(value, list) and not value:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ('true', 'yes', '1'):
            return True
        if isinstance(value, str) and value.strip().lower() in ('false', 'no', '0'):
            return False
        return None

class StructuredQuery(BaseModel):
    """Structured representation including LLM refinements and expansions."""
    original_query: Optional[str] = Field(None, description="The raw query entered by the user.")
    semantic_query: Optional[str] = Field(None, description="Core meaning for vector search.")
    refined_semantic_query: Optional[str] = Field(None, description="LLM-generated alternative semantic query if original is vague.")
    filters: QueryFilters = Field(default_factory=QueryFilters, description="Structured filters extracted.")
    expanded_keywords: Optional[List[str]] = Field(default_factory=list, description="LLM-generated synonyms, related terms, category examples.")
    # New fields for enhanced understanding
    professional_concept_expansions: ProfessionalConceptExpansions = Field(default_factory=ProfessionalConceptExpansions, 
                                                                          description="Domain-specific concept expansions.")
    implicit_needs: Optional[List[str]] = Field(default_factory=list, description="Unstated but implied requirements.")
    query_type: Optional[str] = Field(None, description="Classification of query intent.")
    confidence: Optional[ConfidenceMetrics] = Field(default_factory=ConfidenceMetrics, description="Confidence metrics for query understanding.")

    @field_validator('semantic_query', 'refined_semantic_query', 'query_type', mode='before')
    @classmethod
    def clean_optional_string(cls, value: Any) -> Optional[str]:
        if isinstance(value, list) and not value:
            return None
        if isinstance(value, str): 
            cleaned = value.strip()
            return cleaned if cleaned else None
        return None

    @field_validator('expanded_keywords', 'implicit_needs', mode='before')
    @classmethod
    def clean_string_list(cls, value: Any) -> List[str]:
        return QueryFilters.clean_string_list(value)

# ----- END OF COMPLETE UPDATED CODE BLOCK -----