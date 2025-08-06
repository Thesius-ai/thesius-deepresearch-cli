from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any


class FieldType(str, Enum):
    string = "string"
    number = "number"
    array = "array"
    boolean = "boolean"

class SchemaField(BaseModel):
    key: str = Field(..., description="The unique identifier for the field")
    type: FieldType = Field(..., description="The data type of the field")
    description: str = Field(..., description="Some descriptive information for the field")

class DatasetSchema(BaseModel):
    generated_schema: list[SchemaField]

class DatasetRecords(BaseModel):
    dataset: List[Dict[str, Any]]


class Section(BaseModel):
    section_name: str = Field(..., description="The name of this section of the report without its number")
    sub_sections: List[str] = Field(..., description="Comprehensive descriptions of sub-sections, each combining the sub-section title and its bullet points into a fluid, natural-language description")

class Sections(BaseModel):
    sections: List[Section] = Field(..., description="A list of sections")

class Query(BaseModel):
    query: str = Field(..., description="A search query")

class Queries(BaseModel):
    queries: List[Query] = Field(..., description="A list of search queries")

class SearchResult(BaseModel):
    query: Query = Field(..., description="The search query that was used to retrieve the raw content")
    raw_content: list[str] = Field(..., description="The raw content retrieved from the search")

class Feedback(BaseModel):
    feedback: Union[str, bool] = Field(..., description="Feedback on the report structure. If the content is good for the section, return True (boolean), otherwise return a string of feedback on what is missing or incorrect.")

class SectionOutput(BaseModel):
    # final_section_content: List[str] = Field(..., description="The final section content")
    final_section_dataset: List[Dict[str, Any]] = Field(..., description="The final section dataset")
