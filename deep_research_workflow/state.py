from typing import Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict
from .struct import Section, Feedback, Query, SearchResult, DatasetSchema


class AgentState(TypedDict):
    topic: str
    outline: str
    messages: Annotated[List[BaseMessage], operator.add]
    report_structure: str
    sections: List[Section]
    final_section_dataset: Annotated[List[Dict[str, Any]], operator.add] = []
    final_dataset: List[Dict[str, Any]]
    schema: DatasetSchema

class ResearchState(TypedDict):
    topic: str
    report_structure: str
    section: Section
    knowledge: str
    reflection_feedback: Feedback = Feedback(feedback="")
    generated_queries: List[Query] = []
    searched_queries: Annotated[List[Query], operator.add] = []
    search_results: Annotated[List[SearchResult], operator.add] = []
    accumulated_content: str = ""
    reflection_count: int = 1
    final_section_content: List[str] = []
    schema: DatasetSchema
    final_section_dataset: List[Dict[str, Any]] = []
    error: str