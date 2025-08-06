import time
import json
from openai import RateLimitError, OpenAIError
from pydantic import ValidationError
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command, Send
from typing import Literal
from tavily import TavilyClient

from .struct import DatasetRecords
from .state import AgentState, ResearchState
from .configuration import Configuration
from .utils import init_llm, process_datagen_prompt
from .prompts import (
    SCHEMA_GENERATION_PROMPT,
    REPORT_STRUCTURE_PLANNER_SYSTEM_PROMPT_TEMPLATE,
    SECTION_FORMATTER_SYSTEM_PROMPT_TEMPLATE,
    SECTION_KNOWLEDGE_SYSTEM_PROMPT_TEMPLATE,
    QUERY_GENERATOR_SYSTEM_PROMPT_TEMPLATE,
    RESULT_ACCUMULATOR_SYSTEM_PROMPT_TEMPLATE,
    REFLECTION_FEEDBACK_SYSTEM_PROMPT_TEMPLATE,
    FINAL_SECTION_FORMATTER_SYSTEM_PROMPT_TEMPLATE,
)
from .struct import (
    Sections,
    Queries,
    SearchResult,
    Feedback,
    DatasetSchema,
)
import time
import os

llm = init_llm(model="gpt-4o-mini", provider="openai")
tavily_client = TavilyClient()

def schema_generator_node(state: AgentState, config: RunnableConfig):
    dataset_schema_generator_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SCHEMA_GENERATION_PROMPT),
        HumanMessagePromptTemplate.from_template(
            template="""
            Topic: {topic}
            Outline: {outline}
            """
        ),
        MessagesPlaceholder(variable_name="messages")
    ])

    llm_with_schema_tool = llm.bind_tools(tools=[DatasetSchema], tool_choice="required")
    schema_generator_llm = dataset_schema_generator_system_prompt | llm_with_schema_tool

    result = schema_generator_llm.invoke(state)
    suggested_schema = DatasetSchema.model_validate(result.tool_calls[0]["args"])

    return {"schema": suggested_schema, "messages": [f"Generated schema: \n{[suggested_schema.generated_schema]}"]}

def human_feedback_on_schema_node(state: AgentState, config: RunnableConfig) -> Command[Literal["report_structure_planner", "schema_generator"]]:
    human_message = input("Please provide feedback on the report structure (type 'continue' to continue): ")
    schema = state.get("schema")
    if human_message == "continue":
        return Command(
            goto="report_structure_planner",
            update={"messages": [HumanMessage(content=human_message)], "schema": schema}
        )
    else:
        return Command(
            goto="schema_generator",
            update={"messages": [HumanMessage(content=human_message)]}
        )


def report_structure_planner_node(state: AgentState, config: RunnableConfig):
    report_structure_planner_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(REPORT_STRUCTURE_PLANNER_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(
            template="""
            Topic: {topic}
            Outline: {outline}
            """
        ),
        MessagesPlaceholder(variable_name="messages")
    ])

    report_structure_planner_llm = report_structure_planner_system_prompt | llm
    result = report_structure_planner_llm.invoke(state)

    return {"messages": [result]}

def human_feedback_node(state: AgentState, config: RunnableConfig)->Command[Literal["section_formatter", "report_structure_planner"]]:
    human_message = input("Please provide feedback on the report structure (type 'continue' to continue): ")
    report_structure = state.get("messages")[-1].content
    if human_message == "continue":
        return Command(
            goto="section_formatter",
            update={"messages": [HumanMessage(content=human_message)], "report_structure": report_structure}
        )
    else:
        return Command(
            goto="report_structure_planner",
            update={"messages": [HumanMessage(content=human_message)]}
        )


def section_formatter_node(state: AgentState, config: RunnableConfig) -> Command[Literal["research_agent"]]:
    section_formatter_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SECTION_FORMATTER_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="{report_structure}"),
    ])

    section_formatter_llm = section_formatter_system_prompt | llm.with_structured_output(Sections)
    result = section_formatter_llm.invoke(state)
    schema = state.get("schema")
    report_structure = state.get("report_structure")
    topic = state.get("topic")
    # return {"sections": result.sections}
    return Command(
        update={"sections": result.sections},
        goto=[
            Send(
                "research_agent",
                {
                    "topic": topic,
                    "section": s,
                    "schema": schema,
                    "report_structure": report_structure,
                }
            ) for s in result.sections
        ]
    )

def section_knowledge_node(state: ResearchState, config: RunnableConfig):
    section_knowledge_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SECTION_KNOWLEDGE_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="{section}"),
    ])

    section_knowledge_llm = section_knowledge_system_prompt | llm
    result = section_knowledge_llm.invoke(state)
    return {"knowledge": result.content}


def query_generator_node(state: ResearchState, config: RunnableConfig):
    query_generator_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(QUERY_GENERATOR_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="Section: {section}\nPrevious Queries: {searched_queries}\nReflection Feedback: {reflection_feedback}"),
    ])

    query_generator_llm = query_generator_system_prompt | llm.with_structured_output(Queries)
    state.setdefault("reflection_feedback", "")
    state.setdefault("searched_queries", [])
    configurable = config.get("configurable")

    input_data = {
        **state,
        **configurable  # includes max_queries, search_depth, etc.
    }

    result = query_generator_llm.invoke(input_data, configurable)
    return {"generated_queries": result.queries, "searched_queries": result.queries}

def tavily_search_node(state: ResearchState, config: RunnableConfig):
    queries = state["generated_queries"]
    configurable = config.get("configurable")
    search_results = []
    for query in queries:
        raw_content = []
        response = tavily_client.search(query=query.query, max_results=configurable.get("search_depth"), include_raw_content=True)
        for result in response["results"]:
            raw_content.append(result['content'])
        search_results.append(SearchResult(query=query, raw_content=raw_content))
    return {"search_results": search_results}

def result_accumulator_node(state: ResearchState, config: RunnableConfig):
    result_accumulator_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(RESULT_ACCUMULATOR_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="{search_results}"),
    ])

    result_accumulator_llm = result_accumulator_system_prompt | llm
    result = result_accumulator_llm.invoke(state)
    return {"accumulated_content": result.content}


def reflection_feedback_node(state: ResearchState, config: RunnableConfig) -> Command[Literal["final_section_formatter", "query_generator"]]:
    reflection_feedback_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(REFLECTION_FEEDBACK_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="Section: {section}\nAccumulated Content: {accumulated_content}"),
    ])

    reflection_feedback_llm = reflection_feedback_system_prompt | llm.with_structured_output(Feedback)
    reflection_count = state.get("reflection_count", 0)
    configurable = config.get("configurable")
    result = reflection_feedback_llm.invoke(state)
    feedback = result.feedback
    if (feedback == True) or (feedback.lower() == "true") or (reflection_count < configurable.get("num_reflections")):
        return Command(
            update={"reflection_feedback": feedback},
            goto="final_section_formatter"
        )
    else:
        return Command(
            update={"reflection_feedback": feedback, "reflection_count": reflection_count + 1},
            goto="query_generator"
        )


def final_section_formatter_node(state: ResearchState, config: RunnableConfig):
    final_section_formatter_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(FINAL_SECTION_FORMATTER_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="Internal Knowledge: {knowledge}\nSearch Result content: {accumulated_content}"),
    ])

    final_section_formatter_llm = final_section_formatter_system_prompt | llm
    result = final_section_formatter_llm.invoke(state)
    return {"final_section_content": result.content}


def final_section_dataset_generator_node(state: ResearchState, config: RunnableConfig, max_retries: int = 3, base_wait: float = 2.0):
    schema = state.get("schema")
    max_rows = config.get("configurable").get("max_rows_from_each_section")
    FINAL_SECTION_DATASET_GENERATION_PROMPT = process_datagen_prompt(schema.generated_schema, int(max_rows))

    final_section_dataset_generator_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=FINAL_SECTION_DATASET_GENERATION_PROMPT),
        HumanMessagePromptTemplate.from_template(template="Report Structure: {report_structure}\nSection Contents: {final_section_content}"),
    ])
    final_dataset_generator_llm = final_section_dataset_generator_prompt | llm

    for attempt in range(max_retries):
        try:
            result = final_dataset_generator_llm.invoke(state)
            raw_text = result.content

            # Clean up markdown wrapping
            if raw_text.startswith("```json"):
                raw_text = raw_text[len("```json"):].lstrip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[len("```"):].lstrip()
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].rstrip()

            parsed_json = json.loads(raw_text)
            final_package = {"dataset": parsed_json}
            validated = DatasetRecords(**final_package)

            return {"final_section_dataset": validated.dataset}

        except json.JSONDecodeError as e:
            print(f"[JSON Parse Error] {e}")
            return {"final_section_dataset": [], "error": "JSONDecodeError"}

        except ValidationError as e:
            print(f"[Pydantic Validation Error] {e}")
            return {"final_section_dataset": [], "error": "ValidationError"}

        except RateLimitError:
            wait_time = base_wait * (2 ** attempt)
            print(f"[Rate Limit] Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)

        except OpenAIError as e:
            print(f"[OpenAI Error] {e}")
            wait_time = base_wait * (2 ** attempt)
            time.sleep(wait_time)

        except Exception as e:
            print(f"[Unexpected Error] {e}")
            return {"final_section_dataset": [], "error": str(e)}

    return {"final_section_dataset": [], "error": "Max retries exceeded"}

def final_dataset_aggregator_node(state: AgentState, config: RunnableConfig):
    dataset = []
    section_datasets = state.get("final_section_dataset")

    for section_dataset in section_datasets:
            dataset.append(section_dataset)
    
    return {"final_dataset": dataset}