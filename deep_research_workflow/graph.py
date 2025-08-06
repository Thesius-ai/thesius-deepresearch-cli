from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState, ResearchState
from .struct import SectionOutput
from .nodes import (
    report_structure_planner_node,
    human_feedback_node,
    section_formatter_node,
    section_knowledge_node,
    query_generator_node,
    tavily_search_node,
    result_accumulator_node,
    reflection_feedback_node,
    final_section_formatter_node,
    final_section_dataset_generator_node,
    schema_generator_node,
    human_feedback_on_schema_node,
    final_dataset_aggregator_node
)


research_builder = StateGraph(ResearchState, output=SectionOutput)

research_builder.add_node("section_knowledge", section_knowledge_node)
research_builder.add_node("query_generator", query_generator_node)
research_builder.add_node("tavily_search", tavily_search_node)
research_builder.add_node("result_accumulator", result_accumulator_node)
research_builder.add_node("reflection", reflection_feedback_node)
research_builder.add_node("final_section_formatter", final_section_formatter_node)
research_builder.add_node("final_section_dataset_generator", final_section_dataset_generator_node)

research_builder.add_edge(START, "section_knowledge")
research_builder.add_edge("section_knowledge", "query_generator")
research_builder.add_edge("query_generator", "tavily_search")
research_builder.add_edge("tavily_search", "result_accumulator")
research_builder.add_edge("result_accumulator", "reflection")
research_builder.add_edge("final_section_formatter", "final_section_dataset_generator")
research_builder.add_edge("final_section_dataset_generator", END)


memory_saver = MemorySaver()

builder = StateGraph(AgentState)

builder.add_node("schema_generator", schema_generator_node)
builder.add_node("human_feedback_on_schema", human_feedback_on_schema_node)
builder.add_node("report_structure_planner", report_structure_planner_node)
builder.add_node("human_feedback_report_structure", human_feedback_node)
builder.add_node("section_formatter", section_formatter_node)
builder.add_node("research_agent", research_builder.compile())
builder.add_node("final_dataset_aggregator", final_dataset_aggregator_node)

builder.set_entry_point("schema_generator")
builder.add_edge("schema_generator", "human_feedback_on_schema")
builder.add_edge("report_structure_planner", "human_feedback_report_structure")
builder.add_edge("research_agent", "final_dataset_aggregator")
builder.add_edge("final_dataset_aggregator", END)

agent_graph = builder.compile(checkpointer=memory_saver)