import json
import uuid
import os
from datetime import datetime
from deep_research_workflow.graph import agent_graph as graph

from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

from pyfiglet import Figlet

console = Console()

def render_banner(title: str = "Datalore.ai", subtitle: str = ""):
    figlet = Figlet(font="banner3-D", width=200)
    ascii_art = figlet.renderText(title)

    panel = Panel.fit(
        f"[bold cyan]{ascii_art}[/bold cyan]\n[green]{subtitle}[/green]",
        border_style="bright_magenta",
        padding=(1, 4),
        title="[bold yellow]WELCOME[/bold yellow]",
    )

    console.print(panel)

def print_section(text, content=None):
    width = 100
    title = f"{text}"
    padding = (width - len(title)) // 2

    if padding > 0:
        bar = "#" * width
        line = "#" + " " * (padding-1) + title + " " * (padding-1) + "#"

        if len(line) < width:
            line += " "
    else:
        bar = "#" * len(title)
        line = title

    console.print(f"\n[cyan]{bar}[/cyan]")
    console.print(f"[bold cyan]{line}[/bold cyan]")
    console.print(f"[cyan]{bar}[/cyan]\n")
    if content:
        print(content)


def render_schema(schema_obj):
    if not hasattr(schema_obj, 'generated_schema'):
        print_section("SCHEMA GENERATION", str(schema_obj))
        return

    table = Table(title=None, box=box.ASCII, header_style="bold magenta")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Description", style="white")

    for field in schema_obj.generated_schema:
        field_type = str(field.type.value if hasattr(field.type, "value") else field.type)
        table.add_row(field.key, field_type, field.description)

    console.print("\n")
    print_section("SCHEMA GENERATION")
    console.print(table)

def render_section_formatting(section_data, width=100):
    sections = section_data.get("sections", [])
    if not sections:
        print_section("SECTION FORMATTING AND GENERATION", str(section_data))
        return

    print_section("SECTION FORMATTING AND GENERATION")

    for sec in sections:
        section_name = getattr(sec, "section_name", "Unnamed Section")
        sub_sections = getattr(sec, "sub_sections", [])

        body = "\n\n".join([f"- {sub}" for sub in sub_sections])
        panel = Panel(
            body,
            title=f"[bold magenta]{section_name}[/bold magenta]",
            border_style="cyan",
            padding=(1, 2),
            width=width
        )
        console.print(panel)

def save_json(data, directory="output_files"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"final_dataset_output_{timestamp}.json"
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Saved final dataset to:[/green] {filepath}")

def main():
    render_banner("Datalore.ai", "AI-powered Deep Research & Dataset Engine")

    topic = Prompt.ask("[bold yellow]Enter your topic[/bold yellow]").strip()
    outline = Prompt.ask("[bold yellow]Enter your outline or goal[/bold yellow]").strip()

    if not topic or not outline:
        console.print("[bold red]Error:[/bold red] Topic and outline cannot be empty.")
        return

    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "max_queries": 2,
            "search_depth": 1,
            "num_reflections": 2,
            "max_rows_from_each_section": 5
        }
    }

    console.print(Panel.fit(f"[bold]Topic:[/bold] {topic}\n[bold]Outline:[/bold] {outline}", title=None, border_style="cyan"))

    try:
        for event in graph.stream(
            {"topic": topic, "outline": outline},
            config=thread,
        ):
            if "schema_generator" in event:
                render_schema(event["schema_generator"]["schema"])

            elif "report_structure_planner" in event:
                print_section("REPORT STRUCTURE PLAN", event["report_structure_planner"]["messages"][-1].content)

            elif "section_formatter" in event:
                render_section_formatting(event["section_formatter"])

            elif "research_agent" in event:
                console.print(Panel.fit(f"section dataset generation completed with {len(event['research_agent']['final_section_dataset'])} dataset rows generated", title="RESEARCH AND SECTION-WISE DATASET GENERATION", border_style="green", width=100))

            elif "final_dataset_aggregator" in event:
                print_section("FINAL DATASET AGGREGATION")
                save_json(event["final_dataset_aggregator"])

            elif "human_feedback_on_schema" in event:
                print_section("YOUR FEEDBACK ON SCHEMA", event["human_feedback_on_schema"]["messages"][-1].content)

            elif "human_feedback_report_structure" in event:
                print_section("YOUR FEEDBACK ON REPORT STRUCTURE", event["human_feedback_report_structure"]["messages"][-1].content)

            else:
                console.print("[dim]Waiting for next event...[/dim]")

    except KeyboardInterrupt:
        console.print("\n[bold red]Execution stopped by user.[/bold red]")

if __name__ == "__main__":
    main()