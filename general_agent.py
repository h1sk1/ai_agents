import asyncio
import json
import os
from typing import TypedDict, List, Optional, Dict
from langgraph.graph import StateGraph, END
from langchain.agents import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PlaywrightURLLoader

# Initialize models
deepseek_llm = ChatDeepSeek(
    model_name=os.environ.get("DEEPSEEK_V3_MODEL", "deepseek-chat"),
    api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    api_key=os.environ.get("DEEPSEEK_API_KEY", "your-api-key"),
    temperature=0,
)

deepseek_reasoner_llm = ChatDeepSeek(
    model_name=os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
    api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    api_key=os.environ.get("DEEPSEEK_API_KEY", "your-api-key"),
    temperature=0,
)



class AgentState(TypedDict):
    user_input: str
    all_tasks: List[dict]
    tasks: List[dict]
    task_results: Dict[str, str]
    current_task: Optional[dict]
    needs_tool: Optional[str]  # "True" or "False"
    task_result: Optional[str]  # Temporary result of current task
    search_links: Optional[List[str]]  # Search links
    report: Optional[str]      # Final report

def decompose_tasks(user_input: str) -> List[dict]:
    prompt = PromptTemplate.from_template("""
    Decompose the following user input into a list of sequential tasks. Each task should have a name, description, and a boolean indicating if it needs an external tool.
    The attribute should be named 'name', 'description', and 'may_needs_tool' respectively.
    The attribute 'needs_tool' should be a string value with capital 'True' or 'False' according to whether the task requires an external tool.
    Return the result as a JSON array.
    Input: {input}
    """)
    chain = prompt | deepseek_llm
    response = chain.invoke({"input": user_input}).content
    # Clean up the response
    # Find first occurrence of [ and remove everything before it
    response = response[response.index("["):]
    # Find last occurrence of ] and remove everything after it
    response = response[:response.rindex("]") + 1]

    tasks = json.loads(response) # Parse JSON string to list of dicts

    print(f"Decomposed tasks:")
    for task in tasks:
        print(f"Task: {task['name']}, Description: {task['description']}, May Needs tool: {task['may_needs_tool']}")
    print("\n")
    return tasks

def route_task(task: dict) -> str:
    prompt = PromptTemplate.from_template("""
    Does the following task require external tools like web search? Answer only True or False.
    Notice, the decomposer says whether this task should use web search is {may_needs_tool}.
    Task Detail: {task}
    """)
    chain = prompt | deepseek_llm
    print(f"Routing task: {task}")
    response = chain.invoke({"task": task["description"], "may_needs_tool": task["may_needs_tool"]}).content.strip().upper()
    # Parse response to True or False, small case and then capitalize
    response = response.lower().capitalize()

    if response not in ["True", "False"]:
        raise ValueError(f"Invalid response: {response}")

    needs_external_tool = "[ ]"
    if response == "True":
        needs_external_tool = "[X]"
    print(f"Task {task['name']} needs external tools {needs_external_tool}")

    return response

def web_search(task: dict) -> List[str]:
    print(f"Searching the web for: {task['description']}")
    duckduckgo_results = DuckDuckGoSearchResults().invoke(task["description"])
    # Get all links from the duckduckgo_results string, start with 'link: ', end with ',' or at the end of the string
    duckduckgo_results_links = [link.split(",")[0] for link in duckduckgo_results.split("link: ")[1:]]
    print(f"Web search results: {duckduckgo_results}")
    print(f"links: {duckduckgo_results_links}")
    return duckduckgo_results_links


async def parse_search_links(search_links: List[str], task_description: str) -> str:
    prompt = PromptTemplate.from_template("""
    Extract the main content from documents found in the search results.
    
    Current task: {task}
    
    Search html result parsed document by Playwright: {document}
    """
    )
    loader = PlaywrightURLLoader(urls=search_links, remove_selectors=["header", "footer"])
    data = await loader.aload()
    chain = prompt | deepseek_llm

    search_results = []

    async_tasks = []
    for document in data:
        async def parse_document(document: str):
            response = await chain.ainvoke({"document": document, "task": task_description})
            result = response.content
            search_results.append(result)

        async_tasks.append(parse_document(document))

    await asyncio.gather(*async_tasks)

    search_result = "\n".join(search_results)
    print(f"Search result: {search_result}")
    return search_result

def execute_task(task: dict, task_results: Dict[str, str]) -> str:
    prompt = PromptTemplate.from_template("""
    Complete task: {description}
    Your current knowledge: {knowledge}
    """)
    chain = prompt | deepseek_llm
    print(f"Executing task: {task}, with current knowledge: {task_results}")
    return chain.invoke({"description": task["description"], "knowledge": task_results}).content

def generate_report(results: Dict[str, str]) -> str:
    prompt = PromptTemplate.from_template("Create a detailed report from:\n{results}")
    chain = prompt | deepseek_reasoner_llm
    print(f"Generating report from results: {results}")
    return chain.invoke({"results": results}).content

def decomposer_node(state: AgentState) -> dict:
    tasks = decompose_tasks(state["user_input"])
    return {
        "all_tasks": tasks,
        "tasks": tasks,
        "current_task": tasks[0] if tasks else None
    }

def router_node(state: AgentState) -> dict:
    if not state["current_task"]:
        return {}
    needs_tool = route_task(state["current_task"])
    return {"needs_tool": needs_tool}

def search_agent_node(state: AgentState) -> dict:
    if not state["current_task"]:
        return {}
    search_links = web_search(state["current_task"])
    return {"search_links": search_links}

async def search_result_parser_node(state: AgentState) -> dict:
    if not state["search_links"] or not state["current_task"]:
        return {}
    search_result = await parse_search_links(state["search_links"], state["current_task"]["description"])
    return {"task_result": search_result}

def task_executor_node(state: AgentState) -> dict:
    if not state["current_task"]:
        return {}
    result = execute_task(state["current_task"], state["task_results"])
    return {"task_result": result}

def state_updater_node(state: AgentState) -> dict:
    remaining_tasks = state["tasks"][1:] if state["tasks"] else []

    # Print all the tasks and their status
    print("\n")
    print("Current task status:")
    for task in state["all_tasks"]:
        completed = "[ ]" if task in remaining_tasks else "[X]"
        print(f"Task: {task['name']}, Completed: {completed}")
    print("\n")

    if not state["current_task"] or not state["task_result"]:
        return {}
    new_results = state["task_results"].copy()
    new_results[state["current_task"]["name"]] = state["task_result"]
    next_task = remaining_tasks[0] if remaining_tasks else None
    return {
        "task_results": new_results,
        "tasks": remaining_tasks,
        "current_task": next_task,
        "task_result": None,  # Clear temporary result
        "search_links": None  # Clear search links
    }

def report_generator_node(state: AgentState) -> dict:
    if not state["task_results"]:
        return {"report": "No results to report."}
    report = generate_report(state["task_results"])
    return {"report": report}

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("decomposer", decomposer_node)
workflow.add_node("router", router_node)
workflow.add_node("search_agent", search_agent_node)
workflow.add_node("search_result_parser", search_result_parser_node)
workflow.add_node("task_executor", task_executor_node)
workflow.add_node("state_updater", state_updater_node)
workflow.add_node("report_generator", report_generator_node)

# Define edges
workflow.set_entry_point("decomposer")
workflow.add_edge("decomposer", "router")
workflow.add_conditional_edges(
    "router",
    lambda state: "search_agent" if state["needs_tool"] == "True" else "task_executor",
    {"search_agent": "search_agent", "task_executor": "task_executor"}
)
workflow.add_edge("search_agent", "search_result_parser")
workflow.add_edge("search_result_parser", "state_updater")
workflow.add_edge("task_executor", "state_updater")
workflow.add_conditional_edges(
    "state_updater",
    lambda state: "router" if state["tasks"] else "report_generator",
    {"router": "router", "report_generator": "report_generator"}
)
workflow.add_edge("report_generator", END)

# Compile the agent
agent = workflow.compile()

async def run_agent(your_task: str = None):
    initial_state = {
        "user_input": your_task,
        "tasks": [],
        "task_results": {},
        "current_task": None,
        "needs_tool": None,
        "task_result": None,
        "report": None
    }
    return await agent.ainvoke(initial_state)

your_task = "Search for the latest research articles on ‘quantum computing applications in cryptography’ published in 2025. Summarize the key findings of the top three articles."

result = asyncio.run(run_agent(your_task))

# Write the final report to a file
with open("results/report.md", "w") as f:
    print("Writing report to file...")
    f.write(result["report"])

print("Report generated successfully!")