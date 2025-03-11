import asyncio
import json
import os
import time
import traceback
from typing import TypedDict, List, Optional, Dict
from langgraph.graph import StateGraph, END
from langchain.agents import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools import DuckDuckGoSearchResults, TavilySearchResults, SearxSearchResults
from langchain_community.document_loaders import PlaywrightURLLoader

# Initialize models
# deepseek_llm = ChatDeepSeek(
#     model_name=os.environ.get("DEEPSEEK_V3_MODEL", "deepseek-chat"),
#     api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
#     api_key=os.environ.get("DEEPSEEK_API_KEY", "your-api-key"),
#     temperature=0,
# )

deepseek_llm = ChatDeepSeek(
    model_name=os.environ.get("VOL_DEEPSEEK_V3_MODEL", "deepseek-chat"),
    api_base=os.environ.get("VOL_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    api_key=os.environ.get("VOL_DEEPSEEK_API_KEY", "your-api-key"),
    temperature=0,
)

# deepseek_reasoner_llm = ChatDeepSeek(
#     model_name=os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
#     api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
#     api_key=os.environ.get("DEEPSEEK_API_KEY", "your-api-key"),
#     temperature=0,
# )

deepseek_reasoner_llm = ChatDeepSeek(
    model_name=os.environ.get("VOL_DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
    api_base=os.environ.get("VOL_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    api_key=os.environ.get("VOL_DEEPSEEK_API_KEY", "your-api-key"),
    temperature=0,
)

volcengine_deepseek_llm = ChatOpenAI(
    model_name=os.environ.get("VOL_DEEPSEEK_V3_MODEL", "deepseek-chat"),
    openai_api_base=os.environ.get("VOL_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    openai_api_key=os.environ.get("VOL_DEEPSEEK_API_KEY", "your-api-key"),
    temperature=0,
)

searchXNG = SearxSearchWrapper(
    searx_host=os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080"),
)

# result = searchXNG.results("test", num_results=5, engines=["bing", "duckduckgo", "google", "yahoo", "wikipedia", "wikidata"])
# print(result)

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
    chain = prompt | deepseek_reasoner_llm
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

async def route_task(task: dict) -> str:
    prompt = PromptTemplate.from_template("""
    Does the following task require external tools like web search? Answer only True or False.
    Notice, the decomposer says whether this task should use web search is {may_needs_tool}.
    Task Detail: {task}
    """)
    chain = prompt | deepseek_llm
    print(f"Routing task: {task}")
    # Add timeout for the task
    async def invoke_chain(current_chain):
        try:
            response = await asyncio.wait_for(current_chain.ainvoke({"task": task["description"], "may_needs_tool": task["may_needs_tool"]}), timeout=300)
            response = response.content.strip().lower().capitalize()
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError("Timeout error")
        return response

    response = None
    for i in range(3):
        if i == 2:
            current_chain = prompt | volcengine_deepseek_llm
        else:
            current_chain = chain
        try:
            response = await invoke_chain(current_chain=current_chain)
        except Exception as e:
            print(f"Error in routing task: {traceback.format_exc()}")

            if i == 2:
                raise asyncio.TimeoutError("Timeout error")
            time.sleep(30)
            continue
        break

    need_tool = ""
    if response.find("True") != -1:
        need_tool = "True"
    elif response.find("False") != -1:
        need_tool = "False"
    if need_tool not in ["True", "False"]:
        raise ValueError(f"Invalid response: {response}")

    needs_external_tool = "[x]"
    if response == "True":
        needs_external_tool = "[✔]"
    print(f"Task: {task['name']} needs external tools {needs_external_tool}")

    return response

async def web_search(task: dict, task_results: Dict[str, str]) -> (List[str], str):
    print(f"Searching the web for: {task['description']}")

    # Decompose the task description into multiple search engine queries
    prompt = PromptTemplate.from_template("""
    Decompose the following task description into multiple search engine queries. Only return the queries as a comma-separated string.
    The queries should be separated by a comma, do not add any serial numbers or newline characters, just the queries separated by commas.
    The queries should be based on the task description.
    The queries should be short and concise.
    The queries should be designed specifically for web search, especially for search engines like DuckDuckGo, Google, SearXNG and Tavily.
    Task Description: {description}
    Your current knowledge and requirements: {knowledge}
    """)

    chain = prompt | deepseek_llm
    response = chain.invoke({"description": task["description"], "knowledge": task_results}).content

    queries = response.split(",")

    print(f"Decomposed queries: {queries}")


    async def search_duckduckgo(query: str):
        # return await DuckDuckGoSearchResults(max_results=5).ainvoke(query)
        return ""

    async def search_tavily(query: str):
        # return await TavilySearchResults(max_results=3).ainvoke(query)
        return ""

    async def search_searx(query: str):
        return await searchXNG.aresults(query, num_results=10, engines=["bing", "duckduckgo", "google", "yahoo", "wikipedia", "wikidata"])
        return ""

    tasks = []
    tavily_content_list = []
    results_links = []
    duckduckgo_results_links = []
    tavily_summary = ""
    searxng_results_links = []
    searxng_content_list = []
    content_result = ""

    for query in queries:
        async def gather_search_results(query: str):
            await asyncio.sleep(10)
            for i in range(3):
                try:
                    # Run both search tasks concurrently
                    duckduckgo_results, new_tavily_results, searxng_results = await asyncio.gather(search_duckduckgo(query), search_tavily(query), search_searx(query))
                    # Get all links from the duckduckgo_results string, start with 'link: ', end with ',' or at the end of the string
                    new_duckduckgo_results_links = [link.split(",")[0] for link in duckduckgo_results.split("link: ")[1:]]
                    duckduckgo_results_links.extend(new_duckduckgo_results_links)
                    new_searxng_results_links = [ item["link"] for item in searxng_results ]
                    # searxng_results_links.extend(new_searxng_results_links)
                    searxng_content_list.extend([item["snippet"] for item in searxng_results])

                    new_tavily_contents = "\n".join(new_tavily_results)

                    tavily_content_list.append(new_tavily_contents)
                except Exception as e:
                    print(f"Error in gathering search results: {traceback.format_exc()}")

                    if i == 2:
                        raise e
                    time.sleep(30)
                    continue

        tasks.append(gather_search_results(query))

    await asyncio.gather(*tasks)

    tavily_contents = ""
    for content in tavily_content_list:
        tavily_contents += content + "\n"

    searxng_contents = ""
    for content in searxng_content_list:
        searxng_contents += content + "\n"

    content_result = tavily_contents + searxng_contents

    prompt = PromptTemplate.from_template("""
    Extract the main content from the web search results.
    Do not miss any important information.
    Merge same or similar information.
    Remove any irrelevant information.
    Current task: {task}
    Web search result: {search_result}
    """)

    chain = prompt | deepseek_llm
    content_summary = chain.invoke({"task": task["description"], "search_result": content_result}).content

    print(f"Search content results: {content_summary}")
    print(f"Tavily content: {tavily_contents}")
    print(f"SearXNG content: {searxng_contents}")
    print(f"Duckduckgo links: {duckduckgo_results_links}")
    print(f"SearXNG links: {searxng_results_links}")

    results_links.extend(duckduckgo_results_links)
    results_links.extend(searxng_results_links)
    return results_links, content_summary

async def parse_search_links(search_links: List[str], task_description: str, current_search_result: str) -> str:
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
    for index, document in enumerate(data):
        async def parse_document(document: str):
            response = None
            for i in range(3):
                if i == 2:
                    current_chain = prompt | volcengine_deepseek_llm
                else:
                    current_chain = chain
                try:
                    response = await asyncio.wait_for(current_chain.ainvoke({"document": document, "task": task_description}), timeout=300)
                    break
                except Exception as e:
                    print(f"Timeout error, for document index: {index}, error: {traceback.format_exc()}")
                    if i == 2:
                        raise asyncio.TimeoutError("Timeout error")
                    time.sleep(30)
                    continue
            result = response.content
            search_results.append(result)

        async_tasks.append(parse_document(document))

    await asyncio.gather(*async_tasks)

    search_result = current_search_result + "\n"
    for result in search_results:
        search_result += result + "\n"

    prompt = PromptTemplate.from_template("""
        Extract the main content from the web search results.
        Do not miss any important information.
        Merge same or similar information.
        Remove any irrelevant information.
        Current task: {task}
        Web search results: {search_result}
        """)

    final_search_result = ""
    chain = prompt | deepseek_llm
    for i in range(3):
        if i == 2:
            current_chain = prompt | volcengine_deepseek_llm
        else:
            current_chain = chain
        try:
            final_search_result = current_chain.invoke({"task": task_description, "search_result": search_result}).content
            break
        except Exception as e:
            print(f"Error in parsing search links: {traceback.format_exc()}")

            if i == 2:
                raise asyncio.TimeoutError("Timeout error")
            time.sleep(30)
            continue

    print(f"Search result: {final_search_result}")
    return final_search_result

async def execute_task(task: dict, task_results: Dict[str, str]) -> str:
    prompt = PromptTemplate.from_template("""
    Complete task: {description}
    Your current knowledge: {knowledge}
    """)
    chain = prompt | deepseek_llm
    print(f"Executing task: {task}, with current knowledge: {task_results}")
    for i in range(3):
        if i == 2:
            current_chain = prompt | volcengine_deepseek_llm
        else:
            current_chain = chain
        try:
            response = await asyncio.wait_for(current_chain.ainvoke({"description": task["description"], "knowledge": task_results}), timeout=300)
        except Exception as e:
            print(f"Error in executing task: {traceback.format_exc()}")

            if i == 2:
                raise asyncio.TimeoutError("Timeout error")
            time.sleep(30)
            continue
        return response.content

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

async def router_node(state: AgentState) -> dict:
    if not state["current_task"]:
        return {}
    needs_tool = await route_task(state["current_task"])
    return {"needs_tool": needs_tool}

async def search_agent_node(state: AgentState) -> dict:
    if not state["current_task"]:
        return {}
    search_links, tavily_content = await web_search(state["current_task"], state["task_results"])
    return {"search_links": search_links, "task_result": tavily_content}

async def search_result_parser_node(state: AgentState) -> dict:
    if not state["search_links"] or not state["current_task"]:
        return {}
    search_result = await parse_search_links(state["search_links"], state["current_task"]["description"], state["task_result"])
    return {"task_result": search_result}

async def task_executor_node(state: AgentState) -> dict:
    if not state["current_task"]:
        return {}
    result = await execute_task(state["current_task"], state["task_results"])
    return {"task_result": result}

def state_updater_node(state: AgentState) -> dict:
    remaining_tasks = state["tasks"][1:] if state["tasks"] else []

    # Print all the tasks and their status
    print("\n")
    print("Current task status:")
    for task in state["all_tasks"]:
        completed = "[ ]" if task in remaining_tasks else "[✔]"
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
    return await agent.ainvoke(initial_state, config={"recursion_limit": 100})

your_task = """
I'm currently working on a project to build a evm compatible side-chain for VSYS chain.
Find and compare side-chain and bridging solutions for VSYS chain.
The side-chain needs to be compatible with EVM and support smart contracts
The side-chain can be used for commercial
The side-chain needs to run as a private chain, cannot connect to current main/test blockchain network.
The bridging solution should be secure and efficient. VSYS chain do not have any side-chain or bridging solution yet, so current bridging solutions will not work, we have to start from scratch.
"""

result = asyncio.run(run_agent(your_task))

file_name = "vsys-side-chain.md"
file_path = os.path.join("results", file_name)
# Write the final report to a file
with open(file_path, "w") as f:
    f.write(result["report"])

print(f"Report {file_path} generated successfully!")