import asyncio
import json
import os
import random
import subprocess
import sys
import time
import traceback
from itertools import count
from threading import Thread
from typing import TypedDict, List, Optional, Dict

from huggingface_hub.utils import capture_output
from langchain.agents.agent import RunnableAgent
from langchain.chains.question_answering.map_reduce_prompt import system_template
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableSerializable
from langgraph.graph import StateGraph, END
from langchain.agents import tool
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate
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

deepseek_reasoner_llm = ChatDeepSeek(
    model_name=os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
    api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    api_key=os.environ.get("DEEPSEEK_API_KEY", "your-api-key"),
    temperature=0,
)

# deepseek_reasoner_llm = ChatDeepSeek(
#     model_name=os.environ.get("VOL_DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
#     api_base=os.environ.get("VOL_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
#     api_key=os.environ.get("VOL_DEEPSEEK_API_KEY", "your-api-key"),
#     temperature=0,
# )

volcengine_deepseek_llm = ChatOpenAI(
    model_name=os.environ.get("VOL_DEEPSEEK_V3_MODEL", "deepseek-chat"),
    openai_api_base=os.environ.get("VOL_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    openai_api_key=os.environ.get("VOL_DEEPSEEK_API_KEY", "your-api-key"),
    temperature=0,
)

volcengine_deepseek_reasoner_llm = ChatOpenAI(
    model_name=os.environ.get("VOL_DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
    openai_api_base=os.environ.get("VOL_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    openai_api_key=os.environ.get("VOL_DEEPSEEK_API_KEY", "your-api-key"),
    temperature=0,
)

searchXNG = SearxSearchWrapper(
    searx_host=os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080"),
)

class AgentState(TypedDict):
    user_input: str
    user_task_summary: str
    all_tasks: List[dict]
    tasks: List[dict]
    task_results: Dict[str, str]
    current_task: Optional[dict]
    needs_tool: Optional[bool]  # True or False
    task_result: Optional[str]  # Temporary result of current task
    task_completed: Optional[bool]  # True or False
    task_retry_count: Optional[int] # Number of retries for current task
    search_links: Optional[List[str]]  # Search links
    report: Optional[str]      # Final report

def decompose_tasks(user_input: str) -> (str, List[dict]):
    system_message = SystemMessagePromptTemplate.from_template("""
    You are a professional task analyst.
    Your job is to analyze user input and give a summary of the task. And decompose the user's input into a list of sequential tasks.
    You need to study every detail of the user's input and decompose it into multiple tasks, so please be as detailed as possible.
    The format of your summary should be a JSON array, the details are as follows:
    1. The first object in the array should be the summary of the user's input. The object should only have a single key 'summary' with the value as the summary of the user's input.
        a. The attribute should be named 'summary'.
        b. The attribute 'summary' should be a string value.
    2. Then decompose the following user input into a list of sequential tasks. Each task should have a name, description, and a boolean indicating if it needs an external tool.
        a. The attribute should be named 'name', 'description', and 'may_needs_tool' respectively.
        b. The attribute 'needs_tool' should be a string value with capital 'True' or 'False' according to whether the task requires an external tool.
    Return the result as a JSON array.
    You do not have direct access to the internet, but you can use external tools like web search.
    """)

    current_time = time.strftime("%Y%m%d_%H%M%S")
    human_message = HumanMessagePromptTemplate.from_template(
    """
    Current time: {current_time}
    
    Input: {input}
    """
    )

    prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message
    ])

    chain = prompt | deepseek_reasoner_llm
    response = chain.invoke({"current_time": current_time, "input": user_input}).content
    # Clean up the response
    # Find first occurrence of [ and remove everything before it
    response = response[response.index("["):]
    # Find last occurrence of ] and remove everything after it
    response = response[:response.rindex("]") + 1]

    result = json.loads(response) # Parse JSON string to list of dicts

    # Check if the result format is correct
    # Check if the first object has a key 'summary'
    if "summary" not in result[0]:
        raise ValueError("Invalid task summary format")
    # Check if the decomposed tasks have the correct keys
    for task in result[1:]:
        if "name" not in task or "description" not in task or "may_needs_tool" not in task:
            raise ValueError("Invalid task format")

    user_task_summary = result[0]["summary"]
    tasks = result[1:]

    print(f"User task summary: {user_task_summary}\n\n")

    print(f"Decomposed tasks:")
    for task in tasks:
        print(f"Task: {task['name']}, Description: {task['description']}, May Needs tool: {task['may_needs_tool']}")
    print("\n")

    return user_task_summary, tasks

async def route_task(task: dict) -> bool:
    # current_time = time.strftime("%Y%m%d_%H%M%S")
    # prompt = PromptTemplate.from_template(
    # """
    # Current time: {current_time}
    # Does the following task require external tools like web search? Answer only True or False.
    # Notice, the decomposer says whether this task should use web search is {may_needs_tool}.
    # Task Detail: {task}
    # """)
    # chain = prompt | deepseek_llm
    print(f"Routing task: {task}")
    # # Add timeout for the task
    # async def invoke_chain(current_chain):
    #     try:
    #         response = await asyncio.wait_for(current_chain.ainvoke({"current_time": current_time, "task": task["description"], "may_needs_tool": task["may_needs_tool"]}), timeout=300)
    #         response = response.content.strip().lower().capitalize()
    #     except asyncio.TimeoutError:
    #         raise asyncio.TimeoutError("Timeout error")
    #     return response
    #
    # response = None
    # for i in range(3):
    #     if i == 2:
    #         current_chain = prompt | volcengine_deepseek_llm
    #     else:
    #         current_chain = chain
    #     try:
    #         response = await invoke_chain(current_chain=current_chain)
    #     except Exception as e:
    #         print(f"Error in routing task: {traceback.format_exc()}")
    #
    #         if i == 2:
    #             raise asyncio.TimeoutError("Timeout error")
    #         time.sleep(30)
    #         continue
    #     break

    response = task["may_needs_tool"]

    if response.find("True") != -1:
        need_tool = True
    elif response.find("False") != -1:
        need_tool = False
    else:
        raise ValueError(f"Invalid response: {response}")

    needs_external_tool = "[x]"
    if need_tool:
        needs_external_tool = "[✔]"
    print(f"Task: {task['name']} needs external tools {needs_external_tool}")

    return need_tool

async def web_search(task: dict, task_result: str) -> (List[str], str):
    print(f"Searching the web for: {task['description']}")

    current_time = time.strftime("%Y%m%d_%H%M%S")
    system_message = SystemMessagePromptTemplate.from_template(
    """
    You are a professional search engine expert.
    You need to study every detail of the task with the help of web search.
    Decompose the following task description into multiple search engine queries, based on the task description and your current knowledge.
    Only try to find things that you don't already know and cannot understand from your current knowledge.
    Only return the queries as a comma-separated string.
    The queries should be separated by a comma, do not add any serial numbers or newline characters, just the queries separated by commas.
    The queries should be based on the task description.
    The queries should be short and concise.
    The queries should be designed specifically for web search, especially for search engines like DuckDuckGo, Google, SearXNG and Tavily.
    The queries should first consider to be in English.
    The queries can also be in Chinese if the task description contains queries asking for Chinese information.
    The queries should be less than 10 words each, so you need to be concise, super specific and super precise.
    The queries should not be more than 10 queries, so you should contain all the information you want to search in 10 queries.
    """
    )

    human_message = HumanMessagePromptTemplate.from_template(
    """
    Current time: {current_time}
    Task Description:
    {description}
    Your current knowledge:
    {knowledge}
    """
    )

    # Decompose the task description into multiple search engine queries
    prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message
    ])

    chain = prompt | volcengine_deepseek_reasoner_llm
    response = chain.invoke({"current_time": current_time, "description": task["description"], "knowledge": task_result}).content

    queries = response.strip().split(",")

    print(f"Decomposed queries: {queries}")


    async def search_duckduckgo(query: str):
        # return await DuckDuckGoSearchResults(max_results=5).ainvoke(query)
        return ""

    async def search_tavily(query: str):
        # return await TavilySearchResults(max_results=3).ainvoke(query)
        return ""

    async def search_searx(query: str):
        # search_engine_pool = ["bing", "google", "yahoo", "qwant"]
        # random.seed(time.time())
        # random.shuffle(search_engine_pool)
        # search_engine_list = search_engine_pool[:2]
        # search_engine_list.extend(["wikipedia", "wikidata"])
        search_engine_list = ["bing", "google", "yahoo", "wikipedia", "wikidata"]
        return await searchXNG.aresults(query, num_results=3, engines=search_engine_list)

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
            random.seed(time.time())
            random_num = random.randint(10, 20)
            await asyncio.sleep(random_num)
            for i in range(3):
                try:
                    # Run both search tasks concurrently
                    duckduckgo_results, new_tavily_results, searxng_results = await asyncio.gather(search_duckduckgo(query), search_tavily(query), search_searx(query))
                    # Get all links from the duckduckgo_results string, start with 'link: ', end with ',' or at the end of the string
                    new_duckduckgo_results_links = [link.split(",")[0] for link in duckduckgo_results.split("link: ")[1:]]
                    duckduckgo_results_links.extend(new_duckduckgo_results_links)
                    new_searxng_results_links = [ item["link"] for item in searxng_results if "link" in item ][0:3]
                    searxng_results_links.extend(new_searxng_results_links)
                    # searxng_content_list.extend([item["snippet"] for item in searxng_results])

                    # new_tavily_contents = "\n".join(new_tavily_results)
                    #
                    # tavily_content_list.append(new_tavily_contents)
                    break
                except Exception as e:
                    print(f"Error in gathering search results: {traceback.format_exc()}")

                    if i == 2:
                        raise e
                    time.sleep(5)
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

    content_summary = ""
    # chain = prompt | deepseek_llm
    # content_summary = chain.invoke({"task": task["description"], "search_result": content_result}).content

    print(f"Search content results: {content_summary}")
    print(f"Tavily content: {tavily_contents}")
    print(f"SearXNG content: {searxng_contents}")
    print(f"Duckduckgo links: {duckduckgo_results_links}")
    print(f"SearXNG links: {searxng_results_links}")

    results_links.extend(duckduckgo_results_links)
    results_links.extend(searxng_results_links)
    return results_links, content_summary

async def parse_search_links(search_links: List[str], task_description: str, current_search_result: str) -> str:
    prompt = PromptTemplate.from_template(
    """
    Current time: {current_time}
    Extract the main content from documents found in the search results.
    Summarize the content and remove any irrelevant information.
    Use pithy and concise language
    Do not miss any important information.

    Current task: {task}

    Search html result parsed document by Playwright: {document}
    """
    )

    print(f"Parsing {len(search_links)} search links...")

    # Call scrapy from command line to get the search results
    current_time = time.strftime("%Y%m%d_%H%M%S")
    scrapy_dir = os.path.join(os.path.dirname(__file__), "webcrawler")

    # Divide the search links into batches of 5
    search_links_batches = [search_links[i:i + 5] for i in range(0, len(search_links), 5)]


    # Call scrapy from command line to get the search results
    current_time = time.strftime("%Y%m%d_%H%M%S")
    scrapy_dir = os.path.join(os.path.dirname(__file__), "webcrawler")

    # Divide the search links into batches of 5
    search_links_batches = [search_links[i:i + 5] for i in range(0, len(search_links), 5)]

    # Create a semaphore to limit concurrency
    max_concurrent_processes = 5  # Change this number to your desired limit
    semaphore = asyncio.Semaphore(max_concurrent_processes)

    async def run_scrapy_with_file_monitor(scrapy_dir, search_links, spider_result_file_path, timeout=120):
        # Acquire semaphore before running the process
        async with semaphore:
            random.seed(time.time())
            random_num = random.randint(1, 10)
            await asyncio.sleep(random_num)

            # Start the scrapy process using asyncio subprocess
            process = await asyncio.create_subprocess_exec(
                "scrapy", "crawl", "universal_spider",
                "-a", f"urls={','.join(search_links)}",
                "-a", f"output_path={spider_result_file_path}",
                cwd=scrapy_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                return_code = process.returncode

                if return_code != 0 and not os.path.exists(spider_result_file_path):
                    raise Exception(f"Failed to run scrapy: {stderr.decode()}")
                else:
                    print(f"Scrapy spider completed, output file: {spider_result_file_path}")

            except asyncio.TimeoutError:
                print(f"Scrapy process timed out after {timeout} seconds, terminating...")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    print(f"Process kill wait timed out, continuing anyway...")

    # Use asyncio to run multiple scrapy processes with limited concurrency
    tasks = []
    for index, search_links_batch in enumerate(search_links_batches):
        result_file_name = f"search_results_{current_time}_{index}.jsonl"
        result_file_path = os.path.join(scrapy_dir, "results", result_file_name)
        tasks.append(run_scrapy_with_file_monitor(scrapy_dir, search_links_batch, result_file_path))

    # This will still create all tasks but the semaphore ensures only N run at once
    await asyncio.gather(*tasks)

    print(f"Scrapy processes completed, parsing search results...\n")

    content = []

    for index, search_links_batch in enumerate(search_links_batches):
        result_file_name = f"search_results_{current_time}_{index}.jsonl"
        result_file_path = os.path.join(scrapy_dir, "results", result_file_name)
        # Check if the result file exists
        if not os.path.exists(result_file_path):
            print(f"Result file not found: {result_file_path}")
            continue
        # Read results from the jsonl file
        with open(result_file_path, "r") as f:
            data = f.readlines()
            for line in data:
                json_data = json.loads(line)
                # Check if json_data contains 'content' key
                if "content" in json_data:
                    content.append(json_data["content"])

    search_results = []

    async_tasks = []
    for index, document in enumerate(content):
        async def parse_document(document: str, chain):
            response = None
            for i in range(3):
                if i == 2:
                    current_chain = prompt | volcengine_deepseek_llm
                else:
                    current_chain = chain
                try:
                    response = await asyncio.wait_for(current_chain.ainvoke({"current_time": current_time, "document": document, "task": task_description}), timeout=300)
                    break
                except Exception as e:
                    print(f"Timeout error, for document index: {index}, error: {traceback.format_exc()}")
                    if i == 2:
                        raise asyncio.TimeoutError("Timeout error")
                    time.sleep(30)
                    continue
            result = response.content
            search_results.append(result)

        chain = prompt | deepseek_llm
        async_tasks.append(parse_document(document, chain))

    await asyncio.gather(*async_tasks)

    search_result = current_search_result + "\n"
    for result in search_results:
        search_result += result + "\n"

    system_message = SystemMessagePromptTemplate.from_template(
        """
        You are a professional project analyst.
        Extract the main content from the web search results.
        Do not miss any important information.
        Merge same or similar information.
        Remove any irrelevant information.
        List all technical, metric or article details from search results, do not just summarize.
        """
    )
    human_message = HumanMessagePromptTemplate.from_template(
        """
        Current time: {current_time}
        
        Current task: {task}
        
        Web search results: {search_result}
        """
    )

    prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message
    ])

    final_search_result = ""
    chain = prompt | volcengine_deepseek_reasoner_llm
    for i in range(3):
        if i == 2:
            current_chain = prompt | volcengine_deepseek_llm
        else:
            current_chain = chain
        try:
            final_search_result = current_chain.invoke({"current_time": current_time, "task": task_description, "search_result": search_result}).content
            break
        except Exception as e:
            print(f"Error in parsing search links: {traceback.format_exc()}")

            if i == 2:
                raise asyncio.TimeoutError("Timeout error")
            time.sleep(30)
            continue

    # print(f"Search result: {final_search_result}")
    return final_search_result

async def execute_task(task: dict, task_results: Dict[str, str], task_result: str) -> str:
    current_time = time.strftime("%Y%m%d_%H%M%S")

    system_message = SystemMessagePromptTemplate.from_template(
    """
    You are a professional project analyst.
    Your job is to analyze the user's task and the results obtained from external tools like web search.
    And give a detailed analysis based on the user's task and the results obtained with your previous analysis.
    Do not miss any important information and be as detailed as possible.
    """
    )

    human_message = HumanMessagePromptTemplate.from_template(
    """
    Current time: {current_time}
    Current task: 
    {description}
    
    Previous analysis:
    {analysis}
    
    All Tasks results:
    {knowledge}
    """
    )
    
    prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message
    ])

    chain = prompt | volcengine_deepseek_reasoner_llm
    print(f"Executing task: {task}, with current knowledge: {task_results}\n")
    for i in range(3):
        if i == 2:
            current_chain = prompt | volcengine_deepseek_reasoner_llm
        else:
            current_chain = chain
        try:
            response = await asyncio.wait_for(current_chain.ainvoke({"current_time": current_time, "description": task["description"], "analysis": task_result, "knowledge": task_results}), timeout=300)
            break
        except Exception as e:
            print(f"Error in executing task: {traceback.format_exc()}")

            if i == 2:
                raise asyncio.TimeoutError("Timeout error")
            time.sleep(30)
            continue
    return response.content

async def self_reflection(full_task_summary: str, task_name: str, task_description: str, task_result: str) -> bool:
    current_time = time.strftime("%Y%m%d_%H%M%S")
    system_message = SystemMessagePromptTemplate.from_template(
    """
    You are a professional project analyst.
    Your job is to reflect on the full task summary and the current task with it result obtained from external tools like web search.
    Finally give a result whether you have gather enough information and already able to answer everything in the current task which is enough to complete the full task.
    Important: You have to be honest and have figured out every detail of the task with current knowledge before you can answer 'True'.
    The result should be whether the task is completed or not with a simple 'True' or 'False'.
    """
    )

    human_message = HumanMessagePromptTemplate.from_template(
    """
    Current time: {current_time}
    
    Full Task summary:
    {full_task_summary}
    
    Current task description:
    {task_description}
    
    Current task results:
    {task_result}
    """
    )

    prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message
    ])

    chain = prompt | deepseek_reasoner_llm

    print(f"Self reflection on task: {task_name}")

    for i in range(3):
        if i == 2:
            current_chain = prompt | volcengine_deepseek_llm
        else:
            current_chain = chain
        try:
            response = await asyncio.wait_for(
                current_chain.ainvoke({
                    "full_task_summary": full_task_summary,
                    "current_time": current_time,
                    "task_description": task_description,
                    "task_result": task_result
                }),
                timeout=300,
            )
            response = response.content.strip().lower().capitalize()
            break
        except Exception as e:
            print(f"Error in self reflection: {traceback.format_exc()}")

            if i == 2:
                raise asyncio.TimeoutError("Timeout error")
            time.sleep(30)
            continue

    if response.find("True") != -1:
        completed = True
    elif response.find("False") != -1:
        completed = False
    else:
        completed = False

    completed_mark = "[x]"
    if completed:
        completed_mark = "[✔]"
    print(f"Task: {task_name} is completed: {completed_mark}\n")

    return completed

def generate_report(user_task: str, results: Dict[str, str]) -> str:
    current_time = time.strftime("%Y%m%d_%H%M%S")
    system_message = SystemMessagePromptTemplate.from_template(
    """
    You are a professional project analyst.
    Your job is to analyze the user's task and the results obtained from external tools like web search.
    Finally generate a detailed report based on the user's task and the results with your analysis.
    Please do not miss any important information and be as detailed as possible.
    """
    )
    human_message = HumanMessagePromptTemplate.from_template(
    """
    Current time: {current_time}
    User original task summary is as follows:
    {user_task}
    
    User decomposed tasks and their results which were obtained from external tools like web search or analyzed by analysts:
    {results}
    """
    )

    prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message
    ])

    chain = prompt | deepseek_reasoner_llm
    print(f"Generating report from results: {results}")
    return chain.invoke({"current_time": current_time, "user_task": user_task, "results": results}).content


def generate_file_name(user_task: str, report: str) -> str:
    system_message = SystemMessagePromptTemplate.from_template(
        """
        You are a professional name generator.
        Your job is to analyze a report and generate a file name based on the report.
        The format of the file name should be:
        1. Separate words with underscores '_'.
        2. Use lowercase letters.
        3. Short and concise.
        4. Do not generate the extension, only the file name.
        5. Do not generate anything else, just the file name.
        
        Example:
        1. "example_report"
        2. "my_project_summary"
        3. "data_analysis"
        4. "report"
        """
    )
    human_message = HumanMessagePromptTemplate.from_template(
        """
        User task is as follows:
        {user_task}
        Generate a file name based on the following report:
        {report}
        """
    )

    prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message
    ])

    chain = prompt | deepseek_llm
    print(f"Generating file name...")
    return chain.invoke({"user_task": user_task, "report": report}).content

def decomposer_node(state: AgentState) -> dict:
    user_task_summary, tasks = decompose_tasks(state["user_input"])
    return {
        "user_task_summary": user_task_summary,
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
    search_links, search_content = await web_search(state["current_task"], state["task_result"])
    return {"search_links": search_links, "task_result": state["task_result"] + "\n" + search_content}

async def search_result_parser_node(state: AgentState) -> dict:
    if not state["search_links"] or not state["current_task"]:
        return {}
    search_result = await parse_search_links(state["search_links"], state["current_task"]["description"], state["task_result"])
    return {"task_result": search_result}

async def task_executor_node(state: AgentState) -> dict:
    if not state["current_task"]:
        return {}
    result = await execute_task(state["current_task"], state["task_results"], state["task_result"])
    return {"task_result": state["task_result"] + "\n" + result}

async def self_reflection_node(state: AgentState) -> dict:
    if not state["task_result"]:
        return {"task_completed": False}

    if not state["current_task"]:
        return {"task_completed": True}

    if state["task_retry_count"] >= 3:
        print(f"Task retry count exceeded, task: {state['current_task']}, continue anyway.")
        return {"task_completed": True}

    task_completed = await self_reflection(
        state["user_task_summary"],
        state["current_task"]["name"],
        state["current_task"]["description"],
        state["task_result"]
    )

    return {
        "task_completed": task_completed,
        "task_retry_count": state["task_retry_count"] + 1,
    }

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
        "needs_tool": None,  # Clear needs_tool status
        "task_result": "",  # Clear temporary result
        "task_completed": None,  # Clear task completion status
        "task_retry_count": 0,  # Reset retry count
        "search_links": None  # Clear search links
    }

def report_generator_node(state: AgentState) -> dict:
    if not state["task_results"]:
        return {"report": "No results to report."}
    report = generate_report(state["user_task_summary"], state["task_results"])
    return {"report": report}

def file_generator_node(state: AgentState):
    if not state["report"]:
        return

    report_file_name = generate_file_name(state["user_input"], state["report"])

    # Check if the file name is valid as a file name
    if not report_file_name.isidentifier():
        print(f"Invalid file name: {report_file_name}")
        report_file_name = "report"

    print(f"Generated file name prefix: {report_file_name}")

    report_reference_file_name = report_file_name + "_reference"

    current_time = time.strftime("%Y%m%d_%H%M%S")
    report_file_name = report_file_name + "_" + current_time + ".md"
    report_file_name = os.path.join(os.path.dirname(__file__), "results", report_file_name)
    # Write the final report to a file
    with open(report_file_name, "w") as f:
        f.write(state["report"])

    report_reference_file_name = report_reference_file_name + "_" + current_time + ".md"
    report_reference_file_name = os.path.join(os.path.dirname(__file__), "results", report_reference_file_name)
    # Write the report reference to a file
    with open(report_reference_file_name, "w") as f:
        # Write the task_results to the reference file
        for task, result in state["task_results"].items():
            f.write(f"# Task {task}\n")
            f.write(f"## Analysis\n{result}\n\n")

    print(f"Report {report_file_name} generated successfully!")

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("decomposer", decomposer_node)
workflow.add_node("router", router_node)
workflow.add_node("search_agent", search_agent_node)
workflow.add_node("search_result_parser", search_result_parser_node)
workflow.add_node("task_executor", task_executor_node)
workflow.add_node("self_reflection", self_reflection_node)
workflow.add_node("state_updater", state_updater_node)
workflow.add_node("report_generator", report_generator_node)
workflow.add_node("file_generator", file_generator_node)

# Define edges
workflow.set_entry_point("decomposer")
workflow.add_edge("decomposer", "router")
workflow.add_conditional_edges(
    "router",
    lambda state: "search_agent" if state["needs_tool"] else "task_executor",
    {"search_agent": "search_agent", "task_executor": "task_executor"}
)
workflow.add_edge("search_agent", "search_result_parser")
workflow.add_edge("search_result_parser", "self_reflection")
workflow.add_edge("task_executor", "self_reflection")
workflow.add_conditional_edges(
    "self_reflection",
    lambda state: "state_updater" if state["task_completed"] else "router",
    {"state_updater": "state_updater", "router": "router"}
)
workflow.add_conditional_edges(
    "state_updater",
    lambda state: "router" if state["tasks"] else "report_generator",
    {"router": "router", "report_generator": "report_generator"}
)
workflow.add_edge("report_generator", "file_generator")
workflow.add_edge("file_generator", END)

# Compile the agent
agent = workflow.compile()

async def run_agent(your_task: str = None):
    initial_state = {
        "user_input": your_task,
        "user_task_summary": "",
        "tasks": [],
        "task_results": {},
        "current_task": None,
        "needs_tool": None,
        "task_completed": None,
        "task_retry_count": 0,
        "task_result": "",
        "report": ""
    }
    return await agent.ainvoke(initial_state, config={"recursion_limit": 100})

your_task = """
I'm currently working on a project to build a side-chain for VSYS chain and communicate them through a bridging solution.
Find and compare side-chain and bridging solutions for VSYS chain, the bridge solution should be generic and can be used for bridging VSYS chain with any other blockchain with smart contract support.
The side-chain needs to support smart contracts.
The side-chain can be used for commercial.
The side-chain can be ran as a private chain, without connecting to current main/test blockchain network.
The bridging solution should be secure and efficient, and without touching the funds on both chains, to avoid becoming VASP (Virtual Asset Service Provider)
VSYS chain do not have any side-chain or bridging solution yet, so current bridging solutions will not work, we have to start from scratch.
You need to study how VSYS chain works, including consensus, block generation, transaction processing, etc. from online resources.
You need to study current blockchain with smart contract support, including consensus, block generation, transaction processing, etc. which can be used as a side-chain, and can be used for commercial from online resources.
You need to study how bridging solution works, including cross-chain communication, asset transfer, etc. from online resources.
My final goal is tweaking VSYS chain and creating a bridge to a currently existing blockchain as a side-chain, especially by building a witness (Oracle) node that can communicate with the side-chain, and without touching the funds on both chains, to avoid becoming VASP (Virtual Asset Service Provider).
You should provide a detailed report on the side-chain and bridging solutions you found, including potential sblockchains that can be used as a side-chain, and bridging solutions that can be used to connect VSYS chain with the side-chain.
"""

result = asyncio.run(run_agent(your_task))

print(f"Final report: {result['report']}")
