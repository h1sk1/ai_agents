# backend.py
import asyncio
import os
import time
import uuid
from typing import Dict, Optional, List
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import your agent code
from agent import run_agent, AgentState, workflow

# Create FastAPI app
app = FastAPI(title="Agentic Search API")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Setup templates and static files
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Define models
class TaskRequest(BaseModel):
    task: str

class TaskResponse(BaseModel):
    task_id: str
    status: str

class TaskStatus(BaseModel):
    status: str
    progress: Optional[Dict[str, str]] = None
    report: Optional[str] = None
    tasks: Optional[List[Dict]] = None
    current_task: Optional[Dict] = None
    completed_tasks: Optional[List[str]] = None

# In-memory storage for tasks
tasks = {}
task_statuses = {}
task_progress = {}

# Monkey patch the workflow nodes to track progress
original_decomposer = workflow.get_node("decomposer").fn
original_router = workflow.get_node("router").fn
original_search_agent = workflow.get_node("search_agent").fn
original_search_result_parser = workflow.get_node("search_result_parser").fn
original_task_executor = workflow.get_node("task_executor").fn
original_self_reflection = workflow.get_node("self_reflection").fn
original_state_updater = workflow.get_node("state_updater").fn
original_report_generator = workflow.get_node("report_generator").fn

# Wrapper functions to track progress
def track_progress(node_name):
    def decorator(fn):
        async def wrapper(state):
            task_id = state.get("task_id")
            if task_id and task_id in task_progress:
                current_task = state.get("current_task", {})
                if current_task:
                    task_name = current_task.get('name', 'unknown')

                    # Update progress based on node
                    if node_name == "decomposer":
                        task_progress[task_id] = {"status": "decomposing", "details": "Analyzing and decomposing task..."}
                    elif node_name == "router":
                        task_progress[task_id] = {"status": "routing", "details": f"Determining if task '{task_name}' needs tools"}
                    elif node_name == "search_agent":
                        task_progress[task_id] = {"status": "searching", "details": f"Searching the web for '{task_name}'"}
                    elif node_name == "search_result_parser":
                        task_progress[task_id] = {"status": "parsing", "details": f"Parsing search results for '{task_name}'"}
                    elif node_name == "task_executor":
                        task_progress[task_id] = {"status": "executing", "details": f"Executing task '{task_name}'"}
                    elif node_name == "self_reflection":
                        task_progress[task_id] = {"status": "reflecting", "details": f"Evaluating completeness of task '{task_name}'"}
                    elif node_name == "state_updater":
                        task_progress[task_id] = {"status": "updating", "details": f"Updating state after task '{task_name}'"}
                    elif node_name == "report_generator":
                        task_progress[task_id] = {"status": "reporting", "details": "Generating final report"}

                    # Update tasks info
                    if state.get("all_tasks") and state.get("tasks"):
                        completed_tasks = [task["name"] for task in state["all_tasks"] if task not in state["tasks"]]
                        task_progress[task_id]["completed_tasks"] = completed_tasks
                        task_progress[task_id]["all_tasks"] = [task["name"] for task in state["all_tasks"]]
                        task_progress[task_id]["current_task"] = task_name

            # Call original function
            if asyncio.iscoroutinefunction(fn):
                return await fn(state)
            return fn(state)

        # Make it async if the original is async
        if asyncio.iscoroutinefunction(fn):
            return wrapper
        return wrapper
    return decorator

# Apply wrappers
workflow.update_node("decomposer", track_progress("decomposer")(original_decomposer))
workflow.update_node("router", track_progress("router")(original_router))
workflow.update_node("search_agent", track_progress("search_agent")(original_search_agent))
workflow.update_node("search_result_parser", track_progress("search_result_parser")(original_search_result_parser))
workflow.update_node("task_executor", track_progress("task_executor")(original_task_executor))
workflow.update_node("self_reflection", track_progress("self_reflection")(original_self_reflection))
workflow.update_node("state_updater", track_progress("state_updater")(original_state_updater))
workflow.update_node("report_generator", track_progress("report_generator")(original_report_generator))

# Recompile the agent with wrapped nodes
agent = workflow.compile()

# Background task to update task status
async def run_agent_task(task_id: str, user_task: str):
    try:
        task_statuses[task_id] = "running"
        task_progress[task_id] = {"status": "starting", "details": "Initializing agent..."}

        # Run the agent with the user's task and track progress
        initial_state = {
            "user_input": user_task,
            "user_task_summary": "",
            "tasks": [],
            "all_tasks": [],
            "task_results": {},
            "current_task": None,
            "needs_tool": None,
            "task_completed": None,
            "task_retry_count": 0,
            "task_result": "",
            "report": "",
            "task_id": task_id  # Add task_id to state for tracking
        }

        result = await agent.ainvoke(initial_state, config={"recursion_limit": 100})

        # Store the result
        tasks[task_id] = result
        task_statuses[task_id] = "completed"
        task_progress[task_id] = {"status": "completed", "details": "Task completed"}

        # Save report to file for download
        report_path = os.path.join(RESULTS_DIR, f"{task_id}.md")
        with open(report_path, "w") as f:
            f.write(result.get("report", "No report generated"))

    except Exception as e:
        task_statuses[task_id] = "failed"
        task_progress[task_id] = {"status": "failed", "details": str(e)}
        print(f"Task failed: {e}")

# API endpoints
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/tasks", response_model=TaskResponse)
@limiter.limit("5/hour")  # Rate limiting: 5 requests per hour per IP
async def create_task(task_request: TaskRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_statuses[task_id] = "pending"
    task_progress[task_id] = {"status": "pending", "details": "Task submitted"}

    # Start the task in the background
    background_tasks.add_task(run_agent_task, task_id, task_request.task)

    return {"task_id": task_id, "status": "pending"}

@app.get("/api/tasks/{task_id}/status", response_model=TaskStatus)
async def get_task_status(task_id: str):
    if task_id not in task_statuses:
        raise HTTPException(status_code=404, detail="Task not found")

    status = task_statuses[task_id]
    progress = task_progress.get(task_id, {})

    response = {"status": status, "progress": progress}

    if status == "completed" and task_id in tasks:
        response["report"] = tasks[task_id].get("report", "")

    # Add task information if available
    if "completed_tasks" in progress:
        response["completed_tasks"] = progress["completed_tasks"]
    if "all_tasks" in progress:
        response["tasks"] = progress["all_tasks"]
    if "current_task" in progress:
        response["current_task"] = progress["current_task"]

    return response

@app.get("/api/tasks/{task_id}/download")
async def download_task_result(task_id: str):
    if task_id not in task_statuses or task_statuses[task_id] != "completed":
        raise HTTPException(status_code=404, detail="Task result not found or not completed")

    result_path = os.path.join(RESULTS_DIR, f"{task_id}.md")
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        result_path,
        media_type="text/markdown",
        filename=f"agent_report_{task_id}.md"
    )

@app.get("/api/limits")
async def get_limits():
    return {
        "requests_per_hour": 5,
        "message": "Rate limited to 5 requests per hour per IP address"
    }

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)