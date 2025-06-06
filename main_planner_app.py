import os
import requests
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
import uvicorn # For running the FastAPI app
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from pydantic import BaseModel # For request body if needed, though using query params here

# --- Environment Variable Setup ---
# Create a .env file in the same directory as this script
# and add your API keys like this:
# OPENWEATHERMAP_API_KEY="your_openweathermap_key"
# TAVILY_API_KEY="your_tavily_key"
# OPENAI_API_KEY="your_openai_key"

load_dotenv() # Load environment variables from .env file

OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Global Variables for Agent Components ---
# These will be initialized on FastAPI startup
agent_executor: AgentExecutor = None
app_initialized_successfully = False

# --- Tool Definitions ---

@tool
def get_current_weather(city: str) -> str:
    """
    Fetches the current weather for a given city using OpenWeatherMap API.
    Returns a string describing the weather conditions.
    Example: get_current_weather(city="London")
    """
    if not OPENWEATHERMAP_API_KEY:
        # This error will be caught by the agent or returned if tool is called directly
        return "Error: OpenWeatherMap API key not found. Please set the OPENWEATHERMAP_API_KEY environment variable."
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get("cod") != 200:
            return f"Error fetching weather for {city}: {data.get('message', 'Unknown error from API')}"

        main_weather = data.get("weather", [{}])[0].get("main", "N/A")
        description = data.get("weather", [{}])[0].get("description", "N/A")
        temp = data.get("main", {}).get("temp", "N/A")
        feels_like = data.get("main", {}).get("feels_like", "N/A")
        humidity = data.get("main", {}).get("humidity", "N/A")
        wind_speed = data.get("wind", {}).get("speed", "N/A")

        weather_report = (
            f"Current weather in {city}:\n"
            f"- Condition: {main_weather} ({description})\n"
            f"- Temperature: {temp}°C (feels like {feels_like}°C)\n"
            f"- Humidity: {humidity}%\n"
            f"- Wind Speed: {wind_speed} m/s"
        )
        return weather_report
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 401: return "Error: Invalid OpenWeatherMap API key or unauthorized."
        elif response.status_code == 404: return f"Error: City '{city}' not found by OpenWeatherMap."
        return f"Error: HTTP error occurred while fetching weather for {city}: {http_err} - {response.text}"
    except requests.exceptions.RequestException as req_err:
        return f"Error: Network error occurred while fetching weather for {city}: {req_err}"
    except KeyError as key_err:
        return f"Error: Could not parse weather data for {city}. Unexpected API response structure: {key_err}. Response: {data}"
    except Exception as e:
        return f"Error: An unexpected error occurred while fetching weather for {city}: {e}"

@tool
def search_activity_links_tavily(activity_query: str) -> str:
    """
    Searches for relevant links and information for a given activity or query using Tavily Search.
    The query should be specific, e.g., "Eiffel Tower Paris tickets" or "Best museums in Berlin".
    Returns a concise summary of search results, often including links.
    Example: search_activity_links_tavily(activity_query="Louvre Museum Paris official website")
    """
    if not TAVILY_API_KEY:
        return "Error: Tavily API key not found. Please set the TAVILY_API_KEY environment variable."
    
    try:
        search_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY) # Using API key directly
        results = search_tool.invoke({"query": activity_query})
        
        if not results:
            return f"No search results found for '{activity_query}' using Tavily."

        formatted_results = []
        for res in results:
            url = res.get('url', 'N/A URL')
            content_snippet = res.get('content', 'N/A Content')
            if len(content_snippet) > 200: content_snippet = content_snippet[:197] + "..."
            formatted_results.append(f"- Source: {url}\n  Snippet: {content_snippet}")
        
        return f"Tavily search results for '{activity_query}':\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"Error: An unexpected error occurred during Tavily search for '{activity_query}': {e}"

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """
    Manages startup and shutdown events for the FastAPI application.
    Initializes the LangChain agent and its components on startup.
    """
    global agent_executor, app_initialized_successfully
    print("FastAPI app starting up (lifespan)...")

    # --- API Key Checks ---
    if not OPENWEATHERMAP_API_KEY:
        print("CRITICAL ERROR: OpenWeatherMap API key (OPENWEATHERMAP_API_KEY) is not set.")
        # Not setting app_initialized_successfully to True
    elif not TAVILY_API_KEY:
        print("CRITICAL ERROR: Tavily API key (TAVILY_API_KEY) is not set.")
    elif not OPENAI_API_KEY:
        print("CRITICAL ERROR: OpenAI API key (OPENAI_API_KEY) is not set.")
    else:
        print("All API keys found.")
        # --- LLM Initialization ---
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY)
            llm.invoke("This is a test prompt to verify OpenAI API access.") # Test call
            print("OpenAI LLM (gpt-4o) initialized successfully.")

            # --- Tools List ---
            tools = [get_current_weather, search_activity_links_tavily]

            # --- Agent Prompt ---
            system_message = """
            You are a SmartCity Activity Planner. Your goal is to help users plan activities in a specified city.
            Follow these steps:
            1.  When a user provides a city name, first use the `get_current_weather` tool to fetch the current weather conditions for that city.
            2.  Based on the weather conditions, suggest 3 to 5 appropriate activities.
            3.  For each suggested activity, use the `search_activity_links_tavily` tool to find relevant information or official links. Query should be specific like "[Activity Name] [City] official website".
            4.  Compile all the information (weather, 3-5 activity suggestions with their corresponding links/info from Tavily) into a single, concise, and user-friendly response.
            5.  If a tool fails or cannot find specific information, clearly state that in your response for that part, but try to complete the rest of the request.
            6.  Do not make up weather information or links. Rely solely on the output from the tools.
            7.  Present the final plan directly to the user.
            8.  If the city name is ambiguous or missing, ask the user for clarification.
            """
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_message),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            # --- Agent and Executor ---
            agent = create_openai_tools_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True, # Good for debugging, set to False in production
                handle_parsing_errors="An error occurred while parsing the agent's output. Please check the format and try again."
            )
            print("LangChain Agent and Executor initialized successfully.")
            app_initialized_successfully = True
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize agent components during startup: {e}")
            # app_initialized_successfully remains False
    
    if app_initialized_successfully:
        print("SmartCity Activity Planner agent is ready.")
    else:
        print("SmartCity Activity Planner agent FAILED to initialize properly. Check logs.")
    
    yield # FastAPI app runs after this point

    # --- Shutdown logic (if any) can go here ---
    print("FastAPI app shutting down (lifespan)...")


# --- FastAPI App Setup ---
app = FastAPI(
    title="SmartCity Activity Planner API",
    description="Provides activity suggestions for a city based on weather and Tavily search.",
    version="1.0.0",
    lifespan=lifespan # Use the new lifespan context manager
)


# --- FastAPI Endpoint ---
class PlanResponse(BaseModel):
    city: str
    plan: str | None = None
    error: str | None = None

@app.get("/plan-activity/", response_model=PlanResponse)
async def get_activity_plan(city: str = Query(..., description="The city for which to plan activities.")):
    """
    Provides activity suggestions for a given city.
    It fetches current weather, suggests 3-5 activities, and finds links for them.
    """
    if not app_initialized_successfully or agent_executor is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: The planner agent is not initialized. Check server logs for errors.")

    if not city.strip():
        raise HTTPException(status_code=400, detail="City name cannot be empty.")

    try:
        print(f"Received request for city: {city}")
        # Construct the input for the agent, matching the prompt's "{input}" variable
        agent_input = {"input": f"Provide activity suggestions for {city}."}
        
        response = agent_executor.invoke(agent_input)
        
        output = response.get("output")
        if output:
            return PlanResponse(city=city, plan=output)
        else:
            error_message = "Agent executed but did not produce a standard output. Raw response: " + str(response)
            print(f"Warning: {error_message}")
            return PlanResponse(city=city, error=error_message)

    except Exception as e:
        print(f"Error during agent execution for city '{city}': {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing your request for {city}: {str(e)}")

# --- Main Execution to Run FastAPI App ---
if __name__ == "__main__":
    # This assumes your Python file is named 'main_planner_app.py' (or whatever you are running)
    # If your file is named main_planner_app.py, then the string should be "main_planner_app:app"
    # If your file is named main_planner.py, then the string should be "main_planner:app"
    
    # Get the name of the current file
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    print(f"Attempting to run SmartCity Activity Planner FastAPI app from module: {module_name}...")
    
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=8000, reload=True)
    # reload=True is good for development, remove for production.
    # Using f"{module_name}:app" makes it dynamically use the current file's name.
