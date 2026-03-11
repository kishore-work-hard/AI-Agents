import json
import math
import requests
from datetime import datetime
from groq import Groq
from ddgs import DDGS

# ============================================================
#  SETUP
# ============================================================

client = Groq(api_key="gsk_W1vRtvIkmy64iAKd")  # Replace with your Groq API key
conversation_history = []

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
When you need to calculate something, use the calculator tool.
When asked about weather, use the weather tool.
When you need to look something up, use the web_search tool.
When asked to read a file, use the file_reader tool.
Always use tools when appropriate instead of guessing."""


# ============================================================
#  TOOL FUNCTIONS — the actual logic that runs
# ============================================================

def get_date() -> str:
    """Return the current date and time."""
    now = datetime.now()
    return now.strftime("Today is %A, %B %d, %Y. Current time is %I:%M %p.")


def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Allow only safe math operations
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


def get_weather(city: str) -> str:
    """Get current weather using Open-Meteo (no API key needed)."""
    try:
        # Step 1: Get coordinates for the city
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo = requests.get(geo_url, timeout=8).json()

        if not geo.get("results"):
            return f"Could not find city: {city}"

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        name = geo["results"][0]["name"]
        country = geo["results"][0].get("country", "")

        # Step 2: Get weather for those coordinates
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,weathercode,windspeed_10m,relative_humidity_2m"
            f"&temperature_unit=celsius"
        )
        weather = requests.get(weather_url, timeout=8).json()
        current = weather["current"]

        # Weather code to description mapping
        codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
            61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow",
            80: "Rain showers", 81: "Heavy showers", 95: "Thunderstorm"
        }
        code = current["weathercode"]
        description = codes.get(code, f"Weather code {code}")

        return (
            f"Weather in {name}, {country}: {description}, "
            f"{current['temperature_2m']}°C, "
            f"Humidity: {current['relative_humidity_2m']}%, "
            f"Wind: {current['windspeed_10m']} km/h"
        )
    except Exception as e:
        return f"Weather fetch error: {str(e)}"


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top 3 results."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"- {r['title']}: {r['body']}")
        if results:
            return "\n".join(results)
        return "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"


def file_reader(filepath: str) -> str:
    """Read contents of a .txt or .pdf file."""
    try:
        if filepath.endswith(".pdf"):
            try:
                import PyPDF2
                with open(filepath, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text[:3000]  # Limit to first 3000 chars
            except ImportError:
                return "PDF support requires: pip install PyPDF2"
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return content[:3000]  # Limit to first 3000 chars
    except FileNotFoundError:
        return f"File not found: '{filepath}'"
    except Exception as e:
        return f"File read error: {str(e)}"

def file_writer(filepath: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"File write error: {str(e)}"
# ============================================================
#  TOOL DEFINITIONS — tells the AI what tools exist
#  (This is what you send to the API)
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date and time. Use this whenever the user asks what day, date, or time it is.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use this for any math calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 + 2' or 'sqrt(144)' or '15 * 8'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'London' or 'Tokyo'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on any topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'latest AI news 2025'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_reader",
            "description": "Read the contents of a text or PDF file from the computer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Full or relative path to the file, e.g. 'notes.txt' or 'C:/docs/report.pdf'"
                    }
                },
                "required": ["filepath"]
            }
        }
    },
{
    "type": "function",
    "function": {
        "name": "file_writer",
        "description": "Write or overwrite content to a text file on the computer.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Full or relative path to the file, e.g. 'notes.txt'"
                },
                "content": {
                    "type": "string",
                    "description": "The full content to write into the file"
                }
            },
            "required": ["filepath", "content"]
        }
    }
}
]

# Map tool names to actual functions
TOOL_MAP = {
    "get_date": get_date,
    "calculator": calculator,
    "get_weather": get_weather,
    "web_search": web_search,
    "file_reader": file_reader,
    "file_writer":file_writer,
}


# ============================================================
#  AGENT LOOP — the core of how an agent works
# ============================================================

def run_agent(user_message: str) -> str:
    """
    The agent loop:
    1. Send message to AI
    2. If AI wants to use a tool → run the tool → feed result back
    3. Repeat until AI gives a final answer
    """

    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    while True:
        # Call the AI
        try:
            response = client.chat.completions.create(
                # model="llama-3.3-70b-versatile",
                model="moonshotai/kimi-k2-instruct",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *conversation_history
                ],
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1024,
            )
        except Exception as e:
            # Remove the last user message so history stays clean
            conversation_history.pop()
            return f"Sorry, something went wrong: {str(e)}\nPlease try rephrasing your question."

        message = response.choices[0].message

        # --- Case 1: AI is done, return final answer ---
        if response.choices[0].finish_reason == "stop":
            final_reply = message.content
            conversation_history.append({
                "role": "assistant",
                "content": final_reply
            })
            return final_reply

        # --- Case 2: AI wants to use one or more tools ---
        if response.choices[0].finish_reason == "tool_calls":

            # Add AI's tool request to history
            conversation_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Run each tool the AI requested
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments
                print(f"  DEBUG raw_args = {repr(raw_args)}")  # add this
                parsed = json.loads(raw_args) if raw_args and raw_args.strip() else {}
                tool_args = parsed if isinstance(parsed, dict) else {}
                print(f"  🔧 Using tool: {tool_name}({tool_args})")

                # Call the actual Python function
                tool_fn = TOOL_MAP.get(tool_name)
                if tool_fn:
                    tool_result = tool_fn(**tool_args)
                else:
                    tool_result = f"Unknown tool: {tool_name}"

                print(f"  ✅ Tool result: {tool_result[:100]}...")  # Preview

                # Feed tool result back to AI
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            # Loop again — AI will now form a final answer using tool results


# ============================================================
#  MAIN — chat interface
# ============================================================

def main():
    print("=" * 50)
    print("   AI Agent with Tools — powered by Groq")
    print("   Tools: Calculator, Weather, Web Search, File Reader")
    print("   Type 'quit' to exit | 'clear' to reset memory")
    print("=" * 50)
    print()
    print("Try asking:")
    print("  - What is 15% of 3500?")
    print("  - What's the weather in Tokyo?")
    print("  - Search for latest Python 3.13 features")
    print("  - Read my file notes.txt")
    print()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Agent: Goodbye! 👋")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            print("Agent: Memory cleared! 🧹\n")
            continue

        print()  # spacing
        response = run_agent(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
