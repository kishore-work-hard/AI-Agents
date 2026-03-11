import json
import math
import os
import requests
from datetime import datetime
from groq import Groq
from ddgs import DDGS
from bs4 import BeautifulSoup

# ============================================================
#  SETUP
# ============================================================

client = Groq(api_key="gsk_W1vRtvIkmy64iAKd")  # Replace with your Groq API key

# --- Memory file paths ---
HISTORY_FILE = "chat_history.json"      # Stores every message ever
SUMMARY_FILE = "chat_summary.json"      # Stores AI-generated session summaries

# Loaded at startup, saved at exit
conversation_history = []

SYSTEM_PROMPT = """You are a helpful personal assistant with memory of past conversations.
You have access to a summary of previous sessions and the full current conversation.
Use this context naturally — if the user mentioned something before, acknowledge it.
When using tools, always prefer them over guessing.
"""


# ============================================================
#  MEMORY FUNCTIONS
# ============================================================

def load_history() -> list:
    """Load full conversation history from disk."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"  📂 Loaded {len(data)} messages from history.")
            return data
    return []


def save_history(history: list):
    """Save full conversation history to disk."""
    # Only save user and assistant messages (not tool calls — they clutter the file)
    clean = [m for m in history if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved {len(clean)} messages to history.")


def load_summary() -> str:
    """Load the AI-generated summary of past sessions."""
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("summary", "")
    return ""


def save_summary(summary: str):
    """Save the AI-generated summary to disk."""
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "updated": datetime.now().isoformat()}, f, indent=2)
    print(f"  📝 Session summary saved.")


def generate_summary(history: list, existing_summary: str) -> str:
    """Ask the AI to summarize the conversation for future memory."""

    # Only use user/assistant messages for summarization
    clean = [m for m in history if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)]

    if not clean:
        return existing_summary

    # Build a prompt that includes existing summary + new conversation
    summary_prompt = f"""You are a memory summarizer. Your job is to create a concise summary 
of what has been discussed, so future sessions can remember key facts about the user.

Previous summary (from older sessions):
{existing_summary if existing_summary else "None — this is the first session."}

New conversation to incorporate:
{json.dumps(clean, indent=2)}

Write a short updated summary (max 200 words) capturing:
- Key facts about the user (name, location, preferences, job, etc.)
- Topics they've asked about
- Important things they've mentioned
- Any preferences or patterns noticed

Write in third person. Be concise. Only include meaningful facts, not small talk."""

    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=400,
    )
    return response.choices[0].message.content


# ============================================================
#  TOOL FUNCTIONS
# ============================================================

def get_date() -> str:
    now = datetime.now()
    return now.strftime("Today is %A, %B %d, %Y. Current time is %I:%M %p.")


def calculator(expression: str) -> str:
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


def get_weather(city: str) -> str:
    try:
        geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1", timeout=8).json()
        if not geo.get("results"):
            return f"Could not find city: {city}"
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        name = geo["results"][0]["name"]
        country = geo["results"][0].get("country", "")
        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,weathercode,windspeed_10m,relative_humidity_2m"
            f"&temperature_unit=celsius", timeout=8
        ).json()
        current = weather["current"]
        codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 51: "Light drizzle", 61: "Light rain", 63: "Rain",
            71: "Light snow", 80: "Rain showers", 95: "Thunderstorm"
        }
        desc = codes.get(current["weathercode"], f"Code {current['weathercode']}")
        return (f"Weather in {name}, {country}: {desc}, {current['temperature_2m']}°C, "
                f"Humidity: {current['relative_humidity_2m']}%, Wind: {current['windspeed_10m']} km/h")
    except Exception as e:
        return f"Weather error: {str(e)}"


def web_search(query: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        response = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for r in soup.select(".result__body")[:3]:
            title = r.select_one(".result__title")
            snippet = r.select_one(".result__snippet")
            if title and snippet:
                results.append(f"- {title.get_text()}: {snippet.get_text()}")
        return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"


def file_reader(filepath: str) -> str:
    try:
        if filepath.endswith(".pdf"):
            try:
                import PyPDF2
                with open(filepath, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    return "".join(p.extract_text() + "\n" for p in reader.pages)[:3000]
            except ImportError:
                return "PDF support requires: pip install PyPDF2"
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()[:3000]
    except FileNotFoundError:
        return f"File not found: '{filepath}'"
    except Exception as e:
        return f"File read error: {str(e)}"


def file_writer(filepath: str, content: str) -> str:
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"File write error: {str(e)}"


# ============================================================
#  TOOL DEFINITIONS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression e.g. '15/100 * 3500'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name e.g. 'Tokyo'"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_reader",
            "description": "Read contents of a text or PDF file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to file e.g. 'notes.txt'"}
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_writer",
            "description": "Write content to a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to file"},
                    "content": {"type": "string", "description": "Full content to write"}
                },
                "required": ["filepath", "content"]
            }
        }
    }
]

TOOL_MAP = {
    "get_date": get_date,
    "calculator": calculator,
    "get_weather": get_weather,
    "web_search": web_search,
    "file_reader": file_reader,
    "file_writer": file_writer,
}


# ============================================================
#  AGENT LOOP
# ============================================================

def run_agent(user_message: str) -> str:
    conversation_history.append({"role": "user", "content": user_message})

    while True:
        try:
            response = client.chat.completions.create(
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
            conversation_history.pop()
            return f"Sorry, something went wrong: {str(e)}"

        message = response.choices[0].message

        if response.choices[0].finish_reason == "stop":
            final_reply = message.content
            conversation_history.append({"role": "assistant", "content": final_reply})
            return final_reply

        if response.choices[0].finish_reason == "tool_calls":
            conversation_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in message.tool_calls
                ]
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments
                parsed = json.loads(raw_args) if raw_args and raw_args.strip() else {}
                tool_args = parsed if isinstance(parsed, dict) else {}

                print(f"  🔧 Using tool: {tool_name}({tool_args})")

                tool_fn = TOOL_MAP.get(tool_name)
                tool_result = tool_fn(**tool_args) if tool_fn else f"Unknown tool: {tool_name}"

                print(f"  ✅ Result: {str(tool_result)[:100]}...")

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })


# ============================================================
#  STARTUP — load memory
# ============================================================

def startup():
    """Load history and summary, inject summary into system prompt."""
    global SYSTEM_PROMPT

    print("\n  🧠 Loading memory...")

    # Load full history
    history = load_history()
    conversation_history.extend(history)

    # Load and inject summary into system prompt
    summary = load_summary()
    if summary:
        SYSTEM_PROMPT += f"\n\n--- Memory from past sessions ---\n{summary}\n---"
        print(f"  📋 Past summary loaded.")
    else:
        print(f"  📋 No past summary found (first session).")

    print()


# ============================================================
#  SHUTDOWN — save memory
# ============================================================

def shutdown():
    """Save history and generate+save summary before exiting."""
    print("\n  💾 Saving memory before exit...")

    # Save full conversation history
    save_history(conversation_history)

    # Generate and save smart summary
    existing_summary = load_summary()
    print("  🤔 Generating session summary...")
    new_summary = generate_summary(conversation_history, existing_summary)
    save_summary(new_summary)

    print("\n  Summary of this session:")
    print(f"  {new_summary}")
    print()


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 50)
    print("   AI Agent with Persistent Memory")
    print("   Powered by Groq — remembers across sessions")
    print("   Type 'quit' to save and exit")
    print("   Type 'clear' to wipe all memory")
    print("   Type 'summary' to see current memory summary")
    print("=" * 50)

    startup()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            shutdown()
            print("Agent: Goodbye! See you next time 👋")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            if os.path.exists(SUMMARY_FILE):
                os.remove(SUMMARY_FILE)
            SYSTEM_PROMPT = SYSTEM_PROMPT.split("\n\n--- Memory")[0]  # strip old summary
            print("Agent: All memory wiped! Fresh start 🧹\n")
            continue

        if user_input.lower() == "summary":
            summary = load_summary()
            print(f"\nAgent: Here's what I remember about you:\n{summary if summary else 'Nothing yet!'}\n")
            continue

        print()
        response = run_agent(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
