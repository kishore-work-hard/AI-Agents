"""
Level 4 — ReAct Agent
─────────────────────
The agent thinks, acts, observes, and repeats until
it has enough information to give a final answer.

Flow:
    User asks question
        ↓
    Agent THINKS — what tool do I need?
        ↓
    Agent ACTS   — calls the tool
        ↓
    Agent OBSERVES — sees the result
        ↓
    Agent THINKS again — do I need more info?
        ↓ (repeat until done)
    Agent gives FINAL ANSWER
"""

import json
import os
import math
from groq import Groq

# ── API setup ─────────────────────────────────────────────────────────────────

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL  = "llama3-70b-8192"


# ── Tools ─────────────────────────────────────────────────────────────────────
# These are the same tools you built in Level 2.
# Each tool is just a plain Python function.

def calculator(expression: str) -> str:
    """
    Safely evaluate a math expression.
    Example: calculator("2 + 2") → "4"
    """
    try:
        # We use math module so agent can use sqrt, pow, etc.
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """
    Fake weather tool — in a real project this would
    call a weather API like OpenWeatherMap.
    """
    # Hardcoded for learning purposes
    fake_data = {
        "london":  "15°C, cloudy",
        "kerala":  "32°C, humid",
        "new york": "22°C, sunny",
    }
    return fake_data.get(city.lower(), f"No weather data for {city}")


def search_web(query: str) -> str:
    """
    Fake web search — in a real project this would
    call Google Search API or Tavily.
    """
    # Hardcoded for learning purposes
    fake_results = {
        "population of india":  "India population is approximately 1.44 billion (2024)",
        "who is elon musk":     "Elon Musk is CEO of Tesla, SpaceX, and owner of X (Twitter)",
        "python creator":       "Python was created by Guido van Rossum in 1991",
    }
    # Simple keyword match
    for key, value in fake_results.items():
        if any(word in query.lower() for word in key.split()):
            return value
    return f"No results found for: {query}"


# ── Tool registry ─────────────────────────────────────────────────────────────
# This is a dictionary that maps tool NAMES (strings) to actual FUNCTIONS.
# When the AI says "call calculator", we look it up here and run it.
# This is the core pattern — it's how ALL agent frameworks work internally.

TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "search_web":  search_web,
}


# ── Tool descriptions for the AI ──────────────────────────────────────────────
# We describe each tool to the AI in plain English so it knows:
#   1. What the tool does
#   2. What input it expects
#   3. What it returns
#
# This goes into the system prompt so the AI understands its "toolbox".

TOOL_DESCRIPTIONS = """
You have access to these tools:

1. calculator(expression)
   - Does math calculations
   - Input: a math expression as a string e.g. "2 + 2" or "sqrt(16)"
   - Returns: the result as a string

2. get_weather(city)
   - Gets current weather for a city
   - Input: city name as a string e.g. "London"
   - Returns: temperature and conditions

3. search_web(query)
   - Searches the web for information
   - Input: search query as a string
   - Returns: search result as a string
"""


# ── System prompt ─────────────────────────────────────────────────────────────
# This is the most important part of a ReAct agent.
# We tell the AI EXACTLY how to format its responses so we can parse them.
#
# The AI must respond in one of two formats:
#
# Format 1 — when it wants to use a tool:
#   THOUGHT: I need to calculate something
#   ACTION: calculator
#   INPUT: 15 * 7
#
# Format 2 — when it has the final answer:
#   THOUGHT: I now have all the information
#   FINAL ANSWER: The result is 105

SYSTEM_PROMPT = f"""You are a helpful assistant that solves problems step by step.

{TOOL_DESCRIPTIONS}

INSTRUCTIONS:
- Think through the problem step by step
- Use tools when you need information or calculations
- After getting a tool result, decide if you need more tools or can answer

RESPONSE FORMAT — you must ALWAYS use one of these two formats:

Format 1 (when you need a tool):
THOUGHT: <your reasoning here>
ACTION: <tool name exactly as listed above>
INPUT: <the input for the tool>

Format 2 (when you have the final answer):
THOUGHT: <your reasoning here>
FINAL ANSWER: <your complete answer to the user>

IMPORTANT:
- Never make up tool results — always actually call the tool
- Only use tools that are listed above
- ACTION must be exactly: calculator, get_weather, or search_web
- You MUST call tools one at a time and wait for the OBSERVATION before proceeding
- Never predict or assume tool results
"""


# ── Response parser ───────────────────────────────────────────────────────────
# After the AI responds, we need to read its text and figure out:
#   - Does it want to call a tool? (ACTION + INPUT)
#   - Or is it done? (FINAL ANSWER)
#
# This is called "parsing" — extracting structured data from text.

def parse_response(text: str) -> dict:
    """
    Parse the AI's response and return a dict with:
    
    If tool call:
        {"type": "action", "action": "calculator", "input": "2+2"}
    
    If final answer:
        {"type": "final", "answer": "The result is 4"}
    
    If we can't parse it:
        {"type": "unknown", "raw": "..."}
    """
    lines = text.strip().split("\n")

    result = {}

    for line in lines:
        line = line.strip()

        if line.startswith("THOUGHT:"):
            # Store the thought but don't act on it
            # We print it so the user can see the agent's reasoning
            result["thought"] = line.replace("THOUGHT:", "").strip()

        elif line.startswith("ACTION:"):
            result["type"]   = "action"
            result["action"] = line.replace("ACTION:", "").strip()

        elif line.startswith("INPUT:"):
            result["input"] = line.replace("INPUT:", "").strip()

        elif line.startswith("FINAL ANSWER:"):
            result["type"]   = "final"
            result["answer"] = line.replace("FINAL ANSWER:", "").strip()

    # If we found action but no type was set cleanly, set it
    if "action" in result and "type" not in result:
        result["type"] = "action"

    # Fallback if nothing matched
    if "type" not in result:
        result["type"] = "unknown"
        result["raw"]  = text

    return result


# ── The ReAct loop ────────────────────────────────────────────────────────────
# This is the heart of the agent.
#
# It keeps a "messages" list — the full conversation history including:
#   - User's question
#   - Agent's thoughts and actions
#   - Tool results (observations)
#
# Each iteration of the loop:
#   1. Ask the AI what to do next (given everything so far)
#   2. Parse the response
#   3. If it wants a tool → run the tool, add result to history, loop again
#   4. If it has final answer → return it
#   5. If too many steps → stop (safety limit)

def run_react_agent(user_question: str, max_steps: int = 10) -> str:
    """
    Run the ReAct loop for a user question.
    Returns the final answer as a string.
    
    max_steps prevents infinite loops in case something goes wrong.
    """

    print(f"\n{'='*60}")
    print(f"Question: {user_question}")
    print(f"{'='*60}")

    # Start with the system prompt and the user's question
    # This is the same message history pattern from your Level 3 bot
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_question},
    ]

    # ── Main loop ─────────────────────────────────────────────────────────────
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")

        # Step 1: Ask the AI what to do next
        # We send the ENTIRE conversation history each time
        # so the AI remembers what tools it already called
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,      # 0 = deterministic, better for tool use
            max_tokens=500,
        )

        ai_text = response.choices[0].message.content
        print(f"AI: {ai_text}")

        # Step 2: Parse what the AI said
        parsed = parse_response(ai_text)

        # Step 3a: If AI wants to use a tool
        if parsed["type"] == "action":
            tool_name  = parsed.get("action", "")
            tool_input = parsed.get("input", "")

            # Add the AI's response to history
            messages.append({
                "role":    "assistant",
                "content": ai_text
            })

            # Check if the tool actually exists
            if tool_name not in TOOLS:
                observation = f"Error: Tool '{tool_name}' does not exist."
            else:
                # Run the actual tool function
                print(f"\n→ Calling tool: {tool_name}({tool_input})")
                tool_function = TOOLS[tool_name]
                observation   = tool_function(tool_input)
                print(f"← Tool result: {observation}")

            # Add the tool result to history as a "user" message
            # We label it "OBSERVATION:" so the AI knows it's a tool result
            # This is the key trick — tool results go back into the conversation
            messages.append({
                "role":    "user",
                "content": f"OBSERVATION: {observation}"
            })

            # Loop again — AI will now see the tool result and decide next step

        # Step 3b: If AI has the final answer — we're done
        elif parsed["type"] == "final":
            print(f"\n✅ Final Answer: {parsed['answer']}")
            return parsed["answer"]

        # Step 3c: If we couldn't parse the response
        else:
            print(f"⚠️  Could not parse response, retrying...")
            # Add a hint to the conversation to help the AI format correctly
            messages.append({
                "role":    "assistant",
                "content": ai_text
            })
            messages.append({
                "role":    "user",
                "content": "Please respond using the exact format: THOUGHT, ACTION, INPUT or THOUGHT, FINAL ANSWER"
            })

    # If we hit max_steps without a final answer
    return "Sorry, I could not complete this task within the step limit."


# ── Main chat loop ────────────────────────────────────────────────────────────

def main():
    print("ReAct Agent — Level 4")
    print("The agent will think step by step and use tools to answer.")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        answer = run_react_agent(user_input)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
