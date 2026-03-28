"""
Level 5 — Multi-Agent System
─────────────────────────────
One orchestrator agent breaks down the user's question
and delegates to specialist agents.

Architecture:
                    User
                     ↓
             ORCHESTRATOR AGENT
            /         |         \
     RESEARCH      MATH        WEATHER
      AGENT        AGENT        AGENT
            \         |         /
             ORCHESTRATOR AGENT
                     ↓
               Final Answer
"""

import os
import math
from groq import Groq

# ── Setup ─────────────────────────────────────────────────────────────────────

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL  = "llama3-70b-8192"


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# Same tools as Level 4 — plain Python functions
# ══════════════════════════════════════════════════════════════════════════════

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search_web(query: str) -> str:
    fake_results = {
        "population of india":   "India's population is approximately 1.44 billion (2024)",
        "who is elon musk":      "Elon Musk is CEO of Tesla and SpaceX, and owner of X",
        "python creator":        "Python was created by Guido van Rossum in 1991",
        "guido van rossum age":  "Guido van Rossum was born January 31, 1956. He is 68 years old.",
        "apple stock price":     "Apple (AAPL) stock is trading at approximately $227 (2024)",
        "population of kerala":  "Kerala has a population of approximately 35 million people",
    }
    for key, value in fake_results.items():
        if any(word in query.lower() for word in key.split()):
            return value
    return f"No results found for: {query}"


def get_weather(city: str) -> str:
    fake_data = {
        "london":   "15°C, cloudy",
        "kerala":   "32°C, humid",
        "new york": "22°C, sunny",
        "delhi":    "38°C, hazy",
        "mumbai":   "30°C, partly cloudy",
    }
    return fake_data.get(city.lower(), f"No weather data available for {city}")


# ══════════════════════════════════════════════════════════════════════════════
# BASE AGENT
# A reusable class that any specialist agent inherits from.
# This avoids repeating the same API call code in every agent.
# ══════════════════════════════════════════════════════════════════════════════

class BaseAgent:
    """
    Every agent in the system inherits from this.
    It handles the actual API call to the LLM.
    
    Subclasses just need to define:
      - self.name        → who this agent is
      - self.system_prompt → what this agent's job is
    """

    def __init__(self, name: str, system_prompt: str):
        self.name          = name
        self.system_prompt = system_prompt

    def run(self, task: str) -> str:
        """
        Send a task to this agent and get its response.
        Each agent gets a fresh conversation — no shared history between agents.
        The orchestrator is responsible for combining results.
        """
        print(f"\n  [{self.name}] Working on: {task}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": task},
            ],
            temperature=0,
            max_tokens=500,
        )

        result = response.choices[0].message.content.strip()
        print(f"  [{self.name}] Result: {result}")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SPECIALIST AGENTS
# Each agent has ONE job and a focused system prompt.
# Notice how simple each system prompt is — that's the point.
# ══════════════════════════════════════════════════════════════════════════════

class ResearchAgent(BaseAgent):
    """
    Knows how to search for information.
    It is given the search_web tool and uses it to answer factual questions.
    """

    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            system_prompt="""You are a research specialist. Your ONLY job is to find factual information.

You have one tool:
  search_web(query) — searches the web and returns results

To use the tool, respond EXACTLY like this:
  SEARCH: <your search query>

After you get a result, give a clean factual summary.
If you need to search multiple times, do it one search at a time.
Be concise. Only state facts, no calculations."""
        )

    def run(self, task: str) -> str:
        """
        Override run() to handle the SEARCH: tool call loop.
        This is a mini ReAct loop just for research.
        """
        print(f"\n  [{self.name}] Working on: {task}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": task},
        ]

        # Allow up to 5 search iterations
        for _ in range(5):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()

            # If agent wants to search
            if text.startswith("SEARCH:"):
                query  = text.replace("SEARCH:", "").strip()
                result = search_web(query)
                print(f"  [{self.name}] Searched: {query} → {result}")

                # Add search result back into conversation
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user",      "content": f"Search result: {result}"})

            else:
                # Agent gave a final answer
                print(f"  [{self.name}] Result: {text}")
                return text

        return "Research could not be completed."


class MathAgent(BaseAgent):
    """
    Only does math. Gets the calculator tool.
    Never looks up facts — that's ResearchAgent's job.
    """

    def __init__(self):
        super().__init__(
            name="MathAgent",
            system_prompt="""You are a math specialist. Your ONLY job is to do calculations.

You have one tool:
  calculator(expression) — evaluates math expressions

To use it, respond EXACTLY like this:
  CALCULATE: <expression>

Example expressions: "2 + 2", "sqrt(144)", "15 * 7 + 23"

After getting the result, give a clean answer.
Do NOT look up facts — only calculate numbers you are given."""
        )

    def run(self, task: str) -> str:
        """
        Override run() to handle CALCULATE: tool calls.
        """
        print(f"\n  [{self.name}] Working on: {task}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": task},
        ]

        for _ in range(5):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()

            if text.startswith("CALCULATE:"):
                expression = text.replace("CALCULATE:", "").strip()
                result     = calculator(expression)
                print(f"  [{self.name}] Calculated: {expression} = {result}")

                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user",      "content": f"Calculator result: {result}"})

            else:
                print(f"  [{self.name}] Result: {text}")
                return text

        return "Math calculation could not be completed."


class WeatherAgent(BaseAgent):
    """
    Only checks weather. Simple — no tool loop needed,
    just one direct tool call.
    """

    def __init__(self):
        super().__init__(
            name="WeatherAgent",
            system_prompt="""You are a weather specialist. Your ONLY job is to report weather.
Extract the city name from the task and report the weather clearly.
Be brief: just state the city and its current conditions."""
        )

    def run(self, task: str) -> str:
        """
        Extract city from task and call get_weather directly.
        No LLM loop needed — just one API call to extract city name.
        """
        print(f"\n  [{self.name}] Working on: {task}")

        # Ask LLM to extract just the city name
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Extract only the city name from this text. Reply with just the city name, nothing else."},
                {"role": "user",   "content": task},
            ],
            temperature=0,
            max_tokens=20,
        )
        city   = response.choices[0].message.content.strip()
        result = get_weather(city)

        print(f"  [{self.name}] Weather in {city}: {result}")
        return f"Weather in {city}: {result}"


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR AGENT
# The manager. It:
#   1. Reads the user's question
#   2. Decides which agents to call and in what order
#   3. Collects all results
#   4. Combines them into one final answer
#
# The orchestrator itself does NO research, math, or weather lookups.
# It only plans and combines.
# ══════════════════════════════════════════════════════════════════════════════

class OrchestratorAgent:
    """
    Breaks down complex questions into subtasks,
    delegates to specialists, and combines results.
    """

    def __init__(self):
        # The orchestrator knows about all available agents
        self.agents = {
            "research": ResearchAgent(),
            "math":     MathAgent(),
            "weather":  WeatherAgent(),
        }

        self.system_prompt = """You are an orchestrator that manages specialist agents.

Available agents:
  - research : finds facts, information, and data
  - math     : does calculations and math operations
  - weather  : gets current weather for a city

Your job:
  1. Analyze the user's question
  2. Break it into subtasks
  3. Assign each subtask to the right agent
  4. Combine all results into a final answer

Respond with a JSON plan like this:
{
  "tasks": [
    {"agent": "research", "task": "find the population of India"},
    {"agent": "math",     "task": "calculate 1440000000 divided by 1000000"},
    {"agent": "weather",  "task": "get weather for Delhi"}
  ]
}

IMPORTANT:
- Only use agents listed above
- Break dependent tasks correctly (research first if math needs the research result)
- Each task description must be self-contained and clear"""

    def _plan(self, user_question: str) -> list:
        """
        Ask the LLM to create a plan — which agents to call and with what tasks.
        Returns a list of {"agent": ..., "task": ...} dicts.
        """
        print(f"\n[Orchestrator] Planning for: {user_question}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_question},
            ],
            temperature=0,
            max_tokens=500,
        )

        text = response.choices[0].message.content.strip()
        print(f"[Orchestrator] Plan: {text}")

        # Parse the JSON plan
        # The LLM returns JSON — we extract and parse it
        import json
        import re

        # Find JSON block in response (sometimes LLM adds extra text)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            print("[Orchestrator] Could not parse plan, running as research task")
            return [{"agent": "research", "task": user_question}]

        try:
            plan = json.loads(json_match.group())
            return plan.get("tasks", [])
        except json.JSONDecodeError:
            return [{"agent": "research", "task": user_question}]

    def _combine(self, user_question: str, results: dict) -> str:
        """
        After all agents have run, ask the LLM to combine
        all results into one clean final answer.
        """
        # Build a summary of all agent results
        results_text = "\n".join([
            f"{agent} found: {result}"
            for agent, result in results.items()
        ])

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You combine research results into a clear, concise final answer. Use only the information provided."
                },
                {
                    "role": "user",
                    "content": f"Original question: {user_question}\n\nAgent results:\n{results_text}\n\nGive a clear final answer."
                },
            ],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    def run(self, user_question: str) -> str:
        """
        Main entry point.
        Plan → Delegate → Combine → Return final answer.
        """
        print(f"\n{'='*60}")
        print(f"Question: {user_question}")
        print(f"{'='*60}")

        # Step 1: Create a plan
        tasks = self._plan(user_question)

        if not tasks:
            return "I could not understand the question."

        # Step 2: Execute each task with the right specialist agent
        # Results stored as {"agent_name (task)": "result"}
        results = {}

        for item in tasks:
            agent_name = item.get("agent", "")
            task       = item.get("task", "")

            if agent_name not in self.agents:
                print(f"[Orchestrator] Unknown agent: {agent_name}, skipping")
                continue

            # Delegate to the specialist
            agent  = self.agents[agent_name]
            result = agent.run(task)

            # Store result with a descriptive key
            results[f"{agent_name} ({task})"] = result

        # Step 3: Combine all results into one answer
        print(f"\n[Orchestrator] Combining {len(results)} results...")
        final_answer = self._combine(user_question, results)

        print(f"\n{'='*60}")
        print(f"✅ Final Answer: {final_answer}")
        print(f"{'='*60}")

        return final_answer


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Multi-Agent System — Level 5")
    print("Complex questions handled by specialist agents.")
    print("Type 'quit' to exit\n")

    orchestrator = OrchestratorAgent()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        answer = orchestrator.run(user_input)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
