Level 3 adds persistent memory — conversations are saved to a JSON file on disk and reloaded next time you run the agent. So it'll remember things like:

"Last time you told me your name is Kishore"
"You mentioned you're learning Python"
"You asked about Tokyo weather yesterday"


Two types of memory we'll build
1. Full Conversation History — save every message to a JSON file, reload on startup. Simple and complete.
2. Smart Summary Memory — after each session, the AI writes a short summary of what was discussed. Next session, that summary is injected into the system prompt. This is how production AI assistants work — you can't send 1000 messages of history every time, so you summarize.
We'll build both so you understand the tradeoff.

python agent_memory.py
```

No new installs needed — uses everything from before.

---

## What's new in this agent

**Two files get created automatically on your disk:**

`chat_history.json` — every message saved here, reloaded next session
`chat_summary.json` — AI-written summary of what it knows about you

**New commands:**
- `summary` → see what the agent remembers about you
- `clear` → wipe all memory, fresh start
- `quit` → saves everything before exiting (important — always quit properly!)

---

## How to test it

**Session 1** — run the agent and say:
- *"My name is Kishore and I'm learning Python"*
- *"I live in Kozhikode"*
- Then type `quit`

**Session 2** — run it again and ask:
- *"Do you remember my name?"*
- *"Where do I live?"*

It should remember everything from Session 1. That's persistent memory! 🧠

---

## The key concept
```
Session ends  → AI summarizes everything into ~200 words
Next session  → summary injected into system prompt
              → full history also reloaded
Result        → AI has both detailed + summarized memory
