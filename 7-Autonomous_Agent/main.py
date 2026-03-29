"""
Level 7 — Autonomous Background Agent
───────────────────────────────────────
Runs forever with NO human input.
Watches a folder for new .txt files.
When a new file appears:
    1. Reads it
    2. Sends to AI to summarize
    3. Saves summary to output folder
    4. Logs everything

You just drop a file — the agent does the rest.

Folder structure:
    watch_folder/    ← drop .txt files here
    summaries/       ← agent saves summaries here
    agent.log        ← everything the agent did
"""

import os
import time
import logging
import logging.handlers
from datetime import datetime
from groq import Groq


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# All settings in one place — change these to customize behavior
# ══════════════════════════════════════════════════════════════════════════════

WATCH_FOLDER   = "watch_folder"    # folder the agent monitors for new files
OUTPUT_FOLDER  = "summaries"       # folder where summaries are saved
CHECK_INTERVAL = 10                # seconds between each folder check
LOG_FILE       = "agent.log"       # log file path
client = Groq(api_key="")
MODEL = "moonshotai/kimi-k2-instruct"


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

# Create folders if they don't exist
# exist_ok=True means no error if folder already exists
os.makedirs(WATCH_FOLDER,  exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)



# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# Writes to both terminal AND a log file so you can review what happened later
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)

    # Format: timestamp [LEVEL] message
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Terminal handler — see logs in real time
    # reconfigure to UTF-8 so Windows cp1252 terminal never breaks
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    # File handler — rotating: 1MB max, keeps 3 old files
    # So log never grows forever and takes up disk space
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=3
    )
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

log = setup_logger()


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY
# The agent needs to remember which files it already processed.
# Otherwise every time it wakes up it would re-process everything.
#
# We use a simple Python set() — a collection of unique items.
# When we process a file we add its name to the set.
# Next time we check, we skip files already in the set.
#
# Limitation: this resets when the script restarts.
# Level 10 will persist this to disk.
# ══════════════════════════════════════════════════════════════════════════════

processed_files: set = set()


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def read_file(filepath: str) -> str:
    """
    Read a text file and return its contents.
    Returns None if something goes wrong.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        log.error("Failed to read %s: %s", filepath, e)
        return None


def summarize_with_ai(filename: str, content: str) -> str:
    """
    Send the file content to the AI and get a summary back.

    This is a simple single API call — not a ReAct loop.
    The agent doesn't need to use tools here, just summarize text.

    Notice the system prompt tells the AI exactly what format to use.
    Structured output makes it easy to save and read later.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a summarization assistant. "
                        "When given a document, produce a clean structured summary with:\n"
                        "- SUMMARY: 2-3 sentence overview\n"
                        "- KEY POINTS: bullet list of main points\n"
                        "- WORD COUNT: approximate word count of original\n"
                        "Be concise and factual."
                    )
                },
                {
                    "role": "user",
                    "content": f"File: {filename}\n\nContent:\n{content}"
                }
            ],
            temperature=0,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        log.error("AI summarization failed for %s: %s", filename, e)
        return None


def save_summary(original_filename: str, summary: str) -> str:
    """
    Save the summary to the output folder.

    Naming convention:
        original file:  report.txt
        summary file:   report_summary_2026-03-29_14-32-01.txt

    The timestamp prevents overwriting if the same filename is dropped twice.
    Returns the path where the summary was saved.
    """
    # Strip extension from original filename
    base_name  = os.path.splitext(original_filename)[0]
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_name   = f"{base_name}_summary_{timestamp}.txt"
    out_path   = os.path.join(OUTPUT_FOLDER, out_name)

    # Build the full summary file content
    full_content = (
        f"ORIGINAL FILE : {original_filename}\n"
        f"PROCESSED AT  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'─' * 50}\n\n"
        f"{summary}"
    )

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_content)
        return out_path
    except Exception as e:
        log.error("Failed to save summary for %s: %s", original_filename, e)
        return None


def process_file(filepath: str) -> None:
    """
    Full pipeline for one file:
        read → summarize → save → log

    This is called once per new file the agent finds.
    """
    filename = os.path.basename(filepath)

    log.info("New file detected: %s", filename)

    # Step 1: Read
    content = read_file(filepath)
    if not content:
        log.warning("Skipping %s — empty or unreadable", filename)
        return

    log.info("Read %d characters from %s", len(content), filename)

    # Step 2: Summarize with AI
    log.info("Sending to AI for summarization...")
    summary = summarize_with_ai(filename, content)
    if not summary:
        log.error("Summarization failed for %s — skipping", filename)
        return

    # Step 3: Save summary
    saved_path = save_summary(filename, summary)
    if saved_path:
        log.info("Summary saved -> %s", saved_path)
    else:
        log.error("Could not save summary for %s", filename)
        return

    # Step 4: Print summary to terminal so you can see it live
    print(f"\n{'=' * 60}")
    print(f"File     : {filename}")
    print(f"Saved to : {saved_path}")
    print(f"{'-' * 60}")
    print(summary)
    print(f"{'=' * 60}\n")


def check_for_new_files() -> None:
    """
    Scan the watch folder for .txt files we haven't processed yet.

    This runs every CHECK_INTERVAL seconds.
    It only processes files NOT already in processed_files set.
    """
    try:
        all_files = os.listdir(WATCH_FOLDER)
    except Exception as e:
        log.error("Cannot read watch folder: %s", e)
        return

    # Filter to only .txt files we haven't seen before
    new_files = [
        f for f in all_files
        if f.endswith(".txt") and f not in processed_files
    ]

    if not new_files:
        # Nothing new — just log at debug level (won't show by default)
        log.debug("No new files found. Watching...")
        return

    log.info("Found %d new file(s): %s", len(new_files), new_files)

    for filename in new_files:
        filepath = os.path.join(WATCH_FOLDER, filename)
        process_file(filepath)

        # Mark as processed regardless of success
        # We don't want to keep retrying a broken file forever
        processed_files.add(filename)
        log.info("Marked as processed: %s", filename)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# The heart of an autonomous agent — runs forever until Ctrl+C
#
# Pattern:
#   while True:
#       do the thing
#       sleep
#       repeat
#
# This is exactly how cron jobs, monitoring daemons, and
# production data pipelines work internally.
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("-" * 50)
    log.info("Autonomous Agent - Level 7")
    log.info("Watching folder : %s", os.path.abspath(WATCH_FOLDER))
    log.info("Output folder   : %s", os.path.abspath(OUTPUT_FOLDER))
    log.info("Check interval  : %ds", CHECK_INTERVAL)
    log.info("Drop .txt files into '%s' to trigger summarization", WATCH_FOLDER)
    log.info("Press Ctrl+C to stop")
    log.info("-" * 50)

    # The infinite loop — this is what makes it "autonomous"
    # It never stops unless you kill it
    while True:
        try:
            check_for_new_files()

        except KeyboardInterrupt:
            # Ctrl+C pressed — clean shutdown
            log.info("Agent stopped by user")
            break

        except Exception as e:
            # Something unexpected happened — log it but DON'T crash
            # A production agent must NEVER crash from a single bad file
            log.error("Unexpected error in main loop: %s", e)

        # Sleep until next check
        # The agent does nothing during this time — uses zero CPU
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
