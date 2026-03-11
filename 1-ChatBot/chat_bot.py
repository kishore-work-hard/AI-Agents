# gsk_W1vRtvIkmy64iAKd
from groq import Groq

# --- Setup ---
client = Groq(api_key="gsk_W1vRtvIkmy64iAKd")  # Replace with your Groq API key

# This list stores the entire conversation history
# so the AI remembers what was said before
conversation_history = []

SYSTEM_PROMPT = """You are a helpful and friendly assistant. 
Answer questions clearly and concisely."""


def chat(user_message):
    """Send a message and get a response from the AI."""

    # Add the user's message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Call the Groq API with full conversation history
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Free model on Groq
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *conversation_history  # Send full history so AI has memory
        ],
        temperature=0.7,   # 0 = robotic/precise, 1 = creative/random
        max_tokens=1024,   # Max length of the AI's reply
    )

    # Extract the AI's reply
    ai_reply = response.choices[0].message.content

    # Add AI's reply to history too
    conversation_history.append({
        "role": "assistant",
        "content": ai_reply
    })

    return ai_reply


def main():
    print("=" * 40)
    print("   Simple AI Chatbot powered by Groq")
    print("   Type 'quit' to exit")
    print("   Type 'clear' to reset memory")
    print("=" * 40)
    print()

    while True:
        # Get input from user
        user_input = input("You: ").strip()

        # Skip empty input
        if not user_input:
            continue

        # Quit command
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye! 👋")
            break

        # Clear memory command
        if user_input.lower() == "clear":
            conversation_history.clear()
            print("Chatbot: Memory cleared! Fresh start. 🧹\n")
            continue

        # Get and print AI response
        response = chat(user_input)
        print(f"\nChatbot: {response}\n")


if __name__ == "__main__":
    main()
