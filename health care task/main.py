# ============================================================
# Task 2: General Health Query Chatbot using Hugging Face LLM
# ============================================================

import os

try:
    from huggingface_hub import InferenceClient
except ImportError:
    print("Please install required packages: pip install -r requirements.txt")
    exit(1)

from dotenv import load_dotenv

# ── Load Environment Variables ───────────────────────────────
load_dotenv()
API_TOKEN = os.getenv("HF_TOKEN")

if not API_TOKEN:
    print("WARNING: HF_TOKEN not set. You may hit rate limits quickly.")
    print("Sign up at huggingface.co and add your token to the .env file.\n")

# ── Model Selection ──────────────────────────────────────────
# These models support text_generation on HF's free serverless API.
# We try them in order until one works.
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "HuggingFaceH4/zephyr-7b-gemma-v0.1",
    "microsoft/Phi-3-mini-4k-instruct",
    "tiiuae/falcon-7b-instruct",
]

# ── Safety Filter Keywords ────────────────────────────────────
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe",
    "cannot breathe", "severe bleeding", "unconscious", "overdose",
    "suicide", "poisoning", "dying", "emergency"
]

EMERGENCY_RESPONSE = (
    "⚠️  This sounds like a medical emergency!\n"
    "Please call emergency services (911 or your local emergency number) "
    "or go to the nearest emergency room IMMEDIATELY.\n"
    "Do not wait — get help right now."
)

# ── System Prompt (Prompt Engineering) ───────────────────────
SYSTEM_PROMPT = """You are a helpful, empathetic, and informative general medical assistant.
Your goal is to provide general, educational health information to users.

CRITICAL SAFETY RULES:
1. You MUST NOT provide specific medical diagnoses or personalized treatment plans.
2. If asked about personal symptoms, clearly state you are an AI — not a doctor — and advise consulting a qualified healthcare professional.
3. Keep answers general, objective, and focused on common medical knowledge.
4. If the query involves a life-threatening emergency, immediately instruct the user to call emergency services.
5. Always be warm, clear, and easy to understand.

Answer the following health question in a friendly and informative way:"""


def format_prompt(user_query: str) -> str:
    """
    Formats the prompt using Mistral/Zephyr instruction template.
    <s>[INST] system + user [/INST]
    This format is understood by most open-source instruction-tuned models.
    """
    return f"<s>[INST] {SYSTEM_PROMPT}\n\nUser question: {user_query} [/INST]"


def is_emergency(user_query: str) -> bool:
    """Safety filter: checks if the query contains emergency keywords."""
    query_lower = user_query.lower()
    return any(keyword in query_lower for keyword in EMERGENCY_KEYWORDS)


def get_health_advice(client: InferenceClient, user_query: str) -> str:
    """
    Sends a health query to the HF model using text_generation,
    which is supported by far more free-tier models than chat_completion.
    """
    # ── Safety Filter ─────────────────────────────────────────
    if is_emergency(user_query):
        return EMERGENCY_RESPONSE

    # ── Call the LLM ─────────────────────────────────────────
    try:
        prompt = format_prompt(user_query)
        response = client.text_generation(
            prompt,
            max_new_tokens=300,
            temperature=0.7,
            repetition_penalty=1.1,
            do_sample=True,
            stop_sequences=["</s>", "[INST]", "User:"]
        )
        return response.strip()

    except Exception as e:
        return f"Error communicating with the API: {e}"


def initialize_client() -> tuple:
    """
    Tries each model in the MODELS list and returns the first working one.
    Returns (client, model_name) or (None, None) if all fail.
    """
    for model in MODELS:
        try:
            print(f"  Trying model: {model} ...", end=" ")
            c = InferenceClient(model, token=API_TOKEN)
            # Quick test to verify the model works
            c.text_generation(
                "<s>[INST] Say hello [/INST]",
                max_new_tokens=10
            )
            print("✔ Connected!")
            return c, model
        except Exception as e:
            print(f"✘ ({str(e)[:60]})")
    return None, None


def main():
    print("=" * 50)
    print("   General Health Query Chatbot Assistant")
    print("=" * 50)
    print("\nDisclaimer: I am an AI, not a doctor.")
    print("I provide general health information only.\n")

    # ── Initialize Client ─────────────────────────────────────
    print("Finding an available model...")
    client, model_name = initialize_client()

    if client is None:
        print("\n❌ Could not connect to any model.")
        print("Make sure your HF_TOKEN is valid and try again.")
        return

    print(f"\nUsing model: {model_name}\n")

    # ── Run Example Queries ───────────────────────────────────
    example_queries = [
        "What causes a sore throat?",
        "Is paracetamol safe for children?",
        "I'm having severe chest pain right now, what should I do?"
    ]

    print("--- Running Example Queries ---")
    for q in example_queries:
        print(f"\nUser: {q}")
        print(f"Chatbot: {get_health_advice(client, q)}")
        print("-" * 40)

    print("\n--- Interactive Mode (type 'exit' to quit) ---\n")

    # ── Interactive Chat Loop ─────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nChatbot: Goodbye! Stay healthy. 👋")
            break

        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye! Stay healthy. 👋")
            break

        if not user_input:
            continue

        response = get_health_advice(client, user_input)
        print(f"\nChatbot: {response}\n")


if __name__ == "__main__":
    main()
