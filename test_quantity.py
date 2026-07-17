"""Test short/medium/long answer quantity for religious questions."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from chatbot import ChatbotEngine
engine = ChatbotEngine()

print("=" * 80)
print("SHORT / MEDIUM / LONG ANSWER TESTS")
print("=" * 80)

tests = [
    ("Tell me one line from bhagavad gita", "short"),
    ("Give me the most popular gita verse", "short"),
    ("Give me the best bible verse", "short"),
    ("Tell me about bhagavad gita", "medium"),
    ("What is karma?", "medium"),
    ("Tell me about the prodigal son", "medium"),
    ("Tell me all famous verses from bhagavad gita", "long"),
    ("Give me multiple parables from the bible", "long"),
    ("List all the commandments from the bible", "long"),
]

for question, expected in tests:
    result = engine.get_response(question)
    response_len = len(result["response"])
    intent = result["intent"]

    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"Expected quantity: {expected}")
    print(f"Intent: {intent}")
    print(f"Response length: {response_len} chars")
    print(f"Response preview:")
    print(f"  {result['response'][:300]}...")
