"""Test the religious knowledge bases and chatbot engine."""
import sys
import json

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from chatbot import ChatbotEngine

engine = ChatbotEngine()

print("=" * 80)
print("RELIGIOUS KNOWLEDGE BASE TESTS")
print("=" * 80)

tests = [
    ("What is Bhagavad Gita?", "bhagavad_gita"),
    ("Tell me about the Bible", "bible"),
    ("What is karma?", "bhagavad_gita"),
    ("What is dharma?", "bhagavad_gita"),
    ("What is moksha?", "bhagavad_gita"),
    ("Tell me about reincarnation", "bhagavad_gita"),
    ("How to meditate?", "meditation"),
    ("What is the soul?", "soul"),
    ("Who is God?", "god"),
    ("What are the ten commandments?", "commandment"),
    ("Tell me about the prodigal son", "bible"),
    ("What is forgiveness?", "forgiveness"),
    ("What is faith?", "faith"),
    ("What happens after death?", "afterlife"),
    ("What is the purpose of life?", "purpose_of_life"),
    ("Tell me about karma yoga", "bhagavad_gita"),
    ("Tell me about the good samaritan", "bible"),
    ("What did Jesus teach?", "bible"),
    ("Compare Hinduism and Christianity", "compare_religions"),
    ("What is the meaning of life?", "meaning_of_life"),
]

passed = 0
failed = 0

for question, expected_intent in tests:
    result = engine.get_response(question)
    intent = result["intent"]
    has_content = len(result["response"]) > 50

    if intent == "religious_knowledge" and has_content:
        status = "PASS"
        passed += 1
    elif intent == expected_intent and has_content:
        status = "PASS"
        passed += 1
    else:
        status = "FAIL"
        failed += 1

    print(f"\n{status} | Q: {question}")
    print(f"     | Intent: {intent}")
    print(f"     | Response length: {len(result['response'])} chars")
    if status == "FAIL":
        print(f"     | Expected: {expected_intent}")
        print(f"     | Response: {result['response'][:100]}...")

print("\n" + "=" * 80)
print(f"RESULTS: {passed}/{passed+failed} passed, {failed} failed")
print("=" * 80)

# Also test that non-religious questions still work
print("\n\n" + "=" * 80)
print("NON-RELIGIOUS TESTS (should still work)")
print("=" * 80)

non_religious = [
    "hello",
    "tell me a joke",
    "what is your name",
    "who made you",
    "tell me a fun fact",
]

for question in non_religious:
    result = engine.get_response(question)
    print(f"\nQ: {question}")
    print(f"Intent: {result['intent']}")
    print(f"A: {result['response'][:100]}...")
