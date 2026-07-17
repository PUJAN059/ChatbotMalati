import sys; sys.path.insert(0, r'C:\Users\pujan\Desktop\Chatbot')
from chatbot.engine import ChatbotEngine
e = ChatbotEngine()

# Test 1: Does web search return results?
print("=== TEST 1: Web search ===")
for q in ["who is the president of nepal", "who is the prime minister of nepal 2026", "who is balen shah"]:
    result = e._web_search(q)
    print(f"\nQuery: {q}")
    print(f"Results ({len(result)} chars): {result[:300] if result else 'EMPTY'}")

# Test 2: What messages are being built?
print("\n=== TEST 2: Messages for 'who is the president of nepal' ===")
msgs = e._build_messages("who is the president of nepal")
for m in msgs:
    role = m["role"]
    content = m["content"][:200] if len(m["content"]) > 200 else m["content"]
    print(f"  [{role}]: {content}")

# Test 3: Full response
print("\n=== TEST 3: Full response ===")
resp = e.get_response("who is the president of nepal")
print(f"Response: {resp['response']}")
print(f"Intent: {resp['intent']}")
