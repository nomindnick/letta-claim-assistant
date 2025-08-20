#!/usr/bin/env python3
"""
Quick test script to verify chat modes work via API.
"""

import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

# Test data
test_request = {
    "matter_id": "test-matter-123",
    "query": "What is the project deadline?",
    "k": 5
}

# Test each mode
modes = ["rag", "memory", "combined"]

print("Testing Chat Modes via API")
print("=" * 40)

for mode in modes:
    print(f"\nTesting {mode.upper()} mode:")
    test_request["mode"] = mode
    
    print(f"  Request: {json.dumps(test_request, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json=test_request,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"  ✓ Success: {response.status_code}")
            result = response.json()
            print(f"  - Answer length: {len(result.get('answer', ''))}")
            print(f"  - Sources: {len(result.get('sources', []))}")
            print(f"  - Memory used: {len(result.get('used_memory', []))}")
        else:
            print(f"  ✗ Error: {response.status_code}")
            print(f"  - Message: {response.text[:200]}")
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Connection error: {e}")

print("\n" + "=" * 40)
print("Note: Start the API server with:")
print("  cd /home/nick/Projects/letta-claim-assistant")
print("  source .venv/bin/activate")
print("  uvicorn app.api:app --reload")