#!/usr/bin/env python3
"""
Basic test for Letta server functionality.
"""

import subprocess
import time
import requests
import sys

def test_basic_server():
    """Test basic server start/stop."""
    print("Testing basic Letta server...")
    
    # Start server
    print("Starting server on port 8285...")
    proc = subprocess.Popen(
        ["letta", "server", "--port", "8285", "--host", "localhost"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("Waiting for server to be ready...")
    for i in range(30):
        try:
            response = requests.get("http://localhost:8285/health", timeout=1)
            if response.status_code == 200:
                print(f"✓ Server is healthy after {i+1} seconds")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("✗ Server failed to start within 30 seconds")
        proc.terminate()
        return False
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8285/health")
        print(f"✓ Health check response: {response.status_code}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
    
    # Stop server
    print("Stopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
        print("✓ Server stopped gracefully")
    except subprocess.TimeoutExpired:
        proc.kill()
        print("✓ Server killed")
    
    return True

if __name__ == "__main__":
    # Kill any existing servers first
    subprocess.run(["pkill", "-f", "letta server"], capture_output=True)
    time.sleep(2)
    
    success = test_basic_server()
    sys.exit(0 if success else 1)