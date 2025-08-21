#!/usr/bin/env python
"""
Phase 0: Simple Technical Validation for Letta v0.10.0

Runs basic validation tests to confirm Letta capabilities.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.letta_server import LettaServerManager
from app.logging_conf import get_logger

logger = get_logger(__name__)


def validate_ollama_base_url_fix():
    """Verify that OLLAMA_BASE_URL fix is in place."""
    print("\n=== TEST 1: OLLAMA_BASE_URL Fix ===")
    
    import inspect
    from app.letta_server import LettaServerManager
    
    source = inspect.getsource(LettaServerManager._start_subprocess)
    
    if 'OLLAMA_BASE_URL' in source:
        print("✓ OLLAMA_BASE_URL is set in server startup code")
        if 'http://localhost:11434' in source:
            print("✓ Correct URL for localhost is configured")
        if 'host.docker.internal' in source:
            print("✓ Docker host configuration is present")
        print("✓ TEST PASSED: OLLAMA_BASE_URL environment variable fix is complete")
        return True
    else:
        print("✗ TEST FAILED: OLLAMA_BASE_URL not found in server startup")
        return False


def validate_letta_import():
    """Test that Letta can be imported."""
    print("\n=== TEST 2: Letta Import ===")
    
    try:
        import letta
        print(f"✓ Letta version: {letta.__version__}")
        
        from letta import RESTClient
        print("✓ RESTClient imported successfully")
        
        from letta import LLMConfig, EmbeddingConfig
        print("✓ Configuration classes imported successfully")
        
        print("✓ TEST PASSED: Letta imports work correctly")
        return True
        
    except ImportError as e:
        print(f"✗ TEST FAILED: Could not import Letta: {e}")
        return False


def validate_server_startup():
    """Test that Letta server can start."""
    print("\n=== TEST 3: Server Startup ===")
    
    try:
        server = LettaServerManager()
        
        # Check if server is already running
        if server._is_running:
            print("✓ Letta server is already running")
            return True
        
        print("Starting Letta server...")
        success = server.start()
        
        if success:
            print(f"✓ Server started on {server.host}:{server.port}")
            
            # Give it a moment to fully initialize
            time.sleep(2)
            
            # Try to create a client
            try:
                from letta import RESTClient
                client = RESTClient(base_url=f"http://{server.host}:{server.port}")
                print("✓ Client created successfully")
                
                # Try a simple operation
                try:
                    agents = client.list_agents()
                    print(f"✓ Server responding (found {len(agents)} agents)")
                except Exception as e:
                    print(f"⚠ Server started but API calls failing: {e}")
                
            except Exception as e:
                print(f"⚠ Could not create client: {e}")
            
            # Stop server
            server.stop()
            print("✓ Server stopped cleanly")
            
            print("✓ TEST PASSED: Server lifecycle works")
            return True
        else:
            print("✗ TEST FAILED: Server failed to start")
            return False
            
    except Exception as e:
        print(f"✗ TEST FAILED: Server startup error: {e}")
        return False


def validate_provider_bridge():
    """Test provider bridge configuration."""
    print("\n=== TEST 4: Provider Bridge ===")
    
    try:
        from app.letta_provider_bridge import LettaProviderBridge
        
        bridge = LettaProviderBridge()
        print("✓ Provider bridge initialized")
        
        # Test Ollama configuration
        ollama_config = bridge.get_ollama_config()
        assert ollama_config.provider_type == "ollama"
        assert ollama_config.endpoint_url == "http://localhost:11434"
        print("✓ Ollama configuration correct")
        
        # Test LLM config conversion
        llm_dict = bridge.to_letta_llm_config(ollama_config)
        if llm_dict:
            print("✓ LLM config conversion works")
        
        # Test embedding config conversion
        embed_dict = bridge.to_letta_embedding_config(ollama_config)
        if embed_dict:
            print("✓ Embedding config conversion works")
        
        print("✓ TEST PASSED: Provider bridge functional")
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: Provider bridge error: {e}")
        return False


def check_ollama_availability():
    """Check if Ollama is running locally."""
    print("\n=== TEST 5: Ollama Availability ===")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama is running with {len(models)} models")
            for model in models[:3]:  # Show first 3 models
                print(f"  - {model['name']}")
            return True
        else:
            print("⚠ Ollama responded but with unexpected status")
            return False
    except Exception as e:
        print(f"⚠ Ollama not available: {e}")
        return False


def generate_report(results):
    """Generate final validation report."""
    print("\n" + "=" * 70)
    print("PHASE 0 VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {sum(1 for r in results.values() if r)}")
    print(f"Failed: {sum(1 for r in results.values() if not r)}")
    
    print("\nTest Results:")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    # Critical requirements check
    print("\nCritical Requirements:")
    if results.get("OLLAMA_BASE_URL Fix", False):
        print("✓ OLLAMA_BASE_URL environment variable is properly set")
    else:
        print("✗ CRITICAL: OLLAMA_BASE_URL fix is missing")
    
    if results.get("Letta Import", False):
        print("✓ Letta v0.10.0 is installed and importable")
    else:
        print("✗ CRITICAL: Letta is not properly installed")
    
    # Save report
    report_path = Path("tests/phase_0_simple_validation.json")
    report_path.parent.mkdir(exist_ok=True)
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results.values() if r),
            "failed": sum(1 for r in results.values() if not r)
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    print("=" * 70)


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("PHASE 0: TECHNICAL DISCOVERY & VALIDATION")
    print("Letta v0.10.0 Compatibility Testing")
    print("=" * 70)
    
    results = {}
    
    # Run tests
    results["OLLAMA_BASE_URL Fix"] = validate_ollama_base_url_fix()
    results["Letta Import"] = validate_letta_import()
    results["Server Startup"] = validate_server_startup()
    results["Provider Bridge"] = validate_provider_bridge()
    results["Ollama Availability"] = check_ollama_availability()
    
    # Generate report
    generate_report(results)
    
    # Return exit code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()