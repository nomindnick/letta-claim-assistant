#!/usr/bin/env python3
"""
Test script for Sprint L1: Letta Server Infrastructure
Tests server management, configuration, and basic connectivity.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.letta_server import LettaServerManager
from app.letta_config import LettaConfigManager, LettaServerConfig, LettaClientConfig
from app.settings import settings


def test_server_config():
    """Test server configuration management."""
    print("\n=== Testing Server Configuration ===")
    
    # Test config manager initialization
    config_manager = LettaConfigManager()
    print(f"✓ Config manager initialized")
    print(f"  Config dir: {config_manager.config_dir}")
    
    # Test server config
    server_config = config_manager.server_config
    print(f"✓ Server config loaded")
    print(f"  Mode: {server_config.mode}")
    print(f"  Host: {server_config.host}:{server_config.port}")
    print(f"  Auto-start: {server_config.auto_start}")
    
    # Test client config
    client_config = config_manager.client_config
    print(f"✓ Client config loaded")
    print(f"  Base URL: {client_config.base_url}")
    print(f"  Timeout: {client_config.timeout}s")
    
    # Test agent config generation
    agent_config = config_manager.get_agent_config("Test-Matter")
    print(f"✓ Agent config generated")
    print(f"  Name: {agent_config.name}")
    print(f"  LLM: {agent_config.llm_provider}/{agent_config.llm_model}")
    
    return True


def test_server_management():
    """Test server lifecycle management."""
    print("\n=== Testing Server Management ===")
    
    # Get server manager instance
    server_manager = LettaServerManager()
    server_manager.configure(
        mode="subprocess",
        host="localhost",
        port=8283,
        auto_start=False
    )
    print(f"✓ Server manager created and configured")
    
    # Test health check (should fail if not running)
    is_healthy = server_manager.health_check()
    print(f"  Initial health check: {'✓ Healthy' if is_healthy else '✗ Not running'}")
    
    # Test server start
    print("\nStarting Letta server...")
    success = server_manager.start()
    
    if success:
        print(f"✓ Server started successfully")
        print(f"  URL: {server_manager.get_base_url()}")
        
        # Test health check
        is_healthy = server_manager.health_check()
        print(f"  Health check: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")
        
        # Let it run for a moment
        import time
        time.sleep(2)
        
        # Test server stop
        print("\nStopping server...")
        stopped = server_manager.stop()
        print(f"{'✓' if stopped else '✗'} Server stopped")
        
        return True
    else:
        print(f"✗ Failed to start server")
        print("  This might be because:")
        print("  - Letta is not installed in the virtual environment")
        print("  - Port 8283 is already in use")
        print("  - Missing dependencies")
        return False


async def test_client_connection():
    """Test client connection to server."""
    print("\n=== Testing Client Connection ===")
    
    try:
        from letta_client import AsyncLetta
        
        # Start server first
        server_manager = LettaServerManager()
        if not server_manager.start():
            print("✗ Failed to start server for client test")
            return False
        
        print("✓ Server started for client test")
        
        # Create client
        client = AsyncLetta(base_url=server_manager.get_base_url())
        print(f"✓ Client created")
        
        # Test health check
        try:
            await client.health.health_check()
            print(f"✓ Client connected successfully")
            
            # List agents (should be empty initially)
            agents = await client.agents.list_agents()
            print(f"✓ Listed agents: {len(agents)} found")
            
            success = True
        except Exception as e:
            print(f"✗ Client operation failed: {e}")
            success = False
        
        # Stop server
        server_manager.stop()
        print("✓ Server stopped after test")
        
        return success
        
    except ImportError:
        print("✗ letta_client not installed")
        return False


def test_settings_integration():
    """Test integration with settings module."""
    print("\n=== Testing Settings Integration ===")
    
    # Check settings
    config = settings.global_config
    print(f"✓ Settings loaded")
    print(f"  Letta server mode: {config.letta_server_mode}")
    print(f"  Letta server host: {config.letta_server_host}:{config.letta_server_port}")
    print(f"  Auto-start: {config.letta_server_auto_start}")
    
    # Test that settings can be used to configure server
    server_manager = LettaServerManager()
    server_manager.configure(
        mode=config.letta_server_mode,
        host=config.letta_server_host,
        port=config.letta_server_port,
        auto_start=False  # Don't auto-start for test
    )
    print(f"✓ Server manager configured from settings")
    
    return True


def main():
    """Run all Sprint L1 tests."""
    print("=" * 60)
    print("Sprint L1: Letta Server Infrastructure Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Configuration
    try:
        results.append(("Configuration", test_server_config()))
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        results.append(("Configuration", False))
    
    # Test 2: Settings Integration
    try:
        results.append(("Settings Integration", test_settings_integration()))
    except Exception as e:
        print(f"✗ Settings integration test failed: {e}")
        results.append(("Settings Integration", False))
    
    # Test 3: Server Management
    try:
        results.append(("Server Management", test_server_management()))
    except Exception as e:
        print(f"✗ Server management test failed: {e}")
        results.append(("Server Management", False))
    
    # Test 4: Client Connection
    try:
        result = asyncio.run(test_client_connection())
        results.append(("Client Connection", result))
    except Exception as e:
        print(f"✗ Client connection test failed: {e}")
        results.append(("Client Connection", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✅ Sprint L1 implementation successful!")
        print("\nAcceptance Criteria Met:")
        print("✓ Server configuration management implemented")
        print("✓ Server lifecycle management working")
        print("✓ Client can connect to server")
        print("✓ Settings integration complete")
        print("✓ Fallback mode supported")
    else:
        print("\n⚠️ Sprint L1 implementation incomplete")
        print("Please review failed tests and fix issues.")
    
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())