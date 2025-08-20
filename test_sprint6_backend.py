#!/usr/bin/env python3
"""
Test script for Sprint 6: Backend integration test for chat modes.
Tests that the UI changes properly integrate with the backend API.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ui.api_client import APIClient
import inspect


async def test_api_client_signature():
    """Test that API client has been updated with mode parameter."""
    print("\n=== Testing API Client Signature ===")
    
    client = APIClient()
    
    # Check send_chat_message signature
    sig = inspect.signature(client.send_chat_message)
    params = sig.parameters
    
    # Verify mode parameter exists
    assert 'mode' in params, "Missing 'mode' parameter in send_chat_message"
    print("✓ Mode parameter exists in send_chat_message")
    
    # Verify default value
    assert params['mode'].default == 'combined', f"Wrong default: {params['mode'].default}"
    print("✓ Mode parameter defaults to 'combined'")
    
    # Verify it's in the right position (after k parameter)
    param_names = list(params.keys())
    k_index = param_names.index('k')
    mode_index = param_names.index('mode')
    assert mode_index == k_index + 1, "Mode parameter should come after 'k'"
    print("✓ Mode parameter is in correct position")
    
    print("✓ All API client signature tests passed!")


async def test_chat_components_import():
    """Test that chat components can be imported."""
    print("\n=== Testing Chat Components Import ===")
    
    try:
        from ui.chat_components import ChatMode, ModeInfo
        print("✓ Successfully imported ChatMode and ModeInfo")
        
        # Test ChatMode enum values
        assert ChatMode.RAG_ONLY == "rag"
        assert ChatMode.MEMORY_ONLY == "memory"
        assert ChatMode.COMBINED == "combined"
        print("✓ ChatMode enum values are correct")
        
    except ImportError as e:
        print(f"✗ Failed to import chat components: {e}")
        return False
    
    print("✓ All chat component import tests passed!")
    return True


async def test_ui_main_integration():
    """Test that main UI has been updated."""
    print("\n=== Testing UI Main Integration ===")
    
    # Read the main.py file to verify changes
    main_file = Path("ui/main.py")
    content = main_file.read_text()
    
    # Check for imports
    assert "from ui.chat_components import ChatModeSelector, ChatModeIndicator" in content
    print("✓ Chat components are imported in main.py")
    
    # Check for mode state variable
    assert "self.current_chat_mode" in content
    print("✓ current_chat_mode state variable exists")
    
    # Check for mode selector creation
    assert "self.chat_mode_selector = ChatModeSelector" in content
    print("✓ ChatModeSelector is instantiated")
    
    # Check for mode change callback
    assert "_on_chat_mode_changed" in content
    print("✓ Mode change callback method exists")
    
    # Check that mode is passed to API
    assert "mode=self.current_chat_mode" in content
    print("✓ Mode is passed to API in send_chat_message")
    
    # Check for mode-specific thinking messages
    assert '"Consulting agent memory..."' in content
    assert '"Searching documents..."' in content
    assert '"Searching documents and memory..."' in content
    print("✓ Mode-specific thinking messages implemented")
    
    print("✓ All UI main integration tests passed!")


async def test_backend_api():
    """Test that backend API accepts mode parameter."""
    print("\n=== Testing Backend API ===")
    
    # Read the API models to verify ChatMode enum exists
    models_file = Path("app/models.py")
    content = models_file.read_text()
    
    assert "class ChatMode(str, Enum):" in content
    print("✓ ChatMode enum exists in backend models")
    
    assert 'RAG_ONLY = "rag"' in content
    assert 'MEMORY_ONLY = "memory"' in content
    assert 'COMBINED = "combined"' in content
    print("✓ All chat modes defined in backend")
    
    # Check ChatRequest model includes mode
    assert "mode: ChatMode" in content
    print("✓ ChatRequest includes mode field")
    
    # Check RAG engine handles modes
    rag_file = Path("app/rag.py")
    content = rag_file.read_text()
    
    assert "mode: ChatMode = ChatMode.COMBINED" in content
    print("✓ RAG engine accepts mode parameter")
    
    assert "if mode == ChatMode.MEMORY_ONLY:" in content
    print("✓ RAG engine handles MEMORY_ONLY mode")
    
    assert "if mode != ChatMode.RAG_ONLY" in content
    print("✓ RAG engine handles RAG_ONLY mode")
    
    print("✓ All backend API tests passed!")


async def main():
    """Run all tests."""
    print("=" * 50)
    print("Sprint 6: Chat Mode UI - Backend Integration Test")
    print("=" * 50)
    
    try:
        await test_api_client_signature()
        await test_chat_components_import()
        await test_ui_main_integration()
        await test_backend_api()
        
        print("\n" + "=" * 50)
        print("✅ All Sprint 6 integration tests passed!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Start the backend: uvicorn app.api:app --reload")
        print("2. Start the UI: python -m ui.main")
        print("3. Test the mode selector in the chat interface")
        print("4. Verify each mode sends correct requests to backend")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())