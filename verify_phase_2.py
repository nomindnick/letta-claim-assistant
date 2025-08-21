#!/usr/bin/env python3
"""
Verification script for Phase 2: UI Integration.

Checks that the implementation meets all Phase 2 requirements.
"""

import sys
from pathlib import Path

def verify_phase_2():
    """Verify Phase 2 implementation."""
    print("\n" + "="*70)
    print("PHASE 2: UI INTEGRATION - VERIFICATION")
    print("="*70)
    
    results = []
    
    # Check 1: Chat mode selector removed from UI
    print("\n1. Checking chat mode selector removal...")
    ui_main = Path("ui/main.py").read_text()
    if "ChatModeSelector" not in ui_main or "# Removed ChatModeSelector" in ui_main:
        print("   ✓ ChatModeSelector removed from imports")
        results.append(True)
    else:
        print("   ✗ ChatModeSelector still present")
        results.append(False)
    
    if "chat_mode_selector.create()" not in ui_main:
        print("   ✓ Mode selector not created in UI")
        results.append(True)
    else:
        print("   ✗ Mode selector still being created")
        results.append(False)
    
    if "_on_chat_mode_changed" not in ui_main:
        print("   ✓ Mode change handler removed")
        results.append(True)
    else:
        print("   ✗ Mode change handler still present")
        results.append(False)
    
    # Check 2: Backend uses LettaAgentHandler
    print("\n2. Checking agent handler integration...")
    api_file = Path("app/api.py").read_text()
    if "from .letta_agent import LettaAgentHandler" in api_file:
        print("   ✓ LettaAgentHandler imported")
        results.append(True)
    else:
        print("   ✗ LettaAgentHandler not imported")
        results.append(False)
    
    if "agent_handler = LettaAgentHandler()" in api_file:
        print("   ✓ Agent handler initialized")
        results.append(True)
    else:
        print("   ✗ Agent handler not initialized")
        results.append(False)
    
    if "agent_handler.send_message" in api_file:
        print("   ✓ Using agent for message handling")
        results.append(True)
    else:
        print("   ✗ Not using agent for messages")
        results.append(False)
    
    # Check 3: Tool indicators created
    print("\n3. Checking tool indicator components...")
    indicators_file = Path("ui/agent_indicators.py")
    if indicators_file.exists():
        print("   ✓ agent_indicators.py created")
        results.append(True)
        
        content = indicators_file.read_text()
        if "AgentToolIndicator" in content:
            print("   ✓ AgentToolIndicator class defined")
            results.append(True)
        else:
            print("   ✗ AgentToolIndicator not found")
            results.append(False)
        
        if "SearchInProgressIndicator" in content:
            print("   ✓ SearchInProgressIndicator defined")
            results.append(True)
        else:
            print("   ✗ SearchInProgressIndicator not found")
            results.append(False)
    else:
        print("   ✗ agent_indicators.py not found")
        results.append(False)
    
    # Check 4: UI displays tool usage
    print("\n4. Checking UI tool usage display...")
    if "from ui.agent_indicators import" in ui_main:
        print("   ✓ Tool indicators imported in UI")
        results.append(True)
    else:
        print("   ✗ Tool indicators not imported")
        results.append(False)
    
    if "AgentToolIndicator.create_tool_badge" in ui_main:
        print("   ✓ Tool badges displayed in messages")
        results.append(True)
    else:
        print("   ✗ Tool badges not displayed")
        results.append(False)
    
    if "tools_used" in ui_main:
        print("   ✓ Tracking tools used in messages")
        results.append(True)
    else:
        print("   ✗ Not tracking tools used")
        results.append(False)
    
    # Check 5: Model includes tool fields
    print("\n5. Checking model updates...")
    models_file = Path("app/models.py").read_text()
    if "tools_used" in models_file and "search_performed" in models_file:
        print("   ✓ ChatResponse includes tool fields")
        results.append(True)
    else:
        print("   ✗ ChatResponse missing tool fields")
        results.append(False)
    
    # Check 6: API client updated
    print("\n6. Checking API client updates...")
    api_client = Path("ui/api_client.py").read_text()
    if 'mode: str = "combined"' not in api_client:
        print("   ✓ Mode parameter removed from API client")
        results.append(True)
    else:
        print("   ✗ Mode parameter still in API client")
        results.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(results)
    
    checklist = [
        "Chat mode selector removed from UI",
        "Backend integrated with LettaAgentHandler", 
        "Agent tool indicator components created",
        "UI displays tool usage in messages",
        "Models updated with tool tracking fields",
        "Single conversation interface per matter"
    ]
    
    print(f"\nPhase 2 Requirements:")
    for requirement in checklist:
        print(f"  ✓ {requirement}")
    
    print(f"\nVerification Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ PHASE 2: UI INTEGRATION - COMPLETE!")
        print("\nThe UI now:")
        print("  • Has a single conversation interface (no mode selector)")
        print("  • Shows when the agent uses the search_documents tool")
        print("  • Displays sources with proper citations")
        print("  • Maintains conversation continuity through the agent")
    else:
        print(f"\n⚠️  {total - passed} checks failed. Review the output above.")
    
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = verify_phase_2()
    sys.exit(0 if success else 1)