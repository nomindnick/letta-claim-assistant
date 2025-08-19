# Letta Agent Memory - Frequently Asked Questions

**Version:** 1.0  
**Last Updated:** 2025-08-19

This FAQ addresses common questions about the Letta agent memory features in the Construction Claim Assistant.

---

## General Questions

### What is Letta and why is it used?

**Q**: What exactly is Letta and how does it help with construction claims?

**A**: Letta is an AI agent memory system that gives your Construction Claim Assistant a persistent memory. Unlike regular AI that forgets everything after each conversation, Letta remembers:
- Key facts from your documents
- Previous conversations about your case
- Important deadlines and requirements
- Patterns across different claims

This means your assistant gets smarter about your specific case over time, providing better analysis and more relevant suggestions.

---

### How is this different from regular AI assistants?

**Q**: How is the Letta-powered assistant different from ChatGPT or other AI tools?

**A**: Key differences:
- **Persistent Memory**: Remembers everything across all conversations
- **Case-Specific Learning**: Builds expertise in your particular matter
- **Document Integration**: Connects memories to specific pages and sources
- **Local Privacy**: All memory stored on your computer, not in the cloud
- **Legal Domain Focus**: Specialized for California construction claims

Regular AI assistants start fresh each time and can't build up knowledge about your specific case.

---

### Is my data private and secure?

**Q**: Where is my data stored and who can access it?

**A**: Your data is completely private:
- **Local Storage**: All memory data stored on your computer in `~/.letta-claim/`
- **No Cloud Sync**: Memory never leaves your machine unless you choose external LLM providers
- **Matter Isolation**: Each case has completely separate memory - no cross-contamination
- **You Control Everything**: You own all data and can delete it anytime
- **Optional External APIs**: Gemini/OpenAI only used with explicit consent, and only for generation (not memory storage)

---

## Memory and Learning

### How does the agent learn about my case?

**Q**: What information does the agent remember and how?

**A**: The agent automatically extracts and stores:
- **Entities**: Parties, agencies, contractors, amounts, dates
- **Events**: Contract changes, delays, payments, disputes
- **Legal Issues**: Claims, defenses, statutory requirements
- **Facts**: Important findings from documents and conversations

This happens automatically when you:
- Upload and process documents
- Have conversations with the assistant
- Confirm important details

---

### How long does memory last?

**Q**: Does the agent eventually forget old information?

**A**: Memory persistence depends on your settings:
- **Default**: Memories kept indefinitely
- **Optional Pruning**: Can be configured to remove memories older than X days
- **Importance-Based**: Low-importance items removed first when space is needed
- **Manual Control**: You can export, import, or clear memories anytime

Most users keep all memories for the life of their case.

---

### Can I see what the agent remembers?

**Q**: How do I know what information the agent has stored?

**A**: Several ways to view memory:
- **Memory Dashboard**: Shows count and last sync time in the right panel
- **Memory Summary**: Click "View Summary" for detailed breakdown
- **Enhanced Messages**: Purple glow indicates memory-enhanced responses
- **Memory Tooltips**: Hover over enhanced messages to see specific memories used
- **API Access**: Technical users can query memory directly via API

---

### What if the agent remembers something wrong?

**Q**: How do I correct incorrect information the agent has stored?

**A**: You can correct memories:
- **In Conversation**: Simply tell the agent "Actually, that's not correct..." and provide the right information
- **Immediate Correction**: The agent will update its memory with the correction
- **Memory Export**: Advanced users can export, edit, and re-import memory data
- **Fresh Start**: You can clear all memory and start over if needed (though this is rarely necessary)

The agent is designed to learn from corrections and improve over time.

---

## Technical Questions

### What happens if Letta server fails?

**Q**: What if the Letta server crashes or becomes unavailable?

**A**: The system has automatic fallback:
- **Graceful Degradation**: Application continues working without memory features
- **Automatic Retry**: System attempts to reconnect automatically
- **User Notification**: Clear indicators show when memory features are unavailable
- **No Data Loss**: All existing memory preserved for when service returns
- **Manual Recovery**: You can restart the server or full application if needed

Core RAG functionality (document search and analysis) continues working normally.

---

### Can I use different LLM providers with Letta?

**Q**: Do I have to use Ollama, or can I use other AI providers?

**A**: You have flexible provider options:
- **Default**: Ollama (local, private, free after setup)
- **External Options**: Gemini, OpenAI, Anthropic (with your API keys)
- **Per-Matter Settings**: Different providers for different cases
- **Automatic Fallback**: Falls back to secondary providers if primary fails
- **Cost Control**: Spending limits for external APIs

The agent memory works the same regardless of which LLM provider you choose.

---

### How much disk space does memory use?

**Q**: How much storage space will the agent memory consume?

**A**: Storage requirements are modest:
- **Typical Case**: 10-50 MB for a full construction claims matter
- **Heavy Usage**: 100-200 MB for cases with extensive documentation
- **Database**: Uses efficient SQLite compression
- **Automatic Management**: Old/low-importance items can be pruned automatically
- **Backup Space**: Backups use additional space (compressed)

Most users won't notice the storage impact.

---

### Can I backup and restore memory?

**Q**: How do I backup my agent's memory?

**A**: Multiple backup options:
- **Automatic Backups**: System creates regular backups automatically
- **Manual Export**: Export memory to JSON format anytime
- **Full System Backup**: Include `~/.letta-claim/` in your regular backups
- **Matter-Specific**: Export memory for individual matters
- **Cloud Storage**: Optionally sync backups to cloud storage (user choice)

Restoration is straightforward - import the backup file and restart the application.

---

## Setup and Configuration

### Do I need to install anything extra for Letta?

**Q**: What additional software is required for the memory features?

**A**: Minimal additional requirements:
- **Python Package**: `letta` package (installed automatically)
- **SQLite**: Already included with Python
- **System Resources**: ~1GB RAM and 100MB disk space
- **No External Services**: Everything runs locally

The main application installer handles all Letta dependencies automatically.

---

### Can I disable memory features?

**Q**: What if I don't want to use the agent memory features?

**A**: Memory features are optional:
- **Configuration Setting**: Set `letta.enabled = false` in matter configuration
- **Graceful Fallback**: Application works normally without memory
- **No Performance Impact**: Disabled features don't consume resources
- **Re-enable Anytime**: Turn memory back on whenever you want

The core document analysis features work independently of memory.

---

### How do I upgrade Letta to a newer version?

**Q**: How do I update the Letta memory system?

**A**: Upgrade process:
- **Application Updates**: Letta updated automatically with application
- **Version Compatibility**: System checks version compatibility
- **Data Migration**: Automatic migration of memory data if needed
- **Backup First**: System creates backup before any migration
- **Rollback Option**: Can revert to previous version if issues occur

Most upgrades are seamless and require no user action.

---

## California Construction Specialization

### What California-specific features does the agent have?

**Q**: How is the agent specialized for California construction law?

**A**: Extensive California specialization:
- **Statutory Recognition**: Identifies Public Contract Code, Government Code, Civil Code references
- **Entity Recognition**: Knows Caltrans, DGS, DSA, and other California agencies
- **Deadline Tracking**: Tracks statutory deadlines with day calculations
- **Compliance Checking**: Validates prevailing wage, DIR, and other requirements
- **Specialized Questions**: Generates California-specific follow-up questions
- **Legal Patterns**: Recognizes mechanics liens, stop notices, government claim procedures

---

### Does it work for other states?

**Q**: Can I use this for construction claims in other states?

**A**: Currently optimized for California:
- **California Focus**: Domain knowledge specialized for California public works
- **General Construction**: Basic construction terminology works for any state
- **Customizable**: Can be configured for other states (requires technical knowledge)
- **Future Expansion**: Other state specializations planned for future versions

For non-California cases, you'll get general construction claim assistance without state-specific expertise.

---

### What if I work on both public and private projects?

**Q**: Does the California specialization work for private construction projects?

**A**: Works for both public and private:
- **Public Works**: Full California statutory knowledge (PCC, Government Code)
- **Private Projects**: General construction law (Civil Code, mechanics liens)
- **Mixed Cases**: Handles both public and private aspects of complex projects
- **Context Aware**: Adapts advice based on whether public or private entities involved

The agent automatically adjusts its analysis based on the parties and project type it identifies.

---

## Performance and Reliability

### How fast are memory operations?

**Q**: Will memory features slow down the application?

**A**: Memory operations are designed to be fast:
- **Typical Response**: 200-500ms for memory recall
- **Batch Operations**: Multiple memories processed in parallel
- **Caching**: Frequently accessed memories cached for instant recall
- **Background Processing**: Memory storage happens asynchronously
- **UI Responsiveness**: Memory operations don't block user interface

Most users won't notice any performance impact.

---

### What happens during high memory usage?

**Q**: How does the system handle when memory gets full?

**A**: Automatic memory management:
- **Configurable Limits**: Set maximum memory items per matter
- **Importance Scoring**: Keeps most important memories when space is limited
- **Automatic Pruning**: Removes old, low-importance items first
- **User Notification**: Alerts when approaching limits
- **Manual Control**: You can manually review and clean memory

The system prevents memory exhaustion while preserving the most valuable information.

---

### Can multiple matters share memory?

**Q**: Do different cases share the same memory, or are they isolated?

**A**: Complete isolation between matters:
- **Separate Agents**: Each matter has its own dedicated agent
- **No Cross-Contamination**: Memories never mix between different cases
- **Independent Configuration**: Different LLM providers, settings per matter
- **Parallel Operation**: Multiple matters can be active simultaneously
- **Privacy Protection**: Attorney-client privilege maintained between cases

This ensures confidentiality and prevents confusion between different clients' cases.

---

## Troubleshooting

### Memory features aren't working - what should I check?

**Q**: The memory dashboard shows disconnected and features don't work.

**A**: Common troubleshooting steps:
1. **Check Server Status**: Look for red/yellow connection indicator
2. **Restart Application**: Often fixes temporary connection issues
3. **Check Logs**: Look in `~/.letta-claim/logs/` for error messages
4. **Verify Installation**: Ensure Letta package is installed correctly
5. **Test Connectivity**: Try `curl http://localhost:8283/v1/health/`
6. **Reset Configuration**: Restore default Letta settings if needed

See the full troubleshooting guide for detailed solutions.

---

### Why aren't my conversations being remembered?

**Q**: The agent doesn't seem to remember our previous conversations.

**A**: Several possible causes:
- **Connection Issues**: Check if memory dashboard shows "Connected"
- **Storage Problems**: Verify database permissions and disk space
- **Configuration**: Ensure memory storage is enabled in matter settings
- **Agent Issues**: Agent may need to be recreated
- **Memory Limits**: Check if memory capacity has been exceeded

Try refreshing the memory dashboard and check recent logs for error messages.

---

### How do I reset everything if something goes wrong?

**Q**: What if I need to completely start over with memory?

**A**: Reset options by severity:

**Soft Reset** (preserves documents):
```bash
# Clear just the memory data
rm ~/.letta-claim/letta.db
# Restart application - will recreate agent with fresh memory
```

**Medium Reset** (preserves matters, clears all memory):
```bash
rm -rf ~/.letta-claim/
# Application will reconfigure Letta from scratch
```

**Full Reset** (nuclear option):
```bash
# Backup first!
cp -r ~/LettaClaims ~/LettaClaims.backup
rm -rf ~/.letta-claim/ ~/LettaClaims/*/agent_config.json
# Complete fresh start
```

Always backup your data before any reset operation.

---

## Best Practices

### How can I get the best results from memory features?

**Q**: What practices will help the agent learn most effectively?

**A**: Memory optimization tips:
- **Be Specific**: Provide clear, specific information about facts and dates
- **Confirm Details**: When the agent asks clarifying questions, answer thoroughly
- **Correct Mistakes**: Immediately correct any misunderstandings
- **Reference Sources**: Mention document names and page numbers when possible
- **Use Follow-ups**: Click on suggested follow-up questions - they build memory
- **Review Memory**: Periodically check memory summary to ensure accuracy

---

### Should I use memory for all types of questions?

**Q**: When should I rely on memory vs. regular document search?

**A**: Use memory-enhanced features for:
- **Contextual Questions**: "What did we determine about the delay claim?"
- **Pattern Recognition**: "What contradictions have you found?"
- **Follow-up Analysis**: Building on previous conversations
- **Timeline Questions**: "What happened after the contract amendment?"
- **Comparative Analysis**: "How does this relate to the payment dispute?"

Use regular search for:
- **First-time document exploration**: Initial review of new documents
- **Specific Citations**: Finding exact quotes or references
- **Independent Analysis**: When you want fresh perspective without prior context

---

### How often should I backup memory data?

**Q**: What's a good backup strategy for agent memory?

**A**: Recommended backup frequency:
- **Automatic**: System creates backups every 30 minutes during use
- **Weekly Manual**: Export memory manually once per week for critical cases
- **Before Major Events**: Backup before depositions, mediations, or trial prep
- **System Backups**: Include `~/.letta-claim/` in regular system backups
- **Cloud Sync**: Optionally sync backups to personal cloud storage

The automatic backups are usually sufficient, but manual backups provide extra security for important cases.

---

## Integration and Workflow

### How does memory work with document upload?

**Q**: What happens to memory when I upload new documents?

**A**: Document processing automatically enhances memory:
- **Entity Extraction**: New parties, dates, amounts automatically identified and stored
- **Cross-References**: System connects new information to existing memories
- **Timeline Updates**: Events and deadlines updated with new document info
- **Contradiction Detection**: New information compared against existing memories
- **Knowledge Integration**: New facts seamlessly integrate with conversation history

You don't need to do anything special - the system handles integration automatically.

---

### Can I export memory data for use elsewhere?

**Q**: Can I get the memory data in a format for use in other tools?

**A**: Multiple export formats:
- **JSON**: Structured data for technical use or analysis
- **CSV**: Spreadsheet format for review or manipulation
- **Markdown**: Human-readable format for reports or documentation
- **Full Backup**: Complete database export for migration or archival
- **Selective Export**: Export only specific types of memories or date ranges

Exported data can be reviewed, analyzed, or imported into other systems as needed.

---

## Future Development

### What new features are planned?

**Q**: What improvements are coming to the memory system?

**A**: Planned enhancements include:
- **Visual Memory Maps**: Graphical representation of knowledge relationships
- **Advanced Timeline Views**: Interactive timeline visualization with memory integration
- **Multi-Matter Analysis**: Compare patterns and insights across related cases
- **Team Collaboration**: Share memory insights with team members
- **Enhanced Reasoning**: Improved logical inference and pattern detection
- **Additional State Support**: Specialized knowledge for other states' construction law

Feature requests and feedback help prioritize development.

---

### Can I contribute to development?

**Q**: How can I provide feedback or request features?

**A**: Several ways to contribute:
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Share ideas and best practices
- **User Feedback**: In-app feedback helps prioritize improvements
- **Beta Testing**: Join beta program for early access to new features
- **Documentation**: Suggest improvements to guides and help content

User feedback is essential for making the system better for legal professionals.

---

## Support and Resources

### Where can I get help if I have problems?

**Q**: What support resources are available?

**A**: Support options:
- **Documentation**: Comprehensive guides for configuration, troubleshooting, and API reference
- **GitHub Community**: Questions, discussions, and community support
- **Technical Support**: Enterprise users have access to direct technical support
- **User Guide**: Step-by-step instructions for all features
- **Troubleshooting Guide**: Solutions to common problems

Start with the documentation, then use community resources for additional help.

---

### How do I stay updated on new features?

**Q**: How will I know when new memory features are available?

**A**: Stay informed through:
- **Application Updates**: Automatic notification of new versions
- **Release Notes**: Detailed information about new features and improvements
- **GitHub Releases**: Technical details and changelogs
- **Documentation Updates**: Guides updated with new feature instructions
- **Community Announcements**: Important updates shared in discussions

The application will notify you when updates are available with new memory features.

---

*For additional questions not covered here, please consult the other documentation files or contact support through the appropriate channels.*