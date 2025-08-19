# Letta Agent Memory Features - User Guide

**Version:** 1.0  
**Last Updated:** 2025-08-19

This guide explains how to use the Letta agent memory features in the Construction Claim Assistant to enhance your case analysis with persistent AI memory that learns and improves over time.

---

## What is Letta Agent Memory?

Letta provides your Construction Claim Assistant with a **persistent memory system** that remembers important information from your conversations and documents across all interactions. Unlike standard AI assistants that forget everything after each conversation, your Letta agent builds up knowledge about your specific matter over time.

### Key Benefits

- **Contextual Awareness**: The agent remembers previous conversations, key facts, and important details
- **Learning Over Time**: Analysis quality improves as the agent accumulates more knowledge about your case
- **Proactive Insights**: The agent can connect patterns and suggest relevant follow-up questions
- **Case Continuity**: No need to re-explain context in each new conversation

---

## Understanding Memory-Enhanced Responses

### Visual Indicators

When the agent uses its memory to enhance responses, you'll see several visual indicators:

#### Memory Status Badge
- **Purple badge** appears during memory operations
- Shows "Using Memory..." when actively recalling information
- Changes to green "Memory Enhanced" when complete

#### Memory Dashboard
Located in the right panel, shows:
- **Memory Items**: Total number of facts stored
- **Connection Status**: Agent server connection (green/yellow/red)
- **Last Sync**: When memory was last updated
- **Memory Usage**: Visual progress bar

#### Message Enhancement
- **Purple glow**: Messages enhanced with memory context
- **Memory tooltip**: Hover over enhanced messages to see memory items used
- **Follow-up suggestions**: Contextually relevant questions based on memory

### Memory Operations Notifications
- **Bottom-right toasts** show memory operations in real-time
- **"Storing to memory..."** when saving new information
- **"Recalling from memory..."** when retrieving context
- **Success/error notifications** for completed operations

---

## How Memory Works in Your Workflow

### 1. Initial Document Processing
When you upload documents:
- Agent extracts key entities (parties, dates, amounts, statutes)
- Important facts are automatically stored in memory
- California-specific legal entities and deadlines are identified
- Cross-references between documents are noted

### 2. Chat Interactions
During conversations:
- **Context Recall**: Agent remembers previous discussions about similar topics
- **Knowledge Building**: New facts mentioned are added to memory
- **Pattern Recognition**: Agent connects information across conversations
- **Proactive Questions**: Suggests follow-ups based on memory patterns

### 3. Memory Evolution
Over time, your agent:
- **Builds expertise** in your specific case details
- **Identifies contradictions** between different sources
- **Suggests missing documentation** based on typical claim patterns
- **Provides increasingly sophisticated analysis** as knowledge accumulates

---

## Using the Memory Dashboard

### Accessing Memory Statistics
1. Look at the **right panel** in the main interface
2. Find the **"Agent Memory"** section
3. Click the **refresh button** to update statistics manually
4. Statistics auto-refresh every 30 seconds

### Understanding the Metrics

#### Memory Items Count
- **Total facts stored** in agent memory
- Includes entities, events, legal issues, and factual findings
- Higher counts generally mean better context for responses

#### Connection Status
- **Green circle**: Letta server connected and operational
- **Yellow pulse**: Server connecting or experiencing issues
- **Red circle**: Server disconnected (fallback mode active)

#### Last Sync Time
- Shows when memory was last updated
- "Just now" for recent activity
- "Xm ago" / "Xh ago" for older syncs

#### Memory Usage Bar
- **Visual indicator** of memory utilization
- Estimated based on stored items (100 items ≈ 50% full)
- System automatically manages memory size

### Memory Summary Dialog
- Click **"View Summary"** to see memory contents
- Shows recent memories, key entities, and patterns
- Useful for understanding what the agent knows about your case

---

## Agent Health Monitoring

### Health Indicator Location
The agent health indicator appears in the **application header** and shows:
- **Green check**: Agent healthy and fully operational
- **Yellow warning**: Agent experiencing minor issues but functional
- **Red error**: Agent errors detected, may be in fallback mode
- **Gray help**: Agent status unknown

### Detailed Health Information
Click the **info button** next to the health indicator to see:
- **Connection Status**: Server connectivity details
- **Server Status**: Letta server operational state
- **Provider**: LLM provider being used (Ollama/Gemini/etc.)
- **Response Time**: Average response latency
- **Memory Operations**: Memory system availability

---

## Memory Context in Chat

### Identifying Memory-Enhanced Messages
Look for these indicators in your chat:
- **Purple glow** around message bubbles
- **"Memory Enhanced" badge** next to responses
- **Follow-up suggestions** that reference previous conversations

### Understanding Memory Context
When you see a memory-enhanced response:
1. The agent has **recalled relevant information** from previous interactions
2. **Cross-referenced** your question with stored knowledge
3. **Integrated context** from multiple conversations or documents
4. **Provided richer analysis** than would be possible without memory

### Memory Tooltips
- **Hover** over enhanced messages to see memory context
- Shows **bullet points** of specific memories used
- Helps understand **how the agent** arrived at its response

---

## Best Practices for Memory-Enhanced Workflows

### 1. Building Quality Memory
- **Be specific** when discussing key facts, dates, and parties
- **Confirm important details** when the agent asks clarifying questions
- **Correct mistakes** immediately to prevent them from being stored
- **Reference document names** and page numbers when citing sources

### 2. Leveraging Memory for Analysis
- **Ask follow-up questions** that build on previous conversations
- **Reference earlier discussions**: "What did we determine about the delay claim?"
- **Request comparisons**: "How does this relate to the previous payment dispute?"
- **Seek patterns**: "What contradictions have you identified across documents?"

### 3. Managing Memory Quality
- **Review memory summaries** periodically to ensure accuracy
- **Correct misunderstandings** when you notice them
- **Provide feedback** when the agent's memory seems incomplete
- **Use the refresh function** if memory seems stale

### 4. Optimizing Performance
- **Allow time** for memory operations to complete (purple badges)
- **Don't interrupt** memory-enhanced responses while they're generating
- **Check connection status** if responses seem less contextual than expected
- **Restart the application** if memory features become unresponsive

---

## California Public Works Specialization

Your Letta agent has been specially trained for **California public works construction claims** and will automatically:

### Legal Entity Recognition
- **Identify public agencies**: Caltrans, DGS, DSA, counties, school districts
- **Track special requirements** for each type of public entity
- **Note compliance obligations** specific to California public works

### Statutory Deadline Tracking
- **Extract deadlines** from California Public Contract Code
- **Calculate days remaining** for critical filings
- **Warn about missed deadlines** and their consequences
- **Suggest required notices** and procedural steps

### Compliance Validation
- **Check prevailing wage requirements** and DIR compliance
- **Validate government claim prerequisites** per Gov. Code §910-915
- **Identify mechanics lien** and stop notice requirements
- **Flag False Claims Act** implications

### Expert Analysis Triggers
- **Suggest expert consultations** when complex issues arise
- **Identify when specialized analysis** is needed (quantum, delay, etc.)
- **Recommend document requests** based on claim type patterns

---

## Privacy and Data Security

### Local-First Architecture
- **All memory data** stored locally on your computer
- **No cloud synchronization** unless you explicitly configure external providers
- **Complete data control** - you own all information

### Data Isolation
- **Each matter** has completely separate memory and agent
- **No cross-contamination** between different cases
- **Independent LLM configurations** per matter if desired

### External Provider Consent
- **Explicit consent required** for external LLM providers (Gemini, etc.)
- **Local Ollama preferred** for maximum privacy
- **Clear notifications** when data would be sent externally

---

## Troubleshooting Memory Features

### Common Issues and Solutions

#### Memory Badge Not Appearing
- **Check connection status** in memory dashboard
- **Restart application** if Letta server is unresponsive
- **Verify Letta installation** in system health check

#### Responses Seem Less Contextual
- **Check memory item count** - may be too low for good context
- **Review connection status** - server may be disconnected
- **Try manual refresh** of memory statistics

#### Memory Operations Timing Out
- **Check system resources** - Letta server may be overloaded
- **Reduce concurrent operations** - wait for current operation to finish
- **Restart Letta server** through application restart

#### Memory Summary Shows Incorrect Information
- **Provide corrections** in your next conversation
- **Ask agent to update** specific memories
- **Consider clearing memory** if corruption is extensive

### Getting Help
- **Check the health indicator** for system status
- **Review application logs** for error details
- **Consult troubleshooting guide** for detailed solutions
- **Report persistent issues** through support channels

---

## Advanced Memory Features

### Memory Export and Backup
- **Export memory** to JSON format for backup
- **Import memory** from previous exports
- **Transfer knowledge** between matter versions

### Memory Analytics
- **Pattern detection** across conversations and documents
- **Key actor identification** and relationship mapping
- **Temporal analysis** of claim development
- **Document focus analysis** and gap identification

### Custom Follow-up Templates
- **California-specific questions** automatically generated
- **Context-aware suggestions** based on stored knowledge
- **Priority-based ranking** of most relevant follow-ups
- **Expert consultation triggers** for specialized needs

---

## Future Enhancements

The Letta memory system is continuously evolving. Planned enhancements include:
- **Visual memory maps** showing knowledge relationships
- **Advanced timeline visualization** with memory integration
- **Multi-matter knowledge comparison** and pattern analysis
- **Collaborative memory sharing** for team environments

---

## Support and Feedback

### Documentation Resources
- **Configuration Guide**: Detailed setup instructions
- **API Reference**: Technical integration details
- **Troubleshooting Guide**: Common issues and solutions
- **FAQ**: Frequently asked questions

### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Community Q&A and best practices
- **Technical Reference**: Implementation details for developers

---

*This user guide covers the essential features of the Letta memory system. For technical implementation details, see the Letta Technical Reference and API documentation.*