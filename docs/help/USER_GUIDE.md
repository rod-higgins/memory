# User Guide

This comprehensive guide covers all features of the Personal Memory System web interface.

## Interface Overview

### Navigation Sidebar

The left sidebar provides access to all main features:

- **My AI** - Interactive AI chat with memory context
- **Memories** - Browse, search, and manage memories
- **Connections** - Data source integrations
- **LLMs** - Large Language Model configuration
- **SLMs** - Small/Local Language Model management
- **Admin** - User management (admin users only)

### Header Bar

- **Search** - Global search across all memories (⌘K)
- **Help** - Access documentation and guides (?)
- **User Menu** - Account settings, password change, logout

## My AI (Chat)

The AI chat interface lets you interact with AI assistants enhanced by your personal memory.

### Starting a Conversation

1. Select an AI provider from the dropdown (Claude, GPT, local model)
2. Type your message in the input box
3. Press Enter or click Send

### How Memory Context Works

When you send a message, the system:
1. Analyzes your query for relevant topics
2. Retrieves matching memories from your store
3. Injects this context into the AI prompt
4. Returns a personalized response

### Chat Features

| Feature | Description |
|---------|-------------|
| Provider Selection | Choose between configured AI providers |
| Memory Injection | Automatic context from your memories |
| Conversation History | View and continue past conversations |
| Copy Response | Copy AI responses to clipboard |

### Tips for Better Responses

- **Ask about yourself** - "What do you know about my coding style?"
- **Reference past context** - "Based on my preferences, how should I..."
- **Request personalization** - "Given my experience with Drupal..."

## Memories

The Memories section is where you browse, search, and manage your stored information.

### Memory List

Memories are displayed in a table showing:
- **Content** - The memory text (truncated)
- **Type** - Fact, Preference, Belief, Skill, Event, Context
- **Tier** - Short-term, Long-term, or Persistent
- **Confidence** - How certain the system is (0-100%)
- **Created** - When the memory was added

### Filtering Memories

Use the filter dropdowns to narrow your view:

- **Tier** - Filter by storage tier
- **Type** - Filter by memory type
- **Domain** - Filter by knowledge domain

### Searching Memories

Type in the search box to find memories by content. The search uses:
- **Semantic search** - Finds conceptually related memories
- **Keyword matching** - Exact text matches

### Adding Memories

1. Click **Add Memory**
2. Enter the memory content
3. Select the memory type:
   - **Fact** - Verifiable information
   - **Preference** - Your likes/dislikes
   - **Belief** - Your opinions and values
   - **Skill** - Your capabilities
   - **Event** - Things that happened
   - **Context** - Background information
4. Add tags for organization
5. Click **Save**

### Editing Memories

1. Click on a memory row to select it
2. Click **Edit** or double-click
3. Modify the content or metadata
4. Click **Save**

### Deleting Memories

1. Select the memory
2. Click **Delete**
3. Confirm the deletion

**Note:** Deleted memories cannot be recovered.

### Memory Tiers

| Tier | Purpose | Promotion |
|------|---------|-----------|
| **Short-term** | Temporary, unvalidated | Auto-promotes after validation |
| **Long-term** | Validated, searchable | Promotes based on stability |
| **Persistent** | Core identity, permanent | Manually managed |

## Connections

Connect external data sources to automatically import memories.

### Available Sources

| Source | What It Imports |
|--------|-----------------|
| Claude History | Your Claude Code conversation history |
| Git Repos | Commit messages, code patterns |
| Documents | PDFs, text files, notes |
| Chrome History | Browsing patterns (coming soon) |
| Safari History | Browsing patterns (coming soon) |

### Connecting a Source

1. Go to **Connections**
2. Find the source you want to connect
3. Click **Connect** or **Configure**
4. Follow the source-specific setup instructions
5. Click **Ingest** to import data

### Running Ingestion

- **Manual** - Click "Run Ingestion" to import now
- **Scheduled** - Some sources support automatic periodic imports

### Monitoring Progress

The ingestion progress shows:
- Total items processed
- New memories created
- Errors encountered

## LLM Configuration

Configure Large Language Model providers for the AI chat.

### Supported Providers

| Provider | Models | Requirements |
|----------|--------|--------------|
| Claude (Anthropic) | Opus, Sonnet, Haiku | API key |
| OpenAI | GPT-4, GPT-3.5 | API key |
| Local (Ollama) | Various | Ollama installed |

### Adding a Provider

1. Go to **LLMs**
2. Click **Add Provider**
3. Select the provider type
4. Enter your API key
5. Select available models
6. Click **Save**

### Setting Default Model

1. Find the provider in the list
2. Click **Set as Default** on the desired model

## SLM Management

Manage local Small Language Models for privacy-focused inference.

### What Are SLMs?

Small Language Models run entirely on your machine, ensuring complete privacy. They're smaller than cloud models but can be personalized through training.

### Ollama Integration

The system integrates with Ollama for local model management:

1. **View Models** - See installed Ollama models
2. **Pull Models** - Download new models
3. **Test Models** - Verify model functionality

### Available Models

Popular models for personal use:
- **Phi-2** - Fast, efficient for simple tasks
- **Llama 3.2** - Good balance of size and capability
- **Mistral** - Strong reasoning ability

## Tools

### Context Generator

Generate memory context for external use:

1. Click the **Context Generator** tool
2. Enter a query or topic
3. Set maximum memories to include
4. Click **Generate**
5. Copy the formatted context

### Ingestion Runner

Run data ingestion manually:

1. Click **Run Ingestion**
2. Select sources to process
3. Monitor progress
4. Review imported memories

### Memory Synthesis

Analyze and synthesize insights from your memories:

1. Click **Synthesize**
2. Choose analysis type
3. View generated insights

## Settings

### Change Password

1. Click your username in the top-right
2. Select **Change Password**
3. Enter current password
4. Enter and confirm new password
5. Click **Change Password**

### Account Information

View your account details:
- Username
- Email
- Role (User/Admin)
- Last login

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| ⌘K | Global search |
| Escape | Close modals |
| Enter | Submit forms |

## Troubleshooting

### Can't Log In

- Verify your username and password
- Check that 2FA code is current (codes change every 30 seconds)
- Contact an admin if you've lost your authenticator

### Memories Not Found

- Check filter settings (reset filters)
- Try different search terms
- Verify memory wasn't deleted

### AI Not Responding

- Check LLM provider configuration
- Verify API key is valid
- Try a different provider

### Ingestion Failing

- Check source permissions
- Verify file paths exist
- Review error messages in logs

## Getting More Help

- Check the [Getting Started](GETTING_STARTED.md) guide
- Review the [API Documentation](../API.md)
- Report issues at https://github.com/rod-higgins/memory/issues
