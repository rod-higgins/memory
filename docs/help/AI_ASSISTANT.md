# AI Assistant Guide

The AI Assistant is the heart of the Personal Memory System - an AI chat enhanced with your personal context.

## How It Works

When you chat with the AI:

1. **Query Analysis** - Your message is analyzed for intent and topics
2. **Memory Retrieval** - Relevant memories are fetched from your store
3. **Context Injection** - Memories are formatted and added to the prompt
4. **AI Response** - The enhanced prompt is sent to the AI provider
5. **Learning** - New facts about you may be extracted and stored

```
Your Question → [Memory Lookup] → [Context + Question] → AI → Response
                     ↓
              Your Memories
```

## Using the Chat

### Starting a Conversation

1. Click **My AI** in the sidebar
2. Select an AI provider from the dropdown
3. Type your message in the input box
4. Press Enter or click Send

### Choosing an AI Provider

| Provider | Best For | Speed | Cost |
|----------|----------|-------|------|
| Claude Opus | Complex reasoning | Slower | Higher |
| Claude Sonnet | Balanced tasks | Medium | Medium |
| Claude Haiku | Quick questions | Fast | Lower |
| GPT-4 | General tasks | Medium | Higher |
| GPT-3.5 | Simple queries | Fast | Lower |
| Local (Ollama) | Privacy | Varies | Free |

### Chat Features

- **Provider Selection** - Switch between configured providers
- **Conversation History** - View past conversations
- **Copy Response** - Copy AI responses to clipboard
- **Clear Chat** - Start a fresh conversation

## Getting Personalized Responses

The magic happens when AI uses your memories for context.

### Ask About Yourself

```
You: "What do you know about my programming preferences?"

AI: "Based on your memories, you prefer:
- Python for data processing
- TypeScript for frontend work
- Composition over inheritance
- 4-space indentation
You've also mentioned you're proficient in Drupal..."
```

### Request Personalized Advice

```
You: "How should I structure this new Python project?"

AI: "Given your preference for clean architecture and
your experience with FastAPI (as noted in your memories),
I'd suggest:
1. Use a src/ layout since you prefer that...
2. FastAPI for the API layer...
3. SQLAlchemy for ORM since you're familiar with it..."
```

### Reference Past Context

```
You: "Based on my work experience, what jobs should I target?"

AI: "Your memories indicate 15 years of Drupal experience
and specialization in government websites. Combined with
your Python skills and architecture background, you'd be
well-suited for:
- Senior Drupal Architect roles
- Government tech consulting
- Digital services modernization..."
```

## Effective Prompting

### Ask About Preferences

```
Good: "What code style do I prefer?"
Good: "How do I usually handle error logging?"
Good: "What's my typical project structure?"
```

### Request Contextual Help

```
Good: "Help me write a function that matches my coding style"
Good: "Review this code based on my preferences"
Good: "Suggest improvements consistent with my approach"
```

### Leverage Your History

```
Good: "In my past projects, how did I handle authentication?"
Good: "What patterns have I used for database migrations?"
Good: "Based on my experience, what could go wrong here?"
```

## Memory Context Window

The system includes relevant memories in your prompts, limited by:

- **Max memories:** Typically 10-20 most relevant
- **Token budget:** Stays within provider limits
- **Relevance score:** Only memories above threshold

### What Gets Included

1. **Highly relevant** - Direct topic matches
2. **Supporting context** - Related background
3. **Identity info** - Core facts about you
4. **Recent history** - Recent conversation context

### Viewing Used Context

The Context Generator tool shows what context would be included:

1. Click **Context Generator** in the toolbar
2. Enter a query
3. See the formatted context that would be sent

## Conversation Modes

### Single Query Mode

Ask one question, get one answer. Good for:
- Quick lookups
- Simple questions
- Testing responses

### Conversation Mode

Extended back-and-forth. Good for:
- Complex problem-solving
- Iterative refinement
- Deep discussions

### Compare Mode

Compare responses across providers:

1. Enable **Compare Providers**
2. Enter your query
3. See side-by-side responses from multiple AI providers

## Tips for Better Results

### 1. Build Your Memory Base

The more relevant memories you have, the better the personalization.

- Add your key preferences
- Include your expertise areas
- Document your work context

### 2. Be Specific in Questions

```
Vague: "Help me with code"
Better: "Help me refactor this Python function to match my style"
```

### 3. Reference Your Context

```
"Given what you know about my work..."
"Based on my preferences..."
"Considering my experience with..."
```

### 4. Provide Feedback

When the AI gets something wrong:
- Correct it in the chat
- Update the relevant memory
- The system learns from corrections

### 5. Use the Right Provider

- **Complex tasks:** Claude Opus or GPT-4
- **Quick questions:** Claude Haiku or GPT-3.5
- **Privacy-sensitive:** Local Ollama models

## Troubleshooting

### AI Not Using My Memories

1. Check that relevant memories exist
2. Verify memories have appropriate tags/domains
3. Try more specific questions
4. Review memory confidence scores

### Responses Too Generic

1. Add more specific memories
2. Ask more targeted questions
3. Reference your context explicitly
4. Check your memory coverage

### Wrong Information Used

1. Review the specific memory
2. Correct or delete if inaccurate
3. Add correct information
4. Lower confidence on problematic memories

### Provider Errors

1. Verify API key is valid
2. Check provider status
3. Try a different provider
4. Review error messages

## Privacy Notes

### What's Sent to AI Providers

- Your message text
- Selected memory context
- System prompts

### What Stays Local

- Your full memory database
- Conversation history (local storage)
- Personal identifiers (not included in context)

### For Maximum Privacy

Use Ollama with local models:
- No data leaves your machine
- Completely offline capable
- Full control over the model

## Related Guides

- [Managing Memories](MEMORIES.md) - Build your context base
- [LLM Configuration](USER_GUIDE.md#llm-configuration) - Set up providers
- [Getting Started](GETTING_STARTED.md) - Initial setup
