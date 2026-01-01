# Working with Memories

This guide explains how to effectively manage your memories for optimal AI personalization.

## Understanding Memories

A memory is a piece of information about you that the system stores and uses to personalize AI interactions.

### Memory Anatomy

Each memory contains:

| Field | Description | Example |
|-------|-------------|---------|
| **Content** | The actual information | "I prefer Python for data processing" |
| **Type** | Category of memory | Preference |
| **Tier** | Storage level | Long-term |
| **Confidence** | Certainty score (0-1) | 0.85 |
| **Domains** | Topic areas | ["programming", "python"] |
| **Tags** | Organization labels | ["language", "preference"] |
| **Source** | Where it came from | "manual", "claude-history" |

### Memory Types

| Type | Description | Examples |
|------|-------------|----------|
| **Fact** | Verifiable information | "My birthday is June 15th" |
| **Preference** | Likes and dislikes | "I prefer dark mode in editors" |
| **Belief** | Opinions and values | "Code should be self-documenting" |
| **Skill** | Capabilities | "I'm proficient in TypeScript" |
| **Event** | Things that happened | "I attended PyCon 2024" |
| **Context** | Background info | "I work in healthcare IT" |

### Memory Tiers

The system uses a three-tier memory hierarchy inspired by human cognition:

#### Short-term Memory
- **Purpose:** Temporary storage for new, unvalidated information
- **Duration:** 24-72 hours unless promoted
- **Use case:** Session context, quick notes, unconfirmed observations

#### Long-term Memory
- **Purpose:** Validated, searchable knowledge
- **Duration:** Indefinite until contradicted
- **Use case:** Confirmed facts, stable preferences, domain knowledge

#### Persistent Memory
- **Purpose:** Core identity and absolute truths
- **Duration:** Permanent
- **Use case:** Fundamental facts about you, unchanging preferences

### Automatic Promotion

Memories automatically promote between tiers based on:

**Short-term → Long-term:**
- Confirmed 2+ times from different sources
- Persisted 3+ days without contradiction
- Explicitly validated by you

**Long-term → Persistent:**
- Consistent for 30+ days
- Accessed 10+ times
- Marked as identity/core by you

## Adding Memories

### Manual Entry

The most direct way to add memories:

1. Go to **Memories** tab
2. Click **Add Memory**
3. Fill in the form:
   - Enter clear, specific content
   - Select appropriate type
   - Add relevant tags
   - Set initial confidence (default: 0.5)
4. Click **Save**

### Best Practices for Manual Entry

**Do:**
```
"I prefer composition over inheritance in OOP"
"My primary programming language is Python"
"I have 10 years of experience with Drupal"
```

**Avoid:**
```
"I like coding" (too vague)
"Python" (not a complete statement)
"I'm the best programmer" (not useful for context)
```

### Automatic Ingestion

Import memories from external sources:

1. Go to **Connections**
2. Connect your data sources
3. Run ingestion
4. Review imported memories

Sources include:
- Claude/ChatGPT conversation history
- Git repositories
- Documents and notes
- Browser history

### From AI Conversations

The system can extract memories from your chats:

1. Have a conversation with the AI
2. When you share facts about yourself, they may be captured
3. Review and validate in the Memories tab

## Searching Memories

### Basic Search

Type in the search box to find memories. The search uses:

- **Semantic matching:** Finds conceptually similar memories
- **Keyword matching:** Exact text matches
- **Hybrid ranking:** Combines both for best results

### Search Examples

| Query | Finds |
|-------|-------|
| "coding preferences" | Memories about programming style |
| "Python" | All memories mentioning Python |
| "what I like about work" | Job satisfaction memories |

### Filtering

Use filters to narrow results:

- **Tier:** Short-term, Long-term, Persistent
- **Type:** Fact, Preference, Belief, Skill, Event, Context
- **Domain:** Programming, Work, Personal, etc.
- **Confidence:** Minimum confidence threshold
- **Date range:** When the memory was created

## Organizing Memories

### Using Tags

Tags help categorize and find memories:

```
memory: "I prefer TypeScript over JavaScript"
tags: ["programming", "typescript", "preference", "web-development"]
```

**Tag strategies:**
- Use consistent naming (lowercase, hyphens)
- Create hierarchies: "work", "work-projects", "work-meetings"
- Include technology names: "python", "react", "aws"

### Using Domains

Domains represent knowledge areas:

- **Programming:** Technical skills and preferences
- **Work:** Professional context
- **Personal:** Non-work information
- **Health:** Health-related information
- **Finance:** Financial preferences and facts

### Grouping Related Memories

Create memory clusters for related information:

```
Domain: "drupal"
Memories:
- "I have 15 years of Drupal experience"
- "I prefer Drupal for government websites"
- "I specialize in Drupal content migration"
```

## Curating Memories

### Validating Memories

Review and validate memories for accuracy:

1. Browse the Memories tab
2. Check the content of each memory
3. Mark accurate memories as validated (increases confidence)
4. Delete or correct inaccurate ones

### Resolving Conflicts

When memories contradict each other:

1. The system flags potential conflicts
2. Review both memories
3. Keep the correct one, delete the other
4. Or update to a more accurate statement

### Removing Outdated Information

Memories can become outdated:

```
Old: "I work at Acme Corp"
New: "I work at TechStart Inc" (after job change)
```

**Best practice:** Delete outdated memories rather than leaving them.

## Privacy Considerations

### What's Stored

All memories are stored locally on your machine:
- SQLite database for persistent/long-term
- In-memory cache for short-term
- Vector embeddings for semantic search

### What's Not Stored

- Raw source files (only extracted content)
- Passwords or sensitive credentials
- Financial account numbers

### Controlling What's Captured

You can:
- Disable specific data sources
- Review and delete unwanted memories
- Set up exclusion rules for ingestion

## Advanced Tips

### Memory Quality

High-quality memories lead to better AI personalization:

1. **Be specific:** "I prefer 4-space indentation in Python" vs "I like clean code"
2. **Add context:** Include why, when, or where when relevant
3. **Use proper types:** Classify memories correctly
4. **Keep current:** Remove outdated information

### Memory Density

Balance between:
- **Too few:** AI lacks context for personalization
- **Too many:** Noise can dilute relevance

**Recommendation:** Aim for 100-500 high-quality memories covering your key domains.

### Domain Coverage

Ensure coverage across your important areas:

```
Work:       50 memories (roles, preferences, skills)
Programming: 80 memories (languages, tools, patterns)
Personal:   30 memories (interests, background)
```

## Troubleshooting

### Memories Not Appearing in Search

- Check your filters (reset if needed)
- Try different search terms
- Verify the memory exists in the list

### Duplicate Memories

- The system deduplicates on import
- Manual duplicates should be deleted
- Use the hash column to identify exact duplicates

### Wrong Information Being Used

- Review and correct the inaccurate memory
- Adjust confidence scores
- Delete memories that cause problems

## Related Guides

- [Getting Started](GETTING_STARTED.md) - Initial setup
- [Data Sources](DATA_SOURCES.md) - Automatic ingestion
- [AI Assistant](AI_ASSISTANT.md) - Using memories with AI
