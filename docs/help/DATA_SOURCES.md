# Data Sources Guide

This guide explains how to connect external data sources and automatically import memories.

## Overview

Data sources allow you to automatically extract memories from your digital life:

- AI conversation history (Claude, ChatGPT)
- Code repositories (Git, GitHub)
- Documents and notes
- Browser history
- And more

## Available Sources

### Claude Code History

**What it captures:**
- Your conversations with Claude
- Questions you've asked
- Preferences you've expressed
- Code patterns you use

**Setup:**
1. Go to **Connections**
2. Find "Claude History"
3. Click **Connect**
4. Click **Ingest** to import

**Location:** `~/.claude/` directory

### Git Repositories

**What it captures:**
- Commit messages
- Code comments
- Project structures
- Programming patterns

**Setup:**
1. Go to **Connections**
2. Find "Git Repos"
3. Click **Configure**
4. Enter path to your code directory (e.g., `~/code`)
5. Click **Ingest**

**Options:**
- `path` - Directory containing repos
- `recursive` - Search subdirectories
- `max_depth` - How deep to search

### Documents

**What it captures:**
- Text content from documents
- Notes and memos
- PDF text extraction
- Markdown files

**Supported formats:**
- `.txt` - Plain text
- `.md` - Markdown
- `.pdf` - PDF documents
- `.docx` - Word documents

**Setup:**
1. Go to **Connections**
2. Find "Documents"
3. Click **Configure**
4. Enter path to documents folder
5. Select file types to include
6. Click **Ingest**

### Browser History (Coming Soon)

**Chrome:**
- Browsing patterns
- Frequently visited sites
- Search queries

**Safari:**
- Similar capabilities
- macOS integration

### Obsidian Notes

**What it captures:**
- All your notes
- Tags and links
- Knowledge structure

**Setup:**
1. Go to **Connections**
2. Find "Obsidian"
3. Enter vault path
4. Click **Ingest**

### Apple Notes (Coming Soon)

**What it captures:**
- Personal notes
- Lists and reminders
- Attached content

## Running Ingestion

### Manual Ingestion

1. Go to **Connections**
2. Click **Run Ingestion** on a source
3. Monitor progress in the status bar
4. Review imported memories

### Batch Ingestion

1. Click **Run All**
2. All configured sources process sequentially
3. View combined results

### Ingestion Status

During ingestion, you'll see:

| Metric | Description |
|--------|-------------|
| Processed | Items scanned |
| Created | New memories added |
| Updated | Existing memories modified |
| Skipped | Duplicates or excluded |
| Errors | Failed items |

## What Gets Extracted

### From Conversations

- Direct statements about yourself
- Expressed preferences
- Work context
- Technical knowledge

Example extraction:
```
Conversation: "I usually use pytest for testing Python code"
Memory: "Prefers pytest for Python testing" (Type: Preference)
```

### From Code

- Programming language usage
- Library preferences
- Architecture patterns
- Commit message style

Example extraction:
```
Commit: "Refactored to use dependency injection"
Memory: "Uses dependency injection pattern" (Type: Skill)
```

### From Documents

- Stated facts and preferences
- Work history
- Project details
- Knowledge domain

## Filtering Ingestion

### Include Patterns

Specify what to include:
```
*.py          # Only Python files
**/src/**     # Only src directories
*.md          # Only Markdown
```

### Exclude Patterns

Specify what to skip:
```
**/node_modules/**   # Skip dependencies
**/.git/**           # Skip git internals
*.log                # Skip log files
```

### Privacy Exclusions

Automatically excluded:
- `.env` files
- `credentials.*`
- `secrets.*`
- Password files

## Deduplication

The system prevents duplicate memories:

1. **Content hash** - Exact matches detected
2. **Semantic similarity** - Near-duplicates flagged
3. **Source tracking** - Same source item not re-imported

## Scheduling (Future)

Planned automatic ingestion:

- Daily incremental sync
- Weekly full rescan
- On-demand triggers

## Troubleshooting

### Source Not Found

1. Verify the path exists
2. Check file permissions
3. Ensure source is installed (e.g., Claude Code)

### No Memories Created

1. Check if content is extractable
2. Verify file formats are supported
3. Review exclusion patterns
4. Check for empty files

### Permission Errors

On macOS, grant access in:
- System Preferences > Privacy & Security
- Full Disk Access (for some sources)

### Slow Ingestion

Large sources take time:
- First run scans everything
- Subsequent runs are incremental
- Consider limiting scope initially

## Source Configuration

### Per-Source Settings

Each source has configurable options:

**Git Repos:**
```json
{
  "path": "~/code",
  "include_commits": true,
  "include_code": true,
  "languages": ["python", "javascript"],
  "max_file_size": "1MB"
}
```

**Documents:**
```json
{
  "path": "~/Documents",
  "extensions": [".md", ".txt", ".pdf"],
  "recursive": true,
  "max_depth": 5
}
```

### Global Settings

Apply to all sources:
- Maximum items per run
- Memory type preferences
- Confidence thresholds

## Best Practices

### Start Small

1. Begin with one source (e.g., Claude history)
2. Review the imported memories
3. Adjust settings as needed
4. Add more sources gradually

### Review Imported Data

After ingestion:
1. Go to **Memories**
2. Filter by source
3. Verify quality and accuracy
4. Delete incorrect memories

### Maintain Sources

Keep your sources clean:
- Organize documents in folders
- Use meaningful commit messages
- Keep note structures logical

### Privacy Awareness

Before ingesting:
- Review what data will be captured
- Exclude sensitive directories
- Check for credentials in documents

## Adding Custom Sources

For developers, custom sources can be added:

1. Create a source adapter class
2. Implement the extract method
3. Register in the source registry
4. Configure via the UI or config file

See [API Documentation](../API.md) for details.

## Related Guides

- [Managing Memories](MEMORIES.md) - Organize imported data
- [Getting Started](GETTING_STARTED.md) - Initial setup
- [User Guide](USER_GUIDE.md) - Complete reference
