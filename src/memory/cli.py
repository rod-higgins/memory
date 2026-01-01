"""
Command-line interface for the Personal Memory System.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="memory",
    help="Personal Memory System - Three-tier memory for AI interactions",
)
console = Console()


@app.command()
def info():
    """Show information about available data sources."""

    async def _info():
        from memory.ingestion.coordinator import IngestionCoordinator
        from memory.storage.manager import StorageManager

        storage = StorageManager()
        await storage.initialize()

        coordinator = IngestionCoordinator(storage)
        source_info = await coordinator.get_source_info()

        await storage.close()

        console.print("\n[bold]Available Data Sources[/bold]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Source")
        table.add_column("Status")
        table.add_column("Entries")
        table.add_column("Date Range")

        for source, info in source_info.items():
            status = "[green]Available[/green]" if info.get("exists") else "[red]Not Found[/red]"
            entries = str(info.get("entry_count", "-"))
            first = info.get("first_entry", "")
            last = info.get("last_entry", "")
            date_range = f"{first[:10]} to {last[:10]}" if first and last else "-"

            table.add_row(source, status, entries, date_range)

        console.print(table)

    asyncio.run(_info())


@app.command()
def ingest(
    source: str = typer.Argument("claude-history", help="Source to ingest from"),
    max_entries: int = typer.Option(None, "--max", "-m", help="Maximum entries to process"),
    skip_embeddings: bool = typer.Option(False, "--skip-embeddings", help="Skip embedding generation"),
):
    """Ingest memories from a data source."""

    async def _ingest():
        from memory.ingestion.coordinator import IngestionCoordinator
        from memory.ingestion.enrichment import SimpleEnrichmentPipeline
        from memory.storage.manager import StorageManager

        storage = StorageManager()
        await storage.initialize()

        # Use simple enrichment unless we have Ollama
        enrichment = SimpleEnrichmentPipeline()
        coordinator = IngestionCoordinator(storage, enrichment)

        console.print(f"\n[bold]Ingesting from {source}[/bold]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if source == "claude-history":
                task = progress.add_task("Processing Claude history...", total=None)

                stats = await coordinator.ingest_claude_history(
                    max_entries=max_entries,
                    skip_enrichment=skip_embeddings,
                )

                progress.update(task, completed=True)
            else:
                console.print(f"[red]Unknown source: {source}[/red]")
                await storage.close()
                return

        await storage.close()

        # Show results
        console.print("\n[bold green]Ingestion Complete[/bold green]\n")

        table = Table(show_header=True)
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Processed", str(stats.total_processed))
        table.add_row("Duplicates Skipped", str(stats.duplicates_skipped))
        table.add_row("Errors", str(stats.errors))
        table.add_row("Duration", f"{stats.duration_seconds:.2f}s")

        console.print(table)

    asyncio.run(_ingest())


@app.command()
def stats():
    """Show memory statistics."""

    async def _stats():
        from memory.storage.manager import StorageManager

        storage = StorageManager()
        await storage.initialize()

        info = await storage.get_stats()

        await storage.close()

        console.print("\n[bold]Memory System Statistics[/bold]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Tier")
        table.add_column("Backend")
        table.add_column("Count")
        table.add_column("Path")

        for tier_name, tier_info in info["tiers"].items():
            table.add_row(
                tier_name,
                tier_info["backend"],
                str(tier_info["count"]),
                tier_info.get("path", "in-memory"),
            )

        console.print(table)
        console.print(f"\n[bold]Total Memories:[/bold] {info['counts'].get('total', 0)}")
        console.print(f"[bold]Base Path:[/bold] {info['base_path']}")

    asyncio.run(_stats())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
):
    """Search memories."""

    async def _search():
        from memory.llm.embeddings import SentenceTransformerProvider
        from memory.storage.manager import StorageManager

        storage = StorageManager()
        await storage.initialize()

        # Generate query embedding
        embedder = SentenceTransformerProvider()
        query_embedding = await embedder.embed(query)

        # Search
        memories, scores = await storage.vector_search(
            embedding=query_embedding,
            limit=limit,
        )

        await storage.close()

        if not memories:
            console.print("\n[yellow]No matching memories found.[/yellow]")
            return

        console.print(f"\n[bold]Search Results for:[/bold] {query}\n")

        for i, (memory, score) in enumerate(zip(memories, scores), 1):
            console.print(f"[bold cyan]{i}.[/bold cyan] [{memory.truth_category.value}] [dim]Score: {score:.3f}[/dim]")
            console.print(f"   {memory.content[:200]}...")
            console.print(f"   [dim]Domains: {', '.join(memory.domains[:3])}[/dim]")
            console.print()

    asyncio.run(_search())


@app.command()
def export(
    output: Path = typer.Option(Path("memory_export.json"), "--output", "-o", help="Output file"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json, markdown)"),
):
    """Export memories to a file."""

    async def _export():
        import json

        from memory.storage.manager import StorageManager

        storage = StorageManager()
        await storage.initialize()

        # Get all memories
        counts = await storage.count()
        total = counts.get("total", 0)

        if total == 0:
            console.print("[yellow]No memories to export.[/yellow]")
            await storage.close()
            return

        console.print(f"\n[bold]Exporting {total} memories...[/bold]")

        all_memories = []
        for tier in ["persistent", "long_term", "short_term"]:
            from memory.schema.memory_entry import MemoryTier

            tier_enum = MemoryTier(tier)
            memories = await storage.get_store(tier_enum).list_all(limit=10000)
            all_memories.extend(memories)

        await storage.close()

        if format == "json":
            data = [
                {
                    "id": str(m.id),
                    "content": m.content,
                    "summary": m.summary,
                    "tier": m.tier.value,
                    "truth_category": m.truth_category.value,
                    "memory_type": m.memory_type.value,
                    "confidence": m.confidence.overall,
                    "domains": m.domains,
                    "tags": m.tags,
                    "created_at": m.created_at.isoformat(),
                }
                for m in all_memories
            ]
            output.write_text(json.dumps(data, indent=2))
        else:
            lines = ["# Memory Export\n"]
            for m in all_memories:
                lines.append(f"## [{m.truth_category.value}] {m.summary or m.content[:50]}...")
                lines.append(f"- **Type**: {m.memory_type.value}")
                lines.append(f"- **Confidence**: {m.confidence.overall:.2f}")
                lines.append(f"- **Domains**: {', '.join(m.domains)}")
                lines.append(f"\n> {m.content}\n")
            output.write_text("\n".join(lines))

        console.print(f"[green]Exported to {output}[/green]")

    asyncio.run(_export())


@app.command()
def init():
    """Initialize the memory system."""

    async def _init():
        from memory.schema.identity import IdentityProfile
        from memory.storage.manager import StorageManager

        console.print("\n[bold]Initializing Personal Memory System[/bold]\n")

        storage = StorageManager()
        await storage.initialize()

        # Create default identity
        identity = IdentityProfile.create_default()

        # Save identity to persistent store
        identity_path = Path("~/memory/data/persistent/identity.json").expanduser()
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(identity.model_dump_json(indent=2))

        await storage.close()

        console.print("[green]Memory system initialized![/green]")
        console.print("\n[bold]Identity Profile:[/bold]")
        console.print(f"  Name: {identity.name}")
        console.print(f"  GitHub: {', '.join(identity.github_handles)}")
        console.print(f"  Languages: {', '.join(identity.primary_languages)}")
        console.print(f"  Frameworks: {', '.join(identity.frameworks)}")
        console.print("\n[dim]Configuration saved to ~/memory/data/[/dim]")

    asyncio.run(_init())


if __name__ == "__main__":
    app()
