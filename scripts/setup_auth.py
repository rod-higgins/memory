#!/usr/bin/env python3
"""
Initial authentication setup script for Personal Memory System.

Run this during installation to create the admin account:
    python scripts/setup_auth.py

This will:
1. Create the auth database
2. Prompt for admin credentials
3. Create admin user (MFA setup required on first login)
4. Optionally migrate existing single-user data
"""

import asyncio
import getpass
import shutil
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory.auth import AuthStore, UserCreate, UserRole


async def setup_admin():
    """Run the admin setup wizard."""
    print("\n" + "=" * 60)
    print("  Personal Memory System - Initial Setup")
    print("=" * 60 + "\n")

    # Initialize paths
    data_root = Path("~/memory/data").expanduser()
    auth_path = data_root / "auth"
    auth_path.mkdir(parents=True, exist_ok=True)

    # Initialize auth store
    store = AuthStore(db_path=auth_path / "auth.sqlite")
    await store.initialize()

    # Check if admin already exists
    if await store.has_admin():
        print("An admin user already exists.")
        print("Use the web interface to manage users.")
        await store.close()
        return

    print("Create your admin account:\n")

    # Get username
    while True:
        username = input("Username [admin]: ").strip() or "admin"
        if len(username) < 3:
            print("Username must be at least 3 characters.")
            continue
        if not username.replace("_", "").replace("-", "").isalnum():
            print("Username can only contain letters, numbers, underscores, and hyphens.")
            continue
        break

    # Get email
    while True:
        email = input("Email: ").strip()
        if not email or "@" not in email:
            print("Please enter a valid email address.")
            continue
        break

    # Get password
    while True:
        password = getpass.getpass("Password (min 8 chars): ")
        if len(password) < 8:
            print("Password must be at least 8 characters.")
            continue
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords don't match. Try again.")
            continue
        break

    # Create admin user
    print("\nCreating admin user...")
    try:
        admin = await store.create_user(
            UserCreate(
                username=username,
                email=email,
                password=password,
                role=UserRole.ADMIN,
            )
        )
        print(f"\n  Admin user '{username}' created successfully!")
        print(f"  Data directory: {admin.data_path}")
    except Exception as e:
        print(f"\nError creating admin user: {e}")
        await store.close()
        return

    print("\n" + "-" * 60)
    print("IMPORTANT: Two-Factor Authentication")
    print("-" * 60)
    print("On first login, you'll be prompted to set up two-factor")
    print("authentication using Google Authenticator or a compatible app.")
    print("-" * 60)

    # Check for existing data to migrate
    legacy_persistent = data_root / "persistent"
    if (legacy_persistent / "core.sqlite").exists():
        print("\n" + "-" * 60)
        print("Existing Data Detected")
        print("-" * 60)
        print(f"Found existing memories at: {legacy_persistent}")

        while True:
            migrate = input("\nMigrate existing data to admin account? [Y/n]: ").strip().lower()
            if migrate in ("", "y", "yes"):
                await migrate_legacy_data(admin.data_path, legacy_persistent)
                break
            elif migrate in ("n", "no"):
                print("Skipping migration. Existing data preserved.")
                break
            else:
                print("Please enter 'y' or 'n'.")

    await store.close()

    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print("\nYou can now start the web server:")
    print("  cd ~/memory && python -m memory.web.app")
    print("\nThen open http://localhost:8765 in your browser.")
    print()


async def migrate_legacy_data(target_path: str, source_path: Path):
    """Migrate legacy single-user data to new user directory."""
    print("\nMigrating data...")

    target = Path(target_path)
    backup_path = source_path.parent / "_legacy"

    # Create backup directory
    backup_path.mkdir(exist_ok=True)

    # Create user directories
    (target / "persistent").mkdir(parents=True, exist_ok=True)
    (target / "long_term").mkdir(exist_ok=True)
    (target / "short_term").mkdir(exist_ok=True)

    migrated_files = []

    # Copy persistent data files
    for item in ["core.sqlite", "identity.json", "data_sources.json"]:
        src = source_path / item
        if src.exists():
            # Copy to new location
            shutil.copy2(src, target / "persistent" / item)
            # Backup original
            shutil.copy2(src, backup_path / item)
            migrated_files.append(item)
            print(f"  Migrated: {item}")

    # Copy connections database
    conn_db = source_path.parent / "connections.db"
    if conn_db.exists():
        shutil.copy2(conn_db, target / "connections.db")
        shutil.copy2(conn_db, backup_path / "connections.db")
        migrated_files.append("connections.db")
        print("  Migrated: connections.db")

    # Copy long-term data (LanceDB)
    lt_path = source_path.parent / "long_term"
    if lt_path.exists() and lt_path.is_dir():
        if any(lt_path.iterdir()):
            shutil.copytree(lt_path, target / "long_term", dirs_exist_ok=True)
            shutil.copytree(lt_path, backup_path / "long_term", dirs_exist_ok=True)
            print("  Migrated: long_term/ (LanceDB vectors)")

    # Copy short-term data
    st_path = source_path.parent / "short_term"
    if st_path.exists() and st_path.is_dir():
        if any(st_path.iterdir()):
            shutil.copytree(st_path, target / "short_term", dirs_exist_ok=True)
            print("  Migrated: short_term/")

    print(f"\n  Backup saved to: {backup_path}")
    print(f"  {len(migrated_files)} files migrated successfully.")


def main():
    """Entry point."""
    try:
        asyncio.run(setup_admin())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
