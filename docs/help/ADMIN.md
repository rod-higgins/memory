# Administrator Guide

This guide covers user management and system administration for admin users.

## Admin Overview

As an administrator, you have access to:

- **User Management** - Create, edit, and manage user accounts
- **Access Control** - Assign roles and permissions
- **Security** - Reset passwords and MFA
- **System Health** - Monitor system status

## Accessing Admin Features

1. Log in with an admin account
2. Click **Users** (👥) in the sidebar
3. The Admin panel opens with user management tools

**Note:** The Admin option only appears for users with the admin role.

## User Management

### Viewing Users

The user table displays:

| Column | Description |
|--------|-------------|
| User | Username and email |
| Role | Admin or User |
| Status | Active or Inactive |
| MFA | Two-factor authentication status |
| Last Login | Most recent login time |
| Actions | Edit, Reset MFA, Activate/Deactivate |

### Creating Users

1. Click **Add User**
2. Fill in the required fields:
   - **Username** - Unique identifier (3+ characters)
   - **Email** - Valid email address
   - **Password** - Minimum 8 characters
   - **Role** - User or Admin
3. Click **Create User**

**Note:** New users must set up two-factor authentication on first login.

### Editing Users

1. Click the **Edit** (✏️) button on the user row
2. Modify available fields:
   - Email address
   - Role (cannot change your own role)
3. Click **Save Changes**

### User Roles

| Role | Permissions |
|------|-------------|
| **User** | Access own memories, use AI chat, manage own account |
| **Admin** | All user permissions + manage all users, system settings |

### Changing User Roles

1. Click Edit on the user
2. Select new role from dropdown
3. Save changes

**Important:** You cannot change your own role (prevents accidental lockout).

## Security Management

### Password Resets

To reset a user's password:

1. Click Edit on the user
2. Enter new password in the "Reset Password" field
3. Click **Reset**

The user will:
- Be logged out of all sessions
- Need to log in with the new password

### MFA Resets

If a user loses access to their authenticator:

1. Click the **Reset MFA** (🔑) button on the user row
2. Confirm the reset
3. User will be prompted to set up MFA again on next login

**Security Note:** Verify user identity before resetting MFA.

### Account Deactivation

To temporarily disable an account:

1. Click the **Deactivate** (🚫) button
2. Confirm the action
3. User is immediately logged out

To reactivate:

1. Click the **Activate** (✓) button
2. User can log in again

**Note:** Deactivated users cannot log in but their data is preserved.

## Data Isolation

Each user has isolated data storage:

```
~/memory/data/
├── auth/
│   └── auth.sqlite        # Shared auth database
└── users/
    ├── admin/
    │   ├── persistent/    # Admin's memories
    │   ├── long_term/     # Admin's vectors
    │   └── short_term/    # Admin's cache
    └── username/
        ├── persistent/    # User's memories
        ├── long_term/     # User's vectors
        └── short_term/    # User's cache
```

### What's Isolated

- All memories and memory metadata
- AI conversation history
- LLM provider configurations
- User preferences

### What's Shared

- Authentication database (auth.sqlite)
- System configuration
- Ollama models (if using local inference)

## Initial Setup

### First Admin Account

The first admin account is created during installation:

```bash
python scripts/setup_auth.py
```

This script:
1. Prompts for admin credentials
2. Creates the auth database
3. Optionally migrates existing data

### Adding Additional Admins

1. Create a new user via the Admin panel
2. Set role to "Admin"
3. User completes MFA setup on first login

## Self-Service Features

Users can manage their own accounts:

- **Change Password** - Via user menu dropdown
- **View Account Info** - Username, email, role
- **Logout** - End current session

Admins can also:
- Change their own password
- Cannot change their own role
- Cannot deactivate themselves

## Audit Trail

Current session information is tracked:

- Login times
- Session creation/expiration
- Failed login attempts (in logs)

## Security Best Practices

### Password Policies

Enforce strong passwords:
- Minimum 8 characters
- Mix of letters and numbers recommended
- Avoid common passwords

### MFA Enforcement

All users are required to set up MFA:
- Using Google Authenticator or compatible app
- Backup codes provided at setup
- Cannot be bypassed

### Session Management

- Sessions expire after 24 hours
- Users are logged out on password change
- Admins can force logout by resetting passwords

### Principle of Least Privilege

- Only grant admin role when necessary
- Regular users have access only to their own data
- Review user list periodically

## Troubleshooting

### User Can't Log In

1. Verify account is active (not deactivated)
2. Check password is correct
3. Verify MFA code is current (changes every 30 seconds)
4. Reset password if needed

### User Lost MFA Device

1. Verify user identity
2. Reset MFA via Admin panel
3. User sets up MFA again on next login

### Admin Locked Out

If all admins are locked out:

```bash
# Create a new admin via command line
python scripts/setup_auth.py --force-new-admin
```

### Data Migration

To migrate a user's data to a new account:

1. Copy contents of old user's data directory
2. Move to new user's data directory
3. Update any path references in SQLite

## Common Tasks

### Onboarding New User

1. Create user account
2. Send credentials securely
3. User logs in and sets up MFA
4. User begins adding memories

### Offboarding User

1. Deactivate the account
2. (Optional) Backup user data
3. (Optional) Delete user data directory

### Promoting User to Admin

1. Edit user
2. Change role to Admin
3. Communicate new responsibilities

## Related Guides

- [Getting Started](GETTING_STARTED.md) - Initial setup
- [User Guide](USER_GUIDE.md) - Complete feature reference
