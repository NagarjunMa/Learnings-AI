# CLAUDE.md & AI Context Files: Complete Guide

A reference covering what CLAUDE.md is, why it matters, how it shapes agent behavior, and what equivalent files exist for every major AI coding assistant.

---

## Table of Contents

1. [What Is CLAUDE.md?](#1-what-is-claudemd)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [How Claude Loads It — The Hierarchy](#3-how-claude-loads-it--the-hierarchy)
4. [What Goes Inside CLAUDE.md](#4-what-goes-inside-claudemd)
5. [How It Shapes Agent Behavior](#5-how-it-shapes-agent-behavior)
6. [Equivalent Files for Other AI Assistants](#6-equivalent-files-for-other-ai-assistants)
7. [Universal Best Practices](#7-universal-best-practices)
8. [Is It Essential at Project Start?](#8-is-it-essential-at-project-start)
9. [Real-World Examples](#9-real-world-examples)

---

## 1. What Is CLAUDE.md?

`CLAUDE.md` is a plain Markdown file that Claude Code reads **automatically at the start of every conversation**. It acts as standing instructions — persistent context that you write once and the agent follows every time, without you having to repeat yourself.

Think of it as the **briefing document** you'd give a new engineer joining your team:
- What this project is
- How it's built
- What patterns to follow
- What to avoid

It is **not** a config file, not code, and not a README. It is specifically instructions written *for the AI*.

---

## 2. The Problem It Solves

### Without CLAUDE.md

Every new conversation starts blank. The agent knows nothing about your project:

```
You: "Add a new API endpoint for user profiles"
Agent: [writes raw Express with plain JS, no validation, wrong folder structure]

You: "No — we use NestJS, TypeScript, Zod for validation, and controllers go in src/modules"
Agent: [fixes it]

--- next conversation ---
You: "Add an endpoint for orders"
Agent: [writes raw Express again...]
```

You repeat yourself every single session. The agent makes the same wrong assumptions repeatedly.

### With CLAUDE.md

```
CLAUDE.md contains:
  - Stack: NestJS + TypeScript
  - Validation: always use Zod
  - File structure: controllers in src/modules/<name>/<name>.controller.ts

You: "Add a new API endpoint for user profiles"
Agent: [correctly scaffolds NestJS controller with Zod validation in the right folder]
```

The agent starts every session already knowing your project. You never re-explain.

---

## 3. How Claude Loads It — The Hierarchy

Claude Code looks for `CLAUDE.md` files in three places and merges them all:

```
~/.claude/CLAUDE.md                     ← Global (your personal defaults)
  ↓
/your-project/CLAUDE.md                 ← Project-wide rules
  ↓
/your-project/src/api/CLAUDE.md         ← Subdirectory-specific rules
```

### Global (`~/.claude/CLAUDE.md`)

Applies to **every project** you open with Claude Code. Good for:
- Your personal code style preferences
- Editor/tooling preferences
- Rules that span all your projects ("always write TypeScript, never JavaScript")

### Project Root (`<repo>/CLAUDE.md`)

Applies to the entire repository. This is the most important one. Good for:
- What the application does
- The full tech stack
- Architecture patterns
- Test commands, build commands
- What not to touch

### Subdirectory (`<repo>/src/payments/CLAUDE.md`)

Applies only when Claude is working within that directory subtree. Good for:
- Domain-specific rules ("this module handles PCI-DSS data — never log card numbers")
- Module-specific patterns that differ from the rest of the codebase
- Specialist knowledge about a complex subsystem

### Merge Behavior

All three levels are loaded and merged. If they conflict, the **more specific** file wins:
```
subdirectory > project root > global
```

---

## 4. What Goes Inside CLAUDE.md

### Project Overview

Tell the agent what this thing is, in plain English.

```markdown
## Project Overview
This is the backend API for a B2B SaaS invoicing platform.
It serves a React frontend and mobile apps via REST + GraphQL.
Production handles ~50k requests/day.
```

### Tech Stack

Be exhaustive. Agents make wrong assumptions when they don't know the stack.

```markdown
## Tech Stack
- Runtime: Node.js 20 (LTS)
- Framework: NestJS v10
- Language: TypeScript 5.x (strict mode)
- Database: PostgreSQL 15 via Prisma ORM
- Cache: Redis 7
- Validation: Zod (not class-validator)
- Auth: JWT (access token 15min, refresh token 7d)
- Testing: Jest + Supertest
- Package manager: pnpm (not npm or yarn)
```

### Architecture & Patterns

Tell the agent the structure to follow.

```markdown
## Architecture
- Modular NestJS structure: each domain has its own module in src/modules/<name>/
- Files per module: controller, service, dto, repository, module
- All DB access goes through the repository layer — never query Prisma directly in services
- DTOs use Zod schemas, not class-transformer decorators
- Error handling: throw NestJS HttpException subclasses, never raw Error
```

### Commands

What commands does the agent need to know to run, test, and build your project?

```markdown
## Commands
- Install: pnpm install
- Dev server: pnpm dev
- Tests: pnpm test
- Tests (watch): pnpm test:watch
- Type check: pnpm typecheck
- Lint: pnpm lint
- Build: pnpm build
- DB migrate: pnpm prisma migrate dev
- DB seed: pnpm prisma db seed
```

### Coding Conventions

Name the patterns the agent must follow consistently.

```markdown
## Conventions
- File names: kebab-case (user-profile.service.ts)
- Class names: PascalCase
- Function names: camelCase
- Constants: SCREAMING_SNAKE_CASE
- Always use async/await, never .then()/.catch() chains
- Exports: named exports only, no default exports
- Prefer early returns over nested if/else
- All API responses must go through the ResponseDto wrapper
```

### What NOT to Do

This section is often the most valuable. Negative rules prevent the agent from making mistakes you've already thought about.

```markdown
## Never Do
- Never use `any` type in TypeScript — use `unknown` and narrow it
- Never commit .env files — use .env.example
- Never write raw SQL — use Prisma query builder
- Never catch errors silently (empty catch blocks)
- Never use `console.log` in production code — use the Logger service
- Never modify the migrations/ folder manually
- Never push directly to main — always use a feature branch + PR
- Never install packages without discussing with the team first
```

### Environment & Setup Notes

```markdown
## Environment
- Requires Node 20+ (use nvm: `nvm use`)
- Requires Docker for local PostgreSQL and Redis
- Copy .env.example to .env and fill in values before running
- DB connection string must be in DATABASE_URL env var
```

### External Services

Help the agent understand the external integrations.

```markdown
## External Services
- Stripe: handles all payment processing (src/modules/billing/)
- SendGrid: transactional email (src/modules/notifications/)
- AWS S3: file storage (src/modules/storage/)
- Sentry: error tracking — do not remove Sentry.captureException() calls
```

---

## 5. How It Shapes Agent Behavior

### The agent reads CLAUDE.md before doing anything

Every time you open a conversation in Claude Code, the agent processes all applicable `CLAUDE.md` files first. Your instructions become its operating constraints for that session.

### It prevents wrong assumptions

Without context, agents default to whatever is most common. With CLAUDE.md:
- Agent knows you use `pnpm` — won't suggest `npm install`
- Agent knows you use Zod — won't install `joi` or `yup`
- Agent knows your folder structure — creates files in the right place

### It makes agents refuse bad patterns

If your CLAUDE.md says "Never use `any`", the agent will flag it when it's about to write `any`, explain why it's against your rules, and find a better solution.

### It makes agents consistent across engineers

When multiple engineers use Claude Code on the same project, they all share the same project-level `CLAUDE.md`. The agent behaves the same way for everyone — it follows the team's patterns, not each individual engineer's personal preferences.

### It reduces back-and-forth

The agent makes correct first-draft decisions instead of needing corrections. Each correction you would otherwise give is a round-trip saved — multiply that across dozens of sessions and it's significant.

---

## 6. Equivalent Files for Other AI Assistants

Every major AI coding assistant has its own version of this concept. The name and format differ, but the purpose is identical: **give the AI persistent context about your project**.

---

### Cursor — `.cursor/rules/*.mdc`

**File location:** `.cursor/rules/` directory (one `.mdc` file per rule set)
**Legacy format:** `.cursorrules` (single file at project root — still works but deprecated)

Cursor's rules use **MDC format** (Markdown with frontmatter). You can create multiple rule files, each with a scope.

```
.cursor/
  rules/
    conventions.mdc        ← always applied
    testing.mdc            ← applied when editing test files
    api-patterns.mdc       ← applied when in src/api/
```

**MDC file format:**
```markdown
---
description: TypeScript and NestJS conventions
globs: ["src/**/*.ts"]
alwaysApply: true
---

# Conventions
- Use NestJS modules for all features
- Validate with Zod, not class-validator
- Never use `any` type
```

**Key fields:**
- `alwaysApply: true` — loaded for every file, regardless of globs
- `globs` — only apply this rule when editing files matching the pattern
- `description` — shown in the Cursor UI; helps Cursor decide when to apply it

---

### GitHub Copilot — `.github/copilot-instructions.md`

**File location:** `.github/copilot-instructions.md`
**Format:** Plain Markdown, no frontmatter
**Scope:** Repository-wide

GitHub Copilot reads this file automatically when you're working in the repository (VS Code + Copilot extension, or GitHub.com's Copilot Chat).

```markdown
# Copilot Instructions

## Stack
- Node.js 20, Express 4, TypeScript
- PostgreSQL via Knex.js (not Prisma)

## Rules
- Use ES modules (import/export), not CommonJS
- All routes must have input validation using express-validator
- Tests use Vitest, not Jest
- Never use callback-style async code — use async/await
```

**Note:** As of 2024, GitHub also supports `.github/copilot-instructions/` as a folder with multiple files.

---

### Windsurf (Codeium) — `.windsurfrules`

**File location:** Project root `.windsurfrules`
**Format:** Plain text / Markdown
**Scope:** Project-wide

Windsurf's Cascade agent reads `.windsurfrules` automatically. Also supports a global rules file in settings.

```
# Project Rules

Stack: Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Prisma + PostgreSQL

## File Structure
- Pages: app/(routes)/page.tsx
- Components: components/<name>/<name>.tsx
- Server actions: app/actions/<name>.ts

## Rules
- All components are server components by default
- Use 'use client' only when necessary
- Prefer Tailwind utility classes over custom CSS
- Forms use react-hook-form + Zod
```

---

### Cline — `.clinerules`

**File location:** Project root `.clinerules`
**Format:** Plain Markdown
**Scope:** Project-wide

Cline (VS Code extension, formerly Claude Dev) reads `.clinerules` at the project root. There is also a global rules setting in the extension config.

```markdown
# Project Rules

You are working on a fintech API. Follow these rules strictly:

## Security
- Never log sensitive fields: password, ssn, card_number, cvv
- All financial calculations must use the `decimal.js` library
- Input sanitization required on all user-facing endpoints

## Stack
- Express + TypeScript
- Postgres via pg (no ORM)
- JWT authentication

## Testing
- Every new function needs a unit test
- Use describe/it blocks, not test()
```

---

### Aider — `CONVENTIONS.md` + `.aider.conf.yml`

**Aider** is a terminal-based AI coding assistant. It uses two files:

**`.aider.conf.yml`** — Aider-specific configuration (model, auto-commit behavior):
```yaml
model: claude-3-5-sonnet-20241022
auto-commits: false
dirty-commits: false
read:
  - CONVENTIONS.md     # always load this file into context
```

**`CONVENTIONS.md`** — Your project conventions (loaded via `read:` config or `--read` flag):
```markdown
# Conventions

## Stack
- Python 3.12, FastAPI, SQLAlchemy 2.0, Alembic
- Pydantic v2 for data validation

## Rules
- All endpoints need request/response Pydantic models
- Use dependency injection for DB sessions
- Tests: pytest with pytest-asyncio
- Never use mutable default arguments
```

You can also pass any file into context at startup: `aider --read CONVENTIONS.md`

---

### Continue — `.continuerules`

**File location:** Project root `.continuerules`
**Format:** Plain Markdown
**Scope:** Project-wide

Continue is an open-source AI coding assistant (VS Code / JetBrains). It reads `.continuerules` at project root.

```markdown
# Project Rules
- Framework: Django 5 + Django REST Framework
- Database: PostgreSQL with psycopg3
- Serializers: DRF serializers (not plain dicts)
- Tests: pytest-django
- Linting: ruff + mypy
- Never use Python 2-style string formatting (% or .format()) — use f-strings
```

---

### Zed AI — Assistant Rules

**File:** Via Zed's settings or `.zed/settings.json` using the `assistant` key.

```json
{
  "assistant": {
    "default_model": {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet"
    },
    "inline_alternatives": [],
    "system_prompt": "You are working on a Rust backend API using Axum and Tokio. Follow these rules: use Result<T, AppError> for all fallible functions, never use .unwrap() in production code, prefer tokio::spawn for concurrent tasks."
  }
}
```

For per-project context, Zed uses the assistant panel where you can set rules and include files in the context.

---

### Summary Table

| AI Assistant | Context File | Location | Format | Scope Levels |
|-------------|-------------|----------|--------|-------------|
| Claude Code | `CLAUDE.md` | `~/.claude/`, project root, subdirs | Markdown | Global, Project, Directory |
| Cursor | `.cursor/rules/*.mdc` | Project root | MDC (MD + frontmatter) | Global, Project, File-pattern |
| GitHub Copilot | `copilot-instructions.md` | `.github/` | Markdown | Repository |
| Windsurf | `.windsurfrules` | Project root | Markdown | Global, Project |
| Cline | `.clinerules` | Project root | Markdown | Global, Project |
| Aider | `CONVENTIONS.md` | Anywhere (configured) | Markdown | Configured manually |
| Continue | `.continuerules` | Project root | Markdown | Project |
| Zed AI | `.zed/settings.json` | Project root | JSON | Project |

---

## 7. Universal Best Practices

These apply to **any** AI context file, regardless of tool.

### Be Prescriptive, Not Descriptive

```markdown
# Bad — vague, agent doesn't know what to do
We sometimes use Zod for validation in some parts of the codebase.

# Good — clear rule the agent can follow
Always use Zod for input validation. Never use joi, yup, or class-validator.
```

### Use Negative Rules Explicitly

The agent doesn't know what you want to *avoid* unless you say so.

```markdown
## Never Do
- Never use `any` in TypeScript
- Never write raw SQL strings — use the query builder
- Never catch errors without logging them
- Never commit secrets or API keys
```

### Keep It Concise

Context files are loaded into the agent's context window. Longer = less room for actual code and conversation. Aim for:
- **Under 200 lines** for most projects
- Use bullet points, not paragraphs
- Cut anything the agent can infer from reading the code

### Group by Category

```markdown
## Stack        ← what tools are used
## Architecture ← how the code is structured
## Conventions  ← style and naming rules
## Commands     ← how to run/test/build
## Never Do     ← forbidden patterns
```

### Update It When Architecture Changes

A stale context file is **worse** than no context file. If the agent reads outdated rules:
- It follows rules that no longer apply
- It misses new patterns you've adopted
- It creates inconsistency

Add updating `CLAUDE.md` to your PR checklist when making architectural changes.

### Don't Put Secrets In It

`CLAUDE.md` is often committed to the repository. Never put:
- API keys
- Database passwords
- Internal IP addresses
- PII or customer data

Reference environment variables instead: "The DB URL is in `DATABASE_URL` env var."

---

## 8. Is It Essential at Project Start?

### Short Answer

Not strictly required, but starting without one costs you every session.

### The Compounding Value

```
Day 1:  Write CLAUDE.md — costs 20 minutes
Day 2:  Save 5 minutes of re-explaining context
Day 5:  Save 5 minutes again
Day 10: The agent has been correct first-try 8/10 sessions
Month 1: Saved 2+ hours of back-and-forth
```

The earlier you write it, the more sessions benefit from it.

### Minimum Viable CLAUDE.md

You don't need a perfect file on day one. Even this is worth having:

```markdown
## Stack
- [language], [framework], [database]
- Package manager: [npm/pnpm/yarn/pip/cargo]

## Commands
- Dev: [command]
- Test: [command]
- Build: [command]

## Rules
- [most important convention #1]
- [most important convention #2]
- Never: [most important thing to avoid]
```

Fill in 10 lines, refine over time.

### When It's Especially Critical

| Scenario | Why CLAUDE.md Matters |
|----------|-----------------------|
| Team of 3+ engineers using AI | Ensures everyone gets consistent output |
| Brownfield project (existing codebase) | Agent needs to know existing patterns before adding to them |
| Security-sensitive code | Agent needs explicit rules about what not to log/expose |
| Monorepo | Each package can have its own CLAUDE.md with package-specific rules |
| Non-standard structure | If your folder layout is unusual, the agent will guess wrong without it |

---

## 9. Real-World Examples

### Minimal Starter Template (All Projects)

```markdown
# CLAUDE.md

## Project
[One sentence describing what this app does]

## Stack
- Language: [TypeScript / Python / Go / etc]
- Framework: [Express / Django / Gin / etc]
- Database: [PostgreSQL / MySQL / MongoDB / etc]
- Package manager: [pnpm / pip / cargo / etc]

## Commands
- Install: [command]
- Run dev: [command]
- Run tests: [command]
- Lint/format: [command]

## Conventions
- [Naming convention]
- [Import style]
- [Error handling approach]

## Never Do
- [Most important rule]
- [Second most important rule]
```

---

### Node.js + TypeScript Backend

```markdown
# CLAUDE.md

## Project
REST API for an e-commerce platform. Serves web and mobile clients.

## Stack
- Node.js 20, TypeScript 5 (strict mode)
- NestJS v10
- PostgreSQL 15 via Prisma ORM
- Redis 7 for caching and queues (BullMQ)
- Zod for all validation
- Jest + Supertest for testing
- pnpm (never use npm or yarn)

## Structure
src/
  modules/<name>/
    <name>.controller.ts
    <name>.service.ts
    <name>.repository.ts
    <name>.dto.ts
    <name>.module.ts
  common/           ← shared utilities, guards, interceptors
  config/           ← environment config (use @nestjs/config)

## Commands
- Dev: pnpm dev
- Test: pnpm test
- Test watch: pnpm test:watch
- Lint: pnpm lint
- Build: pnpm build
- Migrate: pnpm prisma migrate dev
- Seed: pnpm prisma db seed

## Rules
- All DB access via repository layer — never use PrismaClient directly in services
- Validate all inputs with Zod DTOs
- Use NestJS Logger, never console.log
- All endpoints require authentication unless decorated with @Public()
- Use async/await — never .then()/.catch() chains
- Named exports only — no default exports

## Never Do
- Never use `any` — use `unknown` and narrow it
- Never write raw SQL
- Never silence errors (empty catch blocks)
- Never expose internal error details to clients
- Never push to main directly — use PRs
```

---

### React Frontend (Next.js)

```markdown
# CLAUDE.md

## Project
Marketing + app dashboard for a SaaS product. Next.js 14 App Router.

## Stack
- Next.js 14 (App Router)
- TypeScript 5, strict mode
- Tailwind CSS v3
- shadcn/ui for components
- React Hook Form + Zod for forms
- TanStack Query (React Query) for server state
- Zustand for client state
- pnpm

## Structure
app/
  (auth)/           ← auth pages (login, signup)
  (dashboard)/      ← authenticated app pages
  api/              ← route handlers
components/
  ui/               ← shadcn/ui components (do not modify directly)
  [feature]/        ← feature-specific components
lib/
  actions/          ← server actions
  api/              ← API client functions

## Rules
- Default to Server Components — add 'use client' only when needed
- Forms: React Hook Form + Zod (not controlled inputs with useState)
- Data fetching: TanStack Query on client, fetch() in server components
- Styling: Tailwind only — no custom CSS files unless absolutely necessary
- Component files: PascalCase. Everything else: kebab-case
- Never install a new UI library without discussion — extend shadcn/ui first

## Commands
- Dev: pnpm dev
- Build: pnpm build
- Test: pnpm test
- Type check: pnpm typecheck
- Lint: pnpm lint
```

---

### Full-Stack Monorepo

```markdown
# CLAUDE.md (root)

## Project
Monorepo for a fintech platform. Contains API, web app, and shared packages.

## Structure
apps/
  api/       ← NestJS REST API (see apps/api/CLAUDE.md)
  web/       ← Next.js frontend (see apps/web/CLAUDE.md)
packages/
  db/        ← Prisma schema + migrations (shared)
  types/     ← shared TypeScript types
  utils/     ← shared utilities

## Stack
- Node.js 20, TypeScript 5 (strict)
- Turborepo for builds
- pnpm workspaces
- See each app's CLAUDE.md for app-specific stack

## Commands (root)
- Install all: pnpm install
- Dev all: pnpm dev
- Build all: pnpm build
- Test all: pnpm test
- Type check all: pnpm typecheck

## Rules
- Shared types live in packages/types — never duplicate them
- DB schema changes: always go through packages/db, never define models in apps
- Never use relative paths across package boundaries — use workspace imports (@repo/types)
- Changesets required for packages/ changes: pnpm changeset

## Never Do
- Never import from apps/ in packages/ — only packages/ → packages/ or apps/ → packages/
- Never commit directly to main
```

---

## Quick Reference

```
File name by tool:
  Claude Code     → CLAUDE.md
  Cursor          → .cursor/rules/*.mdc
  GitHub Copilot  → .github/copilot-instructions.md
  Windsurf        → .windsurfrules
  Cline           → .clinerules
  Aider           → CONVENTIONS.md (+ .aider.conf.yml)
  Continue        → .continuerules

Hierarchy (Claude Code):
  ~/.claude/CLAUDE.md          ← personal defaults
  <repo>/CLAUDE.md             ← project rules
  <repo>/<subdir>/CLAUDE.md    ← module-specific rules

What to always include:
  ✓ Tech stack (specific versions)
  ✓ Dev/test/build commands
  ✓ Folder structure
  ✓ Top 3-5 conventions
  ✓ "Never do" list

What to avoid:
  ✗ Secrets / API keys
  ✗ Prose paragraphs (use bullets)
  ✗ Outdated information
  ✗ Redundant info the agent can read from code
```
