# Database Integration — SQLAlchemy Async

## Why Async SQLAlchemy

FastAPI is async. Blocking DB calls inside async routes block the entire event loop — kills concurrency. `asyncpg` + `SQLAlchemy async` gives non-blocking DB I/O.

```bash
pip install sqlalchemy asyncpg alembic
# For SQLite (testing): pip install aiosqlite
```

---

## Engine and Session Setup

```python
# db/session.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/mydb"
# SQLite for dev: "sqlite+aiosqlite:///./dev.db"

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,          # max persistent connections
    max_overflow=0,        # no extra connections beyond pool_size
    pool_pre_ping=True,    # test connection before use (handles stale connections)
    pool_recycle=3600,     # recycle connections after 1 hour
    echo=False,            # set True to log all SQL (dev only)
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,   # don't expire objects after commit (avoids lazy load errors)
    autoflush=False,
    autocommit=False,
)

class Base(DeclarativeBase):
    pass
```

---

## ORM Models

```python
# models/user.py
from sqlalchemy import String, Integer, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from db.session import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    items: Mapped[list["Item"]] = relationship("Item", back_populates="owner", lazy="selectin")
```

**`Mapped[T]`** — v2 syntax. Replaces `Column(Type)`. Type information is in the annotation.

---

## Session as Dependency (yield pattern)

```python
# dependencies.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from db.session import AsyncSessionLocal

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()   # auto-commit on success
        except Exception:
            await session.rollback() # auto-rollback on exception
            raise
```

```python
# routers/users.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from dependencies import get_db

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

## CRUD Operations

```python
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

# CREATE
async def create_user(db: AsyncSession, username: str, email: str) -> User:
    user = User(username=username, email=email, hashed_password="hashed")
    db.add(user)
    await db.flush()    # assigns id without committing
    await db.refresh(user)  # reload from DB (picks up server defaults)
    return user

# READ one
async def get_user(db: AsyncSession, user_id: int) -> User | None:
    return await db.get(User, user_id)  # uses primary key

# READ with WHERE
async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()

# READ many
async def list_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> list[User]:
    result = await db.execute(select(User).offset(skip).limit(limit))
    return list(result.scalars().all())

# UPDATE
async def update_user(db: AsyncSession, user_id: int, new_email: str) -> User | None:
    user = await db.get(User, user_id)
    if not user:
        return None
    user.email = new_email
    await db.flush()
    return user

# DELETE
async def delete_user(db: AsyncSession, user_id: int) -> bool:
    user = await db.get(User, user_id)
    if not user:
        return False
    await db.delete(user)
    return True
```

---

## Transactions

```python
# Explicit transaction (useful when you need to span multiple operations)
async def transfer_funds(db: AsyncSession, from_id: int, to_id: int, amount: float):
    async with db.begin():  # explicit transaction block
        from_account = await db.get(Account, from_id)
        to_account = await db.get(Account, to_id)

        if from_account.balance < amount:
            raise ValueError("Insufficient funds")

        from_account.balance -= amount
        to_account.balance += amount
        # auto-commits on exit, rolls back on exception

# Savepoints (nested transactions)
async def risky_operation(db: AsyncSession):
    async with db.begin_nested() as savepoint:
        try:
            # risky work here
            ...
        except Exception:
            await savepoint.rollback()
            # outer transaction continues
```

---

## N+1 Prevention — Eager Loading

N+1 problem: loading 100 users then querying items for each = 101 queries.

```python
from sqlalchemy.orm import selectinload, joinedload

# selectinload — separate IN query (best for one-to-many)
async def get_users_with_items(db: AsyncSession) -> list[User]:
    result = await db.execute(
        select(User).options(selectinload(User.items))
    )
    return list(result.scalars().all())
# Executes: SELECT users + SELECT items WHERE user_id IN (1,2,3...)

# joinedload — single JOIN query (best for many-to-one / small related sets)
async def get_items_with_owner(db: AsyncSession) -> list[Item]:
    result = await db.execute(
        select(Item).options(joinedload(Item.owner))
    )
    return list(result.unique().scalars().all())  # .unique() needed with joinedload

# Nested eager loading
result = await db.execute(
    select(User)
    .options(
        selectinload(User.items).selectinload(Item.tags)
    )
)
```

**Rule**: `selectinload` for collections (one-to-many), `joinedload` for single (many-to-one).

---

## Lifespan Database Integration

```python
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from db.session import engine, Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup (use Alembic in production)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Cleanup on shutdown
    await engine.dispose()

app = FastAPI(lifespan=lifespan)
```

---

## Alembic Migrations

```bash
# Init
alembic init alembic

# Create migration (auto-detect ORM changes)
alembic revision --autogenerate -m "add users table"

# Run migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

```python
# alembic/env.py — point to your models
from db.session import Base
from models import user, item  # import models so they register with Base

target_metadata = Base.metadata
```

---

## Complex Queries

```python
from sqlalchemy import select, and_, or_, func, desc

# WHERE with AND/OR
result = await db.execute(
    select(User)
    .where(and_(User.is_active == True, User.age >= 18))
    .order_by(desc(User.created_at))
    .limit(10)
)

# COUNT
result = await db.execute(select(func.count(User.id)).where(User.is_active == True))
count = result.scalar()

# JOIN
result = await db.execute(
    select(User, Item)
    .join(Item, Item.owner_id == User.id)
    .where(Item.price > 100)
)
rows = result.all()  # list of (User, Item) tuples

# EXISTS check
from sqlalchemy import exists
stmt = select(exists().where(User.email == "alice@example.com"))
result = await db.execute(stmt)
email_exists = result.scalar()
```

---

## Interview Cheat Sheet

| Pattern | Code |
|---------|------|
| Async engine | `create_async_engine("postgresql+asyncpg://...")` |
| Session factory | `async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)` |
| Session dependency | `async def get_db() -> AsyncGenerator[AsyncSession, None]: async with AsyncSessionLocal() as s: yield s` |
| Get by PK | `await db.get(Model, id)` |
| SELECT query | `result = await db.execute(select(Model).where(...))` |
| Extract rows | `result.scalars().all()` |
| Create | `db.add(obj); await db.flush()` |
| ORM mode for Pydantic | `ConfigDict(from_attributes=True)` |
| Eager load collection | `options(selectinload(Model.relation))` |
| Eager load single | `options(joinedload(Model.parent))` |
| Transaction | `async with db.begin(): ...` |
