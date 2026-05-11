# Pydantic v2 Advanced

Pydantic v2 is a full rewrite (Rust core). Syntax changes from v1 are breaking — v2 validators use different decorators. FastAPI 0.100+ uses Pydantic v2 by default.

---

## Field() — Constraints and Metadata

```python
from pydantic import BaseModel, Field
from typing import Annotated

class UserCreate(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Alphanumeric username",
        examples=["alice_42"],
    )
    age: int = Field(ge=0, le=150)          # ge=>=, le=<=, gt=>, lt=<
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    email: str = Field(alias="emailAddress")  # JSON key differs from Python attr
    tags: list[str] = Field(default_factory=list)  # mutable default — use factory
```

`Annotated` alternative (preferred for reuse):
```python
PositiveInt = Annotated[int, Field(gt=0)]
ShortStr = Annotated[str, Field(max_length=100)]

class Item(BaseModel):
    quantity: PositiveInt
    name: ShortStr
```

---

## @field_validator (v2 syntax)

Validates or transforms a single field. Replaces v1 `@validator`.

```python
from pydantic import BaseModel, field_validator

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

    @field_validator("username")
    @classmethod
    def username_lower(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("email")
    @classmethod
    def email_valid(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email")
        return v.lower()

    @field_validator("password")
    @classmethod
    def password_strong(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v
```

**`mode` parameter** — controls when validation runs:
```python
@field_validator("username", mode="before")  # runs before type coercion
@classmethod
def strip_whitespace(cls, v):
    return v.strip() if isinstance(v, str) else v

@field_validator("username", mode="after")   # default — runs after type coercion
@classmethod
def check_length(cls, v: str) -> str:
    if len(v) < 2:
        raise ValueError("Too short")
    return v
```

**Multiple fields in one validator:**
```python
@field_validator("first_name", "last_name")
@classmethod
def names_not_empty(cls, v: str) -> str:
    if not v.strip():
        raise ValueError("Name cannot be empty")
    return v.strip()
```

---

## @model_validator — Cross-Field Validation

Validates across multiple fields simultaneously. Replaces v1 `@root_validator`.

```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start_date: str
    end_date: str

    @model_validator(mode="after")
    def check_dates(self) -> "DateRange":
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self

class PasswordConfirm(BaseModel):
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def passwords_match(self) -> "PasswordConfirm":
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self
```

**`mode="before"`** — receives raw dict before field parsing:
```python
class FlexibleInput(BaseModel):
    value: int

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: dict) -> dict:
        # convert string "42" → int before field validation
        if isinstance(data.get("value"), str):
            data["value"] = int(data["value"])
        return data
```

---

## ConfigDict — Model Configuration

Replaces v1 `class Config:` with `model_config = ConfigDict(...)`.

```python
from pydantic import BaseModel, ConfigDict

class UserResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,       # read from ORM objects (replaces orm_mode=True)
        populate_by_name=True,      # accept both alias and field name
        str_strip_whitespace=True,  # auto-strip whitespace from str fields
        str_to_lower=False,         # don't auto-lowercase
        validate_default=True,      # run validators on default values too
        extra="forbid",             # reject unknown fields (default: "ignore")
        # extra="allow"             # accept and store unknown fields
        # extra="ignore"            # silently drop unknown fields
        frozen=True,                # make model immutable (hashable)
        use_enum_values=True,       # store enum values, not enum instances
    )

    id: int
    name: str
```

**`from_attributes=True`** is critical for SQLAlchemy integration:
```python
# ORM model
class UserORM:
    id = 1
    name = "Alice"

# Pydantic schema
class UserSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str

# Works without .model_validate(user.__dict__)
user_schema = UserSchema.model_validate(orm_user)
```

---

## @computed_field

Derived fields that appear in serialization. Not stored — computed on access.

```python
from pydantic import BaseModel, computed_field

class Rectangle(BaseModel):
    width: float
    height: float

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height

    @computed_field
    @property
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

r = Rectangle(width=3, height=4)
print(r.model_dump())
# {"width": 3, "height": 4, "area": 12.0, "perimeter": 14.0}
```

```python
class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    first_name: str
    last_name: str

    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
```

---

## response_model Filtering

Control what gets serialized in response:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserDB(BaseModel):
    id: int
    name: str
    password_hash: str   # never expose this
    internal_notes: str  # never expose this

class UserResponse(BaseModel):
    id: int
    name: str

@app.get("/users/{id}", response_model=UserResponse)
async def get_user(id: int):
    # FastAPI strips password_hash and internal_notes
    return UserDB(id=id, name="Alice", password_hash="hash", internal_notes="vip")
```

**`exclude_unset`** — only return fields explicitly set (not defaults):
```python
@app.patch("/users/{id}", response_model=UserResponse, response_model_exclude_unset=True)
async def patch_user(id: int, update: UserUpdate):
    # Only fields the client sent are returned — perfect for PATCH
    ...
```

**`exclude_none`** — strip null fields from response:
```python
@app.get("/users/{id}", response_model=UserResponse, response_model_exclude_none=True)
async def get_user(id: int):
    ...
```

---

## json_schema_extra — OpenAPI Examples

```python
class UserCreate(BaseModel):
    username: str
    email: str
    age: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "alice_42",
                "email": "alice@example.com",
                "age": 30,
            }
        }
    )
```

Multiple examples:
```python
from fastapi import Body

@app.post("/users/")
async def create_user(
    user: UserCreate = Body(
        openapi_examples={
            "normal": {
                "summary": "Normal user",
                "value": {"username": "alice", "email": "alice@example.com", "age": 30},
            },
            "admin": {
                "summary": "Admin user",
                "value": {"username": "admin", "email": "admin@example.com", "age": 25},
            },
        }
    )
):
    ...
```

---

## Discriminated Unions

When one field determines which schema to use — avoids ambiguous validation.

```python
from typing import Literal, Union
from pydantic import BaseModel

class Cat(BaseModel):
    pet_type: Literal["cat"]
    meows: int

class Dog(BaseModel):
    pet_type: Literal["dog"]
    barks: float

class Owner(BaseModel):
    pet: Union[Cat, Dog] = Field(discriminator="pet_type")

# Pydantic uses pet_type to pick Cat or Dog — no ambiguity, faster validation
owner = Owner(pet={"pet_type": "cat", "meows": 3})
# owner.pet is Cat instance
```

Without discriminator, Pydantic tries each type in order (slow, error-prone on similar schemas).

---

## Custom Types with Annotated

```python
from typing import Annotated
from pydantic import AfterValidator, BeforeValidator, PlainSerializer

def validate_positive(v: int) -> int:
    if v <= 0:
        raise ValueError("Must be positive")
    return v

PositiveInt = Annotated[int, AfterValidator(validate_positive)]

def parse_comma_list(v: str | list) -> list[str]:
    if isinstance(v, str):
        return [x.strip() for x in v.split(",")]
    return v

CommaSeparated = Annotated[list[str], BeforeValidator(parse_comma_list)]

class Filter(BaseModel):
    min_age: PositiveInt
    tags: CommaSeparated   # accepts "python,fastapi" OR ["python", "fastapi"]
```

---

## model_dump and model_validate

```python
user = UserCreate(username="alice", email="alice@example.com", age=30)

# Serialize to dict
user.model_dump()
user.model_dump(exclude={"password"})
user.model_dump(include={"id", "name"})
user.model_dump(exclude_unset=True)   # only fields explicitly set
user.model_dump(exclude_none=True)    # skip None fields
user.model_dump(by_alias=True)        # use Field(alias=...) as key

# Serialize to JSON string
user.model_dump_json()

# Deserialize from dict
UserCreate.model_validate({"username": "alice", "email": "alice@example.com", "age": 30})

# From ORM object
UserResponse.model_validate(orm_object)  # requires from_attributes=True
```

---

## Interview Cheat Sheet

| Pattern | Syntax |
|---------|--------|
| Field constraints | `Field(ge=0, le=100, min_length=3)` |
| Field alias | `Field(alias="emailAddress")` |
| Single field validator | `@field_validator("field") @classmethod def fn(cls, v): ...` |
| Cross-field validator | `@model_validator(mode="after") def fn(self): ...` |
| ORM mode | `model_config = ConfigDict(from_attributes=True)` |
| Reject unknown fields | `ConfigDict(extra="forbid")` |
| Computed field | `@computed_field @property def fn(self) -> T:` |
| Strip null in response | `response_model_exclude_none=True` |
| Only changed fields | `response_model_exclude_unset=True` |
| Discriminated union | `Field(discriminator="type_field")` |
