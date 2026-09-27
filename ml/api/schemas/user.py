"""User Pydantic schemas for request validation and response serialization."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_validator


class UserCreate(BaseModel):
    """Schema for creating a new User."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username (3-50 chars).", examples=["sample_user"])
    email: EmailStr = Field(..., description="Unique valid email address.", examples=["sample@example.com"])

    @field_validator("username")
    @classmethod
    def validate_username_not_empty(cls, v: str) -> str:
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("Username must not be empty or consist solely of whitespace.")
        return v_clean


class UserUpdate(BaseModel):
    """Schema for updating an existing User (partial update)."""
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="Optional new username.", examples=["updated_user"])
    email: Optional[EmailStr] = Field(None, description="Optional new email address.", examples=["updated@example.com"])

    @field_validator("username")
    @classmethod
    def validate_username_if_provided(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v_clean = v.strip()
            if not v_clean:
                raise ValueError("Username must not be empty or consist solely of whitespace.")
            return v_clean
        return v


class UserResponse(BaseModel):
    """Schema for User entity response payload."""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="User ID integer.", examples=[1])
    username: str = Field(..., description="User's unique username.", examples=["sample_user"])
    email: str = Field(..., description="User's unique email address.", examples=["sample@example.com"])
    created_at: datetime = Field(..., description="Creation timestamp.")
    updated_at: datetime = Field(..., description="Last update timestamp.")
