"""User Repository Module for database operations on User table."""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from ml.api.database.models import User


class UserRepository:
    """Repository handling CRUD operations for User model."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, username: str, email: str) -> User:
        """Create a new user record."""
        user = User(username=username, email=email)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by primary key ID."""
        return self.db.get(User, user_id)

    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by unique username."""
        stmt = select(User).where(User.username == username)
        return self.db.scalars(stmt).first()

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by unique email address."""
        stmt = select(User).where(User.email == email)
        return self.db.scalars(stmt).first()

    def update(self, user_id: int, username: Optional[str] = None, email: Optional[str] = None) -> Optional[User]:
        """Update existing user record fields."""
        user = self.get_by_id(user_id)
        if user is None:
            return None

        if username is not None:
            user.username = username
        if email is not None:
            user.email = email

        user.updated_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(user)
        return user

    def delete(self, user_id: int) -> bool:
        """Delete user by ID."""
        user = self.get_by_id(user_id)
        if user is None:
            return False

        self.db.delete(user)
        self.db.commit()
        return True
