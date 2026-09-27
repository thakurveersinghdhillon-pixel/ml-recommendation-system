"""User Service Module implementing business logic and validation for Users."""

import logging
from sqlalchemy.orm import Session

from ml.api.database.repositories.user_repository import UserRepository
from ml.api.schemas.user import UserCreate, UserUpdate, UserResponse
from ml.api.exceptions import UserNotFoundException, DuplicateUserException

logger = logging.getLogger(__name__)


class UserService:
    """Service layer for User entity operations."""

    def __init__(self, db: Session):
        self.db = db
        self.repo = UserRepository(db)

    def create_user(self, payload: UserCreate) -> UserResponse:
        """Create a new user ensuring username and email uniqueness."""
        if self.repo.get_by_username(payload.username):
            raise DuplicateUserException(f"User with username '{payload.username}' already exists.")

        if self.repo.get_by_email(payload.email):
            raise DuplicateUserException(f"User with email '{payload.email}' already exists.")

        user = self.repo.create(username=payload.username, email=payload.email)
        return UserResponse.model_validate(user)

    def get_user_by_id(self, user_id: int) -> UserResponse:
        """Fetch user by ID or raise UserNotFoundException."""
        user = self.repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundException(user_id)
        return UserResponse.model_validate(user)

    def update_user(self, user_id: int, payload: UserUpdate) -> UserResponse:
        """Partially update user attributes."""
        user = self.repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundException(user_id)

        if payload.username is not None and payload.username != user.username:
            existing_user = self.repo.get_by_username(payload.username)
            if existing_user is not None and existing_user.id != user_id:
                raise DuplicateUserException(f"User with username '{payload.username}' already exists.")

        if payload.email is not None and payload.email != user.email:
            existing_email = self.repo.get_by_email(payload.email)
            if existing_email is not None and existing_email.id != user_id:
                raise DuplicateUserException(f"User with email '{payload.email}' already exists.")

        updated_user = self.repo.update(user_id, username=payload.username, email=payload.email)
        return UserResponse.model_validate(updated_user)

    def delete_user(self, user_id: int) -> None:
        """Delete user by ID or raise UserNotFoundException."""
        user = self.repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundException(user_id)
        self.repo.delete(user_id)
