"""Initial database schema migration creating users, ratings, and recommendation_history tables.

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-09-27 22:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # 2. Create ratings table
    op.create_table(
        'ratings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('movie_id', sa.Integer(), nullable=False),
        sa.Column('rating', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('rating >= 0.5 AND rating <= 5.0', name='ck_rating_range'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'movie_id', name='uq_user_movie_rating')
    )
    op.create_index(op.f('ix_ratings_movie_id'), 'ratings', ['movie_id'], unique=False)
    op.create_index(op.f('ix_ratings_user_id'), 'ratings', ['user_id'], unique=False)

    # 3. Create recommendation_history table
    op.create_table(
        'recommendation_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('model', sa.String(length=50), nullable=False),
        sa.Column('seed_movie_id', sa.Integer(), nullable=True),
        sa.Column('top_k', sa.Integer(), nullable=False),
        sa.Column('content_weight', sa.Float(), nullable=True),
        sa.Column('collaborative_weight', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('recommendations', sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_recommendation_history_created_at'), 'recommendation_history', ['created_at'], unique=False)
    op.create_index(op.f('ix_recommendation_history_model'), 'recommendation_history', ['model'], unique=False)
    op.create_index(op.f('ix_recommendation_history_user_id'), 'recommendation_history', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_recommendation_history_user_id'), table_name='recommendation_history')
    op.drop_index(op.f('ix_recommendation_history_model'), table_name='recommendation_history')
    op.drop_index(op.f('ix_recommendation_history_created_at'), table_name='recommendation_history')
    op.drop_table('recommendation_history')

    op.drop_index(op.f('ix_ratings_user_id'), table_name='ratings')
    op.drop_index(op.f('ix_ratings_movie_id'), table_name='ratings')
    op.drop_table('ratings')

    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
