"""Minimal ORM for topic_extraction_service: Episode and TranscriptChunk (whisper schema)."""
from datetime import date
from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import Column, Date, ForeignKey, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Episode(Base):
    __tablename__ = "episodes"
    id: Mapped[int] = mapped_column(primary_key=True)
    podcast_id: Mapped[int] = mapped_column(ForeignKey("podcasts.id"), nullable=False)
    name: Mapped[str]
    date_published: Mapped[date] = Column(Date)
    patreon: Mapped[bool]
    url: Mapped[str]
    transcribed: Mapped[bool]
    embedding_generated: Mapped[bool]
    processed_file_path: Mapped[str]
    transcribed_file_path: Mapped[str]


class TranscriptChunk(Base):
    __tablename__ = "transcript_chunks"
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id"), nullable=False, primary_key=True)
    chunk_index: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str]
    timemark: Mapped[str]
    word_count: Mapped[int]
    embedding: Mapped[Vector] = mapped_column(Vector(768))
    created_at: Mapped[date] = Column(Date, default=func.now())
    updated_at: Mapped[date] = Column(Date, default=func.now(), onupdate=func.now())
