"""
Schema definitions for the voice assistant's memory.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class VoicePreferences(BaseModel):
    """User preferences for voice interaction."""
    preferred_voice: str = "default"
    speaking_rate: float = 1.0
    verbosity_level: str = "medium"  # concise, medium, detailed
    interruption_handling: str = "pause"  # ignore, pause, complete
    confirmation_required: List[str] = Field(default_factory=list)  # actions requiring verbal confirmation


class UserProfile(BaseModel):
    """Structured representation of user information."""
    name: Optional[str] = None
    role: Optional[str] = None
    communication_preferences: Optional[Dict[str, Any]] = None
    interests: List[str] = Field(default_factory=list)
    expertise: List[str] = Field(default_factory=list)
    voice_preferences: VoicePreferences = Field(default_factory=VoicePreferences)


class BusinessEntity(BaseModel):
    """Entity in the business domain (person, project, event, etc.)."""
    name: str
    entity_type: str  # person, project, team, etc.
    attributes: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, str]] = Field(default_factory=list)


class Task(BaseModel):
    """Task tracked by the assistant."""
    description: str
    deadline: Optional[datetime] = None
    priority: str = "medium"
    status: str = "not started"
    related_entities: List[str] = Field(default_factory=list)


class ConversationContext(BaseModel):
    """Tracking context for voice conversations."""
    recent_topics: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    interrupted_flows: List[Dict] = Field(default_factory=list)
    last_interaction: Optional[datetime] = None
