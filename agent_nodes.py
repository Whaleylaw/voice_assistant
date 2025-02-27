"""
LangGraph node implementations for the voice assistant.
"""

from typing import Dict, List, Any, TypedDict, Optional
import logging
from datetime import datetime

from langgraph.config import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI

import config
from voice_io import record_audio, speak, optimize_for_speech
from memory_store import MemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define our state type
class AssistantState(TypedDict):
    """State for the voice assistant."""
    messages: List[Dict]  # Conversation messages
    user_id: str  # User identifier
    memory_search_results: Optional[Dict] = None  # Retrieved memories
    spoken_response: Optional[str] = None  # The response that was spoken


def voice_input(state: AssistantState) -> AssistantState:
    """Record audio from the user and convert to text."""
    logger.info("Recording user input...")
    
    # Record audio and get transcription
    transcription = record_audio()
    
    if not transcription:
        # If no transcription, ask the user to try again
        speak("I didn't catch that. Could you please try again?")
        return state
    
    # Add user message to state
    messages = state.get("messages", [])
    messages.append({"role": "user", "content": transcription})
    
    logger.info(f"User input: {transcription}")
    return {**state, "messages": messages}


def search_memory(state: AssistantState, store: BaseStore) -> AssistantState:
    """Search for relevant memories based on the last user message."""
    logger.info("Searching memory...")
    
    user_id = state.get("user_id", config.DEFAULT_USER_ID)
    messages = state.get("messages", [])
    
    # Skip if no messages
    if not messages:
        return {**state, "memory_search_results": {}}
    
    # Get the last user message
    last_message = None
    for message in reversed(messages):
        if message["role"] == "user":
            last_message = message["content"]
            break
    
    if not last_message:
        return {**state, "memory_search_results": {}}
    
    # Search memory
    memory_manager = MemoryManager(store)
    search_results = memory_manager.search_memory(last_message, user_id)
    
    logger.info(f"Memory search results: {search_results}")
    return {**state, "memory_search_results": search_results}


def generate_response(state: AssistantState, store: BaseStore) -> AssistantState:
    """Generate a response based on the conversation and memory."""
    logger.info("Generating response...")
    
    user_id = state.get("user_id", config.DEFAULT_USER_ID)
    messages = state.get("messages", [])
    memory_results = state.get("memory_search_results", {})
    
    # Skip if no messages
    if not messages:
        return state
    
    # Format memory results for the prompt
    memory_context = _format_memory_for_prompt(memory_results)
    
    # Create LLM
    llm = ChatOpenAI(
        model=config.CHAT_MODEL,
        temperature=0.7
    )
    
    # Prepare system message with memory context
    system_message = {
        "role": "system", 
        "content": f"""You are a helpful voice assistant with access to the user's personal and business information.
        
You should use the memory context provided to personalize your responses.

When responding:
1. Be concise and conversational - you are speaking, not writing
2. Use natural language and avoid complex structures
3. Reference relevant information from the user's profile or business knowledge
4. Keep your responses short enough to be easily spoken

Memory context:
{memory_context}

User ID: {user_id}"""
    }
    
    # Get response from LLM
    response = llm.invoke([system_message, *messages[-config.MAX_CONVERSATION_HISTORY:]])
    
    # Extract response content
    response_content = response.content
    
    # Add to messages
    messages.append({"role": "assistant", "content": response_content})
    
    # Optimize for speech
    spoken_response = optimize_for_speech(response_content)
    
    logger.info(f"Generated response: {response_content[:100]}...")
    
    # Update memory
    _update_memory(messages, user_id, store)
    
    return {
        **state, 
        "messages": messages, 
        "spoken_response": spoken_response
    }


def voice_output(state: AssistantState) -> AssistantState:
    """Convert the assistant's response to speech and play it."""
    logger.info("Converting response to speech...")
    
    spoken_response = state.get("spoken_response")
    
    if spoken_response:
        # Speak the response
        speak(spoken_response)
    
    return state


def _format_memory_for_prompt(memory_results: Dict) -> str:
    """Format memory search results for inclusion in the prompt."""
    formatted = []
    
    # Format user profile
    if memory_results.get("user_profile"):
        profile = memory_results["user_profile"][0]
        formatted.append("USER PROFILE:")
        if profile.name:
            formatted.append(f"- Name: {profile.name}")
        if profile.role:
            formatted.append(f"- Role: {profile.role}")
        if profile.interests:
            formatted.append(f"- Interests: {', '.join(profile.interests)}")
        if profile.expertise:
            formatted.append(f"- Expertise: {', '.join(profile.expertise)}")
    
    # Format business knowledge
    if memory_results.get("business_knowledge"):
        formatted.append("\nBUSINESS KNOWLEDGE:")
        for entity in memory_results["business_knowledge"]:
            formatted.append(f"- {entity.entity_type.capitalize()}: {entity.name}")
            for k, v in entity.attributes.items():
                formatted.append(f"  * {k}: {v}")
    
    # Format conversation history
    if memory_results.get("conversation_history"):
        formatted.append("\nRELEVANT PAST CONVERSATIONS:")
        for conversation in memory_results["conversation_history"]:
            timestamp = datetime.fromisoformat(conversation["timestamp"])
            formatted.append(f"- Date: {timestamp.strftime('%Y-%m-%d %H:%M')}")
            if "messages" in conversation:
                for msg in conversation["messages"][-2:]:  # Just the last exchange
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    if len(content) > 100:
                        content = content[:100] + "..."
                    formatted.append(f"  * {role}: {content}")
    
    return "\n".join(formatted)


def _update_memory(messages: List[Dict], user_id: str, store: BaseStore):
    """Update memory with new information from the conversation."""
    logger.info("Updating memory...")
    
    # Initialize memory manager
    memory_manager = MemoryManager(store)
    
    # Update user profile
    memory_manager.update_user_profile(messages, user_id)
    
    # Update business knowledge
    memory_manager.update_business_knowledge(messages, user_id)
    
    # Save conversation
    memory_manager.save_conversation(messages, user_id)
