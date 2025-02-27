"""
Memory storage and retrieval for the voice assistant.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

import config
from schemas import UserProfile, BusinessEntity, Task, ConversationContext


class MemoryManager:
    """Manages memory storage and retrieval for the assistant."""
    
    def __init__(self, store=None):
        """Initialize the memory manager.
        
        Args:
            store: Optional memory store to use. If not provided, a new InMemoryStore will be created.
        """
        # Initialize store if not provided
        if store is None:
            self.store = InMemoryStore(
                index={
                    "dims": config.EMBEDDING_DIMENSIONS,
                    "embed": config.EMBEDDING_MODEL,
                }
            )
        else:
            self.store = store
            
        # Initialize memory managers for different types of memories
        self.profile_manager = create_memory_store_manager(
            config.CHAT_MODEL,
            namespace=(config.MEMORY_NAMESPACE_USER_PROFILE, "{user_id}"),
            schemas=[UserProfile],
            enable_inserts=False  # Update existing profile only
        )
        
        self.business_manager = create_memory_store_manager(
            config.CHAT_MODEL,
            namespace=(config.MEMORY_NAMESPACE_BUSINESS, "{user_id}"),
            schemas=[BusinessEntity],
            enable_inserts=True
        )
    
    def get_user_profile(self, user_id: str = config.DEFAULT_USER_ID) -> Optional[UserProfile]:
        """Get the user profile from memory.
        
        Args:
            user_id: The user ID to get the profile for.
            
        Returns:
            The user profile, or None if not found.
        """
        namespace = (config.MEMORY_NAMESPACE_USER_PROFILE, user_id)
        results = self.store.search(namespace)
        
        if results:
            # Convert the stored dictionary to a UserProfile object
            return UserProfile.model_validate(results[0].value)
        return None
    
    def update_user_profile(self, messages: List[Dict], user_id: str = config.DEFAULT_USER_ID):
        """Update the user profile based on conversation.
        
        Args:
            messages: The conversation messages.
            user_id: The user ID to update the profile for.
        """
        self.profile_manager.invoke(
            {"messages": messages},
            config={"configurable": {"user_id": user_id}}
        )
    
    def get_business_entities(self, query: str = None, user_id: str = config.DEFAULT_USER_ID) -> List[BusinessEntity]:
        """Get business entities from memory.
        
        Args:
            query: Optional search query to filter entities.
            user_id: The user ID to get entities for.
            
        Returns:
            A list of business entities.
        """
        namespace = (config.MEMORY_NAMESPACE_BUSINESS, user_id)
        
        if query:
            results = self.store.search(namespace, query=query)
        else:
            results = self.store.search(namespace)
        
        # Convert the stored dictionaries to BusinessEntity objects
        return [BusinessEntity.model_validate(item.value) for item in results]
    
    def update_business_knowledge(self, messages: List[Dict], user_id: str = config.DEFAULT_USER_ID):
        """Update business knowledge based on conversation.
        
        Args:
            messages: The conversation messages.
            user_id: The user ID to update knowledge for.
        """
        self.business_manager.invoke(
            {"messages": messages},
            config={"configurable": {"user_id": user_id}}
        )
    
    def save_conversation(self, messages: List[Dict], user_id: str = config.DEFAULT_USER_ID):
        """Save a conversation to memory.
        
        Args:
            messages: The conversation messages.
            user_id: The user ID to save the conversation for.
        """
        namespace = (config.MEMORY_NAMESPACE_CONVERSATION, user_id)
        
        # Generate a unique ID for the conversation
        conversation_id = str(uuid.uuid4())
        
        # Save the conversation
        self.store.put(
            namespace,
            conversation_id,
            {
                "timestamp": datetime.now().isoformat(),
                "messages": messages
            }
        )
    
    def get_conversation_context(self, user_id: str = config.DEFAULT_USER_ID) -> ConversationContext:
        """Get the conversation context from memory.
        
        Args:
            user_id: The user ID to get the context for.
            
        Returns:
            The conversation context.
        """
        namespace = (config.MEMORY_NAMESPACE_CONVERSATION, user_id)
        
        # Get the conversation context
        results = self.store.search(namespace, key="context")
        
        if results:
            # Convert the stored dictionary to a ConversationContext object
            return ConversationContext.model_validate(results[0].value)
        
        # If no context exists, create a new one
        context = ConversationContext()
        self.store.put(namespace, "context", context.model_dump())
        return context
    
    def update_conversation_context(self, context: ConversationContext, user_id: str = config.DEFAULT_USER_ID):
        """Update the conversation context in memory.
        
        Args:
            context: The updated conversation context.
            user_id: The user ID to update the context for.
        """
        namespace = (config.MEMORY_NAMESPACE_CONVERSATION, user_id)
        
        # Update the context
        self.store.put(namespace, "context", context.model_dump())
    
    def search_memory(self, query: str, user_id: str = config.DEFAULT_USER_ID) -> Dict[str, List]:
        """Search all memories for relevant information.
        
        Args:
            query: The search query.
            user_id: The user ID to search memories for.
            
        Returns:
            Dictionary with search results organized by namespace.
        """
        results = {
            "user_profile": [],
            "business_knowledge": [],
            "conversation_history": []
        }
        
        # Search user profile
        profile_namespace = (config.MEMORY_NAMESPACE_USER_PROFILE, user_id)
        profile_results = self.store.search(profile_namespace, query=query, limit=1)
        if profile_results:
            results["user_profile"] = [UserProfile.model_validate(item.value) for item in profile_results]
        
        # Search business knowledge
        business_namespace = (config.MEMORY_NAMESPACE_BUSINESS, user_id)
        business_results = self.store.search(business_namespace, query=query, limit=5)
        if business_results:
            results["business_knowledge"] = [BusinessEntity.model_validate(item.value) for item in business_results]
        
        # Search conversation history
        conversation_namespace = (config.MEMORY_NAMESPACE_CONVERSATION, user_id)
        conversation_results = self.store.search(conversation_namespace, query=query, limit=3)
        if conversation_results:
            results["conversation_history"] = [item.value for item in conversation_results if item.key != "context"]
        
        return results


# Test the memory manager
if __name__ == "__main__":
    memory_manager = MemoryManager()
    
    # Test creating and retrieving a user profile
    test_messages = [
        {"role": "user", "content": "My name is Aaron and I work as a software developer."}
    ]
    
    print("Updating user profile...")
    memory_manager.update_user_profile(test_messages)
    
    print("Retrieving user profile...")
    profile = memory_manager.get_user_profile()
    print(f"Profile: {profile}")
    
    # Test search
    print("Searching memory...")
    search_results = memory_manager.search_memory("software developer")
    print(f"Search results: {search_results}")
