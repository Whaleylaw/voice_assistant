"""
Main entry point for the voice assistant.
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore

import config
from agent_nodes import (
    AssistantState, 
    voice_input, 
    search_memory, 
    generate_response, 
    voice_output
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_assistant(user_id=config.DEFAULT_USER_ID):
    """Create the voice assistant graph.
    
    Args:
        user_id: The user ID to associate with this assistant.
        
    Returns:
        The compiled voice assistant graph.
    """
    logger.info(f"Creating voice assistant for user {user_id}...")
    
    # Create memory store
    store = InMemoryStore(
        index={
            "dims": config.EMBEDDING_DIMENSIONS,
            "embed": config.EMBEDDING_MODEL,
        }
    )
    
    # Create graph builder
    builder = StateGraph(AssistantState)
    
    # Add nodes to the graph
    builder.add_node("voice_input", voice_input)
    builder.add_node("search_memory", search_memory)
    builder.add_node("generate_response", generate_response)
    builder.add_node("voice_output", voice_output)
    
    # Add edges
    builder.add_edge(START, "voice_input")
    builder.add_edge("voice_input", "search_memory")
    builder.add_edge("search_memory", "generate_response")
    builder.add_edge("generate_response", "voice_output")
    builder.add_edge("voice_output", END)
    
    # Compile the graph with the store
    return builder.compile(store=store)


def run_assistant(n_turns=1, user_id=config.DEFAULT_USER_ID):
    """Run the voice assistant for a specified number of turns.
    
    Args:
        n_turns: Number of conversation turns to run.
        user_id: The user ID to associate with this session.
    """
    logger.info(f"Starting voice assistant session for user {user_id}...")
    
    # Create the assistant
    assistant = create_assistant(user_id)
    
    # Initialize state
    state = {
        "messages": [],
        "user_id": user_id
    }
    
    print(f"\n{'='*50}")
    print(f"Voice Assistant ready! You can have {n_turns} turns of conversation.")
    print(f"Press Enter after speaking to end your turn.")
    print(f"{'='*50}\n")
    
    # Run the assistant for n_turns
    for i in range(n_turns):
        print(f"\nTurn {i+1}/{n_turns}")
        
        # Run one turn of the conversation
        result = assistant.invoke(state)
        
        # Update state for next turn
        state = result
    
    print(f"\n{'='*50}")
    print("Voice Assistant session ended.")


if __name__ == "__main__":
    # Run the assistant with default settings
    run_assistant(n_turns=3)
