"""
Test script for the voice assistant.
"""

import sys
import logging
from main import run_assistant

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Get the number of turns from command line arguments
    n_turns = 3  # Default to 3 turns
    if len(sys.argv) > 1:
        try:
            n_turns = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid number of turns: {sys.argv[1]}. Using default.")
    
    # Run the assistant
    logger.info(f"Starting test with {n_turns} turns...")
    run_assistant(n_turns=n_turns)
