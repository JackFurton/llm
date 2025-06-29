#!/usr/bin/env python3
"""
Run the data curation web interface.
"""

import os
import sys
import argparse
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.web_interface import run_app

def parse_args():
    parser = argparse.ArgumentParser(description='Run the data curation web interface')
    
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to run the server on')
    
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on')
    
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    parser.add_argument('--raw-dir', type=str, default='data/raw',
                        help='Directory containing raw data files')
    
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Directory containing processed data files')
    
    parser.add_argument('--curated-dir', type=str, default='data/curated',
                        help='Directory to save curated data files')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create directories if they don't exist
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.curated_dir, exist_ok=True)
    
    # Run the app
    run_app(
        host=args.host,
        port=args.port,
        debug=args.debug,
        raw_data_dir=args.raw_dir,
        processed_data_dir=args.processed_dir,
        curated_data_dir=args.curated_dir
    )

if __name__ == '__main__':
    main()
