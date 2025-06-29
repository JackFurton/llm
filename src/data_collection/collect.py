#!/usr/bin/env python3
"""
Data collection script for the custom LLM project.
This script collects text data from various sources and saves it to the data directory.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_collection.collector import DataCollector
from src.data_collection.sources import (
    WikipediaSource,
    GutenbergSource,
    NewsSource,
    RedditSource
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_collector.cli')

def parse_args():
    parser = argparse.ArgumentParser(description='Collect text data for LLM training')
    
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='Directory to save collected data')
    
    parser.add_argument('--sources', type=str, nargs='+',
                        choices=['wikipedia', 'gutenberg', 'news', 'reddit', 'all'],
                        default=['all'],
                        help='Data sources to collect from')
    
    parser.add_argument('--query', type=str, default='',
                        help='Search query for data collection')
    
    parser.add_argument('--limit', type=int, default=5,
                        help='Maximum number of items to collect per source')
    
    parser.add_argument('--subreddit', type=str, default='',
                        help='Specific subreddit to collect from (for Reddit source)')
    
    parser.add_argument('--news-feed', type=str, default='',
                        help='Specific news feed URL to collect from (for News source)')
    
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum number of parallel workers')
    
    parser.add_argument('--stats-file', type=str, default='data/collection_stats.json',
                        help='File to save collection statistics')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data collector
    collector = DataCollector(output_dir=args.output_dir)
    
    # Add requested sources
    sources = args.sources
    if 'all' in sources:
        sources = ['wikipedia', 'gutenberg', 'news', 'reddit']
    
    for source_name in sources:
        if source_name == 'wikipedia':
            collector.add_source(WikipediaSource())
        elif source_name == 'gutenberg':
            collector.add_source(GutenbergSource())
        elif source_name == 'news':
            collector.add_source(NewsSource())
        elif source_name == 'reddit':
            collector.add_source(RedditSource())
    
    # Collect data
    logger.info(f"Starting data collection from {', '.join(sources)}")
    
    if len(sources) == 1:
        # Collect from a single source
        source_name = sources[0]
        kwargs = {}
        
        if source_name == 'reddit' and args.subreddit:
            kwargs['subreddit'] = args.subreddit
        
        if source_name == 'news' and args.news_feed:
            kwargs['feed'] = args.news_feed
        
        saved_files = collector.collect_from_source(
            source_name=source_name,
            query=args.query,
            limit=args.limit,
            **kwargs
        )
        
        logger.info(f"Collected {len(saved_files)} files from {source_name}")
    else:
        # Collect from all sources
        results = collector.collect_from_all_sources(
            query=args.query,
            limit_per_source=args.limit,
            max_workers=args.max_workers
        )
        
        total_files = sum(len(files) for files in results.values())
        logger.info(f"Collected {total_files} files from {len(results)} sources")
    
    # Save statistics
    stats = collector.get_stats()
    
    os.makedirs(os.path.dirname(args.stats_file), exist_ok=True)
    with open(args.stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Collection statistics saved to {args.stats_file}")
    
    # Print summary
    print("\nCollection Summary:")
    print(f"Total files: {stats['total_files']}")
    print(f"Total characters: {stats['total_chars']}")
    
    for source_name, source_stats in stats['sources'].items():
        if source_stats['files'] > 0:
            print(f"- {source_name}: {source_stats['files']} files, {source_stats['chars']} characters")

if __name__ == '__main__':
    main()
