#!/usr/bin/env python3
"""
Data preprocessing script for the custom LLM project.
This script preprocesses text data and prepares it for training.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_preprocessing.preprocessor import TextPreprocessor
from src.data_preprocessing.filters import (
    LanguageFilter,
    ContentFilter,
    QualityFilter,
    DuplicateFilter
)
from src.data_preprocessing.normalizers import (
    TextNormalizer,
    HTMLCleaner,
    WhitespaceCleaner,
    MarkdownCleaner
)
from src.data_preprocessing.augmenters import (
    SynonymReplacer,
    BackTranslator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_preprocessor.cli')

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess text data for LLM training')
    
    parser.add_argument('--input-dir', type=str, default='data/raw',
                        help='Directory containing input files')
    
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Directory to save processed files')
    
    parser.add_argument('--file-pattern', type=str, default='*.txt',
                        help='Pattern to match input files')
    
    parser.add_argument('--language', type=str, default='en',
                        help='Language filter (default: en)')
    
    parser.add_argument('--no-language-filter', action='store_true',
                        help='Disable language filtering')
    
    parser.add_argument('--no-content-filter', action='store_true',
                        help='Disable content filtering')
    
    parser.add_argument('--no-quality-filter', action='store_true',
                        help='Disable quality filtering')
    
    parser.add_argument('--no-duplicate-filter', action='store_true',
                        help='Disable duplicate filtering')
    
    parser.add_argument('--min-length', type=int, default=100,
                        help='Minimum text length (default: 100)')
    
    parser.add_argument('--max-length', type=int, default=100000,
                        help='Maximum text length (default: 100000)')
    
    parser.add_argument('--normalize', action='store_true',
                        help='Apply text normalization')
    
    parser.add_argument('--clean-html', action='store_true',
                        help='Clean HTML tags and entities')
    
    parser.add_argument('--clean-markdown', action='store_true',
                        help='Clean Markdown formatting')
    
    parser.add_argument('--clean-whitespace', action='store_true',
                        help='Clean and normalize whitespace')
    
    parser.add_argument('--augment', action='store_true',
                        help='Apply text augmentation')
    
    parser.add_argument('--synonym-replace', action='store_true',
                        help='Replace words with synonyms')
    
    parser.add_argument('--back-translate', action='store_true',
                        help='Apply back-translation augmentation')
    
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum number of parallel workers')
    
    parser.add_argument('--stats-file', type=str, default='data/preprocessing_stats.json',
                        help='File to save preprocessing statistics')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create preprocessor
    preprocessor = TextPreprocessor(output_dir=args.output_dir)
    
    # Add filters
    if not args.no_language_filter:
        preprocessor.add_filter(LanguageFilter(allowed_languages=[args.language]))
    
    if not args.no_content_filter:
        preprocessor.add_filter(ContentFilter())
    
    if not args.no_quality_filter:
        preprocessor.add_filter(QualityFilter(
            min_length=args.min_length,
            max_length=args.max_length
        ))
    
    if not args.no_duplicate_filter:
        preprocessor.add_filter(DuplicateFilter())
    
    # Add normalizers
    if args.normalize:
        preprocessor.add_normalizer(TextNormalizer())
    
    if args.clean_html:
        preprocessor.add_normalizer(HTMLCleaner())
    
    if args.clean_whitespace:
        preprocessor.add_normalizer(WhitespaceCleaner())
    
    if args.clean_markdown:
        preprocessor.add_normalizer(MarkdownCleaner())
    
    # Add augmenters
    if args.augment or args.synonym_replace:
        preprocessor.add_augmenter(SynonymReplacer(
            augmentation_probability=0.5,
            replacement_probability=0.1
        ))
    
    if args.augment or args.back_translate:
        preprocessor.add_augmenter(BackTranslator(
            augmentation_probability=0.3
        ))
    
    # Process files
    logger.info(f"Starting preprocessing from {args.input_dir} to {args.output_dir}")
    results = preprocessor.preprocess_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.file_pattern,
        max_workers=args.max_workers
    )
    
    # Save statistics
    stats = preprocessor.get_stats()
    
    os.makedirs(os.path.dirname(args.stats_file), exist_ok=True)
    with open(args.stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Preprocessing statistics saved to {args.stats_file}")
    
    # Print summary
    print("\nPreprocessing Summary:")
    print(f"Total files processed: {stats['total_files_processed']}")
    print(f"Total files output: {stats['total_files_output']}")
    print(f"Files filtered: {stats['files_filtered']}")
    print(f"Input characters: {stats['total_chars_input']}")
    print(f"Output characters: {stats['total_chars_output']}")
    
    # Print filter stats
    for name, processor_stats in stats['processors'].items():
        if processor_stats['type'] == 'filter':
            print(f"- {name}: filtered {processor_stats['files_filtered']} files")
        elif processor_stats['type'] == 'normalizer':
            print(f"- {name}: removed {processor_stats['chars_removed']} characters")
        elif processor_stats['type'] == 'augmenter':
            print(f"- {name}: augmented {processor_stats['files_augmented']} files")

if __name__ == '__main__':
    main()
