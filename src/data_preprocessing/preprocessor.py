import os
import logging
import json
import re
import time
from typing import List, Dict, Any, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('text_preprocessor')

class TextProcessor:
    """Base class for text processors"""
    
    def __init__(self, name: str):
        self.name = name
    
    def process(self, text: str) -> str:
        """
        Process the text
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        raise NotImplementedError("Subclasses must implement process")
    
    def should_process(self, text: str) -> bool:
        """
        Determine if this processor should be applied to the text
        
        Args:
            text: Input text
            
        Returns:
            True if the processor should be applied, False otherwise
        """
        return True


class TextPreprocessor:
    """
    Main class for preprocessing text data
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        self.filters = []
        self.normalizers = []
        self.augmenters = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        self.stats = {
            "total_files_processed": 0,
            "total_files_output": 0,
            "total_chars_input": 0,
            "total_chars_output": 0,
            "files_filtered": 0,
            "processors": {}
        }
    
    def add_filter(self, filter_processor: TextProcessor):
        """Add a filter to the preprocessor"""
        self.filters.append(filter_processor)
        self.stats["processors"][filter_processor.name] = {
            "type": "filter",
            "files_filtered": 0
        }
        logger.info(f"Added filter: {filter_processor.name}")
    
    def add_normalizer(self, normalizer: TextProcessor):
        """Add a normalizer to the preprocessor"""
        self.normalizers.append(normalizer)
        self.stats["processors"][normalizer.name] = {
            "type": "normalizer",
            "chars_removed": 0
        }
        logger.info(f"Added normalizer: {normalizer.name}")
    
    def add_augmenter(self, augmenter: TextProcessor):
        """Add an augmenter to the preprocessor"""
        self.augmenters.append(augmenter)
        self.stats["processors"][augmenter.name] = {
            "type": "augmenter",
            "files_augmented": 0
        }
        logger.info(f"Added augmenter: {augmenter.name}")
    
    def preprocess_file(self, input_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Preprocess a single file
        
        Args:
            input_path: Path to the input file
            output_path: Path to save the output file (if None, will be generated)
            
        Returns:
            Path to the output file if successful, None otherwise
        """
        try:
            # Read input file
            with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            # Update stats
            self.stats["total_files_processed"] += 1
            self.stats["total_chars_input"] += len(text)
            
            # Apply filters
            for filter_proc in self.filters:
                try:
                    if filter_proc.should_process(text) and not filter_proc.process(text):
                        logger.info(f"File {input_path} filtered by {filter_proc.name}")
                        self.stats["files_filtered"] += 1
                        self.stats["processors"][filter_proc.name]["files_filtered"] += 1
                        return None
                except Exception as e:
                    logger.error(f"Error in filter {filter_proc.name}: {e}")
            
            # Apply normalizers
            for normalizer in self.normalizers:
                try:
                    if normalizer.should_process(text):
                        old_len = len(text)
                        text = normalizer.process(text)
                        chars_removed = old_len - len(text)
                        self.stats["processors"][normalizer.name]["chars_removed"] += chars_removed
                        logger.debug(f"{normalizer.name} removed {chars_removed} characters")
                except Exception as e:
                    logger.error(f"Error in normalizer {normalizer.name}: {e}")
            
            # Apply augmenters
            augmented_texts = [text]  # Start with the original text
            for augmenter in self.augmenters:
                try:
                    if augmenter.should_process(text):
                        new_texts = []
                        for t in augmented_texts:
                            augmented = augmenter.process(t)
                            if isinstance(augmented, list):
                                new_texts.extend(augmented)
                            else:
                                new_texts.append(augmented)
                        
                        augmented_texts = new_texts
                        self.stats["processors"][augmenter.name]["files_augmented"] += 1
                except Exception as e:
                    logger.error(f"Error in augmenter {augmenter.name}: {e}")
            
            # Save each augmented text
            saved_paths = []
            for i, augmented_text in enumerate(augmented_texts):
                # Skip empty texts
                if not augmented_text.strip():
                    continue
                
                # Generate output path if not provided
                if output_path is None:
                    # Generate filename based on content hash
                    content_hash = hashlib.md5(augmented_text.encode('utf-8')).hexdigest()[:10]
                    filename = f"processed_{content_hash}.txt"
                    out_path = os.path.join(self.output_dir, filename)
                else:
                    # If multiple augmented texts, add suffix
                    if i > 0:
                        base, ext = os.path.splitext(output_path)
                        out_path = f"{base}_aug{i}{ext}"
                    else:
                        out_path = output_path
                
                # Save to file
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(augmented_text)
                
                # Update stats
                self.stats["total_files_output"] += 1
                self.stats["total_chars_output"] += len(augmented_text)
                
                saved_paths.append(out_path)
                logger.info(f"Saved {len(augmented_text)} chars to {out_path}")
            
            return saved_paths[0] if saved_paths else None
            
        except Exception as e:
            logger.error(f"Error preprocessing file {input_path}: {e}")
            return None
    
    def preprocess_directory(self, 
                           input_dir: str, 
                           output_dir: Optional[str] = None,
                           file_pattern: str = "*.txt",
                           max_workers: int = 4) -> Dict[str, List[str]]:
        """
        Preprocess all files in a directory
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save output files (if None, uses self.output_dir)
            file_pattern: Pattern to match input files
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping input file paths to output file paths
        """
        import glob
        
        # Get list of input files
        input_files = glob.glob(os.path.join(input_dir, file_pattern))
        logger.info(f"Found {len(input_files)} files in {input_dir} matching pattern {file_pattern}")
        
        # Use output_dir if provided, otherwise use self.output_dir
        out_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {}
            
            for input_file in input_files:
                # Generate output path
                rel_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(out_dir, rel_path)
                
                # Create output directory if needed
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Submit preprocessing task
                future = executor.submit(self.preprocess_file, input_file, output_file)
                future_to_file[future] = input_file
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                input_file = future_to_file[future]
                try:
                    output_file = future.result()
                    results[input_file] = output_file
                except Exception as e:
                    logger.error(f"Error processing {input_file}: {e}")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        return self.stats
    
    def save_stats(self, path: str):
        """Save preprocessing statistics to a file"""
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Saved preprocessing statistics to {path}")
