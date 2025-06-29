import os
import logging
import time
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_collector')

class DataSource:
    """Base class for data sources"""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_items(self, query: str = "", limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Get items from the data source
        
        Args:
            query: Search query or parameters
            limit: Maximum number of items to retrieve
            **kwargs: Additional source-specific parameters
            
        Returns:
            List of dictionaries containing the items
        """
        raise NotImplementedError("Subclasses must implement get_items")
    
    def extract_text(self, item: Dict[str, Any]) -> str:
        """
        Extract text content from an item
        
        Args:
            item: Item dictionary from get_items
            
        Returns:
            Extracted text content
        """
        raise NotImplementedError("Subclasses must implement extract_text")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Basic cleaning - subclasses can override for source-specific cleaning
        text = text.strip()
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        return text


class DataCollector:
    """
    Main class for collecting data from various sources
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        self.sources: Dict[str, DataSource] = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        self.stats = {
            "total_files": 0,
            "total_chars": 0,
            "sources": {}
        }
    
    def add_source(self, source: DataSource):
        """Add a data source to the collector"""
        self.sources[source.name] = source
        self.stats["sources"][source.name] = {
            "files": 0,
            "chars": 0
        }
        logger.info(f"Added data source: {source.name}")
    
    def collect_from_source(self, 
                           source_name: str, 
                           query: str = "", 
                           limit: int = 10, 
                           **kwargs) -> List[str]:
        """
        Collect data from a specific source
        
        Args:
            source_name: Name of the source to collect from
            query: Search query or parameters
            limit: Maximum number of items to retrieve
            **kwargs: Additional source-specific parameters
            
        Returns:
            List of file paths where data was saved
        """
        if source_name not in self.sources:
            raise ValueError(f"Unknown source: {source_name}")
        
        source = self.sources[source_name]
        logger.info(f"Collecting data from {source_name} with query: '{query}', limit: {limit}")
        
        # Get items from source
        items = source.get_items(query=query, limit=limit, **kwargs)
        logger.info(f"Retrieved {len(items)} items from {source_name}")
        
        # Process items
        saved_files = []
        for item in items:
            try:
                # Extract and clean text
                text = source.extract_text(item)
                text = source.clean_text(text)
                
                # Skip if text is too short
                if len(text) < 100:
                    logger.warning(f"Skipping short text ({len(text)} chars)")
                    continue
                
                # Generate filename based on content hash
                content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:10]
                filename = f"{source_name}_{content_hash}.txt"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # Update stats
                self.stats["total_files"] += 1
                self.stats["total_chars"] += len(text)
                self.stats["sources"][source_name]["files"] += 1
                self.stats["sources"][source_name]["chars"] += len(text)
                
                saved_files.append(filepath)
                logger.info(f"Saved {len(text)} chars to {filepath}")
                
            except Exception as e:
                logger.error(f"Error processing item: {e}")
        
        return saved_files
    
    def collect_from_all_sources(self, 
                                query: str = "", 
                                limit_per_source: int = 5,
                                max_workers: int = 4) -> Dict[str, List[str]]:
        """
        Collect data from all registered sources in parallel
        
        Args:
            query: Search query or parameters
            limit_per_source: Maximum number of items to retrieve per source
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping source names to lists of saved file paths
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_source = {
                executor.submit(self.collect_from_source, source_name, query, limit_per_source): source_name
                for source_name in self.sources
            }
            
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    saved_files = future.result()
                    results[source_name] = saved_files
                except Exception as e:
                    logger.error(f"Error collecting from {source_name}: {e}")
                    results[source_name] = []
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return self.stats
