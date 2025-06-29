import re
import logging
import string
import unicodedata
from typing import List, Dict, Any, Optional, Set, Union

from .preprocessor import TextProcessor

# Configure logging
logger = logging.getLogger('text_preprocessor.normalizers')

class TextNormalizer(TextProcessor):
    """
    Base text normalizer that applies common normalization techniques
    """
    
    def __init__(self):
        super().__init__(name="text_normalizer")
    
    def process(self, text: str) -> str:
        """
        Apply basic text normalization
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])([^\s])', r'\1 \2', text)  # Add space after punctuation
        
        # Fix quotes
        text = re.sub(r'[""]', '"', text)  # Normalize quotes
        text = re.sub(r'['']', "'", text)  # Normalize apostrophes
        
        # Fix dashes
        text = re.sub(r'[-‐‑‒–—―]', '-', text)  # Normalize dashes
        
        # Fix ellipses
        text = re.sub(r'\.{2,}', '...', text)  # Normalize ellipses
        
        # Remove control characters
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\t\r')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


class HTMLCleaner(TextProcessor):
    """
    Clean HTML tags and entities from text
    """
    
    def __init__(self):
        super().__init__(name="html_cleaner")
        
        # Try to import html parser
        try:
            from html import unescape
            self.unescape = unescape
        except ImportError:
            # Fallback for older Python versions
            import html.parser
            self.unescape = html.parser.HTMLParser().unescape
    
    def process(self, text: str) -> str:
        """
        Remove HTML tags and decode HTML entities
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Check if text contains HTML
        if not re.search(r'<[a-zA-Z]', text) and not re.search(r'&[a-zA-Z]+;', text):
            return text
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Decode HTML entities
        text = self.unescape(text)
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def should_process(self, text: str) -> bool:
        """Only process text that contains HTML tags or entities"""
        return re.search(r'<[a-zA-Z]', text) is not None or re.search(r'&[a-zA-Z]+;', text) is not None


class WhitespaceCleaner(TextProcessor):
    """
    Clean and normalize whitespace in text
    """
    
    def __init__(self):
        super().__init__(name="whitespace_cleaner")
    
    def process(self, text: str) -> str:
        """
        Normalize whitespace in text
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize line breaks (replace multiple line breaks with two)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove spaces at the beginning of lines
        text = re.sub(r'(?<=\n) +', '', text)
        
        # Remove trailing spaces
        text = re.sub(r' +(?=\n|$)', '', text)
        
        # Ensure text starts and ends without whitespace
        text = text.strip()
        
        return text


class MarkdownCleaner(TextProcessor):
    """
    Clean and normalize Markdown formatting
    """
    
    def __init__(self, keep_headings: bool = True, keep_lists: bool = True):
        super().__init__(name="markdown_cleaner")
        self.keep_headings = keep_headings
        self.keep_lists = keep_lists
    
    def process(self, text: str) -> str:
        """
        Clean Markdown formatting
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Check if text contains Markdown
        if not re.search(r'[*_#`\[\]\(\)]', text):
            return text
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Process headings
        if self.keep_headings:
            # Convert headings to plain text with newlines
            text = re.sub(r'^(#{1,6})\s+(.+?)$', lambda m: m.group(2).upper() + '\n', text, flags=re.MULTILINE)
        else:
            # Remove headings
            text = re.sub(r'^#{1,6}\s+.+?$', '', text, flags=re.MULTILINE)
        
        # Process lists
        if self.keep_lists:
            # Convert list items to plain text with dashes
            text = re.sub(r'^(\s*)[-*+]\s+(.+?)$', lambda m: m.group(1) + '- ' + m.group(2), text, flags=re.MULTILINE)
            text = re.sub(r'^(\s*)\d+\.\s+(.+?)$', lambda m: m.group(1) + '- ' + m.group(2), text, flags=re.MULTILINE)
        else:
            # Remove list markers
            text = re.sub(r'^(\s*)[-*+]\s+', r'\1', text, flags=re.MULTILINE)
            text = re.sub(r'^(\s*)\d+\.\s+', r'\1', text, flags=re.MULTILINE)
        
        # Remove emphasis markers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'__(.*?)__', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
        text = re.sub(r'_(.*?)_', r'\1', text)  # Italic
        
        # Remove inline code
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # Convert links to plain text
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)
        
        # Convert images to alt text
        text = re.sub(r'!\[(.*?)\]\((.*?)\)', r'\1', text)
        
        # Remove horizontal rules
        text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\*{3,}$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^_{3,}$', '', text, flags=re.MULTILINE)
        
        # Fix spacing
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
    
    def should_process(self, text: str) -> bool:
        """Only process text that contains Markdown formatting"""
        return re.search(r'[*_#`\[\]\(\)]', text) is not None
