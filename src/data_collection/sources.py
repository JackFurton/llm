import os
import re
import time
import logging
import random
import json
from typing import List, Dict, Any, Optional, Union
import urllib.request
import urllib.parse
import urllib.error
from html import unescape
import ssl

from .collector import DataSource

# Configure logging
logger = logging.getLogger('data_collector.sources')

# Create SSL context that ignores certificate validation for simplicity
# Note: In production, you should use proper certificate validation
ssl_context = ssl._create_unverified_context()

class WikipediaSource(DataSource):
    """
    Data source for Wikipedia articles
    """
    
    def __init__(self):
        super().__init__(name="wikipedia")
        self.api_url = "https://en.wikipedia.org/w/api.php"
    
    def get_items(self, query: str = "", limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Get Wikipedia articles based on search query"""
        if not query:
            # If no query provided, use random articles
            return self._get_random_articles(limit)
        else:
            # Search for articles
            return self._search_articles(query, limit)
    
    def _get_random_articles(self, limit: int) -> List[Dict[str, Any]]:
        """Get random Wikipedia articles"""
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnlimit": str(limit),
            "rnnamespace": "0"  # Main article namespace
        }
        
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url, context=ssl_context) as response:
                data = json.loads(response.read().decode())
                
            articles = []
            for article in data.get("query", {}).get("random", []):
                articles.append({
                    "id": article.get("id"),
                    "title": article.get("title"),
                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(article.get('title', '').replace(' ', '_'))}"
                })
            
            # Fetch content for each article
            for article in articles:
                article["content"] = self._get_article_content(article["title"])
                time.sleep(1)  # Be nice to Wikipedia API
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting random Wikipedia articles: {e}")
            return []
    
    def _search_articles(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for Wikipedia articles"""
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": str(limit)
        }
        
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url, context=ssl_context) as response:
                data = json.loads(response.read().decode())
                
            articles = []
            for article in data.get("query", {}).get("search", []):
                articles.append({
                    "id": article.get("pageid"),
                    "title": article.get("title"),
                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(article.get('title', '').replace(' ', '_'))}"
                })
            
            # Fetch content for each article
            for article in articles:
                article["content"] = self._get_article_content(article["title"])
                time.sleep(1)  # Be nice to Wikipedia API
            
            return articles
            
        except Exception as e:
            logger.error(f"Error searching Wikipedia articles: {e}")
            return []
    
    def _get_article_content(self, title: str) -> str:
        """Get content of a Wikipedia article"""
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": "true",
            "exsectionformat": "plain"
        }
        
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url, context=ssl_context) as response:
                data = json.loads(response.read().decode())
                
            pages = data.get("query", {}).get("pages", {})
            if pages:
                page_id = next(iter(pages))
                return pages[page_id].get("extract", "")
            
            return ""
            
        except Exception as e:
            logger.error(f"Error getting Wikipedia article content: {e}")
            return ""
    
    def extract_text(self, item: Dict[str, Any]) -> str:
        """Extract text from a Wikipedia article"""
        return item.get("content", "")
    
    def clean_text(self, text: str) -> str:
        """Clean Wikipedia article text"""
        text = super().clean_text(text)
        
        # Remove references like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        return text


class GutenbergSource(DataSource):
    """
    Data source for Project Gutenberg books
    """
    
    def __init__(self):
        super().__init__(name="gutenberg")
        self.catalog_url = "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"
        self.mirror_url = "https://www.gutenberg.org/files/"
        
        # List of popular book IDs for when no query is provided
        self.popular_books = [
            1342,   # Pride and Prejudice
            11,     # Alice's Adventures in Wonderland
            1661,   # The Adventures of Sherlock Holmes
            2701,   # Moby Dick
            84,     # Frankenstein
            1400,   # Great Expectations
            98,     # A Tale of Two Cities
            1952,   # The Yellow Wallpaper
            345,    # Dracula
            1080,   # A Modest Proposal
            74,     # The Adventures of Tom Sawyer
            2814,   # Dubliners
            1184,   # The Count of Monte Cristo
            1232,   # The Prince
            2500,   # Siddhartha
            174,    # The Picture of Dorian Gray
            768,    # Wuthering Heights
            1260,   # Jane Eyre
            16328,  # Beowulf
            2591,   # Grimm's Fairy Tales
        ]
    
    def get_items(self, query: str = "", limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Get books from Project Gutenberg"""
        # For simplicity, we'll use a predefined list of popular books
        # In a real implementation, you would parse the catalog or use an API
        
        book_ids = []
        if query:
            # In a real implementation, you would search the catalog
            # For now, we'll just use a subset of popular books
            random.shuffle(self.popular_books)
            book_ids = self.popular_books[:limit]
        else:
            # Use random selection from popular books
            book_ids = random.sample(self.popular_books, min(limit, len(self.popular_books)))
        
        books = []
        for book_id in book_ids:
            try:
                book = self._get_book_info(book_id)
                if book:
                    books.append(book)
                time.sleep(1)  # Be nice to Gutenberg
            except Exception as e:
                logger.error(f"Error getting book {book_id}: {e}")
        
        return books
    
    def _get_book_info(self, book_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a book"""
        # In a real implementation, you would parse the RDF metadata
        # For simplicity, we'll just try to download the text directly
        
        # Try different file patterns
        patterns = [
            f"{book_id}/{book_id}-0.txt",
            f"{book_id}/{book_id}.txt",
            f"{book_id}-0.txt",
            f"{book_id}.txt"
        ]
        
        content = ""
        for pattern in patterns:
            try:
                url = f"{self.mirror_url}{pattern}"
                with urllib.request.urlopen(url, context=ssl_context, timeout=10) as response:
                    content = response.read().decode('utf-8', errors='replace')
                    break
            except Exception:
                continue
        
        if not content:
            return None
        
        # Extract title from content (simple heuristic)
        title_match = re.search(r'Title: (.*?)(\n\n|\r\n\r\n)', content, re.DOTALL)
        title = title_match.group(1).strip() if title_match else f"Book {book_id}"
        
        # Extract author from content (simple heuristic)
        author_match = re.search(r'Author: (.*?)(\n\n|\r\n\r\n)', content, re.DOTALL)
        author = author_match.group(1).strip() if author_match else "Unknown"
        
        return {
            "id": book_id,
            "title": title,
            "author": author,
            "url": f"https://www.gutenberg.org/ebooks/{book_id}",
            "content": content
        }
    
    def extract_text(self, item: Dict[str, Any]) -> str:
        """Extract text from a Gutenberg book"""
        content = item.get("content", "")
        
        # Try to extract the main text, skipping headers and footers
        # This is a simple heuristic and might not work for all books
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG EBOOK",
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "*END THE SMALL PRINT",
            "*** START OF THE PROJECT GUTENBERG",
            "***START OF THE PROJECT GUTENBERG",
        ]
        
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG EBOOK",
            "*** END OF THE PROJECT GUTENBERG EBOOK",
            "End of the Project Gutenberg",
            "End of Project Gutenberg",
            "***END OF THE PROJECT GUTENBERG",
        ]
        
        # Find start of main text
        start_pos = 0
        for marker in start_markers:
            pos = content.find(marker)
            if pos != -1:
                start_pos = content.find("\n", pos) + 1
                break
        
        # Find end of main text
        end_pos = len(content)
        for marker in end_markers:
            pos = content.find(marker)
            if pos != -1:
                end_pos = pos
                break
        
        # Extract main text
        main_text = content[start_pos:end_pos].strip()
        
        # If extraction failed, use the whole content
        if not main_text or len(main_text) < 1000:
            return content
        
        return main_text
    
    def clean_text(self, text: str) -> str:
        """Clean Gutenberg book text"""
        text = super().clean_text(text)
        
        # Remove chapter headings (simple heuristic)
        text = re.sub(r'\n\s*CHAPTER [IVXLCDM0-9]+\.?\s*\n', '\n\n', text, flags=re.IGNORECASE)
        
        return text


class NewsSource(DataSource):
    """
    Data source for news articles
    Uses a simple RSS feed parser
    """
    
    def __init__(self):
        super().__init__(name="news")
        
        # List of RSS feeds
        self.feeds = [
            "http://rss.cnn.com/rss/cnn_topstories.rss",
            "http://feeds.bbci.co.uk/news/rss.xml",
            "http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
            "http://feeds.washingtonpost.com/rss/world",
            "http://feeds.reuters.com/reuters/topNews",
        ]
    
    def get_items(self, query: str = "", limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Get news articles from RSS feeds"""
        articles = []
        
        # Use specified feed if provided in kwargs
        feeds = [kwargs.get("feed")] if kwargs.get("feed") else self.feeds
        
        for feed_url in feeds:
            try:
                feed_articles = self._parse_rss_feed(feed_url, limit)
                articles.extend(feed_articles)
                
                if len(articles) >= limit:
                    articles = articles[:limit]
                    break
                
                time.sleep(1)  # Be nice to news sites
            except Exception as e:
                logger.error(f"Error parsing feed {feed_url}: {e}")
        
        # Filter by query if provided
        if query and articles:
            filtered_articles = []
            for article in articles:
                if (query.lower() in article.get("title", "").lower() or 
                    query.lower() in article.get("description", "").lower()):
                    filtered_articles.append(article)
            
            articles = filtered_articles[:limit]
        
        # Fetch full content for each article
        for article in articles:
            try:
                article["content"] = self._fetch_article_content(article["url"])
                time.sleep(1)  # Be nice to news sites
            except Exception as e:
                logger.error(f"Error fetching article content: {e}")
                article["content"] = article.get("description", "")
        
        return articles
    
    def _parse_rss_feed(self, feed_url: str, limit: int) -> List[Dict[str, Any]]:
        """Parse an RSS feed"""
        try:
            with urllib.request.urlopen(feed_url, context=ssl_context, timeout=10) as response:
                feed_content = response.read().decode('utf-8', errors='replace')
            
            # Simple regex-based parsing (for demonstration)
            # In a real implementation, use a proper XML parser
            items = []
            
            # Extract items
            item_pattern = r'<item>(.*?)</item>'
            for item_match in re.finditer(item_pattern, feed_content, re.DOTALL):
                item_xml = item_match.group(1)
                
                # Extract title
                title_match = re.search(r'<title>(.*?)</title>', item_xml, re.DOTALL)
                title = unescape(title_match.group(1)) if title_match else ""
                
                # Extract link
                link_match = re.search(r'<link>(.*?)</link>', item_xml, re.DOTALL)
                link = link_match.group(1) if link_match else ""
                
                # Extract description
                desc_match = re.search(r'<description>(.*?)</description>', item_xml, re.DOTALL)
                description = unescape(desc_match.group(1)) if desc_match else ""
                
                # Clean up description (remove HTML)
                description = re.sub(r'<[^>]+>', '', description)
                
                if title and link:
                    items.append({
                        "title": title,
                        "url": link,
                        "description": description
                    })
                
                if len(items) >= limit:
                    break
            
            return items
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {e}")
            return []
    
    def _fetch_article_content(self, url: str) -> str:
        """Fetch the content of a news article"""
        try:
            with urllib.request.urlopen(url, context=ssl_context, timeout=10) as response:
                html_content = response.read().decode('utf-8', errors='replace')
            
            # Extract article content (simple heuristic)
            # In a real implementation, use a proper HTML parser or newspaper library
            
            # Remove scripts and styles
            html_content = re.sub(r'<script.*?>.*?</script>', '', html_content, flags=re.DOTALL)
            html_content = re.sub(r'<style.*?>.*?</style>', '', html_content, flags=re.DOTALL)
            
            # Look for article content in common containers
            content_patterns = [
                r'<article[^>]*>(.*?)</article>',
                r'<div[^>]*class="[^"]*article[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*id="[^"]*content[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*story[^"]*"[^>]*>(.*?)</div>'
            ]
            
            content = ""
            for pattern in content_patterns:
                matches = re.finditer(pattern, html_content, re.DOTALL)
                for match in matches:
                    content = match.group(1)
                    if len(content) > 500:  # Reasonable article length
                        break
                
                if content:
                    break
            
            # If no content found, use the whole body
            if not content:
                body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL)
                if body_match:
                    content = body_match.group(1)
            
            # Remove HTML tags
            content = re.sub(r'<[^>]+>', ' ', content)
            
            # Decode HTML entities
            content = unescape(content)
            
            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content
            
        except Exception as e:
            logger.error(f"Error fetching article content from {url}: {e}")
            return ""
    
    def extract_text(self, item: Dict[str, Any]) -> str:
        """Extract text from a news article"""
        content = item.get("content", "")
        if not content:
            content = item.get("description", "")
        
        title = item.get("title", "")
        
        # Combine title and content
        return f"{title}\n\n{content}"
    
    def clean_text(self, text: str) -> str:
        """Clean news article text"""
        text = super().clean_text(text)
        
        # Remove common news article artifacts
        text = re.sub(r'Share this with.*?Email', '', text)
        text = re.sub(r'Image caption.*?Image copyright', '', text)
        text = re.sub(r'Follow us on.*?Twitter', '', text)
        
        return text


class RedditSource(DataSource):
    """
    Data source for Reddit posts and comments
    """
    
    def __init__(self):
        super().__init__(name="reddit")
        
        # List of popular subreddits
        self.subreddits = [
            "AskReddit", "explainlikeimfive", "todayilearned", "science",
            "worldnews", "books", "history", "philosophy", "writing",
            "askscience", "dataisbeautiful", "space", "futurology"
        ]
    
    def get_items(self, query: str = "", limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Get posts from Reddit"""
        subreddit = kwargs.get("subreddit", "")
        
        if not subreddit:
            # If no specific subreddit, choose a random one
            subreddit = random.choice(self.subreddits)
        
        # Construct URL for JSON API
        sort_by = "top"
        time_filter = "month"
        url = f"https://www.reddit.com/r/{subreddit}/{sort_by}.json?t={time_filter}&limit={limit}"
        
        if query:
            # Use search instead
            url = f"https://www.reddit.com/r/{subreddit}/search.json?q={urllib.parse.quote(query)}&restrict_sr=on&limit={limit}"
        
        # Add User-Agent to avoid 429 errors
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            posts = []
            for post in data.get("data", {}).get("children", []):
                post_data = post.get("data", {})
                
                # Skip posts without text
                if not post_data.get("selftext"):
                    continue
                
                posts.append({
                    "id": post_data.get("id"),
                    "title": post_data.get("title"),
                    "url": f"https://www.reddit.com{post_data.get('permalink')}",
                    "author": post_data.get("author"),
                    "subreddit": post_data.get("subreddit"),
                    "selftext": post_data.get("selftext"),
                    "comments": []
                })
            
            # Fetch comments for each post
            for post in posts:
                post["comments"] = self._fetch_comments(post["url"])
                time.sleep(1)  # Be nice to Reddit
            
            return posts
            
        except Exception as e:
            logger.error(f"Error getting Reddit posts: {e}")
            return []
    
    def _fetch_comments(self, post_url: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch comments for a Reddit post"""
        url = f"{post_url}.json"
        
        # Add User-Agent to avoid 429 errors
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            comments = []
            
            # Reddit returns a list with post data and comments data
            if len(data) >= 2:
                comment_data = data[1].get("data", {}).get("children", [])
                
                for comment in comment_data:
                    if comment.get("kind") == "t1":  # Regular comment
                        comment_body = comment.get("data", {}).get("body", "")
                        if comment_body and len(comment_body) > 50:
                            comments.append({
                                "id": comment.get("data", {}).get("id"),
                                "author": comment.get("data", {}).get("author"),
                                "body": comment_body
                            })
                    
                    if len(comments) >= limit:
                        break
            
            return comments
            
        except Exception as e:
            logger.error(f"Error fetching comments for {post_url}: {e}")
            return []
    
    def extract_text(self, item: Dict[str, Any]) -> str:
        """Extract text from a Reddit post and its comments"""
        title = item.get("title", "")
        selftext = item.get("selftext", "")
        subreddit = item.get("subreddit", "")
        
        # Combine post title and body
        text = f"# {title}\n\nFrom r/{subreddit}\n\n{selftext}\n\n"
        
        # Add top comments
        if item.get("comments"):
            text += "## Top Comments:\n\n"
            for comment in item.get("comments", []):
                text += f"{comment.get('body', '')}\n\n"
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean Reddit post text"""
        text = super().clean_text(text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)  # Links
        
        # Remove common Reddit artifacts
        text = re.sub(r'Edit:', '\n\nEdit:', text)
        text = re.sub(r'TL;DR:', '\n\nTL;DR:', text)
        
        return text
