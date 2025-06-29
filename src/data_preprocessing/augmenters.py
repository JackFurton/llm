import re
import logging
import random
from typing import List, Dict, Any, Optional, Set, Union, Tuple

from .preprocessor import TextProcessor

# Configure logging
logger = logging.getLogger('text_preprocessor.augmenters')

class TextAugmenter(TextProcessor):
    """
    Base class for text augmentation
    """
    
    def __init__(self, name: str, augmentation_probability: float = 0.5):
        super().__init__(name=name)
        self.augmentation_probability = augmentation_probability
    
    def process(self, text: str) -> Union[str, List[str]]:
        """
        Augment the text
        
        Args:
            text: Input text
            
        Returns:
            Augmented text or list of augmented texts
        """
        raise NotImplementedError("Subclasses must implement process")
    
    def should_process(self, text: str) -> bool:
        """Only augment with the specified probability"""
        return random.random() < self.augmentation_probability


class SynonymReplacer(TextAugmenter):
    """
    Augment text by replacing words with synonyms
    """
    
    def __init__(self, 
                 replacement_probability: float = 0.1,
                 augmentation_probability: float = 0.5,
                 max_replacements: int = 100):
        super().__init__(name="synonym_replacer", augmentation_probability=augmentation_probability)
        self.replacement_probability = replacement_probability
        self.max_replacements = max_replacements
        
        # Try to import WordNet
        try:
            import nltk
            from nltk.corpus import wordnet
            self.wordnet = wordnet
            self._get_synonyms = self._get_synonyms_wordnet
            
            # Download WordNet if not already downloaded
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                logger.info("Downloading WordNet...")
                nltk.download('wordnet', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
            
            logger.info("Using WordNet for synonym replacement")
            
        except ImportError:
            logger.warning("NLTK WordNet not available, falling back to simple synonym dictionary")
            self._get_synonyms = self._get_synonyms_dict
            
            # Simple synonym dictionary for common words
            self.synonym_dict = {
                # Nouns
                "person": ["individual", "human", "man", "woman"],
                "time": ["moment", "instant", "period", "era"],
                "year": ["season", "annum", "twelvemonth"],
                "way": ["method", "manner", "mode", "means"],
                "day": ["date", "time", "occasion", "period"],
                "thing": ["item", "object", "article", "entity"],
                "man": ["person", "individual", "human", "gentleman"],
                "world": ["earth", "globe", "planet", "universe"],
                "life": ["existence", "being", "living", "lifetime"],
                "hand": ["palm", "grip", "grasp", "worker"],
                
                # Verbs
                "get": ["obtain", "acquire", "receive", "gain"],
                "make": ["create", "produce", "generate", "form"],
                "go": ["move", "travel", "proceed", "advance"],
                "know": ["understand", "comprehend", "recognize", "grasp"],
                "take": ["grab", "seize", "capture", "acquire"],
                "see": ["observe", "view", "witness", "perceive"],
                "come": ["arrive", "appear", "approach", "reach"],
                "think": ["believe", "consider", "contemplate", "reflect"],
                "look": ["see", "view", "watch", "observe"],
                "want": ["desire", "wish", "crave", "need"],
                
                # Adjectives
                "good": ["excellent", "fine", "superior", "quality"],
                "new": ["recent", "fresh", "novel", "modern"],
                "first": ["initial", "primary", "earliest", "foremost"],
                "last": ["final", "ultimate", "concluding", "terminal"],
                "long": ["extended", "lengthy", "prolonged", "extensive"],
                "great": ["excellent", "wonderful", "marvelous", "grand"],
                "little": ["small", "tiny", "slight", "minor"],
                "own": ["personal", "individual", "private", "particular"],
                "other": ["different", "alternative", "additional", "extra"],
                "old": ["aged", "ancient", "antique", "elderly"],
                
                # Adverbs
                "just": ["merely", "simply", "only", "exactly"],
                "well": ["skillfully", "properly", "thoroughly", "completely"],
                "even": ["still", "yet", "nevertheless", "nonetheless"],
                "back": ["backward", "behind", "rearward", "in return"],
                "there": ["in that place", "at that point", "in that respect"],
                "down": ["below", "downward", "underneath", "beneath"],
                "still": ["nevertheless", "yet", "however", "even now"],
                "here": ["at this place", "at this point", "in this situation"]
            }
    
    def _get_synonyms_wordnet(self, word: str) -> List[str]:
        """Get synonyms using WordNet"""
        from nltk import pos_tag
        from nltk.tokenize import word_tokenize
        
        # Get part of speech
        pos = pos_tag([word])[0][1]
        pos_map = {
            'N': self.wordnet.NOUN,
            'V': self.wordnet.VERB,
            'J': self.wordnet.ADJ,
            'R': self.wordnet.ADV
        }
        
        wordnet_pos = pos_map.get(pos[0], None)
        
        # Get synonyms
        synonyms = []
        for synset in self.wordnet.synsets(word, pos=wordnet_pos):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
        
        return synonyms[:5]  # Limit to 5 synonyms
    
    def _get_synonyms_dict(self, word: str) -> List[str]:
        """Get synonyms using simple dictionary"""
        return self.synonym_dict.get(word.lower(), [])
    
    def process(self, text: str) -> str:
        """
        Replace words with synonyms
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        words = re.findall(r'\b\w+\b', text)
        replacements = 0
        
        for word in words:
            # Skip short words
            if len(word) <= 3:
                continue
            
            # Only replace with the specified probability
            if random.random() > self.replacement_probability:
                continue
            
            # Get synonyms
            synonyms = self._get_synonyms(word)
            
            # Skip if no synonyms found
            if not synonyms:
                continue
            
            # Choose a random synonym
            synonym = random.choice(synonyms)
            
            # Replace the word with the synonym (preserve case)
            if word.islower():
                replacement = synonym.lower()
            elif word.isupper():
                replacement = synonym.upper()
            elif word[0].isupper():
                replacement = synonym.capitalize()
            else:
                replacement = synonym
            
            # Replace the word in the text (with word boundaries)
            text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text, count=1)
            
            replacements += 1
            if replacements >= self.max_replacements:
                break
        
        return text


class BackTranslator(TextAugmenter):
    """
    Augment text by translating it to another language and back
    """
    
    def __init__(self, 
                 intermediate_languages: List[str] = ["es", "fr", "de"],
                 augmentation_probability: float = 0.3):
        super().__init__(name="back_translator", augmentation_probability=augmentation_probability)
        self.intermediate_languages = intermediate_languages
        
        # Try to import translation libraries
        self.translator = None
        try:
            from googletrans import Translator
            self.translator = Translator()
            self._translate = self._translate_googletrans
            logger.info("Using googletrans for back-translation")
        except ImportError:
            try:
                import translators
                self._translate = self._translate_translators
                logger.info("Using translators for back-translation")
            except ImportError:
                logger.warning("No translation library available, back-translation disabled")
                self._translate = self._translate_dummy
    
    def _translate_googletrans(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using googletrans"""
        try:
            result = self.translator.translate(text, src=source_lang, dest=target_lang)
            return result.text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def _translate_translators(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using translators"""
        import translators as ts
        try:
            return ts.google(text, from_language=source_lang, to_language=target_lang)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def _translate_dummy(self, text: str, source_lang: str, target_lang: str) -> str:
        """Dummy translation function"""
        return text
    
    def process(self, text: str) -> List[str]:
        """
        Translate text to another language and back
        
        Args:
            text: Input text
            
        Returns:
            List of augmented texts
        """
        # If no translator available, return original text
        if self._translate == self._translate_dummy:
            return [text]
        
        # For long texts, only translate a portion
        if len(text) > 1000:
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Select a random subset of sentences (up to 10)
            if len(sentences) > 10:
                sample_size = min(10, len(sentences) // 2)
                sample_indices = sorted(random.sample(range(len(sentences)), sample_size))
                
                # Translate only the selected sentences
                for idx in sample_indices:
                    intermediate_lang = random.choice(self.intermediate_languages)
                    translated = self._translate(sentences[idx], "en", intermediate_lang)
                    back_translated = self._translate(translated, intermediate_lang, "en")
                    sentences[idx] = back_translated
                
                return [''.join(sentences)]
        
        # For shorter texts, create multiple versions with different languages
        augmented_texts = []
        
        # Original text is always included
        augmented_texts.append(text)
        
        # Create one augmented version per intermediate language
        for lang in self.intermediate_languages:
            try:
                # Translate to intermediate language
                translated = self._translate(text, "en", lang)
                
                # Translate back to English
                back_translated = self._translate(translated, lang, "en")
                
                # Add to augmented texts if different from original
                if back_translated != text:
                    augmented_texts.append(back_translated)
            except Exception as e:
                logger.error(f"Back-translation error with {lang}: {e}")
        
        return augmented_texts
