import unicodedata
import re
from typing import Dict, List

# Use langdetect instead of polyglot
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class MultilingualHandler:
    def __init__(self):
        """
        Handles multilingual PDFs with Unicode-aware parsing
        Supports major languages: English, Japanese, German, Hindi, etc.
        Updated to use langdetect instead of polyglot
        """
        # Language-specific heading keywords
        self.heading_keywords = {
            'en': ['chapter', 'section', 'abstract', 'introduction', 'conclusion', 'appendix'],
            'ja': ['章', '節', '要約', 'はじめに', '結論', '付録'],  # Japanese
            'de': ['kapitel', 'abschnitt', 'zusammenfassung', 'einführung', 'schlussfolgerung', 'anhang'],  # German
            'hi': ['अध्याय', 'खंड', 'सार', 'परिचय', 'निष्कर्ष', 'परिशिष्ट'],  # Hindi
            'zh': ['章', '节', '摘要', '引言', '结论', '附录'],  # Chinese
            'fr': ['chapitre', 'section', 'résumé', 'introduction', 'conclusion', 'annexe'],  # French
            'es': ['capítulo', 'sección', 'resumen', 'introducción', 'conclusión', 'apéndice'],  # Spanish
            'ru': ['глава', 'раздел', 'резюме', 'введение', 'заключение', 'приложение'],  # Russian
            'ar': ['فصل', 'قسم', 'ملخص', 'مقدمة', 'خاتمة', 'ملحق'],  # Arabic
        }

        # Script detection patterns
        self.script_patterns = {
            'latin': re.compile(r'[A-Za-z]'),
            'cyrillic': re.compile(r'[\u0400-\u04FF]'),
            'arabic': re.compile(r'[\u0600-\u06FF]'),
            'chinese': re.compile(r'[\u4E00-\u9FFF]'),
            'japanese_hiragana': re.compile(r'[\u3040-\u309F]'),
            'japanese_katakana': re.compile(r'[\u30A0-\u30FF]'),
            'devanagari': re.compile(r'[\u0900-\u097F]'),  # Hindi
            'hangul': re.compile(r'[\uAC00-\uD7AF]'),  # Korean
        }

        # Language confidence mapping
        self.language_confidence = {}

    def detect_language(self, text: str) -> str:
        """
        Detect language of text using langdetect or fallback methods
        Fast heuristic approach for heading detection
        """
        if not text or len(text.strip()) < 3:
            return 'en'

        text_clean = text.lower().strip()

        # First try langdetect if available
        if LANGDETECT_AVAILABLE:
            try:
                detected_lang = detect(text_clean)
                # Map some language codes to our standard codes
                lang_mapping = {
                    'zh-cn': 'zh',
                    'zh-tw': 'zh',
                    'pt': 'en',  # Default Portuguese to English for simplicity
                    'it': 'en',  # Default Italian to English
                    'nl': 'en',  # Default Dutch to English
                }
                return lang_mapping.get(detected_lang, detected_lang)
            except (LangDetectException, Exception):
                pass

        # Fallback: Check for language-specific keywords
        for lang, keywords in self.heading_keywords.items():
            for keyword in keywords:
                if keyword in text_clean:
                    return lang

        # Fallback: Script detection
        script = self.detect_script(text)
        script_to_lang = {
            'chinese': 'zh',
            'japanese_hiragana': 'ja',
            'japanese_katakana': 'ja',
            'devanagari': 'hi',
            'cyrillic': 'ru',
            'arabic': 'ar',
            'hangul': 'ko'
        }

        detected_lang = script_to_lang.get(script, 'en')
        return detected_lang

    def detect_script(self, text: str) -> str:
        """
        Detect writing script using Unicode ranges
        """
        if not text:
            return 'latin'

        # Count characters for each script
        script_counts = {}
        for script, pattern in self.script_patterns.items():
            matches = pattern.findall(text)
            script_counts[script] = len(matches)

        # Return script with highest count
        if script_counts:
            dominant_script = max(script_counts, key=script_counts.get)
            if script_counts[dominant_script] > 0:
                return dominant_script

        return 'latin'  # Default

    def get_heading_language_score(self, element: Dict) -> float:
        """
        Calculate language-specific heading likelihood score
        """
        text = element["text"].lower()
        language = element.get("language", "en")

        # Check for language-specific heading indicators
        keywords = self.heading_keywords.get(language, [])
        for keyword in keywords:
            if keyword in text:
                return 1.0

        # Check for numbered patterns (universal across languages)
        if re.match(r'^\d+[.)]\s*', text):
            return 0.8

        # Check for Roman numerals (common in many languages)
        if re.match(r'^[IVX]+[.)]\s*', text, re.IGNORECASE):
            return 0.7

        # Check for alphabetic outline patterns (A., B., etc.)
        if re.match(r'^[A-Z][.)]\s*', text):
            return 0.6

        # Check for all-caps headings in Latin scripts
        if language in ['en', 'de', 'fr', 'es'] and text.isupper() and 3 <= len(text) <= 50:
            return 0.5

        return 0.0

    def normalize_unicode_text(self, text: str) -> str:
        """
        Normalize Unicode text for consistent processing
        Handles different Unicode forms and removes combining characters
        """
        if not text:
            return ""

        try:
            # Normalize to NFC form (canonical composition)
            normalized = unicodedata.normalize('NFC', text)

            # Remove zero-width characters that can break processing
            zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u00ad']
            for char in zero_width_chars:
                normalized = normalized.replace(char, '')

            # Remove excessive whitespace
            normalized = re.sub(r'\s+', ' ', normalized)

            return normalized.strip()
        except Exception:
            # Fallback: return original text if normalization fails
            return text.strip()

    def is_rtl_language(self, language: str) -> bool:
        """Check if language is right-to-left"""
        rtl_languages = ['ar', 'he', 'fa', 'ur']
        return language in rtl_languages

    def get_language_specific_patterns(self, language: str) -> List[str]:
        """Get language-specific heading patterns"""
        patterns = {
            'ja': [r'^第\d+章', r'^第\d+節'],  # Chapter/Section in Japanese
            'zh': [r'^第\d+章', r'^第\d+节'],  # Chapter/Section in Chinese
            'ar': [r'^الفصل\s+\d+', r'^القسم\s+\d+'],  # Chapter/Section in Arabic
            'de': [r'^Kapitel\s+\d+', r'^Abschnitt\s+\d+'],  # German
            'fr': [r'^Chapitre\s+\d+', r'^Section\s+\d+'],  # French
            'es': [r'^Capítulo\s+\d+', r'^Sección\s+\d+'],  # Spanish
        }
        return patterns.get(language, [])
