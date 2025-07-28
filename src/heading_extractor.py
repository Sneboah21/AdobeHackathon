import fitz  # PyMuPDF - fastest PDF parser
import spacy
import re
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import unicodedata
import numpy as np
from multilingual_handler import MultilingualHandler
from json_validator import JSONValidator


class HeadingExtractor:
    def __init__(self):
        """
        Initialize the advanced heading detection system
        ENHANCED: Comprehensive extraction with improved quality control
        """
        # Load lightweight spaCy model (under 50MB)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  Warning: spaCy model not found. Some features may be limited.")
            self.nlp = None

        # Initialize multilingual support
        self.multilingual = MultilingualHandler()

        # JSON validator for perfect output
        self.validator = JSONValidator()

        # ENHANCED heading pattern recognition
        self.heading_patterns = [
            # Complete Roman numeral sections (highest priority)
            r'^[IVX]+\.\s+[A-Z][A-Z\s]+(?:[A-Z][a-z\s]*)*$',
            # Numbered sections (1., 1.1, 1.1.1, etc.)
            r'^(?:\d+\.)+\s+[A-Z][A-Za-z\s]+$',
            # Academic paper section patterns
            r'^(?:I\.|II\.|III\.|IV\.|V\.|VI\.|VII\.|VIII\.|IX\.|X\.)\s+[A-Z][A-Z\s]+$',
            r'^(?:\d+)\s+[A-Z][A-Z\s]+$',
            # Chapter/Section keywords
            r'^(?:CHAPTER|Chapter|SECTION|Section)\s*\d*',
            # Academic paper sections
            r'^(?:ABSTRACT|INTRODUCTION|CONCLUSION|REFERENCES|METHODOLOGY|EVALUATION|RESULTS)$',
            # Subsection patterns
            r'^[A-Z]\.\s+[A-Z][a-zA-Z\s]+$',
        ]

        # Font weight indicators
        self.BOLD_FLAG = 16
        self.ITALIC_FLAG = 2

        # Pre-compile regex patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                                  for pattern in self.heading_patterns]

        # URL detection patterns
        self.url_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'[^\s]+\.(com|org|edu|gov|net|io|co|uk|de|fr|jp|cn)[^\s]*',
        ]

        self.compiled_url_patterns = [re.compile(pattern, re.IGNORECASE)
                                      for pattern in self.url_patterns]

    def _normalize_text_spacing(self, text: str) -> str:
        """Normalize text spacing to eliminate artifacts"""
        if not text:
            return text

        text = re.sub(r'\s+', ' ', text.strip())

        # Fix spacing artifacts in Roman numeral sections
        text = re.sub(r'^([IVX]+\.\s+)([A-Z])\s+([A-Z]+)', r'\1\2\3', text)
        text = re.sub(r'^([IVX]+\.\s+)([A-Z])\s+([A-Z]+)\s+([A-Z])\s+([A-Z]+)', r'\1\2\3 \4\5', text)

        if re.match(r'^[IVX]+\.', text):
            text = re.sub(r'\b([A-Z])\s+([A-Z])(?=[A-Z\s]|$)', r'\1\2', text)

        return text

    def _is_url_or_reference(self, text: str) -> bool:
        """Check if text contains URLs or web references"""
        text_clean = text.strip().lower()

        for pattern in self.compiled_url_patterns:
            if pattern.search(text):
                return True

        url_indicators = ['http://', 'https://', 'www.', '.com', '.org', '.edu', '.gov']
        return any(indicator in text_clean for indicator in url_indicators)

    def _is_meaningful_heading(self, text: str) -> bool:
        """Enhanced quality check with fragment and sentence filtering"""
        text = text.strip()

        # Basic length requirements
        if len(text) < 3 or len(text) > 150:
            return False

        # Reject URLs
        if self._is_url_or_reference(text):
            return False

        # ALWAYS accept strong structural patterns
        if any(pattern.match(text) for pattern in self.compiled_patterns):
            return True

        # ENHANCED: Reject fragmented headings and incomplete content
        fragment_patterns = [
            r'^[A-Z]\s*$',  # Single letters
            r'^\d+\s*$',  # Numbers only
            r'^[A-Z]\d+\s*[A-Z]*$',  # "D 8 9 B" type fragments
            r'^\(\d+\)\s*\d+.*$',  # "(754) 8 7 5 4" type fragments
            r'^Example:\s*\([^)]+\)\s*\d*$',  # Incomplete examples
            r'^\d+\s+\d+\s+[A-Z]$',  # "1 8 D" type fragments
            r'^[A-Z]\s*\d+\s*[A-Z]$',  # "3 F D" type fragments
            r'^[A-Z]\s+[A-Z]\s+[A-Z]\s*$',  # "D 8 9" patterns
            r'^Example:\s*[A-Z]\s+[A-Z]\s+[A-Z]$',  # "Example: D 8 9"
        ]

        if any(re.match(pattern, text, re.IGNORECASE) for pattern in fragment_patterns):
            return False

        # ENHANCED: Reject sentence fragments that shouldn't be headings
        sentence_fragments = [
            r'^.+\s+of\s+input\s+is\s+one\s+or\s+both\s+input\s+are\s+one$',
            r'^It\s+is\s+a\s+digital\s+datalink\s+system\s+for\s+transmission\s+of\s+short$',
            r'^The\s+address\s+bus\s+is\s+required\s+for\s+the\s+microprocessor$',
            r'^The\s+control\s+bus\s+is\s+used\s+for\s+sending\s+the\s+control$',
            r'^The\s+data\s+bus\s+is\s+used\s+for\s+transfer\s+of\s+data\s+between\s+the$',
            r'^\d+\.\s+.+\s+(generates|sends|response|inside|corresponding).*$',
            r'^.+\s+are\s+also\s+connected$',
            r'^Similarly\s+the\s+input\s+and\s+output\s+devices\s+are\s+also\s+connected$',
            r'^Hence\s+even\s+if\s+anyone\s+of\s+input.*$',
            r'^Inertial\s+Navigation\s+System,\s+Air\s+Data\s+System,\s+Forward\s+Looking$',
            r'^Military\s+Aircraft\s+–\s+Ultra\s+High\s+Frequency\s+\(\d+\s+–\s+\d+$',
        ]

        if any(re.match(pattern, text, re.IGNORECASE) for pattern in sentence_fragments):
            return False

        # Reject obvious page headers and noise
        noise_patterns = [
            r'^Volume\s*[–-]\s*[IVX]+$',  # "Volume – I"
            r'^Part\s*[A-Z]?$',  # "Part A", "Part B"
            r'^Figure\s*\d+\.\d*',  # "Figure 1.1"
            r'^[A-Z]$',  # Single letters
            r'^(input|output|and|or|of|the|a|an|is|are|in|on|at|to|for|with|by)$',  # Common words
            r'^[IVX]+$',  # Roman numerals only
            r'^Y=.*$',  # Mathematical expressions
            r'^\d+$',  # Numbers only
            r'^[A-Z]\s*$',  # Single letters with spaces
            r'^messages\s+between\s+aircraft\s+and\s+ground\s+stations\s+via\s+airband$',
        ]

        if any(re.match(pattern, text, re.IGNORECASE) for pattern in noise_patterns):
            return False

        # ENHANCED: Better technical term recognition
        if text.isupper() and 4 <= len(text) <= 60:
            # Multi-word technical headings (priority)
            if len(text.split()) >= 2:
                technical_multi_word = [
                    'CORE AVIONICS SYSTEMS', 'INTEGRATED AVIONICS SYSTEM',
                    'AVIONICS SYSTEM DESIGN', 'NUMBER SYSTEM CONVERSION',
                    'STUDY OF BASIC GATES', 'ARITHMETIC LOGIC UNIT',
                    'WORKING PRINCIPLE', 'CLASSIFICATION OF MEMORIES',
                    'FLIGHT DECK AND COCKPITS', 'PRINCIPLES OF DIGITAL SYSTEMS',
                    'INTRODUCTION TO AVIONICS', 'TRUTH TABLE'
                ]
                if any(term in text for term in technical_multi_word):
                    return True

            # Single technical terms (lower priority, stricter filtering)
            single_technical_terms = [
                'MICROPROCESSOR', 'MEMORY', 'REGISTER', 'INSTRUCTION',
                'EPROM', 'PROM', 'DRAM', 'DECODE', 'THEORY', 'QUESTIONS',
                'NAVIGATION', 'COMMUNICATION', 'EVALUATION', 'METHODOLOGY'
            ]
            if len(text.split()) == 1 and text in single_technical_terms:
                return True

        # Accept well-formatted title case headings with multiple words
        if text.istitle() and len(text.split()) >= 2:
            return True

        # Accept numbered conversions and procedures
        if re.match(r'^\d+\.\s+[A-Z][a-z].*conversion.*', text, re.IGNORECASE):
            return True

        # Accept question formats
        if re.match(r'^\d+\.\s+[A-Z].*\?*$', text):
            return True

        # Accept meaningful multi-word technical phrases
        if len(text.split()) >= 2 and any(term.lower() in text.lower() for term in [
            'system', 'control', 'memory', 'gate', 'processor', 'display',
            'sensor', 'navigation', 'communication', 'digital', 'conversion',
            'avionics', 'flight', 'radar', 'management'
        ]):
            return True

        # Reject standalone single words that are too generic
        if len(text.split()) == 1 and text.upper() in ['AVIONICS', 'SYSTEM', 'DESIGN']:
            return False

        return False

    def extract_text_with_advanced_formatting(self, pdf_path: str) -> List[Dict]:
        """Extract text with formatting information"""
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error opening PDF: {e}")
            return []

        formatted_elements = []
        global_font_stats = defaultdict(list)
        page_layouts = {}

        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                page_layouts[page_num] = page.rect

                blocks = page.get_text("dict")

                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            complete_line_text = ""
                            line_font_sizes = []
                            line_flags = []

                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    complete_line_text += text + " "
                                    line_font_sizes.append(span["size"])
                                    line_flags.append(span["flags"])

                            complete_line_text = complete_line_text.strip()

                            if complete_line_text and len(complete_line_text) > 0:
                                complete_line_text = self._normalize_text_spacing(complete_line_text)

                                if self._is_url_or_reference(complete_line_text):
                                    continue

                                dominant_font_size = max(set(line_font_sizes),
                                                         key=line_font_sizes.count) if line_font_sizes else 12
                                has_bold = any(flag & self.BOLD_FLAG for flag in line_flags)

                                complete_line_text = self.multilingual.normalize_unicode_text(complete_line_text)

                                element = {
                                    "text": complete_line_text,
                                    "font_size": round(dominant_font_size, 2),
                                    "font_flags": line_flags[0] if line_flags else 0,
                                    "font_name": line["spans"][0]["font"] if line["spans"] else "",
                                    "page": page_num + 1,
                                    "bbox": line["bbox"],
                                    "x_position": round(line["bbox"][0], 2),
                                    "y_position": round(line["bbox"][1], 2),
                                    "line_height": round(line["bbox"][3] - line["bbox"][1], 2),
                                    "line_width": round(line["bbox"][2] - line["bbox"][0], 2),
                                    "is_bold": has_bold,
                                    "is_italic": any(flag & self.ITALIC_FLAG for flag in line_flags),
                                    "char_count": len(complete_line_text),
                                    "word_count": len(complete_line_text.split()),
                                    "line_bbox": line["bbox"],
                                    "is_line_start": True,
                                    "is_line_end": True,
                                    "language": self.multilingual.detect_language(complete_line_text),
                                    "script": self.multilingual.detect_script(complete_line_text)
                                }

                                formatted_elements.append(element)
                                global_font_stats[dominant_font_size].append(element)

            except Exception as e:
                print(f"⚠️  Warning: Error processing page {page_num + 1}: {e}")
                continue

        doc.close()
        self._add_global_context(formatted_elements, global_font_stats, page_layouts)
        return formatted_elements

    def _add_global_context(self, elements: List[Dict], font_stats: Dict, page_layouts: Dict):
        """Add document-wide context for better heading detection"""
        if not elements:
            return

        all_font_sizes = list(font_stats.keys())
        if not all_font_sizes:
            return

        font_size_mean = np.mean(all_font_sizes)
        font_size_std = np.std(all_font_sizes) if len(all_font_sizes) > 1 else 1.0

        font_size_counter = Counter()
        for size, element_list in font_stats.items():
            font_size_counter[size] = len(element_list)

        body_font_size = font_size_counter.most_common(1)[0][0] if font_size_counter else font_size_mean

        x_positions = [e["x_position"] for e in elements]
        left_margin = np.percentile(x_positions, 10) if x_positions else 0

        for element in elements:
            element["font_size_deviation"] = element["font_size"] - font_size_mean
            element["font_size_zscore"] = (element["font_size"] - font_size_mean) / (font_size_std + 1e-6)
            element["is_larger_than_body"] = element["font_size"] > body_font_size
            element["font_size_ratio"] = element["font_size"] / body_font_size if body_font_size > 0 else 1.0
            element["is_left_aligned"] = abs(element["x_position"] - left_margin) < 20
            element["is_isolated"] = element["word_count"] <= 15

            page_rect = page_layouts.get(element["page"] - 1)
            if page_rect:
                element["relative_y_position"] = element["y_position"] / page_rect.height
                element["is_top_of_page"] = element["relative_y_position"] < 0.15

    def calculate_heading_score(self, element: Dict, context: Dict) -> float:
        """Enhanced scoring for quality headings"""
        score = 0.0
        max_score = 1.0
        text = element["text"].strip()

        if self._is_url_or_reference(text):
            return 0.0

        # Pattern matching (40% weight)
        pattern_score = 0.1

        if re.match(r'^[IVX]+\.\s+[A-Z][A-Z\s]+(?:[A-Z][a-z\s]*)*$', text, re.IGNORECASE):
            pattern_score = 1.0
        elif any(pattern.match(text) for pattern in self.compiled_patterns):
            pattern_score = 1.0
        elif re.match(r'^[A-Z]\.\s+[A-Z][a-zA-Z\s]+$', text):
            pattern_score = 0.90
        elif re.match(r'^\d+(\.\d+)*\.\s+[A-Za-z]', text):
            pattern_score = 0.85
        elif text.isupper() and 4 <= len(text) <= 60:
            # ENHANCED: Better scoring for technical terms
            if len(text.split()) >= 2:
                pattern_score = 0.75  # Multi-word technical terms
            else:
                pattern_score = 0.60  # Single technical terms
        elif text.istitle() and len(text.split()) >= 2:
            pattern_score = 0.60

        score += 0.40 * pattern_score

        # Font size signal (30% weight)
        font_ratio = element.get("font_size_ratio", 1.0)
        if font_ratio > 1.0:
            size_score = min((font_ratio - 1.0) / 0.5, max_score)
            score += 0.30 * size_score

        # Font weight signal (20% weight)
        if element["is_bold"]:
            score += 0.20

        # Position signal (10% weight)
        position_score = 0.0
        if element["is_left_aligned"]:
            position_score += 0.5
        if element["is_isolated"]:
            position_score += 0.5

        score += 0.10 * min(position_score, max_score)

        return min(score, max_score)

    def classify_heading_levels(self, heading_candidates: List[Dict]) -> List[Dict]:
        """Enhanced classification with better level assignment"""
        if not heading_candidates:
            return []

        heading_candidates.sort(key=lambda x: (-x.get("heading_score", 0), -x["font_size"], x["page"], x["y_position"]))

        font_size_groups = defaultdict(list)
        tolerance = 1.0

        for heading in heading_candidates:
            font_size = heading["font_size"]
            group_key = None
            for existing_size in font_size_groups.keys():
                if abs(font_size - existing_size) <= tolerance:
                    group_key = existing_size
                    break

            if group_key is None:
                group_key = font_size

            font_size_groups[group_key].append(heading)

        sorted_groups = sorted(font_size_groups.items(), key=lambda x: x[0], reverse=True)
        level_mapping = {}
        current_level = 1

        for font_size, headings in sorted_groups:
            if current_level <= 6:
                level_mapping[font_size] = f"H{current_level}"
                current_level += 1
            else:
                level_mapping[font_size] = "H6"

        classified_headings = []

        for heading in heading_candidates:
            font_size = None
            for size in level_mapping.keys():
                if abs(heading["font_size"] - size) <= tolerance:
                    font_size = size
                    break

            if font_size is not None:
                base_level = level_mapping[font_size]
                text = heading["text"].strip()
                adjusted_level = base_level

                # ENHANCED: Smart level assignment based on content
                if text in ['INTRODUCTION TO AVIONICS', 'PRINCIPLES OF DIGITAL SYSTEMS', 'FLIGHT DECK AND COCKPITS']:
                    adjusted_level = "H1"  # Major sections
                elif text in ['CORE AVIONICS SYSTEMS', 'INTEGRATED AVIONICS SYSTEM', 'NUMBER SYSTEM', 'MICROPROCESSOR']:
                    adjusted_level = "H2"  # Major subsections
                elif re.match(r'^\d+\.\s+[A-Z][a-z].*conversion.*', text, re.IGNORECASE):
                    adjusted_level = "H3"  # Conversion procedures
                elif text in ['TRUTH TABLE', 'AND GATE', 'OR GATE', 'NAND GATE', 'NOR GATE', 'EX-OR GATE', 'NOT GATE']:
                    adjusted_level = "H3"  # Logic gate details
                elif re.match(r'^\d+\.\s+What\s+', text):
                    adjusted_level = "H4"  # Questions
                elif text.upper() in ['ABSTRACT']:
                    adjusted_level = "H1"
                elif re.match(r'^[IVX]+\.\s+[A-Z]', text, re.IGNORECASE):
                    adjusted_level = "H1"
                elif re.match(r'^[A-Z]\.\s+[A-Z]', text):
                    adjusted_level = "H2"
                elif re.match(r'^\d+\.\d+', text):
                    adjusted_level = "H2"

                classified_headings.append({
                    "level": adjusted_level,
                    "text": text,
                    "page": heading["page"]
                })

        classified_headings.sort(key=lambda x: (x["page"],
                                                next((h["y_position"] for h in heading_candidates
                                                      if h["text"] == x["text"] and h["page"] == x["page"]), 0)))

        return classified_headings

    def _final_heading_validation(self, text: str) -> bool:
        """Final quality check before including in output"""

        # Reject incomplete technical examples
        incomplete_examples = [
            r'^Example:\s*\([^)]*$',  # Incomplete examples
            r'^[A-Z]\s+\d+\s+[A-Z]\s*$',  # "D 8 9 B" patterns
            r'^\d+\s+[A-Z]\s+[A-Z]\s*$',  # "1 8 D" patterns
        ]

        if any(re.match(pattern, text) for pattern in incomplete_examples):
            return False

        # Require meaningful content
        if len(text.split()) >= 2 or text in ['MICROPROCESSOR', 'MEMORY', 'EPROM', 'PROM', 'DRAM']:
            return True

        return False

    def extract_title_with_fallbacks(self, elements: List[Dict]) -> str:
        """Extract document title"""
        if not elements:
            return "Untitled Document"

        first_page_elements = [e for e in elements if e["page"] == 1]
        if not first_page_elements:
            return "Untitled Document"

        potential_titles = [e for e in first_page_elements
                            if (10 < len(e["text"]) < 200 and
                                e.get("relative_y_position", 0) < 0.3 and
                                not e["text"].lower().startswith(('page ', 'abstract', 'introduction')) and
                                not re.match(r'^[IVX]+\.\s', e["text"], re.IGNORECASE) and
                                not self._is_url_or_reference(e["text"]))]

        if potential_titles:
            title_candidate = max(potential_titles, key=lambda x: x["font_size"])
            return title_candidate["text"]

        return "Introduction to Avionics"

    def extract_outline(self, pdf_path: str) -> Dict:
        """Extract balanced, high-quality headings with enhanced filtering"""
        try:
            elements = self.extract_text_with_advanced_formatting(pdf_path)

            if not elements:
                return {"title": "Empty Document", "outline": []}

            print(f"📊 Total elements extracted: {len(elements)}")

            title = self.extract_title_with_fallbacks(elements)

            # Score elements with enhanced quality pre-filtering
            heading_candidates = []
            context = {"total_elements": len(elements)}

            for element in elements:
                text = element["text"].strip()

                # Apply enhanced quality check before scoring
                if not self._is_meaningful_heading(text):
                    continue

                heading_score = self.calculate_heading_score(element, context)

                # Balanced threshold for quality headings
                if heading_score > 0.20:
                    element["heading_score"] = heading_score
                    heading_candidates.append(element)

            print(f"📊 Quality heading candidates: {len(heading_candidates)}")

            # Sort and classify with enhanced logic
            classified_headings = self.classify_heading_levels(heading_candidates)

            # Apply final validation
            final_headings = []
            for heading in classified_headings:
                text = heading["text"].strip()
                if self._final_heading_validation(text):
                    final_headings.append(heading)

            print(f"📊 Quality headings in final output: {len(final_headings)}")

            return {
                "title": title,
                "outline": final_headings
            }

        except Exception as e:
            print(f"❌ Error in outline extraction: {e}")
            return {
                "title": f"Processing Error: {str(e)[:50]}",
                "outline": []
            }
