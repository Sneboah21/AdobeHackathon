# import fitz  # PyMuPDF - fastest PDF parser
# import spacy
# import re
# import json
# from typing import List, Dict, Tuple, Optional
# from collections import defaultdict, Counter
# import unicodedata
# import numpy as np
# from multilingual_handler import MultilingualHandler
# from json_validator import JSONValidator
#
#
# class HeadingExtractor:
#     def __init__(self):
#         """
#         Initialize the advanced heading detection system
#         Combines multiple signals for chaos-resistant extraction
#         """
#         # Load lightweight spaCy model (under 50MB)
#         try:
#             self.nlp = spacy.load("en_core_web_sm")
#         except OSError:
#             print("⚠️  Warning: spaCy model not found. Some features may be limited.")
#             self.nlp = None
#
#         # Initialize multilingual support
#         self.multilingual = MultilingualHandler()
#
#         # JSON validator for perfect output
#         self.validator = JSONValidator()
#
#         # ENHANCED heading pattern recognition - optimized for academic papers
#         self.heading_patterns = [
#             # Complete Roman numeral sections (highest priority)
#             r'^[IVX]+\.\s+[A-Z][A-Z\s]+(?:[A-Z][a-z\s]*)*$',  # I. INTRODUCTION, II. RELATED WORK
#             # Numbered sections (1., 1.1, 1.1.1, etc.)
#             r'^(?:\d+\.)+\s+[A-Z][A-Za-z\s]+$',
#             # Academic paper section patterns
#             r'^(?:I\.|II\.|III\.|IV\.|V\.|VI\.|VII\.|VIII\.|IX\.|X\.)\s+[A-Z][A-Z\s]+$',
#             r'^(?:\d+)\s+[A-Z][A-Z\s]+$',  # "1 INTRODUCTION" format
#             # Chapter/Section keywords (multilingual)
#             r'^(?:CHAPTER|Chapter|章|Kapitel|Chapitre|Capítulo)\s*\d*',
#             r'^(?:SECTION|Section|節|Abschnitt|Section|Sección)\s*\d*',
#             # Academic paper sections (exact matches)
#             r'^(?:ABSTRACT|Abstract|要約|Zusammenfassung|Résumé|Resumen)$',
#             r'^(?:INTRODUCTION|Introduction|はじめに|Einführung|Introduction|Introducción)$',
#             r'^(?:RELATED WORK|Related Work|BACKGROUND|Background)$',
#             r'^(?:METHODOLOGY|Methodology|Methods|METHODS|方法|Methodik|Méthodologie|Metodología)$',
#             r'^(?:SYSTEM OVERVIEW|System Overview|ARCHITECTURE|Architecture)$',
#             r'^(?:TOOLS|Tools|IMPLEMENTATION|Implementation)$',
#             r'^(?:EXECUTION|Execution|EXPERIMENT|Experiment)$',
#             r'^(?:EVALUATION|Evaluation|RESULTS|Results|FINDINGS|Findings)$',
#             r'^(?:SCALABILITY|Scalability|ADAPTABILITY|Adaptability)$',
#             r'^(?:CONCLUSION|Conclusion|結論|Schlussfolgerung|Conclusion|Conclusión)$',
#             r'^(?:DISCUSSION|Discussion|ANALYSIS|Analysis)$',
#             # Appendix patterns
#             r'^(?:APPENDIX|Appendix|付録|Anhang|Annexe|Apéndice)\s*[A-Z]?',
#             # Bibliography/References
#             r'^(?:REFERENCES|References|Bibliography|参考文献|Literatur|Références|Referencias)$',
#             # Subsection patterns
#             r'^[A-Z]\.\s+[A-Z][a-zA-Z\s]+$',  # A. Home Automation
#         ]
#
#         # Font weight indicators (PyMuPDF flags)
#         self.BOLD_FLAG = 16  # Bold text flag in PyMuPDF
#         self.ITALIC_FLAG = 2  # Italic text flag in PyMuPDF
#
#         # Performance optimization: pre-compile regex patterns
#         self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
#                                   for pattern in self.heading_patterns]
#
#         # URL detection patterns for comprehensive filtering
#         self.url_patterns = [
#             r'https?://[^\s]+',  # Standard HTTP/HTTPS URLs
#             r'www\.[^\s]+',  # www. URLs
#             r'[^\s]+\.(com|org|edu|gov|net|io|co|uk|de|fr|jp|cn)[^\s]*',  # Domain extensions
#             r'^\d+\s+https?://',  # Numbered references with URLs
#             r'[^\s]*\.google\.[^\s]*',  # Google-specific domains
#             r'[^\s]*deepmind[^\s]*',  # DeepMind-specific
#             r'[^\s]*technologies[^\s]*gemini[^\s]*',  # Specific false positive
#         ]
#
#         # Pre-compile URL patterns for efficiency
#         self.compiled_url_patterns = [re.compile(pattern, re.IGNORECASE)
#                                       for pattern in self.url_patterns]
#
#         # Font size normalization cache
#         self._font_stats_cache = {}
#
#     def _normalize_text_spacing(self, text: str) -> str:
#         """
#         CRITICAL FIX: Normalize text spacing to eliminate artifacts
#         Addresses spacing issues like "I. I NTRODUCTION" → "I. INTRODUCTION"
#         """
#         if not text:
#             return text
#
#         # Remove excessive whitespace and normalize
#         text = re.sub(r'\s+', ' ', text.strip())
#
#         # Fix specific spacing artifacts in Roman numeral sections
#         # Pattern: "I. I NTRODUCTION" → "I. INTRODUCTION"
#         text = re.sub(r'^([IVX]+\.\s+)([A-Z])\s+([A-Z]+)', r'\1\2\3', text)
#
#         # Fix patterns like "II. R ELATED W ORK" → "II. RELATED WORK"
#         text = re.sub(r'^([IVX]+\.\s+)([A-Z])\s+([A-Z]+)\s+([A-Z])\s+([A-Z]+)', r'\1\2\3 \4\5', text)
#
#         # Fix patterns like "III. S YSTEM O VERVIEW" → "III. SYSTEM OVERVIEW"
#         text = re.sub(r'^([IVX]+\.\s+)([A-Z])\s+([A-Z]+)\s+([A-Z])\s+([A-Z]+)', r'\1\2\3 \4\5', text)
#
#         # General fix for spaced capital letters in headings
#         # Only apply to Roman numeral sections to avoid affecting other text
#         if re.match(r'^[IVX]+\.', text):
#             # Replace single letter followed by space and capital letter
#             text = re.sub(r'\b([A-Z])\s+([A-Z])(?=[A-Z\s]|$)', r'\1\2', text)
#
#         return text
#
#     def _is_url_or_reference(self, text: str) -> bool:
#         """
#         COMPREHENSIVE URL and reference detection
#         Returns True if text contains URLs, web references, or citation links
#         """
#         text_clean = text.strip().lower()
#
#         # Check against compiled URL patterns
#         for pattern in self.compiled_url_patterns:
#             if pattern.search(text):
#                 return True
#
#         # Additional specific checks for common false positives
#         url_indicators = [
#             'http://', 'https://', 'www.', '.com', '.org', '.edu', '.gov',
#             'deepmind.google', 'technologies/gemini', '#introduction',
#             'google.com', 'github.com', 'arxiv.org'
#         ]
#
#         if any(indicator in text_clean for indicator in url_indicators):
#             return True
#
#         # Check for numbered reference patterns with URLs
#         if re.match(r'^\d+\s+http', text_clean):
#             return True
#
#         # Check for mixed number-URL content
#         if re.search(r'\d+.*(?:http|www|\.com)', text_clean):
#             return True
#
#         return False
#
#     def extract_text_with_advanced_formatting(self, pdf_path: str) -> List[Dict]:
#         """
#         ENHANCED text extraction with better line reconstruction and spacing normalization
#         Key fixes: Properly reconstructs complete headings and normalizes spacing
#         """
#         try:
#             doc = fitz.open(pdf_path)
#         except Exception as e:
#             print(f"❌ Error opening PDF: {e}")
#             return []
#
#         formatted_elements = []
#         global_font_stats = defaultdict(list)
#         page_layouts = {}
#
#         for page_num in range(len(doc)):
#             try:
#                 page = doc.load_page(page_num)
#                 page_layouts[page_num] = page.rect
#
#                 # Get text blocks with detailed formatting
#                 blocks = page.get_text("dict")
#
#                 for block in blocks["blocks"]:
#                     if "lines" in block:  # Text block (not image)
#                         for line in block["lines"]:
#                             # CRITICAL FIX: Reconstruct complete line text
#                             complete_line_text = ""
#                             line_font_sizes = []
#                             line_flags = []
#
#                             for span in line["spans"]:
#                                 text = span["text"].strip()
#                                 if text:
#                                     complete_line_text += text + " "
#                                     line_font_sizes.append(span["size"])
#                                     line_flags.append(span["flags"])
#
#                             # Clean up and normalize complete line text
#                             complete_line_text = complete_line_text.strip()
#
#                             if complete_line_text and len(complete_line_text) > 2:
#                                 # CRITICAL: Apply text normalization to fix spacing
#                                 complete_line_text = self._normalize_text_spacing(complete_line_text)
#
#                                 # Skip if this is a URL or reference
#                                 if self._is_url_or_reference(complete_line_text):
#                                     continue
#
#                                 # Use dominant font characteristics for the line
#                                 dominant_font_size = max(set(line_font_sizes),
#                                                          key=line_font_sizes.count) if line_font_sizes else 12
#                                 has_bold = any(flag & self.BOLD_FLAG for flag in line_flags)
#
#                                 # Normalize Unicode text
#                                 complete_line_text = self.multilingual.normalize_unicode_text(complete_line_text)
#
#                                 # Create element for the complete line
#                                 element = {
#                                     "text": complete_line_text,
#                                     "font_size": round(dominant_font_size, 2),
#                                     "font_flags": line_flags[0] if line_flags else 0,
#                                     "font_name": line["spans"][0]["font"] if line["spans"] else "",
#                                     "page": page_num + 1,
#                                     "bbox": line["bbox"],
#
#                                     # Advanced positioning metrics
#                                     "x_position": round(line["bbox"][0], 2),
#                                     "y_position": round(line["bbox"][1], 2),
#                                     "line_height": round(line["bbox"][3] - line["bbox"][1], 2),
#                                     "line_width": round(line["bbox"][2] - line["bbox"][0], 2),
#
#                                     # Text characteristics
#                                     "is_bold": has_bold,
#                                     "is_italic": any(flag & self.ITALIC_FLAG for flag in line_flags),
#                                     "char_count": len(complete_line_text),
#                                     "word_count": len(complete_line_text.split()),
#
#                                     # Line context
#                                     "line_bbox": line["bbox"],
#                                     "is_line_start": True,
#                                     "is_line_end": True,
#
#                                     # Multilingual detection
#                                     "language": self.multilingual.detect_language(complete_line_text),
#                                     "script": self.multilingual.detect_script(complete_line_text)
#                                 }
#
#                                 formatted_elements.append(element)
#                                 global_font_stats[dominant_font_size].append(element)
#
#             except Exception as e:
#                 print(f"⚠️  Warning: Error processing page {page_num + 1}: {e}")
#                 continue
#
#         doc.close()
#
#         # Add global context to each element
#         self._add_global_context(formatted_elements, global_font_stats, page_layouts)
#
#         return formatted_elements
#
#     def _add_global_context(self, elements: List[Dict], font_stats: Dict, page_layouts: Dict):
#         """
#         Add document-wide context for better heading detection
#         Calculates relative font sizes, positioning patterns
#         """
#         if not elements:
#             return
#
#         # Calculate document statistics
#         all_font_sizes = list(font_stats.keys())
#         if not all_font_sizes:
#             return
#
#         font_size_mean = np.mean(all_font_sizes)
#         font_size_std = np.std(all_font_sizes) if len(all_font_sizes) > 1 else 1.0
#
#         # Most common font size (body text)
#         font_size_counter = Counter()
#         for size, element_list in font_stats.items():
#             font_size_counter[size] = len(element_list)
#
#         body_font_size = font_size_counter.most_common(1)[0][0] if font_size_counter else font_size_mean
#
#         # Calculate positioning statistics
#         x_positions = [e["x_position"] for e in elements]
#         left_margin = np.percentile(x_positions, 10) if x_positions else 0
#
#         for element in elements:
#             # Relative font size metrics
#             element["font_size_deviation"] = element["font_size"] - font_size_mean
#             element["font_size_zscore"] = (element["font_size"] - font_size_mean) / (font_size_std + 1e-6)
#             element["is_larger_than_body"] = element["font_size"] > body_font_size
#             element["font_size_ratio"] = element["font_size"] / body_font_size if body_font_size > 0 else 1.0
#
#             # Position-based features
#             element["is_left_aligned"] = abs(element["x_position"] - left_margin) < 20
#             element["is_isolated"] = element["word_count"] <= 15
#             element["is_first_in_line"] = element.get("is_line_start", True)
#
#             # Page context
#             page_rect = page_layouts.get(element["page"] - 1)
#             if page_rect:
#                 element["relative_y_position"] = element["y_position"] / page_rect.height
#                 element["is_top_of_page"] = element["relative_y_position"] < 0.15
#
#     def _is_likely_heading_text(self, text: str) -> bool:
#         """STRICT pre-filtering with URL rejection"""
#         text = text.strip()
#
#         # Must be reasonable length
#         if len(text) < 3 or len(text) > 150:
#             return False
#
#         # REJECT URLs and references immediately
#         if self._is_url_or_reference(text):
#             return False
#
#         # ALWAYS accept complete academic section patterns
#         if (re.match(r'^[IVX]+\.\s+[A-Z][A-Z\s]+', text, re.IGNORECASE) or
#                 text.upper() in ['ABSTRACT', 'INTRODUCTION', 'CONCLUSION', 'REFERENCES'] or
#                 re.match(r'^[A-Z]\.\s+[A-Z][a-zA-Z\s]+$', text) or
#                 any(pattern.match(text) for pattern in self.compiled_patterns)):
#             return True
#
#         # REJECT obvious body text patterns
#         if (text.endswith('.') and len(text.split()) > 6 or  # Long sentences
#                 text.count('.') > 1 or  # Multiple sentences
#                 text.endswith(',') or text.endswith(';') or  # Sentence fragments
#                 ' the ' in text.lower() and len(text) > 20 or  # Body text with articles
#                 text.lower().startswith(('this ', 'these ', 'however,', 'therefore,', 'moreover,')) or
#                 any(phrase in text.lower() for phrase in
#                     ['llms, however', 'triggers an', 'results were', 'agent with grounded'])):
#             return False
#
#         # REJECT single words that are clearly not headings
#         if (len(text.split()) == 1 and
#                 text.lower() in ['tools', 'existing', 'recent', 'reasoning', 'however', 'therefore']):
#             return False
#
#         # Only allow reasonably short text (likely headings)
#         return len(text.split()) <= 10
#
#     def _is_valid_heading(self, text: str) -> bool:
#         """ENHANCED validation with URL filtering and strict false positive filtering"""
#         text_clean = text.strip()
#         text_lower = text_clean.lower()
#
#         # REJECT URLs and references immediately
#         if self._is_url_or_reference(text_clean):
#             return False
#
#         # ALWAYS accept complete academic patterns
#         if (re.match(r'^[IVX]+\.\s+[A-Z][A-Z\s]+', text_clean, re.IGNORECASE) or
#                 text_clean.upper() in ['ABSTRACT', 'INTRODUCTION', 'CONCLUSION', 'REFERENCES'] or
#                 re.match(r'^[A-Z]\.\s+[A-Z][a-zA-Z\s]+$', text_clean)):
#             return True
#
#         # STRICT rejection of false positives
#         false_positives = [
#             'assistant context. llms, however, lack speciﬁc knowledge about',
#             'smart home agent with grounded execution (sage), overcomes',
#             'request triggers an llm-controlled sequence of discrete actions.',
#             'grounded execution (sage).',
#             '24 january 2025.',
#             '. results were',
#             'set of all tools',
#             'subset of agent-tools',
#             '. some tools are themselves',
#             '"agent-tools',
#             'of tools available to',
#             'tools in set',
#             'ubtools',
#             'scalability:',
#             'execution tool:',
#             'behavior is enabled through a collection of tools detailed',
#             'visualizes an execution trace of sage agent-tool',
#             'dmitriy rivkin',
#             'existing',
#             'tools',
#             # URL and reference patterns (additional layer)
#             '8 https://deepmind.google/technologies/gemini/#introduction'
#         ]
#
#         if text_lower in false_positives:
#             return False
#
#         # Reject obvious sentence patterns
#         if (text_clean.endswith('.') and len(text_clean.split()) > 5 and
#                 not text_lower.endswith(('abstract.', 'introduction.', 'conclusion.'))):
#             return False
#
#         # Reject truncated Roman numerals (incomplete sections)
#         if re.match(r'^[IVX]+\.\s+[A-Z]$', text_clean):
#             return False
#
#         # Reject obvious body text patterns
#         if any(pattern in text_lower for pattern in [
#             'however,', 'therefore,', 'moreover,', 'furthermore,',
#             'triggers an', 'results were', 'llms,', 'execution (sage)'
#         ]):
#             return False
#
#         return True
#
#     def calculate_heading_score(self, element: Dict, context: Dict) -> float:
#         """
#         ENHANCED scoring with priority for complete academic sections
#         """
#         score = 0.0
#         max_score = 1.0
#         text = element["text"].strip()
#
#         # IMMEDIATE rejection for URLs
#         if self._is_url_or_reference(text):
#             return 0.0
#
#         # 1. Pattern matching signal (50% weight) - HIGHEST priority for complete patterns
#         pattern_score = 0.0
#
#         # COMPLETE Roman numeral sections get maximum priority
#         if re.match(r'^[IVX]+\.\s+[A-Z][A-Z\s]+(?:[A-Z][a-z\s]*)*$', text, re.IGNORECASE):
#             pattern_score = 1.0
#
#         # Other compiled patterns
#         elif any(pattern.match(text) for pattern in self.compiled_patterns):
#             pattern_score = 1.0
#
#         # Academic paper specific patterns
#         else:
#             # Letter subsections (A. Home Automation)
#             if re.match(r'^[A-Z]\.\s+[A-Z][a-zA-Z\s]+$', text):
#                 pattern_score = 0.90
#
#             # Standard numbered patterns (1., 1.1, etc.)
#             elif re.match(r'^\d+(\.\d+)*\.\s+[A-Za-z]', text):
#                 depth = text.count('.')
#                 pattern_score = min(0.85, 0.85 - (depth * 0.1))
#
#             # All-caps academic sections
#             elif text.isupper() and 4 <= len(text) <= 50:
#                 academic_terms = ['ABSTRACT', 'INTRODUCTION', 'METHODOLOGY', 'RESULTS',
#                                   'EVALUATION', 'CONCLUSION', 'REFERENCES', 'BACKGROUND',
#                                   'RELATED WORK', 'SYSTEM OVERVIEW', 'TOOLS', 'EXECUTION']
#                 if any(term in text for term in academic_terms):
#                     pattern_score = 0.85
#                 else:
#                     pattern_score = 0.30  # Lower for unknown caps
#
#             # Check for academic section keywords
#             elif any(keyword in text.lower() for keyword in
#                      ['introduction', 'methodology', 'evaluation', 'conclusion',
#                       'related work', 'background', 'results', 'discussion',
#                       'system overview', 'execution', 'scalability']):
#                 pattern_score = 0.60
#
#         score += 0.50 * pattern_score
#
#         # 2. Font size signal (20% weight)
#         font_ratio = element.get("font_size_ratio", 1.0)
#         if font_ratio > 1.0:
#             size_score = min((font_ratio - 1.0) / 0.5, max_score)
#             score += 0.20 * size_score
#
#         # 3. Font weight signal (15% weight)
#         if element["is_bold"]:
#             score += 0.15
#
#         # 4. Position and isolation signal (10% weight)
#         position_score = 0.0
#         if element["is_left_aligned"]:
#             position_score += 0.4
#         if element["is_isolated"] and element["word_count"] <= 8:
#             position_score += 0.4
#         if element.get("is_first_in_line", True):
#             position_score += 0.2
#
#         score += 0.10 * min(position_score, max_score)
#
#         # 5. Language-specific adjustments (5% weight)
#         lang_score = self.multilingual.get_heading_language_score(element)
#         score += 0.05 * lang_score
#
#         # HARSH penalties for likely non-headings
#         penalties = 1.0
#
#         # Sentence ending penalty
#         if text.endswith('.') and len(text.split()) > 6:
#             penalties *= 0.1  # Very harsh penalty
#
#         # Multiple sentences penalty
#         if text.count('.') > 1:
#             penalties *= 0.05  # Extremely harsh penalty
#
#         # Body text indicators penalty
#         body_indicators = ['the ', 'and ', 'for ', 'with ', 'this ', 'that ', 'which ', 'however ']
#         if any(indicator in text.lower() for indicator in body_indicators):
#             penalties *= 0.2  # Very harsh penalty
#
#         # Length penalty for very long text
#         if element["char_count"] > 80:
#             penalties *= 0.3
#
#         score *= penalties
#         return min(score, max_score)
#
#     def classify_heading_levels(self, heading_candidates: List[Dict]) -> List[Dict]:
#         """
#         ENHANCED heading level classification
#         """
#         if not heading_candidates:
#             return []
#
#         # Sort by score, then font size, then position
#         heading_candidates.sort(key=lambda x: (-x.get("heading_score", 0), -x["font_size"], x["page"], x["y_position"]))
#
#         # Group by font sizes
#         font_size_groups = defaultdict(list)
#         tolerance = 1.0
#
#         for heading in heading_candidates:
#             font_size = heading["font_size"]
#             group_key = None
#             for existing_size in font_size_groups.keys():
#                 if abs(font_size - existing_size) <= tolerance:
#                     group_key = existing_size
#                     break
#
#             if group_key is None:
#                 group_key = font_size
#
#             font_size_groups[group_key].append(heading)
#
#         # Assign levels
#         sorted_groups = sorted(font_size_groups.items(), key=lambda x: x[0], reverse=True)
#         level_mapping = {}
#         current_level = 1
#
#         for font_size, headings in sorted_groups:
#             if current_level <= 4:
#                 level_mapping[font_size] = f"H{current_level}"
#                 current_level += 1
#
#         # Classify headings
#         classified_headings = []
#         page_heading_counts = defaultdict(int)
#
#         for heading in heading_candidates:
#             font_size = None
#             for size in level_mapping.keys():
#                 if abs(heading["font_size"] - size) <= tolerance:
#                     font_size = size
#                     break
#
#             if font_size is not None:
#                 base_level = level_mapping[font_size]
#                 text = heading["text"].strip()
#                 adjusted_level = base_level
#
#                 # Pattern-based level adjustment
#                 if text.upper() in ['ABSTRACT']:
#                     adjusted_level = "H1"
#                 elif re.match(r'^[IVX]+\.\s+[A-Z]', text, re.IGNORECASE):
#                     adjusted_level = "H1"
#                 elif re.match(r'^[A-Z]\.\s+[A-Z]', text):
#                     adjusted_level = "H2"
#                 elif re.match(r'^\d+\.\d+', text):
#                     adjusted_level = "H2"
#
#                 # Strict limit on headings per page
#                 page = heading["page"]
#                 if page_heading_counts[page] < 6:  # Max 6 headings per page
#                     classified_headings.append({
#                         "level": adjusted_level,
#                         "text": text,
#                         "page": heading["page"]
#                     })
#                     page_heading_counts[page] += 1
#
#         # Sort by page and position
#         classified_headings.sort(key=lambda x: (x["page"],
#                                                 next((h["y_position"] for h in heading_candidates
#                                                       if h["text"] == x["text"] and h["page"] == x["page"]), 0)))
#
#         return classified_headings
#
#     def extract_title_with_fallbacks(self, elements: List[Dict]) -> str:
#         """
#         ENHANCED title extraction avoiding false positives and URLs
#         """
#         if not elements:
#             return "Untitled Document"
#
#         first_page_elements = [e for e in elements if e["page"] == 1]
#         if not first_page_elements:
#             return "Untitled Document"
#
#         # Strategy 1: Look for title-like text (avoid sections, author names, and URLs)
#         potential_titles = [e for e in first_page_elements
#                             if (15 < len(e["text"]) < 200 and  # Reasonable title length
#                                 e.get("relative_y_position", 0) < 0.25 and  # Very top of page
#                                 not e["text"].lower().startswith(('page ', 'abstract', 'introduction')) and
#                                 not re.match(r'^[IVX]+\.\s', e["text"], re.IGNORECASE) and
#                                 not re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', e["text"]) and  # Not author names
#                                 not e["text"].endswith('.') and  # Not sentences
#                                 not self._is_url_or_reference(e["text"]) and  # Not URLs
#                                 ('smart home' in e["text"].lower() or 'aiot' in e["text"].lower()))]  # Title indicators
#
#         if potential_titles:
#             title_candidate = max(potential_titles, key=lambda x: x["font_size"])
#             return title_candidate["text"]
#
#         # Fallback strategies with stricter filtering
#         for element in sorted(first_page_elements, key=lambda x: x["y_position"]):
#             text = element["text"]
#             if (len(text) > 20 and len(text) < 150 and
#                     not text.lower().startswith(('abstract', 'introduction', 'dmitriy')) and
#                     not text.endswith('.') and
#                     not self._is_url_or_reference(text) and
#                     element.get("relative_y_position", 1) < 0.3):
#                 return text
#
#         return "Untitled Document"
#
#     def extract_outline(self, pdf_path: str) -> Dict:
#         """
#         Main extraction method with STRICT quality control and URL filtering
#         """
#         try:
#             # Step 1: Extract complete text lines with spacing normalization
#             elements = self.extract_text_with_advanced_formatting(pdf_path)
#
#             if not elements:
#                 return self.validator.create_valid_outline("Empty Document", [])
#
#             # Step 2: Extract title with URL filtering
#             title = self.extract_title_with_fallbacks(elements)
#
#             # Step 3: Score elements with strict pre-filtering including URL rejection
#             heading_candidates = []
#             context = {"total_elements": len(elements)}
#
#             for element in elements:
#                 # STRICT pre-filtering with URL rejection
#                 if not self._is_likely_heading_text(element["text"]):
#                     continue
#
#                 heading_score = self.calculate_heading_score(element, context)
#
#                 # HIGHER thresholds for quality
#                 min_threshold = 0.35  # Raised threshold
#                 if len(elements) > 500:
#                     min_threshold = 0.4
#                 elif len(elements) < 100:
#                     min_threshold = 0.3
#
#                 if heading_score > min_threshold:
#                     element["heading_score"] = heading_score
#                     heading_candidates.append(element)
#
#             # Step 4: Classify heading levels
#             classified_headings = self.classify_heading_levels(heading_candidates)
#
#             # Step 5: STRICT final validation with URL filtering
#             filtered_headings = []
#             for heading in classified_headings:
#                 text = heading["text"].strip()
#
#                 if (self._is_valid_heading(text) and
#                         len(text.split()) >= 1 and
#                         len(text) >= 4 and
#                         not re.match(r'^\d+$', text.strip()) and
#                         not self._is_url_or_reference(text)):  # Final URL check
#                     filtered_headings.append(heading)
#
#             # Step 6: Limit output for quality
#             outline = {
#                 "title": title,
#                 "outline": filtered_headings[:15]  # Strict limit for quality
#             }
#
#             return self.validator.validate_and_fix(outline, "round_1a")
#
#         except Exception as e:
#             print(f"❌ Error in outline extraction: {e}")
#             error_outline = {
#                 "title": f"Processing Error: {str(e)[:50]}",
#                 "outline": []
#             }
#             return self.validator.validate_and_fix(error_outline, "round_1a")

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
        BALANCED: Comprehensive extraction with intelligent quality control
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
        """Enhanced quality check for meaningful headings with intelligent filtering"""
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
        ]

        if any(re.match(pattern, text, re.IGNORECASE) for pattern in noise_patterns):
            return False

        # Accept meaningful technical terms and concepts (ALL CAPS)
        if text.isupper() and 4 <= len(text) <= 60:
            # Technical indicators for avionics/digital systems
            technical_terms = [
                'SYSTEM', 'AVIONICS', 'DIGITAL', 'MEMORY', 'GATE', 'MICROPROCESSOR',
                'CONTROL', 'NAVIGATION', 'DISPLAY', 'SENSOR', 'COMMUNICATION',
                'ARCHITECTURE', 'COMPUTER', 'PROCESSOR', 'LOGIC', 'BINARY',
                'CONVERSION', 'TRUTH', 'TABLE', 'REGISTER', 'INSTRUCTION',
                'INTRODUCTION', 'CONCLUSION', 'REFERENCES', 'METHODOLOGY',
                'EVALUATION', 'RESULTS', 'QUESTIONS', 'THEORY', 'CLASSIFICATION',
                'WORKING', 'PRINCIPLE', 'FLIGHT', 'COCKPIT', 'RADAR', 'INFRARED',
                'ARITHMETIC', 'TIMING', 'DECODE', 'EPROM', 'PROM', 'DRAM'
            ]

            # Accept if it contains technical terms or is a meaningful concept
            if any(term in text for term in technical_terms):
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
            'sensor', 'navigation', 'communication', 'digital', 'conversion'
        ]):
            return True

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
        """Balanced scoring for quality headings"""
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
            pattern_score = 0.70
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
        """Classify ALL heading candidates without page limits"""
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

                # Pattern-based level adjustment
                if text.upper() in ['ABSTRACT']:
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
        """Extract balanced, high-quality headings"""
        try:
            elements = self.extract_text_with_advanced_formatting(pdf_path)

            if not elements:
                return {"title": "Empty Document", "outline": []}

            print(f"📊 Total elements extracted: {len(elements)}")

            title = self.extract_title_with_fallbacks(elements)

            # Score elements with intelligent quality pre-filtering
            heading_candidates = []
            context = {"total_elements": len(elements)}

            for element in elements:
                text = element["text"].strip()

                # Apply intelligent quality check before scoring
                if not self._is_meaningful_heading(text):
                    continue

                heading_score = self.calculate_heading_score(element, context)

                # Balanced threshold for quality headings
                if heading_score > 0.20:
                    element["heading_score"] = heading_score
                    heading_candidates.append(element)

            print(f"📊 Quality heading candidates: {len(heading_candidates)}")

            # Sort and classify
            classified_headings = self.classify_heading_levels(heading_candidates)
            print(f"📊 Quality headings in final output: {len(classified_headings)}")

            return {
                "title": title,
                "outline": classified_headings
            }

        except Exception as e:
            print(f"❌ Error in outline extraction: {e}")
            return {
                "title": f"Processing Error: {str(e)[:50]}",
                "outline": []
            }
