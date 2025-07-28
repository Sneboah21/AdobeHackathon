import json
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
import os
from heading_extractor import HeadingExtractor
from json_validator import JSONValidator


class PersonaAnalyzer:
    def __init__(self):
        """
        Advanced persona-driven document intelligence
        Reuses HeadingExtractor for modular design
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  Warning: spaCy model not found. Using basic text processing.")
            self.nlp = None

        self.heading_extractor = HeadingExtractor()
        self.validator = JSONValidator()

        # Initialize TF-IDF for semantic matching (lightweight, fast)
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            lowercase=True,
            strip_accents='unicode'
        )

        # ENHANCED: Domain-specific keywords including avionics
        self.domain_keywords = {
            'research': ['methodology', 'analysis', 'findings', 'results', 'discussion', 'literature', 'study',
                         'experiment', 'data', 'hypothesis', 'conclusion', 'evidence', 'investigation'],
            'business': ['strategy', 'market', 'revenue', 'growth', 'analysis', 'performance', 'metrics',
                         'profit', 'investment', 'customer', 'sales', 'competitive', 'financial'],
            'technical': ['implementation', 'algorithm', 'system', 'architecture', 'design', 'specification',
                          'framework', 'development', 'programming', 'software', 'hardware', 'engineering'],
            'academic': ['theory', 'hypothesis', 'experiment', 'conclusion', 'abstract', 'introduction',
                         'reference', 'citation', 'peer review', 'journal', 'conference', 'publication'],
            'medical': ['patient', 'treatment', 'diagnosis', 'clinical', 'therapy', 'symptom', 'disease',
                        'medical', 'health', 'drug', 'pharmaceutical', 'clinical trial'],
            'legal': ['law', 'regulation', 'compliance', 'contract', 'agreement', 'legal', 'court',
                      'statute', 'jurisdiction', 'liability', 'intellectual property', 'patent'],
            'avionics': ['avionics', 'aircraft', 'flight', 'navigation', 'radar', 'autopilot',
                         'cockpit', 'display', 'sensor', 'communication', 'control', 'system',
                         'digital', 'microprocessor', 'integration', 'architecture', 'design',
                         'management', 'automation', 'electronic', 'instrument', 'guidance']
        }

        # Section type importance weights for different job types
        self.job_section_weights = {
            'literature_review': {
                'introduction': 0.9, 'methodology': 0.6, 'discussion': 1.0,
                'conclusion': 0.8, 'results': 0.7, 'abstract': 0.9
            },
            'analysis': {
                'results': 1.0, 'discussion': 1.0, 'methodology': 0.8,
                'abstract': 0.9, 'introduction': 0.6
            },
            'summary': {
                'abstract': 1.0, 'introduction': 0.9, 'conclusion': 1.0,
                'results': 0.7, 'discussion': 0.8
            },
            'methodology': {
                'methodology': 1.0, 'introduction': 0.7, 'discussion': 0.6,
                'results': 0.5, 'abstract': 0.8
            }
        }

    def _is_valid_section_title(self, title: str) -> bool:
        """ENHANCED: Section title validation with better noise filtering"""
        if not title or len(title.strip()) < 3:
            return False

        title_clean = title.strip()

        # Reject page headers and noise
        noise_patterns = [
            r'^Volume\s*[–-]\s*[IVX]+$',  # "Volume – I"
            r'^Page\s*\d+$',  # "Page 1"
            r'^\d+\.\s*.{1,60}$',  # Short numbered items (likely questions/steps)
            r'^[A-Z]\.\s*.{1,40}$',  # Short lettered items
            r'^Part\s*[A-Z–-]*$',  # "Part A", "Part – B"
            r'^Chapter\s*\d*$',  # "Chapter 1"
            r'^Section\s*\d*$',  # "Section 1"
            r'^\d+\.\s+What\s+.{1,50}\?*$',  # Questions starting with "What"
            r'^\d+\.\s+Give\s+.{1,50}$',  # Questions starting with "Give"
            r'^\d+\.\s+List\s+.{1,50}$',  # Questions starting with "List"
            r'^\d+\.\s+Write\s+.{1,50}$',  # Questions starting with "Write"
            r'^\d+\.\s+Explain\s+.{1,50}$',  # Questions starting with "Explain"
            r'^\d+\.\s+Compare\s+.{1,50}$',  # Questions starting with "Compare"
            r'^\d+\.\s+Draw\s+.{1,50}$',  # Questions starting with "Draw"
            r'^\d+\.\s+Describe\s+.{1,50}$',  # Questions starting with "Describe"
        ]

        if any(re.match(pattern, title_clean, re.IGNORECASE) for pattern in noise_patterns):
            return False

        # Reject sentence fragments (but keep proper section titles ending with periods)
        if (title_clean.endswith('.') and
                len(title_clean.split()) > 12 and
                not title_clean.upper().endswith(('INTRODUCTION.', 'CONCLUSION.', 'ABSTRACT.', 'METHODOLOGY.'))):
            return False

        # Reject obvious processing steps and incomplete sentences
        step_patterns = [
            r'^\d+\.\s+The\s+.+\s+(is|are|will|can|should)\s+.+$',  # "1. The microprocessor generates..."
            r'^\d+\.\s+In\s+response\s+to\s+.+$',  # "3. In response to this signal..."
            r'^\d+\.\s+Inside\s+the\s+.+$',  # "1. Inside the microprocessor..."
            r'^\d+\.\s+Similarly\s+the\s+.+$',  # "2.9. Similarly the input..."
            r'^\d+\.\s+Suitable\s+.+$',  # "2. Suitable control signal..."
        ]

        if any(re.match(pattern, title_clean, re.IGNORECASE) for pattern in step_patterns):
            return False

        return True

    def extract_document_sections(self, pdf_path: str) -> List[Dict]:
        """
        Extract semantic sections from PDF using advanced text processing
        Reuses heading detection logic for consistency
        """
        try:
            # Get formatted text elements
            elements = self.heading_extractor.extract_text_with_advanced_formatting(pdf_path)

            if not elements:
                return []

            sections = []
            current_section = {
                "text": "",
                "start_page": 1,
                "end_page": 1,
                "title": "",
                "section_type": "content",
                "word_count": 0,
                "char_count": 0,
                "heading_score": 0.0
            }

            section_break_threshold = 0.4  # Threshold for section breaks

            for i, element in enumerate(elements):
                # Check if this element is likely a section header
                heading_score = self.heading_extractor.calculate_heading_score(
                    element, {"total_elements": len(elements)}
                )

                # Determine if this should start a new section
                is_section_break = (
                        heading_score > section_break_threshold or
                        (element["is_bold"] and element["font_size"] > 12 and element["word_count"] <= 8) or
                        (element["page"] > current_section["end_page"] + 1)  # Page jump
                )

                if is_section_break and current_section["word_count"] > 20:
                    # Save previous section if it has substantial content
                    current_section["text"] = current_section["text"].strip()
                    if current_section["text"]:
                        sections.append(current_section.copy())

                    # Start new section
                    current_section = {
                        "text": element["text"] + "\n\n",
                        "start_page": element["page"],
                        "end_page": element["page"],
                        "title": element["text"],
                        "section_type": self._classify_section_type(element["text"]),
                        "word_count": len(element["text"].split()),
                        "char_count": len(element["text"]),
                        "heading_score": heading_score
                    }
                else:
                    # Add to current section
                    current_section["text"] += element["text"] + " "
                    current_section["end_page"] = element["page"]
                    element_words = len(element["text"].split())
                    current_section["word_count"] += element_words
                    current_section["char_count"] += len(element["text"])

                    # Update section type if we find a better indicator
                    if heading_score > 0.3:
                        new_type = self._classify_section_type(element["text"])
                        if new_type != "content":
                            current_section["section_type"] = new_type

            # Add final section
            if current_section["word_count"] > 20:
                current_section["text"] = current_section["text"].strip()
                if current_section["text"]:
                    sections.append(current_section)

            # ENHANCED: Post-process sections with better quality control
            processed_sections = []
            for section in sections:
                # Validate section title FIRST
                if not self._is_valid_section_title(section["title"]):
                    continue

                # Clean up text
                section["text"] = re.sub(r'\s+', ' ', section["text"]).strip()

                # Skip very short sections
                if section["word_count"] < 15:
                    continue

                # Add text quality metrics
                section["text_quality_score"] = self._calculate_text_quality(section["text"])

                processed_sections.append(section)

            return processed_sections

        except Exception as e:
            print(f"❌ Error extracting sections from {pdf_path}: {e}")
            return []

    def _classify_section_type(self, title: str) -> str:
        """
        Classify section type based on title with improved accuracy
        """
        if not title:
            return 'content'

        title_lower = title.lower().strip()

        # Remove common prefixes
        title_clean = re.sub(r'^(\d+\.?\s*|\w\.\s*)', '', title_lower)

        # Classification patterns
        type_patterns = {
            'abstract': [r'\babstract\b', r'\bsummary\b', r'\bexecutive summary\b', r'\boverview\b'],
            'introduction': [r'\bintroduction\b', r'\bintro\b', r'\bbackground\b', r'\bpreamble\b'],
            'methodology': [r'\bmethodology\b', r'\bmethods?\b', r'\bapproach\b', r'\btechnique\b',
                            r'\bprocedure\b', r'\bexperimental\b'],
            'results': [r'\bresults?\b', r'\bfindings?\b', r'\boutcome\b', r'\bdata\b', r'\bobservation'],
            'discussion': [r'\bdiscussion\b', r'\banalysis\b', r'\binterpretation\b', r'\bimplication'],
            'conclusion': [r'\bconclusion\b', r'\bsummary\b', r'\bfinal\b', r'\bconcluding\b'],
            'references': [r'\breferences?\b', r'\bbibliography\b', r'\bcitations?\b', r'\bworks cited\b'],
            'appendix': [r'\bappendix\b', r'\battachment\b', r'\bsupplement\b']
        }

        for section_type, patterns in type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, title_clean):
                    return section_type

        return 'content'

    def _calculate_text_quality(self, text: str) -> float:
        """
        Calculate text quality score based on various metrics
        """
        if not text:
            return 0.0

        score = 0.0

        # Length appropriateness (not too short, not too long)
        word_count = len(text.split())
        if 20 <= word_count <= 1000:
            score += 0.3
        elif word_count > 1000:
            score += 0.2  # Very long sections get lower quality

        # Sentence structure
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if 8 <= avg_sentence_length <= 25:
            score += 0.2

        # Vocabulary richness (unique words ratio)
        words = text.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            uniqueness_ratio = len(unique_words) / len(words)
            score += 0.2 * min(uniqueness_ratio * 2, 1.0)

        # Professional/academic language indicators
        academic_indicators = ['analysis', 'research', 'study', 'findings', 'methodology',
                               'results', 'discussion', 'conclusion', 'evidence', 'data']
        indicator_count = sum(1 for indicator in academic_indicators if indicator in text.lower())
        score += 0.3 * min(indicator_count / 5, 1.0)

        return min(score, 1.0)

    def calculate_persona_relevance(self, sections: List[Dict], persona: str, job_description: str) -> List[Dict]:
        """
        Calculate relevance scores using advanced semantic matching
        """
        if not sections:
            return []

        try:
            # Combine persona and job description for query
            query_text = f"{persona} {job_description}"

            # Prepare texts for TF-IDF
            section_texts = []
            for section in sections:
                # Combine title and content for better matching, limit for performance
                combined_text = f"{section['title']} {section['text'][:1500]}"
                section_texts.append(combined_text)

            if not section_texts:
                return sections

            # Add query to corpus for TF-IDF
            all_texts = section_texts + [query_text]

            # Calculate TF-IDF matrix
            tfidf_matrix = self.tfidf.fit_transform(all_texts)

            # Calculate similarity between query and each section
            query_vector = tfidf_matrix[-1]  # Last item is the query
            section_vectors = tfidf_matrix[:-1]  # All except last

            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, section_vectors).flatten()

            # Add relevance scores and additional features
            for i, section in enumerate(sections):
                section["relevance_score"] = float(similarities[i])
                section["domain_score"] = self._calculate_domain_score(section, persona)
                section["section_type_score"] = self._calculate_section_type_score(section, job_description)
                section["length_score"] = self._calculate_length_score(section)

                # Combined score with weights
                section["combined_score"] = (
                        0.5 * section["relevance_score"] +
                        0.25 * section["domain_score"] +
                        0.15 * section["section_type_score"] +
                        0.1 * section["length_score"]
                )

                # Boost score for high-quality sections
                section["combined_score"] *= (0.7 + 0.3 * section.get("text_quality_score", 0.5))

            # Sort by combined score
            sections.sort(key=lambda x: x["combined_score"], reverse=True)

            # Add importance ranks
            for i, section in enumerate(sections):
                section["importance_rank"] = i + 1

            return sections

        except Exception as e:
            print(f"❌ Error calculating relevance: {e}")
            # Fallback: return sections with default scores
            for i, section in enumerate(sections):
                section["relevance_score"] = 0.5
                section["combined_score"] = 0.5
                section["importance_rank"] = i + 1
            return sections

    def _calculate_domain_score(self, section: Dict, persona: str) -> float:
        """
        ENHANCED: Calculate domain-specific relevance score with avionics support
        """
        persona_lower = persona.lower()
        section_text = f"{section['title']} {section['text'][:800]}".lower()

        # Identify domain from persona
        domain = 'general'
        domain_scores = {}

        for domain_key in self.domain_keywords:
            if domain_key in persona_lower:
                domain_scores[domain_key] = persona_lower.count(domain_key)

        if domain_scores:
            domain = max(domain_scores.items(), key=lambda x: x[1])[0]

        if domain == 'general':
            return 0.5  # Neutral score

        # Count domain-specific keywords
        domain_words = self.domain_keywords.get(domain, [])
        keyword_matches = sum(1 for keyword in domain_words if keyword in section_text)

        # Normalize by section length and keyword list length
        text_length_factor = min(len(section_text.split()) / 100, 1.0)  # Normalize by length
        score = min(keyword_matches / max(len(domain_words) * 0.3, 1), 1.0) * text_length_factor

        return score

    def _calculate_section_type_score(self, section: Dict, job_description: str) -> float:
        """
        Score section type relevance to job description
        """
        job_lower = job_description.lower()
        section_type = section.get("section_type", "content")

        # Identify job type
        job_type = 'general'
        if any(term in job_lower for term in ['literature review', 'review', 'survey']):
            job_type = 'literature_review'
        elif any(term in job_lower for term in ['analysis', 'analyze', 'examine']):
            job_type = 'analysis'
        elif any(term in job_lower for term in ['summary', 'summarize', 'overview']):
            job_type = 'summary'
        elif any(term in job_lower for term in ['methodology', 'method', 'approach']):
            job_type = 'methodology'

        # Get weights for this job type
        weights = self.job_section_weights.get(job_type, {
            'content': 0.6, 'abstract': 0.7, 'introduction': 0.7,
            'results': 0.6, 'discussion': 0.6, 'conclusion': 0.6
        })

        return weights.get(section_type, 0.5)

    def _calculate_length_score(self, section: Dict) -> float:
        """
        Score based on section length (prefer substantial but not excessive content)
        """
        word_count = section.get("word_count", 0)

        if 50 <= word_count <= 500:
            return 1.0
        elif 30 <= word_count < 50 or 500 < word_count <= 800:
            return 0.8
        elif 20 <= word_count < 30 or 800 < word_count <= 1200:
            return 0.6
        elif word_count > 1200:
            return 0.4
        else:
            return 0.2

    def generate_subsection_analysis(self, top_sections: List[Dict], persona: str, job_description: str) -> List[Dict]:
        """
        Generate refined subsection analysis for top sections
        """
        subsection_analysis = []

        try:
            for section in top_sections[:5]:  # Top 5 sections only
                # Split section into logical paragraphs
                text = section["text"]

                # First try double newlines
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]

                if len(paragraphs) < 2:
                    # Fallback: split by sentence groups
                    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
                    # Group sentences into paragraphs of 2-4 sentences
                    paragraphs = []
                    for i in range(0, len(sentences), 3):
                        para = ". ".join(sentences[i:i + 3])
                        if len(para) > 50:
                            paragraphs.append(para)

                if not paragraphs:
                    # Last resort: chunk by character count
                    chunk_size = 300
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i + chunk_size]
                        if len(chunk.strip()) > 50:
                            paragraphs.append(chunk.strip())

                # Score paragraphs for relevance
                if paragraphs:
                    query_text = f"{persona} {job_description}"

                    try:
                        para_texts = [p[:800] for p in paragraphs]  # Limit paragraph length
                        all_para_texts = para_texts + [query_text]

                        para_tfidf = TfidfVectorizer(
                            max_features=1000,
                            stop_words='english',
                            ngram_range=(1, 2)
                        ).fit_transform(all_para_texts)

                        query_vec = para_tfidf[-1]
                        para_vecs = para_tfidf[:-1]

                        para_similarities = cosine_similarity(query_vec, para_vecs).flatten()

                        # Get top paragraphs with quality filtering
                        para_scores = list(zip(paragraphs, para_similarities))
                        para_scores.sort(key=lambda x: x[1], reverse=True)

                        # Add top 2-3 paragraphs from this section
                        added_count = 0
                        for paragraph, score in para_scores:
                            if score > 0.1 and added_count < 3:  # Minimum relevance threshold
                                # Additional quality checks
                                word_count = len(paragraph.split())
                                if 20 <= word_count <= 300:  # Reasonable length
                                    subsection_analysis.append({
                                        "document": section.get("document", "Unknown"),
                                        "refined_text": paragraph.strip(),
                                        "page_number": section["start_page"]
                                    })
                                    added_count += 1

                    except Exception as e:
                        print(f"⚠️  Warning: Error processing paragraphs: {e}")
                        # Fallback: add first substantial paragraph
                        for para in paragraphs[:2]:
                            if len(para.split()) >= 20:
                                subsection_analysis.append({
                                    "document": section.get("document", "Unknown"),
                                    "refined_text": para.strip(),
                                    "page_number": section["start_page"]
                                })
                                break

        except Exception as e:
            print(f"❌ Error generating subsection analysis: {e}")

        return subsection_analysis

    def analyze_documents(self, document_paths: List[str], persona: str, job_description: str) -> Dict:
        """
        Main analysis method - combines all advanced techniques
        """
        if not document_paths:
            return self._create_empty_analysis(persona, job_description)

        all_sections = []
        processing_errors = []

        print(f"📚 Processing {len(document_paths)} documents...")

        # Process each document
        for doc_path in document_paths:
            try:
                # FIXED: Windows-compatible document name extraction
                doc_name = os.path.basename(doc_path)
                print(f"📄 Processing: {doc_name}")

                sections = self.extract_document_sections(doc_path)

                if not sections:
                    processing_errors.append(f"No sections extracted from {doc_name}")
                    continue

                # Add document metadata
                for section in sections:
                    section["document"] = doc_name

                all_sections.extend(sections)
                print(f"✅ Extracted {len(sections)} sections from {doc_name}")

            except Exception as e:
                error_msg = f"Error processing {doc_path}: {str(e)}"
                processing_errors.append(error_msg)
                print(f"❌ {error_msg}")

        if not all_sections:
            print("❌ No sections were successfully extracted from any document")
            return self._create_empty_analysis(persona, job_description, processing_errors)

        print(f"🔍 Total sections extracted: {len(all_sections)}")

        # Calculate relevance scores
        try:
            relevant_sections = self.calculate_persona_relevance(all_sections, persona, job_description)
        except Exception as e:
            print(f"❌ Error calculating relevance: {e}")
            relevant_sections = all_sections
            for i, section in enumerate(relevant_sections):
                section["importance_rank"] = i + 1

        # Generate outputs
        extracted_sections = []
        for section in relevant_sections[:15]:  # Top 15
            extracted_sections.append({
                "document": section["document"],
                "page_number": section["start_page"],
                "section_title": section["title"][:200],  # Limit title length
                "importance_rank": section["importance_rank"]
            })

        # Generate subsection analysis
        try:
            subsection_analysis = self.generate_subsection_analysis(relevant_sections, persona, job_description)
        except Exception as e:
            print(f"❌ Error generating subsection analysis: {e}")
            subsection_analysis = []

        # FIXED: Clean document names in metadata
        clean_doc_names = [os.path.basename(path) for path in document_paths]

        # Create final result
        result = {
            "metadata": {
                "input_documents": clean_doc_names,
                "persona": persona,
                "job_to_be_done": job_description,
                "processing_timestamp": "2024-01-01T00:00:00Z"
            },
            "extracted_sections": extracted_sections,
            "sub_section_analysis": subsection_analysis
        }

        # Validate and return
        validated_result = self.validator.validate_and_fix(result, "round_1b")

        print(f"✅ Analysis complete: {len(extracted_sections)} sections, {len(subsection_analysis)} subsections")

        return validated_result

    def _create_empty_analysis(self, persona: str, job_description: str, errors: List[str] = None) -> Dict:
        """Create empty analysis result for error cases"""
        return {
            "metadata": {
                "input_documents": [],
                "persona": persona,
                "job_to_be_done": job_description,
                "processing_timestamp": "2024-01-01T00:00:00Z",
                "errors": errors or []
            },
            "extracted_sections": [],
            "sub_section_analysis": []
        }
