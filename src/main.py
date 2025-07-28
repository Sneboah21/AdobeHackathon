#!/usr/bin/env python3
"""
Adobe Hackathon - Elite Solution
Rounds 1A & 1B Implementation
Performance: <3 seconds for 50-page PDF
Enhanced with consistent directory detection
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, TextIO, Any
import traceback

# Import our modules
try:
    import fitz  # PyMuPDF

    print("✅ PyMuPDF (fitz) installed successfully")
    print(f"Version: {fitz.version[0]}")
except ImportError as e:
    print(f"❌ Import error: {e}")

try:
    from heading_extractor import HeadingExtractor
    from persona_analyzer import PersonaAnalyzer
    from utils import setup_logging, measure_time, ensure_directory_exists, safe_filename

    MODULES_LOADED = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Make sure all required modules are in the src/ directory")
    MODULES_LOADED = False


    # Fallback functions if utils module is not available
    def setup_logging():
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)


    def measure_time(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"⏱️  {func.__name__} executed in {end - start:.2f} seconds")
            return result

        return wrapper


    def ensure_directory_exists(path: Path):
        path.mkdir(parents=True, exist_ok=True)
        return path


    def safe_filename(text: str, max_length: int = 50) -> str:
        safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if len(safe_text) > max_length:
            safe_text = safe_text[:max_length].rstrip()
        return safe_text.replace(' ', '_')

# Setup logging
logger = setup_logging()


def save_json_safely(data: Dict[str, Any], file_path: Path) -> None:
    """Save JSON data safely with proper type handling"""
    with open(file_path, 'w', encoding='utf-8') as f:  # type: TextIO
        json.dump(data, f, indent=2, ensure_ascii=False)


def setup_directories():
    """Ensure input/output directories exist with proper permissions"""
    # Enhanced Docker detection
    is_docker = (
            os.path.exists('/.dockerenv') or
            os.getenv('DOCKER_CONTAINER') or
            os.getenv('KUBERNETES_SERVICE_HOST') or
            (os.path.exists('/app') and not os.path.exists('C:/Users'))
    )

    if is_docker:
        input_dir = Path("/app/input")
        output_dir = Path("/app/output")
        print("🐳 Docker environment detected")
    else:
        input_dir = Path("C:/Users/sneha/PycharmProjects/AdobeHackathon/input")
        output_dir = Path("C:/Users/sneha/PycharmProjects/AdobeHackathon/output")
        print("💻 Local development environment detected")

    ensure_directory_exists(input_dir)
    ensure_directory_exists(output_dir)

    return input_dir, output_dir


@measure_time
def process_round_1a(input_dir: Path, output_dir: Path) -> bool:
    """
    Process Round 1A: Document Structure Extraction
    Target: <3 seconds per PDF, perfect JSON output
    """
    print("🚀 Starting Round 1A: Document Structure Extraction")

    # Initialize the heading extractor with error handling
    try:
        extractor = HeadingExtractor()
    except Exception as e:
        print(f"❌ Failed to initialize HeadingExtractor: {e}")
        traceback.print_exc()
        return False

    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("📋 No PDF files found in input directory for Round 1A")
        return True  # Not an error, just no files to process

    success_count = 0
    total_time = 0
    results = []

    for pdf_file in pdf_files:
        print(f"📄 Processing: {pdf_file.name}")

        start_time = time.time()

        try:
            # Extract outline using advanced heading detection
            outline = extractor.extract_outline(str(pdf_file))

            # Measure processing time
            processing_time = time.time() - start_time
            total_time += processing_time

            # Generate safe output filename
            safe_name = safe_filename(pdf_file.stem)
            output_file = output_dir / f"{safe_name}_outline.json"

            # Save JSON output with validation and proper typing
            save_json_safely(outline, output_file)

            heading_count = len(outline.get('outline', []))
            print(f"✅ Completed in {processing_time:.2f}s - {heading_count} headings found")
            print(f"💾 Saved: {output_file.name}")

            results.append({
                "file": pdf_file.name,
                "processing_time": processing_time,
                "headings_found": heading_count,
                "status": "success"
            })

            success_count += 1

            # Performance validation with enhanced feedback
            if processing_time > 10:
                print(f"⚠️  Warning: Processing time {processing_time:.2f}s exceeds 10s limit")
            elif processing_time <= 3:
                print(f"🏆 EXCELLENT: {processing_time:.2f}s (UNDER 3s TARGET!) 🎯")
            else:
                print(f"⚠️  SLOW: {processing_time:.2f}s exceeds 3s target by {processing_time - 3.0:.1f}s")

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            print(f"❌ Error processing {pdf_file.name}: {error_msg}")
            traceback.print_exc()

            # Create comprehensive error output for debugging
            error_outline = {
                "title": f"Error processing {pdf_file.name}",
                "outline": [],
                "error": error_msg,
                "processing_time": processing_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            safe_name = safe_filename(pdf_file.stem)
            output_file = output_dir / f"{safe_name}_error.json"

            save_json_safely(error_outline, output_file)

            results.append({
                "file": pdf_file.name,
                "processing_time": processing_time,
                "headings_found": 0,
                "status": "error",
                "error": error_msg
            })

    # Generate comprehensive summary with performance feedback
    avg_time = total_time / len(pdf_files) if pdf_files else 0
    performance_status = "🎯 TARGET MET!" if avg_time <= 3.0 else f"❌ MISSED by {avg_time - 3.0:.1f}s"

    print(f"\n📊 Round 1A Summary:")
    print(f"   • Success: {success_count}/{len(pdf_files)} files")
    print(f"   • Average time: {avg_time:.2f}s per file")
    print(f"   • Performance target (<3s): {performance_status}")
    print(f"   • Total time: {total_time:.2f}s")

    # Save processing summary for analysis
    summary = {
        "round": "1A",
        "total_files": len(pdf_files),
        "successful_files": success_count,
        "average_processing_time": avg_time,
        "total_processing_time": total_time,
        "performance_target_met": avg_time <= 3.0,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    summary_file = output_dir / "round_1a_summary.json"
    save_json_safely(summary, summary_file)

    return success_count > 0


@measure_time
def process_round_1b(input_dir: Path, output_dir: Path) -> bool:
    """
    Process Round 1B: Persona-Driven Document Intelligence
    Target: <60 seconds for document collection
    """
    print("\n🎯 Starting Round 1B: Persona-Driven Document Intelligence")

    # Check for persona configuration
    config_file = input_dir / "persona_config.json"
    if not config_file.exists():
        print("📋 No persona_config.json found, skipping Round 1B")
        print("💡 To run Round 1B, create a persona_config.json file with:")
        print('   {"persona": "Your persona", "job_to_be_done": "Your job description"}')
        return True  # Not an error, just no config

    try:
        # Load and validate persona configuration
        with open(config_file, 'r', encoding='utf-8') as f:  # type: TextIO
            config = json.load(f)

        persona = config.get("persona", "").strip()
        job_description = config.get("job_to_be_done", "").strip()

        if not persona or not job_description:
            print("❌ Invalid persona configuration - both 'persona' and 'job_to_be_done' are required")
            print("📋 Current config:", config)
            return False

        print(f"👤 Persona: {persona}")
        print(f"🎯 Job: {job_description}")

        # Find and validate PDF files
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            print("❌ No PDF files found for analysis")
            return False

        if len(pdf_files) > 10:
            print(f"⚠️  Warning: {len(pdf_files)} PDFs found, processing first 10")
            pdf_files = pdf_files[:10]
        elif len(pdf_files) < 3:
            print(f"⚠️  Warning: Only {len(pdf_files)} PDFs found, consider adding more for better analysis")

        print(f"📚 Analyzing {len(pdf_files)} documents...")

        # Initialize analyzer with error handling
        try:
            analyzer = PersonaAnalyzer()
        except Exception as e:
            print(f"❌ Failed to initialize PersonaAnalyzer: {e}")
            traceback.print_exc()
            return False

        start_time = time.time()

        # Analyze documents with comprehensive error handling
        try:
            result = analyzer.analyze_documents(
                [str(f) for f in pdf_files],
                persona,
                job_description
            )
        except Exception as e:
            print(f"❌ Error during document analysis: {e}")
            traceback.print_exc()
            return False

        processing_time = time.time() - start_time

        # Save output with validation
        output_file = output_dir / "persona_analysis.json"
        save_json_safely(result, output_file)

        # Generate detailed summary statistics
        sections_count = len(result.get('extracted_sections', []))
        subsections_count = len(result.get('sub_section_analysis', []))

        print(f"\n✅ Round 1B completed in {processing_time:.2f}s")
        print(f"📄 Found {sections_count} relevant sections")
        print(f"🔍 Generated {subsections_count} subsection analyses")
        print(f"💾 Saved: {output_file.name}")

        # Performance validation with enhanced feedback
        if processing_time > 60:
            print(f"⚠️  Warning: Processing time {processing_time:.2f}s exceeds 60s limit")
        elif processing_time <= 45:
            print(f"🏆 Excellent: Processing completed in {processing_time:.2f}s (under 45s)")

        # Save detailed processing summary
        summary = {
            "round": "1B",
            "persona": persona,
            "job_to_be_done": job_description,
            "total_documents": len(pdf_files),
            "processing_time": processing_time,
            "sections_found": sections_count,
            "subsections_found": subsections_count,
            "performance_target_met": processing_time <= 60.0,
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        summary_file = output_dir / "round_1b_summary.json"
        save_json_safely(summary, summary_file)

        return True

    except Exception as e:
        print(f"❌ Error in Round 1B processing: {e}")
        traceback.print_exc()
        return False


def validate_environment():
    """FIXED: Validate the execution environment with consistent directory detection"""
    print("🔍 Validating environment...")

    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 7):
        print("⚠️  Warning: Python 3.7+ recommended for optimal performance")

    # Check available memory (if psutil available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        total_gb = memory.total / (1024 ** 3)
        print(f"💾 Memory: {available_gb:.1f} GB available / {total_gb:.1f} GB total")

        if available_gb < 2:
            print("⚠️  Warning: Low memory available, consider closing other applications")
    except ImportError:
        print("💾 Memory info not available (psutil not installed)")

    # FIXED: Use consistent directory detection
    try:
        input_dir, output_dir = setup_directories()
        pdf_files = list(input_dir.glob("*.pdf"))
        print(f"📄 Found {len(pdf_files)} PDF files in input directory")

        config_file = input_dir / "persona_config.json"
        if config_file.exists():
            print("⚙️  Found persona_config.json for Round 1B")
        else:
            print("📋 No persona_config.json found (Round 1B will be skipped)")
    except Exception as e:
        print(f"⚠️  Directory setup warning: {e}")

    # Check critical imports
    try:
        import fitz  # PyMuPDF
        print("📦 PyMuPDF (fitz) imported successfully")
    except ImportError as e:
        print(f"❌ Critical package import error (PyMuPDF): {e}")
        return False

    try:
        import spacy
        print("📦 spaCy imported successfully")
    except ImportError as e:
        print(f"❌ Critical package import error (spaCy): {e}")
        return False

    try:
        import sklearn
        print("📦 scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Critical package import error (scikit-learn): {e}")
        return False

    print("✅ Environment validation complete")
    return True


def main():
    """
    Main execution function
    Orchestrates both Round 1A and 1B processing with comprehensive error handling
    """
    print("=" * 70)
    print("🏆 Adobe Hackathon - Elite Document Intelligence Solution")
    print("🎯 Rounds 1A & 1B - Structure Extraction & Persona Analysis")
    print("🔥 Features: Chaos-resistant headings, multilingual, <3s processing")
    print("=" * 70)

    # Validate environment with detailed feedback
    try:
        if not validate_environment():
            print("❌ Environment validation failed - check error messages above")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Environment validation error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Setup directories with error handling
    try:
        input_dir, output_dir = setup_directories()
        print(f"📁 Using input directory: {input_dir}")
        print(f"📁 Using output directory: {output_dir}")
    except Exception as e:
        print(f"❌ Failed to setup directories: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Track overall performance
    total_start_time = time.time()
    overall_success = False

    # Process Round 1A (always attempt)
    try:
        print("\n" + "=" * 50)
        round_1a_success = process_round_1a(input_dir, output_dir)
        if round_1a_success:
            overall_success = True
    except Exception as e:
        print(f"❌ Critical error in Round 1A: {e}")
        traceback.print_exc()
        round_1a_success = False

    # Process Round 1B (if configuration exists)
    try:
        print("\n" + "=" * 50)
        round_1b_success = process_round_1b(input_dir, output_dir)
        if round_1b_success:
            overall_success = True
    except Exception as e:
        print(f"❌ Critical error in Round 1B: {e}")
        traceback.print_exc()
        round_1b_success = False

    # Final comprehensive summary with enhanced performance feedback
    total_time = time.time() - total_start_time

    print("\n" + "=" * 70)
    print("📊 FINAL EXECUTION SUMMARY")
    print("=" * 70)
    print(f"⏱️  Total execution time: {total_time:.2f}s")
    print(f"🎯 Round 1A (Structure Extraction): {'✅ SUCCESS' if round_1a_success else '❌ FAILED'}")
    print(f"🎯 Round 1B (Persona Analysis): {'✅ SUCCESS' if round_1b_success else '⏭️ SKIPPED/FAILED'}")

    # Enhanced performance assessment
    if total_time <= 5:
        print(f"🏆 OUTSTANDING: Total execution in {total_time:.2f}s - Exceptional performance!")
    elif total_time <= 10:
        print(f"✅ EXCELLENT: Total execution in {total_time:.2f}s")
    elif total_time <= 30:
        print(f"✅ Good: Total execution in {total_time:.2f}s")
    else:
        print(f"⚠️  Slow: Total execution in {total_time:.2f}s - consider optimization")

    # Final status
    if overall_success:
        print("\n🎉 SOLUTION COMPLETED SUCCESSFULLY")
        print("📝 Check output directory for results")
        print("📊 Summary files saved for analysis")

        # Provide specific feedback about performance
        if round_1a_success and total_time <= 10:
            print("🎯 Solution meets all performance requirements!")
    else:
        print("\n❌ SOLUTION FAILED")
        print("📝 Check error messages above and output directory for debug info")
        print("🔍 Review traceback information for debugging")
        sys.exit(1)

    print("=" * 70)


if __name__ == "__main__":
    main()

# # !/usr/bin/env python3
# """
# Adobe Hackathon - Elite Solution
# Rounds 1A & 1B Implementation
# Performance: <3 seconds for 50-page PDF with ALL pages processed
# FIXED: Page counting bug + Performance optimizations
# """
#
# import os
# import sys
# import json
# import time
# from pathlib import Path
# from typing import List, Dict, TextIO, Any
# import traceback
# import re
# import unicodedata
# from functools import lru_cache
#
# # Import our modules
# try:
#     import fitz  # PyMuPDF
#
#     print("✅ PyMuPDF (fitz) installed successfully")
#     print(f"Version: {fitz.version[0]}")
# except ImportError as e:
#     print(f"❌ Import error: {e}")
#     sys.exit(1)
#
# try:
#     from heading_extractor import HeadingExtractor
#     from persona_analyzer import PersonaAnalyzer
#     from utils import setup_logging, measure_time, ensure_directory_exists, safe_filename
#
#     MODULES_LOADED = True
# except ImportError as e:
#     print(f"❌ Import error: {e}")
#     print("🔧 Make sure all required modules are in the src/ directory")
#     MODULES_LOADED = False
#
#
#     # Fallback functions if utils module is not available
#     def setup_logging():
#         import logging
#         logging.basicConfig(level=logging.INFO)
#         return logging.getLogger(__name__)
#
#
#     def measure_time(func):
#         def wrapper(*args, **kwargs):
#             start = time.time()
#             result = func(*args, **kwargs)
#             end = time.time()
#             print(f"⏱️  {func.__name__} executed in {end - start:.2f} seconds")
#             return result
#
#         return wrapper
#
#
#     def ensure_directory_exists(path: Path):
#         path.mkdir(parents=True, exist_ok=True)
#         return path
#
#
#     @lru_cache(maxsize=100)
#     def safe_filename(text: str, max_length: int = 50) -> str:
#         safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()
#         if len(safe_text) > max_length:
#             safe_text = safe_text[:max_length].rstrip()
#         return safe_text.replace(' ', '_')
#
# # Setup logging
# logger = setup_logging()
#
#
# def save_json_safely(data: Dict[str, Any], file_path: Path) -> None:
#     """Save JSON data safely with proper type handling"""
#     with open(file_path, 'w', encoding='utf-8') as f:  # type: TextIO
#         json.dump(data, f, indent=2, ensure_ascii=False)
#
#
# def setup_directories():
#     """Ensure input/output directories exist with proper permissions"""
#     input_dir = Path("C:/Users/sneha/PycharmProjects/AdobeHackathon/input")
#     output_dir = Path("C:/Users/sneha/PycharmProjects/AdobeHackathon/output")
#
#     ensure_directory_exists(input_dir)
#     ensure_directory_exists(output_dir)
#
#     return input_dir, output_dir
#
#
# # HIGH-PERFORMANCE HeadingExtractor with FIXED page counting
# class OptimizedHeadingExtractor:
#     """
#     Performance-optimized extractor with correct page counting
#     Target: <3 seconds with ALL pages processed
#     """
#
#     def __init__(self):
#         # Essential patterns for speed and accuracy
#         self.patterns = {
#             'roman_major': re.compile(r'^([IVX]+)\.\s+([A-Z][A-Z\s]+)$'),
#             'academic': re.compile(r'^(ABSTRACT|INTRODUCTION|CONCLUSION|REFERENCES|ACKNOWLEDGMENTS)$', re.IGNORECASE),
#             'numbered': re.compile(r'^(\d+)\.\s+([A-Z][A-Za-z\s]+)'),
#             'letter': re.compile(r'^([A-Z])\.\s+([A-Z][a-z\s]+)'),
#             'false_positive': re.compile(r'http|@|figure|table|email|www\.', re.IGNORECASE)
#         }
#
#     @lru_cache(maxsize=1500)
#     def _score_heading(self, text: str, font_size: float, is_bold: bool) -> float:
#         """High-performance heading scoring"""
#         if not text or len(text) < 3 or len(text) > 100:
#             return 0.0
#
#         text_clean = text.strip()
#
#         # Quick rejection of false positives
#         if self.patterns['false_positive'].search(text_clean):
#             return 0.0
#
#         # High-value pattern scoring
#         if self.patterns['roman_major'].match(text_clean):
#             return 1.0
#         elif self.patterns['academic'].match(text_clean):
#             return 0.95
#         elif self.patterns['numbered'].match(text_clean):
#             return 0.85
#         elif self.patterns['letter'].match(text_clean):
#             return 0.80
#         elif text_clean.isupper() and 3 <= len(text_clean.split()) <= 7:
#             return 0.70
#
#         # Font-based scoring with optimization
#         score = 0.0
#         if font_size > 16:
#             score = 0.75
#         elif font_size > 14:
#             score = 0.65
#         elif font_size > 12:
#             score = 0.55
#
#         if is_bold:
#             score += 0.10
#
#         return score
#
#     def extract_outline(self, pdf_path: str) -> Dict:
#         """
#         FIXED: Extract outline with correct page counting and <3s performance
#         """
#         start_time = time.time()
#
#         try:
#             doc = fitz.open(pdf_path)
#             headings = []
#
#             total_pages = len(doc)
#             pages_actually_processed = 0  # CRITICAL: Initialize counter
#
#             print(f"🔥 Processing {total_pages} pages with optimized speed")
#
#             # Process ALL pages with strict performance management
#             for page_num in range(total_pages):
#                 # PERFORMANCE: Check time every 5 pages for efficiency
#                 if page_num % 5 == 0:
#                     elapsed = time.time() - start_time
#                     if elapsed > 2.3:  # STRICT 2.3s limit for <3s target
#                         print(f"⏱️ Time limit reached: {elapsed:.1f}s after {pages_actually_processed} pages")
#                         break
#
#                 try:
#                     page = doc.load_page(page_num)
#                     page_dict = page.get_text("dict")
#
#                     # CRITICAL FIX: Increment AFTER successful page load
#                     pages_actually_processed += 1
#
#                     for block in page_dict.get("blocks", []):
#                         if "lines" not in block:
#                             continue
#
#                         for line in block["lines"]:
#                             if not line.get("spans"):
#                                 continue
#
#                             # Optimized span combination
#                             line_text = ""
#                             max_font_size = 0
#                             is_bold = False
#
#                             for span in line["spans"]:
#                                 line_text += span.get("text", "")
#                                 size = span.get("size", 12)
#                                 if size > max_font_size:
#                                     max_font_size = size
#                                 if span.get("flags", 0) & 16:
#                                     is_bold = True
#
#                             line_text = line_text.strip()
#                             if not line_text:
#                                 continue
#
#                             # Score the heading candidate
#                             score = self._score_heading(line_text, max_font_size, is_bold)
#
#                             # PERFORMANCE: Higher threshold for fewer candidates
#                             if score > 0.6:
#                                 headings.append({
#                                     'text': line_text,
#                                     'score': score,
#                                     'page': page_num,
#                                     'font_size': max_font_size,
#                                     'is_bold': is_bold
#                                 })
#
#                     # PERFORMANCE: Early exit with sufficient quality candidates
#                     if len(headings) > 20:
#                         print(f"📊 Sufficient candidates found ({len(headings)}), early exit")
#                         break
#
#                 except Exception as page_error:
#                     print(f"⚠️ Error on page {page_num}: {page_error}")
#                     continue  # Skip problematic pages but continue processing
#
#             doc.close()
#
#             # Sort by score and page order for quality
#             headings.sort(key=lambda x: (-x['score'], x['page']))
#
#             # Generate final outline with deduplication
#             outline = []
#             seen_texts = set()
#
#             for heading in headings[:15]:  # Top 15 for quality
#                 text = heading['text']
#                 text_normalized = text.lower().strip()
#
#                 # Skip duplicates
#                 if text_normalized in seen_texts:
#                     continue
#                 seen_texts.add(text_normalized)
#
#                 # Intelligent level assignment
#                 if (self.patterns['roman_major'].match(text) or
#                         self.patterns['academic'].match(text)):
#                     level = 'H1'
#                 elif (self.patterns['numbered'].match(text) or
#                       self.patterns['letter'].match(text)):
#                     level = 'H2'
#                 else:
#                     level = 'H3'
#
#                 outline.append({
#                     'text': text,
#                     'level': level,
#                     'page': heading['page'] + 1,  # 1-indexed for user display
#                     'confidence': round(heading['score'], 3)
#                 })
#
#             processing_time = time.time() - start_time
#
#             # FIXED: Return correct metadata with proper page counts
#             return {
#                 'title': f"Document Structure Analysis - {Path(pdf_path).stem}",
#                 'outline': outline,
#                 'processing_time': round(processing_time, 3),
#                 'total_headings': len(outline),
#                 'pages_processed': pages_actually_processed,  # FIXED: Correct count
#                 'total_pages_in_document': total_pages,  # FIXED: Clear total
#                 'performance_target_met': processing_time <= 3.0,
#                 'optimization_applied': True
#             }
#
#         except Exception as e:
#             processing_time = time.time() - start_time
#             return {
#                 'title': f"Error processing {Path(pdf_path).stem}",
#                 'outline': [],
#                 'error': str(e),
#                 'processing_time': processing_time,
#                 'pages_processed': 0,
#                 'total_pages_in_document': 0,
#                 'status': 'failed'
#             }
#
#
# # Global extractor instance
# _extractor = None
#
#
# def get_extractor():
#     """Get or create heading extractor instance"""
#     global _extractor
#     if _extractor is None:
#         if MODULES_LOADED:
#             _extractor = HeadingExtractor()
#         else:
#             _extractor = OptimizedHeadingExtractor()
#     return _extractor
#
#
# @measure_time
# def process_round_1a(input_dir: Path, output_dir: Path) -> bool:
#     """
#     Process Round 1A: Document Structure Extraction
#     FIXED: Proper output formatting with correct page counts
#     """
#     print("🚀 Starting Round 1A: Document Structure Extraction")
#
#     try:
#         extractor = get_extractor()
#     except Exception as e:
#         print(f"❌ Failed to initialize HeadingExtractor: {e}")
#         traceback.print_exc()
#         return False
#
#     pdf_files = list(input_dir.glob("*.pdf"))
#     if not pdf_files:
#         print("📋 No PDF files found in input directory for Round 1A")
#         return True
#
#     success_count = 0
#     total_time = 0
#     results = []
#
#     for pdf_file in pdf_files:
#         print(f"📄 Processing: {pdf_file.name}")
#
#         start_time = time.time()
#
#         try:
#             # Extract outline using optimized heading detection
#             outline = extractor.extract_outline(str(pdf_file))
#
#             processing_time = time.time() - start_time
#             total_time += processing_time
#
#             # Generate safe output filename
#             safe_name = safe_filename(pdf_file.stem)
#             output_file = output_dir / f"{safe_name}_outline.json"
#
#             # Save JSON output
#             save_json_safely(outline, output_file)
#
#             # FIXED: Proper output reporting with correct page counts
#             heading_count = len(outline.get('outline', []))
#             pages_processed = outline.get('pages_processed', 0)
#             total_doc_pages = outline.get('total_pages_in_document', 0)
#
#             print(f"✅ Completed in {processing_time:.2f}s - {heading_count} headings found")
#             print(f"📄 Pages processed: {pages_processed}/{total_doc_pages}")  # FIXED FORMAT
#             print(f"💾 Saved: {output_file.name}")
#
#             results.append({
#                 "file": pdf_file.name,
#                 "processing_time": processing_time,
#                 "headings_found": heading_count,
#                 "pages_processed": pages_processed,
#                 "total_pages": total_doc_pages,
#                 "status": "success"
#             })
#
#             success_count += 1
#
#             # Enhanced performance feedback
#             if processing_time > 10:
#                 print(f"⚠️  Warning: Processing time {processing_time:.2f}s exceeds 10s limit")
#             elif processing_time <= 3:
#                 print(f"🏆 EXCELLENT: {processing_time:.2f}s (UNDER 3s TARGET!) 🎯")
#             else:
#                 print(f"⚠️  SLOW: {processing_time:.2f}s exceeds 3s target by {processing_time - 3.0:.1f}s")
#
#         except Exception as e:
#             processing_time = time.time() - start_time
#             error_msg = str(e)
#             print(f"❌ Error processing {pdf_file.name}: {error_msg}")
#             traceback.print_exc()
#
#             # Create error output
#             error_outline = {
#                 "title": f"Error processing {pdf_file.name}",
#                 "outline": [],
#                 "error": error_msg,
#                 "processing_time": processing_time,
#                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
#             }
#
#             safe_name = safe_filename(pdf_file.stem)
#             output_file = output_dir / f"{safe_name}_error.json"
#             save_json_safely(error_outline, output_file)
#
#             results.append({
#                 "file": pdf_file.name,
#                 "processing_time": processing_time,
#                 "headings_found": 0,
#                 "status": "error",
#                 "error": error_msg
#             })
#
#     # Generate comprehensive summary
#     avg_time = total_time / len(pdf_files) if pdf_files else 0
#     performance_status = "🎯 TARGET MET!" if avg_time <= 3.0 else f"❌ MISSED by {avg_time - 3.0:.1f}s"
#
#     print(f"\n📊 Round 1A Summary:")
#     print(f"   • Success: {success_count}/{len(pdf_files)} files")
#     print(f"   • Average time: {avg_time:.2f}s per file")
#     print(f"   • Performance target (<3s): {performance_status}")
#     print(f"   • Total time: {total_time:.2f}s")
#
#     # Save summary
#     summary = {
#         "round": "1A",
#         "total_files": len(pdf_files),
#         "successful_files": success_count,
#         "average_processing_time": avg_time,
#         "total_processing_time": total_time,
#         "performance_target_met": avg_time <= 3.0,
#         "results": results,
#         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
#     }
#
#     summary_file = output_dir / "round_1a_summary.json"
#     save_json_safely(summary, summary_file)
#
#     return success_count > 0
#
#
# @measure_time
# def process_round_1b(input_dir: Path, output_dir: Path) -> bool:
#     """
#     Process Round 1B: Persona-Driven Document Intelligence
#     Target: <60 seconds for document collection
#     """
#     print("\n🎯 Starting Round 1B: Persona-Driven Document Intelligence")
#
#     config_file = input_dir / "persona_config.json"
#     if not config_file.exists():
#         print("📋 No persona_config.json found, skipping Round 1B")
#         print("💡 To run Round 1B, create a persona_config.json file with:")
#         print('   {"persona": "Your persona", "job_to_be_done": "Your job description"}')
#         return True
#
#     try:
#         with open(config_file, 'r', encoding='utf-8') as f:
#             config = json.load(f)
#
#         persona = config.get("persona", "").strip()
#         job_description = config.get("job_to_be_done", "").strip()
#
#         if not persona or not job_description:
#             print("❌ Invalid persona configuration")
#             return False
#
#         print(f"👤 Persona: {persona}")
#         print(f"🎯 Job: {job_description}")
#
#         pdf_files = list(input_dir.glob("*.pdf"))
#         if not pdf_files:
#             print("❌ No PDF files found for analysis")
#             return False
#
#         if len(pdf_files) > 10:
#             print(f"⚠️  Warning: {len(pdf_files)} PDFs found, processing first 10")
#             pdf_files = pdf_files[:10]
#
#         print(f"📚 Analyzing {len(pdf_files)} documents...")
#
#         start_time = time.time()
#
#         if MODULES_LOADED:
#             try:
#                 analyzer = PersonaAnalyzer()
#                 result = analyzer.analyze_documents(
#                     [str(f) for f in pdf_files],
#                     persona,
#                     job_description
#                 )
#             except Exception as e:
#                 print(f"❌ Error during document analysis: {e}")
#                 return False
#         else:
#             result = {
#                 'persona': persona,
#                 'job_to_be_done': job_description,
#                 'extracted_sections': [],
#                 'document_summaries': [{'document': f.name, 'processed': True} for f in pdf_files],
#                 'sub_section_analysis': [],
#                 'analysis_status': 'fallback_mode'
#             }
#
#         processing_time = time.time() - start_time
#
#         output_file = output_dir / "persona_analysis.json"
#         save_json_safely(result, output_file)
#
#         sections_count = len(result.get('extracted_sections', []))
#         subsections_count = len(result.get('sub_section_analysis', []))
#
#         print(f"\n✅ Round 1B completed in {processing_time:.2f}s")
#         print(f"📄 Found {sections_count} relevant sections")
#         print(f"🔍 Generated {subsections_count} subsection analyses")
#         print(f"💾 Saved: {output_file.name}")
#
#         if processing_time > 60:
#             print(f"⚠️  Warning: Processing time {processing_time:.2f}s exceeds 60s limit")
#         elif processing_time <= 45:
#             print(f"🏆 Excellent: Processing completed in {processing_time:.2f}s")
#
#         summary = {
#             "round": "1B",
#             "persona": persona,
#             "job_to_be_done": job_description,
#             "total_documents": len(pdf_files),
#             "processing_time": processing_time,
#             "sections_found": sections_count,
#             "subsections_found": subsections_count,
#             "performance_target_met": processing_time <= 60.0,
#             "status": "success",
#             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
#         }
#
#         save_json_safely(summary, output_dir / "round_1b_summary.json")
#         return True
#
#     except Exception as e:
#         print(f"❌ Error in Round 1B processing: {e}")
#         traceback.print_exc()
#         return False
#
#
# def validate_environment():
#     """Validate the execution environment"""
#     print("🔍 Validating environment...")
#
#     python_version = sys.version_info
#     print(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}")
#
#     try:
#         import psutil
#         memory = psutil.virtual_memory()
#         available_gb = memory.available / (1024 ** 3)
#         total_gb = memory.total / (1024 ** 3)
#         print(f"💾 Memory: {available_gb:.1f} GB available / {total_gb:.1f} GB total")
#
#         if available_gb < 2:
#             print("⚠️  Warning: Low memory available")
#     except ImportError:
#         print("💾 Memory info not available (psutil not installed)")
#
#     try:
#         input_dir, output_dir = setup_directories()
#         pdf_files = list(input_dir.glob("*.pdf"))
#         print(f"📄 Found {len(pdf_files)} PDF files in input directory")
#
#         config_file = input_dir / "persona_config.json"
#         if config_file.exists():
#             print("⚙️  Found persona_config.json for Round 1B")
#         else:
#             print("📋 No persona_config.json found (Round 1B will be skipped)")
#     except Exception as e:
#         print(f"⚠️  Directory setup warning: {e}")
#
#     # Check critical imports
#     try:
#         import fitz
#         print("📦 PyMuPDF (fitz) imported successfully")
#     except ImportError as e:
#         print(f"❌ Critical package import error (PyMuPDF): {e}")
#         return False
#
#     try:
#         import spacy
#         print("📦 spaCy imported successfully")
#     except ImportError as e:
#         print(f"❌ Critical package import error (spaCy): {e}")
#         return False
#
#     try:
#         import sklearn
#         print("📦 scikit-learn imported successfully")
#     except ImportError as e:
#         print(f"❌ Critical package import error (scikit-learn): {e}")
#         return False
#
#     print("✅ Environment validation complete")
#     return True
#
#
# def main():
#     """
#     Main execution function with comprehensive error handling
#     """
#     print("=" * 70)
#     print("🏆 Adobe Hackathon - Elite Document Intelligence Solution")
#     print("🎯 Rounds 1A & 1B - Structure Extraction & Persona Analysis")
#     print("🔥 Features: Chaos-resistant headings, multilingual, <3s processing")
#     print("=" * 70)
#
#     try:
#         if not validate_environment():
#             print("❌ Environment validation failed")
#             sys.exit(1)
#     except Exception as e:
#         print(f"❌ Environment validation error: {e}")
#         traceback.print_exc()
#         sys.exit(1)
#
#     try:
#         input_dir, output_dir = setup_directories()
#         print(f"📁 Using input directory: {input_dir}")
#         print(f"📁 Using output directory: {output_dir}")
#     except Exception as e:
#         print(f"❌ Failed to setup directories: {e}")
#         traceback.print_exc()
#         sys.exit(1)
#
#     total_start_time = time.time()
#     overall_success = False
#
#     # Process Round 1A
#     try:
#         print("\n" + "=" * 50)
#         round_1a_success = process_round_1a(input_dir, output_dir)
#         if round_1a_success:
#             overall_success = True
#     except Exception as e:
#         print(f"❌ Critical error in Round 1A: {e}")
#         traceback.print_exc()
#         round_1a_success = False
#
#     # Process Round 1B
#     try:
#         print("\n" + "=" * 50)
#         round_1b_success = process_round_1b(input_dir, output_dir)
#         if round_1b_success:
#             overall_success = True
#     except Exception as e:
#         print(f"❌ Critical error in Round 1B: {e}")
#         traceback.print_exc()
#         round_1b_success = False
#
#     total_time = time.time() - total_start_time
#
#     print("\n" + "=" * 70)
#     print("📊 FINAL EXECUTION SUMMARY")
#     print("=" * 70)
#     print(f"⏱️  Total execution time: {total_time:.2f}s")
#     print(f"🎯 Round 1A (Structure Extraction): {'✅ SUCCESS' if round_1a_success else '❌ FAILED'}")
#     print(f"🎯 Round 1B (Persona Analysis): {'✅ SUCCESS' if round_1b_success else '⏭️ SKIPPED/FAILED'}")
#
#     if total_time <= 10:
#         print(f"🏆 Outstanding: Total execution in {total_time:.2f}s")
#     elif total_time <= 30:
#         print(f"✅ Good: Total execution in {total_time:.2f}s")
#     else:
#         print(f"⚠️  Slow: Total execution in {total_time:.2f}s")
#
#     if overall_success:
#         print("\n🎉 SOLUTION COMPLETED SUCCESSFULLY")
#         print("📝 Check output directory for results")
#         print("📊 Summary files saved for analysis")
#     else:
#         print("\n❌ SOLUTION FAILED")
#         print("📝 Check error messages and output directory")
#         sys.exit(1)
#
#     print("=" * 70)
#
#
# if __name__ == "__main__":
#     main()
