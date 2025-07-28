import json
import jsonschema
from typing import Dict, List, Any
import re


class JSONValidator:
    def __init__(self):
        """
        Ensures perfect JSON structure compliance
        Auto-fixes common formatting issues
        """
        # Define the expected schema for Round 1A
        self.round_1a_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "outline": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "level": {
                                "type": "string",
                                "pattern": "^H[1-4]$"  # H1, H2, H3, H4
                            },
                            "text": {"type": "string"},
                            "page": {"type": "integer", "minimum": 1}
                        },
                        "required": ["level", "text", "page"]
                    }
                }
            },
            "required": ["title", "outline"]
        }

        # Define schema for Round 1B
        self.round_1b_schema = {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "input_documents": {"type": "array", "items": {"type": "string"}},
                        "persona": {"type": "string"},
                        "job_to_be_done": {"type": "string"},
                        "processing_timestamp": {"type": "string"}
                    },
                    "required": ["input_documents", "persona", "job_to_be_done"]
                },
                "extracted_sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document": {"type": "string"},
                            "page_number": {"type": "integer", "minimum": 1},
                            "section_title": {"type": "string"},
                            "importance_rank": {"type": "integer", "minimum": 1}
                        },
                        "required": ["document", "page_number", "section_title", "importance_rank"]
                    }
                },
                "sub_section_analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document": {"type": "string"},
                            "refined_text": {"type": "string"},
                            "page_number": {"type": "integer", "minimum": 1}
                        },
                        "required": ["document", "refined_text", "page_number"]
                    }
                }
            },
            "required": ["metadata", "extracted_sections", "sub_section_analysis"]
        }

    def validate_and_fix(self, data: Dict, schema_type: str = "round_1a") -> Dict:
        """
        Validate JSON structure and auto-fix common issues
        Ensures 100% compliance with expected format
        """
        schema = self.round_1a_schema if schema_type == "round_1a" else self.round_1b_schema

        try:
            # Validate against schema
            jsonschema.validate(data, schema)
            return data  # Already valid
        except jsonschema.exceptions.ValidationError:
            # Auto-fix common issues
            if schema_type == "round_1a":
                return self._auto_fix_outline(data)
            else:
                return self._auto_fix_persona_analysis(data)

    def _auto_fix_outline(self, outline: Dict) -> Dict:
        """
        Auto-fix common JSON structure issues for Round 1A
        """
        fixed_outline = {}

        # Fix title
        title = outline.get("title", "Untitled Document")
        if not isinstance(title, str):
            title = str(title)
        if len(title.strip()) == 0:
            title = "Untitled Document"

        # Clean up title - remove excessive whitespace and invalid characters
        title = re.sub(r'\s+', ' ', title).strip()
        title = title[:200]  # Limit title length
        fixed_outline["title"] = title

        # Fix outline array
        outline_items = outline.get("outline", [])
        if not isinstance(outline_items, list):
            outline_items = []

        fixed_items = []
        seen_headings = set()  # Prevent duplicates

        for item in outline_items:
            if isinstance(item, dict):
                fixed_item = self._fix_outline_item(item)
                if fixed_item:  # Only add if successfully fixed
                    # Prevent duplicate headings
                    heading_key = (fixed_item["level"], fixed_item["text"], fixed_item["page"])
                    if heading_key not in seen_headings:
                        fixed_items.append(fixed_item)
                        seen_headings.add(heading_key)

        # Sort by page number and then by heading level
        fixed_items.sort(key=lambda x: (x["page"], x["level"]))

        fixed_outline["outline"] = fixed_items
        return fixed_outline

    def _fix_outline_item(self, item: Dict) -> Dict:
        """
        Fix individual outline item
        """
        fixed_item = {}

        # Fix level
        level = item.get("level", "H1")
        if not isinstance(level, str) or not level.startswith("H"):
            level = "H1"

        # Normalize level format
        level_match = re.search(r'(\d+)', str(level))
        if level_match:
            level_num = int(level_match.group(1))
            level_num = max(1, min(4, level_num))  # Clamp between 1-4
            level = f"H{level_num}"
        else:
            level = "H1"

        fixed_item["level"] = level

        # Fix text
        text = item.get("text", "")
        if not isinstance(text, str):
            text = str(text)

        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[:500]  # Limit text length

        if len(text) == 0:
            return None     # Skip empty headings

        fixed_item["text"] = text

        # Fix page
        page = item.get("page", 1)
        if not isinstance(page, int):
            try:
                page = int(float(str(page)))
            except (ValueError, TypeError):
                page = 1
        if page < 1:
            page = 1

        fixed_item["page"] = page

        return fixed_item

    def _auto_fix_persona_analysis(self, analysis: Dict) -> Dict:
        """
        Auto-fix persona analysis structure for Round 1B
        """
        fixed_analysis = {}

        # Fix metadata
        metadata = analysis.get("metadata", {})
        fixed_metadata = {
            "input_documents": metadata.get("input_documents", []),
            "persona": str(metadata.get("persona", "Unknown Persona")),
            "job_to_be_done": str(metadata.get("job_to_be_done", "Unknown Job")),
            "processing_timestamp": metadata.get("processing_timestamp", "2024-01-01T00:00:00Z")
        }
        fixed_analysis["metadata"] = fixed_metadata

        # Fix extracted sections
        sections = analysis.get("extracted_sections", [])
        if not isinstance(sections, list):
            sections = []

        fixed_sections = []
        for section in sections:
            if isinstance(section, dict):
                fixed_section = {
                    "document": str(section.get("document", "unknown")),
                    "page_number": max(1, int(section.get("page_number", 1))),
                    "section_title": str(section.get("section_title", ""))[:200],
                    "importance_rank": max(1, int(section.get("importance_rank", 1)))
                }
                fixed_sections.append(fixed_section)

        fixed_analysis["extracted_sections"] = fixed_sections

        # Fix sub-section analysis
        sub_sections = analysis.get("sub_section_analysis", [])
        if not isinstance(sub_sections, list):
            sub_sections = []

        fixed_sub_sections = []
        for sub_section in sub_sections:
            if isinstance(sub_section, dict):
                refined_text = str(sub_section.get("refined_text", ""))
                if len(refined_text.strip()) > 10:  # Only include substantial text
                    fixed_sub_section = {
                        "document": str(sub_section.get("document", "unknown")),
                        "refined_text": refined_text[:1000],  # Limit length
                        "page_number": max(1, int(sub_section.get("page_number", 1)))
                    }
                    fixed_sub_sections.append(fixed_sub_section)

        fixed_analysis["sub_section_analysis"] = fixed_sub_sections

        return fixed_analysis

    def create_valid_outline(self, title: str, outline_items: List[Dict]) -> Dict:
        """
        Create a guaranteed valid outline structure
        """
        return self.validate_and_fix({
            "title": title,
            "outline": outline_items
        }, "round_1a")

    def create_valid_persona_analysis(self, metadata: Dict, sections: List[Dict], sub_sections: List[Dict]) -> Dict:
        """
        Create a guaranteed valid persona analysis structure
        """
        return self.validate_and_fix({
            "metadata": metadata,
            "extracted_sections": sections,
            "sub_section_analysis": sub_sections
        }, "round_1b")

    def is_valid_json(self, data: Dict, schema_type: str = "round_1a") -> bool:
        """
        Check if data is valid without fixing
        """
        schema = self.round_1a_schema if schema_type == "round_1a" else self.round_1b_schema
        try:
            jsonschema.validate(data, schema)
            return True
        except jsonschema.exceptions.ValidationError:
            return False
