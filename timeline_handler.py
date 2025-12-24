import logging
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import TypedDict, Any, cast

from indralib.indra_time import IndraTime

class TimelineEvent(TypedDict):
    date_text: str
    event_description: str
    indra_str: str

class TimelineExtractor:
    def __init__(self, model_name: str = "google/gemma-7b-it", device: str | None = None):
        self.log = logging.getLogger("TimelineExtractor")
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.tokenizer = None
        self.model = None
        self.log.info(f"TimelineExtractor initialized using device: {self.device}")

    def _resolve_device(self, device_name: str | None) -> str:
        if device_name is None or device_name == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            elif torch.xpu.is_available():
                return 'xpu'
            else:
                return 'cpu'
        return device_name

    def _load_model(self):
        if self.model is None:
            self.log.info(f"Loading model {self.model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                    device_map=self.device
                )
                self.log.info("Model loaded successfully.")
            except Exception as e:
                self.log.error(f"Failed to load model {self.model_name}: {e}")
                self.model = None

    def normalize_time(self, date_text: str) -> str | None:
        """
        Normalize natural language date strings into IndraTime format.
        """
        date_text = date_text.strip()
        
        # Simple/Direct IndraTime parsing first
        # This handles YYYY, YYYY-MM, YYYY-MM-DD, etc. as well as some BC/BP formats if supported
        # We try to use IndraTime's parsing capabilities first.
        # However, IndraTime.string_time_to_julian parses to float tuple. 
        # We want the canonical string representation.
        
        # First, explicit regex cleanups/transforms to match IndraTime expectations
        
        # 1. "41 thousand years ago" -> "41 ka BP"
        #    "40 to 42 ka" -> "42 ka BP - 40 ka BP" (Note: range order)
        
        # Regex for "N thousand years ago" -> "N ka BP"
        m = re.match(r"([\d\.]+)\s+thousand\s+years\s+ago", date_text, re.IGNORECASE)
        if m:
            return f"{m.group(1)} ka BP"

        # Regex for "N ka" -> "N ka BP"
        m = re.match(r"([\d\.]+)\s+ka\b", date_text, re.IGNORECASE)
        if m and "BP" not in date_text.upper():
             return f"{m.group(1)} ka BP"

        # Regex for "in January 1962" -> "January 1962" (remove preposition)
        if re.match(r"^(in|on|at|during|from)(\s+the)?\s+", date_text, re.IGNORECASE):
            date_text = re.sub(r"^(in|on|at|during|from)(\s+the)?\s+", "", date_text, flags=re.IGNORECASE)

        # Regex for "days of October 16-28, 1962" -> "1962-10-16 - 1962-10-28"
        # This is complex, maybe rely on LLM to give us better intermediate format or specific logic?
        # Let's try to map explicit patterns mentioned in the request.
        
        # "January 18, 1962" -> "1962-01-18" is standard.
        
        # "Jan 1962" -> "1962-01"
        
        # Check if IndraTime can handle it directly or via some simple preprocessing
        # We can implement a trial-parse-reformat loop using IndraTime logic if available,
        # but since IndraTime converts to Julian, we might lose precision if we blindly convert back.
        # But for 'normalization' extracting the string representation from Julian is exactly what is asked 
        # (mostly, except preserving precision).
        # Actually IndraTime format descriptions suggests we should aim for strings like "YYYY-MM-DD".
        
        # Let's try basic manual formatting for common cases if simple pass-through fails or is ambiguous.
        
        # Special case: BC ranges "58 BC to 50 BC" -> "58 BC - 50 BC"
        if " to " in date_text and "BC" in date_text:
            return date_text.replace(" to ", " - ")
            
        # "in the 1st century BC" -> "100 BC - 1 BC"
        m = re.match(r"(\d+)(?:st|nd|rd|th)?\s+century\s+BC", date_text, re.IGNORECASE)
        if m:
            century = int(m.group(1))
            start = century * 100
            end = (century - 1) * 100 + 1
            return f"{start} BC - {end} BC"

        # "Month YYYY" -> "YYYY-MM"
        months = {
            "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
            "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12",
            "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06",
            "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"
        }
        
        # "Month DD, YYYY" -> "YYYY-MM-DD"
        m = re.match(r"^([a-zA-Z]+)\s+(\d{1,2}),\s+(\d{4})$", date_text, re.IGNORECASE)
        if m:
            mon_str = m.group(1).lower()
            day = m.group(2).zfill(2)
            year = m.group(3)
            if mon_str in months:
                return f"{year}-{months[mon_str]}-{day}"
                
        # "Month DD, YYYY" -> "YYYY-MM-DD"
        m = re.match(r"^([a-zA-Z]+)\s+(\d{1,2}),\s+(\d{4})$", date_text, re.IGNORECASE)
        if m:
            mon_str = m.group(1).lower()
            day = m.group(2).zfill(2)
            year = m.group(3)
            if mon_str in months:
                return f"{year}-{months[mon_str]}-{day}"
                
        # "Month YYYY" -> "YYYY-MM"
        m = re.match(r"^([a-zA-Z]+)\s+(\d{3,4})$", date_text, re.IGNORECASE)
        if m:
            mon_str = m.group(1).lower()
            year = m.group(2)
            if mon_str in months:
                return f"{year}-{months[mon_str]}"
        
        # Ranges "YYYY-YYYY" -> "YYYY - YYYY"
        m = re.match(r"^(\d{3,4})[-\s]+(\d{3,4})$", date_text)
        if m:
            return f"{m.group(1)} - {m.group(2)}"
        
        # Range with spaces "YYYY - YYYY" -> preserve (or ensure spacing)
        # IndraTime preferred range format check?
                
        return date_text # Return as-is if no specific rule matches

    def extract_from_text(self, text: str) -> list[TimelineEvent]:
        """
        Extract timeline events from text using LLM and Regex.
        
        Algorithm:
        1. Search for explicit date patterns (Regex) to anchor.
        2. Use LLM to extract events in structured format.
        """
        self._load_model()
        if self.model is None or self.tokenizer is None:
            self.log.error("Model not available for extraction.")
            return []

        # Chunking: Gemma has context limits. Split by paragraphs or chunks.
        # Simple approach: one big chunk (up to limit) or split by newlines.
        # For this implementation, let's truncate to reasonable size (e.g., 2048 tokens approx)
        # or handle chunks loop.
        
        max_len = 4000 # Characters, rough approximation
        chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
        
        all_events = []
        
        for i, chunk in enumerate(chunks):
            prompt = f"""
Analyze the following text and identify all historical events with their associated dates or time periods.
Output the results in strict JSON format as a list of objects.
Each object must have:
- "date_text": The exact time string found in the text.
- "event_description": A concise summary of the event (10-20 words).

Do not include any other text. Do not use Markdown formatting (no backticks).
Ensure proper JSON escaping:
- Use standard straight quotes (") for JSON structure.
- Escape internal quotes as \\".
- Escape newlines and tabs within strings as \\n and \\t.
- Do NOT use typographic/smart quotes (“ ”).
If no events are found, return empty list [].

Text:
{chunk}

JSON:
"""
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=500,
                    do_sample=True, # Allow some creativity/variability
                    temperature=0.2 # But keep it focused
                )
            
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON part
            if "JSON:" in output_text:
                post_json = output_text.split("JSON:")[-1]
            else:
                post_json = output_text
            
            # Cleanup potential Markdown
            post_json = post_json.replace("```json", "").replace("```", "")
            
            json_objects = []
            
            # Attempt 1: Standard JSON parse
            start = post_json.find('[')
            end = post_json.rfind(']')
            if start != -1 and end != -1:
                json_str = post_json[start:end+1]
                try:
                    data = json.loads(json_str, strict=False)
                    if isinstance(data, list):
                        json_objects = data
                except json.JSONDecodeError:
                    # self.log.warning(f"Initial JSON parse failed, attempting heuristic repair.")
                    # Fallback: Heuristic extraction
                    json_objects = self._heuristic_extract(post_json)

            if len(json_objects) > 0:
                self.log.info(f"Extracted {len(json_objects)} events from chunk {i+1}.")
            else:
                self.log.warning("No events extracted from chunk {i+1}.")
            
            for item in json_objects:
                if 'date_text' in item and 'event_description' in item:
                    # Normalize time
                    norm_time = self.normalize_time(item['date_text'])
                    if norm_time:
                        event =   {
                            "date_text": item['date_text'],
                            "event_description": item['event_description'],
                            "indra_str": norm_time
                        }
                        self.log.info(f"Extracted event: {event}")
                        all_events.append(event)
                    else:
                        self.log.warning(f"Invalid event (missing norm_time): {item}")
                else:
                    self.log.warning(f"Invalid event (missing date_text or event_description): {item}")
        return all_events

    def _heuristic_extract(self, text: str) -> list[dict]:
        """
        Extract JSON objects manually using regex when standard parsing fails.
        Handles unescaped quotes and other common LLM artifacts.
        """
        extracted = []
        # Find everything between { and } that looks like an object
        # We assume no nested objects for this specific schema
        object_pattern = re.compile(r"\{.*?\}", re.DOTALL)
        matches = object_pattern.findall(text)
        
        for m in matches:
            # Clean up smart quotes in the block first
            m_clean = m.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
            
            # Extract date_text
            # Look for "date_text": "..." 
            # We assume date_text doesn't contain unescaped quotes usually
            date_match = re.search(r'"date_text"\s*:\s*"(.*?)"', m_clean)
            
            # Extract event_description
            # This is the hard one. It might end with " } or ",
            # We capture everything until the lookahead of the closing quote followed by comma or brace
            desc_match = re.search(r'"event_description"\s*:\s*"(.*)"\s*\}', m_clean, re.DOTALL)
            
            if not desc_match:
                 # Try matching until ", (next field?) - though here description is usually last
                 desc_match = re.search(r'"event_description"\s*:\s*"(.*)"\s*,', m_clean, re.DOTALL)
            
            if date_match and desc_match:
                d_text = date_match.group(1).strip()
                d_desc = desc_match.group(1).strip()
                # If there are residual unescaped quotes in d_desc, we might want to clean them?
                # But mostly we just want the text.
                extracted.append({
                    "date_text": d_text,
                    "event_description": d_desc
                })
                
        return extracted
