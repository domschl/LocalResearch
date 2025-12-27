import logging
import re
import os
import json
import time
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Metal: uv pip install mlx-lm
try:
    import mlx_lm
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Metal: CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python
# Vulkan: CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install --reinstall llama-cpp-python      (requires vulkan-devel)

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

from typing import TypedDict, Any, cast, Literal

from indralib.indra_time import IndraTime
from perf_stats import PerfStats

class TimelineEvent(TypedDict):
    date_text: str
    event_description: str
    indra_str: str

class TimelineExtractor:
    def __init__(self, model_name: str = "google/gemma-2b-it", backend: Literal["pytorch", "mlx", "llama.cpp"] = "pytorch", device: str | None = None, perf_stats: PerfStats | None = None):
        self.log = logging.getLogger("TimelineExtractor")
        self.model_name = model_name
        self.backend = backend
        self.device = self._resolve_device(device)
        self.tokenizer = None
        self.model = None
        self.perf_stats = perf_stats
        self.pt_gen = 0.0
        self.mlx_gen = 0.0
        self.llama_cpp_gen = 0.0
        
        self._validate_backend()
        self.log.info(f"TimelineExtractor initialized using backend: {self.backend}, device: {self.device}")

    def _validate_backend(self):
        if self.backend == "pytorch" and not TRANSFORMERS_AVAILABLE:
            raise ImportError("backend='pytorch' requires 'transformers' and 'torch' libraries.")
        if self.backend == "mlx" and not MLX_AVAILABLE:
            raise ImportError("backend='mlx' requires 'mlx-lm' library.")
        if self.backend == "llama.cpp" and not LLAMA_CPP_AVAILABLE:
            raise ImportError("backend='llama.cpp' requires 'llama-cpp-python' library.")


    def _resolve_device(self, device_name: str | None) -> str:
        if device_name is None or device_name == 'auto':
            if TRANSFORMERS_AVAILABLE:
                if torch.cuda.is_available():
                    return 'cuda'
                elif torch.backends.mps.is_available():
                    return 'mps'
                elif torch.xpu.is_available():
                    return 'xpu'
                else:
                    return 'cpu'
            else:
                # Without torch, we can't easily auto-detect for PyTorch purposes.
                # For MLX, it defaults to GPU if available.
                # For llama.cpp, we set n_gpu_layers=-1 if we think we are on Mac/GPU.
                # Let's default to a safe value or just 'auto' string if backend handles it.
                return 'cpu' # Fallback
        return device_name


    def _load_model(self):
        if self.model is not None:
            return

        self.log.info(f"Loading model {self.model_name} with backend {self.backend}...")
        try:
            if self.backend == "pytorch":
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                    device_map=self.device
                )
                self.log.info(f"Model loaded successfully. Class: {type(self.model)}")
            elif self.backend == "mlx":
                self.model, self.tokenizer = mlx_lm.load(self.model_name)
            elif self.backend == "llama.cpp":
                # For llama.cpp, model_name should ideally be a path to a GGUF file
                # But if it's a HF repo, we might need to handle it differently or expect user to pass local path.
                # For now, let's assume model_name can be a path.
                # If it is a repo, Llama(...) might try to download if properly configured, but usually requires explicit download.
                # Let's assume the user knows what they are doing or we provide a way to download.
                # However, Llama class creates the 'model' instance which doubles as the valid object.
                n_gpu_layers = -1 if self.device in ['mps', 'cuda'] else 0
                self.model = Llama(
                    model_path=self.model_name,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                    n_ctx=4096 
                )
                self.tokenizer = self.model # Llama object handles tokenization

            self.log.info("Model loaded successfully.")
        except Exception as e:
            self.log.error(f"Failed to load model {self.model_name}: {e}")
            self.model = None
            raise e

    def _generate(self, messages: list[dict]) -> str:
        if self.backend == "pytorch":
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            self.log.info(f"Prompt sent to PyTorch backend:\n{prompt}")
            
            if self.perf_stats:
                start_time = time.time()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=2048, # Increased for JSON
                    do_sample=True,
                    temperature=0.1,
                )
            decoded = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            if self.perf_stats:
                dt = time.time() - start_time
                if self.pt_gen == 0.0:
                    self.pt_gen = dt
                else:
                    self.pt_gen = (self.pt_gen * 4 + dt) / 5.0
                self.perf_stats.add_perf(f"{self.model_name}_{self.backend}_{self.device}_tl_extract_generate", self.pt_gen)
            return decoded
            
        elif self.backend == "mlx":
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback if tokenizer doesn't support chat template (unlikely for modern HF tokenizers)
                # But MLX tokenizer might be a slim wrapper? check docs. 
                # Usually mlx_lm.load returns a HF tokenizer.
                prompt = messages[-1]["content"] 

            if self.perf_stats:
                start_time = time.time()
            output = mlx_lm.generate(
                self.model, 
                self.tokenizer, 
                prompt=prompt, 
                max_tokens=2048, 
                verbose=False
            )
            if self.perf_stats:
                dt = time.time() - start_time
                if self.mlx_gen == 0.0:
                    self.mlx_gen = dt
                else:
                    self.mlx_gen = (self.mlx_gen * 4 + dt) / 5.0
                self.perf_stats.add_perf(f"{self.model_name}_{self.backend}_{self.device}_tl_extract_generate", self.mlx_gen)
            return output
            
        elif self.backend == "llama.cpp":
            # Use create_chat_completion for proper template handling
            if self.perf_stats:
                start_time = time.time()
            output = self.model.create_chat_completion(
                messages=messages,
                  max_tokens=2048,
                temperature=0.2,
                # stop=["<eos>"] # Handle automatically by chat format usually
            )
            if self.perf_stats:
                dt = time.time() - start_time
                if self.llama_cpp_gen == 0.0:
                    self.llama_cpp_gen = dt
                else:
                    self.llama_cpp_gen = (self.llama_cpp_gen * 4 + dt) / 5.0
                self.perf_stats.add_perf(f"{self.model_name}_{self.backend}_{self.device}_tl_extract_generate", self.llama_cpp_gen)
            return output['choices'][0]['message']['content']
        
        return ""

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
        # For Llama.cpp, tokenizer is the model instance itself (or not None).
        if self.model is None:
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
            # Construct strict JSON prompt
            user_content = f"""
Analyze the following text and identify all historical events with their associated dates or time periods.
Output the results in strict JSON format as a list of objects.
Each object must have:
- "date_text": The exact time string found in the text.
- "event_description": A concise summary of the event (10-20 words).
If no events are found, return empty list [].

Do not include any other text. Do not use Markdown formatting (no backticks).
Ensure proper JSON escaping.

Text:
{chunk}

JSON:
"""
            messages = [{"role": "user", "content": user_content}]
            output_text = self._generate(messages)
            
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
                self.log.warning(f"No events extracted from chunk {i+1}.")
            
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
