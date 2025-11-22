import re
import logging

class SearchTools:
    def __init__(self):
        self.log = logging.getLogger("SearchTools")

    @staticmethod
    def match(text: str, keys: list[str]) -> bool:
        """
        Checks if the text matches the search keys.
        Keys support:
        - 'word': matches if 'word' is in text (case-insensitive)
        - '!word': matches if 'word' is NOT in text
        - 'w1|w2': matches if 'w1' OR 'w2' is in text
        - 'word*': matches if a word starting with 'word' is in text
        - '*word': matches if a word ending with 'word' is in text
        - '*word*': matches if 'word' is in text (substring match)
        
        All top-level keys in the list must match (AND logic).
        """
        s_text: str = text.lower()
        found: bool = False
        
        # First pass: Check positive matches (AND logic)
        # If there are no positive matches required (only negatives), we assume True initially
        # But usually if keys is empty, we return True? Or False?
        # If keys is empty, usually means "show all", so True.
        if not keys:
            return True

        has_positive_constraints = False
        for key in keys:
            if key.startswith("!"):
                continue
            
            has_positive_constraints = True
            or_keys = key.split("|")
            or_found = False
            for or_key in or_keys:
                if SearchTools._check_single_key(s_text, or_key):
                    or_found = True
                    break
            
            if not or_found:
                return False
        
        # If we had positive constraints and passed all of them, found is effectively True.
        # If we had NO positive constraints (only negatives), we start with True.
        
        # Second pass: Check negative matches (NOT logic)
        for key in keys:
            if not key.startswith("!"):
                continue
            
            neg_key = key[1:]
            if SearchTools._check_single_key(s_text, neg_key):
                return False
                
        return True

    @staticmethod
    def extract_highlight_terms(keys: list[str]) -> list[str]:
        """
        Extracts terms that should be highlighted from the search keys.
        Ignores negated terms.
        Splits OR terms.
        Strips wildcards for substring matching.
        """
        highlight_terms = []
        for key in keys:
            if key.startswith("!"):
                continue
            
            or_keys = key.split("|")
            for sub_key in or_keys:
                # Strip wildcards
                clean_key = sub_key.replace("*", "")
                if clean_key:
                    highlight_terms.append(clean_key)
        return highlight_terms

    @staticmethod
    def _check_single_key(text: str, key: str) -> bool:
        if key.startswith("*"):
            key_pattern = key[1:]
        else:
            key_pattern = r"\b" + key
            
        if key.endswith("*"):
            key_pattern = key_pattern[:-1]
        else:
            key_pattern += r"\b"
            
        # Replace internal * with .* for regex
        key_pattern = key_pattern.lower().replace("*", r".*")
        
        if re.search(key_pattern, text):
            return True
        return False
