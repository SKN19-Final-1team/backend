from typing import Dict, List, Set
import re

try:
    from flashtext import KeywordProcessor
    FLASHTEXT_AVAILABLE = True
except ImportError:
    FLASHTEXT_AVAILABLE = False

# Import vocabulary from app/rag/vocab
from app.rag.vocab.keyword_dict import (
    get_action_synonyms,
    get_card_name_synonyms,
    get_weak_intent_synonyms,
    PAYMENT_SYNONYMS,
)


class VocabRestorer:
    """
    Restores vocabulary terms that were incorrectly spaced during correction.
    
    Uses fuzzy matching to find and fix terms like:
    - "나라 사랑 카드" → "나라사랑카드"
    - "삼성 페이" → "삼성페이"
    - "K - 패스" → "K-패스"
    
    Also performs secondary correction using FlashText for misspelled terms.
    """
    
    def __init__(self):
        """Initialize vocabulary dictionaries."""
        self.card_name_synonyms = get_card_name_synonyms()
        self.payment_synonyms = PAYMENT_SYNONYMS
        self.action_synonyms = get_action_synonyms()
        
        # Build list of all terms to restore
        self._all_terms = set()
        self._build_term_list()
        
        # Initialize FlashText processor for secondary correction
        self._keyword_processor = None
        if FLASHTEXT_AVAILABLE:
            self._build_flashtext_processor()
    
    def _build_term_list(self):
        """Build comprehensive list of terms that should be restored."""
        # Card names (highest priority)
        for canonical, variants in self.card_name_synonyms.items():
            self._all_terms.add(canonical)
            for variant in variants:
                if variant and len(variant) >= 3:
                    self._all_terms.add(variant)
        
        # Payment methods (high priority)
        for canonical, variants in self.payment_synonyms.items():
            self._all_terms.add(canonical)
            for variant in variants:
                if variant and len(variant) >= 3:
                    self._all_terms.add(variant)
        
        # Important action terms (medium priority)
        # Only include specific multi-word terms
        important_actions = {
            '나라사랑카드', '분실도난', '일부결제금액이월약정', '리볼빙',
            '장기카드대출', '카드론', '단기카드대출', '현금서비스'
        }
        for canonical, variants in self.action_synonyms.items():
            if canonical in important_actions or len(canonical) >= 5:
                self._all_terms.add(canonical)
                for variant in variants:
                    if variant and len(variant) >= 5:
                        self._all_terms.add(variant)
    
    def _build_flashtext_processor(self):
        """Build FlashText KeywordProcessor for fast keyword replacement."""
        if not FLASHTEXT_AVAILABLE:
            return
        
        self._keyword_processor = KeywordProcessor(case_sensitive=False)
        
        # Add card names with common misspellings
        for canonical, variants in self.card_name_synonyms.items():
            # Map all variants to canonical form
            for variant in variants:
                if variant and variant != canonical:
                    self._keyword_processor.add_keyword(variant, canonical)
        
        # Add payment methods with common misspellings
        payment_misspellings = {
            '삼성페이': ['삼성 페이', '삼성 pay', '삼성pay'],
            '카카오페이': ['카카오 페이', '카카오 pay', '카카오pay'],
            '네이버페이': ['네이버 페이', '네이버 pay', '네이버pay'],
            '애플페이': ['애플 페이', '애플 pay', '애플pay'],
        }
        for canonical, misspellings in payment_misspellings.items():
            for misspelling in misspellings:
                self._keyword_processor.add_keyword(misspelling, canonical)
        
        # Add common financial term corrections
        financial_corrections = {
            '리볼빙': ['리벌빙', '리벌링', '리볼링', '리불빙'],
            '일부결제금액이월약정': ['일부결제금액 이월약정', '일부 결제금액 이월약정'],
            '장기카드대출': ['장기 카드 대출', '장기카드 대출'],
            '단기카드대출': ['단기 카드 대출', '단기카드 대출'],
            '현금서비스': ['현금 서비스'],
        }
        for canonical, misspellings in financial_corrections.items():
            for misspelling in misspellings:
                self._keyword_processor.add_keyword(misspelling, canonical)
    
    def _generate_spaced_variants(self, term: str) -> List[str]:
        """
        Generate possible spaced variants of a term.
        
        Examples:
        - "나라사랑카드" → ["나라 사랑 카드", "나라사랑 카드", "나라 사랑카드"]
        - "삼성페이" → ["삼성 페이"]
        - "K-패스" → ["K - 패스", "K 패스", "K패스"]
        """
        variants = []
        
        # Remove existing spaces and hyphens for base term
        base = term.replace(" ", "").replace("-", "")
        
        # Add space between every character (extreme case)
        all_spaced = " ".join(base)
        variants.append(all_spaced)
        
        # Add space every 2-3 characters (common case)
        for i in range(2, min(len(base), 6)):
            if len(base) > i:
                spaced = base[:i] + " " + base[i:]
                variants.append(spaced)
                # Try adding more spaces
                for j in range(i+2, len(base)):
                    double_spaced = base[:i] + " " + base[i:j] + " " + base[j:]
                    variants.append(double_spaced)
        
        # Handle hyphenated terms
        if "-" in term:
            # "K-패스" → "K - 패스", "K 패스"
            variants.append(term.replace("-", " - "))
            variants.append(term.replace("-", " "))
        
        # Handle terms with existing spaces
        if " " in term:
            # Keep original spacing
            variants.append(term)
        
        return list(set(variants))  # Remove duplicates
    
    def secondary_correction(self, text: str) -> str:
        """
        Perform secondary correction using FlashText for misspelled vocabulary terms.
        
        Args:
            text: Text to correct
            
        Returns:
            Text with vocabulary terms corrected
        """
        if not FLASHTEXT_AVAILABLE or not self._keyword_processor:
            return text
        
        # Use FlashText to replace misspelled terms
        corrected_text = self._keyword_processor.replace_keywords(text)
        
        return corrected_text
    
    def restore_terms(self, text: str) -> str:
        """
        Restore vocabulary terms that were incorrectly spaced.
        
        Args:
            text: Corrected text that may have incorrectly spaced terms
            
        Returns:
            Text with vocabulary terms restored to correct form
        """
        if not text:
            return text
        
        # Step 1: Secondary correction using FlashText (fast keyword replacement)
        if FLASHTEXT_AVAILABLE and self._keyword_processor:
            text = self.secondary_correction(text)
        
        # Step 2: Restore incorrectly spaced terms
        restored_text = text
        
        # Sort terms by length (longest first) to avoid partial replacements
        sorted_terms = sorted(self._all_terms, key=len, reverse=True)
        
        for term in sorted_terms:
            if not term or len(term) < 3:
                continue
            
            # Generate possible spaced variants
            spaced_variants = self._generate_spaced_variants(term)
            
            # Try to find and replace each variant
            for variant in spaced_variants:
                if variant in restored_text:
                    # Use word boundary to avoid partial matches
                    pattern = re.compile(r'\b' + re.escape(variant) + r'\b', re.IGNORECASE)
                    restored_text = pattern.sub(term, restored_text)
        
        return restored_text


