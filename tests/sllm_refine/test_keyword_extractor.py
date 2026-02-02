import unittest
from unittest.mock import patch, MagicMock
from app.llm.delivery.keyword_extractor import KeywordExtractor, STOPWORDS, CONDITIONAL_STOPWORDS

class TestKeywordExtractorFormatted(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.extractor = KeywordExtractor()
        cls.extractor._ensure_initialized()
        
        # Mock dictionary data to ensure consistent test results irrespective of external files
        cls.extractor._correction_map = {
            "삼송페이": "삼성페이", "결채": "결제", "연헤비": "연회비",
            "리벌빙": "리볼빙", "활부": "할부", "결재": "결제",
            "분실싱고": "분실신고", "나라사랑": "나라사랑카드" 
        }
        
        # Ensure action keywords are populated
        cls.extractor._action_keywords.update({
            "신고", "분실", "재발급", "신청", "발급", "한도", "상향", "결제", "안돼요", "해지"
        })
        
        # Ensure payment keywords
        cls.extractor._payment_keywords.update({
            "삼성페이": "삼성페이", "애플페이": "애플페이", 
            "네이버페이": "네이버페이", "카카오페이": "카카오페이", "티머니": "티머니"
        })

    def print_section(self, title):
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")

    def print_case(self, input_text, result, expected, label_result="결과"):
        # Simple checkmark logic - in a real scenario we assert first, but here we print before assert or after
        print(f"✓ 입력: {input_text}")
        print(f"   {label_result}: {result}")
        print(f"   예상: {expected}")
        print()

    def test_01_stt_correction(self):
        self.print_section("STT 오류 교정 테스트")
        
        test_cases = [
            ("삼송페이 결채가 안돼요", "삼성페이 결제가 안돼요"),
            ("연헤비 없는 카드 추천해주세요", "연회비 없는 카드 추천해주세요"),
            ("리벌빙 해지하고 싶어요", "리볼빙 해지하고 싶어요"),
            ("활부로 결재하려고요", "할부로 결제하려고요"),
            ("나라사랑 분실싱고", "나라사랑카드 분실신고"),
        ]
        
        for input_text, expected in test_cases:
            result = self.extractor._correct_stt_errors(input_text)
            self.print_case(input_text, result, expected, "교정")
            self.assertEqual(result, expected)

    def test_02_action_extraction(self):
        self.print_section("액션 키워드 추출 테스트")
        
        test_cases = [
            ("카드 분실 신고하려고요", ["분실", "신고"], ["분실", "신고"]),
            ("재발급 신청하고 싶어요", ["재발급", "신청"], ["재발급", "신청"]),
            ("한도 상향해주세요", ["한도", "상향"], ["한도", "상향"]), # '한도상향' compound might be extracted via regex
            ("결제가 안돼요", ["결제", "안돼요"], ["결제", "안돼요"]), # Regex might add '결제오류'
            ("해지하고 싶어요", ["해지"], ["해지"]),
        ]

        for input_text, expected_subset, expected_display in test_cases:
            # Note: _extract_actions might return more items (e.g. regex matches)
            # We iterate and check if expected items are in result
            result = self.extractor._extract_actions(input_text)
            
            # Allow some flexibility for regex matches like '분실신고', '결제오류'
            # The user output example shows: ['신고', '분실', '분실신고']
            
            self.print_case(input_text, result, expected_display, "추출")
            
            for item in expected_subset:
                self.assertIn(item, result)

    def test_03_payment_extraction(self):
        self.print_section("결제수단 추출 테스트")
        
        test_cases = [
            ("삼성페이로 결제하고 싶어요", ["삼성페이"]),
            ("애플페이 등록 방법 알려주세요", ["애플페이"]),
            ("네이버페이 연동하려고요", ["네이버페이"]),
            ("카카오페이 결제 오류", ["카카오페이"]),
            ("티머니 충전하려고요", ["티머니"]),
        ]
        
        for input_text, expected in test_cases:
            result = self.extractor._extract_payments(input_text)
            self.print_case(input_text, result, expected, "추출")
            self.assertEqual(result, expected)

    @patch('app.llm.delivery.keyword_extractor.analyze_morphemes')
    def test_04_conditional_stopwords(self, mock_analyze_morphemes):
        self.print_section("조건부 불용어 (Context-Aware) 테스트")
        
        # Setup mock for morphology
        # Cases: 
        # 1. "청년 희망 카드" -> Keep "카드"
        # 2. "카드 발급" -> Drop "카드"
        # 3. "가맹점 문의" -> Keep "문의"
        # 4. "비밀 번호" -> Keep "번호"
        
        scenarios = {
            "청년 희망 카드": [("청년", "NNG"), ("희망", "NNG"), ("카드", "NNG")],
            "카드 발급": [("카드", "NNG"), ("발급", "NNG")],
            "가맹점 문의": [("가맹점", "NNG"), ("문의", "NNG")],
            "문의 드려요": [("문의", "NNG"), ("드리", "VV"), ("어요", "EF")],
            "비밀 번호": [("비밀", "NNG"), ("번호", "NNG")],
            "번호 좀": [("번호", "NNG"), ("좀", "MAG")],
        }
        
        def side_effect(text):
            return scenarios.get(text, [])
        
        mock_analyze_morphemes.side_effect = side_effect
        
        # Enable morphology flag temporarily if needed, though mocking helper usually bypasses check if we call _extract_nouns directly? 
        # _extract_nouns checks MORPHOLOGY_AVAILABLE.
        import app.llm.delivery.keyword_extractor as ke
        old_avail = ke.MORPHOLOGY_AVAILABLE
        ke.MORPHOLOGY_AVAILABLE = True
        
        try:
            test_cases = [
                ("청년 희망 카드", "카드", True),
                ("카드 발급", "카드", False),
                ("가맹점 문의", "문의", True),
                ("문의 드려요", "문의", False),
                ("비밀 번호", "번호", True),
            ]
            
            for input_text, target_word, should_keep in test_cases:
                nouns = self.extractor._extract_nouns(input_text, [], [], [], [])
                
                kept = target_word in nouns
                result_display = nouns
                expected_display = f"['{target_word}' 포함]" if should_keep else f"['{target_word}' 제외]"
                
                self.print_case(input_text, result_display, expected_display, "추출")
                
                if should_keep:
                    self.assertIn(target_word, nouns)
                else:
                    self.assertNotIn(target_word, nouns)
                    
        finally:
            ke.MORPHOLOGY_AVAILABLE = old_avail

if __name__ == '__main__':
    unittest.main()
