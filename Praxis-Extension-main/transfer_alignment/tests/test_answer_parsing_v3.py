import unittest
from transfer_alignment.answer_parsing_v3 import parse_choice
from transfer_alignment.answer_parsing import parse_choice as v2

OPTIONS=['A. First action','B. Second action','C. Third action']


class ParserV3Tests(unittest.TestCase):
    def test_open_tag_does_not_fall_into_reasoning(self):
        text='<answer>B>\n<think>Option A is unsafe. Option C is risky.</think>'
        self.assertEqual(v2(text,OPTIONS),'A')
        self.assertEqual(parse_choice(text,OPTIONS),'B')

    def test_duplicated_label_in_tag_is_consistent(self):
        self.assertEqual(parse_choice('Option A was discussed.\n<Banswer>B</answer>',OPTIONS),'B')
        self.assertIsNone(parse_choice('<Aanswer>B</answer> Option A',OPTIONS))

    def test_invalid_or_multiple_explicit_choices_do_not_fallback(self):
        for text in ('<answer>G</answer> Option A','<answer>A or B</answer>',
                     '<answer>A/B</answer>','<answer>A and C</answer>'):
            self.assertIsNone(parse_choice(text,OPTIONS),text)

    def test_valid_tag_and_untagged_final_retained(self):
        for text in ('<answer>B</answer>','<answer>\nB.\n</answer>',
                     '<answer>B','Option A is unsafe.\nFinal answer: B'):
            self.assertEqual(parse_choice(text,OPTIONS),'B',text)

    def test_unclosed_prose_is_not_reasoning_search(self):
        self.assertIsNone(parse_choice('<answer>Consider Option A before choosing.',OPTIONS))


if __name__=='__main__':unittest.main()
