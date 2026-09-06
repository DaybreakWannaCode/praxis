import unittest
from transfer_alignment.answer_parsing import parse_choice,score_completion

class AnswerParsingTests(unittest.TestCase):
    options=['A. First action','B. Second action','C. Third action','D. Fourth action','E. Fifth action']
    def test_final_declaration_overrides_reasoning(self):
        text='Option A: this is unsafe.\nOption D: a suitable response.\n\nFinal Answer: D.'
        self.assertEqual(parse_choice(text,self.options,profile='legacy'),'A')
        self.assertEqual(parse_choice(text,self.options),'D')
        self.assertEqual(score_completion(text,'D',self.options),(1.,True))
        text="Considering the options, action E doesn't make sense.\nFinal Option: B"
        self.assertEqual(parse_choice(text,self.options),'B')
    def test_tags_and_non_declarations_preserve_behavior(self):
        self.assertEqual(parse_choice('Option A is bad.\n<answer>C</answer>',self.options),'C')
        self.assertIsNone(parse_choice('<think>still reasoning</think>',self.options))
        self.assertEqual(parse_choice('Answer: A\nFinal answer is B.',self.options),'B')
    def test_invalid_and_ambiguous_final_do_not_fall_back(self):
        for tail in ['Final Answer: Z','Final Answer: A or B','Final Answer: A/B']:
            self.assertIsNone(parse_choice('Option D is useful.\n'+tail,self.options))
        with self.assertRaises(ValueError):parse_choice('A',self.options,profile='unknown')
