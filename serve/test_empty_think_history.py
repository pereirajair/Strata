"""serve/test_empty_think_history.py - with thinking off, a history's assistant turns keep the empty thinking block
(keep_empty_think_in_history), so the next turn's prompt is the one the earlier turn generated.

    python -m unittest serve.test_empty_think_history -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import EMPTY_THINK, keep_empty_think_in_history  # noqa: E402


class KeepEmptyThink(unittest.TestCase):
    def test_a_history_turn_without_the_block_gets_it(self):
        text = ("<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\nOK<|im_end|>\n"
                "<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n")
        out = keep_empty_think_in_history(text)
        self.assertIn("<|im_start|>assistant\n" + EMPTY_THINK + "OK<|im_end|>", out)

    def test_the_generation_turn_is_left_as_the_template_wrote_it(self):
        text = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n" + EMPTY_THINK
        self.assertEqual(keep_empty_think_in_history(text), text)

    def test_a_block_already_in_a_turn_is_not_doubled(self):
        text = ("<|im_start|>assistant\n<think>\nreason\n</think>\n\nOK<|im_end|>\n"
                "<|im_start|>assistant\n" + EMPTY_THINK)
        self.assertEqual(keep_empty_think_in_history(text).count("<think>"), 2)

    def test_text_without_assistant_turns_is_unchanged(self):
        text = "<|im_start|>user\nhi<|im_end|>\n"
        self.assertEqual(keep_empty_think_in_history(text), text)


if __name__ == "__main__":
    unittest.main()
