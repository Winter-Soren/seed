import unittest

import numpy as np

from tools.e138_feedback_opportunities import FIELDS, feedback_opportunities


class FeedbackOpportunitiesTests(unittest.TestCase):
    def setUp(self):
        self.obs = np.zeros((12, 5, 1, 103), np.float32)
        self.moves = np.zeros((12, 5, 1), np.int32)
        self.live = np.ones((12, 5, 1), bool)

    def visible(self, i, t, color, patch_index):
        self.obs[i, t, 0, patch_index + 25*(color == "B")] = 1

    def first(self, i, sign, t=0):
        self.visible(i, t, "A", 7)
        self.moves[i, t, 0] = 1
        if t+1 < 5:
            self.obs[i, t+1, 0, 101] = 4*(sign*.25-.003)

    def test_assigned_lives_and_decision_timing(self):
        self.first(0, +1)
        self.visible(0, 1, "A", 11)
        self.visible(0, 2, "A", 17)
        self.visible(0, 2, "B", 13)
        self.moves[0, 2, 0] = 4
        self.first(1, -1)
        self.visible(1, 1, "A", 12)
        self.visible(1, 2, "A", 11)
        self.visible(1, 2, "B", 13)  # Both appeared only after the second contact.
        self.visible(2, 1, "A", 11)  # No contact during the entire life.
        self.first(3, +1)
        self.obs[3, 1, 0, 100] = 1  # Reported energy at upper clip boundary.
        self.first(4, -1, t=4)  # Last action has no subsequent observation.
        self.first(5, -1)
        self.live[5, 1:] = False  # Fatal first contact gives no usable feedback.
        self.first(6, -1)
        self.visible(6, 1, "A", 7)
        self.visible(6, 1, "B", 17)
        self.moves[6, 1, 0] = 2
        self.live[6, 2:] = False  # Fatal second contact still counts.
        self.first(7, +1)
        self.visible(7, 1, "B", 13)  # Reachable, but the creature waits.
        self.first(8, -1)
        self.obs[8, 1, 0, 101] = 0  # Ambiguous energy feedback.
        self.first(9, +1)
        self.visible(9, 1, "A", 11)
        self.moves[9, 1, 0] = 3
        self.first(10, -1)
        self.visible(10, 0, "B", 17)  # Both reachable only *before* feedback.
        self.visible(10, 1, "B", 0)   # Visible at a corner, outside one move.
        self.visible(11, 0, "B", 12)  # Staying over an item is a contact.
        self.obs[11, 1, 0, 101] = 4*(.25-.003)
        self.visible(11, 1, "B", 17)
        self.moves[11, 1, 0] = 2
        flags = feedback_opportunities(self.obs.reshape(3, 2, 2, 5, 1, 103),
                                       self.moves.reshape(3, 2, 2, 5, 1),
                                       self.live.reshape(3, 2, 2, 5, 1))
        expected = {
            "first_feedback_gain": [0, 7, 9, 11], "first_feedback_loss": [1, 6, 10],
            "reachable_item_after_feedback": [0, 1, 6, 7, 9, 11],
            "both_colors_after_feedback": [0, 6], "second_contact": [0, 1, 6, 9, 11],
            "both_colors_at_second_contact": [0, 6], "repeat_after_gain": [9, 11],
            "switch_after_gain": [0], "repeat_after_loss": [1], "switch_after_loss": [6],
        }
        self.assertEqual(set(expected), set(FIELDS))
        for key, lives in expected.items():
            self.assertEqual(flags[key].shape, (3, 2, 2))
            self.assertEqual(np.flatnonzero(flags[key]).tolist(), lives, key)

    def test_both_types_in_one_cell_is_invalid(self):
        self.visible(0, 1, "A", 12)
        self.visible(0, 1, "B", 12)
        with self.assertRaisesRegex(ValueError, "visible item"):
            feedback_opportunities(self.obs, self.moves, self.live)

    def test_illegal_action_is_invalid(self):
        self.moves[1, 1] = -1
        with self.assertRaisesRegex(ValueError, "action"):
            feedback_opportunities(self.obs, self.moves, self.live)


if __name__ == "__main__":
    unittest.main()
