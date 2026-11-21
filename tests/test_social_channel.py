"""Synthetic public movies check E121's timing, controls and gate arithmetic."""
import copy
import unittest

import numpy as np

from scripts import social_channel as channel


def tracked_movie(color=0):
    movie = np.zeros((3, 103), np.float32)
    movie[:, -1] = 1
    visual = movie[:, :100].reshape(3, 4, 5, 5)
    visual[0, 2, 1, 2] = 1
    visual[1:, 2, 1, 3] = 1
    # Next relative offset (-1,+1) plus the camera's south offset means east.
    visual[0, color, 1, 3] = 1
    return movie


def family(index, *, blank=False):
    rule = index % 2
    observed = tracked_movie(rule)
    if blank:
        observed[:, :100] = 0
    yoked, no_demo, scene_only = (observed.copy() for _ in range(3))
    if (index // 2) % 2:
        yoked[:, :25], yoked[:, 25:50] = observed[:, 25:50], observed[:, :25]
    no_demo[:, :100] = 0
    scene_only[:, 50:100] = 0
    row = dict(rule=np.asarray(rule, np.int32), radius=np.asarray(2, np.int32))
    for arm, movie in zip(channel.ARMS, (observed, yoked, no_demo, scene_only)):
        row[arm + "_frames"] = movie
        for method, decoded in channel.decode_frames(movie, rule).items():
            row.update({method + "_" + arm + "_" + key: value for key, value in decoded.items()})
    return row


def item_patch(world, center):
    row, column = center
    patch = world[row - 2:row + 3, column - 2:column + 3]
    return np.stack((patch == 1, patch == 2))


class ChannelDecoderTests(unittest.TestCase):
    def test_track_reads_destination_in_first_frame_using_next_teacher_offset(self):
        movie = tracked_movie()
        visual = movie[:, :100].reshape(3, 4, 5, 5)
        visual[0, 1, 1, 2] = 1  # Current teacher cell is the wrong answer.
        visual[1, 1, 1, 3] = 1  # Next frame's teacher cell is also the wrong answer.
        decoded = channel.track_movie(movie)
        np.testing.assert_array_equal(decoded["step_color"], [0, -1])
        np.testing.assert_array_equal(decoded["step_identified"], [True, True])
        np.testing.assert_array_equal(decoded["votes"], [1, 0])
        self.assertEqual(int(decoded["prediction"]), 0)

    def test_track_requires_both_sightings_and_reachable_destination(self):
        movie = tracked_movie()
        movie[1, 50:75] = 0
        decoded = channel.track_movie(movie)
        np.testing.assert_array_equal(decoded["step_identified"], [False, False])
        self.assertEqual(decoded["votes"].sum(), 0)
        movie = tracked_movie()
        visual = movie[:, :100].reshape(3, 4, 5, 5)
        visual[1, 2] = 0
        visual[1, 2, 4, 4] = 1  # Would require a non-legal multi-cell move.
        self.assertFalse(channel.track_movie(movie)["step_identified"][0])
        movie = tracked_movie()
        visual = movie[:, :100].reshape(3, 4, 5, 5)
        visual[0, 2] = 0
        visual[0, 2, 1, 4] = 1  # East destination falls outside this patch.
        self.assertFalse(channel.track_movie(movie)["step_identified"][0])

    def test_track_votes_on_visible_entry_without_requiring_disappearance(self):
        movie = tracked_movie()
        movie[1, :25] = movie[0, :25]
        self.assertEqual(channel.track_movie(movie)["votes"].tolist(), [1, 0])

    def test_signal_and_labels_cannot_change_the_prediction(self):
        original = tracked_movie()
        changed = original.copy()
        changed[:, 75:100] = np.arange(75).reshape(3, 25) * -100
        changed[:, -3:] = [123, 456, 789]
        first = channel.decode_frames(original, 0)
        other = channel.decode_frames(changed, 1)
        for method in channel.METHODS:
            for field in first[method].keys() - {"score"}:
                np.testing.assert_array_equal(first[method][field], other[method][field])
        self.assertEqual(float(first["track"]["score"]), 1.)
        self.assertEqual(float(other["track"]["score"]), 0.)

    def test_color_swap_reverses_votes_and_blank_or_tied_movies_abstain(self):
        movie = tracked_movie()
        swapped = movie.copy()
        swapped[:, :25], swapped[:, 25:50] = movie[:, 25:50], movie[:, :25]
        decoded = channel.track_movie(swapped)
        np.testing.assert_array_equal(decoded["votes"], [0, 1])
        self.assertEqual(int(decoded["prediction"]), 1)
        visual = movie[:, :100].reshape(3, 4, 5, 5)
        visual[1, 1, 1, 4] = 1
        for rule in (0, 1):
            tied = channel.decode_frames(movie, rule)["track"]
            np.testing.assert_array_equal(tied["votes"], [1, 1])
            self.assertEqual(float(tied["score"]), .5)
            blank = channel.decode_frames(np.zeros_like(movie), rule)
            for method in channel.METHODS:
                self.assertEqual(float(blank[method]["score"]), .5)
                self.assertFalse(blank[method]["step_identified"].any())

    def test_image_alignment_recovers_all_five_camera_moves(self):
        world = np.random.default_rng(8).integers(0, 3, (13, 13))
        first = np.array([6, 6])
        for move in channel.MOVES:
            with self.subTest(move=move.tolist()):
                shifts, identified = channel.image_shifts(np.stack((
                    item_patch(world, first), item_patch(world, first + move))))
                np.testing.assert_array_equal(shifts, move[None])
                np.testing.assert_array_equal(identified, [True])

    def test_scene_contact_uses_two_shifts_and_the_first_item_frame(self):
        world = np.random.default_rng(8).integers(0, 3, (13, 13))
        centers = np.array([[6, 6], [6, 7], [7, 7]])
        world[6, 7] = 1
        movie = np.zeros((3, 103), np.float32)
        for index, center in enumerate(centers):
            movie[index, :50] = item_patch(world, center).ravel()
        decoded = channel.scene_movie(movie)
        np.testing.assert_array_equal(decoded["shifts"], [[0, 1], [1, 0]])
        np.testing.assert_array_equal(decoded["step_color"], [0])
        np.testing.assert_array_equal(decoded["votes"], [1, 0])
        self.assertTrue(decoded["step_identified"].all())
        self.assertEqual(channel.track_movie(movie)["votes"].sum(), 0)
        swapped = movie.copy()
        swapped[:, :25], swapped[:, 25:50] = movie[:, 25:50], movie[:, :25]
        np.testing.assert_array_equal(channel.scene_movie(swapped)["votes"], [0, 1])

    def test_uniform_scenes_abstain_on_ambiguous_camera_alignment(self):
        items = np.zeros((3, 2, 5, 5), bool)
        items[:, 0] = True
        shifts, identified = channel.image_shifts(items)
        self.assertFalse(identified.any())
        np.testing.assert_array_equal(shifts, 0)

    def test_rejects_invalid_or_ambiguous_teacher_pixels(self):
        movie = tracked_movie()
        movie[0, 50] = 1
        with self.assertRaises(ValueError):
            channel.track_movie(movie)
        movie = tracked_movie()
        movie[0, 0] = movie[0, 25] = 1
        with self.assertRaises(ValueError):
            channel.scene_movie(movie)
        with self.assertRaises(ValueError):
            channel.track_movie(np.zeros((3, 102)))

    def test_family_cluster_summary_and_four_gates(self):
        summary = channel.summarize([family(index) for index in range(16)])
        observed = summary["track"]["arms"]["observed"]
        self.assertEqual(observed["accuracy"], dict(mean=1., se=0.))
        self.assertEqual(observed["coverage"], dict(mean=1., se=0.))
        paired = summary["track"]["comparisons"]["yoked"]
        self.assertEqual(paired["mean"], .5)
        self.assertAlmostEqual(paired["se"], np.sqrt(.25 / 15))
        self.assertTrue(all(summary["gate"].values()))
        self.assertEqual(summary["track"]["arms"]["scene_only"]["accuracy"]["mean"], .5)

    def test_empty_evidence_is_kept_and_fails_the_usable_channel_gate(self):
        summary = channel.summarize([family(index, blank=True) for index in range(16)])
        observed = summary["track"]["arms"]["observed"]
        self.assertEqual(observed["accuracy"]["mean"], .5)
        self.assertEqual(observed["no_vote_fraction"]["mean"], 1.)
        self.assertEqual(observed["tie_fraction"]["mean"], 1.)
        self.assertFalse(any(summary["gate"].values()))
        self.assertIsNone(channel.summarize([family(0)])["gate"]["passed"])

    def test_saved_predictions_must_be_reproducible_from_the_pixels(self):
        row = family(0)
        changed = copy.deepcopy(row)
        changed["track_observed_prediction"] = np.asarray(1, np.int32)
        with self.assertRaises(ValueError):
            channel.summarize([changed])
        changed = copy.deepcopy(row)
        changed["track_observed_score"] = np.asarray(.5)
        with self.assertRaises(ValueError):
            channel.summarize([changed])


if __name__ == "__main__":
    unittest.main()
