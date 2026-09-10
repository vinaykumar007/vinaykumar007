import unittest

import numpy as np

from src.thermal_diffusion import simulate, stability_limit


class ThermalDiffusionTests(unittest.TestCase):
    def test_stability_limit(self):
        self.assertAlmostEqual(stability_limit(1.0, 0.1), 0.005)

    def test_fixed_boundaries_are_preserved(self):
        x, temperature = simulate(
            length=1.0,
            nx=21,
            total_time=0.05,
            dt=0.001,
            alpha=0.1,
            left_temperature=100.0,
            right_temperature=20.0,
            initial_temperature=20.0,
        )
        self.assertEqual(len(x), 21)
        np.testing.assert_allclose(temperature[:, 0], 100.0)
        np.testing.assert_allclose(temperature[:, -1], 20.0)

    def test_diffusion_moves_solution_toward_smoother_profile(self):
        _, temperature = simulate(
            length=1.0,
            nx=21,
            total_time=0.02,
            dt=0.001,
            alpha=0.1,
            left_temperature=100.0,
            right_temperature=0.0,
            initial_temperature=0.0,
        )
        self.assertGreater(temperature[-1, 1], temperature[0, 1])
        self.assertLessEqual(temperature[-1].max(), 100.0)
        self.assertGreaterEqual(temperature[-1].min(), 0.0)

    def test_unstable_step_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate(
                length=1.0,
                nx=11,
                total_time=0.1,
                dt=0.02,
                alpha=1.0,
                left_temperature=100.0,
                right_temperature=0.0,
                initial_temperature=0.0,
            )


if __name__ == "__main__":
    unittest.main()
