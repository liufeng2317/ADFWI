import unittest
import sys
import types

import numpy as np

from ADFWI.utils.noise import add_gaussian_noise


SSIM_CALLS = []


def fake_structural_similarity(true_v, inv_v, *, data_range, win_size):
    """Small deterministic stand-in for skimage's SSIM in dependency-light tests."""
    SSIM_CALLS.append(
        {
            "true_v": np.asarray(true_v).copy(),
            "inv_v": np.asarray(inv_v).copy(),
            "data_range": data_range,
            "win_size": win_size,
        }
    )
    return float(np.mean(inv_v - true_v) + data_range + win_size)


skimage_module = types.ModuleType("skimage")
skimage_metrics_module = types.ModuleType("skimage.metrics")
skimage_metrics_module.structural_similarity = fake_structural_similarity
sys.modules.setdefault("skimage", skimage_module)
sys.modules.setdefault("skimage.metrics", skimage_metrics_module)

from ADFWI.utils.assessment_metric import MAPE, MSE, RMSE, SNR, SSIM  # noqa: E402


class UtilsNoiseTests(unittest.TestCase):
    def test_add_gaussian_noise_is_reproducible_with_seed(self):
        data = np.arange(12, dtype=np.float64).reshape(1, 4, 3)

        first = add_gaussian_noise(data, std_noise=0.25, mean_bias_factor=0.1, seed=7)
        second = add_gaussian_noise(data, std_noise=0.25, mean_bias_factor=0.1, seed=7)

        np.testing.assert_allclose(first, second)
        self.assertEqual(first.shape, data.shape)

    def test_add_gaussian_noise_matches_current_numpy_random_formula(self):
        data = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
            ]
        )
        std_noise = 0.5
        mean_bias_factor = 0.25
        seed = 11

        noisy = add_gaussian_noise(data, std_noise, mean_bias_factor=mean_bias_factor, seed=seed)

        trace_means = np.mean(data, axis=1, keepdims=True)
        expected_noise = np.random.RandomState(seed).normal(
            loc=trace_means * mean_bias_factor,
            scale=std_noise,
            size=data.shape,
        )
        np.testing.assert_allclose(noisy, data + expected_noise)

    def test_add_gaussian_noise_seed_none_does_not_reset_global_rng(self):
        data = np.ones((1, 2, 2), dtype=np.float64)
        np.random.seed(13)
        first = add_gaussian_noise(data, std_noise=0.1, mean_bias_factor=0.0, seed=None)
        second = add_gaussian_noise(data, std_noise=0.1, mean_bias_factor=0.0, seed=None)

        self.assertFalse(np.array_equal(first, second))


class UtilsAssessmentMetricTests(unittest.TestCase):
    def test_mse_rmse_mape_snr_for_2d_inputs(self):
        true_v = np.array([[2.0, 4.0], [6.0, 8.0]])
        inv_v = np.array([[1.0, 5.0], [7.0, 10.0]])
        diff = true_v - inv_v

        self.assertEqual(MSE(true_v, inv_v), np.sum(diff**2))
        self.assertEqual(RMSE(true_v, inv_v), np.sqrt(np.sum(diff**2)))
        self.assertEqual(MAPE(true_v, inv_v), 100 / true_v.size * np.sum(np.abs(diff) / true_v))
        self.assertEqual(SNR(true_v, inv_v), 10 * np.log10(np.sum(true_v**2) / np.sum(diff**2)))

    def test_rmse_mape_snr_return_per_model_values_for_batch_inputs(self):
        true_v = np.array([[2.0, 4.0], [6.0, 8.0]])
        inv_v = np.stack(
            [
                np.array([[1.0, 5.0], [7.0, 10.0]]),
                np.array([[2.0, 4.0], [3.0, 4.0]]),
            ]
        )
        diff = true_v[np.newaxis, :, :] - inv_v

        np.testing.assert_allclose(RMSE(true_v, inv_v), np.sqrt(np.sum(diff**2, axis=(1, 2))))
        np.testing.assert_allclose(MAPE(true_v, inv_v), 100 / true_v.size * np.sum(np.abs(diff) / true_v, axis=(1, 2)))
        expected_snr = [10 * np.log10(np.sum(true_v**2) / np.sum((true_v - inv_v[i]) ** 2)) for i in range(inv_v.shape[0])]
        np.testing.assert_allclose(SNR(true_v, inv_v), expected_snr)

    def test_ssim_matches_skimage_for_2d_and_batch_inputs(self):
        SSIM_CALLS.clear()
        true_v = np.arange(25, dtype=np.float64).reshape(5, 5)
        inv_0 = true_v + np.eye(5)
        inv_1 = true_v + np.flipud(np.eye(5))
        inv_batch = np.stack([inv_0, inv_1])
        win_size = 3

        result_2d = SSIM(true_v, inv_0, win_size=win_size)
        result_batch = SSIM(true_v, inv_batch, win_size=win_size)

        self.assertEqual(len(SSIM_CALLS), 3)
        self.assertEqual(SSIM_CALLS[0]["data_range"], max(true_v.max(), inv_0.max()) - min(true_v.min(), inv_0.min()))
        self.assertEqual(SSIM_CALLS[0]["win_size"], win_size)
        self.assertEqual(result_2d, fake_structural_similarity(true_v, inv_0, data_range=SSIM_CALLS[0]["data_range"], win_size=win_size))
        expected_batch = [
            fake_structural_similarity(
                true_v,
                inv_batch[i],
                data_range=max(true_v.max(), inv_batch[i].max()) - min(true_v.min(), inv_batch[i].min()),
                win_size=win_size,
            )
            for i in range(inv_batch.shape[0])
        ]
        np.testing.assert_allclose(result_batch, expected_batch)


if __name__ == "__main__":
    unittest.main()
