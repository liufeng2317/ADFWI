import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ADFWI.survey import Receiver, SeismicData, Source, Survey


class SurveyContractTests(unittest.TestCase):
    def _survey(self):
        nt = 5
        dt = 0.002
        source = Source(nt=nt, dt=dt, f0=4.0)
        wavelet = np.linspace(0.0, 1.0, nt, dtype=np.float32)
        source.add_sources(
            np.array([1, 3]),
            np.array([2, 2]),
            wavelet,
            src_type="mt",
        )
        receiver = Receiver(nt=nt, dt=dt)
        receiver.add_receivers(
            np.array([0, 2, 4]),
            np.array([1, 1, 1]),
            rcv_type="pr",
        )
        return Survey(source, receiver)

    def test_source_receiver_and_survey_shape_contracts(self):
        survey = self._survey()

        self.assertEqual(survey.source.num, 2)
        self.assertEqual(survey.receiver.num, 3)
        self.assertEqual(survey.source.get_loc().shape, (2, 2))
        self.assertEqual(survey.source.get_wavelet().shape, (2, 5))
        self.assertEqual(survey.source.get_moment_tensor().shape, (2, 3, 3))
        self.assertEqual(survey.receiver.get_loc().shape, (3, 2))
        self.assertEqual(survey.source.get_type().tolist(), ["mt", "mt"])
        self.assertEqual(survey.receiver.get_type().tolist(), ["pr", "pr", "pr"])

    def test_source_adders_accept_array_like_and_validate_shapes(self):
        nt = 4
        source = Source(nt=nt, dt=0.002, f0=4.0)

        source.add_source(1, 2, [0.0, 1.0, 0.5, 0.0])
        source.add_sources([3, 5], [2, 2], [0.0, 0.5, 1.0, 0.0])

        self.assertEqual(source.num, 3)
        self.assertEqual(source.get_loc().shape, (3, 2))
        self.assertEqual(source.get_wavelet().shape, (3, nt))
        self.assertEqual(source.get_moment_tensor().shape, (3, 3, 3))

        with self.assertRaisesRegex(ValueError, "Source wavelet"):
            source.add_source(1, 2, [[0.0, 1.0, 0.5, 0.0]])

        with self.assertRaisesRegex(ValueError, "same shape"):
            source.add_sources([1, 2], [1], [0.0, 0.5, 1.0, 0.0])

        with self.assertRaisesRegex(ValueError, "1-D"):
            source.add_sources([[1, 2]], [[1, 2]], [0.0, 0.5, 1.0, 0.0])

        with self.assertRaisesRegex(ValueError, "Moment tensor"):
            source.add_source(1, 2, [0.0, 1.0, 0.5, 0.0], src_mt=None)

    def test_encoded_source_contracts(self):
        nt = 4
        source = Source(nt=nt, dt=0.002, f0=4.0)
        encoded_wavelet = np.ones((2, 2, nt), dtype=np.float32)

        source.add_encoded_sources(
            src_x=np.array([1, 3]),
            src_z=np.array([2, 2]),
            src_wavelet=encoded_wavelet,
        )

        self.assertEqual(source.num, 2)
        self.assertEqual(source.get_loc().shape, (2, 2))
        self.assertEqual(source.get_wavelet().shape, (2, 2, nt))
        self.assertEqual(source.get_moment_tensor().shape, (2, 3, 3))

        with self.assertRaisesRegex(ValueError, "same length"):
            source.add_encoded_sources(
                src_x=np.array([1, 3]),
                src_z=np.array([2, 2]),
                src_wavelet=np.ones((2, 2, nt + 1), dtype=np.float32),
            )

        with self.assertRaisesRegex(ValueError, "array-like"):
            source.add_encoded_sources(
                src_x=np.array(1),
                src_z=np.array(2),
                src_wavelet=encoded_wavelet,
            )

    def test_receiver_adders_accept_array_like_and_validate_shapes(self):
        receiver = Receiver(nt=4, dt=0.002)

        receiver.add_receiver(0, 1, "pr")
        receiver.add_receivers([1, 2], [1, 1], "vx")

        self.assertEqual(receiver.num, 3)
        self.assertEqual(receiver.get_loc().shape, (3, 2))
        self.assertEqual(receiver.get_type().tolist(), ["pr", "vx", "vx"])

        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            receiver.add_receivers([1, 2], [1], "pr")

        with self.assertRaisesRegex(ValueError, "1-D"):
            receiver.add_receivers([[1, 2]], [[1, 2]], "pr")

        with self.assertRaisesRegex(ValueError, "Receiver type"):
            receiver.add_receiver(0, 1, "ux")

    def test_receiver_mask_shape_contract(self):
        survey = self._survey()
        receiver_masks = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)

        survey.set_receiver_masks(receiver_masks)

        self.assertTrue(np.array_equal(survey.receiver_masks, receiver_masks))
        self.assertEqual(survey.receiver_masks.shape, (survey.source.num, survey.receiver.num))

        with self.assertRaisesRegex(ValueError, "Receiver Mask"):
            survey.set_receiver_masks(np.ones((1, 3), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "2-D"):
            survey.set_receiver_masks(np.ones(3, dtype=np.float32))

    def test_receiver_mask_constructor_accepts_array_like_and_preserves_obs_flag(self):
        survey = self._survey()
        receiver_masks = [[1, 0, 1], [0, 1, 1]]

        masked_survey = Survey(
            survey.source,
            survey.receiver,
            receiver_masks=receiver_masks,
            receiver_masks_obs=False,
        )

        self.assertTrue(np.array_equal(masked_survey.receiver_masks, np.asarray(receiver_masks)))
        self.assertFalse(masked_survey.receiver_masks_obs)

    def test_seismic_data_record_save_load_round_trip(self):
        survey = self._survey()
        seismic_data = SeismicData(survey)
        waveform = {
            "p": torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3),
            "u": torch.ones((2, 5, 3), dtype=torch.float32),
            "w": torch.zeros((2, 5, 3), dtype=torch.float32),
        }

        seismic_data.record_data(waveform)

        self.assertIsInstance(waveform["p"], torch.Tensor)
        self.assertIsInstance(seismic_data.data["p"], np.ndarray)
        pressure, u, w = seismic_data.parse_acoustic_data(normalize=False)
        self.assertEqual(pressure.dtype, np.float32)
        self.assertTrue(np.array_equal(pressure, waveform["p"].numpy()))
        self.assertTrue(np.array_equal(u, waveform["u"].numpy()))
        self.assertTrue(np.array_equal(w, waveform["w"].numpy()))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "obs_data.npz"
            seismic_data.save(str(path))

            loaded = SeismicData(survey)
            loaded.load(str(path))

        self.assertEqual(int(loaded.src_num), 2)
        self.assertEqual(int(loaded.rcv_num), 3)
        self.assertEqual(int(loaded.nt), 5)
        self.assertEqual(float(loaded.dt), 0.002)
        self.assertTrue(np.array_equal(loaded.src_loc, survey.source.get_loc()))
        self.assertTrue(np.array_equal(loaded.rcv_loc, survey.receiver.get_loc()))
        self.assertTrue(np.array_equal(loaded.data["p"], pressure))
        self.assertTrue(np.array_equal(loaded.data["u"], u))
        self.assertTrue(np.array_equal(loaded.data["w"], w))

    def test_seismic_data_parse_elastic_and_normalization_contracts(self):
        survey = self._survey()
        seismic_data = SeismicData(survey)
        txx = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
        tzz = np.ones((2, 5, 3), dtype=np.float32)
        txz = np.full((2, 5, 3), 2.0, dtype=np.float32)
        vx = np.zeros((2, 5, 3), dtype=np.float32)
        vz = np.full((2, 5, 3), -3.0, dtype=np.float32)
        seismic_data.data = {"txx": txx, "tzz": tzz, "txz": txz, "vx": vx, "vz": vz}

        pressure, parsed_txz, parsed_vx, parsed_vz = seismic_data.parse_elastic_data(normalize=False)

        self.assertTrue(np.array_equal(pressure, -(txx + tzz)))
        self.assertTrue(np.array_equal(parsed_txz, txz))
        self.assertTrue(np.array_equal(parsed_vx, vx))
        self.assertTrue(np.array_equal(parsed_vz, vz))

        normalized = seismic_data.normalize_and_mask(
            np.array([[[0.0, 0.0], [0.0, 2.0], [0.0, -4.0]]], dtype=np.float32)
        )

        expected = np.array([[[0.0, 0.0], [0.0, 0.5], [0.0, -1.0]]], dtype=np.float32)
        self.assertTrue(np.array_equal(normalized, expected))


if __name__ == "__main__":
    unittest.main()
