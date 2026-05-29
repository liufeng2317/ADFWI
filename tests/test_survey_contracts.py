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

    def test_receiver_mask_shape_contract(self):
        survey = self._survey()
        receiver_masks = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)

        survey.set_receiver_masks(receiver_masks)

        self.assertTrue(np.array_equal(survey.receiver_masks, receiver_masks))

        with self.assertRaisesRegex(ValueError, "Receiver Mask"):
            survey.set_receiver_masks(np.ones((1, 3), dtype=np.float32))

    def test_seismic_data_record_save_load_round_trip(self):
        survey = self._survey()
        seismic_data = SeismicData(survey)
        waveform = {
            "p": torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3),
            "u": torch.ones((2, 5, 3), dtype=torch.float32),
            "w": torch.zeros((2, 5, 3), dtype=torch.float32),
        }

        seismic_data.record_data(waveform)

        pressure, u, w = seismic_data.parse_acoustic_data(normalize=False)
        self.assertTrue(np.array_equal(pressure, waveform["p"]))
        self.assertTrue(np.array_equal(u, waveform["u"]))
        self.assertTrue(np.array_equal(w, waveform["w"]))

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


if __name__ == "__main__":
    unittest.main()
