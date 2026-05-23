import unittest

import numpy as np
import torch

from ADFWI.backends import configure_backend, get_backend
from ADFWI.model import AcousticModel, IsotropicElasticModel
from ADFWI.propagator import AcousticPropagator, ElasticPropagator
from ADFWI.fwi import AcousticFWI
from ADFWI.fwi.regularization import regularization_Tikhonov_1order
from ADFWI.fwi.misfit import Misfit_waveform_L2
from ADFWI.survey import Receiver, SeismicData, Source, Survey


class BackendIntegrationTests(unittest.TestCase):
    def tearDown(self):
        configure_backend("cpu")

    def _model_arrays(self):
        vp = np.ones((6, 8), dtype=np.float32) * 2000.0
        rho = np.ones((6, 8), dtype=np.float32) * 1800.0
        return vp, rho

    def _survey(self):
        nt = 8
        dt = 0.001
        source = Source(nt=nt, dt=dt, f0=5.0)
        source.add_source(2, 2, np.ones(nt, dtype=np.float32), src_type="mt")
        receiver = Receiver(nt=nt, dt=dt)
        receiver.add_receiver(3, 2, rcv_type="pr")
        return Survey(source, receiver)

    def test_acoustic_model_inherits_configured_cpu_backend(self):
        configure_backend("cpu")
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)

        self.assertEqual(model.device.type, "cpu")
        self.assertEqual(model.dtype, torch.float32)
        self.assertEqual(model.vp.device.type, "cpu")
        self.assertTrue(model.vp.requires_grad)

    def test_acoustic_propagator_follows_model_device_by_default(self):
        configure_backend("cpu")
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        propagator = AcousticPropagator(model, self._survey())

        self.assertEqual(propagator.device, model.device)
        self.assertEqual(propagator.dtype, model.dtype)
        self.assertEqual(propagator.src_x.device, model.device)
        self.assertEqual(propagator.rcv_x.device, model.device)


    def test_acoustic_propagator_inherits_model_dtype_by_default(self):
        configure_backend("cpu", dtype=torch.float64)
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        propagator = AcousticPropagator(model, self._survey())

        self.assertEqual(model.dtype, torch.float64)
        self.assertEqual(propagator.dtype, torch.float64)
        self.assertEqual(propagator.wavelet.dtype, torch.float64)

    def test_explicit_model_device_remains_supported(self):
        configure_backend("cpu")
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True, device="cpu")

        self.assertEqual(model.device.type, "cpu")
        self.assertEqual(get_backend().device.type, "cpu")

    def test_acoustic_model_can_inherit_npu_backend_when_available(self):
        npu = getattr(torch, "npu", None)
        is_available = getattr(npu, "is_available", None)
        if not (callable(is_available) and is_available()):
            self.skipTest("NPU is not available on this machine")

        configure_backend("npu:0")
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)

        self.assertEqual(model.device.type, "npu")
        self.assertEqual(model.vp.device.type, "npu")

    def test_regularization_inherits_configured_backend(self):
        configure_backend("cpu", dtype=torch.float64)
        reg = regularization_Tikhonov_1order(8, 6, 10, 10, alphax=1, alphaz=1)

        self.assertEqual(reg.device.type, "cpu")
        self.assertEqual(reg.dtype, torch.float64)
        self.assertEqual(reg.L0.device.type, "cpu")
        self.assertEqual(reg.L0.dtype, torch.float64)

    def test_regularization_explicit_device_and_dtype_remain_supported(self):
        configure_backend("cpu", dtype=torch.float64)
        reg = regularization_Tikhonov_1order(8, 6, 10, 10, alphax=1, alphaz=1, device="cpu", dtype=torch.float32)

        self.assertEqual(reg.device.type, "cpu")
        self.assertEqual(reg.dtype, torch.float32)
        self.assertEqual(reg.L1.dtype, torch.float32)

    def test_elastic_model_and_propagator_inherit_configured_backend(self):
        configure_backend("cpu", dtype=torch.float64)
        vp = np.ones((6, 8), dtype=np.float32) * 2200.0
        vs = np.ones((6, 8), dtype=np.float32) * 1200.0
        rho = np.ones((6, 8), dtype=np.float32) * 2000.0
        model = IsotropicElasticModel(0, 0, 8, 6, 10, 10, vp, vs, rho, vp_grad=True, auto_update_rho=False)
        propagator = ElasticPropagator(model, self._survey())

        self.assertEqual(model.device.type, "cpu")
        self.assertEqual(model.dtype, torch.float64)
        self.assertEqual(model.vp.dtype, torch.float64)
        self.assertEqual(propagator.device, model.device)
        self.assertEqual(propagator.dtype, model.dtype)
        self.assertEqual(propagator.wavelet.dtype, torch.float64)

    def test_acoustic_fwi_aligns_regularization_to_propagator_backend(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        propagator = AcousticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {"p": np.zeros((1, 8, 1), dtype=np.float32)}
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        reg = regularization_Tikhonov_1order(8, 6, 10, 10, alphax=1, alphaz=1, device="cpu", dtype=torch.float64)

        fwi = AcousticFWI(
            propagator,
            model,
            optimizer,
            scheduler,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            regularization_fn=reg,
            cache_result=False,
        )

        self.assertEqual(fwi.device, propagator.device)
        self.assertEqual(reg.device, propagator.device)
        self.assertEqual(reg.dtype, propagator.dtype)
        self.assertEqual(reg.L0.dtype, propagator.dtype)


if __name__ == "__main__":
    unittest.main()
