import unittest

import numpy as np
import torch

from ADFWI.backends import configure_backend, get_backend
from ADFWI.model import AcousticModel, IsotropicElasticModel
from ADFWI.propagator import AcousticPropagator, ElasticPropagator, GradProcessor
from ADFWI.fwi import AcousticFWI, ElasticFWI
from ADFWI.fwi.regularization import regularization_Tikhonov_1order
from ADFWI.fwi.misfit import Misfit_waveform_L2
from ADFWI.fwi.transforms import (
    DataMask,
    DataTransformPipeline,
    LegacyLateWindowMute,
    LegacyLowPassFilter,
    LegacyOffsetMute,
    TraceNormalize,
)
from ADFWI.survey import Receiver, SeismicData, Source, Survey


class DummyRegularization:
    def __init__(self):
        self.alphax = 0.0
        self.alphaz = 0.0
        self.calls = []
        self.device = None
        self.dtype = None

    def forward(self, model_param):
        self.calls.append((self.alphax, self.alphaz))
        return torch.sum(model_param * self.alphax) + torch.sum(model_param * self.alphaz * 0.1)


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

    def _trace_missing_survey(self):
        nt = 8
        dt = 0.001
        source = Source(nt=nt, dt=dt, f0=5.0)
        source.add_source(2, 2, np.ones(nt, dtype=np.float32), src_type="mt")
        receiver = Receiver(nt=nt, dt=dt)
        receiver.add_receivers(np.array([2, 3, 4]), np.array([2, 2, 2]), rcv_type="pr")
        receiver_masks = np.array([[1, 0, 1]], dtype=np.float32)
        return Survey(source, receiver, receiver_masks=receiver_masks, receiver_masks_obs=False)

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

    def test_acoustic_propagator_construction_preserves_survey_tensor_contracts(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._trace_missing_survey()
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)

        propagator = AcousticPropagator(model, survey)

        self.assertEqual(propagator.src_x.shape, (1,))
        self.assertEqual(propagator.src_z.shape, (1,))
        self.assertEqual(propagator.rcv_x.shape, (3,))
        self.assertEqual(propagator.rcv_z.shape, (3,))
        self.assertEqual(propagator.wavelet.shape, (1, survey.source.nt))
        self.assertEqual(propagator.moment_tensor.shape, (1, 3, 3))
        self.assertEqual(propagator.src_x.dtype, torch.long)
        self.assertEqual(propagator.rcv_x.dtype, torch.long)
        self.assertEqual(propagator.wavelet.dtype, propagator.dtype)
        self.assertEqual(propagator.wavelet.device, propagator.device)
        self.assertTrue(np.array_equal(propagator.receiver_masks, survey.receiver_masks))
        self.assertFalse(propagator.receiver_masks_obs)

    def test_acoustic_forward_wavefield_output_can_be_skipped_without_changing_receiver_gradient(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp, rho = self._model_arrays()
        full_model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        skipped_model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        full_propagator = AcousticPropagator(full_model, survey)
        skipped_propagator = AcousticPropagator(skipped_model, survey)

        full_record = full_propagator.forward(checkpoint_segments=1, save_forward_wavefield=True)
        skipped_record = skipped_propagator.forward(checkpoint_segments=1, save_forward_wavefield=False)
        full_loss = full_record["p"].pow(2).mean()
        skipped_loss = skipped_record["p"].pow(2).mean()
        full_loss.backward()
        skipped_loss.backward()

        for key in ("p", "u", "w"):
            self.assertTrue(torch.equal(full_record[key], skipped_record[key]), key)
        self.assertTrue(torch.equal(full_model.vp.grad, skipped_model.vp.grad))
        self.assertEqual(float(full_loss.item()), float(skipped_loss.item()))
        self.assertGreater(float(torch.linalg.norm(full_record["forward_wavefield_p"]).item()), 0.0)
        self.assertEqual(float(torch.linalg.norm(skipped_record["forward_wavefield_p"]).item()), 0.0)

    def test_acoustic_pressure_only_matches_pressure_loss_and_gradient(self):
        configure_backend("cpu", dtype=torch.float64)
        survey = self._survey()
        vp, rho = self._model_arrays()
        full_model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        pressure_model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        full_propagator = AcousticPropagator(full_model, survey)
        pressure_propagator = AcousticPropagator(pressure_model, survey)

        full_record = full_propagator.forward(checkpoint_segments=2, save_forward_wavefield=True)
        pressure_record = pressure_propagator.forward(
            checkpoint_segments=2,
            save_forward_wavefield=True,
            pressure_only=True,
        )
        full_loss = full_record["p"].pow(2).mean()
        pressure_loss = pressure_record["p"].pow(2).mean()
        full_loss.backward()
        pressure_loss.backward()

        self.assertTrue(torch.equal(full_record["p"], pressure_record["p"]))
        self.assertTrue(torch.equal(full_record["forward_wavefield_p"], pressure_record["forward_wavefield_p"]))
        self.assertEqual(float(full_loss.item()), float(pressure_loss.item()))
        self.assertTrue(torch.allclose(full_model.vp.grad, pressure_model.vp.grad, atol=1e-10, rtol=1e-8))
        for key in ("u", "w", "forward_wavefield_u", "forward_wavefield_w"):
            self.assertEqual(float(torch.linalg.norm(pressure_record[key]).item()), 0.0, key)

    def test_acoustic_remat_pressure_strategy_matches_default_pressure_loss(self):
        configure_backend("cpu", dtype=torch.float64)
        survey = self._survey()
        vp, rho = self._model_arrays()
        default_model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        custom_model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        default_propagator = AcousticPropagator(default_model, survey)
        custom_propagator = AcousticPropagator(custom_model, survey)

        default_record = default_propagator.forward(
            checkpoint_segments=1,
            save_forward_wavefield=False,
            pressure_only=True,
        )
        custom_record = custom_propagator.forward(
            checkpoint_segments=1,
            save_forward_wavefield=False,
            pressure_only=True,
            custom_chunk_strategy="remat_pressure_stride2",
        )
        default_loss = default_record["p"].pow(2).mean()
        custom_loss = custom_record["p"].pow(2).mean()
        default_loss.backward()
        custom_loss.backward()

        self.assertTrue(torch.allclose(default_record["p"], custom_record["p"], atol=0.0, rtol=0.0))
        self.assertEqual(float(default_loss.item()), float(custom_loss.item()))
        self.assertTrue(torch.allclose(default_model.vp.grad, custom_model.vp.grad, atol=1e-10, rtol=1e-8))

    def test_acoustic_remat_pressure_strategy_rejects_unsupported_options(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        propagator = AcousticPropagator(model, survey)

        with self.assertRaisesRegex(ValueError, "checkpoint_segments must be positive"):
            propagator.forward(
                checkpoint_segments=0,
                save_forward_wavefield=False,
                pressure_only=True,
                custom_chunk_strategy="remat_pressure_stride2",
            )
        with self.assertRaisesRegex(ValueError, "save_forward_wavefield=True"):
            propagator.forward(
                checkpoint_segments=1,
                save_forward_wavefield=True,
                pressure_only=True,
                custom_chunk_strategy="remat_pressure_stride2",
            )
        with self.assertRaisesRegex(ValueError, "requires pressure_only=True"):
            propagator.forward(
                checkpoint_segments=1,
                save_forward_wavefield=False,
                pressure_only=False,
                custom_chunk_strategy="remat_pressure_stride2",
            )

    def test_acoustic_fwi_rejects_skipped_forward_wavefield_when_illumination_is_active(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        propagator = AcousticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {"p": np.zeros((1, 8, 1), dtype=np.float32)}
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        fwi = AcousticFWI(
            propagator,
            model,
            optimizer,
            scheduler,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            gradient_processor=GradProcessor(forw_illumination=True),
            cache_result=False,
        )

        with self.assertRaisesRegex(ValueError, "forw_illumination=False"):
            fwi.forward(iteration=0, save_forward_wavefield=False)

    def test_acoustic_fwi_allows_skipped_forward_wavefield_without_illumination(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        propagator = AcousticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {"p": np.zeros((1, 8, 1), dtype=np.float32)}
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        fwi = AcousticFWI(
            propagator,
            model,
            optimizer,
            scheduler,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            gradient_processor=GradProcessor(forw_illumination=False),
            cache_result=False,
        )

        fwi.forward(iteration=0, save_forward_wavefield=False)

    def test_acoustic_fwi_skipped_forward_wavefield_matches_default_without_illumination(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp, rho = self._model_arrays()
        observed = SeismicData(survey)
        observed.data = {"p": np.zeros((1, 8, 1), dtype=np.float32)}

        def build_fwi():
            model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
            propagator = AcousticPropagator(model, survey)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            fwi = AcousticFWI(
                propagator,
                model,
                optimizer,
                scheduler,
                Misfit_waveform_L2(dt=survey.source.dt),
                observed,
                gradient_processor=GradProcessor(forw_illumination=False, norm_grad=False),
                waveform_normalize=False,
                cache_result=False,
            )
            return fwi

        default_fwi = build_fwi()
        skipped_fwi = build_fwi()

        default_fwi.forward(iteration=1, save_forward_wavefield=True)
        skipped_fwi.forward(iteration=1, save_forward_wavefield=False)

        self.assertEqual(default_fwi.iter_loss, skipped_fwi.iter_loss)
        self.assertTrue(torch.equal(default_fwi.model.vp, skipped_fwi.model.vp))
        self.assertTrue(torch.equal(default_fwi.model.vp.grad, skipped_fwi.model.vp.grad))

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


    def _elastic_fwi_for_api(self, inversion_component=None, component_weights=None):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp = np.ones((6, 8), dtype=np.float32) * 2200.0
        vs = np.ones((6, 8), dtype=np.float32) * 1200.0
        rho = np.ones((6, 8), dtype=np.float32) * 2000.0
        model = IsotropicElasticModel(0, 0, 8, 6, 10, 10, vp, vs, rho, vp_grad=True, auto_update_rho=False)
        propagator = ElasticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {
            "txx": np.zeros((1, 8, 1), dtype=np.float32),
            "tzz": np.zeros((1, 8, 1), dtype=np.float32),
            "vx": np.zeros((1, 8, 1), dtype=np.float32),
            "vz": np.zeros((1, 8, 1), dtype=np.float32),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        kwargs = {}
        if inversion_component is not None:
            kwargs["inversion_component"] = inversion_component
        if component_weights is not None:
            kwargs["component_weights"] = component_weights
        return ElasticFWI(
            propagator,
            model,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            optimizer=optimizer,
            scheduler=scheduler,
            waveform_normalize=False,
            cache_result=False,
            **kwargs,
        )

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

    def test_elastic_propagator_construction_preserves_survey_tensor_contracts(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._trace_missing_survey()
        vp = np.ones((6, 8), dtype=np.float32) * 2200.0
        vs = np.ones((6, 8), dtype=np.float32) * 1200.0
        rho = np.ones((6, 8), dtype=np.float32) * 2000.0
        model = IsotropicElasticModel(0, 0, 8, 6, 10, 10, vp, vs, rho, vp_grad=True, auto_update_rho=False)

        propagator = ElasticPropagator(model, survey)

        self.assertEqual(propagator.src_x.shape, (1,))
        self.assertEqual(propagator.src_z.shape, (1,))
        self.assertEqual(propagator.rcv_x.shape, (3,))
        self.assertEqual(propagator.rcv_z.shape, (3,))
        self.assertEqual(propagator.wavelet.shape, (1, survey.source.nt))
        self.assertEqual(propagator.moment_tensor.shape, (1, 3, 3))
        self.assertEqual(propagator.src_x.dtype, torch.long)
        self.assertEqual(propagator.rcv_x.dtype, torch.long)
        self.assertEqual(propagator.wavelet.dtype, propagator.dtype)
        self.assertEqual(propagator.moment_tensor.dtype, propagator.dtype)
        self.assertEqual(propagator.wavelet.device, propagator.device)
        self.assertTrue(np.array_equal(propagator.receiver_masks, survey.receiver_masks))
        self.assertFalse(propagator.receiver_masks_obs)

    def test_elastic_fwi_aligns_regularization_to_propagator_backend(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp = np.ones((6, 8), dtype=np.float32) * 2200.0
        vs = np.ones((6, 8), dtype=np.float32) * 1200.0
        rho = np.ones((6, 8), dtype=np.float32) * 2000.0
        model = IsotropicElasticModel(0, 0, 8, 6, 10, 10, vp, vs, rho, vp_grad=True, auto_update_rho=False)
        propagator = ElasticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {
            "txx": np.zeros((1, 8, 1), dtype=np.float32),
            "tzz": np.zeros((1, 8, 1), dtype=np.float32),
            "vx": np.zeros((1, 8, 1), dtype=np.float32),
            "vz": np.zeros((1, 8, 1), dtype=np.float32),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        reg = regularization_Tikhonov_1order(8, 6, 10, 10, alphax=1, alphaz=1, device="cpu", dtype=torch.float64)

        fwi = ElasticFWI(
            propagator,
            model,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            optimizer=optimizer,
            scheduler=scheduler,
            regularization_fn=reg,
            cache_result=False,
        )

        self.assertEqual(fwi.device, propagator.device)
        self.assertEqual(reg.device, propagator.device)
        self.assertEqual(reg.dtype, propagator.dtype)
        self.assertEqual(reg.L0.dtype, propagator.dtype)

    def test_elastic_fwi_rejects_model_propagator_device_mismatch(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp = np.ones((6, 8), dtype=np.float32) * 2200.0
        vs = np.ones((6, 8), dtype=np.float32) * 1200.0
        rho = np.ones((6, 8), dtype=np.float32) * 2000.0
        model = IsotropicElasticModel(0, 0, 8, 6, 10, 10, vp, vs, rho, vp_grad=True, auto_update_rho=False)
        propagator = ElasticPropagator(model, survey)
        model.device = torch.device("meta")
        obs_data = SeismicData(survey)
        obs_data.data = {
            "txx": np.zeros((1, 8, 1), dtype=np.float32),
            "tzz": np.zeros((1, 8, 1), dtype=np.float32),
            "vx": np.zeros((1, 8, 1), dtype=np.float32),
            "vz": np.zeros((1, 8, 1), dtype=np.float32),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        with self.assertRaisesRegex(ValueError, "device .* inconsistent"):
            ElasticFWI(
                propagator,
                model,
                Misfit_waveform_L2(dt=survey.source.dt),
                obs_data,
                optimizer=optimizer,
                scheduler=scheduler,
                cache_result=False,
            )

    def test_elastic_model_regularization_helper_matches_expanded_sum(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp = np.ones((6, 8), dtype=np.float32) * 2200.0
        vs = np.ones((6, 8), dtype=np.float32) * 1200.0
        rho = np.ones((6, 8), dtype=np.float32) * 2000.0
        model = IsotropicElasticModel(
            0, 0, 8, 6, 10, 10, vp, vs, rho,
            vp_grad=True, vs_grad=True, rho_grad=True, auto_update_rho=False,
        )
        propagator = ElasticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {
            "txx": np.zeros((1, 8, 1), dtype=np.float32),
            "tzz": np.zeros((1, 8, 1), dtype=np.float32),
            "vx": np.zeros((1, 8, 1), dtype=np.float32),
            "vz": np.zeros((1, 8, 1), dtype=np.float32),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        reg = DummyRegularization()

        fwi = ElasticFWI(
            propagator,
            model,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            optimizer=optimizer,
            scheduler=scheduler,
            regularization_fn=reg,
            regularization_weights_x=[1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            regularization_weights_z=[4.0, 5.0, 6.0, 0.0, 0.0, 0.0],
            cache_result=False,
        )

        loss = fwi.calculate_model_regularization_loss()
        expected = (
            torch.sum(fwi.model.vp * 1.0) + torch.sum(fwi.model.vp * 4.0 * 0.1)
            + torch.sum(fwi.model.vs * 2.0) + torch.sum(fwi.model.vs * 5.0 * 0.1)
            + torch.sum(fwi.model.rho * 3.0) + torch.sum(fwi.model.rho * 6.0 * 0.1)
        )

        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(reg.calls, [(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)])

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

    def test_acoustic_model_regularization_helper_matches_expanded_sum(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True, rho_grad=True)
        propagator = AcousticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {"p": np.zeros((1, 8, 1), dtype=np.float32)}
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        reg = DummyRegularization()

        fwi = AcousticFWI(
            propagator,
            model,
            optimizer,
            scheduler,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            regularization_fn=reg,
            regularization_weights_x=[1.0, 2.0],
            regularization_weights_z=[3.0, 4.0],
            cache_result=False,
        )

        loss = fwi.calculate_model_regularization_loss()
        expected = (
            torch.sum(fwi.model.vp * 1.0) + torch.sum(fwi.model.vp * 3.0 * 0.1)
            + torch.sum(fwi.model.rho * 2.0) + torch.sum(fwi.model.rho * 4.0 * 0.1)
        )

        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(reg.calls, [(1.0, 3.0), (2.0, 4.0)])

    def test_acoustic_fwi_can_apply_optional_data_transform_pipeline(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        propagator = AcousticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {"p": np.zeros((1, 8, 1), dtype=np.float32)}
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        pipeline = DataTransformPipeline([DataMask(torch.zeros((1, 8, 1)))])

        fwi = AcousticFWI(
            propagator,
            model,
            optimizer,
            scheduler,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            data_transform_pipeline=pipeline,
            waveform_normalize=False,
            cache_result=False,
        )

        synthetic = torch.ones((1, 8, 1), device=fwi.device, dtype=fwi.dtype)
        observed = synthetic * 2
        loss = fwi.calculate_loss(synthetic, observed, False, fwi.loss_fn, shot_index=np.array([0]))

        self.assertEqual(float(loss.detach().cpu().item()), 0.0)

    def test_acoustic_calculate_loss_selects_missing_receivers_when_applying_transforms(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._trace_missing_survey()
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        propagator = AcousticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {"p": np.zeros((1, 8, 2), dtype=np.float32)}
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        fwi = AcousticFWI(
            propagator,
            model,
            optimizer,
            scheduler,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            waveform_normalize=False,
            cache_result=False,
        )

        synthetic = torch.zeros((1, 8, 3), device=fwi.device, dtype=fwi.dtype)
        synthetic[..., 0] = 2.0
        synthetic[..., 2] = 2.0
        observed = torch.full((1, 8, 2), 2.0, device=fwi.device, dtype=fwi.dtype)
        loss = fwi.calculate_loss(synthetic, observed, False, fwi.loss_fn, shot_index=np.array([0]))

        self.assertEqual(float(loss.detach().cpu().item()), 0.0)

    def test_acoustic_fwi_default_normalization_uses_transform_pipeline(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp, rho = self._model_arrays()
        model = AcousticModel(0, 0, 8, 6, 10, 10, vp, rho, vp_grad=True)
        propagator = AcousticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {"p": np.zeros((1, 8, 1), dtype=np.float32)}
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        fwi = AcousticFWI(
            propagator,
            model,
            optimizer,
            scheduler,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            waveform_normalize=True,
            cache_result=False,
        )

        self.assertIsInstance(fwi.data_transform_pipeline, DataTransformPipeline)
        self.assertIsInstance(fwi.data_transform_pipeline.transforms[0], LegacyOffsetMute)
        self.assertIsInstance(fwi.data_transform_pipeline.transforms[1], LegacyLateWindowMute)
        self.assertIsInstance(fwi.data_transform_pipeline.transforms[2], LegacyLowPassFilter)
        self.assertIsInstance(fwi.data_transform_pipeline.transforms[3], DataMask)
        self.assertIsInstance(fwi.data_transform_pipeline.transforms[4], TraceNormalize)
        self.assertFalse(fwi.waveform_normalize)

        synthetic = torch.ones((1, 8, 1), device=fwi.device, dtype=fwi.dtype)
        observed = synthetic * 2
        loss = fwi.calculate_loss(synthetic, observed, fwi.waveform_normalize, fwi.loss_fn, shot_index=np.array([0]))

        self.assertEqual(float(loss.detach().cpu().item()), 0.0)

    def test_elastic_fwi_default_normalization_uses_transform_pipeline(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp = np.ones((6, 8), dtype=np.float32) * 2200.0
        vs = np.ones((6, 8), dtype=np.float32) * 1200.0
        rho = np.ones((6, 8), dtype=np.float32) * 2000.0
        model = IsotropicElasticModel(0, 0, 8, 6, 10, 10, vp, vs, rho, vp_grad=True, auto_update_rho=False)
        propagator = ElasticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {
            "txx": np.zeros((1, 8, 1), dtype=np.float32),
            "tzz": np.zeros((1, 8, 1), dtype=np.float32),
            "vx": np.zeros((1, 8, 1), dtype=np.float32),
            "vz": np.zeros((1, 8, 1), dtype=np.float32),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        fwi = ElasticFWI(
            propagator,
            model,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            optimizer=optimizer,
            scheduler=scheduler,
            waveform_normalize=True,
            cache_result=False,
        )

        self.assertIsInstance(fwi.data_transform_pipeline, DataTransformPipeline)
        self.assertIsInstance(fwi.data_transform_pipeline.transforms[0], LegacyOffsetMute)
        self.assertIsInstance(fwi.data_transform_pipeline.transforms[1], LegacyLateWindowMute)
        self.assertIsInstance(fwi.data_transform_pipeline.transforms[2], LegacyLowPassFilter)
        self.assertIsInstance(fwi.data_transform_pipeline.transforms[3], DataMask)
        self.assertIsInstance(fwi.data_transform_pipeline.transforms[4], TraceNormalize)
        self.assertFalse(fwi.waveform_normalize)

        synthetic = torch.ones((1, 8, 1), device=fwi.device, dtype=fwi.dtype)
        observed = synthetic * 2
        loss = fwi.calculate_loss(synthetic, observed, fwi.waveform_normalize, fwi.loss_fn, shot_index=np.array([0]))

        self.assertEqual(float(loss.detach().cpu().item()), 0.0)

    def test_elastic_calculate_loss_selects_missing_receivers_when_applying_transforms(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._trace_missing_survey()
        vp = np.ones((6, 8), dtype=np.float32) * 2200.0
        vs = np.ones((6, 8), dtype=np.float32) * 1200.0
        rho = np.ones((6, 8), dtype=np.float32) * 2000.0
        model = IsotropicElasticModel(0, 0, 8, 6, 10, 10, vp, vs, rho, vp_grad=True, auto_update_rho=False)
        propagator = ElasticPropagator(model, survey)
        obs_data = SeismicData(survey)
        obs_data.data = {
            "txx": np.zeros((1, 8, 2), dtype=np.float32),
            "tzz": np.zeros((1, 8, 2), dtype=np.float32),
            "vx": np.zeros((1, 8, 2), dtype=np.float32),
            "vz": np.zeros((1, 8, 2), dtype=np.float32),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        fwi = ElasticFWI(
            propagator,
            model,
            Misfit_waveform_L2(dt=survey.source.dt),
            obs_data,
            optimizer=optimizer,
            scheduler=scheduler,
            waveform_normalize=False,
            cache_result=False,
        )

        synthetic = torch.zeros((1, 8, 3), device=fwi.device, dtype=fwi.dtype)
        synthetic[..., 0] = 2.0
        synthetic[..., 2] = 2.0
        observed = torch.full((1, 8, 2), 2.0, device=fwi.device, dtype=fwi.dtype)
        loss = fwi.calculate_loss(synthetic, observed, False, fwi.loss_fn, shot_index=np.array([0]))

        self.assertEqual(float(loss.detach().cpu().item()), 0.0)

    def test_fwi_default_list_arguments_are_not_shared_between_instances(self):
        configure_backend("cpu", dtype=torch.float32)
        survey = self._survey()
        vp, rho = self._model_arrays()

        def acoustic_instance():
            model = AcousticModel(0, 0, 8, 6, 10, 10, vp.copy(), rho.copy(), vp_grad=True)
            propagator = AcousticPropagator(model, survey)
            obs_data = SeismicData(survey)
            obs_data.data = {"p": np.zeros((1, 8, 1), dtype=np.float32)}
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return AcousticFWI(
                propagator,
                model,
                optimizer,
                scheduler,
                Misfit_waveform_L2(dt=survey.source.dt),
                obs_data,
                cache_result=False,
            )

        first = acoustic_instance()
        second = acoustic_instance()
        first.regularization_weights_x[0] = 9

        self.assertEqual(second.regularization_weights_x, [0, 0])

        elastic_first = self._elastic_fwi_for_api()
        elastic_second = self._elastic_fwi_for_api()
        elastic_first.regularization_weights_x[0] = 9
        elastic_first.inversion_component.append("vx")

        self.assertEqual(elastic_second.regularization_weights_x, [0, 0, 0, 0, 0, 0])
        self.assertEqual(elastic_second.inversion_component, ["pressure"])
        self.assertEqual(elastic_second.component_weights, {"pressure": 1.0})

    def test_elastic_fwi_component_weights_default_to_active_components(self):
        fwi = self._elastic_fwi_for_api(inversion_component=["pressure", "vx"])

        self.assertEqual(fwi.component_weights, {"pressure": 1.0, "vx": 1.0})

    def test_elastic_fwi_accepts_explicit_component_weights(self):
        fwi = self._elastic_fwi_for_api(
            inversion_component=["pressure", "vx", "vz"],
            component_weights={"pressure": 2.0, "vz": 0.25},
        )

        self.assertEqual(fwi.component_weights, {"pressure": 2.0, "vx": 1.0, "vz": 0.25})

    def test_elastic_fwi_rejects_invalid_component_weights(self):
        with self.assertRaises(ValueError):
            self._elastic_fwi_for_api(inversion_component=["pressure", "ux"])
        with self.assertRaises(ValueError):
            self._elastic_fwi_for_api(inversion_component=["pressure"], component_weights={"ux": 1.0})
        with self.assertRaises(ValueError):
            self._elastic_fwi_for_api(inversion_component=["pressure"], component_weights={"pressure": -1.0})


if __name__ == "__main__":
    unittest.main()
