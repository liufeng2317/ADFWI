import unittest

import torch

from ADFWI.fwi.misfit import Misfit_waveform_L2, Misfit_waveform_SquaredL2


class SquaredL2MisfitTests(unittest.TestCase):
    def test_squared_l2_exact_match_has_zero_finite_gradient(self):
        obs = torch.ones(1, 4, 2, dtype=torch.float32)
        syn = obs.clone().detach().requires_grad_(True)

        loss = Misfit_waveform_SquaredL2(dt=0.5).forward(obs, syn)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertIsNotNone(syn.grad)
        self.assertTrue(torch.isfinite(syn.grad).all())
        self.assertTrue(torch.equal(syn.grad, torch.zeros_like(syn.grad)))

    def test_squared_l2_sum_matches_classical_half_sum_formula(self):
        obs = torch.tensor([[[1.0], [3.0]]])
        syn = torch.tensor([[[0.0], [1.0]]], requires_grad=True)

        loss = Misfit_waveform_SquaredL2(dt=2.0, reduction="sum").forward(obs, syn)
        loss.backward()

        self.assertTrue(torch.equal(loss.detach(), torch.tensor(5.0)))
        expected_grad = torch.tensor([[[-2.0], [-4.0]]])
        self.assertTrue(torch.equal(syn.grad, expected_grad))

    def test_squared_l2_mean_matches_mean_squared_formula(self):
        obs = torch.tensor([[[1.0], [3.0]]])
        syn = torch.tensor([[[0.0], [1.0]]], requires_grad=True)

        loss = Misfit_waveform_SquaredL2(dt=2.0, reduction="mean").forward(obs, syn)
        loss.backward()

        self.assertTrue(torch.equal(loss.detach(), torch.tensor(5.0)))
        expected_grad = torch.tensor([[[-2.0], [-4.0]]])
        self.assertTrue(torch.equal(syn.grad, expected_grad))

    def test_legacy_l2_exact_match_documents_nan_gradient(self):
        obs = torch.ones(1, 4, 1, dtype=torch.float32)
        syn = obs.clone().detach().requires_grad_(True)

        loss = Misfit_waveform_L2(dt=1.0).forward(obs, syn)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertFalse(torch.isfinite(syn.grad).all())

    def test_squared_l2_rejects_unknown_reduction(self):
        with self.assertRaises(ValueError):
            Misfit_waveform_SquaredL2(reduction="none")


if __name__ == "__main__":
    unittest.main()
