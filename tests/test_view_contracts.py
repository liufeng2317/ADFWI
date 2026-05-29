import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ADFWI.view import (  # noqa: E402
    animate_inversion_process,
    plot_bcx_bcz,
    plot_damp,
    plot_eps_delta_gamma,
    plot_initial_and_inverted,
    plot_lam_mu,
    plot_misfit,
    plot_model,
    plot_survey,
    plot_vp_rho,
    plot_vp_vs_rho,
    plot_waveform2D,
    plot_waveform_trace,
    plot_waveform_wiggle,
    plot_wavelet,
)


class ViewPlotContractTests(unittest.TestCase):
    def setUp(self):
        plt.close("all")
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)

    def tearDown(self):
        plt.close("all")
        self.tmpdir.cleanup()

    def assert_saved_and_closed(self, filename, plot_call):
        path = self.output_dir / filename
        plot_call(path)
        self.assertTrue(path.exists(), f"missing saved figure: {path}")
        self.assertEqual(plt.get_fignums(), [])

    def test_view_package_exports_public_plotting_helpers(self):
        import ADFWI.view as view

        expected_names = {
            "animate_inversion_process",
            "plot_bcx_bcz",
            "plot_damp",
            "plot_eps_delta_gamma",
            "plot_initial_and_inverted",
            "plot_lam_mu",
            "plot_misfit",
            "plot_model",
            "plot_survey",
            "plot_vp_rho",
            "plot_vp_vs_rho",
            "plot_waveform2D",
            "plot_waveform_trace",
            "plot_waveform_wiggle",
            "plot_wavelet",
        }

        self.assertEqual(set(view.__all__), expected_names)
        for name in expected_names:
            self.assertTrue(callable(getattr(view, name)), name)

    def test_velocity_model_plot_helpers_save_and_close(self):
        base = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0

        self.assert_saved_and_closed(
            "vp_rho.png",
            lambda path: plot_vp_rho(base, base + 10, show=False, save_path=str(path)),
        )
        self.assert_saved_and_closed(
            "vp_vs_rho.png",
            lambda path: plot_vp_vs_rho(base, base + 1, base + 2, show=False, save_path=str(path)),
        )
        self.assert_saved_and_closed(
            "eps_delta_gamma.png",
            lambda path: plot_eps_delta_gamma(base, base + 1, base + 2, show=False, save_path=str(path)),
        )
        self.assert_saved_and_closed(
            "lam_mu.png",
            lambda path: plot_lam_mu(base, base + 1, show=False, save_path=str(path)),
        )
        self.assert_saved_and_closed(
            "model.png",
            lambda path: plot_model(base, "vp", show=False, save_path=str(path)),
        )

    def test_plot_eps_delta_gamma_draws_one_image_per_panel_without_spacing(self):
        eps = np.ones((3, 4), dtype=np.float32)
        delta = eps * 2
        gamma = eps * 3

        plot_eps_delta_gamma(eps, delta, gamma, dx=-1, dz=-1, show=True)

        fig = plt.gcf()
        main_axes = fig.axes[:3]
        image_counts = [len(axis.images) for axis in main_axes]
        self.assertEqual(image_counts, [1, 1, 1])

    def test_boundary_survey_and_wavelet_plot_helpers_save_and_close(self):
        array = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
        src_x = np.array([1, 2])
        src_z = np.array([0, 0])
        rcv_x = np.array([0, 1, 2, 3])
        rcv_z = np.array([1, 1, 1, 1])
        t = np.linspace(0.0, 1.0, 16)

        self.assert_saved_and_closed(
            "bcx_bcz.png",
            lambda path: plot_bcx_bcz(array, array + 1, show=False, save_path=str(path)),
        )
        self.assert_saved_and_closed(
            "damp.png",
            lambda path: plot_damp(array, show=False, save_path=str(path)),
        )
        self.assert_saved_and_closed(
            "survey.png",
            lambda path: plot_survey(src_x, src_z, rcv_x, rcv_z, array, show=False, save_path=str(path)),
        )
        self.assert_saved_and_closed(
            "wavelet.png",
            lambda path: plot_wavelet(t, np.sin(t), show=False, save_path=str(path)),
        )

    def test_waveform_plot_helpers_save_and_close(self):
        waveform3d = np.arange(2 * 16 * 4, dtype=np.float32).reshape(2, 16, 4)
        section = waveform3d[0].T
        wiggle = waveform3d[0]

        self.assert_saved_and_closed(
            "trace.png",
            lambda path: plot_waveform_trace(waveform3d, shot=0, trace=1, show=False, save_path=str(path)),
        )
        self.assert_saved_and_closed(
            "section.png",
            lambda path: plot_waveform2D(section, show=False, save_path=str(path)),
        )
        self.assert_saved_and_closed(
            "wiggle.png",
            lambda path: plot_waveform_wiggle(wiggle, show=False, save_path=str(path)),
        )

    def test_waveform_wiggle_uses_a_fresh_figure(self):
        existing = plt.figure()
        plt.imshow(np.ones((2, 2), dtype=np.float32))
        existing_number = existing.number

        path = self.output_dir / "wiggle_fresh.png"
        plot_waveform_wiggle(
            np.arange(16 * 4, dtype=np.float32).reshape(16, 4),
            show=False,
            save_path=str(path),
        )

        self.assertTrue(path.exists())
        self.assertIn(existing_number, plt.get_fignums())

    def test_inversion_plot_helpers_save_and_close(self):
        vp_init = np.arange(12, dtype=np.float32).reshape(3, 4) + 1500.0
        iter_vp = np.stack([vp_init, vp_init + 10.0])

        self.assert_saved_and_closed(
            "misfit.png",
            lambda path: plot_misfit([3.0, 2.0, 1.0], show=False, save_path=str(path)),
        )
        self.assert_saved_and_closed(
            "initial_inverted.png",
            lambda path: plot_initial_and_inverted(vp_init, iter_vp, show=False, save_path=str(path)),
        )

        gif_path = self.output_dir / "inversion.gif"
        animate_inversion_process(iter_vp, save_path=str(gif_path), fps=1, interval=10)
        self.assertTrue(gif_path.exists())
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
