import numpy as np
import pytest

import drjit as dr
import mitsuba as mi


mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")

import mitransient  # noqa: E402,F401 -- registers the plugins for this variant
from mitransient.integrators.transientpath import TransientPath  # noqa: E402
from mitransient.pulses.histogram_pulse import HistogramPulse  # noqa: E402
from mitransient.render.transient_image_block import TransientImageBlock  # noqa: E402


def _as_numpy(value):
    return np.asarray(value, dtype=np.float64)


def test_wall_uv_uses_pixel_centers_and_stays_in_bounds():
    film = mi.load_dict({
        "type": "transient_hdr_film",
        "width": 4,
        "height": 2,
        "temporal_bins": 1,
    })
    film_pos = mi.Point2f(
        mi.Float([0.01, 0.99, 3.01, 3.99]),
        mi.Float([0.01, 0.99, 1.01, 1.99]),
    )

    uv = TransientPath._wall_uv_from_film_pos(film_pos, film)
    u = _as_numpy(uv.x)
    v = _as_numpy(uv.y)

    np.testing.assert_allclose(u, [0.25, 0.25, 0.75, 0.75])
    np.testing.assert_allclose(v, [0.875, 0.875, 0.125, 0.125])
    assert np.all((u > 0.0) & (u < 1.0))
    assert np.all((v > 0.0) & (v < 1.0))


def test_retroreflector_eval_includes_outgoing_cosine():
    bsdf = mi.load_dict({
        "type": "retroreflector",
        "reflectance": 1.0,
        "lobe_exponent": 0.0,
    })
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = mi.Vector3f(0.0, 0.0, 1.0)
    ctx = mi.BSDFContext()
    wo_normal = mi.Vector3f(0.0, 0.0, 1.0)
    wo_grazing = mi.Vector3f(float(np.sqrt(0.75)), 0.0, 0.5)

    value_normal = _as_numpy(bsdf.eval(ctx, si, wo_normal, True))
    value_grazing, pdf_grazing = bsdf.eval_pdf(ctx, si, wo_grazing, True)
    value_grazing = _as_numpy(value_grazing)

    np.testing.assert_allclose(value_grazing, 0.5 * value_normal, rtol=1e-6)
    np.testing.assert_allclose(
        value_grazing,
        _as_numpy(bsdf.eval(ctx, si, wo_grazing, True)),
        rtol=1e-6,
    )
    assert np.all(_as_numpy(pdf_grazing) > 0.0)


def test_projector_pmf_uses_all_rgb_channels():
    projector = mi.load_dict({
        "type": "confocal_projector",
        "spot_positions": mi.TensorXf([[0.0, 0.0], [0.5, 0.5]]),
        "spot_sigmas": mi.TensorXf([0.1, 0.1]),
        "spot_intensities": mi.TensorXf([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        "is_confocal": True,
    })
    np.testing.assert_allclose(_as_numpy(projector.spot_pmf), [0.5, 0.5])


def test_zero_energy_projector_has_valid_proposal():
    projector = mi.load_dict({
        "type": "confocal_projector",
        "spot_positions": mi.TensorXf([[0.0, 0.0], [0.5, 0.5]]),
        "spot_sigmas": mi.TensorXf([0.1, 0.1]),
        "spot_intensities": mi.TensorXf([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "is_confocal": True,
    })
    np.testing.assert_allclose(_as_numpy(projector.spot_pmf), [0.5, 0.5])
    np.testing.assert_allclose(_as_numpy(projector.spot_cdf), [0.5, 1.0])


def test_film_clear_is_safe_before_prepare():
    film = mi.load_dict({
        "type": "transient_hdr_film",
        "width": 1,
        "height": 1,
        "temporal_bins": 1,
    })
    film.clear()


def test_vector_invalid_diagnostics_do_not_scalarize_masks():
    block = TransientImageBlock(
        mi.ScalarVector3u(2, 1, 1),
        mi.ScalarPoint3i(0),
        4,
        mi.load_dict({"type": "box"}),
        warn_invalid=True,
    )
    block.put_(
        mi.Point3f(mi.Float([0.5, 1.5]), 0.5, 0.5),
        [
            mi.Float([1.0, float("nan")]),
            mi.Float(1.0),
            mi.Float(1.0),
            mi.Float(0.0),
        ],
        mi.Bool(True),
    )


def test_all_zero_histogram_pulse_samples_zero_weight():
    pulse = HistogramPulse([0.0, 0.0], start_opl=-1.0, bin_width_opl=1.0)
    sample_time, weight = pulse.sample(mi.Float([0.1, 0.9]))

    np.testing.assert_allclose(_as_numpy(sample_time), [-1.0, -1.0])
    np.testing.assert_allclose(_as_numpy(weight), [0.0, 0.0])
    np.testing.assert_allclose(
        _as_numpy(pulse.eval(mi.Float([-0.5, 0.5]))),
        [0.0, 0.0],
    )


def _make_one_bounce_projector_scene(integrator_type, spp):
    transform = mi.ScalarTransform4f
    integrator = {
        "type": integrator_type,
        "max_depth": 2,
        "confocal_projector": {
            "type": "confocal_projector",
            "grid_rows": 1,
            "grid_cols": 1,
            "grid_sigma": 0.001,
            "grid_intensity": 1.0,
            "fov": 1.0,
            "is_confocal": True,
            "pulse_width_opl": 0.02,
        },
    }
    if integrator_type == "timegated_transient_path":
        integrator["time_sampling"] = "random"

    return mi.load_dict({
        "type": "scene",
        "integrator": integrator,
        "sensor": {
            "type": "perspective",
            "fov": 1.0,
            "to_world": transform.look_at(
                origin=[0, 0, 0],
                target=[0, 0, -1],
                up=[0, 1, 0],
            ),
            "sampler": {"type": "independent", "sample_count": spp},
            "film": {
                "type": "transient_hdr_film",
                "width": 1,
                "height": 1,
                "temporal_bins": 64,
                "start_opl": 3.84,
                "bin_width_opl": 0.005,
                "rfilter": {"type": "box"},
            },
        },
        "wall": {
            "type": "rectangle",
            "to_world": transform.translate([0, 0, -2]).scale([2, 2, 1]),
            "bsdf": {"type": "diffuse", "reflectance": 0.8},
        },
    })


@pytest.mark.slow
def test_timegated_energy_matches_single_projector_estimator():
    spp = 32768
    energy = {}
    for integrator_type in ("transient_path", "timegated_transient_path"):
        scene = _make_one_bounce_projector_scene(integrator_type, spp)
        steady, transient = mi.render(scene, spp=spp, seed=7)
        dr.eval(steady, transient)
        energy[integrator_type] = float(steady.array[0])

    ratio = energy["timegated_transient_path"] / energy["transient_path"]
    assert ratio == pytest.approx(1.0, rel=0.03)
