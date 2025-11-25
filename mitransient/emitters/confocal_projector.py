from __future__ import annotations

from typing import Tuple

import drjit as dr
import mitsuba as mi
import numpy as np


class ConfocalProjector(mi.Emitter):
    r"""
    .. emitter-confocal_projector:

    Confocal Projector (:monosp:`confocal_projector`)
    -------------------------------------------------

    This emitter implements a parametric projector with Gaussian spot patterns,
    suitable for confocal and NLOS imaging setups.

    The projector can be configured in two modes:

    1. **Grid Mode**: Specify grid dimensions and the spots are automatically
       generated in a uniform or random pattern.

    2. **Explicit Mode**: Provide explicit arrays for spot positions, sigmas,
       and intensities.

    .. tabs::

        .. code-tab:: python

            # Grid mode
            {
                'type': 'confocal_projector',
                'grid_rows': 3,
                'grid_cols': 3,
                'grid_sigma': 0.01,
                'grid_intensity': 1000.0,
                'fov': 0.5,
                'is_confocal': True,
            }

        .. code-tab:: python

            # Explicit mode
            {
                'type': 'confocal_projector',
                'spot_positions': mi.TensorXf([[0, 0], [0.5, 0.5]]),
                'spot_sigmas': mi.TensorXf([0.01, 0.01]),
                'spot_intensities': mi.TensorXf([[1000, 1000, 1000], [500, 500, 500]]),
                'fov': 0.5,
                'is_confocal': False,
                'frame': projector_frame,
            }

    .. pluginparameters::

     * - is_confocal
       - |bool|
       - If True, dynamically computes the projector frame from the camera ray
         direction. If False, uses a static frame. (default: True)

     * - fov
       - |float|
       - Field of view in radians for the projector. (default: 0.2)

     * - frame
       - |transform|
       - 3x3 rotation matrix defining the projector's coordinate frame.
         Columns represent [right, up, forward] vectors in world space.
         Only used when is_confocal=False. (default: None)

     * - max_rejection_samples
       - |int|
       - Maximum number of rejection sampling iterations when sampling from
         Gaussian spots. (default: 8)

     * - grid_rows
       - |int|
       - Number of Gaussian spots along Y dimension (grid mode). (default: None)

     * - grid_cols
       - |int|
       - Number of Gaussian spots along X dimension (grid mode). (default: None)

     * - grid_sigma
       - |float|
       - Standard deviation for all spots in normalized coordinates (grid mode). (default: 0.01)

     * - grid_intensity
       - |float|
       - Intensity per spot in RGB (grid mode). (default: 1000.0)

     * - grid_spacing
       - |string|
       - 'uniform' or 'random' positioning (grid mode). (default: 'uniform')

     * - grid_seed
       - |int|
       - Random seed for reproducible random positions (grid mode). (default: None)

     * - spot_positions
       - |array|
       - Nx2 array of spot positions in normalized coordinates [-1, 1] (explicit mode).

     * - spot_sigmas
       - |array|
       - N-element array of standard deviations (explicit mode).

     * - spot_intensities
       - |array|
       - Nx3 array of RGB intensities (explicit mode).
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)

        # Core parameters
        self.is_confocal = props.get("is_confocal", True)
        self.fov = props.get("fov", 0.2)
        self.frame = props.get("frame", None)
        self.max_rejection_samples = props.get("max_rejection_samples", 8)

        # Check for explicit spot mode
        spot_positions = props.get("spot_positions", None)
        spot_sigmas = props.get("spot_sigmas", None)
        spot_intensities = props.get("spot_intensities", None)

        # Check for grid mode
        grid_rows = props.get("grid_rows", None)
        grid_cols = props.get("grid_cols", None)

        if spot_positions is not None and spot_sigmas is not None and spot_intensities is not None:
            # Explicit spot mode
            self._init_from_explicit(spot_positions, spot_sigmas, spot_intensities)
        elif grid_rows is not None and grid_cols is not None:
            # Grid mode
            grid_sigma = props.get("grid_sigma", 0.01)
            grid_intensity = props.get("grid_intensity", 1000.0)
            grid_spacing = props.get("grid_spacing", "uniform")
            grid_seed = props.get("grid_seed", None)
            self._init_from_grid(grid_rows, grid_cols, grid_sigma, grid_intensity, grid_spacing, grid_seed)
        else:
            raise ValueError(
                "ConfocalProjector requires either explicit spot arrays "
                "(spot_positions, spot_sigmas, spot_intensities) or grid parameters "
                "(grid_rows, grid_cols)"
            )

        # Prepare DrJit arrays
        self._prepare()

        self.m_flags = mi.EmitterFlags.Surface

    def _init_from_explicit(self, positions, sigmas, intensities):
        """Initialize from explicit spot arrays."""
        self._spot_positions_np = np.asarray(positions)
        self._spot_sigmas_np = np.asarray(sigmas)
        self._spot_intensities_np = np.asarray(intensities)

    def _init_from_grid(self, grid_rows, grid_cols, sigma, intensity, spacing, seed):
        """Initialize from grid parameters."""
        positions, sigmas, intensities = self._create_parametric_grid(
            grid_rows, grid_cols, sigma, intensity, spacing, seed
        )
        self._spot_positions_np = positions
        self._spot_sigmas_np = sigmas
        self._spot_intensities_np = intensities

    @staticmethod
    def _create_parametric_grid(grid_rows, grid_cols, sigma, intensity, spacing='uniform', random_seed=None):
        """
        Create parametric projector parameters for a grid of Gaussian spots.

        Args:
            grid_rows: Number of Gaussians along Y dimension
            grid_cols: Number of Gaussians along X dimension
            sigma: Standard deviation of each Gaussian in normalized coordinates [-1, 1]
            intensity: Intensity of each spot (RGB uniform)
            spacing: 'uniform' for evenly spaced grid, 'random' for random positions
            random_seed: Random seed for reproducible random positions

        Returns:
            positions: Nx2 array of spot positions in normalized coordinates [-1, 1]
            sigmas: N-element array of standard deviations
            intensities: Nx3 array of RGB intensities
        """
        if spacing == 'uniform':
            # Create uniformly spaced grid in normalized coordinates [-1, 1]
            margin_x = (1.0 / grid_cols if grid_cols > 1 else 0.0)
            margin_y = (1.0 / grid_rows if grid_rows > 1 else 0.0)
            x_positions = np.linspace(-1 + margin_x, 1 - margin_x, grid_cols) if grid_cols > 1 else np.array([0.0])
            y_positions = np.linspace(-1 + margin_y, 1 - margin_y, grid_rows) if grid_rows > 1 else np.array([0.0])

            # Create all combinations
            xx, yy = np.meshgrid(x_positions, y_positions)
            positions = np.stack([xx.flatten(), yy.flatten()], axis=-1)

        elif spacing == 'random':
            if random_seed is not None:
                np.random.seed(random_seed)

            num_spots = grid_rows * grid_cols
            margin = 3 * sigma
            positions = np.random.uniform(-1 + margin, 1 - margin, (num_spots, 2))
        else:
            raise ValueError(f"Unknown spacing mode: {spacing}. Use 'uniform' or 'random'.")

        num_spots = positions.shape[0]
        sigmas = np.full(num_spots, sigma)
        intensities = np.full((num_spots, 3), intensity)

        return positions, sigmas, intensities

    def _prepare(self):
        """
        Prepare parametric projector from Gaussian spot parameters.
        Sets up DrJit arrays for efficient vectorized evaluation and sampling.
        """
        positions = self._spot_positions_np
        sigmas = self._spot_sigmas_np
        intensities = self._spot_intensities_np

        self.num_spots = positions.shape[0]

        # Store spot parameters as DrJit arrays
        self.spot_positions_x = mi.Float(positions[:, 0])
        self.spot_positions_y = mi.Float(positions[:, 1])
        self.spot_sigmas = mi.Float(sigmas)
        self.spot_intensities = mi.Color3f(intensities[:, 0], intensities[:, 1], intensities[:, 2])

        # Compute PMF and CDF for spot selection
        # Integral of 2D Gaussian = 2 * pi * sigma^2 * amplitude
        integrated = intensities[:, 0]
        total = np.sum(integrated)
        pmf = integrated / np.maximum(total, 1e-10)
        self.spot_cdf = mi.Float(np.cumsum(pmf))
        self.spot_pmf = mi.Float(pmf)

    def eval_pattern(self,
                     normalized_x: mi.Float,
                     normalized_y: mi.Float) -> mi.Spectrum:
        """
        Evaluate the parametric projector (Gaussian mixture) at normalized coordinates.

        Args:
            normalized_x: X coordinate in [-1, 1]
            normalized_y: Y coordinate in [-1, 1]

        Returns:
            RGB spectrum value from the Gaussian mixture
        """
        result = mi.Color3f(0.0)

        for i in range(self.num_spots):
            idx = mi.UInt32(i)
            cx = dr.gather(mi.Float, self.spot_positions_x, idx)
            cy = dr.gather(mi.Float, self.spot_positions_y, idx)
            sigma = dr.gather(mi.Float, self.spot_sigmas, idx)
            intensity = dr.gather(mi.Color3f, self.spot_intensities, idx)

            dx = normalized_x - cx
            dy = normalized_y - cy
            dist_sq = dx * dx + dy * dy
            gaussian = dr.exp(-dist_sq / (2.0 * sigma * sigma)) * dr.rcp(2.0 * dr.pi * sigma * sigma)
            result += intensity * gaussian

        return mi.Spectrum(result)

    def sample_spot(self, sampler: mi.Sampler) -> Tuple[mi.UInt32, mi.Float, mi.Float, mi.Float]:
        """
        Sample a spot from the parametric projector with rejection sampling
        to avoid samples outside the FOV (normalized coords outside [-1, 1]).

        Returns:
            spot_idx: Index of sampled spot
            sample_x: X coordinate sampled from the Gaussian (clamped to [-1, 1])
            sample_y: Y coordinate sampled from the Gaussian (clamped to [-1, 1])
            pdf: PDF of the sample (spot selection * Gaussian pdf)
        """
        xi = sampler.next_1d()
        spot_idx = dr.clip(
            dr.binary_search(0, self.num_spots - 1,
                           lambda idx: dr.gather(mi.Float, self.spot_cdf, idx) < xi),
            0, self.num_spots - 1
        )

        cx = dr.gather(mi.Float, self.spot_positions_x, spot_idx)
        cy = dr.gather(mi.Float, self.spot_positions_y, spot_idx)
        sigma = dr.gather(mi.Float, self.spot_sigmas, spot_idx)

        # Initial sample using Box-Muller transform
        u1 = dr.clip(sampler.next_1d(), 1e-10, 1.0)
        u2 = sampler.next_1d()
        r = sigma * dr.sqrt(-2.0 * dr.log(u1))
        theta = 2.0 * dr.pi * u2

        sample_x = cx + r * dr.cos(theta)
        sample_y = cy + r * dr.sin(theta)

        # Final clamp for any remaining outliers
        sample_x = dr.clip(sample_x, -1.0, 1.0)
        sample_y = dr.clip(sample_y, -1.0, 1.0)

        # PDF: P(spot) * P(position|spot)
        spot_pmf = dr.gather(mi.Float, self.spot_pmf, spot_idx)
        gaussian_pdf = dr.rcp(2.0 * dr.pi * sigma * sigma)
        pdf = spot_pmf * gaussian_pdf

        return spot_idx, sample_x, sample_y, pdf

    def build_frame(self, camera_ray_direction: mi.Vector3f = None) -> mi.Matrix3f:
        """
        Build orthonormal frame for projector.

        If is_confocal is False and frame was provided in constructor,
        returns that stored frame. Otherwise, builds a frame aligned with
        the camera ray direction.

        Args:
            camera_ray_direction: Initial camera ray direction (required when
                                  computing frame dynamically)

        Returns:
            3x3 rotation matrix (columns: right, up, forward)
        """
        # Only use stored projector frame when NOT in confocal mode
        if not self.is_confocal and self.frame is not None:
            return self.frame

        # Otherwise, compute from camera ray direction (for confocal mode or when frame not provided)
        if camera_ray_direction is None:
            raise ValueError("camera_ray_direction required when frame is not set or in confocal mode")

        camera_forward = -camera_ray_direction
        camera_up = mi.Vector3f(0.0, 1.0, 0.0)
        camera_right = dr.normalize(dr.cross(camera_forward, camera_up))
        camera_up = dr.normalize(dr.cross(camera_right, camera_forward))
        return mi.Matrix3f(camera_right, camera_up, camera_forward)

    def query_direct(self,
                     scene: mi.Scene,
                     si: mi.SurfaceInteraction3f,
                     camera_origin: mi.Point3f,
                     camera_ray_direction: mi.Vector3f,
                     active: mi.Bool) -> Tuple[mi.DirectionSample3f, mi.Spectrum]:
        """
        Query incoming radiance directly from the parametric projector for the initial
        intersection point. Uses inverse square falloff, no importance sampling.

        Args:
            scene: The scene
            si: Current surface interaction (should be at initial intersection point)
            camera_origin: Camera origin point (projector position)
            camera_ray_direction: Initial camera ray direction
            active: Active mask

        Returns:
            ds: Direction sample toward the projector
            em_weight: Weight including radiance with inverse square falloff
        """
        # Direction from intersection point to projector
        to_projector = camera_origin - si.p
        dist = dr.norm(to_projector)
        to_projector_normalized = dr.normalize(to_projector)

        # Transform incoming direction to projector local space
        world_to_projector = self.build_frame(camera_ray_direction)
        direction_local = world_to_projector @ (-to_projector_normalized)

        # Convert to normalized coordinates [-1, 1]
        tan_half_fov = dr.tan(self.fov / 2.0)
        normalized_x = direction_local.x / (-direction_local.z * tan_half_fov)
        normalized_y = direction_local.y / (-direction_local.z * tan_half_fov)

        # Query parametric projector (Gaussian mixture)
        projector_radiance = self.eval_pattern(normalized_x, normalized_y)

        # Apply inverse square falloff
        falloff = dr.minimum(dr.rcp(dr.maximum(dist * dist, 1e-3)), 10000.0)

        # Create direction sample pointing toward projector
        ds = dr.zeros(mi.DirectionSample3f)
        ds.p = camera_origin
        ds.d = to_projector_normalized
        ds.dist = dist
        ds.pdf = 1.0
        ds.delta = False

        return ds, mi.Spectrum(projector_radiance * falloff)

    def sample_emitter(self,
                       scene: mi.Scene,
                       si: mi.SurfaceInteraction3f,
                       camera_origin: mi.Point3f,
                       camera_ray_direction: mi.Vector3f,
                       sampler: mi.Sampler,
                       active: mi.Bool) -> Tuple[mi.DirectionSample3f, mi.Spectrum]:
        """
        Sample emitter using parametric Gaussian spot importance sampling.

        Samples a point from one of the Gaussian spots, traces a ray from the projector
        to the scene, and computes the contribution to the current vertex.

        Args:
            scene: The scene
            si: Current surface interaction
            camera_origin: Camera origin point (projector position)
            camera_ray_direction: Initial camera ray direction
            sampler: Sampler for random numbers
            active: Active mask

        Returns:
            ds: Direction sample toward the illuminated point
            em_weight: Weight including BRDF, radiance, and PDF
        """
        # Sample a point from the parametric projector
        spot_idx, normalized_x, normalized_y, sample_pdf = self.sample_spot(sampler)

        # Compute direction in projector local space
        tan_half_fov = dr.tan(self.fov / 2.0)
        direction_local = dr.normalize(mi.Vector3f(
            tan_half_fov * normalized_x,
            tan_half_fov * normalized_y,
            -1.0
        ))

        # Transform to world space and find intersection
        projector_to_world = self.build_frame(camera_ray_direction).T
        direction_world = projector_to_world @ direction_local
        si_projector = scene.ray_intersect(mi.Ray3f(camera_origin, direction_world), active)
        dist_projector = dr.norm(si_projector.p - camera_origin)

        # Direction from current vertex to the illuminated point
        to_light = si_projector.p - si.p
        dist = dr.norm(to_light)
        to_light_normalized = dr.normalize(to_light)

        # Check visibility from current vertex to illuminated point
        shadow_ray = mi.Ray3f(si.p + to_light_normalized * 1e-4, to_light_normalized)
        si_shadow = scene.ray_intersect(shadow_ray, active)
        is_visible = (dr.norm(si_shadow.p - si_projector.p) < 1e-4)

        # Evaluate BSDF at the illuminated point (reflection toward current vertex)
        bsdf_ctx = mi.BSDFContext()
        wo_projector = si_projector.to_local(-to_light_normalized)
        bsdf_value = si_projector.bsdf().eval(bsdf_ctx, si_projector, wo_projector, active)
        bsdf_value = si_projector.to_world_mueller(bsdf_value, -wo_projector, si_projector.wi)

        # Get projector radiance at the sampled point
        projector_radiance = self.eval_pattern(normalized_x, normalized_y)

        # Compute geometry terms for PDF conversion
        # We sampled in projector's image plane solid angle, need to convert to
        # the solid angle as seen from the current vertex

        # Solid angle element in projector space
        cos_theta_projector = dr.abs(dr.dot(si_projector.n, -direction_world))
        d_area = (tan_half_fov * tan_half_fov) / dr.maximum(cos_theta_projector, 1e-6)

        # Convert to solid angle at current vertex
        cos_theta_vertex = dr.abs(dr.dot(si_projector.n, -to_light_normalized))
        d_omega_vertex = d_area * cos_theta_vertex / dr.maximum(dist * dist, 1e-3)

        # Weight includes: radiance / pdf * geometry
        pdf_weight = dr.clip(d_omega_vertex / dr.maximum(sample_pdf, 1e-10), 0.0, 10000.0)

        # Fill in direction sample
        ds = dr.zeros(mi.DirectionSample3f)
        ds.p = si_projector.p
        ds.d = to_light_normalized
        ds.dist = dist + dist_projector
        ds.pdf = dr.select(pdf_weight > 0, 1.0 / pdf_weight, 0.0)
        ds.delta = False

        em_weight = dr.select(
            is_visible & active & si_projector.is_valid(),
            projector_radiance * bsdf_value * pdf_weight,
            mi.Spectrum(0.0)
        )
        return ds, em_weight

    def traverse(self, callback: mi.TraversalCallback):
        """Expose parameters for traversal (e.g., for optimization)."""
        super().traverse(callback)
        callback.put("fov", self.fov, mi.ParamFlags.NonDifferentiable)
        callback.put("is_confocal", self.is_confocal, mi.ParamFlags.NonDifferentiable)

    def eval(self, si: mi.SurfaceInteraction3f, active: mi.Bool) -> mi.Spectrum:
        """Standard emitter eval - not typically used for confocal projector."""
        return mi.Spectrum(0.0)

    def sample_ray(self, time: mi.Float, sample1: mi.Float, sample2: mi.Float,
                   sample3: mi.Float, active: mi.Bool) -> Tuple[mi.Ray3f, mi.Spectrum]:
        raise NotImplementedError("ConfocalProjector does not support sample_ray")

    def sample_direction(self, ref: mi.Interaction3f, sample: mi.Point2f,
                         active: mi.Bool) -> Tuple[mi.DirectionSample3f, mi.Spectrum]:
        raise NotImplementedError("ConfocalProjector does not support sample_direction")

    def pdf_direction(self, ref: mi.Interaction3f, ds: mi.DirectionSample3f,
                      active: mi.Bool) -> mi.Float:
        return mi.Float(0.0)

    def eval_direction(self, ref: mi.Interaction3f, ds: mi.DirectionSample3f,
                       active: mi.Bool) -> mi.Spectrum:
        return mi.Spectrum(0.0)

    def to_string(self):
        string = f"{type(self).__name__}[\n"
        string += f"  is_confocal = {self.is_confocal},\n"
        string += f"  fov = {self.fov},\n"
        string += f"  num_spots = {self.num_spots},\n"
        string += f"  max_rejection_samples = {self.max_rejection_samples},\n"
        string += f"]"
        return string


mi.register_emitter('confocal_projector', lambda props: ConfocalProjector(props))
