from __future__ import annotations  # Delayed parsing of type annotations
from typing import Optional, Tuple, List, Callable, Any

import drjit as dr
import mitsuba as mi
import numpy as np
from mitsuba.ad.integrators.common import mis_weight  # type: ignore


from .common import TransientADIntegrator


class TransientPath(TransientADIntegrator):
    r"""
    .. _integrator-transient_path:

    Transient Path (:monosp:`transient_path`)
    -----------------------------------------

    Standard path tracing algorithm which now includes the time dimension.

    .. note::
        If you want to simulate a Non-Line-of-Sight (NLOS) setup, look into the
        ``transient_nlos_path`` plugin, which contains different sampling routines
        specific to NLOS setups that greatly increase the quality of your results.

    .. note::
        This integrator does not handle participating media. Instead, you should use
        our ``transient_prbvolpath`` plugin.

    .. tabs::

        .. code-tab:: xml

            <integrator type="transient_path">
                <integer name="max_depth" value="8"/>
            </integrator>

        .. code-tab:: python

            {
                'type': 'transient_path',
                'max_depth': 8
            }

    .. pluginparameters::

     * - camera_unwarp
       - |bool|
       - If True, does not take into account the distance from the camera origin
         to the camera ray's first intersection point. This allows you to see
         the transient video with the events happening in world time. If False,
         this distance is taken into account, so you see the same thing that you
         would see with a real-world ultra-fast camera. (default: false)

     * - temporal_filter
       - |string|
       - Can be either:
         - 'box' for a box filter (no parameters)
         - 'gaussian' for a Gaussian filter (see gaussian_stddev below)
         - Empty string to use the same filter in the temporal domain as
         the rfilter used in the spatial domain.
         (default: empty string)

     * - gaussian_stddev
       - |float|
       - When temporal_filter == 'gaussian', this marks the standard deviation
         of the Gaussian filter. (default: 2.0)

     * - block_size
       - |int|
       - Size of (square) image blocks to render in parallel (in scalar mode).
         Should be a power of two. (default: 0 i.e. let Mitsuba decide for you)

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (default: 6)

     * - rr_depth
       - |int|
       - Specifies the path depth, at which the implementation will begin to use
         the *russian roulette* path termination criterion. For example, if set to
         1, then path generation many randomly cease after encountering directly
         visible surfaces. (default: 5)

     * - use_confocal_light_source
       - |bool|
       - If True, uses custom emitter sampling that samples the direction from
         the current point to the intersection of the initial camera ray with
         the scene geometry. This is useful for NLOS-type setups. (default: false)

     * - use_nlos_only
       - |bool|
       - If True, only allows contributions where the ray from the current
         intersection point to the camera hits a piece of geometry before
         reaching the camera. This ensures that we only directly illuminate
         the NLOS scene and exclude direct line-of-sight paths. (default: false)

     * - projector_fov
       - |float|
       - Field of view (in radians) for the projector. Controls how the
         projector pattern is mapped onto the scene. (default: 0.2)

     * - projector_frame
       - |transform|
       - 3x3 rotation matrix defining the projector's coordinate frame.
         Columns represent [right, up, forward] vectors in world space.
         If not provided and use_confocal_light_source=True, the frame is
         computed from the camera ray direction. (default: None)

     * - projector_spot_positions
       - |array|
       - Nx2 array of spot positions on the projector image plane in normalized
         coordinates [-1, 1]. Each row is [x, y] position. (default: None)

     * - projector_spot_sigmas
       - |array|
       - N-element array of standard deviations for each Gaussian spot in
         normalized coordinates (same scale as positions). (default: None)

     * - projector_spot_intensities
       - |array|
       - Nx3 array of RGB intensities for each spot. (default: None)

     * - max_rejection_samples
       - |int|
       - Maximum number of rejection sampling iterations when sampling from
         Gaussian spots to avoid samples outside the projector FOV. (default: 8)
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.use_confocal_light_source = props.get("use_confocal_light_source", True)
        self.use_nlos_only = props.get("use_nlos_only", True)

        self.projector_fov = props.get("projector_fov", 0.2)  # Default FOV in radians

        # Projector frame: if provided, use it; otherwise will be computed dynamically
        self.projector_frame = props.get("projector_frame", None)

        # Parametric projector configuration
        self.projector_spot_positions = props.get("projector_spot_positions", None)
        self.projector_spot_sigmas = props.get("projector_spot_sigmas", None)
        self.projector_spot_intensities = props.get("projector_spot_intensities", None)
        self.max_rejection_samples = props.get("max_rejection_samples", 8)

        # Check if we're using parametric mode
        self.use_parametric_projector = (
            self.projector_spot_positions is not None and
            self.projector_spot_sigmas is not None and
            self.projector_spot_intensities is not None
        )

        # Prepare parametric projector if configured
        if self.use_parametric_projector:
            self.prepare_parametric_projector()

    def prepare_parametric_projector(self):
        """
        Prepare parametric projector from Gaussian spot parameters.

        Sets up DrJit arrays for efficient vectorized evaluation and sampling.
        """
        positions = np.asarray(self.projector_spot_positions)
        sigmas = np.asarray(self.projector_spot_sigmas)
        intensities = np.asarray(self.projector_spot_intensities)

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

    def eval_parametric_projector(self,
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

    def sample_parametric_spot(self,
                                sampler: mi.Sampler) -> Tuple[mi.UInt32, mi.Float, mi.Float, mi.Float]:
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

    def build_projector_frame(self, camera_ray_direction: mi.Vector3f = None) -> mi.Matrix3f:
        """
        Build orthonormal frame for projector.

        If use_confocal_light_source is False and projector_frame was provided in constructor,
        returns that stored frame. Otherwise, builds a frame aligned with the camera ray direction.

        Args:
            camera_ray_direction: Initial camera ray direction (required when computing frame dynamically)

        Returns:
            3x3 rotation matrix (columns: right, up, forward)
        """
        # Only use stored projector frame when NOT in confocal mode
        if not self.use_confocal_light_source and self.projector_frame is not None:
            return self.projector_frame

        # Otherwise, compute from camera ray direction (for confocal mode or when frame not provided)
        if camera_ray_direction is None:
            raise ValueError("camera_ray_direction required when projector_frame is not set or in confocal mode")

        camera_forward = -camera_ray_direction
        camera_up = mi.Vector3f(0.0, 1.0, 0.0)
        camera_right = dr.normalize(dr.cross(camera_forward, camera_up))
        camera_up = dr.normalize(dr.cross(camera_right, camera_forward))
        return mi.Matrix3f(camera_right, camera_up, camera_forward)

    def query_projector_direct(self,
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
        world_to_projector = self.build_projector_frame(camera_ray_direction)
        direction_local = world_to_projector @ (-to_projector_normalized)

        # Convert to normalized coordinates [-1, 1]
        tan_half_fov = dr.tan(self.projector_fov / 2.0)
        normalized_x = direction_local.x / (-direction_local.z * tan_half_fov)
        normalized_y = direction_local.y / (-direction_local.z * tan_half_fov)

        # Query parametric projector (Gaussian mixture)
        projector_radiance = self.eval_parametric_projector(normalized_x, normalized_y)

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

    def sample_emitter_parametric(self,
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
        spot_idx, normalized_x, normalized_y, sample_pdf = self.sample_parametric_spot(sampler)

        # Compute direction in projector local space
        tan_half_fov = dr.tan(self.projector_fov / 2.0)
        direction_local = dr.normalize(mi.Vector3f(
            tan_half_fov * normalized_x,
            tan_half_fov * normalized_y,
            -1.0
        ))

        # Transform to world space and find intersection
        projector_to_world = self.build_projector_frame(camera_ray_direction).T
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
        projector_radiance = self.eval_parametric_projector(normalized_x, normalized_y)

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

    @dr.syntax
    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               β: mi.Spectrum,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               # add_transient accepts (spec, distance, wavelengths, active)
               add_transient: Callable[[mi.Spectrum, mi.Float, mi.UnpolarizedSpectrum, mi.Mask], None],
               gather_derivatives_at_distance: Callable[[
                   Any, Any], Any] = None,
               **kwargs  # Absorbs unused arguments
               ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], mi.Spectrum]:
        """
        See ``TransientADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        # Differential/adjoint radiance
        δL = mi.Spectrum(δL if δL is not None else 0)

        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes
        distance = mi.Float(0.0)                      # Distance of the path

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        # Store initial camera ray intersection for NLOS light source mode
        initial_intersection_point = mi.Point3f(0.0)
        si_initial = dr.zeros(mi.SurfaceInteraction3f)
        camera_origin = mi.Point3f(ray.o)  # Store camera origin for NLOS-only check
        camera_ray_direction = mi.Vector3f(ray.d)  # Store initial camera ray direction

        # Track if we've hit at least one non-directly-visible point (for use_nlos_only)
        has_hit_nlos_point = mi.Bool(False)
        should_zero_contribution = mi.Bool(False)
        is_directly_visible = mi.Bool(True)

        if self.camera_unwarp:
            si = scene.ray_intersect(mi.Ray3f(ray),
                                     ray_flags=mi.RayFlags.All,
                                     coherent=mi.Mask(True))

            distance[si.is_valid()] = -si.t

        si_initial = scene.ray_intersect(mi.Ray3f(ray),
                                        ray_flags=mi.RayFlags.All,
                                        coherent=mi.Mask(True))
        initial_intersection_point = si_initial.p

        while dr.hint(active,
                      max_iterations=self.max_depth,
                      label="Transient Path (%s)" % mode.name):
            active_next = mi.Bool(active)

            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All,
                                         coherent=(depth == 0))

            # Update distance
            distance += dr.select(active, si.t, 0.0) * η

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Hide the environment emitter if necessary
            if dr.hint(self.hide_emitters, mode='scalar'):
                active_next &= ~((depth == 0) & ~si.is_valid())

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            Le = mi.Float(0.0)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(
                bsdf.flags(), mi.BSDFFlags.Smooth)

            # Sample emitter: use parametric projector or standard emitter sampling

            # Check if this is the initial intersection point (depth == 0)
            is_initial_intersection = (depth == 0)

            if self.use_parametric_projector:
                # For initial intersection: query projector directly with inverse square falloff
                # For subsequent bounces: importance sample from the Gaussian spots
                if is_initial_intersection:
                    ds, em_weight = self.query_projector_direct(
                        scene, si, camera_origin, camera_ray_direction, active_em
                    )
                else:
                    ds, em_weight = self.sample_emitter_parametric(
                        scene, si, camera_origin, camera_ray_direction, sampler, active_em
                    )
            else:
                # Standard emitter sampling (no projector)
                ds, em_weight = scene.sample_emitter_direction(
                    si, sampler.next_2d(), True, active_em)

            active_em &= (ds.pdf != 0.0)

            with dr.resume_grad(when=not primal):
                if dr.hint(not primal, mode='scalar'):
                    # Given the detached emitter sample, *recompute* its
                    # contribution with AD to enable light source optimization
                    ds.d = dr.replace_grad(ds.d, dr.normalize(ds.p - si.p))
                    em_val = scene.eval_emitter_direction(si, ds, active_em)
                    em_weight = dr.replace_grad(em_weight, dr.select(
                        (ds.pdf != 0), em_val / ds.pdf, 0))
                    dr.disable_grad(ds.d)

                # Evaluate BSDF * cos(theta) differentiably
                wo = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(
                    bsdf_ctx, si, wo, active_em)
                bsdf_value_em = si.to_world_mueller(bsdf_value_em, -wo, si.wi)
                Lr_dir = β * bsdf_value_em * em_weight

            if self.use_nlos_only:
                # Check if current point is directly visible from camera
                # Cast ray from camera toward current intersection point
                point_direction = dr.normalize(si.p - camera_origin)
                visibility_ray_origin = camera_origin
                visibility_ray = mi.Ray3f(visibility_ray_origin, point_direction)

                # Check if ray from camera hits the current point without hitting other geometry
                si_visibility = scene.ray_intersect(visibility_ray, mi.Bool(True))

                # Point is directly visible if the ray hits the current point (within epsilon)
                # Check that the hit point is very close to si.p
                epsilon_distance = 1e-4
                is_directly_visible = si_visibility.is_valid() & (dr.norm(si_visibility.p - si.p) < epsilon_distance)

                # Update tracking: if this point is NOT directly visible, mark that we've hit an NLOS point
                has_hit_nlos_point |= ~is_directly_visible

                # Zero out contributions if point is directly visible AND we haven't hit an NLOS point yet
                should_zero_contribution = ~has_hit_nlos_point
                Lr_dir = dr.select(should_zero_contribution, mi.Spectrum(0.0), Lr_dir)
                L = dr.select(should_zero_contribution, mi.Spectrum(0.0), L)

            # Add contribution direct emitter sampling
            if primal:
                add_transient(Lr_dir, distance + ds.dist *
                              η, ray.wavelengths, active)

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)
            bsdf_weight = si.to_world_mueller(
                bsdf_weight, -bsdf_sample.wo, si.wi)

            # Zero out contributions from directly visible points unless we've already hit an NLOS point
            # when use_nlos_only is True
            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)

            # ---- Update loop variables based on current interaction -----
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β = β * bsdf_weight

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(mi.unpolarized_spectrum(β))
            active_next &= (β_max != 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)
            active_next &= rr_prob > 0

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob) & (rr_prob > 0)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            # ------------------ Differential phase only ------------------

            if dr.hint(not primal, mode='scalar'):
                with dr.resume_grad():
                    # 'L' stores the indirectly reflected radiance at the
                    # current vertex but does not track parameter derivatives.
                    # The following addresses this by canceling the detached
                    # BSDF value and replacing it with an equivalent term that
                    # has derivative tracking enabled. (nit picking: the
                    # direct/indirect terminology isn't 100% accurate here,
                    # since there may be a direct component that is weighted
                    # via multiple importance sampling)

                    # Recompute 'wo' to propagate derivatives to cosine term
                    wo = si.to_local(ray.d)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)
                    bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

                    # Detached version of the above term and inverse
                    bsdf_val_det = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_det = dr.select(bsdf_val_det != 0,
                                                 dr.rcp(bsdf_val_det), 0)

                    # Differentiable version of the reflected indirect
                    # radiance. Minor optional tweak: indicate that the primal
                    # value of the second term is always 1.
                    tmp = inv_bsdf_val_det * bsdf_val
                    tmp_replaced = dr.replace_grad(
                        dr.ones(mi.Float, dr.width(tmp)), tmp)  # FIXME
                    Lr_ind = L * tmp_replaced

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr_dir + Lr_ind

                    attached_contrib = dr.flag(
                        dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo)
                    if dr.hint(attached_contrib, mode='scalar'):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    if dr.hint(mode == dr.ADMode.Backward, mode='scalar'):
                        δL_read = gather_derivatives_at_distance(δL, distance)
                        dr.backward_from(δL_read * Lo)
                    else:
                        δL_part = dr.forward_to(Lo)
                        add_transient(δL_part, distance,
                                      ray.wavelengths, True)
                        δL += δL_part

            depth[si.is_valid()] += 1
            active = active_next

        return (
            L if primal else δL,  # Radiance/differential radiance
            (depth != 0),         # Ray validity flag for alpha blending
            [],                   # Empty typle of AOVs
            L                     # State for the differential phase
        )


mi.register_integrator("transient_path", lambda props: TransientPath(props))

del TransientADIntegrator
