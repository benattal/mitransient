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

     * - projector_texture
       - |bitmap|
       - Bitmap object to use as a projector pattern. When specified,
         the integrator will use this bitmap to modulate the illumination at
         the first intersection point. The texture is queried directly with
         inverse square falloff at depth 0, and importance sampled at subsequent
         depths. (default: None, no projector)

     * - projector_fov
       - |float|
       - Field of view (in radians) for the projector texture. Controls how the
         texture is mapped onto the scene. (default: 0.2)

     * - projector_frame
       - |transform|
       - 3x3 rotation matrix defining the projector's coordinate frame.
         Columns represent [right, up, forward] vectors in world space.
         If not provided and use_confocal_light_source=True, the frame is
         computed from the camera ray direction. If not provided and
         use_confocal_light_source=False but projector_texture is set,
         defaults to identity (aligned with world axes). (default: None)
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.use_confocal_light_source = props.get("use_confocal_light_source", True)
        self.use_nlos_only = props.get("use_nlos_only", True)

        # Projector texture configuration (stored directly on integrator)
        self.projector_texture = props.get("projector_texture", None)
        self.projector_fov = props.get("projector_fov", 0.2)  # Default FOV in radians

        # Projector frame: if provided, use it; otherwise will be computed dynamically
        self.projector_frame = props.get("projector_frame", None)

        # Storage for projector texture and precomputed data
        # Prepare projector if we have a texture, regardless of confocal mode
        if self.projector_texture is not None:
            self.prepare_projector()
        else:
            self.projector_pmf = None
            self.projector_cdf = None
            self.projector_width = None
            self.projector_height = None
            self.projector_texture_data = None

    def prepare_projector(self):
        """
        Prepare projector texture from the configured bitmap object.
        """
        # Access texture data through parameters
        texture_data = self.projector_texture.tensor()

        self.projector_height, self.projector_width = texture_data.shape[0], texture_data.shape[1]
        self.projector_texture_data = texture_data

        # Convert to numpy for PMF/CDF computation
        texture_np = texture_data.numpy().reshape(self.projector_height, self.projector_width, 3)

        # Compute PMF from first channel (assuming grayscale or uniform RGB)
        intensities = texture_np[:, :, 0]
        total_intensity = np.sum(intensities)

        self.projector_pmf = intensities / np.maximum(total_intensity, 1e-6)

        # Flatten and compute CDF for sampling
        pmf_flat = self.projector_pmf.flatten()
        self.projector_cdf = np.cumsum(pmf_flat)

    def query_projector_texture(self,
                                 pixel_x: mi.Float,
                                 pixel_y: mi.Float) -> mi.Spectrum:
        """
        Query the projector texture at continuous pixel coordinates using bilinear interpolation.

        Args:
            pixel_x: Continuous pixel x coordinate (can be fractional)
            pixel_y: Continuous pixel y coordinate (can be fractional)

        Returns:
            RGB spectrum value at the interpolated position
        """
        # Convert pixel coordinates to normalized texture coordinates [0, 1]
        uv = mi.Point2f(pixel_x / self.projector_width, pixel_y / self.projector_height)
        return mi.Spectrum(self.projector_texture.eval(uv))

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

    def sample_emitter(self,
                       scene: mi.Scene,
                       si: mi.SurfaceInteraction3f,
                       initial_intersection_point: mi.Point3f,
                       active: mi.Bool,
                       si_initial: mi.SurfaceInteraction3f = None,
                       camera_origin: mi.Point3f = None,
                       camera_ray_direction: mi.Vector3f = None,
                       sampler: mi.Sampler = None) -> Tuple[mi.DirectionSample3f, mi.Spectrum]:
        """
        Custom emitter sampling that uses a projector pattern or confocal sampling.

        If a projector texture is configured, samples a pixel from the projector based on
        its intensity PMF and computes the radiance contribution. Otherwise, falls back
        to confocal sampling if use_confocal_light_source is True.

        Args:
            scene: The scene
            si: Current surface interaction
            initial_intersection_point: The intersection point of the initial camera ray
            active: Active mask
            si_initial: Surface interaction at the initial intersection point
            camera_origin: Camera origin point
            camera_ray_direction: Initial camera ray direction
            sampler: Sampler for random numbers

        Returns:
            ds: Direction sample toward the sampled projector point
            em_weight: Weight of this sample including BRDF, radiance, and PDF
        """
        # Check if we have a projector configured
        has_projector = (self.projector_texture is not None and
                        self.projector_pmf is not None and
                        sampler is not None)

        if has_projector:
            # Use projector-based sampling (works with or without confocal mode)
            return self.sample_emitter_projector_pmf(
                scene, si, initial_intersection_point, active,
                si_initial, camera_origin, camera_ray_direction, sampler
            )
        else:
            # This shouldn't be reached, but provide a safe fallback
            raise ValueError("sample_emitter called without projector or confocal mode enabled")

    def sample_emitter_with_projector(self,
                                       scene: mi.Scene,
                                       si: mi.SurfaceInteraction3f,
                                       initial_intersection_point: mi.Point3f,
                                       active: mi.Bool,
                                       si_initial: mi.SurfaceInteraction3f,
                                       camera_origin: mi.Point3f,
                                       camera_ray_direction: mi.Vector3f,
                                       sampler: mi.Sampler,
                                       is_initial_intersection: mi.Bool) -> Tuple[mi.DirectionSample3f, mi.Spectrum]:
        """
        Helper function for emitter sampling with projector support.

        For initial intersection (depth == 0): queries projector directly with inverse square falloff
        For subsequent intersections: uses PMF-based sampling

        Args:
            scene: The scene
            si: Current surface interaction
            initial_intersection_point: The intersection point of the initial camera ray
            active: Active mask
            si_initial: Surface interaction at the initial intersection point
            camera_origin: Camera origin point
            camera_ray_direction: Initial camera ray direction
            sampler: Sampler for random numbers
            is_initial_intersection: Boolean mask indicating if this is the first intersection (depth == 0)

        Returns:
            ds: Direction sample
            em_weight: Emitter weight
        """
        # Query projector directly for initial intersection
        ds_direct, em_weight_direct = self.query_projector_direct(
            scene, si, camera_origin, camera_ray_direction, sampler, active, is_initial_intersection
        )

        return ds_direct, em_weight_direct

    def query_projector_direct(self,
                                scene: mi.Scene,
                                si: mi.SurfaceInteraction3f,
                                camera_origin: mi.Point3f,
                                camera_ray_direction: mi.Vector3f,
                                sampler: mi.Sampler,
                                active: mi.Bool,
                                mask: mi.Bool) -> Tuple[mi.DirectionSample3f, mi.Spectrum]:
        """
        Query incoming radiance directly from the projector for the initial intersection point.
        Uses inverse square falloff, no PMF sampling.

        Args:
            scene: The scene
            si: Current surface interaction (should be at initial intersection point)
            camera_origin: Camera origin point (projector position)
            camera_ray_direction: Initial camera ray direction
            sampler: Sampler for random numbers
            active: Active mask
            mask: Mask indicating which lanes should use direct querying

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

        # Convert to pixel coordinates
        tan_half_fov = dr.tan(self.projector_fov / 2.0)
        normalized_x = direction_local.x / (-direction_local.z * tan_half_fov)
        normalized_y = direction_local.y / (-direction_local.z * tan_half_fov)

        # Convert to continuous pixel coordinates and clamp
        pixel_x = dr.clip((normalized_x + 1.0) * 0.5 * self.projector_width, 0, self.projector_width - 1) + 0.5
        pixel_y = dr.clip((normalized_y + 1.0) * 0.5 * self.projector_height, 0, self.projector_height - 1) + 0.5

        # Query texture and apply inverse square falloff
        projector_radiance = self.query_projector_texture(pixel_x, pixel_y)
        falloff = dr.minimum(dr.rcp(dr.maximum(dist * dist, 1e-3)), 10000.0)

        # Create direction sample pointing toward projector
        ds = dr.zeros(mi.DirectionSample3f)
        ds.p = camera_origin
        ds.d = to_projector_normalized
        ds.dist = dist
        ds.pdf = 1.0
        ds.delta = False

        return ds, mi.Spectrum(projector_radiance * falloff)

    def sample_emitter_projector_pmf(self,
                                      scene: mi.Scene,
                                      si: mi.SurfaceInteraction3f,
                                      initial_intersection_point: mi.Point3f,
                                      active: mi.Bool,
                                      si_initial: mi.SurfaceInteraction3f,
                                      camera_origin: mi.Point3f,
                                      camera_ray_direction: mi.Vector3f,
                                      sampler: mi.Sampler) -> Tuple[mi.DirectionSample3f, mi.Spectrum]:
        """
        Sample emitter using PMF-based sampling (for non-initial intersection points).
        """
        # Sample a projector pixel using the intensity PMF
        num_pixels = self.projector_width * self.projector_height
        cdf_dr = mi.Float(self.projector_cdf)
        pixel_idx = dr.clip(dr.binary_search(0, num_pixels - 1,
                                             lambda idx: dr.gather(mi.Float, cdf_dr, idx) < sampler.next_1d()),
                           0, num_pixels - 1)

        # Convert to x, y coordinates with sub-pixel sampling
        pixel_y = (pixel_idx // self.projector_width) + 0.5
        pixel_x = (pixel_idx % self.projector_width) + 0.5

        subpixel_offset_x = sampler.next_1d() - 0.5
        subpixel_offset_y = sampler.next_1d() - 0.5

        # Get pixel PMF value
        pmf_flat_dr = mi.Float(self.projector_pmf.flatten())
        pixel_pmf = dr.gather(mi.Float, pmf_flat_dr, pixel_idx)

        # Convert to normalized coordinates [-1, 1]
        normalized_x = 2.0 * (pixel_x + subpixel_offset_x) / self.projector_width - 1.0
        normalized_y = 2.0 * (pixel_y + subpixel_offset_y) / self.projector_height - 1.0

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

        # Direction from current vertex to intersection point
        to_light = si_projector.p - si.p
        dist = dr.norm(to_light)
        to_light_normalized = dr.normalize(to_light)

        # Check visibility
        shadow_ray = mi.Ray3f(si.p + to_light_normalized * 1e-4, to_light_normalized)
        si_shadow = scene.ray_intersect(shadow_ray, active)
        is_visible = (dr.norm(si_shadow.p - si_projector.p) < 1e-4)

        # Evaluate BSDF at the projector intersection point
        bsdf_ctx = mi.BSDFContext()
        wo_projector = si_projector.to_local(-to_light_normalized)
        bsdf_value = si_projector.bsdf().eval(bsdf_ctx, si_projector, wo_projector, active)
        bsdf_value = si_projector.to_world_mueller(bsdf_value, -wo_projector, si_projector.wi)

        # Get projector radiance
        projector_radiance = self.query_projector_texture(
            mi.Float(pixel_x) + subpixel_offset_x,
            mi.Float(pixel_y) + subpixel_offset_y
        )
        projector_radiance = 1.0

        # Compute PDF weight (convert from projector solid angle to vertex solid angle)
        pixel_area_normalized = (2.0 / self.projector_width) * (2.0 / self.projector_height)
        d_omega_projector = pixel_area_normalized * tan_half_fov * tan_half_fov

        cos_theta_projector = dr.abs(dr.dot(si_projector.n, -direction_world))
        d_area = d_omega_projector / dr.maximum(cos_theta_projector, 1e-6)

        cos_theta_vertex = dr.abs(dr.dot(si_projector.n, -to_light_normalized))
        d_omega_vertex = d_area * cos_theta_vertex / dr.maximum(dist * dist, 1e-3)

        pdf_weight = dr.clip(d_omega_vertex / dr.maximum(pixel_pmf, 1e-6), 0.0, 10000.0)

        # Fill in direction sample
        ds = dr.zeros(mi.DirectionSample3f)
        ds.p = si_projector.p
        ds.d = to_light_normalized
        ds.dist = dist + dist_projector
        ds.pdf = 1.0 / pdf_weight
        ds.delta = False

        em_weight = dr.select(
            is_visible & active,
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

            # Skip emitter sampling for first bounce when use_nlos_only is True
            # Sample emitter: use projector, confocal, or standard emitter sampling

            # Check if this is the initial intersection point (depth == 0)
            is_initial_intersection = (depth == 0)

            # Determine if we should use projector/confocal sampling
            has_projector = (self.projector_texture is not None and self.projector_pmf is not None)

            if has_projector:
                # For initial intersection with projector: query directly
                if is_initial_intersection:
                    # Use helper function that handles both direct query and PMF sampling
                    ds, em_weight = self.sample_emitter_with_projector(
                        scene, si, initial_intersection_point, active_em, si_initial,
                        camera_origin, camera_ray_direction, sampler, is_initial_intersection
                    )
                else:
                    # Projector PMF sampling or confocal sampling for subsequent bounces
                    ds, em_weight = self.sample_emitter(
                        scene, si, initial_intersection_point, active_em, si_initial,
                        camera_origin, camera_ray_direction, sampler)
            else:
                # Standard emitter sampling (no projector, no confocal)
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
