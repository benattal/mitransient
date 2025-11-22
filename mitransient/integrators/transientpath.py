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
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.use_confocal_light_source = props.get("use_confocal_light_source", True)
        self.use_nlos_only = props.get("use_nlos_only", True)

        # Storage for projector emitter and precomputed data
        self.projector_emitter = None
        self.projector_pmf = None
        self.projector_cdf = None
        self.projector_width = None
        self.projector_height = None
        self.projector_fov = None
        self.projector_texture_data = None  # Store the full texture for querying

    def prepare_projector(self, scene: mi.Scene):
        """
        Extract projector emitter information and precompute sampling data.
        """
        # Find the projector emitter (assumed to be the first emitter)
        emitters = scene.emitters()
        if len(emitters) == 0:
            return

        self.projector_emitter = emitters[0]

        # Check if it's actually a projector
        if self.projector_emitter.class_name() != "Projector":
            return

        # Access texture data through scene parameters
        params = mi.traverse(scene)

        # Extract texture data
        try:
            texture_data = params['light.irradiance.data']
            self.projector_height, self.projector_width = texture_data.shape[0], texture_data.shape[1]

            # Store the texture data for later querying
            self.projector_texture_data = texture_data

            # Convert to numpy for PMF/CDF computation
            texture_np = texture_data.numpy().reshape(self.projector_height, self.projector_width, 3)

            # Compute PMF from first channel (assuming grayscale or uniform RGB)
            intensities = texture_np[:, :, 0]
            total_intensity = np.sum(intensities)

            if total_intensity > 0:
                self.projector_pmf = intensities / total_intensity
                # Flatten and compute CDF for sampling
                pmf_flat = self.projector_pmf.flatten()
                self.projector_cdf = np.cumsum(pmf_flat)

            # Extract FOV
            self.projector_fov = params.get('light.x_fov', 0.2)  # Default to 0.2 if not found

        except Exception as e:
            print(f"Warning: Could not extract projector data: {e}")
            self.projector_emitter = None

    def query_projector_texture(self,
                                 pixel_x: mi.Float,
                                 pixel_y: mi.Float,
                                 subpixel_offset_x: mi.Float,
                                 subpixel_offset_y: mi.Float) -> mi.Spectrum:
        """
        Query the projector texture at sub-pixel coordinates using bilinear interpolation.

        Args:
            pixel_x: Integer pixel x coordinate
            pixel_y: Integer pixel y coordinate
            subpixel_offset_x: Fractional offset within pixel [0, 1] in x direction
            subpixel_offset_y: Fractional offset within pixel [0, 1] in y direction

        Returns:
            RGB spectrum value at the interpolated position
        """
        # Convert sub-pixel coordinates to texture indices
        tex_x = pixel_x + subpixel_offset_x
        tex_y = pixel_y + subpixel_offset_y

        # Clamp to valid texture coordinates
        tex_x = dr.clip(tex_x, 0, self.projector_width - 1)
        tex_y = dr.clip(tex_y, 0, self.projector_height - 1)

        # Get floor and fractional parts for bilinear interpolation
        tex_x_floor = mi.Int32(dr.floor(tex_x))
        tex_y_floor = mi.Int32(dr.floor(tex_y))
        tex_x_frac = tex_x - mi.Float(tex_x_floor)
        tex_y_frac = tex_y - mi.Float(tex_y_floor)

        # Clamp neighboring pixel coordinates
        tex_x_ceil = mi.Int32(dr.minimum(tex_x_floor + 1, self.projector_width - 1))
        tex_y_ceil = mi.Int32(dr.minimum(tex_y_floor + 1, self.projector_height - 1))

        # Bilinear interpolation weights
        w_00 = (1 - tex_x_frac) * (1 - tex_y_frac)
        w_10 = tex_x_frac * (1 - tex_y_frac)
        w_01 = (1 - tex_x_frac) * tex_y_frac
        w_11 = tex_x_frac * tex_y_frac

        # Use simple indexing to access texture values
        # texture_data shape: (height, width, 3)
        rgb_00 = self.projector_texture_data[tex_y_floor, tex_x_floor, :]
        rgb_10 = self.projector_texture_data[tex_y_floor, tex_x_ceil, :]
        rgb_01 = self.projector_texture_data[tex_y_ceil, tex_x_floor, :]
        rgb_11 = self.projector_texture_data[tex_y_ceil, tex_x_ceil, :]

        # Interpolate RGB channels
        rgb = w_00 * rgb_00 + w_10 * rgb_10 + w_01 * rgb_01 + w_11 * rgb_11

        return mi.Spectrum(rgb)

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
        Custom emitter sampling that uses a projector pattern aligned with the camera ray.

        If a projector emitter is configured, this samples a pixel from the projector
        based on its intensity PMF, transforms it to align with the camera ray, and
        computes the radiance contribution from that point.

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
        if self.projector_emitter is None or self.projector_pmf is None or sampler is None:
            # Fall back to original confocal behavior
            return self.sample_emitter_confocal(scene, si, initial_intersection_point, active, si_initial)

        # Initialize output
        ds = dr.zeros(mi.DirectionSample3f)
        em_weight = mi.Spectrum(0.0)

        # Sample a projector pixel using the intensity PMF
        # We need to use DrJit-compatible sampling here
        # Use discrete distribution sampling with precomputed CDF

        # Get a uniform random sample - this is a DrJit array
        u = sampler.next_1d()

        # Convert CDF to DrJit array
        cdf_dr = mi.Float(self.projector_cdf)

        # Binary search in CDF - use DrJit searchsorted equivalent
        # For now, use a simple approach: evaluate the CDF at index positions
        num_pixels = self.projector_width * self.projector_height
        pixel_idx = dr.binary_search(
            0, num_pixels - 1,
            lambda idx: dr.gather(mi.Float, cdf_dr, idx) < u
        )

        # Clamp to valid range
        pixel_idx = dr.clip(pixel_idx, 0, num_pixels - 1)

        # Convert to x, y coordinates
        pixel_y = pixel_idx // self.projector_width
        pixel_x = pixel_idx % self.projector_width

        # Get the PMF value for this pixel - use dr.gather to index into PMF array
        pmf_flat_dr = mi.Float(self.projector_pmf.flatten())
        pixel_pmf = dr.gather(mi.Float, pmf_flat_dr, pixel_idx)

        # Sub-pixel sampling: sample a random position within the pixel
        # Get two random samples for x and y offset within the pixel
        subpixel_offset_x = sampler.next_1d()  # Random value in [0, 1]
        subpixel_offset_y = sampler.next_1d()  # Random value in [0, 1]

        # Convert pixel coordinates to normalized coordinates [-1, 1]
        # Use subpixel_offset instead of 0.5 for random position within pixel
        normalized_x = 2.0 * (pixel_x + subpixel_offset_x) / self.projector_width - 1.0
        normalized_y = 2.0 * (pixel_y + subpixel_offset_y) / self.projector_height - 1.0

        # Compute direction in projector local space
        tan_half_fov = dr.tan(self.projector_fov / 2.0)
        dir_x = tan_half_fov * normalized_x
        dir_y = tan_half_fov * normalized_y
        dir_z = -1.0

        direction_local = mi.Vector3f(dir_x, dir_y, dir_z)
        direction_local = dr.normalize(direction_local)

        # Build projector_to_world transform that aligns:
        # - Center pixel with the primary camera ray direction
        # - Origin with camera origin
        # - Up direction with camera up ([0, 1, 0])

        # Camera forward direction (initial ray direction)
        camera_forward = dr.normalize(camera_ray_direction)

        # Camera up direction
        camera_up = mi.Vector3f(0.0, 1.0, 0.0)

        # Build right direction (perpendicular to forward and up)
        camera_right = dr.normalize(dr.cross(camera_forward, camera_up))

        # Recompute up to be orthogonal
        camera_up = dr.normalize(dr.cross(camera_right, camera_forward))

        # Build rotation matrix: columns are [right, up, -forward]
        # Note: We use -forward because projector local space has z=-1 pointing into scene
        projector_to_world_rotation = mi.Matrix3f(
            camera_right,
            camera_up,
            -camera_forward
        )

        # Transform the sampled direction to world space
        # direction_world = projector_to_world_rotation @ direction_local

        # NOTE: Remove
        direction_world = initial_intersection_point - camera_origin
        direction_world = dr.normalize(direction_world)

        # Origin is at camera position
        origin_world = camera_origin

        # Cast ray from projector origin in sampled direction to find intersection
        projector_ray = mi.Ray3f(origin_world, direction_world)
        si_projector = scene.ray_intersect(projector_ray, mi.Bool(True))

        # # TODO: Remove
        # si_projector = si_initial

        # Check if we hit something
        is_valid = mi.Bool(True)

        # Intersection point
        # intersection_point = si_projector.p

        # NOTE: Remove
        intersection_point = initial_intersection_point 

        # Direction from current vertex to intersection point
        to_light = intersection_point - si.p
        dist = dr.norm(to_light)
        to_light_normalized = dr.normalize(to_light)

        # Check visibility from current point to intersection
        epsilon_bump = 1e-4
        shadow_ray = mi.Ray3f(si.p + to_light_normalized * epsilon_bump, to_light_normalized)
        si_shadow = scene.ray_intersect(shadow_ray, is_valid)

        epsilon = 1e-4
        is_visible = (dr.norm(si_shadow.p - intersection_point) < epsilon)

        # Get the BSDF at the projector intersection point
        bsdf_projector = si_projector.bsdf()

        # Outgoing direction at intersection (toward current vertex)
        wo_projector = si_projector.to_local(-to_light_normalized)

        # Evaluate BRDF at the intersection point
        bsdf_ctx = mi.BSDFContext()
        bsdf_value_projector = bsdf_projector.eval(bsdf_ctx, si_projector, wo_projector, is_valid)
        bsdf_value_projector = si_projector.to_world_mueller(bsdf_value_projector, -wo_projector, si_projector.wi)

        # Get projector radiance (from texture intensity)
        # Query the texture at the sub-pixel coordinates using bilinear interpolation
        # projector_radiance = self.query_projector_texture(
        #     pixel_x, pixel_y, subpixel_offset_x, subpixel_offset_y
        # )
        # projector_radiance = (pixel_pmf * self.projector_width * self.projector_height)

        # NOTE: Remove
        projector_radiance = 1.0

        # Compute sampling PDF:
        # pdf_projector_pixel = pixel_pmf (probability of sampling this pixel)
        # We need to convert from projector solid angle to area to solid angle at current vertex

        # Differential solid angle in projector space per pixel
        pixel_area_normalized = (2.0 / self.projector_width) * (2.0 / self.projector_height)
        tan_half_fov_sq = tan_half_fov * tan_half_fov
        # Approximate solid angle per pixel (small angle approximation)
        d_omega_projector = pixel_area_normalized * tan_half_fov_sq

        # Convert to differential area at intersection
        # dA = dω * distance² / cos(θ)
        cos_theta_projector = dr.abs(dr.dot(si_projector.n, -direction_world))
        dist_projector = si_projector.t
        d_area = d_omega_projector * (dist_projector * dist_projector) / dr.maximum(cos_theta_projector, 1e-6)

        # Convert differential area to solid angle at current vertex
        # dω = dA * cos(θ') / distance²
        cos_theta_vertex = dr.abs(dr.dot(si_projector.n, to_light_normalized))
        dist_vertex_to_intersection = dist
        d_omega_vertex = d_area * cos_theta_vertex / dr.maximum(dist_vertex_to_intersection * dist_vertex_to_intersection, 1e-3)

        # PDF in solid angle from current vertex
        pdf_weight = d_omega_vertex / dr.maximum(pixel_pmf, 1e-6)

        # NOTE: Remove
        pdf_weight = dr.rcp(dr.maximum(dist * dist, 1e-6))
        pdf_weight = dr.minimum(pdf_weight, 100.0)

        # Total weight: projector_radiance * BRDF * cos(theta) * visibility / pdf
        # Note: BRDF already includes cos(theta)
        em_weight = dr.select(
            is_visible,
            projector_radiance * bsdf_value_projector * pdf_weight,
            mi.Spectrum(0.0)
        )

        # Fill in direction sample
        ds.p = intersection_point
        ds.d = to_light_normalized
        ds.dist = dist
        # ds.pdf = 1.0 / dr.maximum(pdf_weight, 1e-6)
        # ds.delta = False

        # NOTE: Remove
        ds.pdf = 1.0
        ds.delta = True

        return ds, em_weight

    def sample_emitter_confocal(self,
                                scene: mi.Scene,
                                si: mi.SurfaceInteraction3f,
                                initial_intersection_point: mi.Point3f,
                                active: mi.Bool,
                                si_initial: mi.SurfaceInteraction3f = None) -> Tuple[mi.DirectionSample3f, mi.Spectrum]:
        """
        Original confocal emitter sampling (fallback when no projector is configured).
        """
        # Create a direction sample pointing to the initial intersection
        ds = dr.zeros(mi.DirectionSample3f)
        ds.p = initial_intersection_point
        ds.d = dr.normalize(initial_intersection_point - si.p)
        ds.dist = dr.norm(initial_intersection_point - si.p)
        ds.pdf = 1.0  # Deterministic sampling
        ds.delta = True  # Not a delta distribution

        # Visibility test: check if ray from current point to initial intersection is occluded
        # Bump the shadow ray origin by epsilon along the shadow direction to avoid self-intersection
        epsilon_bump = 1e-4
        shadow_direction = dr.normalize(initial_intersection_point - si.p)
        shadow_origin = si.p + shadow_direction * epsilon_bump

        # Create shadow ray with bumped origin
        shadow_ray = mi.Ray3f(shadow_origin, shadow_direction)

        # Perform visibility test - check if we hit something before reaching the initial point
        # We use a small epsilon to account for numerical precision
        epsilon = 1e-4
        si_shadow = scene.ray_intersect(shadow_ray, active)

        # Check if the shadow ray reaches the initial intersection without hitting other geometry
        # The ray is visible if: (1) it doesn't hit anything, OR (2) the hit is very close to target
        is_visible = (dr.norm(si_shadow.p - initial_intersection_point) < epsilon)

        # Get the BSDF at the initial intersection point
        bsdf_initial = si_initial.bsdf()

        # Direction from initial point to current point (outgoing from initial point's perspective)
        wo_initial = si_initial.to_local(-ds.d)

        # Standard BSDF evaluation context
        bsdf_ctx = mi.BSDFContext()

        # Evaluate BRDF * cos(theta) at the initial intersection point
        bsdf_value = bsdf_initial.eval(bsdf_ctx, si_initial, wo_initial, active)

        # Convert back to world coordinates if using polarization
        bsdf_value = si_initial.to_world_mueller(bsdf_value, -wo_initial, si_initial.wi)

        # Compute inverse square falloff: 1 / dist^2
        # Clip to prevent extremely large values at small distances
        # Using a minimum distance of 0.01 meters (1 cm) to avoid division by zero
        dist_squared = dr.maximum(ds.dist * ds.dist, 0.0001)  # min dist = 0.01m
        inverse_square_falloff = dr.rcp(dist_squared)

        # Clamp the falloff to a reasonable maximum (e.g., 10000)
        inverse_square_falloff = dr.minimum(inverse_square_falloff, 100.0)

        # Weight includes BRDF at initial point, inverse square falloff, and visibility
        # Zero contribution if occluded or if direction is perpendicular to normal
        em_weight = dr.select(is_visible,
                             bsdf_value * mi.Spectrum(inverse_square_falloff),
                             mi.Spectrum(0.0))

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

        # Prepare projector if not already done (only once per render)
        if self.use_confocal_light_source and self.projector_emitter is None:
            self.prepare_projector(scene)

        # Track if we've hit at least one non-directly-visible point (for use_nlos_only)
        has_hit_nlos_point = mi.Bool(False)
        should_zero_contribution = mi.Bool(False)
        is_directly_visible = mi.Bool(True)

        if self.camera_unwarp:
            si = scene.ray_intersect(mi.Ray3f(ray),
                                     ray_flags=mi.RayFlags.All,
                                     coherent=mi.Mask(True))

            distance[si.is_valid()] = -si.t

        if self.use_confocal_light_source:
            si_initial = scene.ray_intersect(mi.Ray3f(ray),
                                            ray_flags=mi.RayFlags.All,
                                            coherent=mi.Mask(True))
            initial_intersection_point = si_initial.p

            distance[si_initial.is_valid()] += si_initial.t

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

            mis = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            with dr.resume_grad(when=not primal):
                Le = β * mi.Spectrum(mis) * \
                    dr.select(self.discard_direct_light, 0,
                              ds.emitter.eval(si, active_next))

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
                Le = dr.select(should_zero_contribution, mi.Spectrum(0.0), Le)
                L = dr.select(should_zero_contribution, mi.Spectrum(0.0), L)

            # Add transient contribution because of emitter found
            if primal:
                add_transient(Le, distance, ray.wavelengths, active)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(
                bsdf.flags(), mi.BSDFFlags.Smooth)

            # Skip emitter sampling for first bounce when use_nlos_only is True
            # Sample emitter: use custom NLOS light source or standard emitter sampling
            if self.use_confocal_light_source:
                # Use custom emitter sampling toward initial intersection point (or projector)
                ds, em_weight = self.sample_emitter(
                    scene, si, initial_intersection_point, active_em, si_initial,
                    camera_origin, camera_ray_direction, sampler)
            else:
                # Standard emitter sampling
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
                mis_em = dr.select(
                    ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = β * mi.Spectrum(mis_em) * bsdf_value_em * em_weight

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
