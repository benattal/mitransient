from __future__ import annotations  # Delayed parsing of type annotations
from typing import Optional, Tuple, List, Callable, Any

import drjit as dr
import mitsuba as mi

from .common import TransientADIntegrator


class TransientPath(TransientADIntegrator):
    r"""
    .. _integrator-transient_path:

    Transient Path (:monosp:`transient_path`)
    -----------------------------------------

    Standard path tracing algorithm which now includes the time dimension.

    This integrator requires a confocal_projector emitter for light source sampling.
    If the projector has a pulse shape configured, pulse time offsets are importance
    sampled and contributions are added at ``path_distance + pulse_time_offset``.

    .. tabs::

        .. code-tab:: xml

            <integrator type="transient_path">
                <integer name="max_depth" value="8"/>
                <emitter type="confocal_projector" name="confocal_projector">
                    ...
                </emitter>
            </integrator>

        .. code-tab:: python

            {
                'type': 'transient_path',
                'max_depth': 8,
                'confocal_projector': { ... }
            }

    .. pluginparameters::

     * - camera_unwarp
       - |bool|
       - If True, does not take into account the distance from the camera origin
         to the camera ray's first intersection point. This allows you to see
         the transient video with the events happening in world time. If False,
         this distance is taken into account, so you see the same thing that you
         would see with a real-world ultra-fast camera. (default: false)

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

     * - confocal_projector
       - |emitter|
       - Reference to a ConfocalProjector emitter for custom light source sampling.
         Uses the projector's sampling methods for emitter sampling. If the projector
         has a pulse shape configured, pulse time offsets are importance sampled and
         contributions are added at ``path_distance + pulse_time_offset``. (required)

     * - use_nlos_only
       - |bool|
       - If True, only allows contributions where the ray from the current
         intersection point to the camera hits a piece of geometry before
         reaching the camera. This ensures that we only directly illuminate
         the NLOS scene and exclude direct line-of-sight paths. (default: false)

     * - pulse_samples
       - |int|
       - Number of samples to take from the pulse distribution per path vertex.
         Multiple samples reduce variance by spreading contributions across the
         pulse shape. Each sample is weighted by 1/(pulse_samples * pdf).
         (default: 1)
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.confocal_projector = props.get("confocal_projector", None)
        self.use_nlos_only = props.get("use_nlos_only", True)
        self.pulse_samples = props.get("pulse_samples", 1)

    def _apply_nlos_filter(self,
                           scene: mi.Scene,
                           si: mi.SurfaceInteraction3f,
                           camera_origin: mi.Point3f,
                           has_hit_nlos_point: mi.Bool,
                           Lr_dir: mi.Spectrum,
                           L: mi.Spectrum) -> Tuple[mi.Bool, mi.Spectrum, mi.Spectrum]:
        """
        Apply NLOS-only filtering to zero out contributions from directly visible points.

        Args:
            scene: The scene
            si: Current surface interaction
            camera_origin: Camera origin position
            has_hit_nlos_point: Tracking state for whether we've hit an NLOS point
            Lr_dir: Direct lighting contribution
            L: Accumulated radiance

        Returns:
            Tuple of (updated_has_hit_nlos_point, filtered_Lr_dir, filtered_L)
        """
        # Check if current point is directly visible from camera
        point_direction = dr.normalize(si.p - camera_origin)
        visibility_ray = mi.Ray3f(camera_origin, point_direction)

        # Check if ray from camera hits the current point without hitting other geometry
        si_visibility = scene.ray_intersect(visibility_ray, mi.Bool(True))

        # Point is directly visible if the ray hits the current point (within epsilon)
        epsilon_distance = 1e-4
        is_directly_visible = si_visibility.is_valid() & (dr.norm(si_visibility.p - si.p) < epsilon_distance)

        # Update tracking: if this point is NOT directly visible, mark that we've hit an NLOS point
        has_hit_nlos_point = has_hit_nlos_point | ~is_directly_visible

        # Zero out contributions if point is directly visible AND we haven't hit an NLOS point yet
        should_zero_contribution = ~has_hit_nlos_point
        Lr_dir = dr.select(should_zero_contribution, mi.Spectrum(0.0), Lr_dir)
        L = dr.select(should_zero_contribution, mi.Spectrum(0.0), L)

        return has_hit_nlos_point, Lr_dir, L

    def _add_pulse_samples(self,
                           sampler: mi.Sampler,
                           add_transient: Callable[[mi.Spectrum, mi.Float, mi.UnpolarizedSpectrum, mi.Mask], None],
                           Lr_dir: mi.Spectrum,
                           path_distance: mi.Float,
                           wavelengths: mi.UnpolarizedSpectrum,
                           active: mi.Bool):
        """
        Take multiple samples from the pulse distribution and add contributions.

        This computes the convolution: integral{path_contribution * pulse(t)} by
        importance sampling from the pulse distribution.

        Args:
            sampler: Random number generator
            add_transient: Callback to add transient contribution
            Lr_dir: Direct lighting contribution (unweighted by pulse)
            path_distance: Path distance to the light source
            wavelengths: Ray wavelengths
            active: Active lanes mask
        """
        for _ in range(self.pulse_samples):
            # Sample time offset from pulse distribution
            pulse_time_offset, pulse_weight = self.confocal_projector.sample_pulse(sampler.next_1d())

            # Weight contribution by pulse_weight / num_samples
            # For normalized pulses, pulse_weight = 1.0
            sample_weight = pulse_weight / self.pulse_samples

            add_transient(Lr_dir * sample_weight, path_distance + pulse_time_offset,
                          wavelengths, active)

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

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0)                            # Radiance accumulator

        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes
        distance = mi.Float(0.0)                      # Distance of the path

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)

        # Store initial camera ray intersection for NLOS light source mode
        camera_origin = mi.Point3f(ray.o)  # Store camera origin for NLOS-only check
        camera_ray_direction = mi.Vector3f(ray.d)  # Store initial camera ray direction

        # Track if we've hit at least one non-directly-visible point (for use_nlos_only)
        has_hit_nlos_point = mi.Bool(False)

        if self.camera_unwarp:
            si = scene.ray_intersect(mi.Ray3f(ray),
                                     ray_flags=mi.RayFlags.All,
                                     coherent=mi.Mask(True))
            distance[si.is_valid()] = -si.t

        while dr.hint(active,
                      max_iterations=self.max_depth,
                      label="Transient Path"):
            active_next = mi.Bool(active)

            # Compute surface interaction
            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=(depth == 0))

            # Update distance
            distance += dr.select(active, si.t, 0.0) * η

            # Get the BSDF
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

            # Check if this is the initial intersection point (depth == 0)
            is_initial_intersection = (depth == 0)

            # Use confocal projector for emitter sampling
            # For initial intersection: query projector directly with inverse square falloff
            # For subsequent bounces: importance sample from the Gaussian spots
            if is_initial_intersection:
                ds, em_weight = self.confocal_projector.query_direct(
                    scene, si, camera_origin, camera_ray_direction, active_em
                )
            else:
                ds, em_weight = self.confocal_projector.sample_emitter(
                    scene, si, camera_origin, camera_ray_direction, sampler, active_em
                )

            active_em &= (ds.pdf != 0.0)

            # Evaluate BSDF * cos(theta)
            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(
                bsdf_ctx, si, wo, active_em)
            bsdf_value_em = si.to_world_mueller(bsdf_value_em, -wo, si.wi)
            Lr_dir = β * bsdf_value_em * em_weight

            if self.use_nlos_only:
                has_hit_nlos_point, Lr_dir, L = self._apply_nlos_filter(
                    scene, si, camera_origin, has_hit_nlos_point, Lr_dir, L
                )

            # Add contribution from direct emitter sampling with pulse sampling
            path_distance = distance + ds.dist * η
            self._add_pulse_samples(sampler, add_transient, Lr_dir, path_distance,
                                    ray.wavelengths, active)

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)
            bsdf_weight = si.to_world_mueller(
                bsdf_weight, -bsdf_sample.wo, si.wi)

            # Accumulate radiance
            L = L + Le + Lr_dir

            # ---- Update loop variables based on current interaction -----
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β = β * bsdf_weight

            # Information about the current vertex needed by the next iteration
            prev_si = dr.detach(si, True)

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

            depth[si.is_valid()] += 1
            active = active_next

        return (
            L,                    # Radiance
            (depth != 0),         # Ray validity flag for alpha blending
            [],                   # Empty tuple of AOVs
            L                     # State for the differential phase
        )


mi.register_integrator("transient_path", lambda props: TransientPath(props))

del TransientADIntegrator
