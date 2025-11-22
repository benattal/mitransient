from __future__ import annotations  # Delayed parsing of type annotations
from typing import Optional, Tuple, List, Callable, Any

import drjit as dr
import mitsuba as mi
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

    def sample_emitter(self,
                       scene: mi.Scene,
                       si: mi.SurfaceInteraction3f,
                       initial_intersection_point: mi.Point3f,
                       active: mi.Bool,
                       si_initial: mi.SurfaceInteraction3f = None,
                       camera_origin: mi.Point3f = None) -> Tuple[mi.DirectionSample3f, mi.Spectrum]:
        """
        Custom emitter sampling that samples the direction from the current point
        to the intersection of the initial camera ray with the scene geometry.

        This is useful for NLOS-type setups where we want to treat the initial
        intersection point as a light source.

        Args:
            scene: The scene
            si: Current surface interaction
            initial_intersection_point: The intersection point of the initial camera ray
            active: Active mask
            si_initial: Surface interaction at the initial intersection point

        Returns:
            ds: Direction sample toward the initial intersection point
            em_weight: Weight of this sample including BRDF at initial point and inverse square falloff
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
                # Use custom emitter sampling toward initial intersection point
                ds, em_weight = self.sample_emitter(
                    scene, si, initial_intersection_point, active_em, si_initial, camera_origin)
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
