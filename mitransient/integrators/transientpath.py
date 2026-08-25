from __future__ import annotations  # Delayed parsing of type annotations
from typing import Optional, Tuple, List, Callable, Any, Union, Sequence

import drjit as dr
import mitsuba as mi

from .common import TransientADIntegrator
from mitsuba import Log, LogLevel
from mitsuba.ad.integrators.common import ADIntegrator  # type: ignore
from ..films.transient_hdr_film import TransientHDRFilm
from ..utils import β_init
from ..version import Version

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

     * - filter_direct
       - |bool|
       - If True, suppresses only the direct line-of-sight component from the
         camera's initial visible intersection while preserving indirect paths.
         This is useful when you want to remove the direct relay-wall peak
         without requiring every kept path to hit an occluded point.
         (default: false)

     * - pulse_samples
       - |int|
       - Number of samples to take from the pulse distribution per path vertex.
         Multiple samples reduce variance by spreading contributions across the
         pulse shape. Each sample is weighted by 1/(pulse_samples * pdf).
         (default: 1)

     * - use_bsdf_dc
       - |bool|
       - Replace the BSDF during transport by its analytical, view-independent
         diffuse constant component. For Principled materials this is
         ``base_color * (1 - metallic) * (1 - spec_trans) / pi``. The cosine
         factor remains part of path transport. (default: false)

    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.confocal_projector = props.get("confocal_projector", None)
        self.use_nlos_only = props.get("use_nlos_only", False)
        self.filter_direct = props.get("filter_direct", False)
        self.wall_sample = props.get("wall_sample", False)
        self.pulse_samples = props.get("pulse_samples", 1)
        self.use_bsdf_dc = props.get("use_bsdf_dc", False)
        self._bsdf_dc_materials = None
        self.wall_name = props.get("wall_name", "elm__4")
        self.wall_id = -1
        self.hidden_shape_name = props.get("hidden_shape_name", "")
        self.hidden_shape_prefix = props.get("hidden_shape_prefix", "")
        self.hidden_shape = None
        self.hidden_shapes = []

    def _prepare_bsdf_dc_materials(self, scene):
        """Cache unique scene BSDFs and constant Principled DC weights."""
        materials = []
        seen = set()
        for shape in scene.shapes():
            material = shape.bsdf()
            if material is None or material.ptr in seen:
                continue
            seen.add(material.ptr)
            params = mi.traverse(material)

            def constant_attribute(name):
                matches = [
                    key for key in params.keys()
                    if key.endswith(f"{name}.value")
                ]
                if len(matches) != 1:
                    return None
                value = params[matches[0]]
                try:
                    return float(value[0])
                except (TypeError, IndexError):
                    return float(value)

            has_metallic = bool(material.has_attribute("metallic"))
            has_spec_trans = bool(material.has_attribute("spec_trans"))
            materials.append((
                material,
                has_metallic,
                has_spec_trans,
                constant_attribute("metallic") if has_metallic else 0.0,
                constant_attribute("spec_trans") if has_spec_trans else 0.0,
            ))
        self._bsdf_dc_materials = materials

    def _eval_bsdf_dc_reflectance(self, bsdf, si, active):
        """Return rho_DC = base_color * (1-metallic) * (1-spec_trans)."""
        dc_reflectance = mi.Spectrum(0.0)
        for material, has_metallic, has_spec_trans, metallic_c, spec_trans_c in \
                self._bsdf_dc_materials:
            material_active = active & (bsdf == material)
            base_color = material.eval_diffuse_reflectance(si, material_active)
            metallic = (
                material.eval_attribute_1("metallic", si, material_active)
                if has_metallic and metallic_c is None else metallic_c
            )
            spec_trans = (
                material.eval_attribute_1("spec_trans", si, material_active)
                if has_spec_trans and spec_trans_c is None else spec_trans_c
            )
            material_weight = (1.0 - metallic) * (1.0 - spec_trans)
            dc_reflectance = dr.select(
                material_active,
                base_color * material_weight,
                dc_reflectance,
            )
        return dc_reflectance

    def _is_directly_visible(self,
                             scene: mi.Scene,
                             si: mi.SurfaceInteraction3f,
                             camera_origin: mi.Point3f) -> mi.Bool:
        """
        Check whether the current surface point is directly visible from the camera.
        """
        point_direction = dr.normalize(si.p - camera_origin)
        visibility_ray = mi.Ray3f(camera_origin, point_direction)
        si_visibility = scene.ray_intersect(visibility_ray, mi.Bool(True))

        epsilon_distance = 1e-4
        return si_visibility.is_valid() & (dr.norm(si_visibility.p - si.p) < epsilon_distance)
        # return ~scene.ray_test(si.spawn_ray_to(camera_origin), mi.Bool(True))

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
        is_directly_visible = self._is_directly_visible(scene, si, camera_origin)

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
    def get_wall_id(self, scene):
        """Helper to find the integer index of the wall shape by name"""
        shapes = scene.shapes()
        for i, shape in enumerate(shapes):
            if shape.id() == self.wall_name:
                return i
        raise Exception(f"Could not find shape with name '{self.wall_name}' in the scene.")


    @staticmethod
    def _wall_uv_from_film_pos(film_pos, film):
        """Map every jittered film sample to its pixel-center wall UV."""
        crop_offset = mi.Point2f(film.crop_offset())
        pixel_center = dr.floor(film_pos - crop_offset) + 0.5
        film_size = film.crop_size()
        return mi.Point2f(
            pixel_center.y / film_size.y,
            1.0 - pixel_center.x / film_size.x,
        )

    def sample_wall_rays(self, scene, sensor, sampler, wall_id):
        """
        Samples rays from a regular grid on the wall corresponding to the sensor resolution.
        Maps sensor pixel (x,y) -> Wall UV (u,v).
        """
        # 1. Call standard camera sampling first
        # We need this to get the 'film_pos' (pixel coordinate) and valid wavelengths.
        camera_ray, _, film_pos = self.sample_rays(scene, sensor, sampler)
        # 2. Steal valid properties
        wavelengths = camera_ray.wavelengths
        time = camera_ray.time
        
        # 3. Map all subpixel samples for a pixel to the same pixel-center UV.
        film = sensor.film()
        uv_coord = self._wall_uv_from_film_pos(film_pos, film)

        # 4. Evaluate the shape at this exact UV
        # This function takes a Point2f (uv) and returns a SurfaceInteraction (p, n, etc.)
        shape = scene.shapes()[wall_id]
        si = shape.eval_parameterization(uv_coord)
        
        # 5. Handle invalid UVs (if mesh doesn't cover [0,1])
        valid_uv = si.is_valid()

        # 6. Construct the Ray
        # Origin: The point on the wall corresponding to that pixel
        # Offset slightly along normal to avoid self-intersection (acne)
        # ray_origin = si.p + si.n * 1e-4

        ray_origin = camera_ray.o
        ray_direction = dr.normalize(si.p - ray_origin)
        ray = mi.Ray3f(ray_origin, ray_direction, time, wavelengths)

        # 7. Set Weight
        # Since we map 1 pixel = 1 UV unit area, we treat weight as 1.0.
        # Mask out invalid UVs by setting weight to 0.
        weight = dr.select(valid_uv, mi.Float(1.0), mi.Float(0.0))

        # Return film_pos so the result is splatted to the correct pixel
        return ray, weight, film_pos
    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: mi.UInt32 = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True,
               progress_callback: function = None) -> Tuple[mi.TensorXf, mi.TensorXf]:
        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        film = sensor.film()

        self.check_transient_(scene, sensor)

        # Pass temporal filter parameters to the film
        if isinstance(film, TransientHDRFilm):
            film.temporal_filter = self.temporal_filter
            film.gaussian_stddev = self.gaussian_stddev

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            samplers_spps = self.prepare(
                scene=scene,
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aov_names()
            )

            # need to re-add in case the spp parameter was set to 0
            # (spp was set through the xml file)
            total_spp = 0
            for _, spp_i in samplers_spps:
                total_spp += spp_i
            if self.wall_sample:
                self.wall_id = self.get_wall_id(scene)
            if self.hidden_shape_name:
                matches = [
                    shape for shape in scene.shapes()
                    if shape.id() == self.hidden_shape_name
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        "hidden_shape_name must identify exactly one scene shape; "
                        f"got {len(matches)} matches for {self.hidden_shape_name!r}"
                    )
                self.hidden_shape = matches[0]
                self.hidden_shapes = matches
            elif self.hidden_shape_prefix:
                self.hidden_shapes = [
                    shape for shape in scene.shapes()
                    if shape.id().startswith(self.hidden_shape_prefix)
                ]
                if not self.hidden_shapes:
                    raise RuntimeError(
                        "hidden_shape_prefix must identify at least one scene shape; "
                        f"got no matches for {self.hidden_shape_prefix!r}"
                    )
            for i, (sampler_i, spp_i) in enumerate(samplers_spps):
                # Generate a set of rays starting at the sensor
                if self.wall_sample:
                    ray, weight, pos = self.sample_wall_rays(scene, sensor, sampler_i, self.wall_id)
                else:
                    ray, weight, pos = self.sample_rays(scene, sensor, sampler_i)

                # Launch the Monte Carlo sampling process in primal mode
                L, valid, aovs, _ = self.sample(
                    mode=dr.ADMode.Primal,
                    scene=scene,
                    sampler=sampler_i,
                    ray=ray,
                    depth=mi.UInt32(0),
                    β=β_init(sensor, ray),
                    δL=None,
                    δaovs=None,
                    state_in=None,
                    active=mi.Bool(True),
                    add_transient=self.add_transient_f(
                        film=film, pos=pos, ray_weight=weight, sample_scale=1.0 / total_spp
                    )
                )

                # Prepare an ImageBlock as specified by the film
                block = film.steady.create_block()

                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp_i >= 4)

                # NOTE(diego): Mitsuba 3.6.X needs extra care when dealing
                # with polarized functions, so we'll our version instead
                splat_function = (
                    ADIntegrator._splat_to_block
                    if Version(mi.__version__) >= Version('3.7.0')
                    else self._splat_to_block
                )
                # Accumulate into the image block
                splat_function(
                    block, film, pos,
                    value=L * mi.Spectrum(weight),
                    weight=1.0,
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    aovs=aovs,
                    wavelengths=ray.wavelengths
                )

                # Explicitly delete any remaining unused variables
                del sampler_i, ray, weight, pos, L, valid

                # Perform the weight division and return an image tensor
                film.steady.put_block(block)

                # Report progress
                if progress_callback:
                    progress_callback((i + 1) / len(samplers_spps))

            steady_image, transient_image = film.develop()
            return steady_image, transient_image

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

        bsdf_ctx = mi.BSDFContext()
        if dr.hint(
                self.use_bsdf_dc and self._bsdf_dc_materials is None,
                mode="scalar"):
            self._prepare_bsdf_dc_materials(scene)

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
            if dr.hint(self.use_bsdf_dc, mode="scalar"):
                dc_reflectance = self._eval_bsdf_dc_reflectance(
                    bsdf, si, active_next & si.is_valid())

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
            if dr.hint(bool(self.hidden_shapes), mode="scalar"):
                # Keep only camera -> relay -> named-hidden-object ->
                # relay/projector paths. A folded relay can otherwise be hit
                # again at depth one and contaminate an NLOS validation capture.
                is_hidden = mi.Bool(False)
                for hidden_shape in self.hidden_shapes:
                    is_hidden |= si.shape == hidden_shape
                active_em &= (depth != 1) | is_hidden

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
            if dr.hint(self.use_bsdf_dc, mode="scalar"):
                cos_theta_o = dr.abs(wo.z)
                same_hemisphere = si.wi.z * wo.z > 0.0
                dc_active_em = active_em & same_hemisphere
                bsdf_value_em = dr.select(
                    dc_active_em,
                    dc_reflectance * (cos_theta_o * dr.inv_pi),
                    0.0,
                )
                bsdf_pdf_em = dr.select(
                    dc_active_em, cos_theta_o * dr.inv_pi, 0.0)
            else:
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(
                    bsdf_ctx, si, wo, active_em)
            bsdf_value_em = si.to_world_mueller(bsdf_value_em, -wo, si.wi)
            Lr_dir = β * bsdf_value_em * em_weight

            if self.filter_direct and is_initial_intersection:
                # The camera ray's first valid intersection is directly
                # visible by construction. Re-tracing a visibility ray here
                # is both redundant and numerically unstable at silhouettes:
                # a neighboring primitive can win the second intersection
                # query and leak a direct-return pixel into an indirect-only
                # capture. Remove the depth-zero contribution exactly.
                Lr_dir = mi.Spectrum(0.0)
                L = mi.Spectrum(0.0)

            if self.use_nlos_only:
                if is_initial_intersection:
                    # As above, depth zero is necessarily camera-visible. Do
                    # not let a silhouette visibility re-test classify it as
                    # the first hidden/NLOS interaction.
                    Lr_dir = mi.Spectrum(0.0)
                    L = mi.Spectrum(0.0)
                else:
                    has_hit_nlos_point, Lr_dir, L = self._apply_nlos_filter(
                        scene, si, camera_origin, has_hit_nlos_point, Lr_dir, L
                    )

            # Add contribution from direct emitter sampling with pulse sampling
            path_distance = distance + ds.dist * η
            self._add_pulse_samples(sampler, add_transient, Lr_dir, path_distance,
                                    ray.wavelengths, active)

            # ------------------ Detached BSDF sampling -------------------

            sample1 = sampler.next_1d()
            sample2 = sampler.next_2d()
            if dr.hint(self.use_bsdf_dc, mode="scalar"):
                bsdf_sample = dr.zeros(mi.BSDFSample3f)
                bsdf_sample.wo = mi.warp.square_to_cosine_hemisphere(sample2)
                # Match Mitsuba's two-sided wrapper exactly: on a back-side
                # interaction it flips the complete local direction, not only
                # its z component.
                bsdf_sample.wo = dr.mulsign(bsdf_sample.wo, si.wi.z)
                bsdf_sample.pdf = dr.abs(bsdf_sample.wo.z) * dr.inv_pi
                bsdf_sample.eta = 1.0
                bsdf_sample.sampled_component = 0
                bsdf_sample.sampled_type = dr.select(
                    si.wi.z > 0.0,
                    mi.UInt32(mi.BSDFFlags.DiffuseReflection |
                              mi.BSDFFlags.FrontSide),
                    mi.UInt32(mi.BSDFFlags.DiffuseReflection |
                              mi.BSDFFlags.BackSide),
                )
                bsdf_weight = dr.select(active_next, dc_reflectance, 0.0)
            else:
                bsdf_sample, bsdf_weight = bsdf.sample(
                    bsdf_ctx, si, sample1, sample2, active_next)
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
