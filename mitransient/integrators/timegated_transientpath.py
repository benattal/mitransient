"""Time-domain transient path integrator."""

from __future__ import annotations

import sys
from typing import Tuple, Union

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import ADIntegrator
from ..films.transient_hdr_film import TransientHDRFilm


class ProgressBar:
    """Simple progress bar for rendering."""

    def __init__(self, total: int = 100, desc: str = "Rendering", bar_width: int = 40):
        self.total = total
        self.desc = desc
        self.bar_width = bar_width
        self.current = 0
        self._displayed = False

    def update(self, progress: float):
        """Update progress bar with a value between 0 and 1."""
        self.current = int(progress * self.total)
        self._display()

    def _display(self):
        """Display the progress bar."""
        fraction = self.current / self.total
        filled = int(self.bar_width * fraction)
        bar = '=' * filled + '-' * (self.bar_width - filled)
        percent = fraction * 100
        sys.stdout.write(f'\r{self.desc}: [{bar}] {percent:5.1f}%')
        sys.stdout.flush()
        self._displayed = True

    def close(self):
        """Close the progress bar."""
        if self._displayed:
            sys.stdout.write('\n')
            sys.stdout.flush()


class TimeGatedTransientPath(ADIntegrator):
    r"""
    .. _integrator-timegated_transient_path:

    Time-Gated Transient Path (:monosp:`timegated_transient_path`)
    -----------------------------------------------------------------

    Time-domain transient path integrator that renders one time instant at a time.

    Unlike the standard ``transient_path`` integrator which bins contributions based on
    path length, this integrator:

    1. Samples a target time ``t`` from the sensor's temporal range
    2. Traces paths and computes total path length ``L``
    3. Evaluates emitter pulse at ``(t - L)`` - contribution is large when ``L ≈ t``
    4. Accumulates to temporal histogram at the sampled target time

    This approach naturally handles pulse shape convolution during rendering.
    The pulse shape is obtained from the confocal_projector emitter.
    Sensor response filtering should be handled by the film.

    .. tabs::

        .. code-tab:: python

            {
                'type': 'timegated_transient_path',
                'max_depth': 8,
            }

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (default: 6)

     * - rr_depth
       - |int|
       - Specifies the path depth at which the implementation will begin to use
         the *russian roulette* path termination criterion. (default: 5)

     * - camera_unwarp
       - |bool|
       - If True, does not take into account the distance from the camera origin
         to the camera ray's first intersection point. (default: false)

     * - confocal_projector
       - |emitter|
       - Reference to a ConfocalProjector emitter for custom light source sampling
         and pulse shape evaluation. (required)

     * - use_nlos_only
       - |bool|
       - If True, only allows contributions where the ray from the current
         intersection point to the camera hits geometry before reaching the camera.
         (default: false)

     * - time_sampling
       - |string|
       - Time sampling strategy: 'stratified' for stratified sampling where the
         temporal range is divided into strata (one per temporal bin) and each
         sample is jittered within its assigned stratum, or 'random' for uniform
         random sampling. Stratified sampling reduces variance by ensuring more
         uniform coverage of the temporal domain. (default: 'stratified')
    """

    # Maximum wavefront size per chunk to avoid memory issues
    MAX_SPP_PER_CHUNK = 100000

    def __init__(self, props: mi.Properties):
        super().__init__(props)

        # Core path tracing parameters
        self.max_depth = props.get("max_depth", 6)
        self.rr_depth = props.get("rr_depth", 5)
        self.camera_unwarp = props.get("camera_unwarp", False)

        # Scene configuration - confocal_projector is required and provides the pulse
        self.confocal_projector = props.get("confocal_projector", None)
        self.use_nlos_only = props.get("use_nlos_only", False)

        # Time sampling strategy
        self.time_sampling = props.get("time_sampling", "stratified")
        if self.time_sampling not in ("random", "stratified"):
            raise ValueError(f"Unknown time_sampling mode: {self.time_sampling}. "
                           f"Must be 'random' or 'stratified'.")

    def _apply_nlos_filter(self,
                           scene: mi.Scene,
                           si: mi.SurfaceInteraction3f,
                           camera_origin: mi.Point3f,
                           has_hit_nlos_point: mi.Bool,
                           Lr_dir: mi.Spectrum) -> Tuple[mi.Bool, mi.Spectrum]:
        """
        Apply NLOS-only filtering to zero out contributions from directly visible points.

        Args:
            scene: The scene
            si: Current surface interaction
            camera_origin: Camera origin position
            has_hit_nlos_point: Tracking state for whether we've hit an NLOS point
            Lr_dir: Direct lighting contribution

        Returns:
            Tuple of (updated_has_hit_nlos_point, filtered_Lr_dir)
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

        return has_hit_nlos_point, Lr_dir

    @dr.syntax
    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               target_time: mi.Float,
               active: mi.Bool = True
               ) -> Tuple[mi.Spectrum, mi.Bool, mi.Float]:
        """
        Trace path and return contribution weighted by emitter pulse.

        Args:
            scene: The scene to render
            sampler: Random number generator
            ray: Camera ray
            target_time: Target time for this sample (in OPL units)
            active: Active lanes mask

        Returns:
            L: Radiance weighted by pulse(target_time - path_length)
            valid: Whether path hit something
            path_length: Total optical path length
        """
        bsdf_ctx = mi.BSDFContext()

        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)
        L = mi.Spectrum(0)
        β = mi.Spectrum(1)  # Path throughput
        η = mi.Float(1)     # Index of refraction
        path_length = mi.Float(0.0)
        active = mi.Bool(active)

        # Store camera info for NLOS mode and confocal projector
        camera_origin = mi.Point3f(ray.o)
        camera_ray_direction = mi.Vector3f(ray.d)
        has_hit_nlos_point = mi.Bool(False)

        # Camera unwarp: subtract first intersection distance
        if self.camera_unwarp:
            si_first = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=True)
            path_length = dr.select(si_first.is_valid(), -si_first.t, mi.Float(0.0))

        # Main path tracing loop
        while dr.hint(active, max_iterations=self.max_depth,
                      label="TimeGatedTransientPath"):
            active_next = mi.Bool(active)

            # Compute surface intersection
            si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All,
                                     coherent=(depth == 0))

            # Update path length
            path_length += dr.select(active, si.t, mi.Float(0.0)) * η

            # Get the BSDF
            bsdf = si.bsdf(ray)

            # Should we continue tracing?
            active_next &= (depth + 1 < self.max_depth) & si.is_valid()

            # ----- Emitter sampling with time-shifted pulse -----
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # Check if this is the initial intersection (depth == 0)
            is_initial_intersection = (depth == 0)

            # Use confocal projector sampling
            if is_initial_intersection:
                ds, em_weight = self.confocal_projector.query_direct(
                    scene, si, camera_origin, camera_ray_direction, active_em
                )
            else:
                ds, em_weight = self.confocal_projector.sample_emitter(
                    scene, si, camera_origin, camera_ray_direction, sampler, active_em
                )

            active_em &= (ds.pdf != 0.0)

            # Time-shifted pulse evaluation
            # The pulse is centered at t=0, so contribution is large when
            # target_time ≈ total_path_length
            total_distance = path_length + ds.dist * η
            shifted_time = target_time - total_distance

            # Evaluate pulse weight from confocal projector
            pulse_weight = self.confocal_projector.eval_pulse(shifted_time)

            # BSDF evaluation
            wo = si.to_local(ds.d)
            bsdf_value, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            bsdf_value = si.to_world_mueller(bsdf_value, -wo, si.wi)

            # MIS weight
            mis_weight = dr.select(ds.delta, mi.Float(1.0),
                                   mi.ad.common.mis_weight(ds.pdf, bsdf_pdf))

            # Direct illumination contribution (weighted by pulse)
            Lr_dir = β * bsdf_value * em_weight * pulse_weight * mis_weight

            # NLOS visibility check
            if self.use_nlos_only:
                has_hit_nlos_point, Lr_dir = self._apply_nlos_filter(
                    scene, si, camera_origin, has_hit_nlos_point, Lr_dir
                )

            # Add direct illumination contribution
            L += dr.select(active_em, Lr_dir, mi.Spectrum(0.0))

            # ----- BSDF sampling for next bounce -----
            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next
            )
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

            # Update ray for next iteration
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β *= bsdf_weight

            # Russian roulette
            β_max = dr.max(mi.unpolarized_spectrum(β))
            active_next &= (β_max != 0)

            rr_prob = dr.minimum(β_max * η**2, 0.95)
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            active_next &= ~rr_active | (sampler.next_1d() < rr_prob)

            depth[si.is_valid()] += 1
            active = active_next

        return L, (depth != 0), path_length

    def _render_chunk(self,
                      scene: mi.Scene,
                      sensor: mi.Sensor,
                      film: TransientHDRFilm,
                      sampler: mi.Sampler,
                      chunk_spp: int,
                      total_spp: int,
                      seed: int,
                      film_size: mi.Vector2u,
                      start_opl: float,
                      temporal_range: float):
        """Render a single chunk of samples."""
        wavefront_size = dr.prod(film_size) * chunk_spp
        sampler.seed(seed, wavefront_size)

        # Generate rays
        ray, weight, pos = self.sample_rays(scene, sensor, sampler)

        # Sample target time based on sampling strategy
        if self.time_sampling == "stratified":
            # Stratified sampling: divide temporal range into bins,
            # assign each sample to a stratum, and jitter within it
            num_strata = film.temporal_bins
            bin_width = film.bin_width_opl

            # Create sample indices for stratification
            # Each pixel gets chunk_spp samples, we cycle through strata
            sample_idx = dr.arange(mi.UInt32, wavefront_size)
            sample_within_pixel = sample_idx % chunk_spp

            # Assign each sample to a stratum (cycle through bins)
            stratum_idx = sample_within_pixel % num_strata

            # Sample uniformly within the assigned stratum
            jitter = sampler.next_1d()
            target_time = start_opl + (mi.Float(stratum_idx) + jitter) * bin_width
        else:
            # Random sampling: uniform over entire temporal range
            target_time = start_opl + sampler.next_1d() * temporal_range

        # Trace paths with time-weighted contribution
        L, valid, path_length = self.sample(
            scene, sampler, ray, target_time, mi.Bool(True)
        )

        # Compute bin index from target_time
        bin_idx_f = (target_time - start_opl) / film.bin_width_opl

        # Scale by temporal range (importance sampling correction)
        # Since we sample uniformly in [start_opl, end_opl], pdf = 1/temporal_range
        # We need to multiply by temporal_range to correct
        L = L * temporal_range

        # Accumulate to transient storage
        coords = mi.Vector3f(pos.x, pos.y, bin_idx_f)
        valid_bin = (bin_idx_f >= 0) & (bin_idx_f < film.temporal_bins)

        film.transient_storage.put(
            pos=coords,
            wavelengths=ray.wavelengths,
            value=L * weight / total_spp,
            alpha=mi.Float(0.0),
            weight=mi.Float(0.0),
            active=valid & valid_bin
        )

        # Also accumulate to steady-state image
        block = film.steady.create_block()
        block.set_coalesce(block.coalesce() and chunk_spp >= 4)

        if mi.is_polarized:
            L_steady = mi.unpolarized_spectrum(L)
        else:
            L_steady = L

        if mi.is_spectral:
            rgb = mi.spectrum_to_srgb(L_steady * weight, ray.wavelengths)
        elif mi.is_monochromatic:
            rgb = mi.Color3f((L_steady * weight).x)
        else:
            rgb = L_steady * weight

        # Scale by chunk_spp/total_spp for proper averaging
        rgb = rgb * (chunk_spp / total_spp)

        if mi.has_flag(film.flags(), mi.FilmFlags.Alpha):
            aovs = [rgb.x, rgb.y, rgb.z, dr.select(valid, mi.Float(1), mi.Float(0)), mi.Float(1.0)]
        else:
            aovs = [rgb.x, rgb.y, rgb.z, mi.Float(1.0)]

        block.put(pos, aovs)
        film.steady.put_block(block)

    def render(self,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True,
               progress_callback=None,
               show_progress: bool = True) -> Tuple[mi.TensorXf, mi.TensorXf]:
        """
        Render transient image by sampling times and accumulating.

        Args:
            scene: The scene to render
            sensor: Sensor index or object
            seed: Random seed
            spp: Samples per pixel (0 = use sampler default)
            develop: Whether to develop the film
            evaluate: Whether to evaluate the result
            progress_callback: Optional callback for progress updates (receives float 0-1)
            show_progress: Whether to show default progress bar (default: True).
                           Ignored if progress_callback is provided.

        Returns:
            Tuple of (steady_state_image, transient_image)
        """
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Validate film type
        if not isinstance(film, TransientHDRFilm):
            raise ValueError("TimeGatedTransientPath requires a transient_hdr_film")

        # Set up progress reporting
        progress_bar = None
        if progress_callback is None and show_progress:
            progress_bar = ProgressBar(desc="Rendering")
            progress_callback = progress_bar.update

        with dr.suspend_grad():
            # Prepare sampler
            original_sampler = sensor.sampler()
            sampler = original_sampler.clone()

            if spp == 0:
                spp = sampler.sample_count()

            film_size = film.crop_size()
            if film.sample_border():
                film_size += 2 * film.rfilter().border_size()

            film.prepare([])

            # Temporal range
            start_opl = film.start_opl
            end_opl = start_opl + film.temporal_bins * film.bin_width_opl
            temporal_range = end_opl - start_opl

            # Chunk the rendering if spp is large
            if spp > self.MAX_SPP_PER_CHUNK:
                num_chunks = (spp + self.MAX_SPP_PER_CHUNK - 1) // self.MAX_SPP_PER_CHUNK
                samples_rendered = 0

                for chunk_idx in range(num_chunks):
                    # Use max chunk size, except last chunk gets the remainder
                    remaining_spp = spp - samples_rendered
                    current_chunk_spp = min(self.MAX_SPP_PER_CHUNK, remaining_spp)

                    if current_chunk_spp <= 0:
                        break

                    sampler.set_sample_count(current_chunk_spp)
                    sampler.set_samples_per_wavefront(current_chunk_spp)

                    # Use different seed for each chunk
                    chunk_seed = seed + chunk_idx

                    self._render_chunk(
                        scene, sensor, film, sampler,
                        current_chunk_spp, spp, chunk_seed,
                        film_size, start_opl, temporal_range
                    )

                    samples_rendered += current_chunk_spp

                    if progress_callback:
                        progress_callback(samples_rendered / spp)
            else:
                # Single pass rendering
                sampler.set_sample_count(spp)
                sampler.set_samples_per_wavefront(spp)

                self._render_chunk(
                    scene, sensor, film, sampler,
                    spp, spp, seed,
                    film_size, start_opl, temporal_range
                )

                if progress_callback:
                    progress_callback(1.0)

            # Close progress bar if we created one
            if progress_bar is not None:
                progress_bar.close()

            return film.develop()

    def to_string(self):
        string = f"{type(self).__name__}[\n"
        string += f"  max_depth = {self.max_depth},\n"
        string += f"  rr_depth = {self.rr_depth},\n"
        string += f"  camera_unwarp = {self.camera_unwarp},\n"
        string += f"  confocal_projector = {self.confocal_projector},\n"
        string += f"  use_nlos_only = {self.use_nlos_only},\n"
        string += f"  time_sampling = {self.time_sampling},\n"
        string += f"]"
        return string


# Register the integrator
mi.register_integrator('timegated_transient_path', lambda props: TimeGatedTransientPath(props))
