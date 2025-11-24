from typing import Sequence

import mitsuba as mi
from mitsuba import (
    Float,
    Int32,
    ScalarInt32,
    TensorXf,
)
import drjit as dr

from mitransient.render.transient_image_block import TransientImageBlock


class TransientHDRFilm(mi.Film):
    r"""

    .. film-transient_hdr_film:

    Transient HDR Film (:monosp:`transient_hdr_film`)
    -------------------------------------------------

    mitransient's equivalent to Mitsuba 3's HDRFilm. The HDRFilm plugin creates a data structure that stores one image.
    Our transient version extends this idea to store a list of images (the transient video).

    **Specifying the start and end times of the video:** You need to specify the exposure time for each frame of the video,
    and the start time of the video.
    These values are specified in **optical path length** and not in time. All the lights of the scene emit at ``t=0``.
    Thus, for example, if you want to capture an event in your scene that starts when light has travelled 1 meter,
    and ends when light has travelled 2 meters, you should set ``start_opl=1.0``, ``bin_width_opl=0.01`` and ``temporal_bins=100``.
    This will store a video of 100 frames that starts when light has travelled 1 meter, and ends when light has travelled 2 meters.

    .. tabs::

        .. code-tab:: xml

            <film type="transient_hdr_film">
                <integer name="width"  value="256"/>
                <integer name="height" value="256"/>
                <integer name="temporal_bins" value="400"/>
                <float name="start_opl" value="1000"/>
                <float name="bin_width_opl" value="6.5"/>
                <rfilter type="box"/>
            </film>

        .. code-tab:: python

            {
                'type': 'transient_hdr_film',
                'width': 256,
                'height': 256,
                'temporal_bins': 400,
                'start_opl': 1000,
                'bin_width_opl': 6.5,
                'rfilter': {'type': 'box'}
            }

    We stores two image blocks simultaneously:

    * Steady block: Accumulates all samples (sum over all the time dimension)
    * Transient block: Accumulates samples separating them in time bins (histogram)

    The results can be retrieved using the ``develop(raw=True)`` method, which returns a ``(steady, transient)`` tuple.
    The ``transient`` image will have shape ``(width, height, temporal_bins, channels)``.

    .. pluginparameters::

     * - temporal_bins
       - |int|
       - Number of bins in the time dimension (histogram representation)

     * - bin_width_opl
       - |float|
       - Width of each bin in the time dimension (histogram representation), measured in optical path length

     * - start_opl
       - |float|
       - Start of the time dimension (histogram representation), measured in optical path length

    See also, from `mi.Film <https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_films.html>`_:

    * `width` (integer)
    * `height` (integer)
    * `crop_width` (integer)
    * `crop_height` (integer)
    * `crop_offset_x` (integer)
    * `crop_offset_y` (integer)
    * `sample_border` (bool)
    * `rfilter` (rfilter)
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.temporal_bins = props.get("temporal_bins", mi.UInt32(2048))
        self.bin_width_opl = props.get("bin_width_opl", mi.Float(0.003))
        self.start_opl = props.get("start_opl", mi.Float(0))

        # Temporal filter parameters (set by integrator during rendering)
        self.temporal_filter = "box"
        self.gaussian_stddev = 0.5

    def end_opl(self):
        return self.start_opl + self.bin_width_opl * self.temporal_bins

    def base_channels_count(self):
        return self.steady.base_channels_count()

    def prepare(self, aovs: Sequence[str]):
        # Prepare steady film
        steady_hdrfilm_dict = {
            'type': 'hdrfilm',
            'width': self.size().x,
            'height': self.size().y,
            'pixel_format': "luminance" if mi.is_monochromatic else "rgb",
            'crop_offset_x': self.crop_offset().x,
            'crop_offset_y': self.crop_offset().y,
            'crop_width': self.crop_size().x,
            'crop_height': self.crop_size().y,
            'sample_border': self.sample_border(),
            'rfilter': self.rfilter()
        }
        self.steady: mi.Film = mi.load_dict(steady_hdrfilm_dict)
        self.steady.prepare(aovs)

        # Prepare transient image block
        channels = self.prepare_transient_(aovs)
        return channels

    def create_block(self):
        return TransientImageBlock(
            size_xyt=self.crop_size_xyt,
            offset_xyt=self.crop_offset_xyt,
            channel_count=len(self.channels),
            rfilter=self.rfilter()
        )

    def gather_derivatives_at_distance(self, pos, δL, distance: mi.Float):
        pos_distance = (distance - self.start_opl) / self.bin_width_opl
        coords = mi.Vector3f(pos.x, pos.y, pos_distance)
        indices = dr.fma(coords.x, self.size().y * self.temporal_bins,
                         dr.fma(coords.y, self.temporal_bins, coords.z))
        alpha = mi.has_flag(self.flags(), mi.FilmFlags.Alpha)
        color_channels = len(self.channels) - (2 if alpha else 1)
        active_g = (indices > 0) & (
            indices < dr.prod(δL.shape) // color_channels)
        result = dr.gather(mi.Spectrum, δL, indices, active=active_g)
        return result

    def prepare_transient_(self, aovs: Sequence[str]):
        alpha = mi.has_flag(self.flags(), mi.FilmFlags.Alpha)

        if mi.is_monochromatic and mi.is_polarized:
            base_channels = "0123" + ("AW" if alpha else "W")
        elif mi.is_monochromatic:
            base_channels = "LAW" if alpha else "LW"
        else:
            # RGB
            base_channels = "RGBAW" if alpha else "RGBW"

        channels = []
        for i in range(len(base_channels)):
            channels.append(base_channels[i])

        for i in range(len(aovs)):
            channels.append(aovs[i])

        self.channels = channels
        self.crop_offset_xyt = mi.ScalarPoint3i(
            self.crop_offset().x, self.crop_offset().y, 0)
        self.crop_size_xyt = mi.ScalarVector3u(
            self.size().x, self.size().y, self.temporal_bins)

        self.transient_storage = self.create_block()
        if len(set(channels)) != len(channels):
            mi.Log(mi.LogLevel.Error,
                   "Film::prepare_transient_(): duplicate channel name.")

        return len(self.channels)

    def clear(self):
        self.steady.clear()

        if self.transient_storage:
            self.transient_storage.clear()

    def develop(self, raw: bool = False):
        steady_image = self.steady.develop(raw=raw)
        transient_image = self.develop_transient_(raw=raw)

        return steady_image, transient_image

    def _create_pulse_kernel(self, filter_type: str, stddev: float, kernel_size: int):
        """
        Create a pulse shape kernel for convolution.

        Args:
            filter_type: 'box' or 'gaussian'
            stddev: Standard deviation for Gaussian filter (in bins)
            kernel_size: Size of the kernel

        Returns:
            DrJit array containing the normalized kernel
        """
        if filter_type == "box":
            # Box filter - just return delta function (no convolution)
            kernel = dr.zeros(Float, kernel_size)
            kernel[kernel_size // 2] = 1.0
            return kernel
        elif filter_type == "gaussian":
            # Gaussian filter
            center = kernel_size // 2
            t = dr.arange(Float, kernel_size) - float(center)
            kernel = dr.exp(-0.5 * (t / float(stddev)) ** 2)
            # Normalize
            kernel_sum = dr.sum(kernel)[0]
            kernel = kernel / kernel_sum
            return kernel
        else:
            mi.Log(mi.LogLevel.Warn,
                   f"Unknown temporal_filter type '{filter_type}', using box filter")
            kernel = dr.zeros(Float, kernel_size)
            kernel[kernel_size // 2] = 1.0
            return kernel

    def _convolve_temporal_drjit(self, data: TensorXf):
        """
        Convolve the transient data with a pulse shape along the temporal dimension.

        Args:
            data: Tensor of shape (height, width, temporal_bins, channels)

        Returns:
            Convolved tensor of the same shape
        """
        if self.temporal_filter == "box":
            # No convolution needed for box filter
            return data

        h, w, t, c = data.shape

        # Create kernel based on filter type
        # Kernel size should be large enough to capture the filter
        kernel_size = self.temporal_bins
        kernel = self._create_pulse_kernel(self.temporal_filter, self.gaussian_stddev, kernel_size)

        # Vectorized convolution along temporal dimension
        half_kernel = kernel_size // 2
        result = dr.zeros(Float, h * w * t * c)

        # For each kernel tap, gather values and accumulate
        for k in range(kernel_size):
            t_offset = k - half_kernel

            # Create indices for this kernel tap
            # Shape: all pixels, all time bins, all channels
            idx = dr.arange(Int32, h * w * t * c)

            # Decompose flat index into (pixel, time, channel)
            pixel_idx = idx // (t * c)
            remainder = idx - pixel_idx * (t * c)
            t_idx = remainder // c
            ch_idx = remainder - t_idx * c

            # Calculate source time index with offset
            t_src = t_idx + t_offset

            # Check if within bounds
            valid = (t_src >= 0) & (t_src < t)

            # Calculate source index
            src_idx = pixel_idx * (t * c) + t_src * c + ch_idx

            # Gather values (with zero for out-of-bounds)
            values = dr.gather(Float, data.array, src_idx, valid)

            # Accumulate weighted contribution
            dr.scatter_reduce(dr.ReduceOp.Add, result, values * kernel[k], idx, valid)

        return TensorXf(result, (h, w, t, c))

    def develop_transient_(self, raw: bool = False):
        if not self.transient_storage:
            mi.Log(mi.LogLevel.Error,
                   "No transient storage allocated, was prepare_transient_() called first?")

        if raw:
            return self.transient_storage.tensor

        data = self.transient_storage.tensor

        pixel_count = dr.prod(data.shape[0:-1])
        source_ch = data.shape[-1]
        # Remove alpha and weight channels
        alpha = mi.has_flag(self.flags(), mi.FilmFlags.Alpha)
        target_ch = source_ch - (ScalarInt32(2) if alpha else ScalarInt32(1))

        idx = dr.arange(Int32, pixel_count * target_ch)
        pixel_idx = idx // target_ch
        channel_idx = dr.fma(pixel_idx, -target_ch, idx)

        values_idx = dr.fma(pixel_idx, source_ch, channel_idx)
        weight_idx = dr.fma(pixel_idx, source_ch, source_ch - 1)

        weight = dr.gather(Float, data.array, weight_idx)
        values_ = dr.gather(Float, data.array, values_idx)

        values = values_ / dr.select((weight == 0.0), 1.0, weight)

        result = TensorXf(values, tuple(list(data.shape[0:-1]) + [target_ch]))

        # Apply pulse shape convolution
        result = self._convolve_temporal_drjit(result)

        return result

    def add_transient_data(self, pos: mi.Vector2f, distance: mi.Float,
                           wavelengths: mi.UnpolarizedSpectrum, spec: mi.Spectrum,
                           ray_weight: mi.Float, active: mi.Bool):
        """
        Add a path's contribution to the film:
        * pos: pixel position
        * distance: distance traveled by the path (opl)
        * wavelengths: for spectral rendering, wavelengths sampled
        * spec: Spectrum / contribution of the path
        * ray_weight: weight of the ray given by the sensor
        * active: mask
        """
        pos_distance = (distance - self.start_opl) / self.bin_width_opl
        coords = mi.Vector3f(pos.x, pos.y, pos_distance)
        mask = (pos_distance >= 0) & (pos_distance < self.temporal_bins)
        self.transient_storage.put(
            pos=coords,
            wavelengths=wavelengths,
            value=spec * mi.Spectrum(ray_weight),
            alpha=mi.Float(0.0),
            # value should have the sample scale already multiplied
            weight=mi.Float(0.0),
            active=active & mask,
        )

    def to_string(self):
        string = "TransientHDRFilm[\n"
        string += f"  size = {self.size()},\n"
        string += f"  crop_size = {self.crop_size()},\n"
        string += f"  crop_offset = {self.crop_offset()},\n"
        string += f"  sample_border = {self.sample_border()},\n"
        string += f"  filter = {self.rfilter()},\n"
        string += f"  temporal_bins = {self.temporal_bins},\n"
        string += f"  bin_width_opl = {self.bin_width_opl},\n"
        string += f"  start_opl = {self.start_opl},\n"
        string += f"]"
        return string

    def traverse(self, callback):
        super().traverse(callback)
        callback.put(
            "temporal_bins", self.temporal_bins, mi.ParamFlags.NonDifferentiable)
        callback.put(
            "bin_width_opl", self.bin_width_opl, mi.ParamFlags.NonDifferentiable)
        callback.put(
            "start_opl", self.start_opl, mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        super().parameters_changed(keys)


mi.register_film("transient_hdr_film", lambda props: TransientHDRFilm(props))
