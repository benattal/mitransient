import drjit as dr
import mitsuba as mi
import math


class Retroreflector(mi.BSDF):
    r"""
    .. bsdf-retroreflector:

    Retroreflector (:monosp:`retroreflector`)
    -----------------------------------------

    This BSDF implements a retroreflector that reflects incoming light
    back along the incident direction with a configurable lobe width.

    Unlike a mirror which reflects at an equal angle on the opposite side of the normal,
    a retroreflector sends light back towards its source.

    .. tabs::

        .. code-tab:: xml

            <bsdf type="retroreflector">
                <spectrum name="reflectance" value="1.0"/>
                <float name="lobe_exponent" value="100.0"/>
            </bsdf>

        .. code-tab:: python

            {
                'type': 'retroreflector',
                'reflectance': 1.0,
                'lobe_exponent': 100.0
            }

    .. pluginparameters::

     * - reflectance
       - |spectrum| or |texture|
       - Specifies the reflectance of the retroreflector. (Default: 1.0)

     * - lobe_exponent
       - |float|
       - Controls the sharpness of the retroreflection lobe. Higher values give
         sharper/more perfect retroreflection. (Default: 100.0)
    """

    def __init__(self, props: mi.Properties):
        mi.BSDF.__init__(self, props)

        self.m_reflectance = props.get('reflectance', mi.Color3f(1.0))
        self.m_exponent = props.get('lobe_exponent', 100.0)

        # Non-delta glossy reflection
        self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [self.m_flags]

    def sample(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
               sample1: mi.Float, sample2: mi.Point2f, active: mi.Bool):
        """
        Sample the BSDF using importance sampling around the retroreflection direction.
        """
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active = active & (cos_theta_i > 0)

        # Sample a direction from a cosine-power distribution around wi
        # This is similar to Phong lobe sampling but centered on wi instead of reflected direction

        # Sample cosine-power distribution
        cos_theta = dr.power(sample2.x, 1.0 / (self.m_exponent + 1.0))
        sin_theta = dr.sqrt(dr.maximum(0.0, 1.0 - cos_theta * cos_theta))
        phi = 2.0 * dr.pi * sample2.y

        # Local direction in frame centered on wi
        local_dir = mi.Vector3f(
            sin_theta * dr.cos(phi),
            sin_theta * dr.sin(phi),
            cos_theta
        )

        # Transform to shading frame - build frame around wi
        wi_frame = mi.Frame3f(si.wi)
        wo = wi_frame.to_world(local_dir)

        # Make sure wo is in the upper hemisphere
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active = active & (cos_theta_o > 0)

        # PDF of the sampled direction
        pdf = (self.m_exponent + 1.0) / (2.0 * dr.pi) * dr.power(cos_theta, self.m_exponent)

        # BSDF value (Phong-like lobe around wi)
        # f = reflectance * (n+2)/(2*pi) * cos^n(theta) / cos_theta_o
        # But we return f * cos_theta_o / pdf = reflectance * (n+2)/(n+1)
        cos_wi_wo = dr.dot(si.wi, wo)
        cos_wi_wo = dr.maximum(cos_wi_wo, 0.0)

        # BSDF value
        normalization = (self.m_exponent + 2.0) / (2.0 * dr.pi)
        bsdf_val = self.m_reflectance * mi.Spectrum(normalization * dr.power(cos_wi_wo, self.m_exponent))

        # Return value is f * cos_theta_o / pdf
        value = dr.select(
            mi.Spectrum(pdf) > 0,
            bsdf_val * mi.Spectrum(cos_theta_o / pdf),
            mi.Spectrum(0.0)
        )

        bs = mi.BSDFSample3f()
        bs.wo = wo
        bs.pdf = pdf
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(mi.BSDFFlags.GlossyReflection)
        bs.sampled_component = 0

        return (bs, dr.select(active, value, mi.Spectrum(0.0)))

    def eval(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
             wo: mi.Vector3f, active: mi.Bool):
        """
        Evaluate the BSDF for given incoming and outgoing directions.
        """
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)

        # Cosine of angle between wi and wo (retroreflection is when this is 1)
        cos_wi_wo = dr.dot(si.wi, wo)
        cos_wi_wo = dr.maximum(cos_wi_wo, 0.0)

        # Phong-like lobe centered on wi
        normalization = (self.m_exponent + 2.0) / (2.0 * dr.pi)
        value = self.m_reflectance * mi.Spectrum(normalization * dr.power(cos_wi_wo, self.m_exponent))

        return dr.select(active, value, mi.Spectrum(0.0))

    def pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
            wo: mi.Vector3f, active: mi.Bool):
        """
        Evaluate the PDF for the given directions.
        """
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)

        # Cosine of angle between wi and wo
        cos_wi_wo = dr.dot(si.wi, wo)
        cos_wi_wo = dr.maximum(cos_wi_wo, 0.0)

        # PDF of cosine-power distribution
        pdf = (self.m_exponent + 1.0) / (2.0 * dr.pi) * dr.power(cos_wi_wo, self.m_exponent)

        return dr.select(active, pdf, mi.Float(0.0))

    def eval_pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
                 wo: mi.Vector3f, active: mi.Bool):
        """
        Evaluate both BSDF value and PDF simultaneously.
        """
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)

        # Cosine of angle between wi and wo
        cos_wi_wo = dr.dot(si.wi, wo)
        cos_wi_wo = dr.maximum(cos_wi_wo, 0.0)

        # BSDF value
        normalization = (self.m_exponent + 2.0) / (2.0 * dr.pi)
        value = self.m_reflectance * mi.Spectrum(normalization * dr.power(cos_wi_wo, self.m_exponent))

        # PDF
        pdf = (self.m_exponent + 1.0) / (2.0 * dr.pi) * dr.power(cos_wi_wo, self.m_exponent)

        return (
            dr.select(active, value, mi.Spectrum(0.0)),
            dr.select(active, pdf, mi.Float(0.0))
        )

    def traverse(self, callback: mi.TraversalCallback):
        # Color3f is not a traversable object, so we don't expose it
        pass

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return f"Retroreflector[reflectance={self.m_reflectance}, lobe_exponent={self.m_exponent}]"


mi.register_bsdf('retroreflector', lambda props: Retroreflector(props))
