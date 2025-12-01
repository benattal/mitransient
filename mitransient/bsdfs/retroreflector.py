import drjit as dr
import mitsuba as mi


class Retroreflector(mi.BSDF):
    r"""
    .. bsdf-retroreflector:

    Retroreflector (:monosp:`retroreflector`)
    -----------------------------------------

    This BSDF implements an ideal retroreflector that reflects all incoming light
    back along the incident direction. This is useful for simulating materials like
    road signs, safety reflectors, or cat's eye reflectors.

    Unlike a mirror which reflects at an equal angle on the opposite side of the normal,
    a retroreflector sends light directly back towards its source.

    .. tabs::

        .. code-tab:: xml

            <bsdf type="retroreflector">
                <spectrum name="reflectance" value="1.0"/>
            </bsdf>

        .. code-tab:: python

            {
                'type': 'retroreflector',
                'reflectance': 1.0
            }

    .. pluginparameters::

     * - reflectance
       - |spectrum| or |texture|
       - Specifies the reflectance of the retroreflector. (Default: 1.0)
    """

    def __init__(self, props: mi.Properties):
        mi.BSDF.__init__(self, props)

        self.m_reflectance = props.get('reflectance', mi.Color3f(1.0))

        # Delta BSDF (like a mirror, but different reflection behavior)
        self.m_flags = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [self.m_flags]

    def sample(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
               sample1: mi.Float, sample2: mi.Point2f, active: mi.Bool):
        """
        Sample the BSDF. For a retroreflector, the outgoing direction equals
        the incoming direction (light goes back where it came from).
        """
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active = active & (cos_theta_i > 0)

        bs = mi.BSDFSample3f()
        bs.pdf = 1.0
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(mi.BSDFFlags.DeltaReflection)
        bs.sampled_component = 0

        # Retroreflection: outgoing direction equals incoming direction
        # In local frame, wo = wi (light goes back the same way)
        bs.wo = si.wi

        value = self.m_reflectance / cos_theta_i

        return (bs, dr.select(active, mi.Spectrum(value), mi.Spectrum(0.0)))

    def eval(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
             wo: mi.Vector3f, active: mi.Bool):
        """
        Evaluate the BSDF. Returns zero since this is a delta distribution.
        """
        return mi.Spectrum(0.0)

    def pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
            wo: mi.Vector3f, active: mi.Bool):
        """
        Evaluate the PDF. Returns zero since this is a delta distribution.
        """
        return mi.Float(0.0)

    def eval_pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
                 wo: mi.Vector3f, active: mi.Bool):
        """
        Evaluate both BSDF value and PDF simultaneously.
        """
        return mi.Spectrum(0.0), mi.Float(0.0)

    def traverse(self, callback: mi.TraversalCallback):
        callback.put_object('reflectance', self.m_reflectance,
                           mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return f"Retroreflector[reflectance={self.m_reflectance}]"


mi.register_bsdf('retroreflector', lambda props: Retroreflector(props))
