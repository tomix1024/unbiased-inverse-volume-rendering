from __future__ import annotations # Delayed parsing of type annotations
import struct

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import RBIntegrator, mis_weight

def index_spectrum(spec, idx):
    m = spec[0]
    if mi.is_rgb:
        m[dr.eq(idx, 1)] = spec[1]
        m[dr.eq(idx, 2)] = spec[2]
    return m

class MyPRBVolpathIntegrator(RBIntegrator):
    r"""
    .. _integrator-prbvolpath:

    Path Replay Backpropagation Volumetric Integrator (:monosp:`prbvolpath`)
    -------------------------------------------------------------------------

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (Default: 6)

     * - rr_depth
       - |int|
       - Specifies the path depth, at which the implementation will begin to use
         the *russian roulette* path termination criterion. For example, if set to
         1, then path generation many randomly cease after encountering directly
         visible surfaces. (Default: 5)

     * - hide_emitters
       - |bool|
       - Hide directly visible emitters. (Default: no, i.e. |false|)


    This class implements a volumetric Path Replay Backpropagation (PRB) integrator
    with the following properties:

    - Differentiable delta tracking for free-flight distance sampling

    - Emitter sampling (a.k.a. next event estimation).

    - Russian Roulette stopping criterion.

    - No reparameterization. This means that the integrator cannot be used for
      shape optimization (it will return incorrect/biased gradients for
      geometric parameters like vertex positions.)

    - Detached sampling. This means that the properties of ideal specular
      objects (e.g., the IOR of a glass vase) cannot be optimized.

    See the paper :cite:`Vicini2021` for details on PRB and differentiable delta
    tracking.

    .. tabs::

        .. code-tab:: python

            'type': 'prbvolpath',
            'max_depth': 8
    """
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', -1)
        self.rr_depth = props.get('rr_depth', 5)
        self.hide_emitters = props.get('hide_emitters', False)

        self.use_nee = False
        self.nee_handle_homogeneous = False
        self.handle_null_scattering = False
        self.is_prepared = False

    def prepare_scene(self, scene):
        if self.is_prepared:
            return

        for shape in scene.shapes():
            for medium in [shape.interior_medium(), shape.exterior_medium()]:
                if medium:
                    # Enable NEE if a medium specifically asks for it
                    self.use_nee = self.use_nee or medium.use_emitter_sampling()
                    self.nee_handle_homogeneous = self.nee_handle_homogeneous or medium.is_homogeneous()
                    self.handle_null_scattering = self.handle_null_scattering or (not medium.is_homogeneous())
        self.is_prepared = True
        # By default enable always NEE in case there are surfaces
        self.use_nee = True

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum,
               mi.Bool, mi.Spectrum]:
        self.prepare_scene(scene)

        if mode == dr.ADMode.Forward:
            raise RuntimeError("PRBVolpathIntegrator doesn't support "
                               "forward-mode differentiation!")

        is_primal = mode == dr.ADMode.Primal

        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)                          # Depth of current vertex
        Lfinal = mi.Spectrum(0 if is_primal else state_in) # The final radiance value for this pixel (if known)
        Lacc = mi.Spectrum(0)                         # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        throughput = mi.Spectrum(1)                   # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)
        escaped = mi.Bool(False)

        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        last_scatter_event = dr.zeros(mi.Interaction3f)
        last_scatter_direction_pdf = mi.Float(1.0)

        # TODO: Support sensors inside media
        medium = dr.zeros(mi.MediumPtr)

        channel = 0
        depth = mi.UInt32(0)
        valid_ray = mi.Bool(False)
        specular_chain = mi.Bool(True)

        # TODO reenable spectral sampling!
        #if mi.is_rgb: # Sample a color channel to sample free-flight distances
        #    n_channels = dr.size_v(mi.Spectrum)
        #    channel = dr.minimum(n_channels * sampler.next_1d(active), n_channels - 1)

        # Additional sampler only used in path replay pass for additional random numbers
        alt_sampler = self.create_alt_sampler(sampler, is_primal)

        loop = mi.Loop(name=f"Path Replay Backpropagation ({mode.name})",
                    state=lambda: (sampler, alt_sampler, active, escaped, depth, ray, medium, si,
                                   throughput, Lacc, needs_intersection,
                                   last_scatter_event, specular_chain, η,
                                   last_scatter_direction_pdf, valid_ray))
        while loop(active):
            # Russian roulette
            active &= dr.any(dr.neq(throughput, 0.0))
            q = dr.minimum(dr.max(throughput) * dr.sqr(η), 0.99)
            perform_rr = (depth > self.rr_depth)
            active &= (sampler.next_1d(active) < q) | ~perform_rr
            throughput[perform_rr] = throughput * dr.rcp(q)

            # Find surface interaction (where needed)
            si[needs_intersection] = scene.ray_intersect(ray, needs_intersection)
            needs_intersection &= False
            ray.maxt[active & si.is_valid()] = si.t # TODO

            # ---- Medium interaction ----

            active_medium = active & dr.neq(medium, None)
            mei, mei_weight = self.sample_real_interaction(medium, ray, sampler, channel, active_medium, is_primal)
            active_scatter = active_medium & mei.is_valid()
            active_surface = active & si.is_valid() & ~active_scatter
            # Detect rays escaped to environment emitter... (if present)
            escaped |= active & (~active_surface) & (~active_scatter)

            # --- Scattering-only gradients (albedo, sigma_t)
            with dr.resume_grad(when=not is_primal):
                sigma_t = dr.select(active_scatter, mei.sigma_t, 1.0)
                albedo = dr.select(active_scatter, medium.get_albedo(mei, active_scatter), 1.0)
                scatter_weight = albedo * sigma_t / dr.detach(sigma_t[channel])
            if not is_primal:
                # TODO implement DRT
                # TODO verify value of throughput! for DRT this is maybe important, for "regular" differentiation this does not matter.
                # TODO Where did the problematic 1/sigma_t disappear?! it's hidden in the nominator of the scatter_weight!
                Lo = Lfinal - Lacc # compute remaining radiance passing through this event
                Li = Lo / dr.maximum(1e-8, albedo*sigma_t)
                with dr.resume_grad(when=not is_primal):
                    Lo = albedo*sigma_t * Li
                    δLo = δL
                    dr.backward_from(δLo * Lo)
                del Li, Lo, δLo
            # ----------

            # Account for albedo on subsequent bounces (no-op if there was no scattering)
            throughput[active_medium] *= mei_weight * scatter_weight
            del albedo, sigma_t, scatter_weight, mei_weight

            # --- Transmittance gradients
            # We resample uniformly along the last step within the medium.
            # Note: this could also be handled by backpropagating through
            # null interactions, but then the 1/sigma_n factor from the
            # pdf also becomes problematic at locations where sigma_t is
            # very close to the majorant.
            if not is_primal:
                # Note: here, `throughput * detach(Tr * sigma_t * albedo)` cancelled out:
                #   δL * (throughput * detach(Tr * sigma_t * albedo)) * (-sigma_t)
                #   * (result / (throughput * detach(Tr * sigma_t * albedo)))
                Lo = Lfinal - Lacc # compute remaining radiance passing through this event
                δLo = δL
                adj_weight = δL * Lo
                self.backpropagate_transmittance(ray, medium, alt_sampler, active, adj_weight)
                del adj_weight, Lo, δLo
            # ----------

            # ---- End medium interaction sampling ----


            # ---------------- Surface intersection with emitters ----------------

            emitter = si.emitter(scene)
            active_e = active_surface & dr.neq(emitter, None) & ~(dr.eq(depth, 0) & self.hide_emitters)
            emitter[~active_e] = dr.zeros(mi.EmitterPtr)

            # Get the PDF of sampling this emitter using next event estimation
            if self.use_nee:
                ds = mi.DirectionSample3f(scene, si, last_scatter_event)
                emitter_pdf = dr.select(specular_chain, 0.0, scene.pdf_emitter_direction(last_scatter_event, ds, active_e))
            else:
                emitter_pdf = 0.0
            with dr.resume_grad(when=not is_primal):
                emitted = emitter.eval(si, active_e)
                contrib = throughput * emitted * mis_weight(last_scatter_direction_pdf, emitter_pdf)
            Lacc[active_e] += contrib
            if not is_primal and dr.grad_enabled(contrib):
                Lo = contrib
                δLo = δL
                dr.backward_from(δLo * Lo)

            # ---- End surface intersection with emitters ----



            # ---- Preparation for BSDF and phase function evaluation ----

            bsdf_ctx = mi.BSDFContext()
            bsdf = si.bsdf(ray)
            bsdf[~active_surface] = dr.zeros(mi.BSDFPtr)

            phase_ctx = mi.PhaseFunctionContext(sampler)
            phase = mei.medium.phase_function()
            phase[~active_scatter] = dr.zeros(mi.PhaseFunctionPtr)

            # ---- End preparation for BSDF and phase function evaluation ----

            # --------------------- Emitter sampling ---------------------
            # TODO fix this..! (really?)
            if self.use_nee:
                # Enable/disable NEE on a per-medium basis.
                # This also affects the surfaces in the medium... (TODO is this a good idea?)
                sample_emitters = mei.medium.use_emitter_sampling()
                active_e_surface = active_surface & sample_emitters & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                active_e_scatter = active_scatter & sample_emitters & ~mi.has_flag(phase.flags(), mi.PhaseFunctionFlags.Delta)
                active_e = active_e_surface | active_e_scatter

                ref_interaction = dr.zeros(mi.Interaction3f)
                ref_interaction[active_scatter] = mei
                ref_interaction[active_surface] = si
                nee_sampler = sampler if is_primal else sampler.clone()
                # Query the BSDF for that emitter-sampled direction
                with dr.resume_grad(when=not is_primal):
                    emitted, ds = self.sample_emitter(ref_interaction, scene, sampler, medium, channel, active_e)
                    bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, si.to_local(ds.d), active_e_surface)
                    phase_val = phase.eval(phase_ctx, mei, ds.d, active_e_scatter)
                    bsdf_or_phase_val = dr.select(active_e_surface, bsdf_val, phase_val)
                    bsdf_or_phase_directional_pdf = dr.detach(dr.select(ds.delta, 0.0, dr.select(active_e_surface, bsdf_pdf, phase_val)))

                    contrib = throughput * bsdf_or_phase_val * mis_weight(ds.pdf, bsdf_or_phase_directional_pdf) * emitted
                Lacc[active_e] += contrib
                if not is_primal:
                    # TODO only backprop if there is something to backpropagate to, i.e. there is a medium?
                    with dr.resume_grad():
                        Lo = contrib
                        δLo = δL
                        adj_weight = δLo * Lo
                        if dr.grad_enabled(contrib):
                            dr.backward_from(adj_weight)
                        backprop_active = active_e & dr.any(dr.neq(adj_weight, 0.0))
                    self.backpropagate_transmittance_null_tr(ref_interaction, ds, scene, nee_sampler, alt_sampler, medium, backprop_active, dr.detach(adj_weight))

            # ---- End emitter sampling ----

            # ---- Advance ray depth ----

            # Don't estimate further lighting if we exceeded number of bounces
            if self.max_depth >= 0:
                active &= depth < self.max_depth
            active_scatter &= active
            active_surface &= active

            # ---- End advance ray depth ----

            # ---- Phase sampling -----

            with dr.resume_grad(when=not is_primal):
                wo, phase_pdf = phase.sample(phase_ctx, mei, sampler.next_1d(active_scatter), sampler.next_2d(active_scatter), active_scatter)
                wo = dr.detach(wo)

            #throughput *= dr.detach(weight)
            if not is_primal and dr.grad_enabled(phase_pdf):
                weight = mi.Spectrum(1.0)
                weight[active_scatter] = dr.replace_grad(weight, phase_pdf / dr.detach(phase_pdf))
                L = Lfinal - Lacc # compute remaining radiance passing through this event
                #Li = dr.detach(dr.select(active_scatter, L / dr.maximum(1e-8, weight), 0.0))
                Li = L
                Lo = weight * Li
                δLo = δL
                dr.backward_from(δLo * Lo)

            ray[active_scatter] = mei.spawn_ray(wo)
            needs_intersection |= active_scatter
            depth[active_scatter] += 1

            # Update the last scatter PDF event if we encountered a medium scatter event
            last_scatter_event[active_scatter] = mei
            last_scatter_direction_pdf[active_scatter] = dr.detach(phase_pdf)

            # ---- End phase sampling ----

            # ----------------------- BSDF sampling ----------------------
            bs, bsdf_weight = bsdf.sample(bsdf_ctx, si, sampler.next_1d(active_surface), sampler.next_2d(active_surface), active_surface)
            active_surface &= bs.pdf > 0

            with dr.resume_grad(when=not is_primal):
                bsdf_eval = bsdf.eval(bsdf_ctx, si, bs.wo, active_surface)

            if not is_primal and dr.grad_enabled(bsdf_eval):
                L = Lfinal - Lacc # Compute remaining radiance passing through this event
                Li = dr.detach(dr.select(active_surface, L / dr.maximum(1e-8, bsdf_eval), 0.0))
                Lo = bsdf_eval * Li
                dr.backward_from(δL * Lo)

            throughput[active_surface] *= bsdf_weight
            η[active_surface] *= bs.eta

            ray[active_surface] = si.spawn_ray(si.to_world(bs.wo))
            needs_intersection |= active_surface
            non_null_bsdf = active_surface & ~mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null)
            depth[non_null_bsdf] += 1

            # Update the last scatter PDF event if we encountered a non-null scatter event
            last_scatter_event[non_null_bsdf] = si
            last_scatter_direction_pdf[non_null_bsdf] = bs.pdf

            # Transmissions might change the current medium
            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)

            # ---- End bsdf sampling ----

            valid_ray |= non_null_bsdf | active_scatter

            # specular_chain[null_bsdf] = Unchanged
            specular_chain[non_null_bsdf] = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
            specular_chain[active_scatter] = False # TODO check for dirac phase function?
            # specular_chain[act_null_scatter] = Unchanged

            active &= (active_surface | active_scatter)


        # --- Envmap contribution
        if is_primal:
            # TODO yes, we do need to consider MIS here.
            # The environment could be/will have been importance sampled via NEE at the previous interaction!
            # Also need to take transmittance into account from the previous surface interaction!!!!
            emitter = scene.environment()
            if emitter is not None:
                # All escaped rays can now query the envmap
                active_e = escaped & ~((depth <= 0) & self.hide_emitters)
                si.p[escaped] = ray.o
                si.wi[escaped] = -ray.d

                if self.use_nee:
                    assert last_scatter_event is not None
                    assert last_scatter_direction_pdf is not None
                    #ds = mi.DirectionSample3f(scene, si, last_scatter_event)
                    ds = dr.zeros(mi.DirectionSample3f)
                    ds.d = ray.d
                    emitter_pdf = emitter.pdf_direction(last_scatter_event, ds, active_e)
                    emitter_pdf = dr.select(specular_chain, 0.0, emitter_pdf)
                else:
                    emitter_pdf = 0.0

                # TODO: envmap gradients
                contrib = emitter.eval(si, active_e)
                Lacc[active_e] += throughput * contrib * mis_weight(last_scatter_direction_pdf, emitter_pdf)
                del emitter, active_e, contrib



        # Output radiance is accumulated radiance
        L = Lacc
        return L, valid_ray, L

    def sample_emitter(self, ref_interaction, scene, sampler, medium, channel, active):
        # Gradient wrt. emitter_val allowed

        active = mi.Mask(active)

        ds, emitter_val = scene.sample_emitter_direction(ref_interaction, sampler.next_2d(active), False, active)
        invalid = dr.eq(ds.pdf, 0.0)
        emitter_val[invalid] = 0.0
        active &= ~invalid
        ds = dr.detach(ds)

        with dr.suspend_grad():
            transmittance = self.estimate_transmittance_null_tr(ref_interaction, ds, scene, sampler, medium, channel, active)
        return emitter_val * transmittance, ds


    def sample_real_interaction(self, medium, ray, sampler, channel, active, is_primal):
        """
        `Medium::sample_interaction` returns an interaction that could be a null interaction.
        Here, we loop until a real interaction is sampled.

        The given ray's `maxt` value must correspond to the closest surface
        interaction (e.g. medium bounding box) in the direction of the ray.
        """

        # Sample the interaction...
        mei, weight, pcg32_state = medium.sample_interaction_real(ray, sampler.get_pcg32_state(), channel, active)
        sampler.set_pcg32_state(pcg32_state, active)

        # Get scattering coefficients with attached gradients.
        with dr.resume_grad(when=not is_primal):
            mei.sigma_s, mei.sigma_n, mei.sigma_t = medium.get_scattering_coefficients(mei, mei.is_valid())

        return mei, weight

    def estimate_transmittance_null_tr(self, it, ds, scene, sampler, medium, channel, active):

        active = mi.Mask(active)
        medium = dr.select(active, medium, dr.zeros(mi.MediumPtr))

        ray = it.spawn_ray(ds.d)
        total_dist = mi.Float(0.0)
        transmittance = mi.Spectrum(1.0)
        loop = mi.Loop(name=f"estimate_transmittance_null_tr",
                       state=lambda: (sampler, active, medium, ray, total_dist, transmittance))
        while loop(active):
            remaining_dist = ds.dist * (1.0 - mi.math.ShadowEpsilon) - total_dist
            ray.maxt = remaining_dist
            active &= remaining_dist > 0.0

            # This ray will not intersect if it reached the end of the segment
            si = scene.ray_intersect(ray, active)
            active_surface = active & si.is_valid()

            # Handle interactions with surfaces
            bsdf = si.bsdf(ray)
            bsdf_val = bsdf.eval_null_transmission(si, active_surface)
            transmittance[active_surface] *= bsdf_val
            # Deactivate upon opaque surface
            active[active_surface] &= dr.any(dr.neq(transmittance, 0))

            # Handle medium absorbtion along the ray
            ray.maxt[active_surface] = si.t
            tr_val = self.estimate_transmittance(ray, medium, sampler, channel, active)
            #if not is_primal:
            #    # TODO adj. weight only available after the loop completed!
            #    # Have to do this in a second loop (or with reservoirs)!
            #    self.backpropagate_transmittance(ray, medium, alt_sampler, active, adj_weight)
            transmittance[active] *= tr_val

            # Update the ray with new origin & t parameter
            ray[active_surface] = si.spawn_ray(mi.Vector3f(ray.d))
            total_dist[active_surface] += si.t
            # ray.maxt is updated at start of next loop

            # Continue tracing through scene if non-zero weights exist
            active &= dr.any(dr.neq(transmittance, 0.0))

            # If a medium transition is taking place: Update the medium pointer
            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)

            # Continue tracing only if we encountered a surface
            active &= active_surface

        # TODO handle derivatives wrt. transmittance and other quantities!!

        return transmittance

    def estimate_transmittance(self, ray, medium, sampler, channel, active):
        """Estimate the transmittance along a ray from `t=0` to `t=ray.maxt`.

        This simplified implementation does not support:
        - presence of surfaces within the medium
        - propagating adjoint radiance (adjoint pass)
        """

        active = mi.Mask(active)
        active &= dr.neq(medium, None)

        transmittance = mi.Spectrum(1.0)

        weight, pcg32_state = medium.estimate_transmittance(ray, sampler.get_pcg32_state(), channel, active)
        sampler.set_pcg32_state(pcg32_state, active)
        transmittance[active] = weight

        return transmittance

    def backpropagate_transmittance_null_tr(self, it, ds, scene, sampler, alt_sampler, medium, active, adj_weight, n_samples=4):

        active = mi.Mask(active)
        medium = dr.select(active, medium, dr.zeros(mi.MediumPtr))

        ray = it.spawn_ray(ds.d)
        total_dist = mi.Float(0.0)
        loop = mi.Loop(name=f"backpropagate_transmittance_null_tr",
                       state=lambda: (sampler, active, medium, ray, total_dist))
        while loop(active):
            remaining_dist = ds.dist * (1.0 - mi.math.ShadowEpsilon) - total_dist
            ray.maxt = remaining_dist
            active &= remaining_dist > 0.0

            # This ray will not intersect if it reached the end of the segment
            si = scene.ray_intersect(ray, active)
            active_surface = active & si.is_valid()

            # Handle medium absorbtion along the ray
            ray.maxt[active_surface] = si.t
            # Backpropagate transmittance along the medium segment.
            # TODO check if the medium actually exists!
            self.backpropagate_transmittance(ray, medium, alt_sampler, active, adj_weight)

            # Update the ray with new origin & t parameter
            ray[active_surface] = si.spawn_ray(mi.Vector3f(ray.d))
            total_dist[active_surface] += si.t
            # ray.maxt is updated at start of next loop

            # If a medium transition is taking place: Update the medium pointer
            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)

            # Continue tracing only if we encountered a surface
            active &= active_surface

    def backpropagate_transmittance(self, ray, medium, alt_sampler, active, adj_weight, n_samples=4):
        mei = dr.zeros(mi.MediumInteraction3f, dr.width(ray))
        mei.medium      = medium
        mei.wi          = -ray.d
        # mei.sh_frame    = mi.Frame3f(mei.wi)
        mei.time        = ray.time
        mei.wavelengths = ray.wavelengths

        contribs = mi.Spectrum(0.)
        # Pick `n_samples` uniformly over the interval
        # TODO: consider stratified sampling
        for _ in range(n_samples):
            mei.t = alt_sampler.next_1d(active) * ray.maxt
            mei.p = ray(mei.t)
            with dr.resume_grad():
                _, _, sigma_t = medium.get_scattering_coefficients(mei, active)
                # The higher sigma_t, the lower the transmittance
                contribs -= sigma_t

        # Probability of sampling each of the new distances
        inv_pdf = ray.maxt / n_samples
        with dr.resume_grad():
            contribs = dr.select(active, contribs, 0.)
            dr.backward_from(adj_weight * contribs * inv_pdf)

    def create_alt_sampler(self, sampler, is_primal):
        alt_sampler = None
        if not is_primal:
            alt_sampler = sampler.fork()
            alt_seed_rnd = alt_sampler.next_1d(True)
            # We need a secondary sampler in order to keep the
            # primary sequence of random numbers identical between
            # the primal and adjoint passes (required by PRB).
            alt_seed = struct.unpack('!I', struct.pack('!f', alt_seed_rnd[0]))[0]
            alt_seed = mi.sample_tea_32(alt_seed, 1)[0]
            alt_sampler.seed(alt_seed, sampler.wavefront_size())
        return alt_sampler

    def to_string(self):
        return f'MyPRBVolpathIntegrator[max_depth = {self.max_depth}]'


mi.register_integrator("myprbvolpath", lambda props: MyPRBVolpathIntegrator(props))
