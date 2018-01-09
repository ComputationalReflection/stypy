
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =======================================
3: Signal processing (:mod:`scipy.signal`)
4: =======================================
5: 
6: Convolution
7: ===========
8: 
9: .. autosummary::
10:    :toctree: generated/
11: 
12:    convolve           -- N-dimensional convolution.
13:    correlate          -- N-dimensional correlation.
14:    fftconvolve        -- N-dimensional convolution using the FFT.
15:    convolve2d         -- 2-dimensional convolution (more options).
16:    correlate2d        -- 2-dimensional correlation (more options).
17:    sepfir2d           -- Convolve with a 2-D separable FIR filter.
18:    choose_conv_method -- Chooses faster of FFT and direct convolution methods.
19: 
20: B-splines
21: =========
22: 
23: .. autosummary::
24:    :toctree: generated/
25: 
26:    bspline        -- B-spline basis function of order n.
27:    cubic          -- B-spline basis function of order 3.
28:    quadratic      -- B-spline basis function of order 2.
29:    gauss_spline   -- Gaussian approximation to the B-spline basis function.
30:    cspline1d      -- Coefficients for 1-D cubic (3rd order) B-spline.
31:    qspline1d      -- Coefficients for 1-D quadratic (2nd order) B-spline.
32:    cspline2d      -- Coefficients for 2-D cubic (3rd order) B-spline.
33:    qspline2d      -- Coefficients for 2-D quadratic (2nd order) B-spline.
34:    cspline1d_eval -- Evaluate a cubic spline at the given points.
35:    qspline1d_eval -- Evaluate a quadratic spline at the given points.
36:    spline_filter  -- Smoothing spline (cubic) filtering of a rank-2 array.
37: 
38: Filtering
39: =========
40: 
41: .. autosummary::
42:    :toctree: generated/
43: 
44:    order_filter  -- N-dimensional order filter.
45:    medfilt       -- N-dimensional median filter.
46:    medfilt2d     -- 2-dimensional median filter (faster).
47:    wiener        -- N-dimensional wiener filter.
48: 
49:    symiirorder1  -- 2nd-order IIR filter (cascade of first-order systems).
50:    symiirorder2  -- 4th-order IIR filter (cascade of second-order systems).
51:    lfilter       -- 1-dimensional FIR and IIR digital linear filtering.
52:    lfiltic       -- Construct initial conditions for `lfilter`.
53:    lfilter_zi    -- Compute an initial state zi for the lfilter function that
54:                  -- corresponds to the steady state of the step response.
55:    filtfilt      -- A forward-backward filter.
56:    savgol_filter -- Filter a signal using the Savitzky-Golay filter.
57: 
58:    deconvolve    -- 1-d deconvolution using lfilter.
59: 
60:    sosfilt       -- 1-dimensional IIR digital linear filtering using
61:                  -- a second-order sections filter representation.
62:    sosfilt_zi    -- Compute an initial state zi for the sosfilt function that
63:                  -- corresponds to the steady state of the step response.
64:    sosfiltfilt   -- A forward-backward filter for second-order sections.
65:    hilbert       -- Compute 1-D analytic signal, using the Hilbert transform.
66:    hilbert2      -- Compute 2-D analytic signal, using the Hilbert transform.
67: 
68:    decimate      -- Downsample a signal.
69:    detrend       -- Remove linear and/or constant trends from data.
70:    resample      -- Resample using Fourier method.
71:    resample_poly -- Resample using polyphase filtering method.
72:    upfirdn       -- Upsample, apply FIR filter, downsample.
73: 
74: Filter design
75: =============
76: 
77: .. autosummary::
78:    :toctree: generated/
79: 
80:    bilinear      -- Digital filter from an analog filter using
81:                     -- the bilinear transform.
82:    findfreqs     -- Find array of frequencies for computing filter response.
83:    firls         -- FIR filter design using least-squares error minimization.
84:    firwin        -- Windowed FIR filter design, with frequency response
85:                     -- defined as pass and stop bands.
86:    firwin2       -- Windowed FIR filter design, with arbitrary frequency
87:                     -- response.
88:    freqs         -- Analog filter frequency response from TF coefficients.
89:    freqs_zpk     -- Analog filter frequency response from ZPK coefficients.
90:    freqz         -- Digital filter frequency response from TF coefficients.
91:    freqz_zpk     -- Digital filter frequency response from ZPK coefficients.
92:    sosfreqz      -- Digital filter frequency response for SOS format filter.
93:    group_delay   -- Digital filter group delay.
94:    iirdesign     -- IIR filter design given bands and gains.
95:    iirfilter     -- IIR filter design given order and critical frequencies.
96:    kaiser_atten  -- Compute the attenuation of a Kaiser FIR filter, given
97:                     -- the number of taps and the transition width at
98:                     -- discontinuities in the frequency response.
99:    kaiser_beta   -- Compute the Kaiser parameter beta, given the desired
100:                     -- FIR filter attenuation.
101:    kaiserord     -- Design a Kaiser window to limit ripple and width of
102:                     -- transition region.
103:    minimum_phase -- Convert a linear phase FIR filter to minimum phase.
104:    savgol_coeffs -- Compute the FIR filter coefficients for a Savitzky-Golay
105:                     -- filter.
106:    remez         -- Optimal FIR filter design.
107: 
108:    unique_roots  -- Unique roots and their multiplicities.
109:    residue       -- Partial fraction expansion of b(s) / a(s).
110:    residuez      -- Partial fraction expansion of b(z) / a(z).
111:    invres        -- Inverse partial fraction expansion for analog filter.
112:    invresz       -- Inverse partial fraction expansion for digital filter.
113:    BadCoefficients  -- Warning on badly conditioned filter coefficients
114: 
115: Lower-level filter design functions:
116: 
117: .. autosummary::
118:    :toctree: generated/
119: 
120:    abcd_normalize -- Check state-space matrices and ensure they are rank-2.
121:    band_stop_obj  -- Band Stop Objective Function for order minimization.
122:    besselap       -- Return (z,p,k) for analog prototype of Bessel filter.
123:    buttap         -- Return (z,p,k) for analog prototype of Butterworth filter.
124:    cheb1ap        -- Return (z,p,k) for type I Chebyshev filter.
125:    cheb2ap        -- Return (z,p,k) for type II Chebyshev filter.
126:    cmplx_sort     -- Sort roots based on magnitude.
127:    ellipap        -- Return (z,p,k) for analog prototype of elliptic filter.
128:    lp2bp          -- Transform a lowpass filter prototype to a bandpass filter.
129:    lp2bs          -- Transform a lowpass filter prototype to a bandstop filter.
130:    lp2hp          -- Transform a lowpass filter prototype to a highpass filter.
131:    lp2lp          -- Transform a lowpass filter prototype to a lowpass filter.
132:    normalize      -- Normalize polynomial representation of a transfer function.
133: 
134: 
135: 
136: Matlab-style IIR filter design
137: ==============================
138: 
139: .. autosummary::
140:    :toctree: generated/
141: 
142:    butter -- Butterworth
143:    buttord
144:    cheby1 -- Chebyshev Type I
145:    cheb1ord
146:    cheby2 -- Chebyshev Type II
147:    cheb2ord
148:    ellip -- Elliptic (Cauer)
149:    ellipord
150:    bessel -- Bessel (no order selection available -- try butterod)
151:    iirnotch      -- Design second-order IIR notch digital filter.
152:    iirpeak       -- Design second-order IIR peak (resonant) digital filter.
153: 
154: Continuous-Time Linear Systems
155: ==============================
156: 
157: .. autosummary::
158:    :toctree: generated/
159: 
160:    lti              -- Continuous-time linear time invariant system base class.
161:    StateSpace       -- Linear time invariant system in state space form.
162:    TransferFunction -- Linear time invariant system in transfer function form.
163:    ZerosPolesGain   -- Linear time invariant system in zeros, poles, gain form.
164:    lsim             -- continuous-time simulation of output to linear system.
165:    lsim2            -- like lsim, but `scipy.integrate.odeint` is used.
166:    impulse          -- impulse response of linear, time-invariant (LTI) system.
167:    impulse2         -- like impulse, but `scipy.integrate.odeint` is used.
168:    step             -- step response of continous-time LTI system.
169:    step2            -- like step, but `scipy.integrate.odeint` is used.
170:    freqresp         -- frequency response of a continuous-time LTI system.
171:    bode             -- Bode magnitude and phase data (continuous-time LTI).
172: 
173: Discrete-Time Linear Systems
174: ============================
175: 
176: .. autosummary::
177:    :toctree: generated/
178: 
179:    dlti             -- Discrete-time linear time invariant system base class.
180:    StateSpace       -- Linear time invariant system in state space form.
181:    TransferFunction -- Linear time invariant system in transfer function form.
182:    ZerosPolesGain   -- Linear time invariant system in zeros, poles, gain form.
183:    dlsim            -- simulation of output to a discrete-time linear system.
184:    dimpulse         -- impulse response of a discrete-time LTI system.
185:    dstep            -- step response of a discrete-time LTI system.
186:    dfreqresp        -- frequency response of a discrete-time LTI system.
187:    dbode            -- Bode magnitude and phase data (discrete-time LTI).
188: 
189: LTI Representations
190: ===================
191: 
192: .. autosummary::
193:    :toctree: generated/
194: 
195:    tf2zpk        -- transfer function to zero-pole-gain.
196:    tf2sos        -- transfer function to second-order sections.
197:    tf2ss         -- transfer function to state-space.
198:    zpk2tf        -- zero-pole-gain to transfer function.
199:    zpk2sos       -- zero-pole-gain to second-order sections.
200:    zpk2ss        -- zero-pole-gain to state-space.
201:    ss2tf         -- state-pace to transfer function.
202:    ss2zpk        -- state-space to pole-zero-gain.
203:    sos2zpk       -- second-order sections to zero-pole-gain.
204:    sos2tf        -- second-order sections to transfer function.
205:    cont2discrete -- continuous-time to discrete-time LTI conversion.
206:    place_poles   -- pole placement.
207: 
208: Waveforms
209: =========
210: 
211: .. autosummary::
212:    :toctree: generated/
213: 
214:    chirp        -- Frequency swept cosine signal, with several freq functions.
215:    gausspulse   -- Gaussian modulated sinusoid
216:    max_len_seq  -- Maximum length sequence
217:    sawtooth     -- Periodic sawtooth
218:    square       -- Square wave
219:    sweep_poly   -- Frequency swept cosine signal; freq is arbitrary polynomial
220:    unit_impulse -- Discrete unit impulse
221: 
222: Window functions
223: ================
224: 
225: .. autosummary::
226:    :toctree: generated/
227: 
228:    get_window        -- Return a window of a given length and type.
229:    barthann          -- Bartlett-Hann window
230:    bartlett          -- Bartlett window
231:    blackman          -- Blackman window
232:    blackmanharris    -- Minimum 4-term Blackman-Harris window
233:    bohman            -- Bohman window
234:    boxcar            -- Boxcar window
235:    chebwin           -- Dolph-Chebyshev window
236:    cosine            -- Cosine window
237:    exponential       -- Exponential window
238:    flattop           -- Flat top window
239:    gaussian          -- Gaussian window
240:    general_gaussian  -- Generalized Gaussian window
241:    hamming           -- Hamming window
242:    hann              -- Hann window
243:    hanning           -- Hann window
244:    kaiser            -- Kaiser window
245:    nuttall           -- Nuttall's minimum 4-term Blackman-Harris window
246:    parzen            -- Parzen window
247:    slepian           -- Slepian window
248:    triang            -- Triangular window
249:    tukey             -- Tukey window
250: 
251: Wavelets
252: ========
253: 
254: .. autosummary::
255:    :toctree: generated/
256: 
257:    cascade  -- compute scaling function and wavelet from coefficients
258:    daub     -- return low-pass
259:    morlet   -- Complex Morlet wavelet.
260:    qmf      -- return quadrature mirror filter from low-pass
261:    ricker   -- return ricker wavelet
262:    cwt      -- perform continuous wavelet transform
263: 
264: Peak finding
265: ============
266: 
267: .. autosummary::
268:    :toctree: generated/
269: 
270:    find_peaks_cwt -- Attempt to find the peaks in the given 1-D array
271:    argrelmin      -- Calculate the relative minima of data
272:    argrelmax      -- Calculate the relative maxima of data
273:    argrelextrema  -- Calculate the relative extrema of data
274: 
275: Spectral Analysis
276: =================
277: 
278: .. autosummary::
279:    :toctree: generated/
280: 
281:    periodogram    -- Compute a (modified) periodogram
282:    welch          -- Compute a periodogram using Welch's method
283:    csd            -- Compute the cross spectral density, using Welch's method
284:    coherence      -- Compute the magnitude squared coherence, using Welch's method
285:    spectrogram    -- Compute the spectrogram
286:    lombscargle    -- Computes the Lomb-Scargle periodogram
287:    vectorstrength -- Computes the vector strength
288:    stft           -- Compute the Short Time Fourier Transform
289:    istft          -- Compute the Inverse Short Time Fourier Transform
290:    check_COLA     -- Check the COLA constraint for iSTFT reconstruction
291: 
292: '''
293: from __future__ import division, print_function, absolute_import
294: 
295: from . import sigtools
296: from .waveforms import *
297: from ._max_len_seq import max_len_seq
298: from ._upfirdn import upfirdn
299: 
300: # The spline module (a C extension) provides:
301: #     cspline2d, qspline2d, sepfir2d, symiirord1, symiirord2
302: from .spline import *
303: 
304: from .bsplines import *
305: from .filter_design import *
306: from .fir_filter_design import *
307: from .ltisys import *
308: from .lti_conversion import *
309: from .windows import *
310: from .signaltools import *
311: from ._savitzky_golay import savgol_coeffs, savgol_filter
312: from .spectral import *
313: from .wavelets import *
314: from ._peak_finding import *
315: 
316: __all__ = [s for s in dir() if not s.startswith('_')]
317: 
318: from scipy._lib._testutils import PytestTester
319: test = PytestTester(__name__)
320: del PytestTester
321: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_288842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, (-1)), 'str', "\n=======================================\nSignal processing (:mod:`scipy.signal`)\n=======================================\n\nConvolution\n===========\n\n.. autosummary::\n   :toctree: generated/\n\n   convolve           -- N-dimensional convolution.\n   correlate          -- N-dimensional correlation.\n   fftconvolve        -- N-dimensional convolution using the FFT.\n   convolve2d         -- 2-dimensional convolution (more options).\n   correlate2d        -- 2-dimensional correlation (more options).\n   sepfir2d           -- Convolve with a 2-D separable FIR filter.\n   choose_conv_method -- Chooses faster of FFT and direct convolution methods.\n\nB-splines\n=========\n\n.. autosummary::\n   :toctree: generated/\n\n   bspline        -- B-spline basis function of order n.\n   cubic          -- B-spline basis function of order 3.\n   quadratic      -- B-spline basis function of order 2.\n   gauss_spline   -- Gaussian approximation to the B-spline basis function.\n   cspline1d      -- Coefficients for 1-D cubic (3rd order) B-spline.\n   qspline1d      -- Coefficients for 1-D quadratic (2nd order) B-spline.\n   cspline2d      -- Coefficients for 2-D cubic (3rd order) B-spline.\n   qspline2d      -- Coefficients for 2-D quadratic (2nd order) B-spline.\n   cspline1d_eval -- Evaluate a cubic spline at the given points.\n   qspline1d_eval -- Evaluate a quadratic spline at the given points.\n   spline_filter  -- Smoothing spline (cubic) filtering of a rank-2 array.\n\nFiltering\n=========\n\n.. autosummary::\n   :toctree: generated/\n\n   order_filter  -- N-dimensional order filter.\n   medfilt       -- N-dimensional median filter.\n   medfilt2d     -- 2-dimensional median filter (faster).\n   wiener        -- N-dimensional wiener filter.\n\n   symiirorder1  -- 2nd-order IIR filter (cascade of first-order systems).\n   symiirorder2  -- 4th-order IIR filter (cascade of second-order systems).\n   lfilter       -- 1-dimensional FIR and IIR digital linear filtering.\n   lfiltic       -- Construct initial conditions for `lfilter`.\n   lfilter_zi    -- Compute an initial state zi for the lfilter function that\n                 -- corresponds to the steady state of the step response.\n   filtfilt      -- A forward-backward filter.\n   savgol_filter -- Filter a signal using the Savitzky-Golay filter.\n\n   deconvolve    -- 1-d deconvolution using lfilter.\n\n   sosfilt       -- 1-dimensional IIR digital linear filtering using\n                 -- a second-order sections filter representation.\n   sosfilt_zi    -- Compute an initial state zi for the sosfilt function that\n                 -- corresponds to the steady state of the step response.\n   sosfiltfilt   -- A forward-backward filter for second-order sections.\n   hilbert       -- Compute 1-D analytic signal, using the Hilbert transform.\n   hilbert2      -- Compute 2-D analytic signal, using the Hilbert transform.\n\n   decimate      -- Downsample a signal.\n   detrend       -- Remove linear and/or constant trends from data.\n   resample      -- Resample using Fourier method.\n   resample_poly -- Resample using polyphase filtering method.\n   upfirdn       -- Upsample, apply FIR filter, downsample.\n\nFilter design\n=============\n\n.. autosummary::\n   :toctree: generated/\n\n   bilinear      -- Digital filter from an analog filter using\n                    -- the bilinear transform.\n   findfreqs     -- Find array of frequencies for computing filter response.\n   firls         -- FIR filter design using least-squares error minimization.\n   firwin        -- Windowed FIR filter design, with frequency response\n                    -- defined as pass and stop bands.\n   firwin2       -- Windowed FIR filter design, with arbitrary frequency\n                    -- response.\n   freqs         -- Analog filter frequency response from TF coefficients.\n   freqs_zpk     -- Analog filter frequency response from ZPK coefficients.\n   freqz         -- Digital filter frequency response from TF coefficients.\n   freqz_zpk     -- Digital filter frequency response from ZPK coefficients.\n   sosfreqz      -- Digital filter frequency response for SOS format filter.\n   group_delay   -- Digital filter group delay.\n   iirdesign     -- IIR filter design given bands and gains.\n   iirfilter     -- IIR filter design given order and critical frequencies.\n   kaiser_atten  -- Compute the attenuation of a Kaiser FIR filter, given\n                    -- the number of taps and the transition width at\n                    -- discontinuities in the frequency response.\n   kaiser_beta   -- Compute the Kaiser parameter beta, given the desired\n                    -- FIR filter attenuation.\n   kaiserord     -- Design a Kaiser window to limit ripple and width of\n                    -- transition region.\n   minimum_phase -- Convert a linear phase FIR filter to minimum phase.\n   savgol_coeffs -- Compute the FIR filter coefficients for a Savitzky-Golay\n                    -- filter.\n   remez         -- Optimal FIR filter design.\n\n   unique_roots  -- Unique roots and their multiplicities.\n   residue       -- Partial fraction expansion of b(s) / a(s).\n   residuez      -- Partial fraction expansion of b(z) / a(z).\n   invres        -- Inverse partial fraction expansion for analog filter.\n   invresz       -- Inverse partial fraction expansion for digital filter.\n   BadCoefficients  -- Warning on badly conditioned filter coefficients\n\nLower-level filter design functions:\n\n.. autosummary::\n   :toctree: generated/\n\n   abcd_normalize -- Check state-space matrices and ensure they are rank-2.\n   band_stop_obj  -- Band Stop Objective Function for order minimization.\n   besselap       -- Return (z,p,k) for analog prototype of Bessel filter.\n   buttap         -- Return (z,p,k) for analog prototype of Butterworth filter.\n   cheb1ap        -- Return (z,p,k) for type I Chebyshev filter.\n   cheb2ap        -- Return (z,p,k) for type II Chebyshev filter.\n   cmplx_sort     -- Sort roots based on magnitude.\n   ellipap        -- Return (z,p,k) for analog prototype of elliptic filter.\n   lp2bp          -- Transform a lowpass filter prototype to a bandpass filter.\n   lp2bs          -- Transform a lowpass filter prototype to a bandstop filter.\n   lp2hp          -- Transform a lowpass filter prototype to a highpass filter.\n   lp2lp          -- Transform a lowpass filter prototype to a lowpass filter.\n   normalize      -- Normalize polynomial representation of a transfer function.\n\n\n\nMatlab-style IIR filter design\n==============================\n\n.. autosummary::\n   :toctree: generated/\n\n   butter -- Butterworth\n   buttord\n   cheby1 -- Chebyshev Type I\n   cheb1ord\n   cheby2 -- Chebyshev Type II\n   cheb2ord\n   ellip -- Elliptic (Cauer)\n   ellipord\n   bessel -- Bessel (no order selection available -- try butterod)\n   iirnotch      -- Design second-order IIR notch digital filter.\n   iirpeak       -- Design second-order IIR peak (resonant) digital filter.\n\nContinuous-Time Linear Systems\n==============================\n\n.. autosummary::\n   :toctree: generated/\n\n   lti              -- Continuous-time linear time invariant system base class.\n   StateSpace       -- Linear time invariant system in state space form.\n   TransferFunction -- Linear time invariant system in transfer function form.\n   ZerosPolesGain   -- Linear time invariant system in zeros, poles, gain form.\n   lsim             -- continuous-time simulation of output to linear system.\n   lsim2            -- like lsim, but `scipy.integrate.odeint` is used.\n   impulse          -- impulse response of linear, time-invariant (LTI) system.\n   impulse2         -- like impulse, but `scipy.integrate.odeint` is used.\n   step             -- step response of continous-time LTI system.\n   step2            -- like step, but `scipy.integrate.odeint` is used.\n   freqresp         -- frequency response of a continuous-time LTI system.\n   bode             -- Bode magnitude and phase data (continuous-time LTI).\n\nDiscrete-Time Linear Systems\n============================\n\n.. autosummary::\n   :toctree: generated/\n\n   dlti             -- Discrete-time linear time invariant system base class.\n   StateSpace       -- Linear time invariant system in state space form.\n   TransferFunction -- Linear time invariant system in transfer function form.\n   ZerosPolesGain   -- Linear time invariant system in zeros, poles, gain form.\n   dlsim            -- simulation of output to a discrete-time linear system.\n   dimpulse         -- impulse response of a discrete-time LTI system.\n   dstep            -- step response of a discrete-time LTI system.\n   dfreqresp        -- frequency response of a discrete-time LTI system.\n   dbode            -- Bode magnitude and phase data (discrete-time LTI).\n\nLTI Representations\n===================\n\n.. autosummary::\n   :toctree: generated/\n\n   tf2zpk        -- transfer function to zero-pole-gain.\n   tf2sos        -- transfer function to second-order sections.\n   tf2ss         -- transfer function to state-space.\n   zpk2tf        -- zero-pole-gain to transfer function.\n   zpk2sos       -- zero-pole-gain to second-order sections.\n   zpk2ss        -- zero-pole-gain to state-space.\n   ss2tf         -- state-pace to transfer function.\n   ss2zpk        -- state-space to pole-zero-gain.\n   sos2zpk       -- second-order sections to zero-pole-gain.\n   sos2tf        -- second-order sections to transfer function.\n   cont2discrete -- continuous-time to discrete-time LTI conversion.\n   place_poles   -- pole placement.\n\nWaveforms\n=========\n\n.. autosummary::\n   :toctree: generated/\n\n   chirp        -- Frequency swept cosine signal, with several freq functions.\n   gausspulse   -- Gaussian modulated sinusoid\n   max_len_seq  -- Maximum length sequence\n   sawtooth     -- Periodic sawtooth\n   square       -- Square wave\n   sweep_poly   -- Frequency swept cosine signal; freq is arbitrary polynomial\n   unit_impulse -- Discrete unit impulse\n\nWindow functions\n================\n\n.. autosummary::\n   :toctree: generated/\n\n   get_window        -- Return a window of a given length and type.\n   barthann          -- Bartlett-Hann window\n   bartlett          -- Bartlett window\n   blackman          -- Blackman window\n   blackmanharris    -- Minimum 4-term Blackman-Harris window\n   bohman            -- Bohman window\n   boxcar            -- Boxcar window\n   chebwin           -- Dolph-Chebyshev window\n   cosine            -- Cosine window\n   exponential       -- Exponential window\n   flattop           -- Flat top window\n   gaussian          -- Gaussian window\n   general_gaussian  -- Generalized Gaussian window\n   hamming           -- Hamming window\n   hann              -- Hann window\n   hanning           -- Hann window\n   kaiser            -- Kaiser window\n   nuttall           -- Nuttall's minimum 4-term Blackman-Harris window\n   parzen            -- Parzen window\n   slepian           -- Slepian window\n   triang            -- Triangular window\n   tukey             -- Tukey window\n\nWavelets\n========\n\n.. autosummary::\n   :toctree: generated/\n\n   cascade  -- compute scaling function and wavelet from coefficients\n   daub     -- return low-pass\n   morlet   -- Complex Morlet wavelet.\n   qmf      -- return quadrature mirror filter from low-pass\n   ricker   -- return ricker wavelet\n   cwt      -- perform continuous wavelet transform\n\nPeak finding\n============\n\n.. autosummary::\n   :toctree: generated/\n\n   find_peaks_cwt -- Attempt to find the peaks in the given 1-D array\n   argrelmin      -- Calculate the relative minima of data\n   argrelmax      -- Calculate the relative maxima of data\n   argrelextrema  -- Calculate the relative extrema of data\n\nSpectral Analysis\n=================\n\n.. autosummary::\n   :toctree: generated/\n\n   periodogram    -- Compute a (modified) periodogram\n   welch          -- Compute a periodogram using Welch's method\n   csd            -- Compute the cross spectral density, using Welch's method\n   coherence      -- Compute the magnitude squared coherence, using Welch's method\n   spectrogram    -- Compute the spectrogram\n   lombscargle    -- Computes the Lomb-Scargle periodogram\n   vectorstrength -- Computes the vector strength\n   stft           -- Compute the Short Time Fourier Transform\n   istft          -- Compute the Inverse Short Time Fourier Transform\n   check_COLA     -- Check the COLA constraint for iSTFT reconstruction\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 295, 0))

# 'from scipy.signal import sigtools' statement (line 295)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288843 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 295, 0), 'scipy.signal')

if (type(import_288843) is not StypyTypeError):

    if (import_288843 != 'pyd_module'):
        __import__(import_288843)
        sys_modules_288844 = sys.modules[import_288843]
        import_from_module(stypy.reporting.localization.Localization(__file__, 295, 0), 'scipy.signal', sys_modules_288844.module_type_store, module_type_store, ['sigtools'])
        nest_module(stypy.reporting.localization.Localization(__file__, 295, 0), __file__, sys_modules_288844, sys_modules_288844.module_type_store, module_type_store)
    else:
        from scipy.signal import sigtools

        import_from_module(stypy.reporting.localization.Localization(__file__, 295, 0), 'scipy.signal', None, module_type_store, ['sigtools'], [sigtools])

else:
    # Assigning a type to the variable 'scipy.signal' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'scipy.signal', import_288843)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 296, 0))

# 'from scipy.signal.waveforms import ' statement (line 296)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288845 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 296, 0), 'scipy.signal.waveforms')

if (type(import_288845) is not StypyTypeError):

    if (import_288845 != 'pyd_module'):
        __import__(import_288845)
        sys_modules_288846 = sys.modules[import_288845]
        import_from_module(stypy.reporting.localization.Localization(__file__, 296, 0), 'scipy.signal.waveforms', sys_modules_288846.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 296, 0), __file__, sys_modules_288846, sys_modules_288846.module_type_store, module_type_store)
    else:
        from scipy.signal.waveforms import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 296, 0), 'scipy.signal.waveforms', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal.waveforms' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'scipy.signal.waveforms', import_288845)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 297, 0))

# 'from scipy.signal._max_len_seq import max_len_seq' statement (line 297)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288847 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 297, 0), 'scipy.signal._max_len_seq')

if (type(import_288847) is not StypyTypeError):

    if (import_288847 != 'pyd_module'):
        __import__(import_288847)
        sys_modules_288848 = sys.modules[import_288847]
        import_from_module(stypy.reporting.localization.Localization(__file__, 297, 0), 'scipy.signal._max_len_seq', sys_modules_288848.module_type_store, module_type_store, ['max_len_seq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 297, 0), __file__, sys_modules_288848, sys_modules_288848.module_type_store, module_type_store)
    else:
        from scipy.signal._max_len_seq import max_len_seq

        import_from_module(stypy.reporting.localization.Localization(__file__, 297, 0), 'scipy.signal._max_len_seq', None, module_type_store, ['max_len_seq'], [max_len_seq])

else:
    # Assigning a type to the variable 'scipy.signal._max_len_seq' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'scipy.signal._max_len_seq', import_288847)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 298, 0))

# 'from scipy.signal._upfirdn import upfirdn' statement (line 298)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288849 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 298, 0), 'scipy.signal._upfirdn')

if (type(import_288849) is not StypyTypeError):

    if (import_288849 != 'pyd_module'):
        __import__(import_288849)
        sys_modules_288850 = sys.modules[import_288849]
        import_from_module(stypy.reporting.localization.Localization(__file__, 298, 0), 'scipy.signal._upfirdn', sys_modules_288850.module_type_store, module_type_store, ['upfirdn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 298, 0), __file__, sys_modules_288850, sys_modules_288850.module_type_store, module_type_store)
    else:
        from scipy.signal._upfirdn import upfirdn

        import_from_module(stypy.reporting.localization.Localization(__file__, 298, 0), 'scipy.signal._upfirdn', None, module_type_store, ['upfirdn'], [upfirdn])

else:
    # Assigning a type to the variable 'scipy.signal._upfirdn' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'scipy.signal._upfirdn', import_288849)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 302, 0))

# 'from scipy.signal.spline import ' statement (line 302)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288851 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 302, 0), 'scipy.signal.spline')

if (type(import_288851) is not StypyTypeError):

    if (import_288851 != 'pyd_module'):
        __import__(import_288851)
        sys_modules_288852 = sys.modules[import_288851]
        import_from_module(stypy.reporting.localization.Localization(__file__, 302, 0), 'scipy.signal.spline', sys_modules_288852.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 302, 0), __file__, sys_modules_288852, sys_modules_288852.module_type_store, module_type_store)
    else:
        from scipy.signal.spline import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 302, 0), 'scipy.signal.spline', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal.spline' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 0), 'scipy.signal.spline', import_288851)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 304, 0))

# 'from scipy.signal.bsplines import ' statement (line 304)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288853 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 304, 0), 'scipy.signal.bsplines')

if (type(import_288853) is not StypyTypeError):

    if (import_288853 != 'pyd_module'):
        __import__(import_288853)
        sys_modules_288854 = sys.modules[import_288853]
        import_from_module(stypy.reporting.localization.Localization(__file__, 304, 0), 'scipy.signal.bsplines', sys_modules_288854.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 304, 0), __file__, sys_modules_288854, sys_modules_288854.module_type_store, module_type_store)
    else:
        from scipy.signal.bsplines import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 304, 0), 'scipy.signal.bsplines', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal.bsplines' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'scipy.signal.bsplines', import_288853)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 305, 0))

# 'from scipy.signal.filter_design import ' statement (line 305)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288855 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 305, 0), 'scipy.signal.filter_design')

if (type(import_288855) is not StypyTypeError):

    if (import_288855 != 'pyd_module'):
        __import__(import_288855)
        sys_modules_288856 = sys.modules[import_288855]
        import_from_module(stypy.reporting.localization.Localization(__file__, 305, 0), 'scipy.signal.filter_design', sys_modules_288856.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 305, 0), __file__, sys_modules_288856, sys_modules_288856.module_type_store, module_type_store)
    else:
        from scipy.signal.filter_design import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 305, 0), 'scipy.signal.filter_design', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal.filter_design' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 0), 'scipy.signal.filter_design', import_288855)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 306, 0))

# 'from scipy.signal.fir_filter_design import ' statement (line 306)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288857 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 306, 0), 'scipy.signal.fir_filter_design')

if (type(import_288857) is not StypyTypeError):

    if (import_288857 != 'pyd_module'):
        __import__(import_288857)
        sys_modules_288858 = sys.modules[import_288857]
        import_from_module(stypy.reporting.localization.Localization(__file__, 306, 0), 'scipy.signal.fir_filter_design', sys_modules_288858.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 306, 0), __file__, sys_modules_288858, sys_modules_288858.module_type_store, module_type_store)
    else:
        from scipy.signal.fir_filter_design import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 306, 0), 'scipy.signal.fir_filter_design', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal.fir_filter_design' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), 'scipy.signal.fir_filter_design', import_288857)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 307, 0))

# 'from scipy.signal.ltisys import ' statement (line 307)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288859 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 307, 0), 'scipy.signal.ltisys')

if (type(import_288859) is not StypyTypeError):

    if (import_288859 != 'pyd_module'):
        __import__(import_288859)
        sys_modules_288860 = sys.modules[import_288859]
        import_from_module(stypy.reporting.localization.Localization(__file__, 307, 0), 'scipy.signal.ltisys', sys_modules_288860.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 307, 0), __file__, sys_modules_288860, sys_modules_288860.module_type_store, module_type_store)
    else:
        from scipy.signal.ltisys import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 307, 0), 'scipy.signal.ltisys', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal.ltisys' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'scipy.signal.ltisys', import_288859)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 308, 0))

# 'from scipy.signal.lti_conversion import ' statement (line 308)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288861 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 308, 0), 'scipy.signal.lti_conversion')

if (type(import_288861) is not StypyTypeError):

    if (import_288861 != 'pyd_module'):
        __import__(import_288861)
        sys_modules_288862 = sys.modules[import_288861]
        import_from_module(stypy.reporting.localization.Localization(__file__, 308, 0), 'scipy.signal.lti_conversion', sys_modules_288862.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 308, 0), __file__, sys_modules_288862, sys_modules_288862.module_type_store, module_type_store)
    else:
        from scipy.signal.lti_conversion import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 308, 0), 'scipy.signal.lti_conversion', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal.lti_conversion' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 0), 'scipy.signal.lti_conversion', import_288861)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 309, 0))

# 'from scipy.signal.windows import ' statement (line 309)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288863 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 309, 0), 'scipy.signal.windows')

if (type(import_288863) is not StypyTypeError):

    if (import_288863 != 'pyd_module'):
        __import__(import_288863)
        sys_modules_288864 = sys.modules[import_288863]
        import_from_module(stypy.reporting.localization.Localization(__file__, 309, 0), 'scipy.signal.windows', sys_modules_288864.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 309, 0), __file__, sys_modules_288864, sys_modules_288864.module_type_store, module_type_store)
    else:
        from scipy.signal.windows import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 309, 0), 'scipy.signal.windows', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal.windows' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 0), 'scipy.signal.windows', import_288863)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 310, 0))

# 'from scipy.signal.signaltools import ' statement (line 310)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288865 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 310, 0), 'scipy.signal.signaltools')

if (type(import_288865) is not StypyTypeError):

    if (import_288865 != 'pyd_module'):
        __import__(import_288865)
        sys_modules_288866 = sys.modules[import_288865]
        import_from_module(stypy.reporting.localization.Localization(__file__, 310, 0), 'scipy.signal.signaltools', sys_modules_288866.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 310, 0), __file__, sys_modules_288866, sys_modules_288866.module_type_store, module_type_store)
    else:
        from scipy.signal.signaltools import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 310, 0), 'scipy.signal.signaltools', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal.signaltools' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 0), 'scipy.signal.signaltools', import_288865)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 311, 0))

# 'from scipy.signal._savitzky_golay import savgol_coeffs, savgol_filter' statement (line 311)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288867 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 311, 0), 'scipy.signal._savitzky_golay')

if (type(import_288867) is not StypyTypeError):

    if (import_288867 != 'pyd_module'):
        __import__(import_288867)
        sys_modules_288868 = sys.modules[import_288867]
        import_from_module(stypy.reporting.localization.Localization(__file__, 311, 0), 'scipy.signal._savitzky_golay', sys_modules_288868.module_type_store, module_type_store, ['savgol_coeffs', 'savgol_filter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 311, 0), __file__, sys_modules_288868, sys_modules_288868.module_type_store, module_type_store)
    else:
        from scipy.signal._savitzky_golay import savgol_coeffs, savgol_filter

        import_from_module(stypy.reporting.localization.Localization(__file__, 311, 0), 'scipy.signal._savitzky_golay', None, module_type_store, ['savgol_coeffs', 'savgol_filter'], [savgol_coeffs, savgol_filter])

else:
    # Assigning a type to the variable 'scipy.signal._savitzky_golay' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 0), 'scipy.signal._savitzky_golay', import_288867)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 312, 0))

# 'from scipy.signal.spectral import ' statement (line 312)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288869 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 312, 0), 'scipy.signal.spectral')

if (type(import_288869) is not StypyTypeError):

    if (import_288869 != 'pyd_module'):
        __import__(import_288869)
        sys_modules_288870 = sys.modules[import_288869]
        import_from_module(stypy.reporting.localization.Localization(__file__, 312, 0), 'scipy.signal.spectral', sys_modules_288870.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 312, 0), __file__, sys_modules_288870, sys_modules_288870.module_type_store, module_type_store)
    else:
        from scipy.signal.spectral import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 312, 0), 'scipy.signal.spectral', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal.spectral' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'scipy.signal.spectral', import_288869)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 313, 0))

# 'from scipy.signal.wavelets import ' statement (line 313)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288871 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 313, 0), 'scipy.signal.wavelets')

if (type(import_288871) is not StypyTypeError):

    if (import_288871 != 'pyd_module'):
        __import__(import_288871)
        sys_modules_288872 = sys.modules[import_288871]
        import_from_module(stypy.reporting.localization.Localization(__file__, 313, 0), 'scipy.signal.wavelets', sys_modules_288872.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 313, 0), __file__, sys_modules_288872, sys_modules_288872.module_type_store, module_type_store)
    else:
        from scipy.signal.wavelets import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 313, 0), 'scipy.signal.wavelets', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal.wavelets' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 0), 'scipy.signal.wavelets', import_288871)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 314, 0))

# 'from scipy.signal._peak_finding import ' statement (line 314)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288873 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 314, 0), 'scipy.signal._peak_finding')

if (type(import_288873) is not StypyTypeError):

    if (import_288873 != 'pyd_module'):
        __import__(import_288873)
        sys_modules_288874 = sys.modules[import_288873]
        import_from_module(stypy.reporting.localization.Localization(__file__, 314, 0), 'scipy.signal._peak_finding', sys_modules_288874.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 314, 0), __file__, sys_modules_288874, sys_modules_288874.module_type_store, module_type_store)
    else:
        from scipy.signal._peak_finding import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 314, 0), 'scipy.signal._peak_finding', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.signal._peak_finding' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 0), 'scipy.signal._peak_finding', import_288873)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a ListComp to a Name (line 316):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 316)
# Processing the call keyword arguments (line 316)
kwargs_288883 = {}
# Getting the type of 'dir' (line 316)
dir_288882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 22), 'dir', False)
# Calling dir(args, kwargs) (line 316)
dir_call_result_288884 = invoke(stypy.reporting.localization.Localization(__file__, 316, 22), dir_288882, *[], **kwargs_288883)

comprehension_288885 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 11), dir_call_result_288884)
# Assigning a type to the variable 's' (line 316)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 11), 's', comprehension_288885)


# Call to startswith(...): (line 316)
# Processing the call arguments (line 316)
str_288878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 48), 'str', '_')
# Processing the call keyword arguments (line 316)
kwargs_288879 = {}
# Getting the type of 's' (line 316)
s_288876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 316)
startswith_288877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 35), s_288876, 'startswith')
# Calling startswith(args, kwargs) (line 316)
startswith_call_result_288880 = invoke(stypy.reporting.localization.Localization(__file__, 316, 35), startswith_288877, *[str_288878], **kwargs_288879)

# Applying the 'not' unary operator (line 316)
result_not__288881 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 31), 'not', startswith_call_result_288880)

# Getting the type of 's' (line 316)
s_288875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 11), 's')
list_288886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 11), list_288886, s_288875)
# Assigning a type to the variable '__all__' (line 316)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 0), '__all__', list_288886)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 318, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 318)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288887 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 318, 0), 'scipy._lib._testutils')

if (type(import_288887) is not StypyTypeError):

    if (import_288887 != 'pyd_module'):
        __import__(import_288887)
        sys_modules_288888 = sys.modules[import_288887]
        import_from_module(stypy.reporting.localization.Localization(__file__, 318, 0), 'scipy._lib._testutils', sys_modules_288888.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 318, 0), __file__, sys_modules_288888, sys_modules_288888.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 318, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'scipy._lib._testutils', import_288887)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a Call to a Name (line 319):

# Call to PytestTester(...): (line 319)
# Processing the call arguments (line 319)
# Getting the type of '__name__' (line 319)
name___288890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 20), '__name__', False)
# Processing the call keyword arguments (line 319)
kwargs_288891 = {}
# Getting the type of 'PytestTester' (line 319)
PytestTester_288889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 319)
PytestTester_call_result_288892 = invoke(stypy.reporting.localization.Localization(__file__, 319, 7), PytestTester_288889, *[name___288890], **kwargs_288891)

# Assigning a type to the variable 'test' (line 319)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 0), 'test', PytestTester_call_result_288892)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 320, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
