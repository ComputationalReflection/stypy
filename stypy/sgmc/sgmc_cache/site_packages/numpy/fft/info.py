
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Discrete Fourier Transform (:mod:`numpy.fft`)
3: =============================================
4: 
5: .. currentmodule:: numpy.fft
6: 
7: Standard FFTs
8: -------------
9: 
10: .. autosummary::
11:    :toctree: generated/
12: 
13:    fft       Discrete Fourier transform.
14:    ifft      Inverse discrete Fourier transform.
15:    fft2      Discrete Fourier transform in two dimensions.
16:    ifft2     Inverse discrete Fourier transform in two dimensions.
17:    fftn      Discrete Fourier transform in N-dimensions.
18:    ifftn     Inverse discrete Fourier transform in N dimensions.
19: 
20: Real FFTs
21: ---------
22: 
23: .. autosummary::
24:    :toctree: generated/
25: 
26:    rfft      Real discrete Fourier transform.
27:    irfft     Inverse real discrete Fourier transform.
28:    rfft2     Real discrete Fourier transform in two dimensions.
29:    irfft2    Inverse real discrete Fourier transform in two dimensions.
30:    rfftn     Real discrete Fourier transform in N dimensions.
31:    irfftn    Inverse real discrete Fourier transform in N dimensions.
32: 
33: Hermitian FFTs
34: --------------
35: 
36: .. autosummary::
37:    :toctree: generated/
38: 
39:    hfft      Hermitian discrete Fourier transform.
40:    ihfft     Inverse Hermitian discrete Fourier transform.
41: 
42: Helper routines
43: ---------------
44: 
45: .. autosummary::
46:    :toctree: generated/
47: 
48:    fftfreq   Discrete Fourier Transform sample frequencies.
49:    rfftfreq  DFT sample frequencies (for usage with rfft, irfft).
50:    fftshift  Shift zero-frequency component to center of spectrum.
51:    ifftshift Inverse of fftshift.
52: 
53: 
54: Background information
55: ----------------------
56: 
57: Fourier analysis is fundamentally a method for expressing a function as a
58: sum of periodic components, and for recovering the function from those
59: components.  When both the function and its Fourier transform are
60: replaced with discretized counterparts, it is called the discrete Fourier
61: transform (DFT).  The DFT has become a mainstay of numerical computing in
62: part because of a very fast algorithm for computing it, called the Fast
63: Fourier Transform (FFT), which was known to Gauss (1805) and was brought
64: to light in its current form by Cooley and Tukey [CT]_.  Press et al. [NR]_
65: provide an accessible introduction to Fourier analysis and its
66: applications.
67: 
68: Because the discrete Fourier transform separates its input into
69: components that contribute at discrete frequencies, it has a great number
70: of applications in digital signal processing, e.g., for filtering, and in
71: this context the discretized input to the transform is customarily
72: referred to as a *signal*, which exists in the *time domain*.  The output
73: is called a *spectrum* or *transform* and exists in the *frequency
74: domain*.
75: 
76: Implementation details
77: ----------------------
78: 
79: There are many ways to define the DFT, varying in the sign of the
80: exponent, normalization, etc.  In this implementation, the DFT is defined
81: as
82: 
83: .. math::
84:    A_k =  \\sum_{m=0}^{n-1} a_m \\exp\\left\\{-2\\pi i{mk \\over n}\\right\\}
85:    \\qquad k = 0,\\ldots,n-1.
86: 
87: The DFT is in general defined for complex inputs and outputs, and a
88: single-frequency component at linear frequency :math:`f` is
89: represented by a complex exponential
90: :math:`a_m = \\exp\\{2\\pi i\\,f m\\Delta t\\}`, where :math:`\\Delta t`
91: is the sampling interval.
92: 
93: The values in the result follow so-called "standard" order: If ``A =
94: fft(a, n)``, then ``A[0]`` contains the zero-frequency term (the sum of
95: the signal), which is always purely real for real inputs. Then ``A[1:n/2]``
96: contains the positive-frequency terms, and ``A[n/2+1:]`` contains the
97: negative-frequency terms, in order of decreasingly negative frequency.
98: For an even number of input points, ``A[n/2]`` represents both positive and
99: negative Nyquist frequency, and is also purely real for real input.  For
100: an odd number of input points, ``A[(n-1)/2]`` contains the largest positive
101: frequency, while ``A[(n+1)/2]`` contains the largest negative frequency.
102: The routine ``np.fft.fftfreq(n)`` returns an array giving the frequencies
103: of corresponding elements in the output.  The routine
104: ``np.fft.fftshift(A)`` shifts transforms and their frequencies to put the
105: zero-frequency components in the middle, and ``np.fft.ifftshift(A)`` undoes
106: that shift.
107: 
108: When the input `a` is a time-domain signal and ``A = fft(a)``, ``np.abs(A)``
109: is its amplitude spectrum and ``np.abs(A)**2`` is its power spectrum.
110: The phase spectrum is obtained by ``np.angle(A)``.
111: 
112: The inverse DFT is defined as
113: 
114: .. math::
115:    a_m = \\frac{1}{n}\\sum_{k=0}^{n-1}A_k\\exp\\left\\{2\\pi i{mk\\over n}\\right\\}
116:    \\qquad m = 0,\\ldots,n-1.
117: 
118: It differs from the forward transform by the sign of the exponential
119: argument and the default normalization by :math:`1/n`.
120: 
121: Normalization
122: -------------
123: The default normalization has the direct transforms unscaled and the inverse
124: transforms are scaled by :math:`1/n`. It is possible to obtain unitary
125: transforms by setting the keyword argument ``norm`` to ``"ortho"`` (default is
126: `None`) so that both direct and inverse transforms will be scaled by
127: :math:`1/\\sqrt{n}`.
128: 
129: Real and Hermitian transforms
130: -----------------------------
131: 
132: When the input is purely real, its transform is Hermitian, i.e., the
133: component at frequency :math:`f_k` is the complex conjugate of the
134: component at frequency :math:`-f_k`, which means that for real
135: inputs there is no information in the negative frequency components that
136: is not already available from the positive frequency components.
137: The family of `rfft` functions is
138: designed to operate on real inputs, and exploits this symmetry by
139: computing only the positive frequency components, up to and including the
140: Nyquist frequency.  Thus, ``n`` input points produce ``n/2+1`` complex
141: output points.  The inverses of this family assumes the same symmetry of
142: its input, and for an output of ``n`` points uses ``n/2+1`` input points.
143: 
144: Correspondingly, when the spectrum is purely real, the signal is
145: Hermitian.  The `hfft` family of functions exploits this symmetry by
146: using ``n/2+1`` complex points in the input (time) domain for ``n`` real
147: points in the frequency domain.
148: 
149: In higher dimensions, FFTs are used, e.g., for image analysis and
150: filtering.  The computational efficiency of the FFT means that it can
151: also be a faster way to compute large convolutions, using the property
152: that a convolution in the time domain is equivalent to a point-by-point
153: multiplication in the frequency domain.
154: 
155: Higher dimensions
156: -----------------
157: 
158: In two dimensions, the DFT is defined as
159: 
160: .. math::
161:    A_{kl} =  \\sum_{m=0}^{M-1} \\sum_{n=0}^{N-1}
162:    a_{mn}\\exp\\left\\{-2\\pi i \\left({mk\\over M}+{nl\\over N}\\right)\\right\\}
163:    \\qquad k = 0, \\ldots, M-1;\\quad l = 0, \\ldots, N-1,
164: 
165: which extends in the obvious way to higher dimensions, and the inverses
166: in higher dimensions also extend in the same way.
167: 
168: References
169: ----------
170: 
171: .. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
172:         machine calculation of complex Fourier series," *Math. Comput.*
173:         19: 297-301.
174: 
175: .. [NR] Press, W., Teukolsky, S., Vetterline, W.T., and Flannery, B.P.,
176:         2007, *Numerical Recipes: The Art of Scientific Computing*, ch.
177:         12-13.  Cambridge Univ. Press, Cambridge, UK.
178: 
179: Examples
180: --------
181: 
182: For examples, see the various functions.
183: 
184: '''
185: from __future__ import division, absolute_import, print_function
186: 
187: depends = ['core']
188: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_101081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, (-1)), 'str', '\nDiscrete Fourier Transform (:mod:`numpy.fft`)\n=============================================\n\n.. currentmodule:: numpy.fft\n\nStandard FFTs\n-------------\n\n.. autosummary::\n   :toctree: generated/\n\n   fft       Discrete Fourier transform.\n   ifft      Inverse discrete Fourier transform.\n   fft2      Discrete Fourier transform in two dimensions.\n   ifft2     Inverse discrete Fourier transform in two dimensions.\n   fftn      Discrete Fourier transform in N-dimensions.\n   ifftn     Inverse discrete Fourier transform in N dimensions.\n\nReal FFTs\n---------\n\n.. autosummary::\n   :toctree: generated/\n\n   rfft      Real discrete Fourier transform.\n   irfft     Inverse real discrete Fourier transform.\n   rfft2     Real discrete Fourier transform in two dimensions.\n   irfft2    Inverse real discrete Fourier transform in two dimensions.\n   rfftn     Real discrete Fourier transform in N dimensions.\n   irfftn    Inverse real discrete Fourier transform in N dimensions.\n\nHermitian FFTs\n--------------\n\n.. autosummary::\n   :toctree: generated/\n\n   hfft      Hermitian discrete Fourier transform.\n   ihfft     Inverse Hermitian discrete Fourier transform.\n\nHelper routines\n---------------\n\n.. autosummary::\n   :toctree: generated/\n\n   fftfreq   Discrete Fourier Transform sample frequencies.\n   rfftfreq  DFT sample frequencies (for usage with rfft, irfft).\n   fftshift  Shift zero-frequency component to center of spectrum.\n   ifftshift Inverse of fftshift.\n\n\nBackground information\n----------------------\n\nFourier analysis is fundamentally a method for expressing a function as a\nsum of periodic components, and for recovering the function from those\ncomponents.  When both the function and its Fourier transform are\nreplaced with discretized counterparts, it is called the discrete Fourier\ntransform (DFT).  The DFT has become a mainstay of numerical computing in\npart because of a very fast algorithm for computing it, called the Fast\nFourier Transform (FFT), which was known to Gauss (1805) and was brought\nto light in its current form by Cooley and Tukey [CT]_.  Press et al. [NR]_\nprovide an accessible introduction to Fourier analysis and its\napplications.\n\nBecause the discrete Fourier transform separates its input into\ncomponents that contribute at discrete frequencies, it has a great number\nof applications in digital signal processing, e.g., for filtering, and in\nthis context the discretized input to the transform is customarily\nreferred to as a *signal*, which exists in the *time domain*.  The output\nis called a *spectrum* or *transform* and exists in the *frequency\ndomain*.\n\nImplementation details\n----------------------\n\nThere are many ways to define the DFT, varying in the sign of the\nexponent, normalization, etc.  In this implementation, the DFT is defined\nas\n\n.. math::\n   A_k =  \\sum_{m=0}^{n-1} a_m \\exp\\left\\{-2\\pi i{mk \\over n}\\right\\}\n   \\qquad k = 0,\\ldots,n-1.\n\nThe DFT is in general defined for complex inputs and outputs, and a\nsingle-frequency component at linear frequency :math:`f` is\nrepresented by a complex exponential\n:math:`a_m = \\exp\\{2\\pi i\\,f m\\Delta t\\}`, where :math:`\\Delta t`\nis the sampling interval.\n\nThe values in the result follow so-called "standard" order: If ``A =\nfft(a, n)``, then ``A[0]`` contains the zero-frequency term (the sum of\nthe signal), which is always purely real for real inputs. Then ``A[1:n/2]``\ncontains the positive-frequency terms, and ``A[n/2+1:]`` contains the\nnegative-frequency terms, in order of decreasingly negative frequency.\nFor an even number of input points, ``A[n/2]`` represents both positive and\nnegative Nyquist frequency, and is also purely real for real input.  For\nan odd number of input points, ``A[(n-1)/2]`` contains the largest positive\nfrequency, while ``A[(n+1)/2]`` contains the largest negative frequency.\nThe routine ``np.fft.fftfreq(n)`` returns an array giving the frequencies\nof corresponding elements in the output.  The routine\n``np.fft.fftshift(A)`` shifts transforms and their frequencies to put the\nzero-frequency components in the middle, and ``np.fft.ifftshift(A)`` undoes\nthat shift.\n\nWhen the input `a` is a time-domain signal and ``A = fft(a)``, ``np.abs(A)``\nis its amplitude spectrum and ``np.abs(A)**2`` is its power spectrum.\nThe phase spectrum is obtained by ``np.angle(A)``.\n\nThe inverse DFT is defined as\n\n.. math::\n   a_m = \\frac{1}{n}\\sum_{k=0}^{n-1}A_k\\exp\\left\\{2\\pi i{mk\\over n}\\right\\}\n   \\qquad m = 0,\\ldots,n-1.\n\nIt differs from the forward transform by the sign of the exponential\nargument and the default normalization by :math:`1/n`.\n\nNormalization\n-------------\nThe default normalization has the direct transforms unscaled and the inverse\ntransforms are scaled by :math:`1/n`. It is possible to obtain unitary\ntransforms by setting the keyword argument ``norm`` to ``"ortho"`` (default is\n`None`) so that both direct and inverse transforms will be scaled by\n:math:`1/\\sqrt{n}`.\n\nReal and Hermitian transforms\n-----------------------------\n\nWhen the input is purely real, its transform is Hermitian, i.e., the\ncomponent at frequency :math:`f_k` is the complex conjugate of the\ncomponent at frequency :math:`-f_k`, which means that for real\ninputs there is no information in the negative frequency components that\nis not already available from the positive frequency components.\nThe family of `rfft` functions is\ndesigned to operate on real inputs, and exploits this symmetry by\ncomputing only the positive frequency components, up to and including the\nNyquist frequency.  Thus, ``n`` input points produce ``n/2+1`` complex\noutput points.  The inverses of this family assumes the same symmetry of\nits input, and for an output of ``n`` points uses ``n/2+1`` input points.\n\nCorrespondingly, when the spectrum is purely real, the signal is\nHermitian.  The `hfft` family of functions exploits this symmetry by\nusing ``n/2+1`` complex points in the input (time) domain for ``n`` real\npoints in the frequency domain.\n\nIn higher dimensions, FFTs are used, e.g., for image analysis and\nfiltering.  The computational efficiency of the FFT means that it can\nalso be a faster way to compute large convolutions, using the property\nthat a convolution in the time domain is equivalent to a point-by-point\nmultiplication in the frequency domain.\n\nHigher dimensions\n-----------------\n\nIn two dimensions, the DFT is defined as\n\n.. math::\n   A_{kl} =  \\sum_{m=0}^{M-1} \\sum_{n=0}^{N-1}\n   a_{mn}\\exp\\left\\{-2\\pi i \\left({mk\\over M}+{nl\\over N}\\right)\\right\\}\n   \\qquad k = 0, \\ldots, M-1;\\quad l = 0, \\ldots, N-1,\n\nwhich extends in the obvious way to higher dimensions, and the inverses\nin higher dimensions also extend in the same way.\n\nReferences\n----------\n\n.. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the\n        machine calculation of complex Fourier series," *Math. Comput.*\n        19: 297-301.\n\n.. [NR] Press, W., Teukolsky, S., Vetterline, W.T., and Flannery, B.P.,\n        2007, *Numerical Recipes: The Art of Scientific Computing*, ch.\n        12-13.  Cambridge Univ. Press, Cambridge, UK.\n\nExamples\n--------\n\nFor examples, see the various functions.\n\n')

# Assigning a List to a Name (line 187):

# Obtaining an instance of the builtin type 'list' (line 187)
list_101082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 187)
# Adding element type (line 187)
str_101083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 11), 'str', 'core')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 10), list_101082, str_101083)

# Assigning a type to the variable 'depends' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'depends', list_101082)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
