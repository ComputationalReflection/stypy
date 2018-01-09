
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.dual import eig
5: from scipy.special import comb
6: from scipy import linspace, pi, exp
7: from scipy.signal import convolve
8: 
9: __all__ = ['daub', 'qmf', 'cascade', 'morlet', 'ricker', 'cwt']
10: 
11: 
12: def daub(p):
13:     '''
14:     The coefficients for the FIR low-pass filter producing Daubechies wavelets.
15: 
16:     p>=1 gives the order of the zero at f=1/2.
17:     There are 2p filter coefficients.
18: 
19:     Parameters
20:     ----------
21:     p : int
22:         Order of the zero at f=1/2, can have values from 1 to 34.
23: 
24:     Returns
25:     -------
26:     daub : ndarray
27:         Return
28: 
29:     '''
30:     sqrt = np.sqrt
31:     if p < 1:
32:         raise ValueError("p must be at least 1.")
33:     if p == 1:
34:         c = 1 / sqrt(2)
35:         return np.array([c, c])
36:     elif p == 2:
37:         f = sqrt(2) / 8
38:         c = sqrt(3)
39:         return f * np.array([1 + c, 3 + c, 3 - c, 1 - c])
40:     elif p == 3:
41:         tmp = 12 * sqrt(10)
42:         z1 = 1.5 + sqrt(15 + tmp) / 6 - 1j * (sqrt(15) + sqrt(tmp - 15)) / 6
43:         z1c = np.conj(z1)
44:         f = sqrt(2) / 8
45:         d0 = np.real((1 - z1) * (1 - z1c))
46:         a0 = np.real(z1 * z1c)
47:         a1 = 2 * np.real(z1)
48:         return f / d0 * np.array([a0, 3 * a0 - a1, 3 * a0 - 3 * a1 + 1,
49:                                   a0 - 3 * a1 + 3, 3 - a1, 1])
50:     elif p < 35:
51:         # construct polynomial and factor it
52:         if p < 35:
53:             P = [comb(p - 1 + k, k, exact=1) for k in range(p)][::-1]
54:             yj = np.roots(P)
55:         else:  # try different polynomial --- needs work
56:             P = [comb(p - 1 + k, k, exact=1) / 4.0**k
57:                  for k in range(p)][::-1]
58:             yj = np.roots(P) / 4
59:         # for each root, compute two z roots, select the one with |z|>1
60:         # Build up final polynomial
61:         c = np.poly1d([1, 1])**p
62:         q = np.poly1d([1])
63:         for k in range(p - 1):
64:             yval = yj[k]
65:             part = 2 * sqrt(yval * (yval - 1))
66:             const = 1 - 2 * yval
67:             z1 = const + part
68:             if (abs(z1)) < 1:
69:                 z1 = const - part
70:             q = q * [1, -z1]
71: 
72:         q = c * np.real(q)
73:         # Normalize result
74:         q = q / np.sum(q) * sqrt(2)
75:         return q.c[::-1]
76:     else:
77:         raise ValueError("Polynomial factorization does not work "
78:                          "well for p too large.")
79: 
80: 
81: def qmf(hk):
82:     '''
83:     Return high-pass qmf filter from low-pass
84: 
85:     Parameters
86:     ----------
87:     hk : array_like
88:         Coefficients of high-pass filter.
89: 
90:     '''
91:     N = len(hk) - 1
92:     asgn = [{0: 1, 1: -1}[k % 2] for k in range(N + 1)]
93:     return hk[::-1] * np.array(asgn)
94: 
95: 
96: def cascade(hk, J=7):
97:     '''
98:     Return (x, phi, psi) at dyadic points ``K/2**J`` from filter coefficients.
99: 
100:     Parameters
101:     ----------
102:     hk : array_like
103:         Coefficients of low-pass filter.
104:     J : int, optional
105:         Values will be computed at grid points ``K/2**J``. Default is 7.
106: 
107:     Returns
108:     -------
109:     x : ndarray
110:         The dyadic points ``K/2**J`` for ``K=0...N * (2**J)-1`` where
111:         ``len(hk) = len(gk) = N+1``.
112:     phi : ndarray
113:         The scaling function ``phi(x)`` at `x`:
114:         ``phi(x) = sum(hk * phi(2x-k))``, where k is from 0 to N.
115:     psi : ndarray, optional
116:         The wavelet function ``psi(x)`` at `x`:
117:         ``phi(x) = sum(gk * phi(2x-k))``, where k is from 0 to N.
118:         `psi` is only returned if `gk` is not None.
119: 
120:     Notes
121:     -----
122:     The algorithm uses the vector cascade algorithm described by Strang and
123:     Nguyen in "Wavelets and Filter Banks".  It builds a dictionary of values
124:     and slices for quick reuse.  Then inserts vectors into final vector at the
125:     end.
126: 
127:     '''
128:     N = len(hk) - 1
129: 
130:     if (J > 30 - np.log2(N + 1)):
131:         raise ValueError("Too many levels.")
132:     if (J < 1):
133:         raise ValueError("Too few levels.")
134: 
135:     # construct matrices needed
136:     nn, kk = np.ogrid[:N, :N]
137:     s2 = np.sqrt(2)
138:     # append a zero so that take works
139:     thk = np.r_[hk, 0]
140:     gk = qmf(hk)
141:     tgk = np.r_[gk, 0]
142: 
143:     indx1 = np.clip(2 * nn - kk, -1, N + 1)
144:     indx2 = np.clip(2 * nn - kk + 1, -1, N + 1)
145:     m = np.zeros((2, 2, N, N), 'd')
146:     m[0, 0] = np.take(thk, indx1, 0)
147:     m[0, 1] = np.take(thk, indx2, 0)
148:     m[1, 0] = np.take(tgk, indx1, 0)
149:     m[1, 1] = np.take(tgk, indx2, 0)
150:     m *= s2
151: 
152:     # construct the grid of points
153:     x = np.arange(0, N * (1 << J), dtype=float) / (1 << J)
154:     phi = 0 * x
155: 
156:     psi = 0 * x
157: 
158:     # find phi0, and phi1
159:     lam, v = eig(m[0, 0])
160:     ind = np.argmin(np.absolute(lam - 1))
161:     # a dictionary with a binary representation of the
162:     #   evaluation points x < 1 -- i.e. position is 0.xxxx
163:     v = np.real(v[:, ind])
164:     # need scaling function to integrate to 1 so find
165:     #  eigenvector normalized to sum(v,axis=0)=1
166:     sm = np.sum(v)
167:     if sm < 0:  # need scaling function to integrate to 1
168:         v = -v
169:         sm = -sm
170:     bitdic = {'0': v / sm}
171:     bitdic['1'] = np.dot(m[0, 1], bitdic['0'])
172:     step = 1 << J
173:     phi[::step] = bitdic['0']
174:     phi[(1 << (J - 1))::step] = bitdic['1']
175:     psi[::step] = np.dot(m[1, 0], bitdic['0'])
176:     psi[(1 << (J - 1))::step] = np.dot(m[1, 1], bitdic['0'])
177:     # descend down the levels inserting more and more values
178:     #  into bitdic -- store the values in the correct location once we
179:     #  have computed them -- stored in the dictionary
180:     #  for quicker use later.
181:     prevkeys = ['1']
182:     for level in range(2, J + 1):
183:         newkeys = ['%d%s' % (xx, yy) for xx in [0, 1] for yy in prevkeys]
184:         fac = 1 << (J - level)
185:         for key in newkeys:
186:             # convert key to number
187:             num = 0
188:             for pos in range(level):
189:                 if key[pos] == '1':
190:                     num += (1 << (level - 1 - pos))
191:             pastphi = bitdic[key[1:]]
192:             ii = int(key[0])
193:             temp = np.dot(m[0, ii], pastphi)
194:             bitdic[key] = temp
195:             phi[num * fac::step] = temp
196:             psi[num * fac::step] = np.dot(m[1, ii], pastphi)
197:         prevkeys = newkeys
198: 
199:     return x, phi, psi
200: 
201: 
202: def morlet(M, w=5.0, s=1.0, complete=True):
203:     '''
204:     Complex Morlet wavelet.
205: 
206:     Parameters
207:     ----------
208:     M : int
209:         Length of the wavelet.
210:     w : float, optional
211:         Omega0. Default is 5
212:     s : float, optional
213:         Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.
214:     complete : bool, optional
215:         Whether to use the complete or the standard version.
216: 
217:     Returns
218:     -------
219:     morlet : (M,) ndarray
220: 
221:     See Also
222:     --------
223:     scipy.signal.gausspulse
224: 
225:     Notes
226:     -----
227:     The standard version::
228: 
229:         pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
230: 
231:     This commonly used wavelet is often referred to simply as the
232:     Morlet wavelet.  Note that this simplified version can cause
233:     admissibility problems at low values of `w`.
234: 
235:     The complete version::
236: 
237:         pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))
238: 
239:     This version has a correction
240:     term to improve admissibility. For `w` greater than 5, the
241:     correction term is negligible.
242: 
243:     Note that the energy of the return wavelet is not normalised
244:     according to `s`.
245: 
246:     The fundamental frequency of this wavelet in Hz is given
247:     by ``f = 2*s*w*r / M`` where `r` is the sampling rate.
248:     
249:     Note: This function was created before `cwt` and is not compatible
250:     with it.
251: 
252:     '''
253:     x = linspace(-s * 2 * pi, s * 2 * pi, M)
254:     output = exp(1j * w * x)
255: 
256:     if complete:
257:         output -= exp(-0.5 * (w**2))
258: 
259:     output *= exp(-0.5 * (x**2)) * pi**(-0.25)
260: 
261:     return output
262: 
263: 
264: def ricker(points, a):
265:     '''
266:     Return a Ricker wavelet, also known as the "Mexican hat wavelet".
267: 
268:     It models the function:
269: 
270:         ``A (1 - x^2/a^2) exp(-x^2/2 a^2)``,
271: 
272:     where ``A = 2/sqrt(3a)pi^1/4``.
273: 
274:     Parameters
275:     ----------
276:     points : int
277:         Number of points in `vector`.
278:         Will be centered around 0.
279:     a : scalar
280:         Width parameter of the wavelet.
281: 
282:     Returns
283:     -------
284:     vector : (N,) ndarray
285:         Array of length `points` in shape of ricker curve.
286: 
287:     Examples
288:     --------
289:     >>> from scipy import signal
290:     >>> import matplotlib.pyplot as plt
291: 
292:     >>> points = 100
293:     >>> a = 4.0
294:     >>> vec2 = signal.ricker(points, a)
295:     >>> print(len(vec2))
296:     100
297:     >>> plt.plot(vec2)
298:     >>> plt.show()
299: 
300:     '''
301:     A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
302:     wsq = a**2
303:     vec = np.arange(0, points) - (points - 1.0) / 2
304:     xsq = vec**2
305:     mod = (1 - xsq / wsq)
306:     gauss = np.exp(-xsq / (2 * wsq))
307:     total = A * mod * gauss
308:     return total
309: 
310: 
311: def cwt(data, wavelet, widths):
312:     '''
313:     Continuous wavelet transform.
314: 
315:     Performs a continuous wavelet transform on `data`,
316:     using the `wavelet` function. A CWT performs a convolution
317:     with `data` using the `wavelet` function, which is characterized
318:     by a width parameter and length parameter.
319: 
320:     Parameters
321:     ----------
322:     data : (N,) ndarray
323:         data on which to perform the transform.
324:     wavelet : function
325:         Wavelet function, which should take 2 arguments.
326:         The first argument is the number of points that the returned vector
327:         will have (len(wavelet(length,width)) == length).
328:         The second is a width parameter, defining the size of the wavelet
329:         (e.g. standard deviation of a gaussian). See `ricker`, which
330:         satisfies these requirements.
331:     widths : (M,) sequence
332:         Widths to use for transform.
333: 
334:     Returns
335:     -------
336:     cwt: (M, N) ndarray
337:         Will have shape of (len(widths), len(data)).
338: 
339:     Notes
340:     -----
341:     ::
342: 
343:         length = min(10 * width[ii], len(data))
344:         cwt[ii,:] = signal.convolve(data, wavelet(length,
345:                                     width[ii]), mode='same')
346: 
347:     Examples
348:     --------
349:     >>> from scipy import signal
350:     >>> import matplotlib.pyplot as plt
351:     >>> t = np.linspace(-1, 1, 200, endpoint=False)
352:     >>> sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
353:     >>> widths = np.arange(1, 31)
354:     >>> cwtmatr = signal.cwt(sig, signal.ricker, widths)
355:     >>> plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
356:     ...            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
357:     >>> plt.show()
358: 
359:     '''
360:     output = np.zeros([len(widths), len(data)])
361:     for ind, width in enumerate(widths):
362:         wavelet_data = wavelet(min(10 * width, len(data)), width)
363:         output[ind, :] = convolve(data, wavelet_data,
364:                                   mode='same')
365:     return output
366: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_283698 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_283698) is not StypyTypeError):

    if (import_283698 != 'pyd_module'):
        __import__(import_283698)
        sys_modules_283699 = sys.modules[import_283698]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_283699.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_283698)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.dual import eig' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_283700 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.dual')

if (type(import_283700) is not StypyTypeError):

    if (import_283700 != 'pyd_module'):
        __import__(import_283700)
        sys_modules_283701 = sys.modules[import_283700]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.dual', sys_modules_283701.module_type_store, module_type_store, ['eig'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_283701, sys_modules_283701.module_type_store, module_type_store)
    else:
        from numpy.dual import eig

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.dual', None, module_type_store, ['eig'], [eig])

else:
    # Assigning a type to the variable 'numpy.dual' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.dual', import_283700)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.special import comb' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_283702 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special')

if (type(import_283702) is not StypyTypeError):

    if (import_283702 != 'pyd_module'):
        __import__(import_283702)
        sys_modules_283703 = sys.modules[import_283702]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', sys_modules_283703.module_type_store, module_type_store, ['comb'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_283703, sys_modules_283703.module_type_store, module_type_store)
    else:
        from scipy.special import comb

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', None, module_type_store, ['comb'], [comb])

else:
    # Assigning a type to the variable 'scipy.special' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', import_283702)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy import linspace, pi, exp' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_283704 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy')

if (type(import_283704) is not StypyTypeError):

    if (import_283704 != 'pyd_module'):
        __import__(import_283704)
        sys_modules_283705 = sys.modules[import_283704]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy', sys_modules_283705.module_type_store, module_type_store, ['linspace', 'pi', 'exp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_283705, sys_modules_283705.module_type_store, module_type_store)
    else:
        from scipy import linspace, pi, exp

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy', None, module_type_store, ['linspace', 'pi', 'exp'], [linspace, pi, exp])

else:
    # Assigning a type to the variable 'scipy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy', import_283704)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.signal import convolve' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_283706 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.signal')

if (type(import_283706) is not StypyTypeError):

    if (import_283706 != 'pyd_module'):
        __import__(import_283706)
        sys_modules_283707 = sys.modules[import_283706]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.signal', sys_modules_283707.module_type_store, module_type_store, ['convolve'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_283707, sys_modules_283707.module_type_store, module_type_store)
    else:
        from scipy.signal import convolve

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.signal', None, module_type_store, ['convolve'], [convolve])

else:
    # Assigning a type to the variable 'scipy.signal' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.signal', import_283706)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a List to a Name (line 9):

# Assigning a List to a Name (line 9):
__all__ = ['daub', 'qmf', 'cascade', 'morlet', 'ricker', 'cwt']
module_type_store.set_exportable_members(['daub', 'qmf', 'cascade', 'morlet', 'ricker', 'cwt'])

# Obtaining an instance of the builtin type 'list' (line 9)
list_283708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
str_283709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'daub')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_283708, str_283709)
# Adding element type (line 9)
str_283710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'str', 'qmf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_283708, str_283710)
# Adding element type (line 9)
str_283711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 26), 'str', 'cascade')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_283708, str_283711)
# Adding element type (line 9)
str_283712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 37), 'str', 'morlet')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_283708, str_283712)
# Adding element type (line 9)
str_283713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 47), 'str', 'ricker')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_283708, str_283713)
# Adding element type (line 9)
str_283714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 57), 'str', 'cwt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_283708, str_283714)

# Assigning a type to the variable '__all__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__all__', list_283708)

@norecursion
def daub(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'daub'
    module_type_store = module_type_store.open_function_context('daub', 12, 0, False)
    
    # Passed parameters checking function
    daub.stypy_localization = localization
    daub.stypy_type_of_self = None
    daub.stypy_type_store = module_type_store
    daub.stypy_function_name = 'daub'
    daub.stypy_param_names_list = ['p']
    daub.stypy_varargs_param_name = None
    daub.stypy_kwargs_param_name = None
    daub.stypy_call_defaults = defaults
    daub.stypy_call_varargs = varargs
    daub.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'daub', ['p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'daub', localization, ['p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'daub(...)' code ##################

    str_283715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'str', '\n    The coefficients for the FIR low-pass filter producing Daubechies wavelets.\n\n    p>=1 gives the order of the zero at f=1/2.\n    There are 2p filter coefficients.\n\n    Parameters\n    ----------\n    p : int\n        Order of the zero at f=1/2, can have values from 1 to 34.\n\n    Returns\n    -------\n    daub : ndarray\n        Return\n\n    ')
    
    # Assigning a Attribute to a Name (line 30):
    
    # Assigning a Attribute to a Name (line 30):
    # Getting the type of 'np' (line 30)
    np_283716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'np')
    # Obtaining the member 'sqrt' of a type (line 30)
    sqrt_283717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 11), np_283716, 'sqrt')
    # Assigning a type to the variable 'sqrt' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'sqrt', sqrt_283717)
    
    
    # Getting the type of 'p' (line 31)
    p_283718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'p')
    int_283719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 11), 'int')
    # Applying the binary operator '<' (line 31)
    result_lt_283720 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 7), '<', p_283718, int_283719)
    
    # Testing the type of an if condition (line 31)
    if_condition_283721 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 4), result_lt_283720)
    # Assigning a type to the variable 'if_condition_283721' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'if_condition_283721', if_condition_283721)
    # SSA begins for if statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 32)
    # Processing the call arguments (line 32)
    str_283723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'str', 'p must be at least 1.')
    # Processing the call keyword arguments (line 32)
    kwargs_283724 = {}
    # Getting the type of 'ValueError' (line 32)
    ValueError_283722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 32)
    ValueError_call_result_283725 = invoke(stypy.reporting.localization.Localization(__file__, 32, 14), ValueError_283722, *[str_283723], **kwargs_283724)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 32, 8), ValueError_call_result_283725, 'raise parameter', BaseException)
    # SSA join for if statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'p' (line 33)
    p_283726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'p')
    int_283727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 12), 'int')
    # Applying the binary operator '==' (line 33)
    result_eq_283728 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 7), '==', p_283726, int_283727)
    
    # Testing the type of an if condition (line 33)
    if_condition_283729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 4), result_eq_283728)
    # Assigning a type to the variable 'if_condition_283729' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'if_condition_283729', if_condition_283729)
    # SSA begins for if statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 34):
    
    # Assigning a BinOp to a Name (line 34):
    int_283730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'int')
    
    # Call to sqrt(...): (line 34)
    # Processing the call arguments (line 34)
    int_283732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'int')
    # Processing the call keyword arguments (line 34)
    kwargs_283733 = {}
    # Getting the type of 'sqrt' (line 34)
    sqrt_283731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 34)
    sqrt_call_result_283734 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), sqrt_283731, *[int_283732], **kwargs_283733)
    
    # Applying the binary operator 'div' (line 34)
    result_div_283735 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 12), 'div', int_283730, sqrt_call_result_283734)
    
    # Assigning a type to the variable 'c' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'c', result_div_283735)
    
    # Call to array(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_283738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    # Getting the type of 'c' (line 35)
    c_283739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 24), list_283738, c_283739)
    # Adding element type (line 35)
    # Getting the type of 'c' (line 35)
    c_283740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 28), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 24), list_283738, c_283740)
    
    # Processing the call keyword arguments (line 35)
    kwargs_283741 = {}
    # Getting the type of 'np' (line 35)
    np_283736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 35)
    array_283737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), np_283736, 'array')
    # Calling array(args, kwargs) (line 35)
    array_call_result_283742 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), array_283737, *[list_283738], **kwargs_283741)
    
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', array_call_result_283742)
    # SSA branch for the else part of an if statement (line 33)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'p' (line 36)
    p_283743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'p')
    int_283744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'int')
    # Applying the binary operator '==' (line 36)
    result_eq_283745 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 9), '==', p_283743, int_283744)
    
    # Testing the type of an if condition (line 36)
    if_condition_283746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 9), result_eq_283745)
    # Assigning a type to the variable 'if_condition_283746' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'if_condition_283746', if_condition_283746)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 37):
    
    # Assigning a BinOp to a Name (line 37):
    
    # Call to sqrt(...): (line 37)
    # Processing the call arguments (line 37)
    int_283748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 17), 'int')
    # Processing the call keyword arguments (line 37)
    kwargs_283749 = {}
    # Getting the type of 'sqrt' (line 37)
    sqrt_283747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 37)
    sqrt_call_result_283750 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), sqrt_283747, *[int_283748], **kwargs_283749)
    
    int_283751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 22), 'int')
    # Applying the binary operator 'div' (line 37)
    result_div_283752 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 12), 'div', sqrt_call_result_283750, int_283751)
    
    # Assigning a type to the variable 'f' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'f', result_div_283752)
    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to sqrt(...): (line 38)
    # Processing the call arguments (line 38)
    int_283754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'int')
    # Processing the call keyword arguments (line 38)
    kwargs_283755 = {}
    # Getting the type of 'sqrt' (line 38)
    sqrt_283753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 38)
    sqrt_call_result_283756 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), sqrt_283753, *[int_283754], **kwargs_283755)
    
    # Assigning a type to the variable 'c' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'c', sqrt_call_result_283756)
    # Getting the type of 'f' (line 39)
    f_283757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'f')
    
    # Call to array(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_283760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    int_283761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 29), 'int')
    # Getting the type of 'c' (line 39)
    c_283762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), 'c', False)
    # Applying the binary operator '+' (line 39)
    result_add_283763 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 29), '+', int_283761, c_283762)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 28), list_283760, result_add_283763)
    # Adding element type (line 39)
    int_283764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'int')
    # Getting the type of 'c' (line 39)
    c_283765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 40), 'c', False)
    # Applying the binary operator '+' (line 39)
    result_add_283766 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 36), '+', int_283764, c_283765)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 28), list_283760, result_add_283766)
    # Adding element type (line 39)
    int_283767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 43), 'int')
    # Getting the type of 'c' (line 39)
    c_283768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 47), 'c', False)
    # Applying the binary operator '-' (line 39)
    result_sub_283769 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 43), '-', int_283767, c_283768)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 28), list_283760, result_sub_283769)
    # Adding element type (line 39)
    int_283770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 50), 'int')
    # Getting the type of 'c' (line 39)
    c_283771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 54), 'c', False)
    # Applying the binary operator '-' (line 39)
    result_sub_283772 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 50), '-', int_283770, c_283771)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 28), list_283760, result_sub_283772)
    
    # Processing the call keyword arguments (line 39)
    kwargs_283773 = {}
    # Getting the type of 'np' (line 39)
    np_283758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'np', False)
    # Obtaining the member 'array' of a type (line 39)
    array_283759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 19), np_283758, 'array')
    # Calling array(args, kwargs) (line 39)
    array_call_result_283774 = invoke(stypy.reporting.localization.Localization(__file__, 39, 19), array_283759, *[list_283760], **kwargs_283773)
    
    # Applying the binary operator '*' (line 39)
    result_mul_283775 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), '*', f_283757, array_call_result_283774)
    
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', result_mul_283775)
    # SSA branch for the else part of an if statement (line 36)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'p' (line 40)
    p_283776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 9), 'p')
    int_283777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 14), 'int')
    # Applying the binary operator '==' (line 40)
    result_eq_283778 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 9), '==', p_283776, int_283777)
    
    # Testing the type of an if condition (line 40)
    if_condition_283779 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 9), result_eq_283778)
    # Assigning a type to the variable 'if_condition_283779' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 9), 'if_condition_283779', if_condition_283779)
    # SSA begins for if statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 41):
    
    # Assigning a BinOp to a Name (line 41):
    int_283780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'int')
    
    # Call to sqrt(...): (line 41)
    # Processing the call arguments (line 41)
    int_283782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_283783 = {}
    # Getting the type of 'sqrt' (line 41)
    sqrt_283781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 41)
    sqrt_call_result_283784 = invoke(stypy.reporting.localization.Localization(__file__, 41, 19), sqrt_283781, *[int_283782], **kwargs_283783)
    
    # Applying the binary operator '*' (line 41)
    result_mul_283785 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 14), '*', int_283780, sqrt_call_result_283784)
    
    # Assigning a type to the variable 'tmp' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'tmp', result_mul_283785)
    
    # Assigning a BinOp to a Name (line 42):
    
    # Assigning a BinOp to a Name (line 42):
    float_283786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 13), 'float')
    
    # Call to sqrt(...): (line 42)
    # Processing the call arguments (line 42)
    int_283788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'int')
    # Getting the type of 'tmp' (line 42)
    tmp_283789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 29), 'tmp', False)
    # Applying the binary operator '+' (line 42)
    result_add_283790 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 24), '+', int_283788, tmp_283789)
    
    # Processing the call keyword arguments (line 42)
    kwargs_283791 = {}
    # Getting the type of 'sqrt' (line 42)
    sqrt_283787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 42)
    sqrt_call_result_283792 = invoke(stypy.reporting.localization.Localization(__file__, 42, 19), sqrt_283787, *[result_add_283790], **kwargs_283791)
    
    int_283793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 36), 'int')
    # Applying the binary operator 'div' (line 42)
    result_div_283794 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 19), 'div', sqrt_call_result_283792, int_283793)
    
    # Applying the binary operator '+' (line 42)
    result_add_283795 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 13), '+', float_283786, result_div_283794)
    
    complex_283796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 40), 'complex')
    
    # Call to sqrt(...): (line 42)
    # Processing the call arguments (line 42)
    int_283798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 51), 'int')
    # Processing the call keyword arguments (line 42)
    kwargs_283799 = {}
    # Getting the type of 'sqrt' (line 42)
    sqrt_283797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 46), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 42)
    sqrt_call_result_283800 = invoke(stypy.reporting.localization.Localization(__file__, 42, 46), sqrt_283797, *[int_283798], **kwargs_283799)
    
    
    # Call to sqrt(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'tmp' (line 42)
    tmp_283802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 62), 'tmp', False)
    int_283803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 68), 'int')
    # Applying the binary operator '-' (line 42)
    result_sub_283804 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 62), '-', tmp_283802, int_283803)
    
    # Processing the call keyword arguments (line 42)
    kwargs_283805 = {}
    # Getting the type of 'sqrt' (line 42)
    sqrt_283801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 57), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 42)
    sqrt_call_result_283806 = invoke(stypy.reporting.localization.Localization(__file__, 42, 57), sqrt_283801, *[result_sub_283804], **kwargs_283805)
    
    # Applying the binary operator '+' (line 42)
    result_add_283807 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 46), '+', sqrt_call_result_283800, sqrt_call_result_283806)
    
    # Applying the binary operator '*' (line 42)
    result_mul_283808 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 40), '*', complex_283796, result_add_283807)
    
    int_283809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 75), 'int')
    # Applying the binary operator 'div' (line 42)
    result_div_283810 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 73), 'div', result_mul_283808, int_283809)
    
    # Applying the binary operator '-' (line 42)
    result_sub_283811 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 38), '-', result_add_283795, result_div_283810)
    
    # Assigning a type to the variable 'z1' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'z1', result_sub_283811)
    
    # Assigning a Call to a Name (line 43):
    
    # Assigning a Call to a Name (line 43):
    
    # Call to conj(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'z1' (line 43)
    z1_283814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'z1', False)
    # Processing the call keyword arguments (line 43)
    kwargs_283815 = {}
    # Getting the type of 'np' (line 43)
    np_283812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'np', False)
    # Obtaining the member 'conj' of a type (line 43)
    conj_283813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 14), np_283812, 'conj')
    # Calling conj(args, kwargs) (line 43)
    conj_call_result_283816 = invoke(stypy.reporting.localization.Localization(__file__, 43, 14), conj_283813, *[z1_283814], **kwargs_283815)
    
    # Assigning a type to the variable 'z1c' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'z1c', conj_call_result_283816)
    
    # Assigning a BinOp to a Name (line 44):
    
    # Assigning a BinOp to a Name (line 44):
    
    # Call to sqrt(...): (line 44)
    # Processing the call arguments (line 44)
    int_283818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 17), 'int')
    # Processing the call keyword arguments (line 44)
    kwargs_283819 = {}
    # Getting the type of 'sqrt' (line 44)
    sqrt_283817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 44)
    sqrt_call_result_283820 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), sqrt_283817, *[int_283818], **kwargs_283819)
    
    int_283821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'int')
    # Applying the binary operator 'div' (line 44)
    result_div_283822 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 12), 'div', sqrt_call_result_283820, int_283821)
    
    # Assigning a type to the variable 'f' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'f', result_div_283822)
    
    # Assigning a Call to a Name (line 45):
    
    # Assigning a Call to a Name (line 45):
    
    # Call to real(...): (line 45)
    # Processing the call arguments (line 45)
    int_283825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 22), 'int')
    # Getting the type of 'z1' (line 45)
    z1_283826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'z1', False)
    # Applying the binary operator '-' (line 45)
    result_sub_283827 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 22), '-', int_283825, z1_283826)
    
    int_283828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 33), 'int')
    # Getting the type of 'z1c' (line 45)
    z1c_283829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 37), 'z1c', False)
    # Applying the binary operator '-' (line 45)
    result_sub_283830 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 33), '-', int_283828, z1c_283829)
    
    # Applying the binary operator '*' (line 45)
    result_mul_283831 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 21), '*', result_sub_283827, result_sub_283830)
    
    # Processing the call keyword arguments (line 45)
    kwargs_283832 = {}
    # Getting the type of 'np' (line 45)
    np_283823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'np', False)
    # Obtaining the member 'real' of a type (line 45)
    real_283824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 13), np_283823, 'real')
    # Calling real(args, kwargs) (line 45)
    real_call_result_283833 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), real_283824, *[result_mul_283831], **kwargs_283832)
    
    # Assigning a type to the variable 'd0' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'd0', real_call_result_283833)
    
    # Assigning a Call to a Name (line 46):
    
    # Assigning a Call to a Name (line 46):
    
    # Call to real(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'z1' (line 46)
    z1_283836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'z1', False)
    # Getting the type of 'z1c' (line 46)
    z1c_283837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'z1c', False)
    # Applying the binary operator '*' (line 46)
    result_mul_283838 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 21), '*', z1_283836, z1c_283837)
    
    # Processing the call keyword arguments (line 46)
    kwargs_283839 = {}
    # Getting the type of 'np' (line 46)
    np_283834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'np', False)
    # Obtaining the member 'real' of a type (line 46)
    real_283835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 13), np_283834, 'real')
    # Calling real(args, kwargs) (line 46)
    real_call_result_283840 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), real_283835, *[result_mul_283838], **kwargs_283839)
    
    # Assigning a type to the variable 'a0' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'a0', real_call_result_283840)
    
    # Assigning a BinOp to a Name (line 47):
    
    # Assigning a BinOp to a Name (line 47):
    int_283841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 13), 'int')
    
    # Call to real(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'z1' (line 47)
    z1_283844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'z1', False)
    # Processing the call keyword arguments (line 47)
    kwargs_283845 = {}
    # Getting the type of 'np' (line 47)
    np_283842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'np', False)
    # Obtaining the member 'real' of a type (line 47)
    real_283843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 17), np_283842, 'real')
    # Calling real(args, kwargs) (line 47)
    real_call_result_283846 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), real_283843, *[z1_283844], **kwargs_283845)
    
    # Applying the binary operator '*' (line 47)
    result_mul_283847 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 13), '*', int_283841, real_call_result_283846)
    
    # Assigning a type to the variable 'a1' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'a1', result_mul_283847)
    # Getting the type of 'f' (line 48)
    f_283848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'f')
    # Getting the type of 'd0' (line 48)
    d0_283849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'd0')
    # Applying the binary operator 'div' (line 48)
    result_div_283850 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 15), 'div', f_283848, d0_283849)
    
    
    # Call to array(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_283853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    # Adding element type (line 48)
    # Getting the type of 'a0' (line 48)
    a0_283854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'a0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 33), list_283853, a0_283854)
    # Adding element type (line 48)
    int_283855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 38), 'int')
    # Getting the type of 'a0' (line 48)
    a0_283856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 42), 'a0', False)
    # Applying the binary operator '*' (line 48)
    result_mul_283857 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 38), '*', int_283855, a0_283856)
    
    # Getting the type of 'a1' (line 48)
    a1_283858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 47), 'a1', False)
    # Applying the binary operator '-' (line 48)
    result_sub_283859 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 38), '-', result_mul_283857, a1_283858)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 33), list_283853, result_sub_283859)
    # Adding element type (line 48)
    int_283860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 51), 'int')
    # Getting the type of 'a0' (line 48)
    a0_283861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 55), 'a0', False)
    # Applying the binary operator '*' (line 48)
    result_mul_283862 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 51), '*', int_283860, a0_283861)
    
    int_283863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 60), 'int')
    # Getting the type of 'a1' (line 48)
    a1_283864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 64), 'a1', False)
    # Applying the binary operator '*' (line 48)
    result_mul_283865 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 60), '*', int_283863, a1_283864)
    
    # Applying the binary operator '-' (line 48)
    result_sub_283866 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 51), '-', result_mul_283862, result_mul_283865)
    
    int_283867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 69), 'int')
    # Applying the binary operator '+' (line 48)
    result_add_283868 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 67), '+', result_sub_283866, int_283867)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 33), list_283853, result_add_283868)
    # Adding element type (line 48)
    # Getting the type of 'a0' (line 49)
    a0_283869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'a0', False)
    int_283870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 39), 'int')
    # Getting the type of 'a1' (line 49)
    a1_283871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 43), 'a1', False)
    # Applying the binary operator '*' (line 49)
    result_mul_283872 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 39), '*', int_283870, a1_283871)
    
    # Applying the binary operator '-' (line 49)
    result_sub_283873 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 34), '-', a0_283869, result_mul_283872)
    
    int_283874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 48), 'int')
    # Applying the binary operator '+' (line 49)
    result_add_283875 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 46), '+', result_sub_283873, int_283874)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 33), list_283853, result_add_283875)
    # Adding element type (line 48)
    int_283876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 51), 'int')
    # Getting the type of 'a1' (line 49)
    a1_283877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 55), 'a1', False)
    # Applying the binary operator '-' (line 49)
    result_sub_283878 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 51), '-', int_283876, a1_283877)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 33), list_283853, result_sub_283878)
    # Adding element type (line 48)
    int_283879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 33), list_283853, int_283879)
    
    # Processing the call keyword arguments (line 48)
    kwargs_283880 = {}
    # Getting the type of 'np' (line 48)
    np_283851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'np', False)
    # Obtaining the member 'array' of a type (line 48)
    array_283852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 24), np_283851, 'array')
    # Calling array(args, kwargs) (line 48)
    array_call_result_283881 = invoke(stypy.reporting.localization.Localization(__file__, 48, 24), array_283852, *[list_283853], **kwargs_283880)
    
    # Applying the binary operator '*' (line 48)
    result_mul_283882 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 22), '*', result_div_283850, array_call_result_283881)
    
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', result_mul_283882)
    # SSA branch for the else part of an if statement (line 40)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'p' (line 50)
    p_283883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 9), 'p')
    int_283884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 13), 'int')
    # Applying the binary operator '<' (line 50)
    result_lt_283885 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 9), '<', p_283883, int_283884)
    
    # Testing the type of an if condition (line 50)
    if_condition_283886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 9), result_lt_283885)
    # Assigning a type to the variable 'if_condition_283886' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 9), 'if_condition_283886', if_condition_283886)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'p' (line 52)
    p_283887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'p')
    int_283888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 15), 'int')
    # Applying the binary operator '<' (line 52)
    result_lt_283889 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 11), '<', p_283887, int_283888)
    
    # Testing the type of an if condition (line 52)
    if_condition_283890 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 8), result_lt_283889)
    # Assigning a type to the variable 'if_condition_283890' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'if_condition_283890', if_condition_283890)
    # SSA begins for if statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 53):
    
    # Assigning a Subscript to a Name (line 53):
    
    # Obtaining the type of the subscript
    int_283891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 66), 'int')
    slice_283892 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 53, 17), None, None, int_283891)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'p' (line 53)
    p_283905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 60), 'p', False)
    # Processing the call keyword arguments (line 53)
    kwargs_283906 = {}
    # Getting the type of 'range' (line 53)
    range_283904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 54), 'range', False)
    # Calling range(args, kwargs) (line 53)
    range_call_result_283907 = invoke(stypy.reporting.localization.Localization(__file__, 53, 54), range_283904, *[p_283905], **kwargs_283906)
    
    comprehension_283908 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), range_call_result_283907)
    # Assigning a type to the variable 'k' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'k', comprehension_283908)
    
    # Call to comb(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'p' (line 53)
    p_283894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'p', False)
    int_283895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 26), 'int')
    # Applying the binary operator '-' (line 53)
    result_sub_283896 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 22), '-', p_283894, int_283895)
    
    # Getting the type of 'k' (line 53)
    k_283897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'k', False)
    # Applying the binary operator '+' (line 53)
    result_add_283898 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 28), '+', result_sub_283896, k_283897)
    
    # Getting the type of 'k' (line 53)
    k_283899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'k', False)
    # Processing the call keyword arguments (line 53)
    int_283900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 42), 'int')
    keyword_283901 = int_283900
    kwargs_283902 = {'exact': keyword_283901}
    # Getting the type of 'comb' (line 53)
    comb_283893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'comb', False)
    # Calling comb(args, kwargs) (line 53)
    comb_call_result_283903 = invoke(stypy.reporting.localization.Localization(__file__, 53, 17), comb_283893, *[result_add_283898, k_283899], **kwargs_283902)
    
    list_283909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), list_283909, comb_call_result_283903)
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___283910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 17), list_283909, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_283911 = invoke(stypy.reporting.localization.Localization(__file__, 53, 17), getitem___283910, slice_283892)
    
    # Assigning a type to the variable 'P' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'P', subscript_call_result_283911)
    
    # Assigning a Call to a Name (line 54):
    
    # Assigning a Call to a Name (line 54):
    
    # Call to roots(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'P' (line 54)
    P_283914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 26), 'P', False)
    # Processing the call keyword arguments (line 54)
    kwargs_283915 = {}
    # Getting the type of 'np' (line 54)
    np_283912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'np', False)
    # Obtaining the member 'roots' of a type (line 54)
    roots_283913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 17), np_283912, 'roots')
    # Calling roots(args, kwargs) (line 54)
    roots_call_result_283916 = invoke(stypy.reporting.localization.Localization(__file__, 54, 17), roots_283913, *[P_283914], **kwargs_283915)
    
    # Assigning a type to the variable 'yj' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'yj', roots_call_result_283916)
    # SSA branch for the else part of an if statement (line 52)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 56):
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_283917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 38), 'int')
    slice_283918 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 17), None, None, int_283917)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'p' (line 57)
    p_283935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 32), 'p', False)
    # Processing the call keyword arguments (line 57)
    kwargs_283936 = {}
    # Getting the type of 'range' (line 57)
    range_283934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'range', False)
    # Calling range(args, kwargs) (line 57)
    range_call_result_283937 = invoke(stypy.reporting.localization.Localization(__file__, 57, 26), range_283934, *[p_283935], **kwargs_283936)
    
    comprehension_283938 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 17), range_call_result_283937)
    # Assigning a type to the variable 'k' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'k', comprehension_283938)
    
    # Call to comb(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'p' (line 56)
    p_283920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'p', False)
    int_283921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 26), 'int')
    # Applying the binary operator '-' (line 56)
    result_sub_283922 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 22), '-', p_283920, int_283921)
    
    # Getting the type of 'k' (line 56)
    k_283923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'k', False)
    # Applying the binary operator '+' (line 56)
    result_add_283924 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 28), '+', result_sub_283922, k_283923)
    
    # Getting the type of 'k' (line 56)
    k_283925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'k', False)
    # Processing the call keyword arguments (line 56)
    int_283926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 42), 'int')
    keyword_283927 = int_283926
    kwargs_283928 = {'exact': keyword_283927}
    # Getting the type of 'comb' (line 56)
    comb_283919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'comb', False)
    # Calling comb(args, kwargs) (line 56)
    comb_call_result_283929 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), comb_283919, *[result_add_283924, k_283925], **kwargs_283928)
    
    float_283930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 47), 'float')
    # Getting the type of 'k' (line 56)
    k_283931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 52), 'k')
    # Applying the binary operator '**' (line 56)
    result_pow_283932 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 47), '**', float_283930, k_283931)
    
    # Applying the binary operator 'div' (line 56)
    result_div_283933 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 17), 'div', comb_call_result_283929, result_pow_283932)
    
    list_283939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 17), list_283939, result_div_283933)
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___283940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), list_283939, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_283941 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), getitem___283940, slice_283918)
    
    # Assigning a type to the variable 'P' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'P', subscript_call_result_283941)
    
    # Assigning a BinOp to a Name (line 58):
    
    # Assigning a BinOp to a Name (line 58):
    
    # Call to roots(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'P' (line 58)
    P_283944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'P', False)
    # Processing the call keyword arguments (line 58)
    kwargs_283945 = {}
    # Getting the type of 'np' (line 58)
    np_283942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'np', False)
    # Obtaining the member 'roots' of a type (line 58)
    roots_283943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 17), np_283942, 'roots')
    # Calling roots(args, kwargs) (line 58)
    roots_call_result_283946 = invoke(stypy.reporting.localization.Localization(__file__, 58, 17), roots_283943, *[P_283944], **kwargs_283945)
    
    int_283947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'int')
    # Applying the binary operator 'div' (line 58)
    result_div_283948 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 17), 'div', roots_call_result_283946, int_283947)
    
    # Assigning a type to the variable 'yj' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'yj', result_div_283948)
    # SSA join for if statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 61):
    
    # Assigning a BinOp to a Name (line 61):
    
    # Call to poly1d(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Obtaining an instance of the builtin type 'list' (line 61)
    list_283951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 61)
    # Adding element type (line 61)
    int_283952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 22), list_283951, int_283952)
    # Adding element type (line 61)
    int_283953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 22), list_283951, int_283953)
    
    # Processing the call keyword arguments (line 61)
    kwargs_283954 = {}
    # Getting the type of 'np' (line 61)
    np_283949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'np', False)
    # Obtaining the member 'poly1d' of a type (line 61)
    poly1d_283950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), np_283949, 'poly1d')
    # Calling poly1d(args, kwargs) (line 61)
    poly1d_call_result_283955 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), poly1d_283950, *[list_283951], **kwargs_283954)
    
    # Getting the type of 'p' (line 61)
    p_283956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 31), 'p')
    # Applying the binary operator '**' (line 61)
    result_pow_283957 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 12), '**', poly1d_call_result_283955, p_283956)
    
    # Assigning a type to the variable 'c' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'c', result_pow_283957)
    
    # Assigning a Call to a Name (line 62):
    
    # Assigning a Call to a Name (line 62):
    
    # Call to poly1d(...): (line 62)
    # Processing the call arguments (line 62)
    
    # Obtaining an instance of the builtin type 'list' (line 62)
    list_283960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 62)
    # Adding element type (line 62)
    int_283961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 22), list_283960, int_283961)
    
    # Processing the call keyword arguments (line 62)
    kwargs_283962 = {}
    # Getting the type of 'np' (line 62)
    np_283958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'np', False)
    # Obtaining the member 'poly1d' of a type (line 62)
    poly1d_283959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), np_283958, 'poly1d')
    # Calling poly1d(args, kwargs) (line 62)
    poly1d_call_result_283963 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), poly1d_283959, *[list_283960], **kwargs_283962)
    
    # Assigning a type to the variable 'q' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'q', poly1d_call_result_283963)
    
    
    # Call to range(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'p' (line 63)
    p_283965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'p', False)
    int_283966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'int')
    # Applying the binary operator '-' (line 63)
    result_sub_283967 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 23), '-', p_283965, int_283966)
    
    # Processing the call keyword arguments (line 63)
    kwargs_283968 = {}
    # Getting the type of 'range' (line 63)
    range_283964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'range', False)
    # Calling range(args, kwargs) (line 63)
    range_call_result_283969 = invoke(stypy.reporting.localization.Localization(__file__, 63, 17), range_283964, *[result_sub_283967], **kwargs_283968)
    
    # Testing the type of a for loop iterable (line 63)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 8), range_call_result_283969)
    # Getting the type of the for loop variable (line 63)
    for_loop_var_283970 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 8), range_call_result_283969)
    # Assigning a type to the variable 'k' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'k', for_loop_var_283970)
    # SSA begins for a for statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 64):
    
    # Assigning a Subscript to a Name (line 64):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 64)
    k_283971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'k')
    # Getting the type of 'yj' (line 64)
    yj_283972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'yj')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___283973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), yj_283972, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_283974 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), getitem___283973, k_283971)
    
    # Assigning a type to the variable 'yval' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'yval', subscript_call_result_283974)
    
    # Assigning a BinOp to a Name (line 65):
    
    # Assigning a BinOp to a Name (line 65):
    int_283975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'int')
    
    # Call to sqrt(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'yval' (line 65)
    yval_283977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'yval', False)
    # Getting the type of 'yval' (line 65)
    yval_283978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'yval', False)
    int_283979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 43), 'int')
    # Applying the binary operator '-' (line 65)
    result_sub_283980 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 36), '-', yval_283978, int_283979)
    
    # Applying the binary operator '*' (line 65)
    result_mul_283981 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 28), '*', yval_283977, result_sub_283980)
    
    # Processing the call keyword arguments (line 65)
    kwargs_283982 = {}
    # Getting the type of 'sqrt' (line 65)
    sqrt_283976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 65)
    sqrt_call_result_283983 = invoke(stypy.reporting.localization.Localization(__file__, 65, 23), sqrt_283976, *[result_mul_283981], **kwargs_283982)
    
    # Applying the binary operator '*' (line 65)
    result_mul_283984 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 19), '*', int_283975, sqrt_call_result_283983)
    
    # Assigning a type to the variable 'part' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'part', result_mul_283984)
    
    # Assigning a BinOp to a Name (line 66):
    
    # Assigning a BinOp to a Name (line 66):
    int_283985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'int')
    int_283986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 24), 'int')
    # Getting the type of 'yval' (line 66)
    yval_283987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'yval')
    # Applying the binary operator '*' (line 66)
    result_mul_283988 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 24), '*', int_283986, yval_283987)
    
    # Applying the binary operator '-' (line 66)
    result_sub_283989 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 20), '-', int_283985, result_mul_283988)
    
    # Assigning a type to the variable 'const' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'const', result_sub_283989)
    
    # Assigning a BinOp to a Name (line 67):
    
    # Assigning a BinOp to a Name (line 67):
    # Getting the type of 'const' (line 67)
    const_283990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'const')
    # Getting the type of 'part' (line 67)
    part_283991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'part')
    # Applying the binary operator '+' (line 67)
    result_add_283992 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 17), '+', const_283990, part_283991)
    
    # Assigning a type to the variable 'z1' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'z1', result_add_283992)
    
    
    
    # Call to abs(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'z1' (line 68)
    z1_283994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'z1', False)
    # Processing the call keyword arguments (line 68)
    kwargs_283995 = {}
    # Getting the type of 'abs' (line 68)
    abs_283993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'abs', False)
    # Calling abs(args, kwargs) (line 68)
    abs_call_result_283996 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), abs_283993, *[z1_283994], **kwargs_283995)
    
    int_283997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'int')
    # Applying the binary operator '<' (line 68)
    result_lt_283998 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 15), '<', abs_call_result_283996, int_283997)
    
    # Testing the type of an if condition (line 68)
    if_condition_283999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 12), result_lt_283998)
    # Assigning a type to the variable 'if_condition_283999' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'if_condition_283999', if_condition_283999)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 69):
    
    # Assigning a BinOp to a Name (line 69):
    # Getting the type of 'const' (line 69)
    const_284000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'const')
    # Getting the type of 'part' (line 69)
    part_284001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'part')
    # Applying the binary operator '-' (line 69)
    result_sub_284002 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 21), '-', const_284000, part_284001)
    
    # Assigning a type to the variable 'z1' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'z1', result_sub_284002)
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 70):
    
    # Assigning a BinOp to a Name (line 70):
    # Getting the type of 'q' (line 70)
    q_284003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'q')
    
    # Obtaining an instance of the builtin type 'list' (line 70)
    list_284004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 70)
    # Adding element type (line 70)
    int_284005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 20), list_284004, int_284005)
    # Adding element type (line 70)
    
    # Getting the type of 'z1' (line 70)
    z1_284006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'z1')
    # Applying the 'usub' unary operator (line 70)
    result___neg___284007 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 24), 'usub', z1_284006)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 20), list_284004, result___neg___284007)
    
    # Applying the binary operator '*' (line 70)
    result_mul_284008 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 16), '*', q_284003, list_284004)
    
    # Assigning a type to the variable 'q' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'q', result_mul_284008)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 72):
    
    # Assigning a BinOp to a Name (line 72):
    # Getting the type of 'c' (line 72)
    c_284009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'c')
    
    # Call to real(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'q' (line 72)
    q_284012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'q', False)
    # Processing the call keyword arguments (line 72)
    kwargs_284013 = {}
    # Getting the type of 'np' (line 72)
    np_284010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'np', False)
    # Obtaining the member 'real' of a type (line 72)
    real_284011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), np_284010, 'real')
    # Calling real(args, kwargs) (line 72)
    real_call_result_284014 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), real_284011, *[q_284012], **kwargs_284013)
    
    # Applying the binary operator '*' (line 72)
    result_mul_284015 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 12), '*', c_284009, real_call_result_284014)
    
    # Assigning a type to the variable 'q' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'q', result_mul_284015)
    
    # Assigning a BinOp to a Name (line 74):
    
    # Assigning a BinOp to a Name (line 74):
    # Getting the type of 'q' (line 74)
    q_284016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'q')
    
    # Call to sum(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'q' (line 74)
    q_284019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'q', False)
    # Processing the call keyword arguments (line 74)
    kwargs_284020 = {}
    # Getting the type of 'np' (line 74)
    np_284017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'np', False)
    # Obtaining the member 'sum' of a type (line 74)
    sum_284018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), np_284017, 'sum')
    # Calling sum(args, kwargs) (line 74)
    sum_call_result_284021 = invoke(stypy.reporting.localization.Localization(__file__, 74, 16), sum_284018, *[q_284019], **kwargs_284020)
    
    # Applying the binary operator 'div' (line 74)
    result_div_284022 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 12), 'div', q_284016, sum_call_result_284021)
    
    
    # Call to sqrt(...): (line 74)
    # Processing the call arguments (line 74)
    int_284024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 33), 'int')
    # Processing the call keyword arguments (line 74)
    kwargs_284025 = {}
    # Getting the type of 'sqrt' (line 74)
    sqrt_284023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 74)
    sqrt_call_result_284026 = invoke(stypy.reporting.localization.Localization(__file__, 74, 28), sqrt_284023, *[int_284024], **kwargs_284025)
    
    # Applying the binary operator '*' (line 74)
    result_mul_284027 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 26), '*', result_div_284022, sqrt_call_result_284026)
    
    # Assigning a type to the variable 'q' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'q', result_mul_284027)
    
    # Obtaining the type of the subscript
    int_284028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 21), 'int')
    slice_284029 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 75, 15), None, None, int_284028)
    # Getting the type of 'q' (line 75)
    q_284030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'q')
    # Obtaining the member 'c' of a type (line 75)
    c_284031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), q_284030, 'c')
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___284032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), c_284031, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_284033 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___284032, slice_284029)
    
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type', subscript_call_result_284033)
    # SSA branch for the else part of an if statement (line 50)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 77)
    # Processing the call arguments (line 77)
    str_284035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'str', 'Polynomial factorization does not work well for p too large.')
    # Processing the call keyword arguments (line 77)
    kwargs_284036 = {}
    # Getting the type of 'ValueError' (line 77)
    ValueError_284034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 77)
    ValueError_call_result_284037 = invoke(stypy.reporting.localization.Localization(__file__, 77, 14), ValueError_284034, *[str_284035], **kwargs_284036)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 77, 8), ValueError_call_result_284037, 'raise parameter', BaseException)
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 40)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'daub(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'daub' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_284038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284038)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'daub'
    return stypy_return_type_284038

# Assigning a type to the variable 'daub' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'daub', daub)

@norecursion
def qmf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'qmf'
    module_type_store = module_type_store.open_function_context('qmf', 81, 0, False)
    
    # Passed parameters checking function
    qmf.stypy_localization = localization
    qmf.stypy_type_of_self = None
    qmf.stypy_type_store = module_type_store
    qmf.stypy_function_name = 'qmf'
    qmf.stypy_param_names_list = ['hk']
    qmf.stypy_varargs_param_name = None
    qmf.stypy_kwargs_param_name = None
    qmf.stypy_call_defaults = defaults
    qmf.stypy_call_varargs = varargs
    qmf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'qmf', ['hk'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'qmf', localization, ['hk'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'qmf(...)' code ##################

    str_284039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'str', '\n    Return high-pass qmf filter from low-pass\n\n    Parameters\n    ----------\n    hk : array_like\n        Coefficients of high-pass filter.\n\n    ')
    
    # Assigning a BinOp to a Name (line 91):
    
    # Assigning a BinOp to a Name (line 91):
    
    # Call to len(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'hk' (line 91)
    hk_284041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'hk', False)
    # Processing the call keyword arguments (line 91)
    kwargs_284042 = {}
    # Getting the type of 'len' (line 91)
    len_284040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'len', False)
    # Calling len(args, kwargs) (line 91)
    len_call_result_284043 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), len_284040, *[hk_284041], **kwargs_284042)
    
    int_284044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 18), 'int')
    # Applying the binary operator '-' (line 91)
    result_sub_284045 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 8), '-', len_call_result_284043, int_284044)
    
    # Assigning a type to the variable 'N' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'N', result_sub_284045)
    
    # Assigning a ListComp to a Name (line 92):
    
    # Assigning a ListComp to a Name (line 92):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'N' (line 92)
    N_284057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 48), 'N', False)
    int_284058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 52), 'int')
    # Applying the binary operator '+' (line 92)
    result_add_284059 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 48), '+', N_284057, int_284058)
    
    # Processing the call keyword arguments (line 92)
    kwargs_284060 = {}
    # Getting the type of 'range' (line 92)
    range_284056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'range', False)
    # Calling range(args, kwargs) (line 92)
    range_call_result_284061 = invoke(stypy.reporting.localization.Localization(__file__, 92, 42), range_284056, *[result_add_284059], **kwargs_284060)
    
    comprehension_284062 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 12), range_call_result_284061)
    # Assigning a type to the variable 'k' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'k', comprehension_284062)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 92)
    k_284046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 26), 'k')
    int_284047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 30), 'int')
    # Applying the binary operator '%' (line 92)
    result_mod_284048 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 26), '%', k_284046, int_284047)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 92)
    dict_284049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 92)
    # Adding element type (key, value) (line 92)
    int_284050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 13), 'int')
    int_284051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 16), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 12), dict_284049, (int_284050, int_284051))
    # Adding element type (key, value) (line 92)
    int_284052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'int')
    int_284053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 12), dict_284049, (int_284052, int_284053))
    
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___284054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), dict_284049, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_284055 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), getitem___284054, result_mod_284048)
    
    list_284063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 12), list_284063, subscript_call_result_284055)
    # Assigning a type to the variable 'asgn' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'asgn', list_284063)
    
    # Obtaining the type of the subscript
    int_284064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 16), 'int')
    slice_284065 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 93, 11), None, None, int_284064)
    # Getting the type of 'hk' (line 93)
    hk_284066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'hk')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___284067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 11), hk_284066, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_284068 = invoke(stypy.reporting.localization.Localization(__file__, 93, 11), getitem___284067, slice_284065)
    
    
    # Call to array(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'asgn' (line 93)
    asgn_284071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'asgn', False)
    # Processing the call keyword arguments (line 93)
    kwargs_284072 = {}
    # Getting the type of 'np' (line 93)
    np_284069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'np', False)
    # Obtaining the member 'array' of a type (line 93)
    array_284070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 22), np_284069, 'array')
    # Calling array(args, kwargs) (line 93)
    array_call_result_284073 = invoke(stypy.reporting.localization.Localization(__file__, 93, 22), array_284070, *[asgn_284071], **kwargs_284072)
    
    # Applying the binary operator '*' (line 93)
    result_mul_284074 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 11), '*', subscript_call_result_284068, array_call_result_284073)
    
    # Assigning a type to the variable 'stypy_return_type' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type', result_mul_284074)
    
    # ################# End of 'qmf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'qmf' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_284075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284075)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'qmf'
    return stypy_return_type_284075

# Assigning a type to the variable 'qmf' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'qmf', qmf)

@norecursion
def cascade(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_284076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 18), 'int')
    defaults = [int_284076]
    # Create a new context for function 'cascade'
    module_type_store = module_type_store.open_function_context('cascade', 96, 0, False)
    
    # Passed parameters checking function
    cascade.stypy_localization = localization
    cascade.stypy_type_of_self = None
    cascade.stypy_type_store = module_type_store
    cascade.stypy_function_name = 'cascade'
    cascade.stypy_param_names_list = ['hk', 'J']
    cascade.stypy_varargs_param_name = None
    cascade.stypy_kwargs_param_name = None
    cascade.stypy_call_defaults = defaults
    cascade.stypy_call_varargs = varargs
    cascade.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cascade', ['hk', 'J'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cascade', localization, ['hk', 'J'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cascade(...)' code ##################

    str_284077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, (-1)), 'str', '\n    Return (x, phi, psi) at dyadic points ``K/2**J`` from filter coefficients.\n\n    Parameters\n    ----------\n    hk : array_like\n        Coefficients of low-pass filter.\n    J : int, optional\n        Values will be computed at grid points ``K/2**J``. Default is 7.\n\n    Returns\n    -------\n    x : ndarray\n        The dyadic points ``K/2**J`` for ``K=0...N * (2**J)-1`` where\n        ``len(hk) = len(gk) = N+1``.\n    phi : ndarray\n        The scaling function ``phi(x)`` at `x`:\n        ``phi(x) = sum(hk * phi(2x-k))``, where k is from 0 to N.\n    psi : ndarray, optional\n        The wavelet function ``psi(x)`` at `x`:\n        ``phi(x) = sum(gk * phi(2x-k))``, where k is from 0 to N.\n        `psi` is only returned if `gk` is not None.\n\n    Notes\n    -----\n    The algorithm uses the vector cascade algorithm described by Strang and\n    Nguyen in "Wavelets and Filter Banks".  It builds a dictionary of values\n    and slices for quick reuse.  Then inserts vectors into final vector at the\n    end.\n\n    ')
    
    # Assigning a BinOp to a Name (line 128):
    
    # Assigning a BinOp to a Name (line 128):
    
    # Call to len(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'hk' (line 128)
    hk_284079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'hk', False)
    # Processing the call keyword arguments (line 128)
    kwargs_284080 = {}
    # Getting the type of 'len' (line 128)
    len_284078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'len', False)
    # Calling len(args, kwargs) (line 128)
    len_call_result_284081 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), len_284078, *[hk_284079], **kwargs_284080)
    
    int_284082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 18), 'int')
    # Applying the binary operator '-' (line 128)
    result_sub_284083 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 8), '-', len_call_result_284081, int_284082)
    
    # Assigning a type to the variable 'N' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'N', result_sub_284083)
    
    
    # Getting the type of 'J' (line 130)
    J_284084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'J')
    int_284085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 12), 'int')
    
    # Call to log2(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'N' (line 130)
    N_284088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'N', False)
    int_284089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 29), 'int')
    # Applying the binary operator '+' (line 130)
    result_add_284090 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 25), '+', N_284088, int_284089)
    
    # Processing the call keyword arguments (line 130)
    kwargs_284091 = {}
    # Getting the type of 'np' (line 130)
    np_284086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 17), 'np', False)
    # Obtaining the member 'log2' of a type (line 130)
    log2_284087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 17), np_284086, 'log2')
    # Calling log2(args, kwargs) (line 130)
    log2_call_result_284092 = invoke(stypy.reporting.localization.Localization(__file__, 130, 17), log2_284087, *[result_add_284090], **kwargs_284091)
    
    # Applying the binary operator '-' (line 130)
    result_sub_284093 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 12), '-', int_284085, log2_call_result_284092)
    
    # Applying the binary operator '>' (line 130)
    result_gt_284094 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 8), '>', J_284084, result_sub_284093)
    
    # Testing the type of an if condition (line 130)
    if_condition_284095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), result_gt_284094)
    # Assigning a type to the variable 'if_condition_284095' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_284095', if_condition_284095)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 131)
    # Processing the call arguments (line 131)
    str_284097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 25), 'str', 'Too many levels.')
    # Processing the call keyword arguments (line 131)
    kwargs_284098 = {}
    # Getting the type of 'ValueError' (line 131)
    ValueError_284096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 131)
    ValueError_call_result_284099 = invoke(stypy.reporting.localization.Localization(__file__, 131, 14), ValueError_284096, *[str_284097], **kwargs_284098)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 131, 8), ValueError_call_result_284099, 'raise parameter', BaseException)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'J' (line 132)
    J_284100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'J')
    int_284101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 12), 'int')
    # Applying the binary operator '<' (line 132)
    result_lt_284102 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 8), '<', J_284100, int_284101)
    
    # Testing the type of an if condition (line 132)
    if_condition_284103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 4), result_lt_284102)
    # Assigning a type to the variable 'if_condition_284103' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'if_condition_284103', if_condition_284103)
    # SSA begins for if statement (line 132)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 133)
    # Processing the call arguments (line 133)
    str_284105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'str', 'Too few levels.')
    # Processing the call keyword arguments (line 133)
    kwargs_284106 = {}
    # Getting the type of 'ValueError' (line 133)
    ValueError_284104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 133)
    ValueError_call_result_284107 = invoke(stypy.reporting.localization.Localization(__file__, 133, 14), ValueError_284104, *[str_284105], **kwargs_284106)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 133, 8), ValueError_call_result_284107, 'raise parameter', BaseException)
    # SSA join for if statement (line 132)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Tuple (line 136):
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    int_284108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'N' (line 136)
    N_284109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'N')
    slice_284110 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 136, 13), None, N_284109, None)
    # Getting the type of 'N' (line 136)
    N_284111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'N')
    slice_284112 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 136, 13), None, N_284111, None)
    # Getting the type of 'np' (line 136)
    np_284113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 13), 'np')
    # Obtaining the member 'ogrid' of a type (line 136)
    ogrid_284114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 13), np_284113, 'ogrid')
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___284115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 13), ogrid_284114, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_284116 = invoke(stypy.reporting.localization.Localization(__file__, 136, 13), getitem___284115, (slice_284110, slice_284112))
    
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___284117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 4), subscript_call_result_284116, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_284118 = invoke(stypy.reporting.localization.Localization(__file__, 136, 4), getitem___284117, int_284108)
    
    # Assigning a type to the variable 'tuple_var_assignment_283694' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_283694', subscript_call_result_284118)
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    int_284119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'N' (line 136)
    N_284120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'N')
    slice_284121 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 136, 13), None, N_284120, None)
    # Getting the type of 'N' (line 136)
    N_284122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'N')
    slice_284123 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 136, 13), None, N_284122, None)
    # Getting the type of 'np' (line 136)
    np_284124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 13), 'np')
    # Obtaining the member 'ogrid' of a type (line 136)
    ogrid_284125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 13), np_284124, 'ogrid')
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___284126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 13), ogrid_284125, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_284127 = invoke(stypy.reporting.localization.Localization(__file__, 136, 13), getitem___284126, (slice_284121, slice_284123))
    
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___284128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 4), subscript_call_result_284127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_284129 = invoke(stypy.reporting.localization.Localization(__file__, 136, 4), getitem___284128, int_284119)
    
    # Assigning a type to the variable 'tuple_var_assignment_283695' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_283695', subscript_call_result_284129)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_var_assignment_283694' (line 136)
    tuple_var_assignment_283694_284130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_283694')
    # Assigning a type to the variable 'nn' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'nn', tuple_var_assignment_283694_284130)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_var_assignment_283695' (line 136)
    tuple_var_assignment_283695_284131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_var_assignment_283695')
    # Assigning a type to the variable 'kk' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'kk', tuple_var_assignment_283695_284131)
    
    # Assigning a Call to a Name (line 137):
    
    # Assigning a Call to a Name (line 137):
    
    # Call to sqrt(...): (line 137)
    # Processing the call arguments (line 137)
    int_284134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 17), 'int')
    # Processing the call keyword arguments (line 137)
    kwargs_284135 = {}
    # Getting the type of 'np' (line 137)
    np_284132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 9), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 137)
    sqrt_284133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 9), np_284132, 'sqrt')
    # Calling sqrt(args, kwargs) (line 137)
    sqrt_call_result_284136 = invoke(stypy.reporting.localization.Localization(__file__, 137, 9), sqrt_284133, *[int_284134], **kwargs_284135)
    
    # Assigning a type to the variable 's2' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 's2', sqrt_call_result_284136)
    
    # Assigning a Subscript to a Name (line 139):
    
    # Assigning a Subscript to a Name (line 139):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 139)
    tuple_284137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 139)
    # Adding element type (line 139)
    # Getting the type of 'hk' (line 139)
    hk_284138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'hk')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), tuple_284137, hk_284138)
    # Adding element type (line 139)
    int_284139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), tuple_284137, int_284139)
    
    # Getting the type of 'np' (line 139)
    np_284140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 10), 'np')
    # Obtaining the member 'r_' of a type (line 139)
    r__284141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 10), np_284140, 'r_')
    # Obtaining the member '__getitem__' of a type (line 139)
    getitem___284142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 10), r__284141, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 139)
    subscript_call_result_284143 = invoke(stypy.reporting.localization.Localization(__file__, 139, 10), getitem___284142, tuple_284137)
    
    # Assigning a type to the variable 'thk' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'thk', subscript_call_result_284143)
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to qmf(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'hk' (line 140)
    hk_284145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 13), 'hk', False)
    # Processing the call keyword arguments (line 140)
    kwargs_284146 = {}
    # Getting the type of 'qmf' (line 140)
    qmf_284144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 9), 'qmf', False)
    # Calling qmf(args, kwargs) (line 140)
    qmf_call_result_284147 = invoke(stypy.reporting.localization.Localization(__file__, 140, 9), qmf_284144, *[hk_284145], **kwargs_284146)
    
    # Assigning a type to the variable 'gk' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'gk', qmf_call_result_284147)
    
    # Assigning a Subscript to a Name (line 141):
    
    # Assigning a Subscript to a Name (line 141):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 141)
    tuple_284148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 141)
    # Adding element type (line 141)
    # Getting the type of 'gk' (line 141)
    gk_284149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'gk')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 16), tuple_284148, gk_284149)
    # Adding element type (line 141)
    int_284150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 16), tuple_284148, int_284150)
    
    # Getting the type of 'np' (line 141)
    np_284151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 10), 'np')
    # Obtaining the member 'r_' of a type (line 141)
    r__284152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 10), np_284151, 'r_')
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___284153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 10), r__284152, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_284154 = invoke(stypy.reporting.localization.Localization(__file__, 141, 10), getitem___284153, tuple_284148)
    
    # Assigning a type to the variable 'tgk' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'tgk', subscript_call_result_284154)
    
    # Assigning a Call to a Name (line 143):
    
    # Assigning a Call to a Name (line 143):
    
    # Call to clip(...): (line 143)
    # Processing the call arguments (line 143)
    int_284157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'int')
    # Getting the type of 'nn' (line 143)
    nn_284158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'nn', False)
    # Applying the binary operator '*' (line 143)
    result_mul_284159 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 20), '*', int_284157, nn_284158)
    
    # Getting the type of 'kk' (line 143)
    kk_284160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'kk', False)
    # Applying the binary operator '-' (line 143)
    result_sub_284161 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 20), '-', result_mul_284159, kk_284160)
    
    int_284162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 33), 'int')
    # Getting the type of 'N' (line 143)
    N_284163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 37), 'N', False)
    int_284164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 41), 'int')
    # Applying the binary operator '+' (line 143)
    result_add_284165 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 37), '+', N_284163, int_284164)
    
    # Processing the call keyword arguments (line 143)
    kwargs_284166 = {}
    # Getting the type of 'np' (line 143)
    np_284155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'np', False)
    # Obtaining the member 'clip' of a type (line 143)
    clip_284156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), np_284155, 'clip')
    # Calling clip(args, kwargs) (line 143)
    clip_call_result_284167 = invoke(stypy.reporting.localization.Localization(__file__, 143, 12), clip_284156, *[result_sub_284161, int_284162, result_add_284165], **kwargs_284166)
    
    # Assigning a type to the variable 'indx1' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'indx1', clip_call_result_284167)
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 144):
    
    # Call to clip(...): (line 144)
    # Processing the call arguments (line 144)
    int_284170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 20), 'int')
    # Getting the type of 'nn' (line 144)
    nn_284171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'nn', False)
    # Applying the binary operator '*' (line 144)
    result_mul_284172 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 20), '*', int_284170, nn_284171)
    
    # Getting the type of 'kk' (line 144)
    kk_284173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 29), 'kk', False)
    # Applying the binary operator '-' (line 144)
    result_sub_284174 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 20), '-', result_mul_284172, kk_284173)
    
    int_284175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 34), 'int')
    # Applying the binary operator '+' (line 144)
    result_add_284176 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 32), '+', result_sub_284174, int_284175)
    
    int_284177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 37), 'int')
    # Getting the type of 'N' (line 144)
    N_284178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 41), 'N', False)
    int_284179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 45), 'int')
    # Applying the binary operator '+' (line 144)
    result_add_284180 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 41), '+', N_284178, int_284179)
    
    # Processing the call keyword arguments (line 144)
    kwargs_284181 = {}
    # Getting the type of 'np' (line 144)
    np_284168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'np', False)
    # Obtaining the member 'clip' of a type (line 144)
    clip_284169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), np_284168, 'clip')
    # Calling clip(args, kwargs) (line 144)
    clip_call_result_284182 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), clip_284169, *[result_add_284176, int_284177, result_add_284180], **kwargs_284181)
    
    # Assigning a type to the variable 'indx2' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'indx2', clip_call_result_284182)
    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to zeros(...): (line 145)
    # Processing the call arguments (line 145)
    
    # Obtaining an instance of the builtin type 'tuple' (line 145)
    tuple_284185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 145)
    # Adding element type (line 145)
    int_284186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), tuple_284185, int_284186)
    # Adding element type (line 145)
    int_284187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), tuple_284185, int_284187)
    # Adding element type (line 145)
    # Getting the type of 'N' (line 145)
    N_284188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), tuple_284185, N_284188)
    # Adding element type (line 145)
    # Getting the type of 'N' (line 145)
    N_284189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 27), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), tuple_284185, N_284189)
    
    str_284190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 31), 'str', 'd')
    # Processing the call keyword arguments (line 145)
    kwargs_284191 = {}
    # Getting the type of 'np' (line 145)
    np_284183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 145)
    zeros_284184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), np_284183, 'zeros')
    # Calling zeros(args, kwargs) (line 145)
    zeros_call_result_284192 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), zeros_284184, *[tuple_284185, str_284190], **kwargs_284191)
    
    # Assigning a type to the variable 'm' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'm', zeros_call_result_284192)
    
    # Assigning a Call to a Subscript (line 146):
    
    # Assigning a Call to a Subscript (line 146):
    
    # Call to take(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'thk' (line 146)
    thk_284195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 22), 'thk', False)
    # Getting the type of 'indx1' (line 146)
    indx1_284196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'indx1', False)
    int_284197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 34), 'int')
    # Processing the call keyword arguments (line 146)
    kwargs_284198 = {}
    # Getting the type of 'np' (line 146)
    np_284193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 14), 'np', False)
    # Obtaining the member 'take' of a type (line 146)
    take_284194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 14), np_284193, 'take')
    # Calling take(args, kwargs) (line 146)
    take_call_result_284199 = invoke(stypy.reporting.localization.Localization(__file__, 146, 14), take_284194, *[thk_284195, indx1_284196, int_284197], **kwargs_284198)
    
    # Getting the type of 'm' (line 146)
    m_284200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'm')
    
    # Obtaining an instance of the builtin type 'tuple' (line 146)
    tuple_284201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 146)
    # Adding element type (line 146)
    int_284202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 6), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 6), tuple_284201, int_284202)
    # Adding element type (line 146)
    int_284203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 6), tuple_284201, int_284203)
    
    # Storing an element on a container (line 146)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 4), m_284200, (tuple_284201, take_call_result_284199))
    
    # Assigning a Call to a Subscript (line 147):
    
    # Assigning a Call to a Subscript (line 147):
    
    # Call to take(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'thk' (line 147)
    thk_284206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'thk', False)
    # Getting the type of 'indx2' (line 147)
    indx2_284207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'indx2', False)
    int_284208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 34), 'int')
    # Processing the call keyword arguments (line 147)
    kwargs_284209 = {}
    # Getting the type of 'np' (line 147)
    np_284204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'np', False)
    # Obtaining the member 'take' of a type (line 147)
    take_284205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 14), np_284204, 'take')
    # Calling take(args, kwargs) (line 147)
    take_call_result_284210 = invoke(stypy.reporting.localization.Localization(__file__, 147, 14), take_284205, *[thk_284206, indx2_284207, int_284208], **kwargs_284209)
    
    # Getting the type of 'm' (line 147)
    m_284211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'm')
    
    # Obtaining an instance of the builtin type 'tuple' (line 147)
    tuple_284212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 147)
    # Adding element type (line 147)
    int_284213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 6), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 6), tuple_284212, int_284213)
    # Adding element type (line 147)
    int_284214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 6), tuple_284212, int_284214)
    
    # Storing an element on a container (line 147)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 4), m_284211, (tuple_284212, take_call_result_284210))
    
    # Assigning a Call to a Subscript (line 148):
    
    # Assigning a Call to a Subscript (line 148):
    
    # Call to take(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'tgk' (line 148)
    tgk_284217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 'tgk', False)
    # Getting the type of 'indx1' (line 148)
    indx1_284218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'indx1', False)
    int_284219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 34), 'int')
    # Processing the call keyword arguments (line 148)
    kwargs_284220 = {}
    # Getting the type of 'np' (line 148)
    np_284215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 14), 'np', False)
    # Obtaining the member 'take' of a type (line 148)
    take_284216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 14), np_284215, 'take')
    # Calling take(args, kwargs) (line 148)
    take_call_result_284221 = invoke(stypy.reporting.localization.Localization(__file__, 148, 14), take_284216, *[tgk_284217, indx1_284218, int_284219], **kwargs_284220)
    
    # Getting the type of 'm' (line 148)
    m_284222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'm')
    
    # Obtaining an instance of the builtin type 'tuple' (line 148)
    tuple_284223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 148)
    # Adding element type (line 148)
    int_284224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 6), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 6), tuple_284223, int_284224)
    # Adding element type (line 148)
    int_284225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 6), tuple_284223, int_284225)
    
    # Storing an element on a container (line 148)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 4), m_284222, (tuple_284223, take_call_result_284221))
    
    # Assigning a Call to a Subscript (line 149):
    
    # Assigning a Call to a Subscript (line 149):
    
    # Call to take(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'tgk' (line 149)
    tgk_284228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), 'tgk', False)
    # Getting the type of 'indx2' (line 149)
    indx2_284229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'indx2', False)
    int_284230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 34), 'int')
    # Processing the call keyword arguments (line 149)
    kwargs_284231 = {}
    # Getting the type of 'np' (line 149)
    np_284226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 14), 'np', False)
    # Obtaining the member 'take' of a type (line 149)
    take_284227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 14), np_284226, 'take')
    # Calling take(args, kwargs) (line 149)
    take_call_result_284232 = invoke(stypy.reporting.localization.Localization(__file__, 149, 14), take_284227, *[tgk_284228, indx2_284229, int_284230], **kwargs_284231)
    
    # Getting the type of 'm' (line 149)
    m_284233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'm')
    
    # Obtaining an instance of the builtin type 'tuple' (line 149)
    tuple_284234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 149)
    # Adding element type (line 149)
    int_284235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 6), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 6), tuple_284234, int_284235)
    # Adding element type (line 149)
    int_284236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 6), tuple_284234, int_284236)
    
    # Storing an element on a container (line 149)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 4), m_284233, (tuple_284234, take_call_result_284232))
    
    # Getting the type of 'm' (line 150)
    m_284237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'm')
    # Getting the type of 's2' (line 150)
    s2_284238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 9), 's2')
    # Applying the binary operator '*=' (line 150)
    result_imul_284239 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 4), '*=', m_284237, s2_284238)
    # Assigning a type to the variable 'm' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'm', result_imul_284239)
    
    
    # Assigning a BinOp to a Name (line 153):
    
    # Assigning a BinOp to a Name (line 153):
    
    # Call to arange(...): (line 153)
    # Processing the call arguments (line 153)
    int_284242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 18), 'int')
    # Getting the type of 'N' (line 153)
    N_284243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'N', False)
    int_284244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'int')
    # Getting the type of 'J' (line 153)
    J_284245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'J', False)
    # Applying the binary operator '<<' (line 153)
    result_lshift_284246 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 26), '<<', int_284244, J_284245)
    
    # Applying the binary operator '*' (line 153)
    result_mul_284247 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 21), '*', N_284243, result_lshift_284246)
    
    # Processing the call keyword arguments (line 153)
    # Getting the type of 'float' (line 153)
    float_284248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 41), 'float', False)
    keyword_284249 = float_284248
    kwargs_284250 = {'dtype': keyword_284249}
    # Getting the type of 'np' (line 153)
    np_284240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 153)
    arange_284241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), np_284240, 'arange')
    # Calling arange(args, kwargs) (line 153)
    arange_call_result_284251 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), arange_284241, *[int_284242, result_mul_284247], **kwargs_284250)
    
    int_284252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 51), 'int')
    # Getting the type of 'J' (line 153)
    J_284253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 56), 'J')
    # Applying the binary operator '<<' (line 153)
    result_lshift_284254 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 51), '<<', int_284252, J_284253)
    
    # Applying the binary operator 'div' (line 153)
    result_div_284255 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 8), 'div', arange_call_result_284251, result_lshift_284254)
    
    # Assigning a type to the variable 'x' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'x', result_div_284255)
    
    # Assigning a BinOp to a Name (line 154):
    
    # Assigning a BinOp to a Name (line 154):
    int_284256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 10), 'int')
    # Getting the type of 'x' (line 154)
    x_284257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 14), 'x')
    # Applying the binary operator '*' (line 154)
    result_mul_284258 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 10), '*', int_284256, x_284257)
    
    # Assigning a type to the variable 'phi' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'phi', result_mul_284258)
    
    # Assigning a BinOp to a Name (line 156):
    
    # Assigning a BinOp to a Name (line 156):
    int_284259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 10), 'int')
    # Getting the type of 'x' (line 156)
    x_284260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 14), 'x')
    # Applying the binary operator '*' (line 156)
    result_mul_284261 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 10), '*', int_284259, x_284260)
    
    # Assigning a type to the variable 'psi' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'psi', result_mul_284261)
    
    # Assigning a Call to a Tuple (line 159):
    
    # Assigning a Subscript to a Name (line 159):
    
    # Obtaining the type of the subscript
    int_284262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 4), 'int')
    
    # Call to eig(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 159)
    tuple_284264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 159)
    # Adding element type (line 159)
    int_284265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), tuple_284264, int_284265)
    # Adding element type (line 159)
    int_284266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), tuple_284264, int_284266)
    
    # Getting the type of 'm' (line 159)
    m_284267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 17), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___284268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 17), m_284267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_284269 = invoke(stypy.reporting.localization.Localization(__file__, 159, 17), getitem___284268, tuple_284264)
    
    # Processing the call keyword arguments (line 159)
    kwargs_284270 = {}
    # Getting the type of 'eig' (line 159)
    eig_284263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'eig', False)
    # Calling eig(args, kwargs) (line 159)
    eig_call_result_284271 = invoke(stypy.reporting.localization.Localization(__file__, 159, 13), eig_284263, *[subscript_call_result_284269], **kwargs_284270)
    
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___284272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 4), eig_call_result_284271, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_284273 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), getitem___284272, int_284262)
    
    # Assigning a type to the variable 'tuple_var_assignment_283696' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_var_assignment_283696', subscript_call_result_284273)
    
    # Assigning a Subscript to a Name (line 159):
    
    # Obtaining the type of the subscript
    int_284274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 4), 'int')
    
    # Call to eig(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 159)
    tuple_284276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 159)
    # Adding element type (line 159)
    int_284277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), tuple_284276, int_284277)
    # Adding element type (line 159)
    int_284278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), tuple_284276, int_284278)
    
    # Getting the type of 'm' (line 159)
    m_284279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 17), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___284280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 17), m_284279, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_284281 = invoke(stypy.reporting.localization.Localization(__file__, 159, 17), getitem___284280, tuple_284276)
    
    # Processing the call keyword arguments (line 159)
    kwargs_284282 = {}
    # Getting the type of 'eig' (line 159)
    eig_284275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'eig', False)
    # Calling eig(args, kwargs) (line 159)
    eig_call_result_284283 = invoke(stypy.reporting.localization.Localization(__file__, 159, 13), eig_284275, *[subscript_call_result_284281], **kwargs_284282)
    
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___284284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 4), eig_call_result_284283, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_284285 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), getitem___284284, int_284274)
    
    # Assigning a type to the variable 'tuple_var_assignment_283697' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_var_assignment_283697', subscript_call_result_284285)
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'tuple_var_assignment_283696' (line 159)
    tuple_var_assignment_283696_284286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_var_assignment_283696')
    # Assigning a type to the variable 'lam' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'lam', tuple_var_assignment_283696_284286)
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'tuple_var_assignment_283697' (line 159)
    tuple_var_assignment_283697_284287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_var_assignment_283697')
    # Assigning a type to the variable 'v' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 9), 'v', tuple_var_assignment_283697_284287)
    
    # Assigning a Call to a Name (line 160):
    
    # Assigning a Call to a Name (line 160):
    
    # Call to argmin(...): (line 160)
    # Processing the call arguments (line 160)
    
    # Call to absolute(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'lam' (line 160)
    lam_284292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'lam', False)
    int_284293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 38), 'int')
    # Applying the binary operator '-' (line 160)
    result_sub_284294 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 32), '-', lam_284292, int_284293)
    
    # Processing the call keyword arguments (line 160)
    kwargs_284295 = {}
    # Getting the type of 'np' (line 160)
    np_284290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'np', False)
    # Obtaining the member 'absolute' of a type (line 160)
    absolute_284291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), np_284290, 'absolute')
    # Calling absolute(args, kwargs) (line 160)
    absolute_call_result_284296 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), absolute_284291, *[result_sub_284294], **kwargs_284295)
    
    # Processing the call keyword arguments (line 160)
    kwargs_284297 = {}
    # Getting the type of 'np' (line 160)
    np_284288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 10), 'np', False)
    # Obtaining the member 'argmin' of a type (line 160)
    argmin_284289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 10), np_284288, 'argmin')
    # Calling argmin(args, kwargs) (line 160)
    argmin_call_result_284298 = invoke(stypy.reporting.localization.Localization(__file__, 160, 10), argmin_284289, *[absolute_call_result_284296], **kwargs_284297)
    
    # Assigning a type to the variable 'ind' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'ind', argmin_call_result_284298)
    
    # Assigning a Call to a Name (line 163):
    
    # Assigning a Call to a Name (line 163):
    
    # Call to real(...): (line 163)
    # Processing the call arguments (line 163)
    
    # Obtaining the type of the subscript
    slice_284301 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 163, 16), None, None, None)
    # Getting the type of 'ind' (line 163)
    ind_284302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'ind', False)
    # Getting the type of 'v' (line 163)
    v_284303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'v', False)
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___284304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 16), v_284303, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_284305 = invoke(stypy.reporting.localization.Localization(__file__, 163, 16), getitem___284304, (slice_284301, ind_284302))
    
    # Processing the call keyword arguments (line 163)
    kwargs_284306 = {}
    # Getting the type of 'np' (line 163)
    np_284299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'np', False)
    # Obtaining the member 'real' of a type (line 163)
    real_284300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), np_284299, 'real')
    # Calling real(args, kwargs) (line 163)
    real_call_result_284307 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), real_284300, *[subscript_call_result_284305], **kwargs_284306)
    
    # Assigning a type to the variable 'v' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'v', real_call_result_284307)
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to sum(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'v' (line 166)
    v_284310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'v', False)
    # Processing the call keyword arguments (line 166)
    kwargs_284311 = {}
    # Getting the type of 'np' (line 166)
    np_284308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 9), 'np', False)
    # Obtaining the member 'sum' of a type (line 166)
    sum_284309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 9), np_284308, 'sum')
    # Calling sum(args, kwargs) (line 166)
    sum_call_result_284312 = invoke(stypy.reporting.localization.Localization(__file__, 166, 9), sum_284309, *[v_284310], **kwargs_284311)
    
    # Assigning a type to the variable 'sm' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'sm', sum_call_result_284312)
    
    
    # Getting the type of 'sm' (line 167)
    sm_284313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 7), 'sm')
    int_284314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 12), 'int')
    # Applying the binary operator '<' (line 167)
    result_lt_284315 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 7), '<', sm_284313, int_284314)
    
    # Testing the type of an if condition (line 167)
    if_condition_284316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 4), result_lt_284315)
    # Assigning a type to the variable 'if_condition_284316' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'if_condition_284316', if_condition_284316)
    # SSA begins for if statement (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a UnaryOp to a Name (line 168):
    
    # Assigning a UnaryOp to a Name (line 168):
    
    # Getting the type of 'v' (line 168)
    v_284317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'v')
    # Applying the 'usub' unary operator (line 168)
    result___neg___284318 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 12), 'usub', v_284317)
    
    # Assigning a type to the variable 'v' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'v', result___neg___284318)
    
    # Assigning a UnaryOp to a Name (line 169):
    
    # Assigning a UnaryOp to a Name (line 169):
    
    # Getting the type of 'sm' (line 169)
    sm_284319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 14), 'sm')
    # Applying the 'usub' unary operator (line 169)
    result___neg___284320 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 13), 'usub', sm_284319)
    
    # Assigning a type to the variable 'sm' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'sm', result___neg___284320)
    # SSA join for if statement (line 167)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 170):
    
    # Assigning a Dict to a Name (line 170):
    
    # Obtaining an instance of the builtin type 'dict' (line 170)
    dict_284321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 170)
    # Adding element type (key, value) (line 170)
    str_284322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 14), 'str', '0')
    # Getting the type of 'v' (line 170)
    v_284323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'v')
    # Getting the type of 'sm' (line 170)
    sm_284324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'sm')
    # Applying the binary operator 'div' (line 170)
    result_div_284325 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 19), 'div', v_284323, sm_284324)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 13), dict_284321, (str_284322, result_div_284325))
    
    # Assigning a type to the variable 'bitdic' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'bitdic', dict_284321)
    
    # Assigning a Call to a Subscript (line 171):
    
    # Assigning a Call to a Subscript (line 171):
    
    # Call to dot(...): (line 171)
    # Processing the call arguments (line 171)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 171)
    tuple_284328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 171)
    # Adding element type (line 171)
    int_284329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 27), tuple_284328, int_284329)
    # Adding element type (line 171)
    int_284330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 27), tuple_284328, int_284330)
    
    # Getting the type of 'm' (line 171)
    m_284331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 25), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___284332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 25), m_284331, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_284333 = invoke(stypy.reporting.localization.Localization(__file__, 171, 25), getitem___284332, tuple_284328)
    
    
    # Obtaining the type of the subscript
    str_284334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 41), 'str', '0')
    # Getting the type of 'bitdic' (line 171)
    bitdic_284335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 34), 'bitdic', False)
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___284336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 34), bitdic_284335, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_284337 = invoke(stypy.reporting.localization.Localization(__file__, 171, 34), getitem___284336, str_284334)
    
    # Processing the call keyword arguments (line 171)
    kwargs_284338 = {}
    # Getting the type of 'np' (line 171)
    np_284326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 171)
    dot_284327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 18), np_284326, 'dot')
    # Calling dot(args, kwargs) (line 171)
    dot_call_result_284339 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), dot_284327, *[subscript_call_result_284333, subscript_call_result_284337], **kwargs_284338)
    
    # Getting the type of 'bitdic' (line 171)
    bitdic_284340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'bitdic')
    str_284341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 11), 'str', '1')
    # Storing an element on a container (line 171)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 4), bitdic_284340, (str_284341, dot_call_result_284339))
    
    # Assigning a BinOp to a Name (line 172):
    
    # Assigning a BinOp to a Name (line 172):
    int_284342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 11), 'int')
    # Getting the type of 'J' (line 172)
    J_284343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'J')
    # Applying the binary operator '<<' (line 172)
    result_lshift_284344 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 11), '<<', int_284342, J_284343)
    
    # Assigning a type to the variable 'step' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'step', result_lshift_284344)
    
    # Assigning a Subscript to a Subscript (line 173):
    
    # Assigning a Subscript to a Subscript (line 173):
    
    # Obtaining the type of the subscript
    str_284345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 25), 'str', '0')
    # Getting the type of 'bitdic' (line 173)
    bitdic_284346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 18), 'bitdic')
    # Obtaining the member '__getitem__' of a type (line 173)
    getitem___284347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 18), bitdic_284346, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 173)
    subscript_call_result_284348 = invoke(stypy.reporting.localization.Localization(__file__, 173, 18), getitem___284347, str_284345)
    
    # Getting the type of 'phi' (line 173)
    phi_284349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'phi')
    # Getting the type of 'step' (line 173)
    step_284350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 10), 'step')
    slice_284351 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 173, 4), None, None, step_284350)
    # Storing an element on a container (line 173)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 4), phi_284349, (slice_284351, subscript_call_result_284348))
    
    # Assigning a Subscript to a Subscript (line 174):
    
    # Assigning a Subscript to a Subscript (line 174):
    
    # Obtaining the type of the subscript
    str_284352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 39), 'str', '1')
    # Getting the type of 'bitdic' (line 174)
    bitdic_284353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'bitdic')
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___284354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 32), bitdic_284353, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_284355 = invoke(stypy.reporting.localization.Localization(__file__, 174, 32), getitem___284354, str_284352)
    
    # Getting the type of 'phi' (line 174)
    phi_284356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'phi')
    int_284357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 9), 'int')
    # Getting the type of 'J' (line 174)
    J_284358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'J')
    int_284359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 19), 'int')
    # Applying the binary operator '-' (line 174)
    result_sub_284360 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), '-', J_284358, int_284359)
    
    # Applying the binary operator '<<' (line 174)
    result_lshift_284361 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 9), '<<', int_284357, result_sub_284360)
    
    # Getting the type of 'step' (line 174)
    step_284362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'step')
    slice_284363 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 4), result_lshift_284361, None, step_284362)
    # Storing an element on a container (line 174)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 4), phi_284356, (slice_284363, subscript_call_result_284355))
    
    # Assigning a Call to a Subscript (line 175):
    
    # Assigning a Call to a Subscript (line 175):
    
    # Call to dot(...): (line 175)
    # Processing the call arguments (line 175)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 175)
    tuple_284366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 175)
    # Adding element type (line 175)
    int_284367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 27), tuple_284366, int_284367)
    # Adding element type (line 175)
    int_284368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 27), tuple_284366, int_284368)
    
    # Getting the type of 'm' (line 175)
    m_284369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 175)
    getitem___284370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 25), m_284369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 175)
    subscript_call_result_284371 = invoke(stypy.reporting.localization.Localization(__file__, 175, 25), getitem___284370, tuple_284366)
    
    
    # Obtaining the type of the subscript
    str_284372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'str', '0')
    # Getting the type of 'bitdic' (line 175)
    bitdic_284373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 34), 'bitdic', False)
    # Obtaining the member '__getitem__' of a type (line 175)
    getitem___284374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 34), bitdic_284373, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 175)
    subscript_call_result_284375 = invoke(stypy.reporting.localization.Localization(__file__, 175, 34), getitem___284374, str_284372)
    
    # Processing the call keyword arguments (line 175)
    kwargs_284376 = {}
    # Getting the type of 'np' (line 175)
    np_284364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'np', False)
    # Obtaining the member 'dot' of a type (line 175)
    dot_284365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 18), np_284364, 'dot')
    # Calling dot(args, kwargs) (line 175)
    dot_call_result_284377 = invoke(stypy.reporting.localization.Localization(__file__, 175, 18), dot_284365, *[subscript_call_result_284371, subscript_call_result_284375], **kwargs_284376)
    
    # Getting the type of 'psi' (line 175)
    psi_284378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'psi')
    # Getting the type of 'step' (line 175)
    step_284379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 10), 'step')
    slice_284380 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 175, 4), None, None, step_284379)
    # Storing an element on a container (line 175)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), psi_284378, (slice_284380, dot_call_result_284377))
    
    # Assigning a Call to a Subscript (line 176):
    
    # Assigning a Call to a Subscript (line 176):
    
    # Call to dot(...): (line 176)
    # Processing the call arguments (line 176)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 176)
    tuple_284383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 176)
    # Adding element type (line 176)
    int_284384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 41), tuple_284383, int_284384)
    # Adding element type (line 176)
    int_284385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 41), tuple_284383, int_284385)
    
    # Getting the type of 'm' (line 176)
    m_284386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 39), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___284387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 39), m_284386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_284388 = invoke(stypy.reporting.localization.Localization(__file__, 176, 39), getitem___284387, tuple_284383)
    
    
    # Obtaining the type of the subscript
    str_284389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 55), 'str', '0')
    # Getting the type of 'bitdic' (line 176)
    bitdic_284390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 48), 'bitdic', False)
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___284391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 48), bitdic_284390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_284392 = invoke(stypy.reporting.localization.Localization(__file__, 176, 48), getitem___284391, str_284389)
    
    # Processing the call keyword arguments (line 176)
    kwargs_284393 = {}
    # Getting the type of 'np' (line 176)
    np_284381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'np', False)
    # Obtaining the member 'dot' of a type (line 176)
    dot_284382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 32), np_284381, 'dot')
    # Calling dot(args, kwargs) (line 176)
    dot_call_result_284394 = invoke(stypy.reporting.localization.Localization(__file__, 176, 32), dot_284382, *[subscript_call_result_284388, subscript_call_result_284392], **kwargs_284393)
    
    # Getting the type of 'psi' (line 176)
    psi_284395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'psi')
    int_284396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 9), 'int')
    # Getting the type of 'J' (line 176)
    J_284397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'J')
    int_284398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 19), 'int')
    # Applying the binary operator '-' (line 176)
    result_sub_284399 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 15), '-', J_284397, int_284398)
    
    # Applying the binary operator '<<' (line 176)
    result_lshift_284400 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 9), '<<', int_284396, result_sub_284399)
    
    # Getting the type of 'step' (line 176)
    step_284401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'step')
    slice_284402 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 176, 4), result_lshift_284400, None, step_284401)
    # Storing an element on a container (line 176)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 4), psi_284395, (slice_284402, dot_call_result_284394))
    
    # Assigning a List to a Name (line 181):
    
    # Assigning a List to a Name (line 181):
    
    # Obtaining an instance of the builtin type 'list' (line 181)
    list_284403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 181)
    # Adding element type (line 181)
    str_284404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 16), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 15), list_284403, str_284404)
    
    # Assigning a type to the variable 'prevkeys' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'prevkeys', list_284403)
    
    
    # Call to range(...): (line 182)
    # Processing the call arguments (line 182)
    int_284406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 23), 'int')
    # Getting the type of 'J' (line 182)
    J_284407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 26), 'J', False)
    int_284408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 30), 'int')
    # Applying the binary operator '+' (line 182)
    result_add_284409 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 26), '+', J_284407, int_284408)
    
    # Processing the call keyword arguments (line 182)
    kwargs_284410 = {}
    # Getting the type of 'range' (line 182)
    range_284405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'range', False)
    # Calling range(args, kwargs) (line 182)
    range_call_result_284411 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), range_284405, *[int_284406, result_add_284409], **kwargs_284410)
    
    # Testing the type of a for loop iterable (line 182)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 4), range_call_result_284411)
    # Getting the type of the for loop variable (line 182)
    for_loop_var_284412 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 4), range_call_result_284411)
    # Assigning a type to the variable 'level' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'level', for_loop_var_284412)
    # SSA begins for a for statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a ListComp to a Name (line 183):
    
    # Assigning a ListComp to a Name (line 183):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 183)
    list_284418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 183)
    # Adding element type (line 183)
    int_284419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 47), list_284418, int_284419)
    # Adding element type (line 183)
    int_284420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 47), list_284418, int_284420)
    
    comprehension_284421 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 19), list_284418)
    # Assigning a type to the variable 'xx' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'xx', comprehension_284421)
    # Calculating comprehension expression
    # Getting the type of 'prevkeys' (line 183)
    prevkeys_284422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 64), 'prevkeys')
    comprehension_284423 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 19), prevkeys_284422)
    # Assigning a type to the variable 'yy' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'yy', comprehension_284423)
    str_284413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 19), 'str', '%d%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 183)
    tuple_284414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 183)
    # Adding element type (line 183)
    # Getting the type of 'xx' (line 183)
    xx_284415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 29), 'xx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 29), tuple_284414, xx_284415)
    # Adding element type (line 183)
    # Getting the type of 'yy' (line 183)
    yy_284416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 33), 'yy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 29), tuple_284414, yy_284416)
    
    # Applying the binary operator '%' (line 183)
    result_mod_284417 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 19), '%', str_284413, tuple_284414)
    
    list_284424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 19), list_284424, result_mod_284417)
    # Assigning a type to the variable 'newkeys' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'newkeys', list_284424)
    
    # Assigning a BinOp to a Name (line 184):
    
    # Assigning a BinOp to a Name (line 184):
    int_284425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 14), 'int')
    # Getting the type of 'J' (line 184)
    J_284426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'J')
    # Getting the type of 'level' (line 184)
    level_284427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 24), 'level')
    # Applying the binary operator '-' (line 184)
    result_sub_284428 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 20), '-', J_284426, level_284427)
    
    # Applying the binary operator '<<' (line 184)
    result_lshift_284429 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 14), '<<', int_284425, result_sub_284428)
    
    # Assigning a type to the variable 'fac' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'fac', result_lshift_284429)
    
    # Getting the type of 'newkeys' (line 185)
    newkeys_284430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'newkeys')
    # Testing the type of a for loop iterable (line 185)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 185, 8), newkeys_284430)
    # Getting the type of the for loop variable (line 185)
    for_loop_var_284431 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 185, 8), newkeys_284430)
    # Assigning a type to the variable 'key' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'key', for_loop_var_284431)
    # SSA begins for a for statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Num to a Name (line 187):
    
    # Assigning a Num to a Name (line 187):
    int_284432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 18), 'int')
    # Assigning a type to the variable 'num' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'num', int_284432)
    
    
    # Call to range(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'level' (line 188)
    level_284434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 29), 'level', False)
    # Processing the call keyword arguments (line 188)
    kwargs_284435 = {}
    # Getting the type of 'range' (line 188)
    range_284433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'range', False)
    # Calling range(args, kwargs) (line 188)
    range_call_result_284436 = invoke(stypy.reporting.localization.Localization(__file__, 188, 23), range_284433, *[level_284434], **kwargs_284435)
    
    # Testing the type of a for loop iterable (line 188)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 188, 12), range_call_result_284436)
    # Getting the type of the for loop variable (line 188)
    for_loop_var_284437 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 188, 12), range_call_result_284436)
    # Assigning a type to the variable 'pos' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'pos', for_loop_var_284437)
    # SSA begins for a for statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'pos' (line 189)
    pos_284438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 23), 'pos')
    # Getting the type of 'key' (line 189)
    key_284439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 19), 'key')
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___284440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 19), key_284439, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_284441 = invoke(stypy.reporting.localization.Localization(__file__, 189, 19), getitem___284440, pos_284438)
    
    str_284442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 31), 'str', '1')
    # Applying the binary operator '==' (line 189)
    result_eq_284443 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 19), '==', subscript_call_result_284441, str_284442)
    
    # Testing the type of an if condition (line 189)
    if_condition_284444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 16), result_eq_284443)
    # Assigning a type to the variable 'if_condition_284444' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'if_condition_284444', if_condition_284444)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'num' (line 190)
    num_284445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'num')
    int_284446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 28), 'int')
    # Getting the type of 'level' (line 190)
    level_284447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 34), 'level')
    int_284448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 42), 'int')
    # Applying the binary operator '-' (line 190)
    result_sub_284449 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 34), '-', level_284447, int_284448)
    
    # Getting the type of 'pos' (line 190)
    pos_284450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 46), 'pos')
    # Applying the binary operator '-' (line 190)
    result_sub_284451 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 44), '-', result_sub_284449, pos_284450)
    
    # Applying the binary operator '<<' (line 190)
    result_lshift_284452 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 28), '<<', int_284446, result_sub_284451)
    
    # Applying the binary operator '+=' (line 190)
    result_iadd_284453 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 20), '+=', num_284445, result_lshift_284452)
    # Assigning a type to the variable 'num' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'num', result_iadd_284453)
    
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 191):
    
    # Assigning a Subscript to a Name (line 191):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_284454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 33), 'int')
    slice_284455 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 191, 29), int_284454, None, None)
    # Getting the type of 'key' (line 191)
    key_284456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 29), 'key')
    # Obtaining the member '__getitem__' of a type (line 191)
    getitem___284457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 29), key_284456, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
    subscript_call_result_284458 = invoke(stypy.reporting.localization.Localization(__file__, 191, 29), getitem___284457, slice_284455)
    
    # Getting the type of 'bitdic' (line 191)
    bitdic_284459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'bitdic')
    # Obtaining the member '__getitem__' of a type (line 191)
    getitem___284460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 22), bitdic_284459, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
    subscript_call_result_284461 = invoke(stypy.reporting.localization.Localization(__file__, 191, 22), getitem___284460, subscript_call_result_284458)
    
    # Assigning a type to the variable 'pastphi' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'pastphi', subscript_call_result_284461)
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to int(...): (line 192)
    # Processing the call arguments (line 192)
    
    # Obtaining the type of the subscript
    int_284463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 25), 'int')
    # Getting the type of 'key' (line 192)
    key_284464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 21), 'key', False)
    # Obtaining the member '__getitem__' of a type (line 192)
    getitem___284465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 21), key_284464, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 192)
    subscript_call_result_284466 = invoke(stypy.reporting.localization.Localization(__file__, 192, 21), getitem___284465, int_284463)
    
    # Processing the call keyword arguments (line 192)
    kwargs_284467 = {}
    # Getting the type of 'int' (line 192)
    int_284462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'int', False)
    # Calling int(args, kwargs) (line 192)
    int_call_result_284468 = invoke(stypy.reporting.localization.Localization(__file__, 192, 17), int_284462, *[subscript_call_result_284466], **kwargs_284467)
    
    # Assigning a type to the variable 'ii' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'ii', int_call_result_284468)
    
    # Assigning a Call to a Name (line 193):
    
    # Assigning a Call to a Name (line 193):
    
    # Call to dot(...): (line 193)
    # Processing the call arguments (line 193)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 193)
    tuple_284471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 193)
    # Adding element type (line 193)
    int_284472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 28), tuple_284471, int_284472)
    # Adding element type (line 193)
    # Getting the type of 'ii' (line 193)
    ii_284473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 31), 'ii', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 28), tuple_284471, ii_284473)
    
    # Getting the type of 'm' (line 193)
    m_284474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 193)
    getitem___284475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 26), m_284474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 193)
    subscript_call_result_284476 = invoke(stypy.reporting.localization.Localization(__file__, 193, 26), getitem___284475, tuple_284471)
    
    # Getting the type of 'pastphi' (line 193)
    pastphi_284477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 36), 'pastphi', False)
    # Processing the call keyword arguments (line 193)
    kwargs_284478 = {}
    # Getting the type of 'np' (line 193)
    np_284469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'np', False)
    # Obtaining the member 'dot' of a type (line 193)
    dot_284470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 19), np_284469, 'dot')
    # Calling dot(args, kwargs) (line 193)
    dot_call_result_284479 = invoke(stypy.reporting.localization.Localization(__file__, 193, 19), dot_284470, *[subscript_call_result_284476, pastphi_284477], **kwargs_284478)
    
    # Assigning a type to the variable 'temp' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'temp', dot_call_result_284479)
    
    # Assigning a Name to a Subscript (line 194):
    
    # Assigning a Name to a Subscript (line 194):
    # Getting the type of 'temp' (line 194)
    temp_284480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 26), 'temp')
    # Getting the type of 'bitdic' (line 194)
    bitdic_284481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'bitdic')
    # Getting the type of 'key' (line 194)
    key_284482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 19), 'key')
    # Storing an element on a container (line 194)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 12), bitdic_284481, (key_284482, temp_284480))
    
    # Assigning a Name to a Subscript (line 195):
    
    # Assigning a Name to a Subscript (line 195):
    # Getting the type of 'temp' (line 195)
    temp_284483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 35), 'temp')
    # Getting the type of 'phi' (line 195)
    phi_284484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'phi')
    # Getting the type of 'num' (line 195)
    num_284485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'num')
    # Getting the type of 'fac' (line 195)
    fac_284486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'fac')
    # Applying the binary operator '*' (line 195)
    result_mul_284487 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 16), '*', num_284485, fac_284486)
    
    # Getting the type of 'step' (line 195)
    step_284488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 27), 'step')
    slice_284489 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 195, 12), result_mul_284487, None, step_284488)
    # Storing an element on a container (line 195)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 12), phi_284484, (slice_284489, temp_284483))
    
    # Assigning a Call to a Subscript (line 196):
    
    # Assigning a Call to a Subscript (line 196):
    
    # Call to dot(...): (line 196)
    # Processing the call arguments (line 196)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 196)
    tuple_284492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 196)
    # Adding element type (line 196)
    int_284493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 44), tuple_284492, int_284493)
    # Adding element type (line 196)
    # Getting the type of 'ii' (line 196)
    ii_284494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 47), 'ii', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 44), tuple_284492, ii_284494)
    
    # Getting the type of 'm' (line 196)
    m_284495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 42), 'm', False)
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___284496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 42), m_284495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 196)
    subscript_call_result_284497 = invoke(stypy.reporting.localization.Localization(__file__, 196, 42), getitem___284496, tuple_284492)
    
    # Getting the type of 'pastphi' (line 196)
    pastphi_284498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 52), 'pastphi', False)
    # Processing the call keyword arguments (line 196)
    kwargs_284499 = {}
    # Getting the type of 'np' (line 196)
    np_284490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 35), 'np', False)
    # Obtaining the member 'dot' of a type (line 196)
    dot_284491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 35), np_284490, 'dot')
    # Calling dot(args, kwargs) (line 196)
    dot_call_result_284500 = invoke(stypy.reporting.localization.Localization(__file__, 196, 35), dot_284491, *[subscript_call_result_284497, pastphi_284498], **kwargs_284499)
    
    # Getting the type of 'psi' (line 196)
    psi_284501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'psi')
    # Getting the type of 'num' (line 196)
    num_284502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'num')
    # Getting the type of 'fac' (line 196)
    fac_284503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 22), 'fac')
    # Applying the binary operator '*' (line 196)
    result_mul_284504 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 16), '*', num_284502, fac_284503)
    
    # Getting the type of 'step' (line 196)
    step_284505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), 'step')
    slice_284506 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 196, 12), result_mul_284504, None, step_284505)
    # Storing an element on a container (line 196)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 12), psi_284501, (slice_284506, dot_call_result_284500))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 197):
    
    # Assigning a Name to a Name (line 197):
    # Getting the type of 'newkeys' (line 197)
    newkeys_284507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'newkeys')
    # Assigning a type to the variable 'prevkeys' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'prevkeys', newkeys_284507)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 199)
    tuple_284508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 199)
    # Adding element type (line 199)
    # Getting the type of 'x' (line 199)
    x_284509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 11), tuple_284508, x_284509)
    # Adding element type (line 199)
    # Getting the type of 'phi' (line 199)
    phi_284510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 14), 'phi')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 11), tuple_284508, phi_284510)
    # Adding element type (line 199)
    # Getting the type of 'psi' (line 199)
    psi_284511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'psi')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 11), tuple_284508, psi_284511)
    
    # Assigning a type to the variable 'stypy_return_type' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'stypy_return_type', tuple_284508)
    
    # ################# End of 'cascade(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cascade' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_284512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284512)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cascade'
    return stypy_return_type_284512

# Assigning a type to the variable 'cascade' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'cascade', cascade)

@norecursion
def morlet(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_284513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 16), 'float')
    float_284514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 23), 'float')
    # Getting the type of 'True' (line 202)
    True_284515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 37), 'True')
    defaults = [float_284513, float_284514, True_284515]
    # Create a new context for function 'morlet'
    module_type_store = module_type_store.open_function_context('morlet', 202, 0, False)
    
    # Passed parameters checking function
    morlet.stypy_localization = localization
    morlet.stypy_type_of_self = None
    morlet.stypy_type_store = module_type_store
    morlet.stypy_function_name = 'morlet'
    morlet.stypy_param_names_list = ['M', 'w', 's', 'complete']
    morlet.stypy_varargs_param_name = None
    morlet.stypy_kwargs_param_name = None
    morlet.stypy_call_defaults = defaults
    morlet.stypy_call_varargs = varargs
    morlet.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'morlet', ['M', 'w', 's', 'complete'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'morlet', localization, ['M', 'w', 's', 'complete'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'morlet(...)' code ##################

    str_284516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, (-1)), 'str', '\n    Complex Morlet wavelet.\n\n    Parameters\n    ----------\n    M : int\n        Length of the wavelet.\n    w : float, optional\n        Omega0. Default is 5\n    s : float, optional\n        Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.\n    complete : bool, optional\n        Whether to use the complete or the standard version.\n\n    Returns\n    -------\n    morlet : (M,) ndarray\n\n    See Also\n    --------\n    scipy.signal.gausspulse\n\n    Notes\n    -----\n    The standard version::\n\n        pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))\n\n    This commonly used wavelet is often referred to simply as the\n    Morlet wavelet.  Note that this simplified version can cause\n    admissibility problems at low values of `w`.\n\n    The complete version::\n\n        pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))\n\n    This version has a correction\n    term to improve admissibility. For `w` greater than 5, the\n    correction term is negligible.\n\n    Note that the energy of the return wavelet is not normalised\n    according to `s`.\n\n    The fundamental frequency of this wavelet in Hz is given\n    by ``f = 2*s*w*r / M`` where `r` is the sampling rate.\n    \n    Note: This function was created before `cwt` and is not compatible\n    with it.\n\n    ')
    
    # Assigning a Call to a Name (line 253):
    
    # Assigning a Call to a Name (line 253):
    
    # Call to linspace(...): (line 253)
    # Processing the call arguments (line 253)
    
    # Getting the type of 's' (line 253)
    s_284518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 18), 's', False)
    # Applying the 'usub' unary operator (line 253)
    result___neg___284519 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 17), 'usub', s_284518)
    
    int_284520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 22), 'int')
    # Applying the binary operator '*' (line 253)
    result_mul_284521 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 17), '*', result___neg___284519, int_284520)
    
    # Getting the type of 'pi' (line 253)
    pi_284522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'pi', False)
    # Applying the binary operator '*' (line 253)
    result_mul_284523 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 24), '*', result_mul_284521, pi_284522)
    
    # Getting the type of 's' (line 253)
    s_284524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 30), 's', False)
    int_284525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 34), 'int')
    # Applying the binary operator '*' (line 253)
    result_mul_284526 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 30), '*', s_284524, int_284525)
    
    # Getting the type of 'pi' (line 253)
    pi_284527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 38), 'pi', False)
    # Applying the binary operator '*' (line 253)
    result_mul_284528 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 36), '*', result_mul_284526, pi_284527)
    
    # Getting the type of 'M' (line 253)
    M_284529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 42), 'M', False)
    # Processing the call keyword arguments (line 253)
    kwargs_284530 = {}
    # Getting the type of 'linspace' (line 253)
    linspace_284517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'linspace', False)
    # Calling linspace(args, kwargs) (line 253)
    linspace_call_result_284531 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), linspace_284517, *[result_mul_284523, result_mul_284528, M_284529], **kwargs_284530)
    
    # Assigning a type to the variable 'x' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'x', linspace_call_result_284531)
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to exp(...): (line 254)
    # Processing the call arguments (line 254)
    complex_284533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 17), 'complex')
    # Getting the type of 'w' (line 254)
    w_284534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 22), 'w', False)
    # Applying the binary operator '*' (line 254)
    result_mul_284535 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 17), '*', complex_284533, w_284534)
    
    # Getting the type of 'x' (line 254)
    x_284536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'x', False)
    # Applying the binary operator '*' (line 254)
    result_mul_284537 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 24), '*', result_mul_284535, x_284536)
    
    # Processing the call keyword arguments (line 254)
    kwargs_284538 = {}
    # Getting the type of 'exp' (line 254)
    exp_284532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 13), 'exp', False)
    # Calling exp(args, kwargs) (line 254)
    exp_call_result_284539 = invoke(stypy.reporting.localization.Localization(__file__, 254, 13), exp_284532, *[result_mul_284537], **kwargs_284538)
    
    # Assigning a type to the variable 'output' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'output', exp_call_result_284539)
    
    # Getting the type of 'complete' (line 256)
    complete_284540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 7), 'complete')
    # Testing the type of an if condition (line 256)
    if_condition_284541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 4), complete_284540)
    # Assigning a type to the variable 'if_condition_284541' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'if_condition_284541', if_condition_284541)
    # SSA begins for if statement (line 256)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'output' (line 257)
    output_284542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'output')
    
    # Call to exp(...): (line 257)
    # Processing the call arguments (line 257)
    float_284544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 22), 'float')
    # Getting the type of 'w' (line 257)
    w_284545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 30), 'w', False)
    int_284546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 33), 'int')
    # Applying the binary operator '**' (line 257)
    result_pow_284547 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 30), '**', w_284545, int_284546)
    
    # Applying the binary operator '*' (line 257)
    result_mul_284548 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 22), '*', float_284544, result_pow_284547)
    
    # Processing the call keyword arguments (line 257)
    kwargs_284549 = {}
    # Getting the type of 'exp' (line 257)
    exp_284543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 18), 'exp', False)
    # Calling exp(args, kwargs) (line 257)
    exp_call_result_284550 = invoke(stypy.reporting.localization.Localization(__file__, 257, 18), exp_284543, *[result_mul_284548], **kwargs_284549)
    
    # Applying the binary operator '-=' (line 257)
    result_isub_284551 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 8), '-=', output_284542, exp_call_result_284550)
    # Assigning a type to the variable 'output' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'output', result_isub_284551)
    
    # SSA join for if statement (line 256)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'output' (line 259)
    output_284552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'output')
    
    # Call to exp(...): (line 259)
    # Processing the call arguments (line 259)
    float_284554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 18), 'float')
    # Getting the type of 'x' (line 259)
    x_284555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 26), 'x', False)
    int_284556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 29), 'int')
    # Applying the binary operator '**' (line 259)
    result_pow_284557 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 26), '**', x_284555, int_284556)
    
    # Applying the binary operator '*' (line 259)
    result_mul_284558 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 18), '*', float_284554, result_pow_284557)
    
    # Processing the call keyword arguments (line 259)
    kwargs_284559 = {}
    # Getting the type of 'exp' (line 259)
    exp_284553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 14), 'exp', False)
    # Calling exp(args, kwargs) (line 259)
    exp_call_result_284560 = invoke(stypy.reporting.localization.Localization(__file__, 259, 14), exp_284553, *[result_mul_284558], **kwargs_284559)
    
    # Getting the type of 'pi' (line 259)
    pi_284561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 35), 'pi')
    float_284562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 40), 'float')
    # Applying the binary operator '**' (line 259)
    result_pow_284563 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 35), '**', pi_284561, float_284562)
    
    # Applying the binary operator '*' (line 259)
    result_mul_284564 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 14), '*', exp_call_result_284560, result_pow_284563)
    
    # Applying the binary operator '*=' (line 259)
    result_imul_284565 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 4), '*=', output_284552, result_mul_284564)
    # Assigning a type to the variable 'output' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'output', result_imul_284565)
    
    # Getting the type of 'output' (line 261)
    output_284566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type', output_284566)
    
    # ################# End of 'morlet(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'morlet' in the type store
    # Getting the type of 'stypy_return_type' (line 202)
    stypy_return_type_284567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284567)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'morlet'
    return stypy_return_type_284567

# Assigning a type to the variable 'morlet' (line 202)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'morlet', morlet)

@norecursion
def ricker(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ricker'
    module_type_store = module_type_store.open_function_context('ricker', 264, 0, False)
    
    # Passed parameters checking function
    ricker.stypy_localization = localization
    ricker.stypy_type_of_self = None
    ricker.stypy_type_store = module_type_store
    ricker.stypy_function_name = 'ricker'
    ricker.stypy_param_names_list = ['points', 'a']
    ricker.stypy_varargs_param_name = None
    ricker.stypy_kwargs_param_name = None
    ricker.stypy_call_defaults = defaults
    ricker.stypy_call_varargs = varargs
    ricker.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ricker', ['points', 'a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ricker', localization, ['points', 'a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ricker(...)' code ##################

    str_284568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, (-1)), 'str', '\n    Return a Ricker wavelet, also known as the "Mexican hat wavelet".\n\n    It models the function:\n\n        ``A (1 - x^2/a^2) exp(-x^2/2 a^2)``,\n\n    where ``A = 2/sqrt(3a)pi^1/4``.\n\n    Parameters\n    ----------\n    points : int\n        Number of points in `vector`.\n        Will be centered around 0.\n    a : scalar\n        Width parameter of the wavelet.\n\n    Returns\n    -------\n    vector : (N,) ndarray\n        Array of length `points` in shape of ricker curve.\n\n    Examples\n    --------\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n\n    >>> points = 100\n    >>> a = 4.0\n    >>> vec2 = signal.ricker(points, a)\n    >>> print(len(vec2))\n    100\n    >>> plt.plot(vec2)\n    >>> plt.show()\n\n    ')
    
    # Assigning a BinOp to a Name (line 301):
    
    # Assigning a BinOp to a Name (line 301):
    int_284569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 8), 'int')
    
    # Call to sqrt(...): (line 301)
    # Processing the call arguments (line 301)
    int_284572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 21), 'int')
    # Getting the type of 'a' (line 301)
    a_284573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'a', False)
    # Applying the binary operator '*' (line 301)
    result_mul_284574 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 21), '*', int_284572, a_284573)
    
    # Processing the call keyword arguments (line 301)
    kwargs_284575 = {}
    # Getting the type of 'np' (line 301)
    np_284570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 13), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 301)
    sqrt_284571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 13), np_284570, 'sqrt')
    # Calling sqrt(args, kwargs) (line 301)
    sqrt_call_result_284576 = invoke(stypy.reporting.localization.Localization(__file__, 301, 13), sqrt_284571, *[result_mul_284574], **kwargs_284575)
    
    # Getting the type of 'np' (line 301)
    np_284577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 31), 'np')
    # Obtaining the member 'pi' of a type (line 301)
    pi_284578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 31), np_284577, 'pi')
    float_284579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 38), 'float')
    # Applying the binary operator '**' (line 301)
    result_pow_284580 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 31), '**', pi_284578, float_284579)
    
    # Applying the binary operator '*' (line 301)
    result_mul_284581 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 13), '*', sqrt_call_result_284576, result_pow_284580)
    
    # Applying the binary operator 'div' (line 301)
    result_div_284582 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 8), 'div', int_284569, result_mul_284581)
    
    # Assigning a type to the variable 'A' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'A', result_div_284582)
    
    # Assigning a BinOp to a Name (line 302):
    
    # Assigning a BinOp to a Name (line 302):
    # Getting the type of 'a' (line 302)
    a_284583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 10), 'a')
    int_284584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 13), 'int')
    # Applying the binary operator '**' (line 302)
    result_pow_284585 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 10), '**', a_284583, int_284584)
    
    # Assigning a type to the variable 'wsq' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'wsq', result_pow_284585)
    
    # Assigning a BinOp to a Name (line 303):
    
    # Assigning a BinOp to a Name (line 303):
    
    # Call to arange(...): (line 303)
    # Processing the call arguments (line 303)
    int_284588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 20), 'int')
    # Getting the type of 'points' (line 303)
    points_284589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'points', False)
    # Processing the call keyword arguments (line 303)
    kwargs_284590 = {}
    # Getting the type of 'np' (line 303)
    np_284586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 10), 'np', False)
    # Obtaining the member 'arange' of a type (line 303)
    arange_284587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 10), np_284586, 'arange')
    # Calling arange(args, kwargs) (line 303)
    arange_call_result_284591 = invoke(stypy.reporting.localization.Localization(__file__, 303, 10), arange_284587, *[int_284588, points_284589], **kwargs_284590)
    
    # Getting the type of 'points' (line 303)
    points_284592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 34), 'points')
    float_284593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 43), 'float')
    # Applying the binary operator '-' (line 303)
    result_sub_284594 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 34), '-', points_284592, float_284593)
    
    int_284595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 50), 'int')
    # Applying the binary operator 'div' (line 303)
    result_div_284596 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 33), 'div', result_sub_284594, int_284595)
    
    # Applying the binary operator '-' (line 303)
    result_sub_284597 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 10), '-', arange_call_result_284591, result_div_284596)
    
    # Assigning a type to the variable 'vec' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'vec', result_sub_284597)
    
    # Assigning a BinOp to a Name (line 304):
    
    # Assigning a BinOp to a Name (line 304):
    # Getting the type of 'vec' (line 304)
    vec_284598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 10), 'vec')
    int_284599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 15), 'int')
    # Applying the binary operator '**' (line 304)
    result_pow_284600 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 10), '**', vec_284598, int_284599)
    
    # Assigning a type to the variable 'xsq' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'xsq', result_pow_284600)
    
    # Assigning a BinOp to a Name (line 305):
    
    # Assigning a BinOp to a Name (line 305):
    int_284601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 11), 'int')
    # Getting the type of 'xsq' (line 305)
    xsq_284602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'xsq')
    # Getting the type of 'wsq' (line 305)
    wsq_284603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 21), 'wsq')
    # Applying the binary operator 'div' (line 305)
    result_div_284604 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 15), 'div', xsq_284602, wsq_284603)
    
    # Applying the binary operator '-' (line 305)
    result_sub_284605 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 11), '-', int_284601, result_div_284604)
    
    # Assigning a type to the variable 'mod' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'mod', result_sub_284605)
    
    # Assigning a Call to a Name (line 306):
    
    # Assigning a Call to a Name (line 306):
    
    # Call to exp(...): (line 306)
    # Processing the call arguments (line 306)
    
    # Getting the type of 'xsq' (line 306)
    xsq_284608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 20), 'xsq', False)
    # Applying the 'usub' unary operator (line 306)
    result___neg___284609 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 19), 'usub', xsq_284608)
    
    int_284610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 27), 'int')
    # Getting the type of 'wsq' (line 306)
    wsq_284611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 31), 'wsq', False)
    # Applying the binary operator '*' (line 306)
    result_mul_284612 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 27), '*', int_284610, wsq_284611)
    
    # Applying the binary operator 'div' (line 306)
    result_div_284613 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 19), 'div', result___neg___284609, result_mul_284612)
    
    # Processing the call keyword arguments (line 306)
    kwargs_284614 = {}
    # Getting the type of 'np' (line 306)
    np_284606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'np', False)
    # Obtaining the member 'exp' of a type (line 306)
    exp_284607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), np_284606, 'exp')
    # Calling exp(args, kwargs) (line 306)
    exp_call_result_284615 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), exp_284607, *[result_div_284613], **kwargs_284614)
    
    # Assigning a type to the variable 'gauss' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'gauss', exp_call_result_284615)
    
    # Assigning a BinOp to a Name (line 307):
    
    # Assigning a BinOp to a Name (line 307):
    # Getting the type of 'A' (line 307)
    A_284616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'A')
    # Getting the type of 'mod' (line 307)
    mod_284617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'mod')
    # Applying the binary operator '*' (line 307)
    result_mul_284618 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 12), '*', A_284616, mod_284617)
    
    # Getting the type of 'gauss' (line 307)
    gauss_284619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 22), 'gauss')
    # Applying the binary operator '*' (line 307)
    result_mul_284620 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 20), '*', result_mul_284618, gauss_284619)
    
    # Assigning a type to the variable 'total' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'total', result_mul_284620)
    # Getting the type of 'total' (line 308)
    total_284621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 11), 'total')
    # Assigning a type to the variable 'stypy_return_type' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'stypy_return_type', total_284621)
    
    # ################# End of 'ricker(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ricker' in the type store
    # Getting the type of 'stypy_return_type' (line 264)
    stypy_return_type_284622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284622)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ricker'
    return stypy_return_type_284622

# Assigning a type to the variable 'ricker' (line 264)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'ricker', ricker)

@norecursion
def cwt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cwt'
    module_type_store = module_type_store.open_function_context('cwt', 311, 0, False)
    
    # Passed parameters checking function
    cwt.stypy_localization = localization
    cwt.stypy_type_of_self = None
    cwt.stypy_type_store = module_type_store
    cwt.stypy_function_name = 'cwt'
    cwt.stypy_param_names_list = ['data', 'wavelet', 'widths']
    cwt.stypy_varargs_param_name = None
    cwt.stypy_kwargs_param_name = None
    cwt.stypy_call_defaults = defaults
    cwt.stypy_call_varargs = varargs
    cwt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cwt', ['data', 'wavelet', 'widths'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cwt', localization, ['data', 'wavelet', 'widths'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cwt(...)' code ##################

    str_284623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, (-1)), 'str', "\n    Continuous wavelet transform.\n\n    Performs a continuous wavelet transform on `data`,\n    using the `wavelet` function. A CWT performs a convolution\n    with `data` using the `wavelet` function, which is characterized\n    by a width parameter and length parameter.\n\n    Parameters\n    ----------\n    data : (N,) ndarray\n        data on which to perform the transform.\n    wavelet : function\n        Wavelet function, which should take 2 arguments.\n        The first argument is the number of points that the returned vector\n        will have (len(wavelet(length,width)) == length).\n        The second is a width parameter, defining the size of the wavelet\n        (e.g. standard deviation of a gaussian). See `ricker`, which\n        satisfies these requirements.\n    widths : (M,) sequence\n        Widths to use for transform.\n\n    Returns\n    -------\n    cwt: (M, N) ndarray\n        Will have shape of (len(widths), len(data)).\n\n    Notes\n    -----\n    ::\n\n        length = min(10 * width[ii], len(data))\n        cwt[ii,:] = signal.convolve(data, wavelet(length,\n                                    width[ii]), mode='same')\n\n    Examples\n    --------\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> t = np.linspace(-1, 1, 200, endpoint=False)\n    >>> sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)\n    >>> widths = np.arange(1, 31)\n    >>> cwtmatr = signal.cwt(sig, signal.ricker, widths)\n    >>> plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',\n    ...            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())\n    >>> plt.show()\n\n    ")
    
    # Assigning a Call to a Name (line 360):
    
    # Assigning a Call to a Name (line 360):
    
    # Call to zeros(...): (line 360)
    # Processing the call arguments (line 360)
    
    # Obtaining an instance of the builtin type 'list' (line 360)
    list_284626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 360)
    # Adding element type (line 360)
    
    # Call to len(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'widths' (line 360)
    widths_284628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 27), 'widths', False)
    # Processing the call keyword arguments (line 360)
    kwargs_284629 = {}
    # Getting the type of 'len' (line 360)
    len_284627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'len', False)
    # Calling len(args, kwargs) (line 360)
    len_call_result_284630 = invoke(stypy.reporting.localization.Localization(__file__, 360, 23), len_284627, *[widths_284628], **kwargs_284629)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 22), list_284626, len_call_result_284630)
    # Adding element type (line 360)
    
    # Call to len(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'data' (line 360)
    data_284632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 40), 'data', False)
    # Processing the call keyword arguments (line 360)
    kwargs_284633 = {}
    # Getting the type of 'len' (line 360)
    len_284631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 36), 'len', False)
    # Calling len(args, kwargs) (line 360)
    len_call_result_284634 = invoke(stypy.reporting.localization.Localization(__file__, 360, 36), len_284631, *[data_284632], **kwargs_284633)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 22), list_284626, len_call_result_284634)
    
    # Processing the call keyword arguments (line 360)
    kwargs_284635 = {}
    # Getting the type of 'np' (line 360)
    np_284624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 13), 'np', False)
    # Obtaining the member 'zeros' of a type (line 360)
    zeros_284625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 13), np_284624, 'zeros')
    # Calling zeros(args, kwargs) (line 360)
    zeros_call_result_284636 = invoke(stypy.reporting.localization.Localization(__file__, 360, 13), zeros_284625, *[list_284626], **kwargs_284635)
    
    # Assigning a type to the variable 'output' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'output', zeros_call_result_284636)
    
    
    # Call to enumerate(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'widths' (line 361)
    widths_284638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 32), 'widths', False)
    # Processing the call keyword arguments (line 361)
    kwargs_284639 = {}
    # Getting the type of 'enumerate' (line 361)
    enumerate_284637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 22), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 361)
    enumerate_call_result_284640 = invoke(stypy.reporting.localization.Localization(__file__, 361, 22), enumerate_284637, *[widths_284638], **kwargs_284639)
    
    # Testing the type of a for loop iterable (line 361)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 361, 4), enumerate_call_result_284640)
    # Getting the type of the for loop variable (line 361)
    for_loop_var_284641 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 361, 4), enumerate_call_result_284640)
    # Assigning a type to the variable 'ind' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 4), for_loop_var_284641))
    # Assigning a type to the variable 'width' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'width', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 4), for_loop_var_284641))
    # SSA begins for a for statement (line 361)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 362):
    
    # Assigning a Call to a Name (line 362):
    
    # Call to wavelet(...): (line 362)
    # Processing the call arguments (line 362)
    
    # Call to min(...): (line 362)
    # Processing the call arguments (line 362)
    int_284644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 35), 'int')
    # Getting the type of 'width' (line 362)
    width_284645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 40), 'width', False)
    # Applying the binary operator '*' (line 362)
    result_mul_284646 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 35), '*', int_284644, width_284645)
    
    
    # Call to len(...): (line 362)
    # Processing the call arguments (line 362)
    # Getting the type of 'data' (line 362)
    data_284648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 51), 'data', False)
    # Processing the call keyword arguments (line 362)
    kwargs_284649 = {}
    # Getting the type of 'len' (line 362)
    len_284647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 47), 'len', False)
    # Calling len(args, kwargs) (line 362)
    len_call_result_284650 = invoke(stypy.reporting.localization.Localization(__file__, 362, 47), len_284647, *[data_284648], **kwargs_284649)
    
    # Processing the call keyword arguments (line 362)
    kwargs_284651 = {}
    # Getting the type of 'min' (line 362)
    min_284643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 31), 'min', False)
    # Calling min(args, kwargs) (line 362)
    min_call_result_284652 = invoke(stypy.reporting.localization.Localization(__file__, 362, 31), min_284643, *[result_mul_284646, len_call_result_284650], **kwargs_284651)
    
    # Getting the type of 'width' (line 362)
    width_284653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 59), 'width', False)
    # Processing the call keyword arguments (line 362)
    kwargs_284654 = {}
    # Getting the type of 'wavelet' (line 362)
    wavelet_284642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 23), 'wavelet', False)
    # Calling wavelet(args, kwargs) (line 362)
    wavelet_call_result_284655 = invoke(stypy.reporting.localization.Localization(__file__, 362, 23), wavelet_284642, *[min_call_result_284652, width_284653], **kwargs_284654)
    
    # Assigning a type to the variable 'wavelet_data' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'wavelet_data', wavelet_call_result_284655)
    
    # Assigning a Call to a Subscript (line 363):
    
    # Assigning a Call to a Subscript (line 363):
    
    # Call to convolve(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'data' (line 363)
    data_284657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 34), 'data', False)
    # Getting the type of 'wavelet_data' (line 363)
    wavelet_data_284658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 40), 'wavelet_data', False)
    # Processing the call keyword arguments (line 363)
    str_284659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 39), 'str', 'same')
    keyword_284660 = str_284659
    kwargs_284661 = {'mode': keyword_284660}
    # Getting the type of 'convolve' (line 363)
    convolve_284656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'convolve', False)
    # Calling convolve(args, kwargs) (line 363)
    convolve_call_result_284662 = invoke(stypy.reporting.localization.Localization(__file__, 363, 25), convolve_284656, *[data_284657, wavelet_data_284658], **kwargs_284661)
    
    # Getting the type of 'output' (line 363)
    output_284663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'output')
    # Getting the type of 'ind' (line 363)
    ind_284664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'ind')
    slice_284665 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 363, 8), None, None, None)
    # Storing an element on a container (line 363)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 8), output_284663, ((ind_284664, slice_284665), convolve_call_result_284662))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'output' (line 365)
    output_284666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 11), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'stypy_return_type', output_284666)
    
    # ################# End of 'cwt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cwt' in the type store
    # Getting the type of 'stypy_return_type' (line 311)
    stypy_return_type_284667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284667)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cwt'
    return stypy_return_type_284667

# Assigning a type to the variable 'cwt' (line 311)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 0), 'cwt', cwt)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
