
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ltisys -- a collection of functions to convert linear time invariant systems
3: from one representation to another.
4: '''
5: from __future__ import division, print_function, absolute_import
6: 
7: import numpy
8: import numpy as np
9: from numpy import (r_, eye, atleast_2d, poly, dot,
10:                    asarray, product, zeros, array, outer)
11: from scipy import linalg
12: 
13: from .filter_design import tf2zpk, zpk2tf, normalize
14: 
15: 
16: __all__ = ['tf2ss', 'abcd_normalize', 'ss2tf', 'zpk2ss', 'ss2zpk',
17:            'cont2discrete']
18: 
19: 
20: def tf2ss(num, den):
21:     r'''Transfer function to state-space representation.
22: 
23:     Parameters
24:     ----------
25:     num, den : array_like
26:         Sequences representing the coefficients of the numerator and
27:         denominator polynomials, in order of descending degree. The
28:         denominator needs to be at least as long as the numerator.
29: 
30:     Returns
31:     -------
32:     A, B, C, D : ndarray
33:         State space representation of the system, in controller canonical
34:         form.
35: 
36:     Examples
37:     --------
38:     Convert the transfer function:
39: 
40:     .. math:: H(s) = \frac{s^2 + 3s + 3}{s^2 + 2s + 1}
41: 
42:     >>> num = [1, 3, 3]
43:     >>> den = [1, 2, 1]
44: 
45:     to the state-space representation:
46: 
47:     .. math::
48: 
49:         \dot{\textbf{x}}(t) =
50:         \begin{bmatrix} -2 & -1 \\ 1 & 0 \end{bmatrix} \textbf{x}(t) +
51:         \begin{bmatrix} 1 \\ 0 \end{bmatrix} \textbf{u}(t) \\
52: 
53:         \textbf{y}(t) = \begin{bmatrix} 1 & 2 \end{bmatrix} \textbf{x}(t) +
54:         \begin{bmatrix} 1 \end{bmatrix} \textbf{u}(t)
55: 
56:     >>> from scipy.signal import tf2ss
57:     >>> A, B, C, D = tf2ss(num, den)
58:     >>> A
59:     array([[-2., -1.],
60:            [ 1.,  0.]])
61:     >>> B
62:     array([[ 1.],
63:            [ 0.]])
64:     >>> C
65:     array([[ 1.,  2.]])
66:     >>> D
67:     array([[ 1.]])
68:     '''
69:     # Controller canonical state-space representation.
70:     #  if M+1 = len(num) and K+1 = len(den) then we must have M <= K
71:     #  states are found by asserting that X(s) = U(s) / D(s)
72:     #  then Y(s) = N(s) * X(s)
73:     #
74:     #   A, B, C, and D follow quite naturally.
75:     #
76:     num, den = normalize(num, den)   # Strips zeros, checks arrays
77:     nn = len(num.shape)
78:     if nn == 1:
79:         num = asarray([num], num.dtype)
80:     M = num.shape[1]
81:     K = len(den)
82:     if M > K:
83:         msg = "Improper transfer function. `num` is longer than `den`."
84:         raise ValueError(msg)
85:     if M == 0 or K == 0:  # Null system
86:         return (array([], float), array([], float), array([], float),
87:                 array([], float))
88: 
89:     # pad numerator to have same number of columns has denominator
90:     num = r_['-1', zeros((num.shape[0], K - M), num.dtype), num]
91: 
92:     if num.shape[-1] > 0:
93:         D = atleast_2d(num[:, 0])
94: 
95:     else:
96:         # We don't assign it an empty array because this system
97:         # is not 'null'. It just doesn't have a non-zero D
98:         # matrix. Thus, it should have a non-zero shape so that
99:         # it can be operated on by functions like 'ss2tf'
100:         D = array([[0]], float)
101: 
102:     if K == 1:
103:         D = D.reshape(num.shape)
104: 
105:         return (zeros((1, 1)), zeros((1, D.shape[1])),
106:                 zeros((D.shape[0], 1)), D)
107: 
108:     frow = -array([den[1:]])
109:     A = r_[frow, eye(K - 2, K - 1)]
110:     B = eye(K - 1, 1)
111:     C = num[:, 1:] - outer(num[:, 0], den[1:])
112:     D = D.reshape((C.shape[0], B.shape[1]))
113: 
114:     return A, B, C, D
115: 
116: 
117: def _none_to_empty_2d(arg):
118:     if arg is None:
119:         return zeros((0, 0))
120:     else:
121:         return arg
122: 
123: 
124: def _atleast_2d_or_none(arg):
125:     if arg is not None:
126:         return atleast_2d(arg)
127: 
128: 
129: def _shape_or_none(M):
130:     if M is not None:
131:         return M.shape
132:     else:
133:         return (None,) * 2
134: 
135: 
136: def _choice_not_none(*args):
137:     for arg in args:
138:         if arg is not None:
139:             return arg
140: 
141: 
142: def _restore(M, shape):
143:     if M.shape == (0, 0):
144:         return zeros(shape)
145:     else:
146:         if M.shape != shape:
147:             raise ValueError("The input arrays have incompatible shapes.")
148:         return M
149: 
150: 
151: def abcd_normalize(A=None, B=None, C=None, D=None):
152:     '''Check state-space matrices and ensure they are two-dimensional.
153: 
154:     If enough information on the system is provided, that is, enough
155:     properly-shaped arrays are passed to the function, the missing ones
156:     are built from this information, ensuring the correct number of
157:     rows and columns. Otherwise a ValueError is raised.
158: 
159:     Parameters
160:     ----------
161:     A, B, C, D : array_like, optional
162:         State-space matrices. All of them are None (missing) by default.
163:         See `ss2tf` for format.
164: 
165:     Returns
166:     -------
167:     A, B, C, D : array
168:         Properly shaped state-space matrices.
169: 
170:     Raises
171:     ------
172:     ValueError
173:         If not enough information on the system was provided.
174: 
175:     '''
176:     A, B, C, D = map(_atleast_2d_or_none, (A, B, C, D))
177: 
178:     MA, NA = _shape_or_none(A)
179:     MB, NB = _shape_or_none(B)
180:     MC, NC = _shape_or_none(C)
181:     MD, ND = _shape_or_none(D)
182: 
183:     p = _choice_not_none(MA, MB, NC)
184:     q = _choice_not_none(NB, ND)
185:     r = _choice_not_none(MC, MD)
186:     if p is None or q is None or r is None:
187:         raise ValueError("Not enough information on the system.")
188: 
189:     A, B, C, D = map(_none_to_empty_2d, (A, B, C, D))
190:     A = _restore(A, (p, p))
191:     B = _restore(B, (p, q))
192:     C = _restore(C, (r, p))
193:     D = _restore(D, (r, q))
194: 
195:     return A, B, C, D
196: 
197: 
198: def ss2tf(A, B, C, D, input=0):
199:     r'''State-space to transfer function.
200: 
201:     A, B, C, D defines a linear state-space system with `p` inputs,
202:     `q` outputs, and `n` state variables.
203: 
204:     Parameters
205:     ----------
206:     A : array_like
207:         State (or system) matrix of shape ``(n, n)``
208:     B : array_like
209:         Input matrix of shape ``(n, p)``
210:     C : array_like
211:         Output matrix of shape ``(q, n)``
212:     D : array_like
213:         Feedthrough (or feedforward) matrix of shape ``(q, p)``
214:     input : int, optional
215:         For multiple-input systems, the index of the input to use.
216: 
217:     Returns
218:     -------
219:     num : 2-D ndarray
220:         Numerator(s) of the resulting transfer function(s).  `num` has one row
221:         for each of the system's outputs. Each row is a sequence representation
222:         of the numerator polynomial.
223:     den : 1-D ndarray
224:         Denominator of the resulting transfer function(s).  `den` is a sequence
225:         representation of the denominator polynomial.
226: 
227:     Examples
228:     --------
229:     Convert the state-space representation:
230: 
231:     .. math::
232: 
233:         \dot{\textbf{x}}(t) =
234:         \begin{bmatrix} -2 & -1 \\ 1 & 0 \end{bmatrix} \textbf{x}(t) +
235:         \begin{bmatrix} 1 \\ 0 \end{bmatrix} \textbf{u}(t) \\
236: 
237:         \textbf{y}(t) = \begin{bmatrix} 1 & 2 \end{bmatrix} \textbf{x}(t) +
238:         \begin{bmatrix} 1 \end{bmatrix} \textbf{u}(t)
239: 
240:     >>> A = [[-2, -1], [1, 0]]
241:     >>> B = [[1], [0]]  # 2-dimensional column vector
242:     >>> C = [[1, 2]]    # 2-dimensional row vector
243:     >>> D = 1
244: 
245:     to the transfer function:
246: 
247:     .. math:: H(s) = \frac{s^2 + 3s + 3}{s^2 + 2s + 1}
248: 
249:     >>> from scipy.signal import ss2tf
250:     >>> ss2tf(A, B, C, D)
251:     (array([[1, 3, 3]]), array([ 1.,  2.,  1.]))
252:     '''
253:     # transfer function is C (sI - A)**(-1) B + D
254: 
255:     # Check consistency and make them all rank-2 arrays
256:     A, B, C, D = abcd_normalize(A, B, C, D)
257: 
258:     nout, nin = D.shape
259:     if input >= nin:
260:         raise ValueError("System does not have the input specified.")
261: 
262:     # make SIMO from possibly MIMO system.
263:     B = B[:, input:input + 1]
264:     D = D[:, input:input + 1]
265: 
266:     try:
267:         den = poly(A)
268:     except ValueError:
269:         den = 1
270: 
271:     if (product(B.shape, axis=0) == 0) and (product(C.shape, axis=0) == 0):
272:         num = numpy.ravel(D)
273:         if (product(D.shape, axis=0) == 0) and (product(A.shape, axis=0) == 0):
274:             den = []
275:         return num, den
276: 
277:     num_states = A.shape[0]
278:     type_test = A[:, 0] + B[:, 0] + C[0, :] + D
279:     num = numpy.zeros((nout, num_states + 1), type_test.dtype)
280:     for k in range(nout):
281:         Ck = atleast_2d(C[k, :])
282:         num[k] = poly(A - dot(B, Ck)) + (D[k] - 1) * den
283: 
284:     return num, den
285: 
286: 
287: def zpk2ss(z, p, k):
288:     '''Zero-pole-gain representation to state-space representation
289: 
290:     Parameters
291:     ----------
292:     z, p : sequence
293:         Zeros and poles.
294:     k : float
295:         System gain.
296: 
297:     Returns
298:     -------
299:     A, B, C, D : ndarray
300:         State space representation of the system, in controller canonical
301:         form.
302: 
303:     '''
304:     return tf2ss(*zpk2tf(z, p, k))
305: 
306: 
307: def ss2zpk(A, B, C, D, input=0):
308:     '''State-space representation to zero-pole-gain representation.
309: 
310:     A, B, C, D defines a linear state-space system with `p` inputs,
311:     `q` outputs, and `n` state variables.
312: 
313:     Parameters
314:     ----------
315:     A : array_like
316:         State (or system) matrix of shape ``(n, n)``
317:     B : array_like
318:         Input matrix of shape ``(n, p)``
319:     C : array_like
320:         Output matrix of shape ``(q, n)``
321:     D : array_like
322:         Feedthrough (or feedforward) matrix of shape ``(q, p)``
323:     input : int, optional
324:         For multiple-input systems, the index of the input to use.
325: 
326:     Returns
327:     -------
328:     z, p : sequence
329:         Zeros and poles.
330:     k : float
331:         System gain.
332: 
333:     '''
334:     return tf2zpk(*ss2tf(A, B, C, D, input=input))
335: 
336: 
337: def cont2discrete(system, dt, method="zoh", alpha=None):
338:     '''
339:     Transform a continuous to a discrete state-space system.
340: 
341:     Parameters
342:     ----------
343:     system : a tuple describing the system or an instance of `lti`
344:         The following gives the number of elements in the tuple and
345:         the interpretation:
346: 
347:             * 1: (instance of `lti`)
348:             * 2: (num, den)
349:             * 3: (zeros, poles, gain)
350:             * 4: (A, B, C, D)
351: 
352:     dt : float
353:         The discretization time step.
354:     method : {"gbt", "bilinear", "euler", "backward_diff", "zoh"}, optional
355:         Which method to use:
356: 
357:             * gbt: generalized bilinear transformation
358:             * bilinear: Tustin's approximation ("gbt" with alpha=0.5)
359:             * euler: Euler (or forward differencing) method ("gbt" with alpha=0)
360:             * backward_diff: Backwards differencing ("gbt" with alpha=1.0)
361:             * zoh: zero-order hold (default)
362: 
363:     alpha : float within [0, 1], optional
364:         The generalized bilinear transformation weighting parameter, which
365:         should only be specified with method="gbt", and is ignored otherwise
366: 
367:     Returns
368:     -------
369:     sysd : tuple containing the discrete system
370:         Based on the input type, the output will be of the form
371: 
372:         * (num, den, dt)   for transfer function input
373:         * (zeros, poles, gain, dt)   for zeros-poles-gain input
374:         * (A, B, C, D, dt) for state-space system input
375: 
376:     Notes
377:     -----
378:     By default, the routine uses a Zero-Order Hold (zoh) method to perform
379:     the transformation.  Alternatively, a generalized bilinear transformation
380:     may be used, which includes the common Tustin's bilinear approximation,
381:     an Euler's method technique, or a backwards differencing technique.
382: 
383:     The Zero-Order Hold (zoh) method is based on [1]_, the generalized bilinear
384:     approximation is based on [2]_ and [3]_.
385: 
386:     References
387:     ----------
388:     .. [1] http://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
389: 
390:     .. [2] http://techteach.no/publications/discretetime_signals_systems/discrete.pdf
391: 
392:     .. [3] G. Zhang, X. Chen, and T. Chen, Digital redesign via the generalized
393:         bilinear transformation, Int. J. Control, vol. 82, no. 4, pp. 741-754,
394:         2009.
395:         (http://www.mypolyuweb.hk/~magzhang/Research/ZCC09_IJC.pdf)
396: 
397:     '''
398:     if len(system) == 1:
399:         return system.to_discrete()
400:     if len(system) == 2:
401:         sysd = cont2discrete(tf2ss(system[0], system[1]), dt, method=method,
402:                              alpha=alpha)
403:         return ss2tf(sysd[0], sysd[1], sysd[2], sysd[3]) + (dt,)
404:     elif len(system) == 3:
405:         sysd = cont2discrete(zpk2ss(system[0], system[1], system[2]), dt,
406:                              method=method, alpha=alpha)
407:         return ss2zpk(sysd[0], sysd[1], sysd[2], sysd[3]) + (dt,)
408:     elif len(system) == 4:
409:         a, b, c, d = system
410:     else:
411:         raise ValueError("First argument must either be a tuple of 2 (tf), "
412:                          "3 (zpk), or 4 (ss) arrays.")
413: 
414:     if method == 'gbt':
415:         if alpha is None:
416:             raise ValueError("Alpha parameter must be specified for the "
417:                              "generalized bilinear transform (gbt) method")
418:         elif alpha < 0 or alpha > 1:
419:             raise ValueError("Alpha parameter must be within the interval "
420:                              "[0,1] for the gbt method")
421: 
422:     if method == 'gbt':
423:         # This parameter is used repeatedly - compute once here
424:         ima = np.eye(a.shape[0]) - alpha*dt*a
425:         ad = linalg.solve(ima, np.eye(a.shape[0]) + (1.0-alpha)*dt*a)
426:         bd = linalg.solve(ima, dt*b)
427: 
428:         # Similarly solve for the output equation matrices
429:         cd = linalg.solve(ima.transpose(), c.transpose())
430:         cd = cd.transpose()
431:         dd = d + alpha*np.dot(c, bd)
432: 
433:     elif method == 'bilinear' or method == 'tustin':
434:         return cont2discrete(system, dt, method="gbt", alpha=0.5)
435: 
436:     elif method == 'euler' or method == 'forward_diff':
437:         return cont2discrete(system, dt, method="gbt", alpha=0.0)
438: 
439:     elif method == 'backward_diff':
440:         return cont2discrete(system, dt, method="gbt", alpha=1.0)
441: 
442:     elif method == 'zoh':
443:         # Build an exponential matrix
444:         em_upper = np.hstack((a, b))
445: 
446:         # Need to stack zeros under the a and b matrices
447:         em_lower = np.hstack((np.zeros((b.shape[1], a.shape[0])),
448:                               np.zeros((b.shape[1], b.shape[1]))))
449: 
450:         em = np.vstack((em_upper, em_lower))
451:         ms = linalg.expm(dt * em)
452: 
453:         # Dispose of the lower rows
454:         ms = ms[:a.shape[0], :]
455: 
456:         ad = ms[:, 0:a.shape[1]]
457:         bd = ms[:, a.shape[1]:]
458: 
459:         cd = c
460:         dd = d
461: 
462:     else:
463:         raise ValueError("Unknown transformation method '%s'" % method)
464: 
465:     return ad, bd, cd, dd, dt
466: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_273028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nltisys -- a collection of functions to convert linear time invariant systems\nfrom one representation to another.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_273029 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_273029) is not StypyTypeError):

    if (import_273029 != 'pyd_module'):
        __import__(import_273029)
        sys_modules_273030 = sys.modules[import_273029]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', sys_modules_273030.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_273029)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_273031 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_273031) is not StypyTypeError):

    if (import_273031 != 'pyd_module'):
        __import__(import_273031)
        sys_modules_273032 = sys.modules[import_273031]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_273032.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_273031)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy import r_, eye, atleast_2d, poly, dot, asarray, product, zeros, array, outer' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_273033 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_273033) is not StypyTypeError):

    if (import_273033 != 'pyd_module'):
        __import__(import_273033)
        sys_modules_273034 = sys.modules[import_273033]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', sys_modules_273034.module_type_store, module_type_store, ['r_', 'eye', 'atleast_2d', 'poly', 'dot', 'asarray', 'product', 'zeros', 'array', 'outer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_273034, sys_modules_273034.module_type_store, module_type_store)
    else:
        from numpy import r_, eye, atleast_2d, poly, dot, asarray, product, zeros, array, outer

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', None, module_type_store, ['r_', 'eye', 'atleast_2d', 'poly', 'dot', 'asarray', 'product', 'zeros', 'array', 'outer'], [r_, eye, atleast_2d, poly, dot, asarray, product, zeros, array, outer])

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_273033)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy import linalg' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_273035 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy')

if (type(import_273035) is not StypyTypeError):

    if (import_273035 != 'pyd_module'):
        __import__(import_273035)
        sys_modules_273036 = sys.modules[import_273035]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', sys_modules_273036.module_type_store, module_type_store, ['linalg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_273036, sys_modules_273036.module_type_store, module_type_store)
    else:
        from scipy import linalg

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', None, module_type_store, ['linalg'], [linalg])

else:
    # Assigning a type to the variable 'scipy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', import_273035)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.signal.filter_design import tf2zpk, zpk2tf, normalize' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_273037 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.signal.filter_design')

if (type(import_273037) is not StypyTypeError):

    if (import_273037 != 'pyd_module'):
        __import__(import_273037)
        sys_modules_273038 = sys.modules[import_273037]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.signal.filter_design', sys_modules_273038.module_type_store, module_type_store, ['tf2zpk', 'zpk2tf', 'normalize'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_273038, sys_modules_273038.module_type_store, module_type_store)
    else:
        from scipy.signal.filter_design import tf2zpk, zpk2tf, normalize

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.signal.filter_design', None, module_type_store, ['tf2zpk', 'zpk2tf', 'normalize'], [tf2zpk, zpk2tf, normalize])

else:
    # Assigning a type to the variable 'scipy.signal.filter_design' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.signal.filter_design', import_273037)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a List to a Name (line 16):

# Assigning a List to a Name (line 16):
__all__ = ['tf2ss', 'abcd_normalize', 'ss2tf', 'zpk2ss', 'ss2zpk', 'cont2discrete']
module_type_store.set_exportable_members(['tf2ss', 'abcd_normalize', 'ss2tf', 'zpk2ss', 'ss2zpk', 'cont2discrete'])

# Obtaining an instance of the builtin type 'list' (line 16)
list_273039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_273040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'tf2ss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_273039, str_273040)
# Adding element type (line 16)
str_273041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'str', 'abcd_normalize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_273039, str_273041)
# Adding element type (line 16)
str_273042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 38), 'str', 'ss2tf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_273039, str_273042)
# Adding element type (line 16)
str_273043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 47), 'str', 'zpk2ss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_273039, str_273043)
# Adding element type (line 16)
str_273044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 57), 'str', 'ss2zpk')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_273039, str_273044)
# Adding element type (line 16)
str_273045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'cont2discrete')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_273039, str_273045)

# Assigning a type to the variable '__all__' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__all__', list_273039)

@norecursion
def tf2ss(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tf2ss'
    module_type_store = module_type_store.open_function_context('tf2ss', 20, 0, False)
    
    # Passed parameters checking function
    tf2ss.stypy_localization = localization
    tf2ss.stypy_type_of_self = None
    tf2ss.stypy_type_store = module_type_store
    tf2ss.stypy_function_name = 'tf2ss'
    tf2ss.stypy_param_names_list = ['num', 'den']
    tf2ss.stypy_varargs_param_name = None
    tf2ss.stypy_kwargs_param_name = None
    tf2ss.stypy_call_defaults = defaults
    tf2ss.stypy_call_varargs = varargs
    tf2ss.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tf2ss', ['num', 'den'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tf2ss', localization, ['num', 'den'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tf2ss(...)' code ##################

    str_273046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', 'Transfer function to state-space representation.\n\n    Parameters\n    ----------\n    num, den : array_like\n        Sequences representing the coefficients of the numerator and\n        denominator polynomials, in order of descending degree. The\n        denominator needs to be at least as long as the numerator.\n\n    Returns\n    -------\n    A, B, C, D : ndarray\n        State space representation of the system, in controller canonical\n        form.\n\n    Examples\n    --------\n    Convert the transfer function:\n\n    .. math:: H(s) = \\frac{s^2 + 3s + 3}{s^2 + 2s + 1}\n\n    >>> num = [1, 3, 3]\n    >>> den = [1, 2, 1]\n\n    to the state-space representation:\n\n    .. math::\n\n        \\dot{\\textbf{x}}(t) =\n        \\begin{bmatrix} -2 & -1 \\\\ 1 & 0 \\end{bmatrix} \\textbf{x}(t) +\n        \\begin{bmatrix} 1 \\\\ 0 \\end{bmatrix} \\textbf{u}(t) \\\\\n\n        \\textbf{y}(t) = \\begin{bmatrix} 1 & 2 \\end{bmatrix} \\textbf{x}(t) +\n        \\begin{bmatrix} 1 \\end{bmatrix} \\textbf{u}(t)\n\n    >>> from scipy.signal import tf2ss\n    >>> A, B, C, D = tf2ss(num, den)\n    >>> A\n    array([[-2., -1.],\n           [ 1.,  0.]])\n    >>> B\n    array([[ 1.],\n           [ 0.]])\n    >>> C\n    array([[ 1.,  2.]])\n    >>> D\n    array([[ 1.]])\n    ')
    
    # Assigning a Call to a Tuple (line 76):
    
    # Assigning a Subscript to a Name (line 76):
    
    # Obtaining the type of the subscript
    int_273047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'int')
    
    # Call to normalize(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'num' (line 76)
    num_273049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'num', False)
    # Getting the type of 'den' (line 76)
    den_273050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'den', False)
    # Processing the call keyword arguments (line 76)
    kwargs_273051 = {}
    # Getting the type of 'normalize' (line 76)
    normalize_273048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'normalize', False)
    # Calling normalize(args, kwargs) (line 76)
    normalize_call_result_273052 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), normalize_273048, *[num_273049, den_273050], **kwargs_273051)
    
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___273053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 4), normalize_call_result_273052, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_273054 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), getitem___273053, int_273047)
    
    # Assigning a type to the variable 'tuple_var_assignment_273000' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_273000', subscript_call_result_273054)
    
    # Assigning a Subscript to a Name (line 76):
    
    # Obtaining the type of the subscript
    int_273055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'int')
    
    # Call to normalize(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'num' (line 76)
    num_273057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'num', False)
    # Getting the type of 'den' (line 76)
    den_273058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'den', False)
    # Processing the call keyword arguments (line 76)
    kwargs_273059 = {}
    # Getting the type of 'normalize' (line 76)
    normalize_273056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'normalize', False)
    # Calling normalize(args, kwargs) (line 76)
    normalize_call_result_273060 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), normalize_273056, *[num_273057, den_273058], **kwargs_273059)
    
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___273061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 4), normalize_call_result_273060, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_273062 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), getitem___273061, int_273055)
    
    # Assigning a type to the variable 'tuple_var_assignment_273001' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_273001', subscript_call_result_273062)
    
    # Assigning a Name to a Name (line 76):
    # Getting the type of 'tuple_var_assignment_273000' (line 76)
    tuple_var_assignment_273000_273063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_273000')
    # Assigning a type to the variable 'num' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'num', tuple_var_assignment_273000_273063)
    
    # Assigning a Name to a Name (line 76):
    # Getting the type of 'tuple_var_assignment_273001' (line 76)
    tuple_var_assignment_273001_273064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'tuple_var_assignment_273001')
    # Assigning a type to the variable 'den' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 9), 'den', tuple_var_assignment_273001_273064)
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to len(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'num' (line 77)
    num_273066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'num', False)
    # Obtaining the member 'shape' of a type (line 77)
    shape_273067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), num_273066, 'shape')
    # Processing the call keyword arguments (line 77)
    kwargs_273068 = {}
    # Getting the type of 'len' (line 77)
    len_273065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'len', False)
    # Calling len(args, kwargs) (line 77)
    len_call_result_273069 = invoke(stypy.reporting.localization.Localization(__file__, 77, 9), len_273065, *[shape_273067], **kwargs_273068)
    
    # Assigning a type to the variable 'nn' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'nn', len_call_result_273069)
    
    
    # Getting the type of 'nn' (line 78)
    nn_273070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 7), 'nn')
    int_273071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 13), 'int')
    # Applying the binary operator '==' (line 78)
    result_eq_273072 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 7), '==', nn_273070, int_273071)
    
    # Testing the type of an if condition (line 78)
    if_condition_273073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 4), result_eq_273072)
    # Assigning a type to the variable 'if_condition_273073' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'if_condition_273073', if_condition_273073)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to asarray(...): (line 79)
    # Processing the call arguments (line 79)
    
    # Obtaining an instance of the builtin type 'list' (line 79)
    list_273075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 79)
    # Adding element type (line 79)
    # Getting the type of 'num' (line 79)
    num_273076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'num', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 22), list_273075, num_273076)
    
    # Getting the type of 'num' (line 79)
    num_273077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'num', False)
    # Obtaining the member 'dtype' of a type (line 79)
    dtype_273078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 29), num_273077, 'dtype')
    # Processing the call keyword arguments (line 79)
    kwargs_273079 = {}
    # Getting the type of 'asarray' (line 79)
    asarray_273074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 14), 'asarray', False)
    # Calling asarray(args, kwargs) (line 79)
    asarray_call_result_273080 = invoke(stypy.reporting.localization.Localization(__file__, 79, 14), asarray_273074, *[list_273075, dtype_273078], **kwargs_273079)
    
    # Assigning a type to the variable 'num' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'num', asarray_call_result_273080)
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 80):
    
    # Assigning a Subscript to a Name (line 80):
    
    # Obtaining the type of the subscript
    int_273081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 18), 'int')
    # Getting the type of 'num' (line 80)
    num_273082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'num')
    # Obtaining the member 'shape' of a type (line 80)
    shape_273083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), num_273082, 'shape')
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___273084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), shape_273083, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_273085 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), getitem___273084, int_273081)
    
    # Assigning a type to the variable 'M' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'M', subscript_call_result_273085)
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to len(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'den' (line 81)
    den_273087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'den', False)
    # Processing the call keyword arguments (line 81)
    kwargs_273088 = {}
    # Getting the type of 'len' (line 81)
    len_273086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'len', False)
    # Calling len(args, kwargs) (line 81)
    len_call_result_273089 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), len_273086, *[den_273087], **kwargs_273088)
    
    # Assigning a type to the variable 'K' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'K', len_call_result_273089)
    
    
    # Getting the type of 'M' (line 82)
    M_273090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 7), 'M')
    # Getting the type of 'K' (line 82)
    K_273091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'K')
    # Applying the binary operator '>' (line 82)
    result_gt_273092 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 7), '>', M_273090, K_273091)
    
    # Testing the type of an if condition (line 82)
    if_condition_273093 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 4), result_gt_273092)
    # Assigning a type to the variable 'if_condition_273093' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'if_condition_273093', if_condition_273093)
    # SSA begins for if statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 83):
    
    # Assigning a Str to a Name (line 83):
    str_273094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 14), 'str', 'Improper transfer function. `num` is longer than `den`.')
    # Assigning a type to the variable 'msg' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'msg', str_273094)
    
    # Call to ValueError(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'msg' (line 84)
    msg_273096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'msg', False)
    # Processing the call keyword arguments (line 84)
    kwargs_273097 = {}
    # Getting the type of 'ValueError' (line 84)
    ValueError_273095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 84)
    ValueError_call_result_273098 = invoke(stypy.reporting.localization.Localization(__file__, 84, 14), ValueError_273095, *[msg_273096], **kwargs_273097)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 84, 8), ValueError_call_result_273098, 'raise parameter', BaseException)
    # SSA join for if statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'M' (line 85)
    M_273099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 7), 'M')
    int_273100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'int')
    # Applying the binary operator '==' (line 85)
    result_eq_273101 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 7), '==', M_273099, int_273100)
    
    
    # Getting the type of 'K' (line 85)
    K_273102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'K')
    int_273103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'int')
    # Applying the binary operator '==' (line 85)
    result_eq_273104 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 17), '==', K_273102, int_273103)
    
    # Applying the binary operator 'or' (line 85)
    result_or_keyword_273105 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 7), 'or', result_eq_273101, result_eq_273104)
    
    # Testing the type of an if condition (line 85)
    if_condition_273106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 4), result_or_keyword_273105)
    # Assigning a type to the variable 'if_condition_273106' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'if_condition_273106', if_condition_273106)
    # SSA begins for if statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 86)
    tuple_273107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 86)
    # Adding element type (line 86)
    
    # Call to array(...): (line 86)
    # Processing the call arguments (line 86)
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_273109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    
    # Getting the type of 'float' (line 86)
    float_273110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 26), 'float', False)
    # Processing the call keyword arguments (line 86)
    kwargs_273111 = {}
    # Getting the type of 'array' (line 86)
    array_273108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'array', False)
    # Calling array(args, kwargs) (line 86)
    array_call_result_273112 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), array_273108, *[list_273109, float_273110], **kwargs_273111)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 16), tuple_273107, array_call_result_273112)
    # Adding element type (line 86)
    
    # Call to array(...): (line 86)
    # Processing the call arguments (line 86)
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_273114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    
    # Getting the type of 'float' (line 86)
    float_273115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 44), 'float', False)
    # Processing the call keyword arguments (line 86)
    kwargs_273116 = {}
    # Getting the type of 'array' (line 86)
    array_273113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 34), 'array', False)
    # Calling array(args, kwargs) (line 86)
    array_call_result_273117 = invoke(stypy.reporting.localization.Localization(__file__, 86, 34), array_273113, *[list_273114, float_273115], **kwargs_273116)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 16), tuple_273107, array_call_result_273117)
    # Adding element type (line 86)
    
    # Call to array(...): (line 86)
    # Processing the call arguments (line 86)
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_273119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 58), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    
    # Getting the type of 'float' (line 86)
    float_273120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 62), 'float', False)
    # Processing the call keyword arguments (line 86)
    kwargs_273121 = {}
    # Getting the type of 'array' (line 86)
    array_273118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 52), 'array', False)
    # Calling array(args, kwargs) (line 86)
    array_call_result_273122 = invoke(stypy.reporting.localization.Localization(__file__, 86, 52), array_273118, *[list_273119, float_273120], **kwargs_273121)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 16), tuple_273107, array_call_result_273122)
    # Adding element type (line 86)
    
    # Call to array(...): (line 87)
    # Processing the call arguments (line 87)
    
    # Obtaining an instance of the builtin type 'list' (line 87)
    list_273124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 87)
    
    # Getting the type of 'float' (line 87)
    float_273125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'float', False)
    # Processing the call keyword arguments (line 87)
    kwargs_273126 = {}
    # Getting the type of 'array' (line 87)
    array_273123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'array', False)
    # Calling array(args, kwargs) (line 87)
    array_call_result_273127 = invoke(stypy.reporting.localization.Localization(__file__, 87, 16), array_273123, *[list_273124, float_273125], **kwargs_273126)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 16), tuple_273107, array_call_result_273127)
    
    # Assigning a type to the variable 'stypy_return_type' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', tuple_273107)
    # SSA join for if statement (line 85)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 90):
    
    # Assigning a Subscript to a Name (line 90):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 90)
    tuple_273128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 90)
    # Adding element type (line 90)
    str_273129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 13), 'str', '-1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 13), tuple_273128, str_273129)
    # Adding element type (line 90)
    
    # Call to zeros(...): (line 90)
    # Processing the call arguments (line 90)
    
    # Obtaining an instance of the builtin type 'tuple' (line 90)
    tuple_273131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 90)
    # Adding element type (line 90)
    
    # Obtaining the type of the subscript
    int_273132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 36), 'int')
    # Getting the type of 'num' (line 90)
    num_273133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'num', False)
    # Obtaining the member 'shape' of a type (line 90)
    shape_273134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 26), num_273133, 'shape')
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___273135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 26), shape_273134, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_273136 = invoke(stypy.reporting.localization.Localization(__file__, 90, 26), getitem___273135, int_273132)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 26), tuple_273131, subscript_call_result_273136)
    # Adding element type (line 90)
    # Getting the type of 'K' (line 90)
    K_273137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 40), 'K', False)
    # Getting the type of 'M' (line 90)
    M_273138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 44), 'M', False)
    # Applying the binary operator '-' (line 90)
    result_sub_273139 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 40), '-', K_273137, M_273138)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 26), tuple_273131, result_sub_273139)
    
    # Getting the type of 'num' (line 90)
    num_273140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 48), 'num', False)
    # Obtaining the member 'dtype' of a type (line 90)
    dtype_273141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 48), num_273140, 'dtype')
    # Processing the call keyword arguments (line 90)
    kwargs_273142 = {}
    # Getting the type of 'zeros' (line 90)
    zeros_273130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'zeros', False)
    # Calling zeros(args, kwargs) (line 90)
    zeros_call_result_273143 = invoke(stypy.reporting.localization.Localization(__file__, 90, 19), zeros_273130, *[tuple_273131, dtype_273141], **kwargs_273142)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 13), tuple_273128, zeros_call_result_273143)
    # Adding element type (line 90)
    # Getting the type of 'num' (line 90)
    num_273144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 60), 'num')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 13), tuple_273128, num_273144)
    
    # Getting the type of 'r_' (line 90)
    r__273145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 10), 'r_')
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___273146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 10), r__273145, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_273147 = invoke(stypy.reporting.localization.Localization(__file__, 90, 10), getitem___273146, tuple_273128)
    
    # Assigning a type to the variable 'num' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'num', subscript_call_result_273147)
    
    
    
    # Obtaining the type of the subscript
    int_273148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 17), 'int')
    # Getting the type of 'num' (line 92)
    num_273149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 7), 'num')
    # Obtaining the member 'shape' of a type (line 92)
    shape_273150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 7), num_273149, 'shape')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___273151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 7), shape_273150, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_273152 = invoke(stypy.reporting.localization.Localization(__file__, 92, 7), getitem___273151, int_273148)
    
    int_273153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 23), 'int')
    # Applying the binary operator '>' (line 92)
    result_gt_273154 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 7), '>', subscript_call_result_273152, int_273153)
    
    # Testing the type of an if condition (line 92)
    if_condition_273155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 4), result_gt_273154)
    # Assigning a type to the variable 'if_condition_273155' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'if_condition_273155', if_condition_273155)
    # SSA begins for if statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 93):
    
    # Assigning a Call to a Name (line 93):
    
    # Call to atleast_2d(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Obtaining the type of the subscript
    slice_273157 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 93, 23), None, None, None)
    int_273158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 30), 'int')
    # Getting the type of 'num' (line 93)
    num_273159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'num', False)
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___273160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 23), num_273159, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_273161 = invoke(stypy.reporting.localization.Localization(__file__, 93, 23), getitem___273160, (slice_273157, int_273158))
    
    # Processing the call keyword arguments (line 93)
    kwargs_273162 = {}
    # Getting the type of 'atleast_2d' (line 93)
    atleast_2d_273156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'atleast_2d', False)
    # Calling atleast_2d(args, kwargs) (line 93)
    atleast_2d_call_result_273163 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), atleast_2d_273156, *[subscript_call_result_273161], **kwargs_273162)
    
    # Assigning a type to the variable 'D' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'D', atleast_2d_call_result_273163)
    # SSA branch for the else part of an if statement (line 92)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 100):
    
    # Assigning a Call to a Name (line 100):
    
    # Call to array(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_273165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    # Adding element type (line 100)
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_273166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    # Adding element type (line 100)
    int_273167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 19), list_273166, int_273167)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 18), list_273165, list_273166)
    
    # Getting the type of 'float' (line 100)
    float_273168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'float', False)
    # Processing the call keyword arguments (line 100)
    kwargs_273169 = {}
    # Getting the type of 'array' (line 100)
    array_273164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'array', False)
    # Calling array(args, kwargs) (line 100)
    array_call_result_273170 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), array_273164, *[list_273165, float_273168], **kwargs_273169)
    
    # Assigning a type to the variable 'D' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'D', array_call_result_273170)
    # SSA join for if statement (line 92)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'K' (line 102)
    K_273171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 7), 'K')
    int_273172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 12), 'int')
    # Applying the binary operator '==' (line 102)
    result_eq_273173 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 7), '==', K_273171, int_273172)
    
    # Testing the type of an if condition (line 102)
    if_condition_273174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 4), result_eq_273173)
    # Assigning a type to the variable 'if_condition_273174' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'if_condition_273174', if_condition_273174)
    # SSA begins for if statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to reshape(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'num' (line 103)
    num_273177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'num', False)
    # Obtaining the member 'shape' of a type (line 103)
    shape_273178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 22), num_273177, 'shape')
    # Processing the call keyword arguments (line 103)
    kwargs_273179 = {}
    # Getting the type of 'D' (line 103)
    D_273175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'D', False)
    # Obtaining the member 'reshape' of a type (line 103)
    reshape_273176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), D_273175, 'reshape')
    # Calling reshape(args, kwargs) (line 103)
    reshape_call_result_273180 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), reshape_273176, *[shape_273178], **kwargs_273179)
    
    # Assigning a type to the variable 'D' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'D', reshape_call_result_273180)
    
    # Obtaining an instance of the builtin type 'tuple' (line 105)
    tuple_273181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 105)
    # Adding element type (line 105)
    
    # Call to zeros(...): (line 105)
    # Processing the call arguments (line 105)
    
    # Obtaining an instance of the builtin type 'tuple' (line 105)
    tuple_273183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 105)
    # Adding element type (line 105)
    int_273184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 23), tuple_273183, int_273184)
    # Adding element type (line 105)
    int_273185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 23), tuple_273183, int_273185)
    
    # Processing the call keyword arguments (line 105)
    kwargs_273186 = {}
    # Getting the type of 'zeros' (line 105)
    zeros_273182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'zeros', False)
    # Calling zeros(args, kwargs) (line 105)
    zeros_call_result_273187 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), zeros_273182, *[tuple_273183], **kwargs_273186)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 16), tuple_273181, zeros_call_result_273187)
    # Adding element type (line 105)
    
    # Call to zeros(...): (line 105)
    # Processing the call arguments (line 105)
    
    # Obtaining an instance of the builtin type 'tuple' (line 105)
    tuple_273189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 105)
    # Adding element type (line 105)
    int_273190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 38), tuple_273189, int_273190)
    # Adding element type (line 105)
    
    # Obtaining the type of the subscript
    int_273191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 49), 'int')
    # Getting the type of 'D' (line 105)
    D_273192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 41), 'D', False)
    # Obtaining the member 'shape' of a type (line 105)
    shape_273193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 41), D_273192, 'shape')
    # Obtaining the member '__getitem__' of a type (line 105)
    getitem___273194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 41), shape_273193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 105)
    subscript_call_result_273195 = invoke(stypy.reporting.localization.Localization(__file__, 105, 41), getitem___273194, int_273191)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 38), tuple_273189, subscript_call_result_273195)
    
    # Processing the call keyword arguments (line 105)
    kwargs_273196 = {}
    # Getting the type of 'zeros' (line 105)
    zeros_273188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'zeros', False)
    # Calling zeros(args, kwargs) (line 105)
    zeros_call_result_273197 = invoke(stypy.reporting.localization.Localization(__file__, 105, 31), zeros_273188, *[tuple_273189], **kwargs_273196)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 16), tuple_273181, zeros_call_result_273197)
    # Adding element type (line 105)
    
    # Call to zeros(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Obtaining an instance of the builtin type 'tuple' (line 106)
    tuple_273199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 106)
    # Adding element type (line 106)
    
    # Obtaining the type of the subscript
    int_273200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 31), 'int')
    # Getting the type of 'D' (line 106)
    D_273201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'D', False)
    # Obtaining the member 'shape' of a type (line 106)
    shape_273202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 23), D_273201, 'shape')
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___273203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 23), shape_273202, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_273204 = invoke(stypy.reporting.localization.Localization(__file__, 106, 23), getitem___273203, int_273200)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 23), tuple_273199, subscript_call_result_273204)
    # Adding element type (line 106)
    int_273205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 23), tuple_273199, int_273205)
    
    # Processing the call keyword arguments (line 106)
    kwargs_273206 = {}
    # Getting the type of 'zeros' (line 106)
    zeros_273198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'zeros', False)
    # Calling zeros(args, kwargs) (line 106)
    zeros_call_result_273207 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), zeros_273198, *[tuple_273199], **kwargs_273206)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 16), tuple_273181, zeros_call_result_273207)
    # Adding element type (line 105)
    # Getting the type of 'D' (line 106)
    D_273208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'D')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 16), tuple_273181, D_273208)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', tuple_273181)
    # SSA join for if statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a UnaryOp to a Name (line 108):
    
    # Assigning a UnaryOp to a Name (line 108):
    
    
    # Call to array(...): (line 108)
    # Processing the call arguments (line 108)
    
    # Obtaining an instance of the builtin type 'list' (line 108)
    list_273210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 108)
    # Adding element type (line 108)
    
    # Obtaining the type of the subscript
    int_273211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 23), 'int')
    slice_273212 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 108, 19), int_273211, None, None)
    # Getting the type of 'den' (line 108)
    den_273213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'den', False)
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___273214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 19), den_273213, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_273215 = invoke(stypy.reporting.localization.Localization(__file__, 108, 19), getitem___273214, slice_273212)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 18), list_273210, subscript_call_result_273215)
    
    # Processing the call keyword arguments (line 108)
    kwargs_273216 = {}
    # Getting the type of 'array' (line 108)
    array_273209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'array', False)
    # Calling array(args, kwargs) (line 108)
    array_call_result_273217 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), array_273209, *[list_273210], **kwargs_273216)
    
    # Applying the 'usub' unary operator (line 108)
    result___neg___273218 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 11), 'usub', array_call_result_273217)
    
    # Assigning a type to the variable 'frow' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'frow', result___neg___273218)
    
    # Assigning a Subscript to a Name (line 109):
    
    # Assigning a Subscript to a Name (line 109):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 109)
    tuple_273219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 109)
    # Adding element type (line 109)
    # Getting the type of 'frow' (line 109)
    frow_273220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'frow')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 11), tuple_273219, frow_273220)
    # Adding element type (line 109)
    
    # Call to eye(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'K' (line 109)
    K_273222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'K', False)
    int_273223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'int')
    # Applying the binary operator '-' (line 109)
    result_sub_273224 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 21), '-', K_273222, int_273223)
    
    # Getting the type of 'K' (line 109)
    K_273225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'K', False)
    int_273226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 32), 'int')
    # Applying the binary operator '-' (line 109)
    result_sub_273227 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 28), '-', K_273225, int_273226)
    
    # Processing the call keyword arguments (line 109)
    kwargs_273228 = {}
    # Getting the type of 'eye' (line 109)
    eye_273221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'eye', False)
    # Calling eye(args, kwargs) (line 109)
    eye_call_result_273229 = invoke(stypy.reporting.localization.Localization(__file__, 109, 17), eye_273221, *[result_sub_273224, result_sub_273227], **kwargs_273228)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 11), tuple_273219, eye_call_result_273229)
    
    # Getting the type of 'r_' (line 109)
    r__273230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'r_')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___273231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), r__273230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_273232 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), getitem___273231, tuple_273219)
    
    # Assigning a type to the variable 'A' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'A', subscript_call_result_273232)
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to eye(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'K' (line 110)
    K_273234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'K', False)
    int_273235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 16), 'int')
    # Applying the binary operator '-' (line 110)
    result_sub_273236 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 12), '-', K_273234, int_273235)
    
    int_273237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 19), 'int')
    # Processing the call keyword arguments (line 110)
    kwargs_273238 = {}
    # Getting the type of 'eye' (line 110)
    eye_273233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'eye', False)
    # Calling eye(args, kwargs) (line 110)
    eye_call_result_273239 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), eye_273233, *[result_sub_273236, int_273237], **kwargs_273238)
    
    # Assigning a type to the variable 'B' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'B', eye_call_result_273239)
    
    # Assigning a BinOp to a Name (line 111):
    
    # Assigning a BinOp to a Name (line 111):
    
    # Obtaining the type of the subscript
    slice_273240 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 111, 8), None, None, None)
    int_273241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 15), 'int')
    slice_273242 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 111, 8), int_273241, None, None)
    # Getting the type of 'num' (line 111)
    num_273243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'num')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___273244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), num_273243, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_273245 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), getitem___273244, (slice_273240, slice_273242))
    
    
    # Call to outer(...): (line 111)
    # Processing the call arguments (line 111)
    
    # Obtaining the type of the subscript
    slice_273247 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 111, 27), None, None, None)
    int_273248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 34), 'int')
    # Getting the type of 'num' (line 111)
    num_273249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'num', False)
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___273250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 27), num_273249, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_273251 = invoke(stypy.reporting.localization.Localization(__file__, 111, 27), getitem___273250, (slice_273247, int_273248))
    
    
    # Obtaining the type of the subscript
    int_273252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 42), 'int')
    slice_273253 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 111, 38), int_273252, None, None)
    # Getting the type of 'den' (line 111)
    den_273254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), 'den', False)
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___273255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 38), den_273254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_273256 = invoke(stypy.reporting.localization.Localization(__file__, 111, 38), getitem___273255, slice_273253)
    
    # Processing the call keyword arguments (line 111)
    kwargs_273257 = {}
    # Getting the type of 'outer' (line 111)
    outer_273246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'outer', False)
    # Calling outer(args, kwargs) (line 111)
    outer_call_result_273258 = invoke(stypy.reporting.localization.Localization(__file__, 111, 21), outer_273246, *[subscript_call_result_273251, subscript_call_result_273256], **kwargs_273257)
    
    # Applying the binary operator '-' (line 111)
    result_sub_273259 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 8), '-', subscript_call_result_273245, outer_call_result_273258)
    
    # Assigning a type to the variable 'C' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'C', result_sub_273259)
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to reshape(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_273262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    
    # Obtaining the type of the subscript
    int_273263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 27), 'int')
    # Getting the type of 'C' (line 112)
    C_273264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'C', False)
    # Obtaining the member 'shape' of a type (line 112)
    shape_273265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 19), C_273264, 'shape')
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___273266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 19), shape_273265, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_273267 = invoke(stypy.reporting.localization.Localization(__file__, 112, 19), getitem___273266, int_273263)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 19), tuple_273262, subscript_call_result_273267)
    # Adding element type (line 112)
    
    # Obtaining the type of the subscript
    int_273268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 39), 'int')
    # Getting the type of 'B' (line 112)
    B_273269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'B', False)
    # Obtaining the member 'shape' of a type (line 112)
    shape_273270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 31), B_273269, 'shape')
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___273271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 31), shape_273270, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_273272 = invoke(stypy.reporting.localization.Localization(__file__, 112, 31), getitem___273271, int_273268)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 19), tuple_273262, subscript_call_result_273272)
    
    # Processing the call keyword arguments (line 112)
    kwargs_273273 = {}
    # Getting the type of 'D' (line 112)
    D_273260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'D', False)
    # Obtaining the member 'reshape' of a type (line 112)
    reshape_273261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), D_273260, 'reshape')
    # Calling reshape(args, kwargs) (line 112)
    reshape_call_result_273274 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), reshape_273261, *[tuple_273262], **kwargs_273273)
    
    # Assigning a type to the variable 'D' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'D', reshape_call_result_273274)
    
    # Obtaining an instance of the builtin type 'tuple' (line 114)
    tuple_273275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 114)
    # Adding element type (line 114)
    # Getting the type of 'A' (line 114)
    A_273276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 11), tuple_273275, A_273276)
    # Adding element type (line 114)
    # Getting the type of 'B' (line 114)
    B_273277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 11), tuple_273275, B_273277)
    # Adding element type (line 114)
    # Getting the type of 'C' (line 114)
    C_273278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'C')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 11), tuple_273275, C_273278)
    # Adding element type (line 114)
    # Getting the type of 'D' (line 114)
    D_273279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'D')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 11), tuple_273275, D_273279)
    
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type', tuple_273275)
    
    # ################# End of 'tf2ss(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tf2ss' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_273280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273280)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tf2ss'
    return stypy_return_type_273280

# Assigning a type to the variable 'tf2ss' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'tf2ss', tf2ss)

@norecursion
def _none_to_empty_2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_none_to_empty_2d'
    module_type_store = module_type_store.open_function_context('_none_to_empty_2d', 117, 0, False)
    
    # Passed parameters checking function
    _none_to_empty_2d.stypy_localization = localization
    _none_to_empty_2d.stypy_type_of_self = None
    _none_to_empty_2d.stypy_type_store = module_type_store
    _none_to_empty_2d.stypy_function_name = '_none_to_empty_2d'
    _none_to_empty_2d.stypy_param_names_list = ['arg']
    _none_to_empty_2d.stypy_varargs_param_name = None
    _none_to_empty_2d.stypy_kwargs_param_name = None
    _none_to_empty_2d.stypy_call_defaults = defaults
    _none_to_empty_2d.stypy_call_varargs = varargs
    _none_to_empty_2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_none_to_empty_2d', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_none_to_empty_2d', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_none_to_empty_2d(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 118)
    # Getting the type of 'arg' (line 118)
    arg_273281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 7), 'arg')
    # Getting the type of 'None' (line 118)
    None_273282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'None')
    
    (may_be_273283, more_types_in_union_273284) = may_be_none(arg_273281, None_273282)

    if may_be_273283:

        if more_types_in_union_273284:
            # Runtime conditional SSA (line 118)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to zeros(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Obtaining an instance of the builtin type 'tuple' (line 119)
        tuple_273286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 119)
        # Adding element type (line 119)
        int_273287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 22), tuple_273286, int_273287)
        # Adding element type (line 119)
        int_273288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 22), tuple_273286, int_273288)
        
        # Processing the call keyword arguments (line 119)
        kwargs_273289 = {}
        # Getting the type of 'zeros' (line 119)
        zeros_273285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'zeros', False)
        # Calling zeros(args, kwargs) (line 119)
        zeros_call_result_273290 = invoke(stypy.reporting.localization.Localization(__file__, 119, 15), zeros_273285, *[tuple_273286], **kwargs_273289)
        
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'stypy_return_type', zeros_call_result_273290)

        if more_types_in_union_273284:
            # Runtime conditional SSA for else branch (line 118)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_273283) or more_types_in_union_273284):
        # Getting the type of 'arg' (line 121)
        arg_273291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'arg')
        # Assigning a type to the variable 'stypy_return_type' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', arg_273291)

        if (may_be_273283 and more_types_in_union_273284):
            # SSA join for if statement (line 118)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_none_to_empty_2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_none_to_empty_2d' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_273292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273292)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_none_to_empty_2d'
    return stypy_return_type_273292

# Assigning a type to the variable '_none_to_empty_2d' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), '_none_to_empty_2d', _none_to_empty_2d)

@norecursion
def _atleast_2d_or_none(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_atleast_2d_or_none'
    module_type_store = module_type_store.open_function_context('_atleast_2d_or_none', 124, 0, False)
    
    # Passed parameters checking function
    _atleast_2d_or_none.stypy_localization = localization
    _atleast_2d_or_none.stypy_type_of_self = None
    _atleast_2d_or_none.stypy_type_store = module_type_store
    _atleast_2d_or_none.stypy_function_name = '_atleast_2d_or_none'
    _atleast_2d_or_none.stypy_param_names_list = ['arg']
    _atleast_2d_or_none.stypy_varargs_param_name = None
    _atleast_2d_or_none.stypy_kwargs_param_name = None
    _atleast_2d_or_none.stypy_call_defaults = defaults
    _atleast_2d_or_none.stypy_call_varargs = varargs
    _atleast_2d_or_none.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_atleast_2d_or_none', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_atleast_2d_or_none', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_atleast_2d_or_none(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 125)
    # Getting the type of 'arg' (line 125)
    arg_273293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'arg')
    # Getting the type of 'None' (line 125)
    None_273294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 18), 'None')
    
    (may_be_273295, more_types_in_union_273296) = may_not_be_none(arg_273293, None_273294)

    if may_be_273295:

        if more_types_in_union_273296:
            # Runtime conditional SSA (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to atleast_2d(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'arg' (line 126)
        arg_273298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'arg', False)
        # Processing the call keyword arguments (line 126)
        kwargs_273299 = {}
        # Getting the type of 'atleast_2d' (line 126)
        atleast_2d_273297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'atleast_2d', False)
        # Calling atleast_2d(args, kwargs) (line 126)
        atleast_2d_call_result_273300 = invoke(stypy.reporting.localization.Localization(__file__, 126, 15), atleast_2d_273297, *[arg_273298], **kwargs_273299)
        
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', atleast_2d_call_result_273300)

        if more_types_in_union_273296:
            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_atleast_2d_or_none(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_atleast_2d_or_none' in the type store
    # Getting the type of 'stypy_return_type' (line 124)
    stypy_return_type_273301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273301)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_atleast_2d_or_none'
    return stypy_return_type_273301

# Assigning a type to the variable '_atleast_2d_or_none' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), '_atleast_2d_or_none', _atleast_2d_or_none)

@norecursion
def _shape_or_none(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_shape_or_none'
    module_type_store = module_type_store.open_function_context('_shape_or_none', 129, 0, False)
    
    # Passed parameters checking function
    _shape_or_none.stypy_localization = localization
    _shape_or_none.stypy_type_of_self = None
    _shape_or_none.stypy_type_store = module_type_store
    _shape_or_none.stypy_function_name = '_shape_or_none'
    _shape_or_none.stypy_param_names_list = ['M']
    _shape_or_none.stypy_varargs_param_name = None
    _shape_or_none.stypy_kwargs_param_name = None
    _shape_or_none.stypy_call_defaults = defaults
    _shape_or_none.stypy_call_varargs = varargs
    _shape_or_none.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_shape_or_none', ['M'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_shape_or_none', localization, ['M'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_shape_or_none(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 130)
    # Getting the type of 'M' (line 130)
    M_273302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'M')
    # Getting the type of 'None' (line 130)
    None_273303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'None')
    
    (may_be_273304, more_types_in_union_273305) = may_not_be_none(M_273302, None_273303)

    if may_be_273304:

        if more_types_in_union_273305:
            # Runtime conditional SSA (line 130)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'M' (line 131)
        M_273306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'M')
        # Obtaining the member 'shape' of a type (line 131)
        shape_273307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 15), M_273306, 'shape')
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', shape_273307)

        if more_types_in_union_273305:
            # Runtime conditional SSA for else branch (line 130)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_273304) or more_types_in_union_273305):
        
        # Obtaining an instance of the builtin type 'tuple' (line 133)
        tuple_273308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 133)
        # Adding element type (line 133)
        # Getting the type of 'None' (line 133)
        None_273309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 16), tuple_273308, None_273309)
        
        int_273310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'int')
        # Applying the binary operator '*' (line 133)
        result_mul_273311 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 15), '*', tuple_273308, int_273310)
        
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', result_mul_273311)

        if (may_be_273304 and more_types_in_union_273305):
            # SSA join for if statement (line 130)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_shape_or_none(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_shape_or_none' in the type store
    # Getting the type of 'stypy_return_type' (line 129)
    stypy_return_type_273312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273312)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_shape_or_none'
    return stypy_return_type_273312

# Assigning a type to the variable '_shape_or_none' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), '_shape_or_none', _shape_or_none)

@norecursion
def _choice_not_none(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_choice_not_none'
    module_type_store = module_type_store.open_function_context('_choice_not_none', 136, 0, False)
    
    # Passed parameters checking function
    _choice_not_none.stypy_localization = localization
    _choice_not_none.stypy_type_of_self = None
    _choice_not_none.stypy_type_store = module_type_store
    _choice_not_none.stypy_function_name = '_choice_not_none'
    _choice_not_none.stypy_param_names_list = []
    _choice_not_none.stypy_varargs_param_name = 'args'
    _choice_not_none.stypy_kwargs_param_name = None
    _choice_not_none.stypy_call_defaults = defaults
    _choice_not_none.stypy_call_varargs = varargs
    _choice_not_none.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_choice_not_none', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_choice_not_none', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_choice_not_none(...)' code ##################

    
    # Getting the type of 'args' (line 137)
    args_273313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'args')
    # Testing the type of a for loop iterable (line 137)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 4), args_273313)
    # Getting the type of the for loop variable (line 137)
    for_loop_var_273314 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 4), args_273313)
    # Assigning a type to the variable 'arg' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'arg', for_loop_var_273314)
    # SSA begins for a for statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 138)
    # Getting the type of 'arg' (line 138)
    arg_273315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'arg')
    # Getting the type of 'None' (line 138)
    None_273316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 22), 'None')
    
    (may_be_273317, more_types_in_union_273318) = may_not_be_none(arg_273315, None_273316)

    if may_be_273317:

        if more_types_in_union_273318:
            # Runtime conditional SSA (line 138)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'arg' (line 139)
        arg_273319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'arg')
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'stypy_return_type', arg_273319)

        if more_types_in_union_273318:
            # SSA join for if statement (line 138)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_choice_not_none(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_choice_not_none' in the type store
    # Getting the type of 'stypy_return_type' (line 136)
    stypy_return_type_273320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273320)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_choice_not_none'
    return stypy_return_type_273320

# Assigning a type to the variable '_choice_not_none' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), '_choice_not_none', _choice_not_none)

@norecursion
def _restore(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_restore'
    module_type_store = module_type_store.open_function_context('_restore', 142, 0, False)
    
    # Passed parameters checking function
    _restore.stypy_localization = localization
    _restore.stypy_type_of_self = None
    _restore.stypy_type_store = module_type_store
    _restore.stypy_function_name = '_restore'
    _restore.stypy_param_names_list = ['M', 'shape']
    _restore.stypy_varargs_param_name = None
    _restore.stypy_kwargs_param_name = None
    _restore.stypy_call_defaults = defaults
    _restore.stypy_call_varargs = varargs
    _restore.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_restore', ['M', 'shape'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_restore', localization, ['M', 'shape'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_restore(...)' code ##################

    
    
    # Getting the type of 'M' (line 143)
    M_273321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 7), 'M')
    # Obtaining the member 'shape' of a type (line 143)
    shape_273322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 7), M_273321, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 143)
    tuple_273323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 143)
    # Adding element type (line 143)
    int_273324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 19), tuple_273323, int_273324)
    # Adding element type (line 143)
    int_273325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 19), tuple_273323, int_273325)
    
    # Applying the binary operator '==' (line 143)
    result_eq_273326 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 7), '==', shape_273322, tuple_273323)
    
    # Testing the type of an if condition (line 143)
    if_condition_273327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 4), result_eq_273326)
    # Assigning a type to the variable 'if_condition_273327' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'if_condition_273327', if_condition_273327)
    # SSA begins for if statement (line 143)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to zeros(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'shape' (line 144)
    shape_273329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'shape', False)
    # Processing the call keyword arguments (line 144)
    kwargs_273330 = {}
    # Getting the type of 'zeros' (line 144)
    zeros_273328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'zeros', False)
    # Calling zeros(args, kwargs) (line 144)
    zeros_call_result_273331 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), zeros_273328, *[shape_273329], **kwargs_273330)
    
    # Assigning a type to the variable 'stypy_return_type' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'stypy_return_type', zeros_call_result_273331)
    # SSA branch for the else part of an if statement (line 143)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'M' (line 146)
    M_273332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'M')
    # Obtaining the member 'shape' of a type (line 146)
    shape_273333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 11), M_273332, 'shape')
    # Getting the type of 'shape' (line 146)
    shape_273334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 22), 'shape')
    # Applying the binary operator '!=' (line 146)
    result_ne_273335 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 11), '!=', shape_273333, shape_273334)
    
    # Testing the type of an if condition (line 146)
    if_condition_273336 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), result_ne_273335)
    # Assigning a type to the variable 'if_condition_273336' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_273336', if_condition_273336)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 147)
    # Processing the call arguments (line 147)
    str_273338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 29), 'str', 'The input arrays have incompatible shapes.')
    # Processing the call keyword arguments (line 147)
    kwargs_273339 = {}
    # Getting the type of 'ValueError' (line 147)
    ValueError_273337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 147)
    ValueError_call_result_273340 = invoke(stypy.reporting.localization.Localization(__file__, 147, 18), ValueError_273337, *[str_273338], **kwargs_273339)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 147, 12), ValueError_call_result_273340, 'raise parameter', BaseException)
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'M' (line 148)
    M_273341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'M')
    # Assigning a type to the variable 'stypy_return_type' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type', M_273341)
    # SSA join for if statement (line 143)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_restore(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_restore' in the type store
    # Getting the type of 'stypy_return_type' (line 142)
    stypy_return_type_273342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273342)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_restore'
    return stypy_return_type_273342

# Assigning a type to the variable '_restore' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), '_restore', _restore)

@norecursion
def abcd_normalize(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 151)
    None_273343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'None')
    # Getting the type of 'None' (line 151)
    None_273344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'None')
    # Getting the type of 'None' (line 151)
    None_273345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 37), 'None')
    # Getting the type of 'None' (line 151)
    None_273346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 45), 'None')
    defaults = [None_273343, None_273344, None_273345, None_273346]
    # Create a new context for function 'abcd_normalize'
    module_type_store = module_type_store.open_function_context('abcd_normalize', 151, 0, False)
    
    # Passed parameters checking function
    abcd_normalize.stypy_localization = localization
    abcd_normalize.stypy_type_of_self = None
    abcd_normalize.stypy_type_store = module_type_store
    abcd_normalize.stypy_function_name = 'abcd_normalize'
    abcd_normalize.stypy_param_names_list = ['A', 'B', 'C', 'D']
    abcd_normalize.stypy_varargs_param_name = None
    abcd_normalize.stypy_kwargs_param_name = None
    abcd_normalize.stypy_call_defaults = defaults
    abcd_normalize.stypy_call_varargs = varargs
    abcd_normalize.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'abcd_normalize', ['A', 'B', 'C', 'D'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'abcd_normalize', localization, ['A', 'B', 'C', 'D'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'abcd_normalize(...)' code ##################

    str_273347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, (-1)), 'str', 'Check state-space matrices and ensure they are two-dimensional.\n\n    If enough information on the system is provided, that is, enough\n    properly-shaped arrays are passed to the function, the missing ones\n    are built from this information, ensuring the correct number of\n    rows and columns. Otherwise a ValueError is raised.\n\n    Parameters\n    ----------\n    A, B, C, D : array_like, optional\n        State-space matrices. All of them are None (missing) by default.\n        See `ss2tf` for format.\n\n    Returns\n    -------\n    A, B, C, D : array\n        Properly shaped state-space matrices.\n\n    Raises\n    ------\n    ValueError\n        If not enough information on the system was provided.\n\n    ')
    
    # Assigning a Call to a Tuple (line 176):
    
    # Assigning a Subscript to a Name (line 176):
    
    # Obtaining the type of the subscript
    int_273348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 4), 'int')
    
    # Call to map(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of '_atleast_2d_or_none' (line 176)
    _atleast_2d_or_none_273350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), '_atleast_2d_or_none', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 176)
    tuple_273351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 176)
    # Adding element type (line 176)
    # Getting the type of 'A' (line 176)
    A_273352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273351, A_273352)
    # Adding element type (line 176)
    # Getting the type of 'B' (line 176)
    B_273353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 46), 'B', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273351, B_273353)
    # Adding element type (line 176)
    # Getting the type of 'C' (line 176)
    C_273354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 49), 'C', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273351, C_273354)
    # Adding element type (line 176)
    # Getting the type of 'D' (line 176)
    D_273355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 52), 'D', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273351, D_273355)
    
    # Processing the call keyword arguments (line 176)
    kwargs_273356 = {}
    # Getting the type of 'map' (line 176)
    map_273349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 17), 'map', False)
    # Calling map(args, kwargs) (line 176)
    map_call_result_273357 = invoke(stypy.reporting.localization.Localization(__file__, 176, 17), map_273349, *[_atleast_2d_or_none_273350, tuple_273351], **kwargs_273356)
    
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___273358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), map_call_result_273357, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_273359 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), getitem___273358, int_273348)
    
    # Assigning a type to the variable 'tuple_var_assignment_273002' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_273002', subscript_call_result_273359)
    
    # Assigning a Subscript to a Name (line 176):
    
    # Obtaining the type of the subscript
    int_273360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 4), 'int')
    
    # Call to map(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of '_atleast_2d_or_none' (line 176)
    _atleast_2d_or_none_273362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), '_atleast_2d_or_none', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 176)
    tuple_273363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 176)
    # Adding element type (line 176)
    # Getting the type of 'A' (line 176)
    A_273364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273363, A_273364)
    # Adding element type (line 176)
    # Getting the type of 'B' (line 176)
    B_273365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 46), 'B', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273363, B_273365)
    # Adding element type (line 176)
    # Getting the type of 'C' (line 176)
    C_273366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 49), 'C', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273363, C_273366)
    # Adding element type (line 176)
    # Getting the type of 'D' (line 176)
    D_273367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 52), 'D', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273363, D_273367)
    
    # Processing the call keyword arguments (line 176)
    kwargs_273368 = {}
    # Getting the type of 'map' (line 176)
    map_273361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 17), 'map', False)
    # Calling map(args, kwargs) (line 176)
    map_call_result_273369 = invoke(stypy.reporting.localization.Localization(__file__, 176, 17), map_273361, *[_atleast_2d_or_none_273362, tuple_273363], **kwargs_273368)
    
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___273370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), map_call_result_273369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_273371 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), getitem___273370, int_273360)
    
    # Assigning a type to the variable 'tuple_var_assignment_273003' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_273003', subscript_call_result_273371)
    
    # Assigning a Subscript to a Name (line 176):
    
    # Obtaining the type of the subscript
    int_273372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 4), 'int')
    
    # Call to map(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of '_atleast_2d_or_none' (line 176)
    _atleast_2d_or_none_273374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), '_atleast_2d_or_none', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 176)
    tuple_273375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 176)
    # Adding element type (line 176)
    # Getting the type of 'A' (line 176)
    A_273376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273375, A_273376)
    # Adding element type (line 176)
    # Getting the type of 'B' (line 176)
    B_273377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 46), 'B', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273375, B_273377)
    # Adding element type (line 176)
    # Getting the type of 'C' (line 176)
    C_273378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 49), 'C', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273375, C_273378)
    # Adding element type (line 176)
    # Getting the type of 'D' (line 176)
    D_273379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 52), 'D', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273375, D_273379)
    
    # Processing the call keyword arguments (line 176)
    kwargs_273380 = {}
    # Getting the type of 'map' (line 176)
    map_273373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 17), 'map', False)
    # Calling map(args, kwargs) (line 176)
    map_call_result_273381 = invoke(stypy.reporting.localization.Localization(__file__, 176, 17), map_273373, *[_atleast_2d_or_none_273374, tuple_273375], **kwargs_273380)
    
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___273382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), map_call_result_273381, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_273383 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), getitem___273382, int_273372)
    
    # Assigning a type to the variable 'tuple_var_assignment_273004' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_273004', subscript_call_result_273383)
    
    # Assigning a Subscript to a Name (line 176):
    
    # Obtaining the type of the subscript
    int_273384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 4), 'int')
    
    # Call to map(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of '_atleast_2d_or_none' (line 176)
    _atleast_2d_or_none_273386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), '_atleast_2d_or_none', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 176)
    tuple_273387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 176)
    # Adding element type (line 176)
    # Getting the type of 'A' (line 176)
    A_273388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273387, A_273388)
    # Adding element type (line 176)
    # Getting the type of 'B' (line 176)
    B_273389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 46), 'B', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273387, B_273389)
    # Adding element type (line 176)
    # Getting the type of 'C' (line 176)
    C_273390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 49), 'C', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273387, C_273390)
    # Adding element type (line 176)
    # Getting the type of 'D' (line 176)
    D_273391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 52), 'D', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), tuple_273387, D_273391)
    
    # Processing the call keyword arguments (line 176)
    kwargs_273392 = {}
    # Getting the type of 'map' (line 176)
    map_273385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 17), 'map', False)
    # Calling map(args, kwargs) (line 176)
    map_call_result_273393 = invoke(stypy.reporting.localization.Localization(__file__, 176, 17), map_273385, *[_atleast_2d_or_none_273386, tuple_273387], **kwargs_273392)
    
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___273394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), map_call_result_273393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_273395 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), getitem___273394, int_273384)
    
    # Assigning a type to the variable 'tuple_var_assignment_273005' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_273005', subscript_call_result_273395)
    
    # Assigning a Name to a Name (line 176):
    # Getting the type of 'tuple_var_assignment_273002' (line 176)
    tuple_var_assignment_273002_273396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_273002')
    # Assigning a type to the variable 'A' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'A', tuple_var_assignment_273002_273396)
    
    # Assigning a Name to a Name (line 176):
    # Getting the type of 'tuple_var_assignment_273003' (line 176)
    tuple_var_assignment_273003_273397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_273003')
    # Assigning a type to the variable 'B' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 7), 'B', tuple_var_assignment_273003_273397)
    
    # Assigning a Name to a Name (line 176):
    # Getting the type of 'tuple_var_assignment_273004' (line 176)
    tuple_var_assignment_273004_273398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_273004')
    # Assigning a type to the variable 'C' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 10), 'C', tuple_var_assignment_273004_273398)
    
    # Assigning a Name to a Name (line 176):
    # Getting the type of 'tuple_var_assignment_273005' (line 176)
    tuple_var_assignment_273005_273399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'tuple_var_assignment_273005')
    # Assigning a type to the variable 'D' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'D', tuple_var_assignment_273005_273399)
    
    # Assigning a Call to a Tuple (line 178):
    
    # Assigning a Subscript to a Name (line 178):
    
    # Obtaining the type of the subscript
    int_273400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 4), 'int')
    
    # Call to _shape_or_none(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'A' (line 178)
    A_273402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'A', False)
    # Processing the call keyword arguments (line 178)
    kwargs_273403 = {}
    # Getting the type of '_shape_or_none' (line 178)
    _shape_or_none_273401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), '_shape_or_none', False)
    # Calling _shape_or_none(args, kwargs) (line 178)
    _shape_or_none_call_result_273404 = invoke(stypy.reporting.localization.Localization(__file__, 178, 13), _shape_or_none_273401, *[A_273402], **kwargs_273403)
    
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___273405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 4), _shape_or_none_call_result_273404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_273406 = invoke(stypy.reporting.localization.Localization(__file__, 178, 4), getitem___273405, int_273400)
    
    # Assigning a type to the variable 'tuple_var_assignment_273006' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_var_assignment_273006', subscript_call_result_273406)
    
    # Assigning a Subscript to a Name (line 178):
    
    # Obtaining the type of the subscript
    int_273407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 4), 'int')
    
    # Call to _shape_or_none(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'A' (line 178)
    A_273409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'A', False)
    # Processing the call keyword arguments (line 178)
    kwargs_273410 = {}
    # Getting the type of '_shape_or_none' (line 178)
    _shape_or_none_273408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), '_shape_or_none', False)
    # Calling _shape_or_none(args, kwargs) (line 178)
    _shape_or_none_call_result_273411 = invoke(stypy.reporting.localization.Localization(__file__, 178, 13), _shape_or_none_273408, *[A_273409], **kwargs_273410)
    
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___273412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 4), _shape_or_none_call_result_273411, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_273413 = invoke(stypy.reporting.localization.Localization(__file__, 178, 4), getitem___273412, int_273407)
    
    # Assigning a type to the variable 'tuple_var_assignment_273007' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_var_assignment_273007', subscript_call_result_273413)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_var_assignment_273006' (line 178)
    tuple_var_assignment_273006_273414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_var_assignment_273006')
    # Assigning a type to the variable 'MA' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'MA', tuple_var_assignment_273006_273414)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_var_assignment_273007' (line 178)
    tuple_var_assignment_273007_273415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_var_assignment_273007')
    # Assigning a type to the variable 'NA' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'NA', tuple_var_assignment_273007_273415)
    
    # Assigning a Call to a Tuple (line 179):
    
    # Assigning a Subscript to a Name (line 179):
    
    # Obtaining the type of the subscript
    int_273416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 4), 'int')
    
    # Call to _shape_or_none(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'B' (line 179)
    B_273418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 28), 'B', False)
    # Processing the call keyword arguments (line 179)
    kwargs_273419 = {}
    # Getting the type of '_shape_or_none' (line 179)
    _shape_or_none_273417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), '_shape_or_none', False)
    # Calling _shape_or_none(args, kwargs) (line 179)
    _shape_or_none_call_result_273420 = invoke(stypy.reporting.localization.Localization(__file__, 179, 13), _shape_or_none_273417, *[B_273418], **kwargs_273419)
    
    # Obtaining the member '__getitem__' of a type (line 179)
    getitem___273421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 4), _shape_or_none_call_result_273420, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 179)
    subscript_call_result_273422 = invoke(stypy.reporting.localization.Localization(__file__, 179, 4), getitem___273421, int_273416)
    
    # Assigning a type to the variable 'tuple_var_assignment_273008' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'tuple_var_assignment_273008', subscript_call_result_273422)
    
    # Assigning a Subscript to a Name (line 179):
    
    # Obtaining the type of the subscript
    int_273423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 4), 'int')
    
    # Call to _shape_or_none(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'B' (line 179)
    B_273425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 28), 'B', False)
    # Processing the call keyword arguments (line 179)
    kwargs_273426 = {}
    # Getting the type of '_shape_or_none' (line 179)
    _shape_or_none_273424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), '_shape_or_none', False)
    # Calling _shape_or_none(args, kwargs) (line 179)
    _shape_or_none_call_result_273427 = invoke(stypy.reporting.localization.Localization(__file__, 179, 13), _shape_or_none_273424, *[B_273425], **kwargs_273426)
    
    # Obtaining the member '__getitem__' of a type (line 179)
    getitem___273428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 4), _shape_or_none_call_result_273427, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 179)
    subscript_call_result_273429 = invoke(stypy.reporting.localization.Localization(__file__, 179, 4), getitem___273428, int_273423)
    
    # Assigning a type to the variable 'tuple_var_assignment_273009' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'tuple_var_assignment_273009', subscript_call_result_273429)
    
    # Assigning a Name to a Name (line 179):
    # Getting the type of 'tuple_var_assignment_273008' (line 179)
    tuple_var_assignment_273008_273430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'tuple_var_assignment_273008')
    # Assigning a type to the variable 'MB' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'MB', tuple_var_assignment_273008_273430)
    
    # Assigning a Name to a Name (line 179):
    # Getting the type of 'tuple_var_assignment_273009' (line 179)
    tuple_var_assignment_273009_273431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'tuple_var_assignment_273009')
    # Assigning a type to the variable 'NB' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'NB', tuple_var_assignment_273009_273431)
    
    # Assigning a Call to a Tuple (line 180):
    
    # Assigning a Subscript to a Name (line 180):
    
    # Obtaining the type of the subscript
    int_273432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 4), 'int')
    
    # Call to _shape_or_none(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'C' (line 180)
    C_273434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'C', False)
    # Processing the call keyword arguments (line 180)
    kwargs_273435 = {}
    # Getting the type of '_shape_or_none' (line 180)
    _shape_or_none_273433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 13), '_shape_or_none', False)
    # Calling _shape_or_none(args, kwargs) (line 180)
    _shape_or_none_call_result_273436 = invoke(stypy.reporting.localization.Localization(__file__, 180, 13), _shape_or_none_273433, *[C_273434], **kwargs_273435)
    
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___273437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 4), _shape_or_none_call_result_273436, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_273438 = invoke(stypy.reporting.localization.Localization(__file__, 180, 4), getitem___273437, int_273432)
    
    # Assigning a type to the variable 'tuple_var_assignment_273010' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'tuple_var_assignment_273010', subscript_call_result_273438)
    
    # Assigning a Subscript to a Name (line 180):
    
    # Obtaining the type of the subscript
    int_273439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 4), 'int')
    
    # Call to _shape_or_none(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'C' (line 180)
    C_273441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'C', False)
    # Processing the call keyword arguments (line 180)
    kwargs_273442 = {}
    # Getting the type of '_shape_or_none' (line 180)
    _shape_or_none_273440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 13), '_shape_or_none', False)
    # Calling _shape_or_none(args, kwargs) (line 180)
    _shape_or_none_call_result_273443 = invoke(stypy.reporting.localization.Localization(__file__, 180, 13), _shape_or_none_273440, *[C_273441], **kwargs_273442)
    
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___273444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 4), _shape_or_none_call_result_273443, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_273445 = invoke(stypy.reporting.localization.Localization(__file__, 180, 4), getitem___273444, int_273439)
    
    # Assigning a type to the variable 'tuple_var_assignment_273011' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'tuple_var_assignment_273011', subscript_call_result_273445)
    
    # Assigning a Name to a Name (line 180):
    # Getting the type of 'tuple_var_assignment_273010' (line 180)
    tuple_var_assignment_273010_273446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'tuple_var_assignment_273010')
    # Assigning a type to the variable 'MC' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'MC', tuple_var_assignment_273010_273446)
    
    # Assigning a Name to a Name (line 180):
    # Getting the type of 'tuple_var_assignment_273011' (line 180)
    tuple_var_assignment_273011_273447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'tuple_var_assignment_273011')
    # Assigning a type to the variable 'NC' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'NC', tuple_var_assignment_273011_273447)
    
    # Assigning a Call to a Tuple (line 181):
    
    # Assigning a Subscript to a Name (line 181):
    
    # Obtaining the type of the subscript
    int_273448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 4), 'int')
    
    # Call to _shape_or_none(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'D' (line 181)
    D_273450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'D', False)
    # Processing the call keyword arguments (line 181)
    kwargs_273451 = {}
    # Getting the type of '_shape_or_none' (line 181)
    _shape_or_none_273449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 13), '_shape_or_none', False)
    # Calling _shape_or_none(args, kwargs) (line 181)
    _shape_or_none_call_result_273452 = invoke(stypy.reporting.localization.Localization(__file__, 181, 13), _shape_or_none_273449, *[D_273450], **kwargs_273451)
    
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___273453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 4), _shape_or_none_call_result_273452, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_273454 = invoke(stypy.reporting.localization.Localization(__file__, 181, 4), getitem___273453, int_273448)
    
    # Assigning a type to the variable 'tuple_var_assignment_273012' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_273012', subscript_call_result_273454)
    
    # Assigning a Subscript to a Name (line 181):
    
    # Obtaining the type of the subscript
    int_273455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 4), 'int')
    
    # Call to _shape_or_none(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'D' (line 181)
    D_273457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'D', False)
    # Processing the call keyword arguments (line 181)
    kwargs_273458 = {}
    # Getting the type of '_shape_or_none' (line 181)
    _shape_or_none_273456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 13), '_shape_or_none', False)
    # Calling _shape_or_none(args, kwargs) (line 181)
    _shape_or_none_call_result_273459 = invoke(stypy.reporting.localization.Localization(__file__, 181, 13), _shape_or_none_273456, *[D_273457], **kwargs_273458)
    
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___273460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 4), _shape_or_none_call_result_273459, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_273461 = invoke(stypy.reporting.localization.Localization(__file__, 181, 4), getitem___273460, int_273455)
    
    # Assigning a type to the variable 'tuple_var_assignment_273013' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_273013', subscript_call_result_273461)
    
    # Assigning a Name to a Name (line 181):
    # Getting the type of 'tuple_var_assignment_273012' (line 181)
    tuple_var_assignment_273012_273462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_273012')
    # Assigning a type to the variable 'MD' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'MD', tuple_var_assignment_273012_273462)
    
    # Assigning a Name to a Name (line 181):
    # Getting the type of 'tuple_var_assignment_273013' (line 181)
    tuple_var_assignment_273013_273463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'tuple_var_assignment_273013')
    # Assigning a type to the variable 'ND' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'ND', tuple_var_assignment_273013_273463)
    
    # Assigning a Call to a Name (line 183):
    
    # Assigning a Call to a Name (line 183):
    
    # Call to _choice_not_none(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'MA' (line 183)
    MA_273465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 25), 'MA', False)
    # Getting the type of 'MB' (line 183)
    MB_273466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 29), 'MB', False)
    # Getting the type of 'NC' (line 183)
    NC_273467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 33), 'NC', False)
    # Processing the call keyword arguments (line 183)
    kwargs_273468 = {}
    # Getting the type of '_choice_not_none' (line 183)
    _choice_not_none_273464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), '_choice_not_none', False)
    # Calling _choice_not_none(args, kwargs) (line 183)
    _choice_not_none_call_result_273469 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), _choice_not_none_273464, *[MA_273465, MB_273466, NC_273467], **kwargs_273468)
    
    # Assigning a type to the variable 'p' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'p', _choice_not_none_call_result_273469)
    
    # Assigning a Call to a Name (line 184):
    
    # Assigning a Call to a Name (line 184):
    
    # Call to _choice_not_none(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'NB' (line 184)
    NB_273471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 25), 'NB', False)
    # Getting the type of 'ND' (line 184)
    ND_273472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 29), 'ND', False)
    # Processing the call keyword arguments (line 184)
    kwargs_273473 = {}
    # Getting the type of '_choice_not_none' (line 184)
    _choice_not_none_273470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), '_choice_not_none', False)
    # Calling _choice_not_none(args, kwargs) (line 184)
    _choice_not_none_call_result_273474 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), _choice_not_none_273470, *[NB_273471, ND_273472], **kwargs_273473)
    
    # Assigning a type to the variable 'q' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'q', _choice_not_none_call_result_273474)
    
    # Assigning a Call to a Name (line 185):
    
    # Assigning a Call to a Name (line 185):
    
    # Call to _choice_not_none(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'MC' (line 185)
    MC_273476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 25), 'MC', False)
    # Getting the type of 'MD' (line 185)
    MD_273477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 29), 'MD', False)
    # Processing the call keyword arguments (line 185)
    kwargs_273478 = {}
    # Getting the type of '_choice_not_none' (line 185)
    _choice_not_none_273475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), '_choice_not_none', False)
    # Calling _choice_not_none(args, kwargs) (line 185)
    _choice_not_none_call_result_273479 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), _choice_not_none_273475, *[MC_273476, MD_273477], **kwargs_273478)
    
    # Assigning a type to the variable 'r' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'r', _choice_not_none_call_result_273479)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'p' (line 186)
    p_273480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 7), 'p')
    # Getting the type of 'None' (line 186)
    None_273481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'None')
    # Applying the binary operator 'is' (line 186)
    result_is__273482 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 7), 'is', p_273480, None_273481)
    
    
    # Getting the type of 'q' (line 186)
    q_273483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'q')
    # Getting the type of 'None' (line 186)
    None_273484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'None')
    # Applying the binary operator 'is' (line 186)
    result_is__273485 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 20), 'is', q_273483, None_273484)
    
    # Applying the binary operator 'or' (line 186)
    result_or_keyword_273486 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 7), 'or', result_is__273482, result_is__273485)
    
    # Getting the type of 'r' (line 186)
    r_273487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 33), 'r')
    # Getting the type of 'None' (line 186)
    None_273488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 38), 'None')
    # Applying the binary operator 'is' (line 186)
    result_is__273489 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 33), 'is', r_273487, None_273488)
    
    # Applying the binary operator 'or' (line 186)
    result_or_keyword_273490 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 7), 'or', result_or_keyword_273486, result_is__273489)
    
    # Testing the type of an if condition (line 186)
    if_condition_273491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 4), result_or_keyword_273490)
    # Assigning a type to the variable 'if_condition_273491' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'if_condition_273491', if_condition_273491)
    # SSA begins for if statement (line 186)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 187)
    # Processing the call arguments (line 187)
    str_273493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 25), 'str', 'Not enough information on the system.')
    # Processing the call keyword arguments (line 187)
    kwargs_273494 = {}
    # Getting the type of 'ValueError' (line 187)
    ValueError_273492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 187)
    ValueError_call_result_273495 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), ValueError_273492, *[str_273493], **kwargs_273494)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 187, 8), ValueError_call_result_273495, 'raise parameter', BaseException)
    # SSA join for if statement (line 186)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 189):
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    int_273496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 4), 'int')
    
    # Call to map(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of '_none_to_empty_2d' (line 189)
    _none_to_empty_2d_273498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), '_none_to_empty_2d', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 189)
    tuple_273499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 189)
    # Adding element type (line 189)
    # Getting the type of 'A' (line 189)
    A_273500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 41), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273499, A_273500)
    # Adding element type (line 189)
    # Getting the type of 'B' (line 189)
    B_273501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 44), 'B', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273499, B_273501)
    # Adding element type (line 189)
    # Getting the type of 'C' (line 189)
    C_273502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 47), 'C', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273499, C_273502)
    # Adding element type (line 189)
    # Getting the type of 'D' (line 189)
    D_273503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 50), 'D', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273499, D_273503)
    
    # Processing the call keyword arguments (line 189)
    kwargs_273504 = {}
    # Getting the type of 'map' (line 189)
    map_273497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 17), 'map', False)
    # Calling map(args, kwargs) (line 189)
    map_call_result_273505 = invoke(stypy.reporting.localization.Localization(__file__, 189, 17), map_273497, *[_none_to_empty_2d_273498, tuple_273499], **kwargs_273504)
    
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___273506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 4), map_call_result_273505, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_273507 = invoke(stypy.reporting.localization.Localization(__file__, 189, 4), getitem___273506, int_273496)
    
    # Assigning a type to the variable 'tuple_var_assignment_273014' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_273014', subscript_call_result_273507)
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    int_273508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 4), 'int')
    
    # Call to map(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of '_none_to_empty_2d' (line 189)
    _none_to_empty_2d_273510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), '_none_to_empty_2d', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 189)
    tuple_273511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 189)
    # Adding element type (line 189)
    # Getting the type of 'A' (line 189)
    A_273512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 41), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273511, A_273512)
    # Adding element type (line 189)
    # Getting the type of 'B' (line 189)
    B_273513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 44), 'B', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273511, B_273513)
    # Adding element type (line 189)
    # Getting the type of 'C' (line 189)
    C_273514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 47), 'C', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273511, C_273514)
    # Adding element type (line 189)
    # Getting the type of 'D' (line 189)
    D_273515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 50), 'D', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273511, D_273515)
    
    # Processing the call keyword arguments (line 189)
    kwargs_273516 = {}
    # Getting the type of 'map' (line 189)
    map_273509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 17), 'map', False)
    # Calling map(args, kwargs) (line 189)
    map_call_result_273517 = invoke(stypy.reporting.localization.Localization(__file__, 189, 17), map_273509, *[_none_to_empty_2d_273510, tuple_273511], **kwargs_273516)
    
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___273518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 4), map_call_result_273517, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_273519 = invoke(stypy.reporting.localization.Localization(__file__, 189, 4), getitem___273518, int_273508)
    
    # Assigning a type to the variable 'tuple_var_assignment_273015' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_273015', subscript_call_result_273519)
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    int_273520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 4), 'int')
    
    # Call to map(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of '_none_to_empty_2d' (line 189)
    _none_to_empty_2d_273522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), '_none_to_empty_2d', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 189)
    tuple_273523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 189)
    # Adding element type (line 189)
    # Getting the type of 'A' (line 189)
    A_273524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 41), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273523, A_273524)
    # Adding element type (line 189)
    # Getting the type of 'B' (line 189)
    B_273525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 44), 'B', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273523, B_273525)
    # Adding element type (line 189)
    # Getting the type of 'C' (line 189)
    C_273526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 47), 'C', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273523, C_273526)
    # Adding element type (line 189)
    # Getting the type of 'D' (line 189)
    D_273527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 50), 'D', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273523, D_273527)
    
    # Processing the call keyword arguments (line 189)
    kwargs_273528 = {}
    # Getting the type of 'map' (line 189)
    map_273521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 17), 'map', False)
    # Calling map(args, kwargs) (line 189)
    map_call_result_273529 = invoke(stypy.reporting.localization.Localization(__file__, 189, 17), map_273521, *[_none_to_empty_2d_273522, tuple_273523], **kwargs_273528)
    
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___273530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 4), map_call_result_273529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_273531 = invoke(stypy.reporting.localization.Localization(__file__, 189, 4), getitem___273530, int_273520)
    
    # Assigning a type to the variable 'tuple_var_assignment_273016' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_273016', subscript_call_result_273531)
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    int_273532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 4), 'int')
    
    # Call to map(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of '_none_to_empty_2d' (line 189)
    _none_to_empty_2d_273534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), '_none_to_empty_2d', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 189)
    tuple_273535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 189)
    # Adding element type (line 189)
    # Getting the type of 'A' (line 189)
    A_273536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 41), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273535, A_273536)
    # Adding element type (line 189)
    # Getting the type of 'B' (line 189)
    B_273537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 44), 'B', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273535, B_273537)
    # Adding element type (line 189)
    # Getting the type of 'C' (line 189)
    C_273538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 47), 'C', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273535, C_273538)
    # Adding element type (line 189)
    # Getting the type of 'D' (line 189)
    D_273539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 50), 'D', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 41), tuple_273535, D_273539)
    
    # Processing the call keyword arguments (line 189)
    kwargs_273540 = {}
    # Getting the type of 'map' (line 189)
    map_273533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 17), 'map', False)
    # Calling map(args, kwargs) (line 189)
    map_call_result_273541 = invoke(stypy.reporting.localization.Localization(__file__, 189, 17), map_273533, *[_none_to_empty_2d_273534, tuple_273535], **kwargs_273540)
    
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___273542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 4), map_call_result_273541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_273543 = invoke(stypy.reporting.localization.Localization(__file__, 189, 4), getitem___273542, int_273532)
    
    # Assigning a type to the variable 'tuple_var_assignment_273017' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_273017', subscript_call_result_273543)
    
    # Assigning a Name to a Name (line 189):
    # Getting the type of 'tuple_var_assignment_273014' (line 189)
    tuple_var_assignment_273014_273544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_273014')
    # Assigning a type to the variable 'A' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'A', tuple_var_assignment_273014_273544)
    
    # Assigning a Name to a Name (line 189):
    # Getting the type of 'tuple_var_assignment_273015' (line 189)
    tuple_var_assignment_273015_273545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_273015')
    # Assigning a type to the variable 'B' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 7), 'B', tuple_var_assignment_273015_273545)
    
    # Assigning a Name to a Name (line 189):
    # Getting the type of 'tuple_var_assignment_273016' (line 189)
    tuple_var_assignment_273016_273546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_273016')
    # Assigning a type to the variable 'C' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 10), 'C', tuple_var_assignment_273016_273546)
    
    # Assigning a Name to a Name (line 189):
    # Getting the type of 'tuple_var_assignment_273017' (line 189)
    tuple_var_assignment_273017_273547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'tuple_var_assignment_273017')
    # Assigning a type to the variable 'D' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 13), 'D', tuple_var_assignment_273017_273547)
    
    # Assigning a Call to a Name (line 190):
    
    # Assigning a Call to a Name (line 190):
    
    # Call to _restore(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'A' (line 190)
    A_273549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'A', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 190)
    tuple_273550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 190)
    # Adding element type (line 190)
    # Getting the type of 'p' (line 190)
    p_273551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 21), 'p', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 21), tuple_273550, p_273551)
    # Adding element type (line 190)
    # Getting the type of 'p' (line 190)
    p_273552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 24), 'p', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 21), tuple_273550, p_273552)
    
    # Processing the call keyword arguments (line 190)
    kwargs_273553 = {}
    # Getting the type of '_restore' (line 190)
    _restore_273548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), '_restore', False)
    # Calling _restore(args, kwargs) (line 190)
    _restore_call_result_273554 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), _restore_273548, *[A_273549, tuple_273550], **kwargs_273553)
    
    # Assigning a type to the variable 'A' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'A', _restore_call_result_273554)
    
    # Assigning a Call to a Name (line 191):
    
    # Assigning a Call to a Name (line 191):
    
    # Call to _restore(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'B' (line 191)
    B_273556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 17), 'B', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 191)
    tuple_273557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 191)
    # Adding element type (line 191)
    # Getting the type of 'p' (line 191)
    p_273558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 21), 'p', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 21), tuple_273557, p_273558)
    # Adding element type (line 191)
    # Getting the type of 'q' (line 191)
    q_273559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'q', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 21), tuple_273557, q_273559)
    
    # Processing the call keyword arguments (line 191)
    kwargs_273560 = {}
    # Getting the type of '_restore' (line 191)
    _restore_273555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), '_restore', False)
    # Calling _restore(args, kwargs) (line 191)
    _restore_call_result_273561 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), _restore_273555, *[B_273556, tuple_273557], **kwargs_273560)
    
    # Assigning a type to the variable 'B' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'B', _restore_call_result_273561)
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to _restore(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'C' (line 192)
    C_273563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'C', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 192)
    tuple_273564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 192)
    # Adding element type (line 192)
    # Getting the type of 'r' (line 192)
    r_273565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 21), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 21), tuple_273564, r_273565)
    # Adding element type (line 192)
    # Getting the type of 'p' (line 192)
    p_273566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'p', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 21), tuple_273564, p_273566)
    
    # Processing the call keyword arguments (line 192)
    kwargs_273567 = {}
    # Getting the type of '_restore' (line 192)
    _restore_273562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), '_restore', False)
    # Calling _restore(args, kwargs) (line 192)
    _restore_call_result_273568 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), _restore_273562, *[C_273563, tuple_273564], **kwargs_273567)
    
    # Assigning a type to the variable 'C' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'C', _restore_call_result_273568)
    
    # Assigning a Call to a Name (line 193):
    
    # Assigning a Call to a Name (line 193):
    
    # Call to _restore(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'D' (line 193)
    D_273570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 17), 'D', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 193)
    tuple_273571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 193)
    # Adding element type (line 193)
    # Getting the type of 'r' (line 193)
    r_273572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 21), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), tuple_273571, r_273572)
    # Adding element type (line 193)
    # Getting the type of 'q' (line 193)
    q_273573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 24), 'q', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 21), tuple_273571, q_273573)
    
    # Processing the call keyword arguments (line 193)
    kwargs_273574 = {}
    # Getting the type of '_restore' (line 193)
    _restore_273569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), '_restore', False)
    # Calling _restore(args, kwargs) (line 193)
    _restore_call_result_273575 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), _restore_273569, *[D_273570, tuple_273571], **kwargs_273574)
    
    # Assigning a type to the variable 'D' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'D', _restore_call_result_273575)
    
    # Obtaining an instance of the builtin type 'tuple' (line 195)
    tuple_273576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 195)
    # Adding element type (line 195)
    # Getting the type of 'A' (line 195)
    A_273577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 11), tuple_273576, A_273577)
    # Adding element type (line 195)
    # Getting the type of 'B' (line 195)
    B_273578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 14), 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 11), tuple_273576, B_273578)
    # Adding element type (line 195)
    # Getting the type of 'C' (line 195)
    C_273579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 17), 'C')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 11), tuple_273576, C_273579)
    # Adding element type (line 195)
    # Getting the type of 'D' (line 195)
    D_273580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'D')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 11), tuple_273576, D_273580)
    
    # Assigning a type to the variable 'stypy_return_type' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type', tuple_273576)
    
    # ################# End of 'abcd_normalize(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'abcd_normalize' in the type store
    # Getting the type of 'stypy_return_type' (line 151)
    stypy_return_type_273581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273581)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'abcd_normalize'
    return stypy_return_type_273581

# Assigning a type to the variable 'abcd_normalize' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'abcd_normalize', abcd_normalize)

@norecursion
def ss2tf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_273582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 28), 'int')
    defaults = [int_273582]
    # Create a new context for function 'ss2tf'
    module_type_store = module_type_store.open_function_context('ss2tf', 198, 0, False)
    
    # Passed parameters checking function
    ss2tf.stypy_localization = localization
    ss2tf.stypy_type_of_self = None
    ss2tf.stypy_type_store = module_type_store
    ss2tf.stypy_function_name = 'ss2tf'
    ss2tf.stypy_param_names_list = ['A', 'B', 'C', 'D', 'input']
    ss2tf.stypy_varargs_param_name = None
    ss2tf.stypy_kwargs_param_name = None
    ss2tf.stypy_call_defaults = defaults
    ss2tf.stypy_call_varargs = varargs
    ss2tf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ss2tf', ['A', 'B', 'C', 'D', 'input'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ss2tf', localization, ['A', 'B', 'C', 'D', 'input'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ss2tf(...)' code ##################

    str_273583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, (-1)), 'str', "State-space to transfer function.\n\n    A, B, C, D defines a linear state-space system with `p` inputs,\n    `q` outputs, and `n` state variables.\n\n    Parameters\n    ----------\n    A : array_like\n        State (or system) matrix of shape ``(n, n)``\n    B : array_like\n        Input matrix of shape ``(n, p)``\n    C : array_like\n        Output matrix of shape ``(q, n)``\n    D : array_like\n        Feedthrough (or feedforward) matrix of shape ``(q, p)``\n    input : int, optional\n        For multiple-input systems, the index of the input to use.\n\n    Returns\n    -------\n    num : 2-D ndarray\n        Numerator(s) of the resulting transfer function(s).  `num` has one row\n        for each of the system's outputs. Each row is a sequence representation\n        of the numerator polynomial.\n    den : 1-D ndarray\n        Denominator of the resulting transfer function(s).  `den` is a sequence\n        representation of the denominator polynomial.\n\n    Examples\n    --------\n    Convert the state-space representation:\n\n    .. math::\n\n        \\dot{\\textbf{x}}(t) =\n        \\begin{bmatrix} -2 & -1 \\\\ 1 & 0 \\end{bmatrix} \\textbf{x}(t) +\n        \\begin{bmatrix} 1 \\\\ 0 \\end{bmatrix} \\textbf{u}(t) \\\\\n\n        \\textbf{y}(t) = \\begin{bmatrix} 1 & 2 \\end{bmatrix} \\textbf{x}(t) +\n        \\begin{bmatrix} 1 \\end{bmatrix} \\textbf{u}(t)\n\n    >>> A = [[-2, -1], [1, 0]]\n    >>> B = [[1], [0]]  # 2-dimensional column vector\n    >>> C = [[1, 2]]    # 2-dimensional row vector\n    >>> D = 1\n\n    to the transfer function:\n\n    .. math:: H(s) = \\frac{s^2 + 3s + 3}{s^2 + 2s + 1}\n\n    >>> from scipy.signal import ss2tf\n    >>> ss2tf(A, B, C, D)\n    (array([[1, 3, 3]]), array([ 1.,  2.,  1.]))\n    ")
    
    # Assigning a Call to a Tuple (line 256):
    
    # Assigning a Subscript to a Name (line 256):
    
    # Obtaining the type of the subscript
    int_273584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 4), 'int')
    
    # Call to abcd_normalize(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'A' (line 256)
    A_273586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'A', False)
    # Getting the type of 'B' (line 256)
    B_273587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 35), 'B', False)
    # Getting the type of 'C' (line 256)
    C_273588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 38), 'C', False)
    # Getting the type of 'D' (line 256)
    D_273589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 41), 'D', False)
    # Processing the call keyword arguments (line 256)
    kwargs_273590 = {}
    # Getting the type of 'abcd_normalize' (line 256)
    abcd_normalize_273585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 17), 'abcd_normalize', False)
    # Calling abcd_normalize(args, kwargs) (line 256)
    abcd_normalize_call_result_273591 = invoke(stypy.reporting.localization.Localization(__file__, 256, 17), abcd_normalize_273585, *[A_273586, B_273587, C_273588, D_273589], **kwargs_273590)
    
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___273592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 4), abcd_normalize_call_result_273591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_273593 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), getitem___273592, int_273584)
    
    # Assigning a type to the variable 'tuple_var_assignment_273018' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_273018', subscript_call_result_273593)
    
    # Assigning a Subscript to a Name (line 256):
    
    # Obtaining the type of the subscript
    int_273594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 4), 'int')
    
    # Call to abcd_normalize(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'A' (line 256)
    A_273596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'A', False)
    # Getting the type of 'B' (line 256)
    B_273597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 35), 'B', False)
    # Getting the type of 'C' (line 256)
    C_273598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 38), 'C', False)
    # Getting the type of 'D' (line 256)
    D_273599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 41), 'D', False)
    # Processing the call keyword arguments (line 256)
    kwargs_273600 = {}
    # Getting the type of 'abcd_normalize' (line 256)
    abcd_normalize_273595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 17), 'abcd_normalize', False)
    # Calling abcd_normalize(args, kwargs) (line 256)
    abcd_normalize_call_result_273601 = invoke(stypy.reporting.localization.Localization(__file__, 256, 17), abcd_normalize_273595, *[A_273596, B_273597, C_273598, D_273599], **kwargs_273600)
    
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___273602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 4), abcd_normalize_call_result_273601, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_273603 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), getitem___273602, int_273594)
    
    # Assigning a type to the variable 'tuple_var_assignment_273019' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_273019', subscript_call_result_273603)
    
    # Assigning a Subscript to a Name (line 256):
    
    # Obtaining the type of the subscript
    int_273604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 4), 'int')
    
    # Call to abcd_normalize(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'A' (line 256)
    A_273606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'A', False)
    # Getting the type of 'B' (line 256)
    B_273607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 35), 'B', False)
    # Getting the type of 'C' (line 256)
    C_273608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 38), 'C', False)
    # Getting the type of 'D' (line 256)
    D_273609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 41), 'D', False)
    # Processing the call keyword arguments (line 256)
    kwargs_273610 = {}
    # Getting the type of 'abcd_normalize' (line 256)
    abcd_normalize_273605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 17), 'abcd_normalize', False)
    # Calling abcd_normalize(args, kwargs) (line 256)
    abcd_normalize_call_result_273611 = invoke(stypy.reporting.localization.Localization(__file__, 256, 17), abcd_normalize_273605, *[A_273606, B_273607, C_273608, D_273609], **kwargs_273610)
    
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___273612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 4), abcd_normalize_call_result_273611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_273613 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), getitem___273612, int_273604)
    
    # Assigning a type to the variable 'tuple_var_assignment_273020' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_273020', subscript_call_result_273613)
    
    # Assigning a Subscript to a Name (line 256):
    
    # Obtaining the type of the subscript
    int_273614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 4), 'int')
    
    # Call to abcd_normalize(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'A' (line 256)
    A_273616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'A', False)
    # Getting the type of 'B' (line 256)
    B_273617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 35), 'B', False)
    # Getting the type of 'C' (line 256)
    C_273618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 38), 'C', False)
    # Getting the type of 'D' (line 256)
    D_273619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 41), 'D', False)
    # Processing the call keyword arguments (line 256)
    kwargs_273620 = {}
    # Getting the type of 'abcd_normalize' (line 256)
    abcd_normalize_273615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 17), 'abcd_normalize', False)
    # Calling abcd_normalize(args, kwargs) (line 256)
    abcd_normalize_call_result_273621 = invoke(stypy.reporting.localization.Localization(__file__, 256, 17), abcd_normalize_273615, *[A_273616, B_273617, C_273618, D_273619], **kwargs_273620)
    
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___273622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 4), abcd_normalize_call_result_273621, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_273623 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), getitem___273622, int_273614)
    
    # Assigning a type to the variable 'tuple_var_assignment_273021' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_273021', subscript_call_result_273623)
    
    # Assigning a Name to a Name (line 256):
    # Getting the type of 'tuple_var_assignment_273018' (line 256)
    tuple_var_assignment_273018_273624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_273018')
    # Assigning a type to the variable 'A' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'A', tuple_var_assignment_273018_273624)
    
    # Assigning a Name to a Name (line 256):
    # Getting the type of 'tuple_var_assignment_273019' (line 256)
    tuple_var_assignment_273019_273625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_273019')
    # Assigning a type to the variable 'B' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 7), 'B', tuple_var_assignment_273019_273625)
    
    # Assigning a Name to a Name (line 256):
    # Getting the type of 'tuple_var_assignment_273020' (line 256)
    tuple_var_assignment_273020_273626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_273020')
    # Assigning a type to the variable 'C' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 10), 'C', tuple_var_assignment_273020_273626)
    
    # Assigning a Name to a Name (line 256):
    # Getting the type of 'tuple_var_assignment_273021' (line 256)
    tuple_var_assignment_273021_273627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_273021')
    # Assigning a type to the variable 'D' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 13), 'D', tuple_var_assignment_273021_273627)
    
    # Assigning a Attribute to a Tuple (line 258):
    
    # Assigning a Subscript to a Name (line 258):
    
    # Obtaining the type of the subscript
    int_273628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 4), 'int')
    # Getting the type of 'D' (line 258)
    D_273629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'D')
    # Obtaining the member 'shape' of a type (line 258)
    shape_273630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 16), D_273629, 'shape')
    # Obtaining the member '__getitem__' of a type (line 258)
    getitem___273631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 4), shape_273630, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 258)
    subscript_call_result_273632 = invoke(stypy.reporting.localization.Localization(__file__, 258, 4), getitem___273631, int_273628)
    
    # Assigning a type to the variable 'tuple_var_assignment_273022' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'tuple_var_assignment_273022', subscript_call_result_273632)
    
    # Assigning a Subscript to a Name (line 258):
    
    # Obtaining the type of the subscript
    int_273633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 4), 'int')
    # Getting the type of 'D' (line 258)
    D_273634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'D')
    # Obtaining the member 'shape' of a type (line 258)
    shape_273635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 16), D_273634, 'shape')
    # Obtaining the member '__getitem__' of a type (line 258)
    getitem___273636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 4), shape_273635, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 258)
    subscript_call_result_273637 = invoke(stypy.reporting.localization.Localization(__file__, 258, 4), getitem___273636, int_273633)
    
    # Assigning a type to the variable 'tuple_var_assignment_273023' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'tuple_var_assignment_273023', subscript_call_result_273637)
    
    # Assigning a Name to a Name (line 258):
    # Getting the type of 'tuple_var_assignment_273022' (line 258)
    tuple_var_assignment_273022_273638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'tuple_var_assignment_273022')
    # Assigning a type to the variable 'nout' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'nout', tuple_var_assignment_273022_273638)
    
    # Assigning a Name to a Name (line 258):
    # Getting the type of 'tuple_var_assignment_273023' (line 258)
    tuple_var_assignment_273023_273639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'tuple_var_assignment_273023')
    # Assigning a type to the variable 'nin' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 10), 'nin', tuple_var_assignment_273023_273639)
    
    
    # Getting the type of 'input' (line 259)
    input_273640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 7), 'input')
    # Getting the type of 'nin' (line 259)
    nin_273641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'nin')
    # Applying the binary operator '>=' (line 259)
    result_ge_273642 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 7), '>=', input_273640, nin_273641)
    
    # Testing the type of an if condition (line 259)
    if_condition_273643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 4), result_ge_273642)
    # Assigning a type to the variable 'if_condition_273643' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'if_condition_273643', if_condition_273643)
    # SSA begins for if statement (line 259)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 260)
    # Processing the call arguments (line 260)
    str_273645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 25), 'str', 'System does not have the input specified.')
    # Processing the call keyword arguments (line 260)
    kwargs_273646 = {}
    # Getting the type of 'ValueError' (line 260)
    ValueError_273644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 260)
    ValueError_call_result_273647 = invoke(stypy.reporting.localization.Localization(__file__, 260, 14), ValueError_273644, *[str_273645], **kwargs_273646)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 260, 8), ValueError_call_result_273647, 'raise parameter', BaseException)
    # SSA join for if statement (line 259)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 263):
    
    # Assigning a Subscript to a Name (line 263):
    
    # Obtaining the type of the subscript
    slice_273648 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 263, 8), None, None, None)
    # Getting the type of 'input' (line 263)
    input_273649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), 'input')
    # Getting the type of 'input' (line 263)
    input_273650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'input')
    int_273651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 27), 'int')
    # Applying the binary operator '+' (line 263)
    result_add_273652 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 19), '+', input_273650, int_273651)
    
    slice_273653 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 263, 8), input_273649, result_add_273652, None)
    # Getting the type of 'B' (line 263)
    B_273654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'B')
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___273655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), B_273654, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_273656 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), getitem___273655, (slice_273648, slice_273653))
    
    # Assigning a type to the variable 'B' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'B', subscript_call_result_273656)
    
    # Assigning a Subscript to a Name (line 264):
    
    # Assigning a Subscript to a Name (line 264):
    
    # Obtaining the type of the subscript
    slice_273657 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 264, 8), None, None, None)
    # Getting the type of 'input' (line 264)
    input_273658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'input')
    # Getting the type of 'input' (line 264)
    input_273659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 19), 'input')
    int_273660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 27), 'int')
    # Applying the binary operator '+' (line 264)
    result_add_273661 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 19), '+', input_273659, int_273660)
    
    slice_273662 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 264, 8), input_273658, result_add_273661, None)
    # Getting the type of 'D' (line 264)
    D_273663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'D')
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___273664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), D_273663, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_273665 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), getitem___273664, (slice_273657, slice_273662))
    
    # Assigning a type to the variable 'D' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'D', subscript_call_result_273665)
    
    
    # SSA begins for try-except statement (line 266)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 267):
    
    # Assigning a Call to a Name (line 267):
    
    # Call to poly(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'A' (line 267)
    A_273667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 19), 'A', False)
    # Processing the call keyword arguments (line 267)
    kwargs_273668 = {}
    # Getting the type of 'poly' (line 267)
    poly_273666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 14), 'poly', False)
    # Calling poly(args, kwargs) (line 267)
    poly_call_result_273669 = invoke(stypy.reporting.localization.Localization(__file__, 267, 14), poly_273666, *[A_273667], **kwargs_273668)
    
    # Assigning a type to the variable 'den' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'den', poly_call_result_273669)
    # SSA branch for the except part of a try statement (line 266)
    # SSA branch for the except 'ValueError' branch of a try statement (line 266)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Num to a Name (line 269):
    
    # Assigning a Num to a Name (line 269):
    int_273670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 14), 'int')
    # Assigning a type to the variable 'den' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'den', int_273670)
    # SSA join for try-except statement (line 266)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to product(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'B' (line 271)
    B_273672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'B', False)
    # Obtaining the member 'shape' of a type (line 271)
    shape_273673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), B_273672, 'shape')
    # Processing the call keyword arguments (line 271)
    int_273674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 30), 'int')
    keyword_273675 = int_273674
    kwargs_273676 = {'axis': keyword_273675}
    # Getting the type of 'product' (line 271)
    product_273671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'product', False)
    # Calling product(args, kwargs) (line 271)
    product_call_result_273677 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), product_273671, *[shape_273673], **kwargs_273676)
    
    int_273678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 36), 'int')
    # Applying the binary operator '==' (line 271)
    result_eq_273679 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 8), '==', product_call_result_273677, int_273678)
    
    
    
    # Call to product(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'C' (line 271)
    C_273681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 52), 'C', False)
    # Obtaining the member 'shape' of a type (line 271)
    shape_273682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 52), C_273681, 'shape')
    # Processing the call keyword arguments (line 271)
    int_273683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 66), 'int')
    keyword_273684 = int_273683
    kwargs_273685 = {'axis': keyword_273684}
    # Getting the type of 'product' (line 271)
    product_273680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 44), 'product', False)
    # Calling product(args, kwargs) (line 271)
    product_call_result_273686 = invoke(stypy.reporting.localization.Localization(__file__, 271, 44), product_273680, *[shape_273682], **kwargs_273685)
    
    int_273687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 72), 'int')
    # Applying the binary operator '==' (line 271)
    result_eq_273688 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 44), '==', product_call_result_273686, int_273687)
    
    # Applying the binary operator 'and' (line 271)
    result_and_keyword_273689 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 7), 'and', result_eq_273679, result_eq_273688)
    
    # Testing the type of an if condition (line 271)
    if_condition_273690 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 4), result_and_keyword_273689)
    # Assigning a type to the variable 'if_condition_273690' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'if_condition_273690', if_condition_273690)
    # SSA begins for if statement (line 271)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 272):
    
    # Assigning a Call to a Name (line 272):
    
    # Call to ravel(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'D' (line 272)
    D_273693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 26), 'D', False)
    # Processing the call keyword arguments (line 272)
    kwargs_273694 = {}
    # Getting the type of 'numpy' (line 272)
    numpy_273691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 14), 'numpy', False)
    # Obtaining the member 'ravel' of a type (line 272)
    ravel_273692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 14), numpy_273691, 'ravel')
    # Calling ravel(args, kwargs) (line 272)
    ravel_call_result_273695 = invoke(stypy.reporting.localization.Localization(__file__, 272, 14), ravel_273692, *[D_273693], **kwargs_273694)
    
    # Assigning a type to the variable 'num' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'num', ravel_call_result_273695)
    
    
    # Evaluating a boolean operation
    
    
    # Call to product(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'D' (line 273)
    D_273697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'D', False)
    # Obtaining the member 'shape' of a type (line 273)
    shape_273698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 20), D_273697, 'shape')
    # Processing the call keyword arguments (line 273)
    int_273699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 34), 'int')
    keyword_273700 = int_273699
    kwargs_273701 = {'axis': keyword_273700}
    # Getting the type of 'product' (line 273)
    product_273696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'product', False)
    # Calling product(args, kwargs) (line 273)
    product_call_result_273702 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), product_273696, *[shape_273698], **kwargs_273701)
    
    int_273703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 40), 'int')
    # Applying the binary operator '==' (line 273)
    result_eq_273704 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 12), '==', product_call_result_273702, int_273703)
    
    
    
    # Call to product(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'A' (line 273)
    A_273706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 56), 'A', False)
    # Obtaining the member 'shape' of a type (line 273)
    shape_273707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 56), A_273706, 'shape')
    # Processing the call keyword arguments (line 273)
    int_273708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 70), 'int')
    keyword_273709 = int_273708
    kwargs_273710 = {'axis': keyword_273709}
    # Getting the type of 'product' (line 273)
    product_273705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 48), 'product', False)
    # Calling product(args, kwargs) (line 273)
    product_call_result_273711 = invoke(stypy.reporting.localization.Localization(__file__, 273, 48), product_273705, *[shape_273707], **kwargs_273710)
    
    int_273712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 76), 'int')
    # Applying the binary operator '==' (line 273)
    result_eq_273713 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 48), '==', product_call_result_273711, int_273712)
    
    # Applying the binary operator 'and' (line 273)
    result_and_keyword_273714 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), 'and', result_eq_273704, result_eq_273713)
    
    # Testing the type of an if condition (line 273)
    if_condition_273715 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_and_keyword_273714)
    # Assigning a type to the variable 'if_condition_273715' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_273715', if_condition_273715)
    # SSA begins for if statement (line 273)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 274):
    
    # Assigning a List to a Name (line 274):
    
    # Obtaining an instance of the builtin type 'list' (line 274)
    list_273716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 274)
    
    # Assigning a type to the variable 'den' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'den', list_273716)
    # SSA join for if statement (line 273)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 275)
    tuple_273717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 275)
    # Adding element type (line 275)
    # Getting the type of 'num' (line 275)
    num_273718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'num')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 15), tuple_273717, num_273718)
    # Adding element type (line 275)
    # Getting the type of 'den' (line 275)
    den_273719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 20), 'den')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 15), tuple_273717, den_273719)
    
    # Assigning a type to the variable 'stypy_return_type' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'stypy_return_type', tuple_273717)
    # SSA join for if statement (line 271)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 277):
    
    # Assigning a Subscript to a Name (line 277):
    
    # Obtaining the type of the subscript
    int_273720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 25), 'int')
    # Getting the type of 'A' (line 277)
    A_273721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 17), 'A')
    # Obtaining the member 'shape' of a type (line 277)
    shape_273722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 17), A_273721, 'shape')
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___273723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 17), shape_273722, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_273724 = invoke(stypy.reporting.localization.Localization(__file__, 277, 17), getitem___273723, int_273720)
    
    # Assigning a type to the variable 'num_states' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'num_states', subscript_call_result_273724)
    
    # Assigning a BinOp to a Name (line 278):
    
    # Assigning a BinOp to a Name (line 278):
    
    # Obtaining the type of the subscript
    slice_273725 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 278, 16), None, None, None)
    int_273726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 21), 'int')
    # Getting the type of 'A' (line 278)
    A_273727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'A')
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___273728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 16), A_273727, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_273729 = invoke(stypy.reporting.localization.Localization(__file__, 278, 16), getitem___273728, (slice_273725, int_273726))
    
    
    # Obtaining the type of the subscript
    slice_273730 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 278, 26), None, None, None)
    int_273731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 31), 'int')
    # Getting the type of 'B' (line 278)
    B_273732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 26), 'B')
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___273733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 26), B_273732, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_273734 = invoke(stypy.reporting.localization.Localization(__file__, 278, 26), getitem___273733, (slice_273730, int_273731))
    
    # Applying the binary operator '+' (line 278)
    result_add_273735 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 16), '+', subscript_call_result_273729, subscript_call_result_273734)
    
    
    # Obtaining the type of the subscript
    int_273736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 38), 'int')
    slice_273737 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 278, 36), None, None, None)
    # Getting the type of 'C' (line 278)
    C_273738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 36), 'C')
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___273739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 36), C_273738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_273740 = invoke(stypy.reporting.localization.Localization(__file__, 278, 36), getitem___273739, (int_273736, slice_273737))
    
    # Applying the binary operator '+' (line 278)
    result_add_273741 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 34), '+', result_add_273735, subscript_call_result_273740)
    
    # Getting the type of 'D' (line 278)
    D_273742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 46), 'D')
    # Applying the binary operator '+' (line 278)
    result_add_273743 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 44), '+', result_add_273741, D_273742)
    
    # Assigning a type to the variable 'type_test' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'type_test', result_add_273743)
    
    # Assigning a Call to a Name (line 279):
    
    # Assigning a Call to a Name (line 279):
    
    # Call to zeros(...): (line 279)
    # Processing the call arguments (line 279)
    
    # Obtaining an instance of the builtin type 'tuple' (line 279)
    tuple_273746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 279)
    # Adding element type (line 279)
    # Getting the type of 'nout' (line 279)
    nout_273747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 23), 'nout', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 23), tuple_273746, nout_273747)
    # Adding element type (line 279)
    # Getting the type of 'num_states' (line 279)
    num_states_273748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 29), 'num_states', False)
    int_273749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 42), 'int')
    # Applying the binary operator '+' (line 279)
    result_add_273750 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 29), '+', num_states_273748, int_273749)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 23), tuple_273746, result_add_273750)
    
    # Getting the type of 'type_test' (line 279)
    type_test_273751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 46), 'type_test', False)
    # Obtaining the member 'dtype' of a type (line 279)
    dtype_273752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 46), type_test_273751, 'dtype')
    # Processing the call keyword arguments (line 279)
    kwargs_273753 = {}
    # Getting the type of 'numpy' (line 279)
    numpy_273744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 10), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 279)
    zeros_273745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 10), numpy_273744, 'zeros')
    # Calling zeros(args, kwargs) (line 279)
    zeros_call_result_273754 = invoke(stypy.reporting.localization.Localization(__file__, 279, 10), zeros_273745, *[tuple_273746, dtype_273752], **kwargs_273753)
    
    # Assigning a type to the variable 'num' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'num', zeros_call_result_273754)
    
    
    # Call to range(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'nout' (line 280)
    nout_273756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'nout', False)
    # Processing the call keyword arguments (line 280)
    kwargs_273757 = {}
    # Getting the type of 'range' (line 280)
    range_273755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 13), 'range', False)
    # Calling range(args, kwargs) (line 280)
    range_call_result_273758 = invoke(stypy.reporting.localization.Localization(__file__, 280, 13), range_273755, *[nout_273756], **kwargs_273757)
    
    # Testing the type of a for loop iterable (line 280)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 280, 4), range_call_result_273758)
    # Getting the type of the for loop variable (line 280)
    for_loop_var_273759 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 280, 4), range_call_result_273758)
    # Assigning a type to the variable 'k' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'k', for_loop_var_273759)
    # SSA begins for a for statement (line 280)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 281):
    
    # Assigning a Call to a Name (line 281):
    
    # Call to atleast_2d(...): (line 281)
    # Processing the call arguments (line 281)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 281)
    k_273761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 26), 'k', False)
    slice_273762 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 281, 24), None, None, None)
    # Getting the type of 'C' (line 281)
    C_273763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'C', False)
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___273764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 24), C_273763, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_273765 = invoke(stypy.reporting.localization.Localization(__file__, 281, 24), getitem___273764, (k_273761, slice_273762))
    
    # Processing the call keyword arguments (line 281)
    kwargs_273766 = {}
    # Getting the type of 'atleast_2d' (line 281)
    atleast_2d_273760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 13), 'atleast_2d', False)
    # Calling atleast_2d(args, kwargs) (line 281)
    atleast_2d_call_result_273767 = invoke(stypy.reporting.localization.Localization(__file__, 281, 13), atleast_2d_273760, *[subscript_call_result_273765], **kwargs_273766)
    
    # Assigning a type to the variable 'Ck' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'Ck', atleast_2d_call_result_273767)
    
    # Assigning a BinOp to a Subscript (line 282):
    
    # Assigning a BinOp to a Subscript (line 282):
    
    # Call to poly(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'A' (line 282)
    A_273769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 22), 'A', False)
    
    # Call to dot(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'B' (line 282)
    B_273771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 30), 'B', False)
    # Getting the type of 'Ck' (line 282)
    Ck_273772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 33), 'Ck', False)
    # Processing the call keyword arguments (line 282)
    kwargs_273773 = {}
    # Getting the type of 'dot' (line 282)
    dot_273770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 26), 'dot', False)
    # Calling dot(args, kwargs) (line 282)
    dot_call_result_273774 = invoke(stypy.reporting.localization.Localization(__file__, 282, 26), dot_273770, *[B_273771, Ck_273772], **kwargs_273773)
    
    # Applying the binary operator '-' (line 282)
    result_sub_273775 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 22), '-', A_273769, dot_call_result_273774)
    
    # Processing the call keyword arguments (line 282)
    kwargs_273776 = {}
    # Getting the type of 'poly' (line 282)
    poly_273768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'poly', False)
    # Calling poly(args, kwargs) (line 282)
    poly_call_result_273777 = invoke(stypy.reporting.localization.Localization(__file__, 282, 17), poly_273768, *[result_sub_273775], **kwargs_273776)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 282)
    k_273778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 43), 'k')
    # Getting the type of 'D' (line 282)
    D_273779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 41), 'D')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___273780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 41), D_273779, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_273781 = invoke(stypy.reporting.localization.Localization(__file__, 282, 41), getitem___273780, k_273778)
    
    int_273782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 48), 'int')
    # Applying the binary operator '-' (line 282)
    result_sub_273783 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 41), '-', subscript_call_result_273781, int_273782)
    
    # Getting the type of 'den' (line 282)
    den_273784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 53), 'den')
    # Applying the binary operator '*' (line 282)
    result_mul_273785 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 40), '*', result_sub_273783, den_273784)
    
    # Applying the binary operator '+' (line 282)
    result_add_273786 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 17), '+', poly_call_result_273777, result_mul_273785)
    
    # Getting the type of 'num' (line 282)
    num_273787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'num')
    # Getting the type of 'k' (line 282)
    k_273788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'k')
    # Storing an element on a container (line 282)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 8), num_273787, (k_273788, result_add_273786))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 284)
    tuple_273789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 284)
    # Adding element type (line 284)
    # Getting the type of 'num' (line 284)
    num_273790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'num')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 11), tuple_273789, num_273790)
    # Adding element type (line 284)
    # Getting the type of 'den' (line 284)
    den_273791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'den')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 11), tuple_273789, den_273791)
    
    # Assigning a type to the variable 'stypy_return_type' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type', tuple_273789)
    
    # ################# End of 'ss2tf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ss2tf' in the type store
    # Getting the type of 'stypy_return_type' (line 198)
    stypy_return_type_273792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273792)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ss2tf'
    return stypy_return_type_273792

# Assigning a type to the variable 'ss2tf' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'ss2tf', ss2tf)

@norecursion
def zpk2ss(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'zpk2ss'
    module_type_store = module_type_store.open_function_context('zpk2ss', 287, 0, False)
    
    # Passed parameters checking function
    zpk2ss.stypy_localization = localization
    zpk2ss.stypy_type_of_self = None
    zpk2ss.stypy_type_store = module_type_store
    zpk2ss.stypy_function_name = 'zpk2ss'
    zpk2ss.stypy_param_names_list = ['z', 'p', 'k']
    zpk2ss.stypy_varargs_param_name = None
    zpk2ss.stypy_kwargs_param_name = None
    zpk2ss.stypy_call_defaults = defaults
    zpk2ss.stypy_call_varargs = varargs
    zpk2ss.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'zpk2ss', ['z', 'p', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'zpk2ss', localization, ['z', 'p', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'zpk2ss(...)' code ##################

    str_273793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, (-1)), 'str', 'Zero-pole-gain representation to state-space representation\n\n    Parameters\n    ----------\n    z, p : sequence\n        Zeros and poles.\n    k : float\n        System gain.\n\n    Returns\n    -------\n    A, B, C, D : ndarray\n        State space representation of the system, in controller canonical\n        form.\n\n    ')
    
    # Call to tf2ss(...): (line 304)
    
    # Call to zpk2tf(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'z' (line 304)
    z_273796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'z', False)
    # Getting the type of 'p' (line 304)
    p_273797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 28), 'p', False)
    # Getting the type of 'k' (line 304)
    k_273798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 31), 'k', False)
    # Processing the call keyword arguments (line 304)
    kwargs_273799 = {}
    # Getting the type of 'zpk2tf' (line 304)
    zpk2tf_273795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 18), 'zpk2tf', False)
    # Calling zpk2tf(args, kwargs) (line 304)
    zpk2tf_call_result_273800 = invoke(stypy.reporting.localization.Localization(__file__, 304, 18), zpk2tf_273795, *[z_273796, p_273797, k_273798], **kwargs_273799)
    
    # Processing the call keyword arguments (line 304)
    kwargs_273801 = {}
    # Getting the type of 'tf2ss' (line 304)
    tf2ss_273794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 11), 'tf2ss', False)
    # Calling tf2ss(args, kwargs) (line 304)
    tf2ss_call_result_273802 = invoke(stypy.reporting.localization.Localization(__file__, 304, 11), tf2ss_273794, *[zpk2tf_call_result_273800], **kwargs_273801)
    
    # Assigning a type to the variable 'stypy_return_type' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type', tf2ss_call_result_273802)
    
    # ################# End of 'zpk2ss(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'zpk2ss' in the type store
    # Getting the type of 'stypy_return_type' (line 287)
    stypy_return_type_273803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273803)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'zpk2ss'
    return stypy_return_type_273803

# Assigning a type to the variable 'zpk2ss' (line 287)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'zpk2ss', zpk2ss)

@norecursion
def ss2zpk(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_273804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 29), 'int')
    defaults = [int_273804]
    # Create a new context for function 'ss2zpk'
    module_type_store = module_type_store.open_function_context('ss2zpk', 307, 0, False)
    
    # Passed parameters checking function
    ss2zpk.stypy_localization = localization
    ss2zpk.stypy_type_of_self = None
    ss2zpk.stypy_type_store = module_type_store
    ss2zpk.stypy_function_name = 'ss2zpk'
    ss2zpk.stypy_param_names_list = ['A', 'B', 'C', 'D', 'input']
    ss2zpk.stypy_varargs_param_name = None
    ss2zpk.stypy_kwargs_param_name = None
    ss2zpk.stypy_call_defaults = defaults
    ss2zpk.stypy_call_varargs = varargs
    ss2zpk.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ss2zpk', ['A', 'B', 'C', 'D', 'input'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ss2zpk', localization, ['A', 'B', 'C', 'D', 'input'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ss2zpk(...)' code ##################

    str_273805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, (-1)), 'str', 'State-space representation to zero-pole-gain representation.\n\n    A, B, C, D defines a linear state-space system with `p` inputs,\n    `q` outputs, and `n` state variables.\n\n    Parameters\n    ----------\n    A : array_like\n        State (or system) matrix of shape ``(n, n)``\n    B : array_like\n        Input matrix of shape ``(n, p)``\n    C : array_like\n        Output matrix of shape ``(q, n)``\n    D : array_like\n        Feedthrough (or feedforward) matrix of shape ``(q, p)``\n    input : int, optional\n        For multiple-input systems, the index of the input to use.\n\n    Returns\n    -------\n    z, p : sequence\n        Zeros and poles.\n    k : float\n        System gain.\n\n    ')
    
    # Call to tf2zpk(...): (line 334)
    
    # Call to ss2tf(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'A' (line 334)
    A_273808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 25), 'A', False)
    # Getting the type of 'B' (line 334)
    B_273809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 28), 'B', False)
    # Getting the type of 'C' (line 334)
    C_273810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 31), 'C', False)
    # Getting the type of 'D' (line 334)
    D_273811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 34), 'D', False)
    # Processing the call keyword arguments (line 334)
    # Getting the type of 'input' (line 334)
    input_273812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 43), 'input', False)
    keyword_273813 = input_273812
    kwargs_273814 = {'input': keyword_273813}
    # Getting the type of 'ss2tf' (line 334)
    ss2tf_273807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'ss2tf', False)
    # Calling ss2tf(args, kwargs) (line 334)
    ss2tf_call_result_273815 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), ss2tf_273807, *[A_273808, B_273809, C_273810, D_273811], **kwargs_273814)
    
    # Processing the call keyword arguments (line 334)
    kwargs_273816 = {}
    # Getting the type of 'tf2zpk' (line 334)
    tf2zpk_273806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 'tf2zpk', False)
    # Calling tf2zpk(args, kwargs) (line 334)
    tf2zpk_call_result_273817 = invoke(stypy.reporting.localization.Localization(__file__, 334, 11), tf2zpk_273806, *[ss2tf_call_result_273815], **kwargs_273816)
    
    # Assigning a type to the variable 'stypy_return_type' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type', tf2zpk_call_result_273817)
    
    # ################# End of 'ss2zpk(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ss2zpk' in the type store
    # Getting the type of 'stypy_return_type' (line 307)
    stypy_return_type_273818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273818)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ss2zpk'
    return stypy_return_type_273818

# Assigning a type to the variable 'ss2zpk' (line 307)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'ss2zpk', ss2zpk)

@norecursion
def cont2discrete(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_273819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 37), 'str', 'zoh')
    # Getting the type of 'None' (line 337)
    None_273820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 50), 'None')
    defaults = [str_273819, None_273820]
    # Create a new context for function 'cont2discrete'
    module_type_store = module_type_store.open_function_context('cont2discrete', 337, 0, False)
    
    # Passed parameters checking function
    cont2discrete.stypy_localization = localization
    cont2discrete.stypy_type_of_self = None
    cont2discrete.stypy_type_store = module_type_store
    cont2discrete.stypy_function_name = 'cont2discrete'
    cont2discrete.stypy_param_names_list = ['system', 'dt', 'method', 'alpha']
    cont2discrete.stypy_varargs_param_name = None
    cont2discrete.stypy_kwargs_param_name = None
    cont2discrete.stypy_call_defaults = defaults
    cont2discrete.stypy_call_varargs = varargs
    cont2discrete.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cont2discrete', ['system', 'dt', 'method', 'alpha'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cont2discrete', localization, ['system', 'dt', 'method', 'alpha'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cont2discrete(...)' code ##################

    str_273821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, (-1)), 'str', '\n    Transform a continuous to a discrete state-space system.\n\n    Parameters\n    ----------\n    system : a tuple describing the system or an instance of `lti`\n        The following gives the number of elements in the tuple and\n        the interpretation:\n\n            * 1: (instance of `lti`)\n            * 2: (num, den)\n            * 3: (zeros, poles, gain)\n            * 4: (A, B, C, D)\n\n    dt : float\n        The discretization time step.\n    method : {"gbt", "bilinear", "euler", "backward_diff", "zoh"}, optional\n        Which method to use:\n\n            * gbt: generalized bilinear transformation\n            * bilinear: Tustin\'s approximation ("gbt" with alpha=0.5)\n            * euler: Euler (or forward differencing) method ("gbt" with alpha=0)\n            * backward_diff: Backwards differencing ("gbt" with alpha=1.0)\n            * zoh: zero-order hold (default)\n\n    alpha : float within [0, 1], optional\n        The generalized bilinear transformation weighting parameter, which\n        should only be specified with method="gbt", and is ignored otherwise\n\n    Returns\n    -------\n    sysd : tuple containing the discrete system\n        Based on the input type, the output will be of the form\n\n        * (num, den, dt)   for transfer function input\n        * (zeros, poles, gain, dt)   for zeros-poles-gain input\n        * (A, B, C, D, dt) for state-space system input\n\n    Notes\n    -----\n    By default, the routine uses a Zero-Order Hold (zoh) method to perform\n    the transformation.  Alternatively, a generalized bilinear transformation\n    may be used, which includes the common Tustin\'s bilinear approximation,\n    an Euler\'s method technique, or a backwards differencing technique.\n\n    The Zero-Order Hold (zoh) method is based on [1]_, the generalized bilinear\n    approximation is based on [2]_ and [3]_.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models\n\n    .. [2] http://techteach.no/publications/discretetime_signals_systems/discrete.pdf\n\n    .. [3] G. Zhang, X. Chen, and T. Chen, Digital redesign via the generalized\n        bilinear transformation, Int. J. Control, vol. 82, no. 4, pp. 741-754,\n        2009.\n        (http://www.mypolyuweb.hk/~magzhang/Research/ZCC09_IJC.pdf)\n\n    ')
    
    
    
    # Call to len(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'system' (line 398)
    system_273823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 11), 'system', False)
    # Processing the call keyword arguments (line 398)
    kwargs_273824 = {}
    # Getting the type of 'len' (line 398)
    len_273822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 7), 'len', False)
    # Calling len(args, kwargs) (line 398)
    len_call_result_273825 = invoke(stypy.reporting.localization.Localization(__file__, 398, 7), len_273822, *[system_273823], **kwargs_273824)
    
    int_273826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 22), 'int')
    # Applying the binary operator '==' (line 398)
    result_eq_273827 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 7), '==', len_call_result_273825, int_273826)
    
    # Testing the type of an if condition (line 398)
    if_condition_273828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 398, 4), result_eq_273827)
    # Assigning a type to the variable 'if_condition_273828' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'if_condition_273828', if_condition_273828)
    # SSA begins for if statement (line 398)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to to_discrete(...): (line 399)
    # Processing the call keyword arguments (line 399)
    kwargs_273831 = {}
    # Getting the type of 'system' (line 399)
    system_273829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 15), 'system', False)
    # Obtaining the member 'to_discrete' of a type (line 399)
    to_discrete_273830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 15), system_273829, 'to_discrete')
    # Calling to_discrete(args, kwargs) (line 399)
    to_discrete_call_result_273832 = invoke(stypy.reporting.localization.Localization(__file__, 399, 15), to_discrete_273830, *[], **kwargs_273831)
    
    # Assigning a type to the variable 'stypy_return_type' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'stypy_return_type', to_discrete_call_result_273832)
    # SSA join for if statement (line 398)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 400)
    # Processing the call arguments (line 400)
    # Getting the type of 'system' (line 400)
    system_273834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 11), 'system', False)
    # Processing the call keyword arguments (line 400)
    kwargs_273835 = {}
    # Getting the type of 'len' (line 400)
    len_273833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 7), 'len', False)
    # Calling len(args, kwargs) (line 400)
    len_call_result_273836 = invoke(stypy.reporting.localization.Localization(__file__, 400, 7), len_273833, *[system_273834], **kwargs_273835)
    
    int_273837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 22), 'int')
    # Applying the binary operator '==' (line 400)
    result_eq_273838 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 7), '==', len_call_result_273836, int_273837)
    
    # Testing the type of an if condition (line 400)
    if_condition_273839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 400, 4), result_eq_273838)
    # Assigning a type to the variable 'if_condition_273839' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'if_condition_273839', if_condition_273839)
    # SSA begins for if statement (line 400)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 401):
    
    # Assigning a Call to a Name (line 401):
    
    # Call to cont2discrete(...): (line 401)
    # Processing the call arguments (line 401)
    
    # Call to tf2ss(...): (line 401)
    # Processing the call arguments (line 401)
    
    # Obtaining the type of the subscript
    int_273842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 42), 'int')
    # Getting the type of 'system' (line 401)
    system_273843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 35), 'system', False)
    # Obtaining the member '__getitem__' of a type (line 401)
    getitem___273844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 35), system_273843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 401)
    subscript_call_result_273845 = invoke(stypy.reporting.localization.Localization(__file__, 401, 35), getitem___273844, int_273842)
    
    
    # Obtaining the type of the subscript
    int_273846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 53), 'int')
    # Getting the type of 'system' (line 401)
    system_273847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 46), 'system', False)
    # Obtaining the member '__getitem__' of a type (line 401)
    getitem___273848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 46), system_273847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 401)
    subscript_call_result_273849 = invoke(stypy.reporting.localization.Localization(__file__, 401, 46), getitem___273848, int_273846)
    
    # Processing the call keyword arguments (line 401)
    kwargs_273850 = {}
    # Getting the type of 'tf2ss' (line 401)
    tf2ss_273841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 29), 'tf2ss', False)
    # Calling tf2ss(args, kwargs) (line 401)
    tf2ss_call_result_273851 = invoke(stypy.reporting.localization.Localization(__file__, 401, 29), tf2ss_273841, *[subscript_call_result_273845, subscript_call_result_273849], **kwargs_273850)
    
    # Getting the type of 'dt' (line 401)
    dt_273852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 58), 'dt', False)
    # Processing the call keyword arguments (line 401)
    # Getting the type of 'method' (line 401)
    method_273853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 69), 'method', False)
    keyword_273854 = method_273853
    # Getting the type of 'alpha' (line 402)
    alpha_273855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 35), 'alpha', False)
    keyword_273856 = alpha_273855
    kwargs_273857 = {'alpha': keyword_273856, 'method': keyword_273854}
    # Getting the type of 'cont2discrete' (line 401)
    cont2discrete_273840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'cont2discrete', False)
    # Calling cont2discrete(args, kwargs) (line 401)
    cont2discrete_call_result_273858 = invoke(stypy.reporting.localization.Localization(__file__, 401, 15), cont2discrete_273840, *[tf2ss_call_result_273851, dt_273852], **kwargs_273857)
    
    # Assigning a type to the variable 'sysd' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'sysd', cont2discrete_call_result_273858)
    
    # Call to ss2tf(...): (line 403)
    # Processing the call arguments (line 403)
    
    # Obtaining the type of the subscript
    int_273860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 26), 'int')
    # Getting the type of 'sysd' (line 403)
    sysd_273861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 21), 'sysd', False)
    # Obtaining the member '__getitem__' of a type (line 403)
    getitem___273862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 21), sysd_273861, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 403)
    subscript_call_result_273863 = invoke(stypy.reporting.localization.Localization(__file__, 403, 21), getitem___273862, int_273860)
    
    
    # Obtaining the type of the subscript
    int_273864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 35), 'int')
    # Getting the type of 'sysd' (line 403)
    sysd_273865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 30), 'sysd', False)
    # Obtaining the member '__getitem__' of a type (line 403)
    getitem___273866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 30), sysd_273865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 403)
    subscript_call_result_273867 = invoke(stypy.reporting.localization.Localization(__file__, 403, 30), getitem___273866, int_273864)
    
    
    # Obtaining the type of the subscript
    int_273868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 44), 'int')
    # Getting the type of 'sysd' (line 403)
    sysd_273869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 39), 'sysd', False)
    # Obtaining the member '__getitem__' of a type (line 403)
    getitem___273870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 39), sysd_273869, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 403)
    subscript_call_result_273871 = invoke(stypy.reporting.localization.Localization(__file__, 403, 39), getitem___273870, int_273868)
    
    
    # Obtaining the type of the subscript
    int_273872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 53), 'int')
    # Getting the type of 'sysd' (line 403)
    sysd_273873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 48), 'sysd', False)
    # Obtaining the member '__getitem__' of a type (line 403)
    getitem___273874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 48), sysd_273873, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 403)
    subscript_call_result_273875 = invoke(stypy.reporting.localization.Localization(__file__, 403, 48), getitem___273874, int_273872)
    
    # Processing the call keyword arguments (line 403)
    kwargs_273876 = {}
    # Getting the type of 'ss2tf' (line 403)
    ss2tf_273859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 15), 'ss2tf', False)
    # Calling ss2tf(args, kwargs) (line 403)
    ss2tf_call_result_273877 = invoke(stypy.reporting.localization.Localization(__file__, 403, 15), ss2tf_273859, *[subscript_call_result_273863, subscript_call_result_273867, subscript_call_result_273871, subscript_call_result_273875], **kwargs_273876)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 403)
    tuple_273878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 403)
    # Adding element type (line 403)
    # Getting the type of 'dt' (line 403)
    dt_273879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 60), 'dt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 60), tuple_273878, dt_273879)
    
    # Applying the binary operator '+' (line 403)
    result_add_273880 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 15), '+', ss2tf_call_result_273877, tuple_273878)
    
    # Assigning a type to the variable 'stypy_return_type' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'stypy_return_type', result_add_273880)
    # SSA branch for the else part of an if statement (line 400)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'system' (line 404)
    system_273882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 13), 'system', False)
    # Processing the call keyword arguments (line 404)
    kwargs_273883 = {}
    # Getting the type of 'len' (line 404)
    len_273881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 9), 'len', False)
    # Calling len(args, kwargs) (line 404)
    len_call_result_273884 = invoke(stypy.reporting.localization.Localization(__file__, 404, 9), len_273881, *[system_273882], **kwargs_273883)
    
    int_273885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 24), 'int')
    # Applying the binary operator '==' (line 404)
    result_eq_273886 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 9), '==', len_call_result_273884, int_273885)
    
    # Testing the type of an if condition (line 404)
    if_condition_273887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 9), result_eq_273886)
    # Assigning a type to the variable 'if_condition_273887' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 9), 'if_condition_273887', if_condition_273887)
    # SSA begins for if statement (line 404)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 405):
    
    # Assigning a Call to a Name (line 405):
    
    # Call to cont2discrete(...): (line 405)
    # Processing the call arguments (line 405)
    
    # Call to zpk2ss(...): (line 405)
    # Processing the call arguments (line 405)
    
    # Obtaining the type of the subscript
    int_273890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 43), 'int')
    # Getting the type of 'system' (line 405)
    system_273891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 36), 'system', False)
    # Obtaining the member '__getitem__' of a type (line 405)
    getitem___273892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 36), system_273891, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 405)
    subscript_call_result_273893 = invoke(stypy.reporting.localization.Localization(__file__, 405, 36), getitem___273892, int_273890)
    
    
    # Obtaining the type of the subscript
    int_273894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 54), 'int')
    # Getting the type of 'system' (line 405)
    system_273895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 47), 'system', False)
    # Obtaining the member '__getitem__' of a type (line 405)
    getitem___273896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 47), system_273895, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 405)
    subscript_call_result_273897 = invoke(stypy.reporting.localization.Localization(__file__, 405, 47), getitem___273896, int_273894)
    
    
    # Obtaining the type of the subscript
    int_273898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 65), 'int')
    # Getting the type of 'system' (line 405)
    system_273899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 58), 'system', False)
    # Obtaining the member '__getitem__' of a type (line 405)
    getitem___273900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 58), system_273899, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 405)
    subscript_call_result_273901 = invoke(stypy.reporting.localization.Localization(__file__, 405, 58), getitem___273900, int_273898)
    
    # Processing the call keyword arguments (line 405)
    kwargs_273902 = {}
    # Getting the type of 'zpk2ss' (line 405)
    zpk2ss_273889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 29), 'zpk2ss', False)
    # Calling zpk2ss(args, kwargs) (line 405)
    zpk2ss_call_result_273903 = invoke(stypy.reporting.localization.Localization(__file__, 405, 29), zpk2ss_273889, *[subscript_call_result_273893, subscript_call_result_273897, subscript_call_result_273901], **kwargs_273902)
    
    # Getting the type of 'dt' (line 405)
    dt_273904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 70), 'dt', False)
    # Processing the call keyword arguments (line 405)
    # Getting the type of 'method' (line 406)
    method_273905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 36), 'method', False)
    keyword_273906 = method_273905
    # Getting the type of 'alpha' (line 406)
    alpha_273907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 50), 'alpha', False)
    keyword_273908 = alpha_273907
    kwargs_273909 = {'alpha': keyword_273908, 'method': keyword_273906}
    # Getting the type of 'cont2discrete' (line 405)
    cont2discrete_273888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 15), 'cont2discrete', False)
    # Calling cont2discrete(args, kwargs) (line 405)
    cont2discrete_call_result_273910 = invoke(stypy.reporting.localization.Localization(__file__, 405, 15), cont2discrete_273888, *[zpk2ss_call_result_273903, dt_273904], **kwargs_273909)
    
    # Assigning a type to the variable 'sysd' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'sysd', cont2discrete_call_result_273910)
    
    # Call to ss2zpk(...): (line 407)
    # Processing the call arguments (line 407)
    
    # Obtaining the type of the subscript
    int_273912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 27), 'int')
    # Getting the type of 'sysd' (line 407)
    sysd_273913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 22), 'sysd', False)
    # Obtaining the member '__getitem__' of a type (line 407)
    getitem___273914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 22), sysd_273913, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 407)
    subscript_call_result_273915 = invoke(stypy.reporting.localization.Localization(__file__, 407, 22), getitem___273914, int_273912)
    
    
    # Obtaining the type of the subscript
    int_273916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 36), 'int')
    # Getting the type of 'sysd' (line 407)
    sysd_273917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 31), 'sysd', False)
    # Obtaining the member '__getitem__' of a type (line 407)
    getitem___273918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 31), sysd_273917, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 407)
    subscript_call_result_273919 = invoke(stypy.reporting.localization.Localization(__file__, 407, 31), getitem___273918, int_273916)
    
    
    # Obtaining the type of the subscript
    int_273920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 45), 'int')
    # Getting the type of 'sysd' (line 407)
    sysd_273921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 40), 'sysd', False)
    # Obtaining the member '__getitem__' of a type (line 407)
    getitem___273922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 40), sysd_273921, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 407)
    subscript_call_result_273923 = invoke(stypy.reporting.localization.Localization(__file__, 407, 40), getitem___273922, int_273920)
    
    
    # Obtaining the type of the subscript
    int_273924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 54), 'int')
    # Getting the type of 'sysd' (line 407)
    sysd_273925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 49), 'sysd', False)
    # Obtaining the member '__getitem__' of a type (line 407)
    getitem___273926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 49), sysd_273925, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 407)
    subscript_call_result_273927 = invoke(stypy.reporting.localization.Localization(__file__, 407, 49), getitem___273926, int_273924)
    
    # Processing the call keyword arguments (line 407)
    kwargs_273928 = {}
    # Getting the type of 'ss2zpk' (line 407)
    ss2zpk_273911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 15), 'ss2zpk', False)
    # Calling ss2zpk(args, kwargs) (line 407)
    ss2zpk_call_result_273929 = invoke(stypy.reporting.localization.Localization(__file__, 407, 15), ss2zpk_273911, *[subscript_call_result_273915, subscript_call_result_273919, subscript_call_result_273923, subscript_call_result_273927], **kwargs_273928)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 407)
    tuple_273930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 61), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 407)
    # Adding element type (line 407)
    # Getting the type of 'dt' (line 407)
    dt_273931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 61), 'dt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 61), tuple_273930, dt_273931)
    
    # Applying the binary operator '+' (line 407)
    result_add_273932 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 15), '+', ss2zpk_call_result_273929, tuple_273930)
    
    # Assigning a type to the variable 'stypy_return_type' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'stypy_return_type', result_add_273932)
    # SSA branch for the else part of an if statement (line 404)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'system' (line 408)
    system_273934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 13), 'system', False)
    # Processing the call keyword arguments (line 408)
    kwargs_273935 = {}
    # Getting the type of 'len' (line 408)
    len_273933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 9), 'len', False)
    # Calling len(args, kwargs) (line 408)
    len_call_result_273936 = invoke(stypy.reporting.localization.Localization(__file__, 408, 9), len_273933, *[system_273934], **kwargs_273935)
    
    int_273937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 24), 'int')
    # Applying the binary operator '==' (line 408)
    result_eq_273938 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 9), '==', len_call_result_273936, int_273937)
    
    # Testing the type of an if condition (line 408)
    if_condition_273939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 408, 9), result_eq_273938)
    # Assigning a type to the variable 'if_condition_273939' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 9), 'if_condition_273939', if_condition_273939)
    # SSA begins for if statement (line 408)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Tuple (line 409):
    
    # Assigning a Subscript to a Name (line 409):
    
    # Obtaining the type of the subscript
    int_273940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 8), 'int')
    # Getting the type of 'system' (line 409)
    system_273941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 21), 'system')
    # Obtaining the member '__getitem__' of a type (line 409)
    getitem___273942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), system_273941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 409)
    subscript_call_result_273943 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), getitem___273942, int_273940)
    
    # Assigning a type to the variable 'tuple_var_assignment_273024' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_var_assignment_273024', subscript_call_result_273943)
    
    # Assigning a Subscript to a Name (line 409):
    
    # Obtaining the type of the subscript
    int_273944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 8), 'int')
    # Getting the type of 'system' (line 409)
    system_273945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 21), 'system')
    # Obtaining the member '__getitem__' of a type (line 409)
    getitem___273946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), system_273945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 409)
    subscript_call_result_273947 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), getitem___273946, int_273944)
    
    # Assigning a type to the variable 'tuple_var_assignment_273025' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_var_assignment_273025', subscript_call_result_273947)
    
    # Assigning a Subscript to a Name (line 409):
    
    # Obtaining the type of the subscript
    int_273948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 8), 'int')
    # Getting the type of 'system' (line 409)
    system_273949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 21), 'system')
    # Obtaining the member '__getitem__' of a type (line 409)
    getitem___273950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), system_273949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 409)
    subscript_call_result_273951 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), getitem___273950, int_273948)
    
    # Assigning a type to the variable 'tuple_var_assignment_273026' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_var_assignment_273026', subscript_call_result_273951)
    
    # Assigning a Subscript to a Name (line 409):
    
    # Obtaining the type of the subscript
    int_273952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 8), 'int')
    # Getting the type of 'system' (line 409)
    system_273953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 21), 'system')
    # Obtaining the member '__getitem__' of a type (line 409)
    getitem___273954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), system_273953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 409)
    subscript_call_result_273955 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), getitem___273954, int_273952)
    
    # Assigning a type to the variable 'tuple_var_assignment_273027' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_var_assignment_273027', subscript_call_result_273955)
    
    # Assigning a Name to a Name (line 409):
    # Getting the type of 'tuple_var_assignment_273024' (line 409)
    tuple_var_assignment_273024_273956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_var_assignment_273024')
    # Assigning a type to the variable 'a' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'a', tuple_var_assignment_273024_273956)
    
    # Assigning a Name to a Name (line 409):
    # Getting the type of 'tuple_var_assignment_273025' (line 409)
    tuple_var_assignment_273025_273957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_var_assignment_273025')
    # Assigning a type to the variable 'b' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), 'b', tuple_var_assignment_273025_273957)
    
    # Assigning a Name to a Name (line 409):
    # Getting the type of 'tuple_var_assignment_273026' (line 409)
    tuple_var_assignment_273026_273958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_var_assignment_273026')
    # Assigning a type to the variable 'c' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 14), 'c', tuple_var_assignment_273026_273958)
    
    # Assigning a Name to a Name (line 409):
    # Getting the type of 'tuple_var_assignment_273027' (line 409)
    tuple_var_assignment_273027_273959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'tuple_var_assignment_273027')
    # Assigning a type to the variable 'd' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 17), 'd', tuple_var_assignment_273027_273959)
    # SSA branch for the else part of an if statement (line 408)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 411)
    # Processing the call arguments (line 411)
    str_273961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 25), 'str', 'First argument must either be a tuple of 2 (tf), 3 (zpk), or 4 (ss) arrays.')
    # Processing the call keyword arguments (line 411)
    kwargs_273962 = {}
    # Getting the type of 'ValueError' (line 411)
    ValueError_273960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 411)
    ValueError_call_result_273963 = invoke(stypy.reporting.localization.Localization(__file__, 411, 14), ValueError_273960, *[str_273961], **kwargs_273962)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 411, 8), ValueError_call_result_273963, 'raise parameter', BaseException)
    # SSA join for if statement (line 408)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 404)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 400)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'method' (line 414)
    method_273964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 7), 'method')
    str_273965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 17), 'str', 'gbt')
    # Applying the binary operator '==' (line 414)
    result_eq_273966 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 7), '==', method_273964, str_273965)
    
    # Testing the type of an if condition (line 414)
    if_condition_273967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 414, 4), result_eq_273966)
    # Assigning a type to the variable 'if_condition_273967' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'if_condition_273967', if_condition_273967)
    # SSA begins for if statement (line 414)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 415)
    # Getting the type of 'alpha' (line 415)
    alpha_273968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 11), 'alpha')
    # Getting the type of 'None' (line 415)
    None_273969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 20), 'None')
    
    (may_be_273970, more_types_in_union_273971) = may_be_none(alpha_273968, None_273969)

    if may_be_273970:

        if more_types_in_union_273971:
            # Runtime conditional SSA (line 415)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 416)
        # Processing the call arguments (line 416)
        str_273973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 29), 'str', 'Alpha parameter must be specified for the generalized bilinear transform (gbt) method')
        # Processing the call keyword arguments (line 416)
        kwargs_273974 = {}
        # Getting the type of 'ValueError' (line 416)
        ValueError_273972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 416)
        ValueError_call_result_273975 = invoke(stypy.reporting.localization.Localization(__file__, 416, 18), ValueError_273972, *[str_273973], **kwargs_273974)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 416, 12), ValueError_call_result_273975, 'raise parameter', BaseException)

        if more_types_in_union_273971:
            # Runtime conditional SSA for else branch (line 415)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_273970) or more_types_in_union_273971):
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'alpha' (line 418)
        alpha_273976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 13), 'alpha')
        int_273977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 21), 'int')
        # Applying the binary operator '<' (line 418)
        result_lt_273978 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 13), '<', alpha_273976, int_273977)
        
        
        # Getting the type of 'alpha' (line 418)
        alpha_273979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 26), 'alpha')
        int_273980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 34), 'int')
        # Applying the binary operator '>' (line 418)
        result_gt_273981 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 26), '>', alpha_273979, int_273980)
        
        # Applying the binary operator 'or' (line 418)
        result_or_keyword_273982 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 13), 'or', result_lt_273978, result_gt_273981)
        
        # Testing the type of an if condition (line 418)
        if_condition_273983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 13), result_or_keyword_273982)
        # Assigning a type to the variable 'if_condition_273983' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 13), 'if_condition_273983', if_condition_273983)
        # SSA begins for if statement (line 418)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 419)
        # Processing the call arguments (line 419)
        str_273985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 29), 'str', 'Alpha parameter must be within the interval [0,1] for the gbt method')
        # Processing the call keyword arguments (line 419)
        kwargs_273986 = {}
        # Getting the type of 'ValueError' (line 419)
        ValueError_273984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 419)
        ValueError_call_result_273987 = invoke(stypy.reporting.localization.Localization(__file__, 419, 18), ValueError_273984, *[str_273985], **kwargs_273986)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 419, 12), ValueError_call_result_273987, 'raise parameter', BaseException)
        # SSA join for if statement (line 418)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_273970 and more_types_in_union_273971):
            # SSA join for if statement (line 415)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 414)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'method' (line 422)
    method_273988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 7), 'method')
    str_273989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 17), 'str', 'gbt')
    # Applying the binary operator '==' (line 422)
    result_eq_273990 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 7), '==', method_273988, str_273989)
    
    # Testing the type of an if condition (line 422)
    if_condition_273991 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 4), result_eq_273990)
    # Assigning a type to the variable 'if_condition_273991' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'if_condition_273991', if_condition_273991)
    # SSA begins for if statement (line 422)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 424):
    
    # Assigning a BinOp to a Name (line 424):
    
    # Call to eye(...): (line 424)
    # Processing the call arguments (line 424)
    
    # Obtaining the type of the subscript
    int_273994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 29), 'int')
    # Getting the type of 'a' (line 424)
    a_273995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 21), 'a', False)
    # Obtaining the member 'shape' of a type (line 424)
    shape_273996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 21), a_273995, 'shape')
    # Obtaining the member '__getitem__' of a type (line 424)
    getitem___273997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 21), shape_273996, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 424)
    subscript_call_result_273998 = invoke(stypy.reporting.localization.Localization(__file__, 424, 21), getitem___273997, int_273994)
    
    # Processing the call keyword arguments (line 424)
    kwargs_273999 = {}
    # Getting the type of 'np' (line 424)
    np_273992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 14), 'np', False)
    # Obtaining the member 'eye' of a type (line 424)
    eye_273993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 14), np_273992, 'eye')
    # Calling eye(args, kwargs) (line 424)
    eye_call_result_274000 = invoke(stypy.reporting.localization.Localization(__file__, 424, 14), eye_273993, *[subscript_call_result_273998], **kwargs_273999)
    
    # Getting the type of 'alpha' (line 424)
    alpha_274001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 35), 'alpha')
    # Getting the type of 'dt' (line 424)
    dt_274002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 41), 'dt')
    # Applying the binary operator '*' (line 424)
    result_mul_274003 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 35), '*', alpha_274001, dt_274002)
    
    # Getting the type of 'a' (line 424)
    a_274004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 44), 'a')
    # Applying the binary operator '*' (line 424)
    result_mul_274005 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 43), '*', result_mul_274003, a_274004)
    
    # Applying the binary operator '-' (line 424)
    result_sub_274006 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 14), '-', eye_call_result_274000, result_mul_274005)
    
    # Assigning a type to the variable 'ima' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'ima', result_sub_274006)
    
    # Assigning a Call to a Name (line 425):
    
    # Assigning a Call to a Name (line 425):
    
    # Call to solve(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'ima' (line 425)
    ima_274009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 26), 'ima', False)
    
    # Call to eye(...): (line 425)
    # Processing the call arguments (line 425)
    
    # Obtaining the type of the subscript
    int_274012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 46), 'int')
    # Getting the type of 'a' (line 425)
    a_274013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 38), 'a', False)
    # Obtaining the member 'shape' of a type (line 425)
    shape_274014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 38), a_274013, 'shape')
    # Obtaining the member '__getitem__' of a type (line 425)
    getitem___274015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 38), shape_274014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 425)
    subscript_call_result_274016 = invoke(stypy.reporting.localization.Localization(__file__, 425, 38), getitem___274015, int_274012)
    
    # Processing the call keyword arguments (line 425)
    kwargs_274017 = {}
    # Getting the type of 'np' (line 425)
    np_274010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 31), 'np', False)
    # Obtaining the member 'eye' of a type (line 425)
    eye_274011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 31), np_274010, 'eye')
    # Calling eye(args, kwargs) (line 425)
    eye_call_result_274018 = invoke(stypy.reporting.localization.Localization(__file__, 425, 31), eye_274011, *[subscript_call_result_274016], **kwargs_274017)
    
    float_274019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 53), 'float')
    # Getting the type of 'alpha' (line 425)
    alpha_274020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 57), 'alpha', False)
    # Applying the binary operator '-' (line 425)
    result_sub_274021 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 53), '-', float_274019, alpha_274020)
    
    # Getting the type of 'dt' (line 425)
    dt_274022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 64), 'dt', False)
    # Applying the binary operator '*' (line 425)
    result_mul_274023 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 52), '*', result_sub_274021, dt_274022)
    
    # Getting the type of 'a' (line 425)
    a_274024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 67), 'a', False)
    # Applying the binary operator '*' (line 425)
    result_mul_274025 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 66), '*', result_mul_274023, a_274024)
    
    # Applying the binary operator '+' (line 425)
    result_add_274026 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 31), '+', eye_call_result_274018, result_mul_274025)
    
    # Processing the call keyword arguments (line 425)
    kwargs_274027 = {}
    # Getting the type of 'linalg' (line 425)
    linalg_274007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 13), 'linalg', False)
    # Obtaining the member 'solve' of a type (line 425)
    solve_274008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 13), linalg_274007, 'solve')
    # Calling solve(args, kwargs) (line 425)
    solve_call_result_274028 = invoke(stypy.reporting.localization.Localization(__file__, 425, 13), solve_274008, *[ima_274009, result_add_274026], **kwargs_274027)
    
    # Assigning a type to the variable 'ad' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'ad', solve_call_result_274028)
    
    # Assigning a Call to a Name (line 426):
    
    # Assigning a Call to a Name (line 426):
    
    # Call to solve(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'ima' (line 426)
    ima_274031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 26), 'ima', False)
    # Getting the type of 'dt' (line 426)
    dt_274032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 31), 'dt', False)
    # Getting the type of 'b' (line 426)
    b_274033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 34), 'b', False)
    # Applying the binary operator '*' (line 426)
    result_mul_274034 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 31), '*', dt_274032, b_274033)
    
    # Processing the call keyword arguments (line 426)
    kwargs_274035 = {}
    # Getting the type of 'linalg' (line 426)
    linalg_274029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 13), 'linalg', False)
    # Obtaining the member 'solve' of a type (line 426)
    solve_274030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 13), linalg_274029, 'solve')
    # Calling solve(args, kwargs) (line 426)
    solve_call_result_274036 = invoke(stypy.reporting.localization.Localization(__file__, 426, 13), solve_274030, *[ima_274031, result_mul_274034], **kwargs_274035)
    
    # Assigning a type to the variable 'bd' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'bd', solve_call_result_274036)
    
    # Assigning a Call to a Name (line 429):
    
    # Assigning a Call to a Name (line 429):
    
    # Call to solve(...): (line 429)
    # Processing the call arguments (line 429)
    
    # Call to transpose(...): (line 429)
    # Processing the call keyword arguments (line 429)
    kwargs_274041 = {}
    # Getting the type of 'ima' (line 429)
    ima_274039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 26), 'ima', False)
    # Obtaining the member 'transpose' of a type (line 429)
    transpose_274040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 26), ima_274039, 'transpose')
    # Calling transpose(args, kwargs) (line 429)
    transpose_call_result_274042 = invoke(stypy.reporting.localization.Localization(__file__, 429, 26), transpose_274040, *[], **kwargs_274041)
    
    
    # Call to transpose(...): (line 429)
    # Processing the call keyword arguments (line 429)
    kwargs_274045 = {}
    # Getting the type of 'c' (line 429)
    c_274043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 43), 'c', False)
    # Obtaining the member 'transpose' of a type (line 429)
    transpose_274044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 43), c_274043, 'transpose')
    # Calling transpose(args, kwargs) (line 429)
    transpose_call_result_274046 = invoke(stypy.reporting.localization.Localization(__file__, 429, 43), transpose_274044, *[], **kwargs_274045)
    
    # Processing the call keyword arguments (line 429)
    kwargs_274047 = {}
    # Getting the type of 'linalg' (line 429)
    linalg_274037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 13), 'linalg', False)
    # Obtaining the member 'solve' of a type (line 429)
    solve_274038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 13), linalg_274037, 'solve')
    # Calling solve(args, kwargs) (line 429)
    solve_call_result_274048 = invoke(stypy.reporting.localization.Localization(__file__, 429, 13), solve_274038, *[transpose_call_result_274042, transpose_call_result_274046], **kwargs_274047)
    
    # Assigning a type to the variable 'cd' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'cd', solve_call_result_274048)
    
    # Assigning a Call to a Name (line 430):
    
    # Assigning a Call to a Name (line 430):
    
    # Call to transpose(...): (line 430)
    # Processing the call keyword arguments (line 430)
    kwargs_274051 = {}
    # Getting the type of 'cd' (line 430)
    cd_274049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 13), 'cd', False)
    # Obtaining the member 'transpose' of a type (line 430)
    transpose_274050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 13), cd_274049, 'transpose')
    # Calling transpose(args, kwargs) (line 430)
    transpose_call_result_274052 = invoke(stypy.reporting.localization.Localization(__file__, 430, 13), transpose_274050, *[], **kwargs_274051)
    
    # Assigning a type to the variable 'cd' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'cd', transpose_call_result_274052)
    
    # Assigning a BinOp to a Name (line 431):
    
    # Assigning a BinOp to a Name (line 431):
    # Getting the type of 'd' (line 431)
    d_274053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 13), 'd')
    # Getting the type of 'alpha' (line 431)
    alpha_274054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 17), 'alpha')
    
    # Call to dot(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'c' (line 431)
    c_274057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 30), 'c', False)
    # Getting the type of 'bd' (line 431)
    bd_274058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 33), 'bd', False)
    # Processing the call keyword arguments (line 431)
    kwargs_274059 = {}
    # Getting the type of 'np' (line 431)
    np_274055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 23), 'np', False)
    # Obtaining the member 'dot' of a type (line 431)
    dot_274056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 23), np_274055, 'dot')
    # Calling dot(args, kwargs) (line 431)
    dot_call_result_274060 = invoke(stypy.reporting.localization.Localization(__file__, 431, 23), dot_274056, *[c_274057, bd_274058], **kwargs_274059)
    
    # Applying the binary operator '*' (line 431)
    result_mul_274061 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 17), '*', alpha_274054, dot_call_result_274060)
    
    # Applying the binary operator '+' (line 431)
    result_add_274062 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 13), '+', d_274053, result_mul_274061)
    
    # Assigning a type to the variable 'dd' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'dd', result_add_274062)
    # SSA branch for the else part of an if statement (line 422)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'method' (line 433)
    method_274063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 9), 'method')
    str_274064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 19), 'str', 'bilinear')
    # Applying the binary operator '==' (line 433)
    result_eq_274065 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 9), '==', method_274063, str_274064)
    
    
    # Getting the type of 'method' (line 433)
    method_274066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 33), 'method')
    str_274067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 43), 'str', 'tustin')
    # Applying the binary operator '==' (line 433)
    result_eq_274068 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 33), '==', method_274066, str_274067)
    
    # Applying the binary operator 'or' (line 433)
    result_or_keyword_274069 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 9), 'or', result_eq_274065, result_eq_274068)
    
    # Testing the type of an if condition (line 433)
    if_condition_274070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 9), result_or_keyword_274069)
    # Assigning a type to the variable 'if_condition_274070' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 9), 'if_condition_274070', if_condition_274070)
    # SSA begins for if statement (line 433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cont2discrete(...): (line 434)
    # Processing the call arguments (line 434)
    # Getting the type of 'system' (line 434)
    system_274072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 29), 'system', False)
    # Getting the type of 'dt' (line 434)
    dt_274073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 37), 'dt', False)
    # Processing the call keyword arguments (line 434)
    str_274074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 48), 'str', 'gbt')
    keyword_274075 = str_274074
    float_274076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 61), 'float')
    keyword_274077 = float_274076
    kwargs_274078 = {'alpha': keyword_274077, 'method': keyword_274075}
    # Getting the type of 'cont2discrete' (line 434)
    cont2discrete_274071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'cont2discrete', False)
    # Calling cont2discrete(args, kwargs) (line 434)
    cont2discrete_call_result_274079 = invoke(stypy.reporting.localization.Localization(__file__, 434, 15), cont2discrete_274071, *[system_274072, dt_274073], **kwargs_274078)
    
    # Assigning a type to the variable 'stypy_return_type' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'stypy_return_type', cont2discrete_call_result_274079)
    # SSA branch for the else part of an if statement (line 433)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'method' (line 436)
    method_274080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 9), 'method')
    str_274081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 19), 'str', 'euler')
    # Applying the binary operator '==' (line 436)
    result_eq_274082 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 9), '==', method_274080, str_274081)
    
    
    # Getting the type of 'method' (line 436)
    method_274083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 30), 'method')
    str_274084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 40), 'str', 'forward_diff')
    # Applying the binary operator '==' (line 436)
    result_eq_274085 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 30), '==', method_274083, str_274084)
    
    # Applying the binary operator 'or' (line 436)
    result_or_keyword_274086 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 9), 'or', result_eq_274082, result_eq_274085)
    
    # Testing the type of an if condition (line 436)
    if_condition_274087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 9), result_or_keyword_274086)
    # Assigning a type to the variable 'if_condition_274087' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 9), 'if_condition_274087', if_condition_274087)
    # SSA begins for if statement (line 436)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cont2discrete(...): (line 437)
    # Processing the call arguments (line 437)
    # Getting the type of 'system' (line 437)
    system_274089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 29), 'system', False)
    # Getting the type of 'dt' (line 437)
    dt_274090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 37), 'dt', False)
    # Processing the call keyword arguments (line 437)
    str_274091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 48), 'str', 'gbt')
    keyword_274092 = str_274091
    float_274093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 61), 'float')
    keyword_274094 = float_274093
    kwargs_274095 = {'alpha': keyword_274094, 'method': keyword_274092}
    # Getting the type of 'cont2discrete' (line 437)
    cont2discrete_274088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'cont2discrete', False)
    # Calling cont2discrete(args, kwargs) (line 437)
    cont2discrete_call_result_274096 = invoke(stypy.reporting.localization.Localization(__file__, 437, 15), cont2discrete_274088, *[system_274089, dt_274090], **kwargs_274095)
    
    # Assigning a type to the variable 'stypy_return_type' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'stypy_return_type', cont2discrete_call_result_274096)
    # SSA branch for the else part of an if statement (line 436)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 439)
    method_274097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 9), 'method')
    str_274098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 19), 'str', 'backward_diff')
    # Applying the binary operator '==' (line 439)
    result_eq_274099 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 9), '==', method_274097, str_274098)
    
    # Testing the type of an if condition (line 439)
    if_condition_274100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 439, 9), result_eq_274099)
    # Assigning a type to the variable 'if_condition_274100' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 9), 'if_condition_274100', if_condition_274100)
    # SSA begins for if statement (line 439)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cont2discrete(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'system' (line 440)
    system_274102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 29), 'system', False)
    # Getting the type of 'dt' (line 440)
    dt_274103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 37), 'dt', False)
    # Processing the call keyword arguments (line 440)
    str_274104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 48), 'str', 'gbt')
    keyword_274105 = str_274104
    float_274106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 61), 'float')
    keyword_274107 = float_274106
    kwargs_274108 = {'alpha': keyword_274107, 'method': keyword_274105}
    # Getting the type of 'cont2discrete' (line 440)
    cont2discrete_274101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'cont2discrete', False)
    # Calling cont2discrete(args, kwargs) (line 440)
    cont2discrete_call_result_274109 = invoke(stypy.reporting.localization.Localization(__file__, 440, 15), cont2discrete_274101, *[system_274102, dt_274103], **kwargs_274108)
    
    # Assigning a type to the variable 'stypy_return_type' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'stypy_return_type', cont2discrete_call_result_274109)
    # SSA branch for the else part of an if statement (line 439)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 442)
    method_274110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 9), 'method')
    str_274111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 19), 'str', 'zoh')
    # Applying the binary operator '==' (line 442)
    result_eq_274112 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 9), '==', method_274110, str_274111)
    
    # Testing the type of an if condition (line 442)
    if_condition_274113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 9), result_eq_274112)
    # Assigning a type to the variable 'if_condition_274113' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 9), 'if_condition_274113', if_condition_274113)
    # SSA begins for if statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 444):
    
    # Assigning a Call to a Name (line 444):
    
    # Call to hstack(...): (line 444)
    # Processing the call arguments (line 444)
    
    # Obtaining an instance of the builtin type 'tuple' (line 444)
    tuple_274116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 444)
    # Adding element type (line 444)
    # Getting the type of 'a' (line 444)
    a_274117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 30), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 30), tuple_274116, a_274117)
    # Adding element type (line 444)
    # Getting the type of 'b' (line 444)
    b_274118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 33), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 30), tuple_274116, b_274118)
    
    # Processing the call keyword arguments (line 444)
    kwargs_274119 = {}
    # Getting the type of 'np' (line 444)
    np_274114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 19), 'np', False)
    # Obtaining the member 'hstack' of a type (line 444)
    hstack_274115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 19), np_274114, 'hstack')
    # Calling hstack(args, kwargs) (line 444)
    hstack_call_result_274120 = invoke(stypy.reporting.localization.Localization(__file__, 444, 19), hstack_274115, *[tuple_274116], **kwargs_274119)
    
    # Assigning a type to the variable 'em_upper' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'em_upper', hstack_call_result_274120)
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to hstack(...): (line 447)
    # Processing the call arguments (line 447)
    
    # Obtaining an instance of the builtin type 'tuple' (line 447)
    tuple_274123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 447)
    # Adding element type (line 447)
    
    # Call to zeros(...): (line 447)
    # Processing the call arguments (line 447)
    
    # Obtaining an instance of the builtin type 'tuple' (line 447)
    tuple_274126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 447)
    # Adding element type (line 447)
    
    # Obtaining the type of the subscript
    int_274127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 48), 'int')
    # Getting the type of 'b' (line 447)
    b_274128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 40), 'b', False)
    # Obtaining the member 'shape' of a type (line 447)
    shape_274129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 40), b_274128, 'shape')
    # Obtaining the member '__getitem__' of a type (line 447)
    getitem___274130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 40), shape_274129, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 447)
    subscript_call_result_274131 = invoke(stypy.reporting.localization.Localization(__file__, 447, 40), getitem___274130, int_274127)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 40), tuple_274126, subscript_call_result_274131)
    # Adding element type (line 447)
    
    # Obtaining the type of the subscript
    int_274132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 60), 'int')
    # Getting the type of 'a' (line 447)
    a_274133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 52), 'a', False)
    # Obtaining the member 'shape' of a type (line 447)
    shape_274134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 52), a_274133, 'shape')
    # Obtaining the member '__getitem__' of a type (line 447)
    getitem___274135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 52), shape_274134, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 447)
    subscript_call_result_274136 = invoke(stypy.reporting.localization.Localization(__file__, 447, 52), getitem___274135, int_274132)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 40), tuple_274126, subscript_call_result_274136)
    
    # Processing the call keyword arguments (line 447)
    kwargs_274137 = {}
    # Getting the type of 'np' (line 447)
    np_274124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 30), 'np', False)
    # Obtaining the member 'zeros' of a type (line 447)
    zeros_274125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 30), np_274124, 'zeros')
    # Calling zeros(args, kwargs) (line 447)
    zeros_call_result_274138 = invoke(stypy.reporting.localization.Localization(__file__, 447, 30), zeros_274125, *[tuple_274126], **kwargs_274137)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 30), tuple_274123, zeros_call_result_274138)
    # Adding element type (line 447)
    
    # Call to zeros(...): (line 448)
    # Processing the call arguments (line 448)
    
    # Obtaining an instance of the builtin type 'tuple' (line 448)
    tuple_274141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 448)
    # Adding element type (line 448)
    
    # Obtaining the type of the subscript
    int_274142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 48), 'int')
    # Getting the type of 'b' (line 448)
    b_274143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 40), 'b', False)
    # Obtaining the member 'shape' of a type (line 448)
    shape_274144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 40), b_274143, 'shape')
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___274145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 40), shape_274144, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_274146 = invoke(stypy.reporting.localization.Localization(__file__, 448, 40), getitem___274145, int_274142)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 40), tuple_274141, subscript_call_result_274146)
    # Adding element type (line 448)
    
    # Obtaining the type of the subscript
    int_274147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 60), 'int')
    # Getting the type of 'b' (line 448)
    b_274148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 52), 'b', False)
    # Obtaining the member 'shape' of a type (line 448)
    shape_274149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 52), b_274148, 'shape')
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___274150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 52), shape_274149, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_274151 = invoke(stypy.reporting.localization.Localization(__file__, 448, 52), getitem___274150, int_274147)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 40), tuple_274141, subscript_call_result_274151)
    
    # Processing the call keyword arguments (line 448)
    kwargs_274152 = {}
    # Getting the type of 'np' (line 448)
    np_274139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 30), 'np', False)
    # Obtaining the member 'zeros' of a type (line 448)
    zeros_274140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 30), np_274139, 'zeros')
    # Calling zeros(args, kwargs) (line 448)
    zeros_call_result_274153 = invoke(stypy.reporting.localization.Localization(__file__, 448, 30), zeros_274140, *[tuple_274141], **kwargs_274152)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 30), tuple_274123, zeros_call_result_274153)
    
    # Processing the call keyword arguments (line 447)
    kwargs_274154 = {}
    # Getting the type of 'np' (line 447)
    np_274121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 19), 'np', False)
    # Obtaining the member 'hstack' of a type (line 447)
    hstack_274122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 19), np_274121, 'hstack')
    # Calling hstack(args, kwargs) (line 447)
    hstack_call_result_274155 = invoke(stypy.reporting.localization.Localization(__file__, 447, 19), hstack_274122, *[tuple_274123], **kwargs_274154)
    
    # Assigning a type to the variable 'em_lower' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'em_lower', hstack_call_result_274155)
    
    # Assigning a Call to a Name (line 450):
    
    # Assigning a Call to a Name (line 450):
    
    # Call to vstack(...): (line 450)
    # Processing the call arguments (line 450)
    
    # Obtaining an instance of the builtin type 'tuple' (line 450)
    tuple_274158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 450)
    # Adding element type (line 450)
    # Getting the type of 'em_upper' (line 450)
    em_upper_274159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 24), 'em_upper', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 24), tuple_274158, em_upper_274159)
    # Adding element type (line 450)
    # Getting the type of 'em_lower' (line 450)
    em_lower_274160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 34), 'em_lower', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 24), tuple_274158, em_lower_274160)
    
    # Processing the call keyword arguments (line 450)
    kwargs_274161 = {}
    # Getting the type of 'np' (line 450)
    np_274156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 13), 'np', False)
    # Obtaining the member 'vstack' of a type (line 450)
    vstack_274157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 13), np_274156, 'vstack')
    # Calling vstack(args, kwargs) (line 450)
    vstack_call_result_274162 = invoke(stypy.reporting.localization.Localization(__file__, 450, 13), vstack_274157, *[tuple_274158], **kwargs_274161)
    
    # Assigning a type to the variable 'em' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'em', vstack_call_result_274162)
    
    # Assigning a Call to a Name (line 451):
    
    # Assigning a Call to a Name (line 451):
    
    # Call to expm(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'dt' (line 451)
    dt_274165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 25), 'dt', False)
    # Getting the type of 'em' (line 451)
    em_274166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 30), 'em', False)
    # Applying the binary operator '*' (line 451)
    result_mul_274167 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 25), '*', dt_274165, em_274166)
    
    # Processing the call keyword arguments (line 451)
    kwargs_274168 = {}
    # Getting the type of 'linalg' (line 451)
    linalg_274163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 13), 'linalg', False)
    # Obtaining the member 'expm' of a type (line 451)
    expm_274164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 13), linalg_274163, 'expm')
    # Calling expm(args, kwargs) (line 451)
    expm_call_result_274169 = invoke(stypy.reporting.localization.Localization(__file__, 451, 13), expm_274164, *[result_mul_274167], **kwargs_274168)
    
    # Assigning a type to the variable 'ms' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'ms', expm_call_result_274169)
    
    # Assigning a Subscript to a Name (line 454):
    
    # Assigning a Subscript to a Name (line 454):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_274170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 25), 'int')
    # Getting the type of 'a' (line 454)
    a_274171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 17), 'a')
    # Obtaining the member 'shape' of a type (line 454)
    shape_274172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 17), a_274171, 'shape')
    # Obtaining the member '__getitem__' of a type (line 454)
    getitem___274173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 17), shape_274172, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 454)
    subscript_call_result_274174 = invoke(stypy.reporting.localization.Localization(__file__, 454, 17), getitem___274173, int_274170)
    
    slice_274175 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 454, 13), None, subscript_call_result_274174, None)
    slice_274176 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 454, 13), None, None, None)
    # Getting the type of 'ms' (line 454)
    ms_274177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 13), 'ms')
    # Obtaining the member '__getitem__' of a type (line 454)
    getitem___274178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 13), ms_274177, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 454)
    subscript_call_result_274179 = invoke(stypy.reporting.localization.Localization(__file__, 454, 13), getitem___274178, (slice_274175, slice_274176))
    
    # Assigning a type to the variable 'ms' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'ms', subscript_call_result_274179)
    
    # Assigning a Subscript to a Name (line 456):
    
    # Assigning a Subscript to a Name (line 456):
    
    # Obtaining the type of the subscript
    slice_274180 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 456, 13), None, None, None)
    int_274181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 19), 'int')
    
    # Obtaining the type of the subscript
    int_274182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 29), 'int')
    # Getting the type of 'a' (line 456)
    a_274183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 21), 'a')
    # Obtaining the member 'shape' of a type (line 456)
    shape_274184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 21), a_274183, 'shape')
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___274185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 21), shape_274184, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_274186 = invoke(stypy.reporting.localization.Localization(__file__, 456, 21), getitem___274185, int_274182)
    
    slice_274187 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 456, 13), int_274181, subscript_call_result_274186, None)
    # Getting the type of 'ms' (line 456)
    ms_274188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 13), 'ms')
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___274189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 13), ms_274188, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_274190 = invoke(stypy.reporting.localization.Localization(__file__, 456, 13), getitem___274189, (slice_274180, slice_274187))
    
    # Assigning a type to the variable 'ad' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'ad', subscript_call_result_274190)
    
    # Assigning a Subscript to a Name (line 457):
    
    # Assigning a Subscript to a Name (line 457):
    
    # Obtaining the type of the subscript
    slice_274191 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 457, 13), None, None, None)
    
    # Obtaining the type of the subscript
    int_274192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 27), 'int')
    # Getting the type of 'a' (line 457)
    a_274193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 19), 'a')
    # Obtaining the member 'shape' of a type (line 457)
    shape_274194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 19), a_274193, 'shape')
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___274195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 19), shape_274194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_274196 = invoke(stypy.reporting.localization.Localization(__file__, 457, 19), getitem___274195, int_274192)
    
    slice_274197 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 457, 13), subscript_call_result_274196, None, None)
    # Getting the type of 'ms' (line 457)
    ms_274198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 13), 'ms')
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___274199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 13), ms_274198, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_274200 = invoke(stypy.reporting.localization.Localization(__file__, 457, 13), getitem___274199, (slice_274191, slice_274197))
    
    # Assigning a type to the variable 'bd' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'bd', subscript_call_result_274200)
    
    # Assigning a Name to a Name (line 459):
    
    # Assigning a Name to a Name (line 459):
    # Getting the type of 'c' (line 459)
    c_274201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 13), 'c')
    # Assigning a type to the variable 'cd' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'cd', c_274201)
    
    # Assigning a Name to a Name (line 460):
    
    # Assigning a Name to a Name (line 460):
    # Getting the type of 'd' (line 460)
    d_274202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 13), 'd')
    # Assigning a type to the variable 'dd' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'dd', d_274202)
    # SSA branch for the else part of an if statement (line 442)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 463)
    # Processing the call arguments (line 463)
    str_274204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 25), 'str', "Unknown transformation method '%s'")
    # Getting the type of 'method' (line 463)
    method_274205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 64), 'method', False)
    # Applying the binary operator '%' (line 463)
    result_mod_274206 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 25), '%', str_274204, method_274205)
    
    # Processing the call keyword arguments (line 463)
    kwargs_274207 = {}
    # Getting the type of 'ValueError' (line 463)
    ValueError_274203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 463)
    ValueError_call_result_274208 = invoke(stypy.reporting.localization.Localization(__file__, 463, 14), ValueError_274203, *[result_mod_274206], **kwargs_274207)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 463, 8), ValueError_call_result_274208, 'raise parameter', BaseException)
    # SSA join for if statement (line 442)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 439)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 436)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 433)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 422)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 465)
    tuple_274209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 465)
    # Adding element type (line 465)
    # Getting the type of 'ad' (line 465)
    ad_274210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 11), 'ad')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 11), tuple_274209, ad_274210)
    # Adding element type (line 465)
    # Getting the type of 'bd' (line 465)
    bd_274211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'bd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 11), tuple_274209, bd_274211)
    # Adding element type (line 465)
    # Getting the type of 'cd' (line 465)
    cd_274212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 19), 'cd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 11), tuple_274209, cd_274212)
    # Adding element type (line 465)
    # Getting the type of 'dd' (line 465)
    dd_274213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 23), 'dd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 11), tuple_274209, dd_274213)
    # Adding element type (line 465)
    # Getting the type of 'dt' (line 465)
    dt_274214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 27), 'dt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 11), tuple_274209, dt_274214)
    
    # Assigning a type to the variable 'stypy_return_type' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'stypy_return_type', tuple_274209)
    
    # ################# End of 'cont2discrete(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cont2discrete' in the type store
    # Getting the type of 'stypy_return_type' (line 337)
    stypy_return_type_274215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_274215)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cont2discrete'
    return stypy_return_type_274215

# Assigning a type to the variable 'cont2discrete' (line 337)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 0), 'cont2discrete', cont2discrete)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
