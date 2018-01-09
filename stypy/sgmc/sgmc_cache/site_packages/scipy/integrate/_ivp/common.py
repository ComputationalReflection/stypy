
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: from itertools import groupby
3: from warnings import warn
4: import numpy as np
5: from scipy.sparse import find, coo_matrix
6: 
7: 
8: EPS = np.finfo(float).eps
9: 
10: 
11: def validate_max_step(max_step):
12:     '''Assert that max_Step is valid and return it.'''
13:     if max_step <= 0:
14:         raise ValueError("`max_step` must be positive.")
15:     return max_step
16: 
17: 
18: def warn_extraneous(extraneous):
19:     '''Display a warning for extraneous keyword arguments.
20: 
21:     The initializer of each solver class is expected to collect keyword
22:     arguments that it doesn't understand and warn about them. This function
23:     prints a warning for each key in the supplied dictionary.
24: 
25:     Parameters
26:     ----------
27:     extraneous : dict
28:         Extraneous keyword arguments
29:     '''
30:     if extraneous:
31:         warn("The following arguments have no effect for a chosen solver: {}."
32:              .format(", ".join("`{}`".format(x) for x in extraneous)))
33: 
34: 
35: def validate_tol(rtol, atol, n):
36:     '''Validate tolerance values.'''
37:     if rtol < 100 * EPS:
38:         warn("`rtol` is too low, setting to {}".format(100 * EPS))
39:         rtol = 100 * EPS
40: 
41:     atol = np.asarray(atol)
42:     if atol.ndim > 0 and atol.shape != (n,):
43:         raise ValueError("`atol` has wrong shape.")
44: 
45:     if np.any(atol < 0):
46:         raise ValueError("`atol` must be positive.")
47: 
48:     return rtol, atol
49: 
50: 
51: def norm(x):
52:     '''Compute RMS norm.'''
53:     return np.linalg.norm(x) / x.size ** 0.5
54: 
55: 
56: def select_initial_step(fun, t0, y0, f0, direction, order, rtol, atol):
57:     '''Empirically select a good initial step.
58: 
59:     The algorithm is described in [1]_.
60: 
61:     Parameters
62:     ----------
63:     fun : callable
64:         Right-hand side of the system.
65:     t0 : float
66:         Initial value of the independent variable.
67:     y0 : ndarray, shape (n,)
68:         Initial value of the dependent variable.
69:     f0 : ndarray, shape (n,)
70:         Initial value of the derivative, i. e. ``fun(t0, y0)``.
71:     direction : float
72:         Integration direction.
73:     order : float
74:         Method order.
75:     rtol : float
76:         Desired relative tolerance.
77:     atol : float
78:         Desired absolute tolerance.
79: 
80:     Returns
81:     -------
82:     h_abs : float
83:         Absolute value of the suggested initial step.
84: 
85:     References
86:     ----------
87:     .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
88:            Equations I: Nonstiff Problems", Sec. II.4.
89:     '''
90:     if y0.size == 0:
91:         return np.inf
92: 
93:     scale = atol + np.abs(y0) * rtol
94:     d0 = norm(y0 / scale)
95:     d1 = norm(f0 / scale)
96:     if d0 < 1e-5 or d1 < 1e-5:
97:         h0 = 1e-6
98:     else:
99:         h0 = 0.01 * d0 / d1
100: 
101:     y1 = y0 + h0 * direction * f0
102:     f1 = fun(t0 + h0 * direction, y1)
103:     d2 = norm((f1 - f0) / scale) / h0
104: 
105:     if d1 <= 1e-15 and d2 <= 1e-15:
106:         h1 = max(1e-6, h0 * 1e-3)
107:     else:
108:         h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))
109: 
110:     return min(100 * h0, h1)
111: 
112: 
113: class OdeSolution(object):
114:     '''Continuous ODE solution.
115: 
116:     It is organized as a collection of `DenseOutput` objects which represent
117:     local interpolants. It provides an algorithm to select a right interpolant
118:     for each given point.
119: 
120:     The interpolants cover the range between `t_min` and `t_max` (see
121:     Attributes below). Evaluation outside this interval is not forbidden, but
122:     the accuracy is not guaranteed.
123: 
124:     When evaluating at a breakpoint (one of the values in `ts`) a segment with
125:     the lower index is selected.
126: 
127:     Parameters
128:     ----------
129:     ts : array_like, shape (n_segments + 1,)
130:         Time instants between which local interpolants are defined. Must
131:         be strictly increasing or decreasing (zero segment with two points is
132:         also allowed).
133:     interpolants : list of DenseOutput with n_segments elements
134:         Local interpolants. An i-th interpolant is assumed to be defined
135:         between ``ts[i]`` and ``ts[i + 1]``.
136: 
137:     Attributes
138:     ----------
139:     t_min, t_max : float
140:         Time range of the interpolation.
141:     '''
142:     def __init__(self, ts, interpolants):
143:         ts = np.asarray(ts)
144:         d = np.diff(ts)
145:         # The first case covers integration on zero segment.
146:         if not ((ts.size == 2 and ts[0] == ts[-1])
147:                 or np.all(d > 0) or np.all(d < 0)):
148:             raise ValueError("`ts` must be strictly increasing or decreasing.")
149: 
150:         self.n_segments = len(interpolants)
151:         if ts.shape != (self.n_segments + 1,):
152:             raise ValueError("Numbers of time stamps and interpolants "
153:                              "don't match.")
154: 
155:         self.ts = ts
156:         self.interpolants = interpolants
157:         if ts[-1] >= ts[0]:
158:             self.t_min = ts[0]
159:             self.t_max = ts[-1]
160:             self.ascending = True
161:             self.ts_sorted = ts
162:         else:
163:             self.t_min = ts[-1]
164:             self.t_max = ts[0]
165:             self.ascending = False
166:             self.ts_sorted = ts[::-1]
167: 
168:     def _call_single(self, t):
169:         # Here we preserve a certain symmetry that when t is in self.ts,
170:         # then we prioritize a segment with a lower index.
171:         if self.ascending:
172:             ind = np.searchsorted(self.ts_sorted, t, side='left')
173:         else:
174:             ind = np.searchsorted(self.ts_sorted, t, side='right')
175: 
176:         segment = min(max(ind - 1, 0), self.n_segments - 1)
177:         if not self.ascending:
178:             segment = self.n_segments - 1 - segment
179: 
180:         return self.interpolants[segment](t)
181: 
182:     def __call__(self, t):
183:         '''Evaluate the solution.
184: 
185:         Parameters
186:         ----------
187:         t : float or array_like with shape (n_points,)
188:             Points to evaluate at.
189: 
190:         Returns
191:         -------
192:         y : ndarray, shape (n_states,) or (n_states, n_points)
193:             Computed values. Shape depends on whether `t` is a scalar or a
194:             1-d array.
195:         '''
196:         t = np.asarray(t)
197: 
198:         if t.ndim == 0:
199:             return self._call_single(t)
200: 
201:         order = np.argsort(t)
202:         reverse = np.empty_like(order)
203:         reverse[order] = np.arange(order.shape[0])
204:         t_sorted = t[order]
205: 
206:         # See comment in self._call_single.
207:         if self.ascending:
208:             segments = np.searchsorted(self.ts_sorted, t_sorted, side='left')
209:         else:
210:             segments = np.searchsorted(self.ts_sorted, t_sorted, side='right')
211:         segments -= 1
212:         segments[segments < 0] = 0
213:         segments[segments > self.n_segments - 1] = self.n_segments - 1
214:         if not self.ascending:
215:             segments = self.n_segments - 1 - segments
216: 
217:         ys = []
218:         group_start = 0
219:         for segment, group in groupby(segments):
220:             group_end = group_start + len(list(group))
221:             y = self.interpolants[segment](t_sorted[group_start:group_end])
222:             ys.append(y)
223:             group_start = group_end
224: 
225:         ys = np.hstack(ys)
226:         ys = ys[:, reverse]
227: 
228:         return ys
229: 
230: 
231: NUM_JAC_DIFF_REJECT = EPS ** 0.875
232: NUM_JAC_DIFF_SMALL = EPS ** 0.75
233: NUM_JAC_DIFF_BIG = EPS ** 0.25
234: NUM_JAC_MIN_FACTOR = 1e3 * EPS
235: NUM_JAC_FACTOR_INCREASE = 10
236: NUM_JAC_FACTOR_DECREASE = 0.1
237: 
238: 
239: def num_jac(fun, t, y, f, threshold, factor, sparsity=None):
240:     '''Finite differences Jacobian approximation tailored for ODE solvers.
241: 
242:     This function computes finite difference approximation to the Jacobian
243:     matrix of `fun` with respect to `y` using forward differences.
244:     The Jacobian matrix has shape (n, n) and its element (i, j) is equal to
245:     ``d f_i / d y_j``.
246: 
247:     A special feature of this function is the ability to correct the step
248:     size from iteration to iteration. The main idea is to keep the finite
249:     difference significantly separated from its round-off error which
250:     approximately equals ``EPS * np.abs(f)``. It reduces a possibility of a
251:     huge error and assures that the estimated derivative are reasonably close
252:     to the true values (i.e. the finite difference approximation is at least
253:     qualitatively reflects the structure of the true Jacobian).
254: 
255:     Parameters
256:     ----------
257:     fun : callable
258:         Right-hand side of the system implemented in a vectorized fashion.
259:     t : float
260:         Current time.
261:     y : ndarray, shape (n,)
262:         Current state.
263:     f : ndarray, shape (n,)
264:         Value of the right hand side at (t, y).
265:     threshold : float
266:         Threshold for `y` value used for computing the step size as
267:         ``factor * np.maximum(np.abs(y), threshold)``. Typically the value of
268:         absolute tolerance (atol) for a solver should be passed as `threshold`.
269:     factor : ndarray with shape (n,) or None
270:         Factor to use for computing the step size. Pass None for the very
271:         evaluation, then use the value returned from this function.
272:     sparsity : tuple (structure, groups) or None
273:         Sparsity structure of the Jacobian, `structure` must be csc_matrix.
274: 
275:     Returns
276:     -------
277:     J : ndarray or csc_matrix, shape (n, n)
278:         Jacobian matrix.
279:     factor : ndarray, shape (n,)
280:         Suggested `factor` for the next evaluation.
281:     '''
282:     y = np.asarray(y)
283:     n = y.shape[0]
284:     if n == 0:
285:         return np.empty((0, 0)), factor
286: 
287:     if factor is None:
288:         factor = np.ones(n) * EPS ** 0.5
289:     else:
290:         factor = factor.copy()
291: 
292:     # Direct the step as ODE dictates, hoping that such a step won't lead to
293:     # a problematic region. For complex ODEs it makes sense to use the real
294:     # part of f as we use steps along real axis.
295:     f_sign = 2 * (np.real(f) >= 0).astype(float) - 1
296:     y_scale = f_sign * np.maximum(threshold, np.abs(y))
297:     h = (y + factor * y_scale) - y
298: 
299:     # Make sure that the step is not 0 to start with. Not likely it will be
300:     # executed often.
301:     for i in np.nonzero(h == 0)[0]:
302:         while h[i] == 0:
303:             factor[i] *= 10
304:             h[i] = (y[i] + factor[i] * y_scale[i]) - y[i]
305: 
306:     if sparsity is None:
307:         return _dense_num_jac(fun, t, y, f, h, factor, y_scale)
308:     else:
309:         structure, groups = sparsity
310:         return _sparse_num_jac(fun, t, y, f, h, factor, y_scale,
311:                                structure, groups)
312: 
313: 
314: def _dense_num_jac(fun, t, y, f, h, factor, y_scale):
315:     n = y.shape[0]
316:     h_vecs = np.diag(h)
317:     f_new = fun(t, y[:, None] + h_vecs)
318:     diff = f_new - f[:, None]
319:     max_ind = np.argmax(np.abs(diff), axis=0)
320:     r = np.arange(n)
321:     max_diff = np.abs(diff[max_ind, r])
322:     scale = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))
323: 
324:     diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
325:     if np.any(diff_too_small):
326:         ind, = np.nonzero(diff_too_small)
327:         new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
328:         h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
329:         h_vecs[ind, ind] = h_new
330:         f_new = fun(t, y[:, None] + h_vecs[:, ind])
331:         diff_new = f_new - f[:, None]
332:         max_ind = np.argmax(np.abs(diff_new), axis=0)
333:         r = np.arange(ind.shape[0])
334:         max_diff_new = np.abs(diff_new[max_ind, r])
335:         scale_new = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))
336: 
337:         update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
338:         if np.any(update):
339:             update, = np.where(update)
340:             update_ind = ind[update]
341:             factor[update_ind] = new_factor[update]
342:             h[update_ind] = h_new[update]
343:             diff[:, update_ind] = diff_new[:, update]
344:             scale[update_ind] = scale_new[update]
345:             max_diff[update_ind] = max_diff_new[update]
346: 
347:     diff /= h
348: 
349:     factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
350:     factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
351:     factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)
352: 
353:     return diff, factor
354: 
355: 
356: def _sparse_num_jac(fun, t, y, f, h, factor, y_scale, structure, groups):
357:     n = y.shape[0]
358:     n_groups = np.max(groups) + 1
359:     h_vecs = np.empty((n_groups, n))
360:     for group in range(n_groups):
361:         e = np.equal(group, groups)
362:         h_vecs[group] = h * e
363:     h_vecs = h_vecs.T
364: 
365:     f_new = fun(t, y[:, None] + h_vecs)
366:     df = f_new - f[:, None]
367: 
368:     i, j, _ = find(structure)
369:     diff = coo_matrix((df[i, groups[j]], (i, j)), shape=(n, n)).tocsc()
370:     max_ind = np.array(abs(diff).argmax(axis=0)).ravel()
371:     r = np.arange(n)
372:     max_diff = np.asarray(np.abs(diff[max_ind, r])).ravel()
373:     scale = np.maximum(np.abs(f[max_ind]),
374:                        np.abs(f_new[max_ind, groups[r]]))
375: 
376:     diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
377:     if np.any(diff_too_small):
378:         ind, = np.nonzero(diff_too_small)
379:         new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
380:         h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
381:         h_new_all = np.zeros(n)
382:         h_new_all[ind] = h_new
383: 
384:         groups_unique = np.unique(groups[ind])
385:         groups_map = np.empty(n_groups, dtype=int)
386:         h_vecs = np.empty((groups_unique.shape[0], n))
387:         for k, group in enumerate(groups_unique):
388:             e = np.equal(group, groups)
389:             h_vecs[k] = h_new_all * e
390:             groups_map[group] = k
391:         h_vecs = h_vecs.T
392: 
393:         f_new = fun(t, y[:, None] + h_vecs)
394:         df = f_new - f[:, None]
395:         i, j, _ = find(structure[:, ind])
396:         diff_new = coo_matrix((df[i, groups_map[groups[ind[j]]]],
397:                                (i, j)), shape=(n, ind.shape[0])).tocsc()
398: 
399:         max_ind_new = np.array(abs(diff_new).argmax(axis=0)).ravel()
400:         r = np.arange(ind.shape[0])
401:         max_diff_new = np.asarray(np.abs(diff_new[max_ind_new, r])).ravel()
402:         scale_new = np.maximum(
403:             np.abs(f[max_ind_new]),
404:             np.abs(f_new[max_ind_new, groups_map[groups[ind]]]))
405: 
406:         update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
407:         if np.any(update):
408:             update, = np.where(update)
409:             update_ind = ind[update]
410:             factor[update_ind] = new_factor[update]
411:             h[update_ind] = h_new[update]
412:             diff[:, update_ind] = diff_new[:, update]
413:             scale[update_ind] = scale_new[update]
414:             max_diff[update_ind] = max_diff_new[update]
415: 
416:     diff.data /= np.repeat(h, np.diff(diff.indptr))
417: 
418:     factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
419:     factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
420:     factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)
421: 
422:     return diff, factor
423: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from itertools import groupby' statement (line 2)
try:
    from itertools import groupby

except:
    groupby = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'itertools', None, module_type_store, ['groupby'], [groupby])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from warnings import warn' statement (line 3)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_54020 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_54020) is not StypyTypeError):

    if (import_54020 != 'pyd_module'):
        __import__(import_54020)
        sys_modules_54021 = sys.modules[import_54020]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_54021.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_54020)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.sparse import find, coo_matrix' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_54022 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse')

if (type(import_54022) is not StypyTypeError):

    if (import_54022 != 'pyd_module'):
        __import__(import_54022)
        sys_modules_54023 = sys.modules[import_54022]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', sys_modules_54023.module_type_store, module_type_store, ['find', 'coo_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_54023, sys_modules_54023.module_type_store, module_type_store)
    else:
        from scipy.sparse import find, coo_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', None, module_type_store, ['find', 'coo_matrix'], [find, coo_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', import_54022)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')


# Assigning a Attribute to a Name (line 8):

# Assigning a Attribute to a Name (line 8):

# Call to finfo(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'float' (line 8)
float_54026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'float', False)
# Processing the call keyword arguments (line 8)
kwargs_54027 = {}
# Getting the type of 'np' (line 8)
np_54024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 6), 'np', False)
# Obtaining the member 'finfo' of a type (line 8)
finfo_54025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 6), np_54024, 'finfo')
# Calling finfo(args, kwargs) (line 8)
finfo_call_result_54028 = invoke(stypy.reporting.localization.Localization(__file__, 8, 6), finfo_54025, *[float_54026], **kwargs_54027)

# Obtaining the member 'eps' of a type (line 8)
eps_54029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 6), finfo_call_result_54028, 'eps')
# Assigning a type to the variable 'EPS' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'EPS', eps_54029)

@norecursion
def validate_max_step(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'validate_max_step'
    module_type_store = module_type_store.open_function_context('validate_max_step', 11, 0, False)
    
    # Passed parameters checking function
    validate_max_step.stypy_localization = localization
    validate_max_step.stypy_type_of_self = None
    validate_max_step.stypy_type_store = module_type_store
    validate_max_step.stypy_function_name = 'validate_max_step'
    validate_max_step.stypy_param_names_list = ['max_step']
    validate_max_step.stypy_varargs_param_name = None
    validate_max_step.stypy_kwargs_param_name = None
    validate_max_step.stypy_call_defaults = defaults
    validate_max_step.stypy_call_varargs = varargs
    validate_max_step.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'validate_max_step', ['max_step'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'validate_max_step', localization, ['max_step'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'validate_max_step(...)' code ##################

    str_54030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'str', 'Assert that max_Step is valid and return it.')
    
    
    # Getting the type of 'max_step' (line 13)
    max_step_54031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 7), 'max_step')
    int_54032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'int')
    # Applying the binary operator '<=' (line 13)
    result_le_54033 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 7), '<=', max_step_54031, int_54032)
    
    # Testing the type of an if condition (line 13)
    if_condition_54034 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 4), result_le_54033)
    # Assigning a type to the variable 'if_condition_54034' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'if_condition_54034', if_condition_54034)
    # SSA begins for if statement (line 13)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 14)
    # Processing the call arguments (line 14)
    str_54036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'str', '`max_step` must be positive.')
    # Processing the call keyword arguments (line 14)
    kwargs_54037 = {}
    # Getting the type of 'ValueError' (line 14)
    ValueError_54035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 14)
    ValueError_call_result_54038 = invoke(stypy.reporting.localization.Localization(__file__, 14, 14), ValueError_54035, *[str_54036], **kwargs_54037)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 14, 8), ValueError_call_result_54038, 'raise parameter', BaseException)
    # SSA join for if statement (line 13)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'max_step' (line 15)
    max_step_54039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'max_step')
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type', max_step_54039)
    
    # ################# End of 'validate_max_step(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'validate_max_step' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_54040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_54040)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'validate_max_step'
    return stypy_return_type_54040

# Assigning a type to the variable 'validate_max_step' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'validate_max_step', validate_max_step)

@norecursion
def warn_extraneous(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'warn_extraneous'
    module_type_store = module_type_store.open_function_context('warn_extraneous', 18, 0, False)
    
    # Passed parameters checking function
    warn_extraneous.stypy_localization = localization
    warn_extraneous.stypy_type_of_self = None
    warn_extraneous.stypy_type_store = module_type_store
    warn_extraneous.stypy_function_name = 'warn_extraneous'
    warn_extraneous.stypy_param_names_list = ['extraneous']
    warn_extraneous.stypy_varargs_param_name = None
    warn_extraneous.stypy_kwargs_param_name = None
    warn_extraneous.stypy_call_defaults = defaults
    warn_extraneous.stypy_call_varargs = varargs
    warn_extraneous.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'warn_extraneous', ['extraneous'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'warn_extraneous', localization, ['extraneous'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'warn_extraneous(...)' code ##################

    str_54041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'str', "Display a warning for extraneous keyword arguments.\n\n    The initializer of each solver class is expected to collect keyword\n    arguments that it doesn't understand and warn about them. This function\n    prints a warning for each key in the supplied dictionary.\n\n    Parameters\n    ----------\n    extraneous : dict\n        Extraneous keyword arguments\n    ")
    
    # Getting the type of 'extraneous' (line 30)
    extraneous_54042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'extraneous')
    # Testing the type of an if condition (line 30)
    if_condition_54043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 4), extraneous_54042)
    # Assigning a type to the variable 'if_condition_54043' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'if_condition_54043', if_condition_54043)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to format(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to join(...): (line 32)
    # Processing the call arguments (line 32)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 32, 31, True)
    # Calculating comprehension expression
    # Getting the type of 'extraneous' (line 32)
    extraneous_54054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 57), 'extraneous', False)
    comprehension_54055 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 31), extraneous_54054)
    # Assigning a type to the variable 'x' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 31), 'x', comprehension_54055)
    
    # Call to format(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'x' (line 32)
    x_54051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 45), 'x', False)
    # Processing the call keyword arguments (line 32)
    kwargs_54052 = {}
    str_54049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 31), 'str', '`{}`')
    # Obtaining the member 'format' of a type (line 32)
    format_54050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 31), str_54049, 'format')
    # Calling format(args, kwargs) (line 32)
    format_call_result_54053 = invoke(stypy.reporting.localization.Localization(__file__, 32, 31), format_54050, *[x_54051], **kwargs_54052)
    
    list_54056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 31), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 31), list_54056, format_call_result_54053)
    # Processing the call keyword arguments (line 32)
    kwargs_54057 = {}
    str_54047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'str', ', ')
    # Obtaining the member 'join' of a type (line 32)
    join_54048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 21), str_54047, 'join')
    # Calling join(args, kwargs) (line 32)
    join_call_result_54058 = invoke(stypy.reporting.localization.Localization(__file__, 32, 21), join_54048, *[list_54056], **kwargs_54057)
    
    # Processing the call keyword arguments (line 31)
    kwargs_54059 = {}
    str_54045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 13), 'str', 'The following arguments have no effect for a chosen solver: {}.')
    # Obtaining the member 'format' of a type (line 31)
    format_54046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), str_54045, 'format')
    # Calling format(args, kwargs) (line 31)
    format_call_result_54060 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), format_54046, *[join_call_result_54058], **kwargs_54059)
    
    # Processing the call keyword arguments (line 31)
    kwargs_54061 = {}
    # Getting the type of 'warn' (line 31)
    warn_54044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 31)
    warn_call_result_54062 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), warn_54044, *[format_call_result_54060], **kwargs_54061)
    
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'warn_extraneous(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'warn_extraneous' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_54063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_54063)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'warn_extraneous'
    return stypy_return_type_54063

# Assigning a type to the variable 'warn_extraneous' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'warn_extraneous', warn_extraneous)

@norecursion
def validate_tol(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'validate_tol'
    module_type_store = module_type_store.open_function_context('validate_tol', 35, 0, False)
    
    # Passed parameters checking function
    validate_tol.stypy_localization = localization
    validate_tol.stypy_type_of_self = None
    validate_tol.stypy_type_store = module_type_store
    validate_tol.stypy_function_name = 'validate_tol'
    validate_tol.stypy_param_names_list = ['rtol', 'atol', 'n']
    validate_tol.stypy_varargs_param_name = None
    validate_tol.stypy_kwargs_param_name = None
    validate_tol.stypy_call_defaults = defaults
    validate_tol.stypy_call_varargs = varargs
    validate_tol.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'validate_tol', ['rtol', 'atol', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'validate_tol', localization, ['rtol', 'atol', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'validate_tol(...)' code ##################

    str_54064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', 'Validate tolerance values.')
    
    
    # Getting the type of 'rtol' (line 37)
    rtol_54065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 7), 'rtol')
    int_54066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 14), 'int')
    # Getting the type of 'EPS' (line 37)
    EPS_54067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'EPS')
    # Applying the binary operator '*' (line 37)
    result_mul_54068 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 14), '*', int_54066, EPS_54067)
    
    # Applying the binary operator '<' (line 37)
    result_lt_54069 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 7), '<', rtol_54065, result_mul_54068)
    
    # Testing the type of an if condition (line 37)
    if_condition_54070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 4), result_lt_54069)
    # Assigning a type to the variable 'if_condition_54070' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'if_condition_54070', if_condition_54070)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 38)
    # Processing the call arguments (line 38)
    
    # Call to format(...): (line 38)
    # Processing the call arguments (line 38)
    int_54074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 55), 'int')
    # Getting the type of 'EPS' (line 38)
    EPS_54075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 61), 'EPS', False)
    # Applying the binary operator '*' (line 38)
    result_mul_54076 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 55), '*', int_54074, EPS_54075)
    
    # Processing the call keyword arguments (line 38)
    kwargs_54077 = {}
    str_54072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 13), 'str', '`rtol` is too low, setting to {}')
    # Obtaining the member 'format' of a type (line 38)
    format_54073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 13), str_54072, 'format')
    # Calling format(args, kwargs) (line 38)
    format_call_result_54078 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), format_54073, *[result_mul_54076], **kwargs_54077)
    
    # Processing the call keyword arguments (line 38)
    kwargs_54079 = {}
    # Getting the type of 'warn' (line 38)
    warn_54071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 38)
    warn_call_result_54080 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), warn_54071, *[format_call_result_54078], **kwargs_54079)
    
    
    # Assigning a BinOp to a Name (line 39):
    
    # Assigning a BinOp to a Name (line 39):
    int_54081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 15), 'int')
    # Getting the type of 'EPS' (line 39)
    EPS_54082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'EPS')
    # Applying the binary operator '*' (line 39)
    result_mul_54083 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), '*', int_54081, EPS_54082)
    
    # Assigning a type to the variable 'rtol' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'rtol', result_mul_54083)
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 41):
    
    # Assigning a Call to a Name (line 41):
    
    # Call to asarray(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'atol' (line 41)
    atol_54086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'atol', False)
    # Processing the call keyword arguments (line 41)
    kwargs_54087 = {}
    # Getting the type of 'np' (line 41)
    np_54084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'np', False)
    # Obtaining the member 'asarray' of a type (line 41)
    asarray_54085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 11), np_54084, 'asarray')
    # Calling asarray(args, kwargs) (line 41)
    asarray_call_result_54088 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), asarray_54085, *[atol_54086], **kwargs_54087)
    
    # Assigning a type to the variable 'atol' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'atol', asarray_call_result_54088)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'atol' (line 42)
    atol_54089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 7), 'atol')
    # Obtaining the member 'ndim' of a type (line 42)
    ndim_54090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 7), atol_54089, 'ndim')
    int_54091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'int')
    # Applying the binary operator '>' (line 42)
    result_gt_54092 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 7), '>', ndim_54090, int_54091)
    
    
    # Getting the type of 'atol' (line 42)
    atol_54093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'atol')
    # Obtaining the member 'shape' of a type (line 42)
    shape_54094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 25), atol_54093, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_54095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    # Getting the type of 'n' (line 42)
    n_54096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 40), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 40), tuple_54095, n_54096)
    
    # Applying the binary operator '!=' (line 42)
    result_ne_54097 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 25), '!=', shape_54094, tuple_54095)
    
    # Applying the binary operator 'and' (line 42)
    result_and_keyword_54098 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 7), 'and', result_gt_54092, result_ne_54097)
    
    # Testing the type of an if condition (line 42)
    if_condition_54099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 4), result_and_keyword_54098)
    # Assigning a type to the variable 'if_condition_54099' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_condition_54099', if_condition_54099)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 43)
    # Processing the call arguments (line 43)
    str_54101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'str', '`atol` has wrong shape.')
    # Processing the call keyword arguments (line 43)
    kwargs_54102 = {}
    # Getting the type of 'ValueError' (line 43)
    ValueError_54100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 43)
    ValueError_call_result_54103 = invoke(stypy.reporting.localization.Localization(__file__, 43, 14), ValueError_54100, *[str_54101], **kwargs_54102)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 43, 8), ValueError_call_result_54103, 'raise parameter', BaseException)
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Getting the type of 'atol' (line 45)
    atol_54106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), 'atol', False)
    int_54107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 21), 'int')
    # Applying the binary operator '<' (line 45)
    result_lt_54108 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 14), '<', atol_54106, int_54107)
    
    # Processing the call keyword arguments (line 45)
    kwargs_54109 = {}
    # Getting the type of 'np' (line 45)
    np_54104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 45)
    any_54105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 7), np_54104, 'any')
    # Calling any(args, kwargs) (line 45)
    any_call_result_54110 = invoke(stypy.reporting.localization.Localization(__file__, 45, 7), any_54105, *[result_lt_54108], **kwargs_54109)
    
    # Testing the type of an if condition (line 45)
    if_condition_54111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 4), any_call_result_54110)
    # Assigning a type to the variable 'if_condition_54111' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'if_condition_54111', if_condition_54111)
    # SSA begins for if statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 46)
    # Processing the call arguments (line 46)
    str_54113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 25), 'str', '`atol` must be positive.')
    # Processing the call keyword arguments (line 46)
    kwargs_54114 = {}
    # Getting the type of 'ValueError' (line 46)
    ValueError_54112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 46)
    ValueError_call_result_54115 = invoke(stypy.reporting.localization.Localization(__file__, 46, 14), ValueError_54112, *[str_54113], **kwargs_54114)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 46, 8), ValueError_call_result_54115, 'raise parameter', BaseException)
    # SSA join for if statement (line 45)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 48)
    tuple_54116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 48)
    # Adding element type (line 48)
    # Getting the type of 'rtol' (line 48)
    rtol_54117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'rtol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), tuple_54116, rtol_54117)
    # Adding element type (line 48)
    # Getting the type of 'atol' (line 48)
    atol_54118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'atol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), tuple_54116, atol_54118)
    
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type', tuple_54116)
    
    # ################# End of 'validate_tol(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'validate_tol' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_54119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_54119)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'validate_tol'
    return stypy_return_type_54119

# Assigning a type to the variable 'validate_tol' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'validate_tol', validate_tol)

@norecursion
def norm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'norm'
    module_type_store = module_type_store.open_function_context('norm', 51, 0, False)
    
    # Passed parameters checking function
    norm.stypy_localization = localization
    norm.stypy_type_of_self = None
    norm.stypy_type_store = module_type_store
    norm.stypy_function_name = 'norm'
    norm.stypy_param_names_list = ['x']
    norm.stypy_varargs_param_name = None
    norm.stypy_kwargs_param_name = None
    norm.stypy_call_defaults = defaults
    norm.stypy_call_varargs = varargs
    norm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'norm', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'norm', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'norm(...)' code ##################

    str_54120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'str', 'Compute RMS norm.')
    
    # Call to norm(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'x' (line 53)
    x_54124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'x', False)
    # Processing the call keyword arguments (line 53)
    kwargs_54125 = {}
    # Getting the type of 'np' (line 53)
    np_54121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'np', False)
    # Obtaining the member 'linalg' of a type (line 53)
    linalg_54122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 11), np_54121, 'linalg')
    # Obtaining the member 'norm' of a type (line 53)
    norm_54123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 11), linalg_54122, 'norm')
    # Calling norm(args, kwargs) (line 53)
    norm_call_result_54126 = invoke(stypy.reporting.localization.Localization(__file__, 53, 11), norm_54123, *[x_54124], **kwargs_54125)
    
    # Getting the type of 'x' (line 53)
    x_54127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'x')
    # Obtaining the member 'size' of a type (line 53)
    size_54128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 31), x_54127, 'size')
    float_54129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 41), 'float')
    # Applying the binary operator '**' (line 53)
    result_pow_54130 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 31), '**', size_54128, float_54129)
    
    # Applying the binary operator 'div' (line 53)
    result_div_54131 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 11), 'div', norm_call_result_54126, result_pow_54130)
    
    # Assigning a type to the variable 'stypy_return_type' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type', result_div_54131)
    
    # ################# End of 'norm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'norm' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_54132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_54132)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'norm'
    return stypy_return_type_54132

# Assigning a type to the variable 'norm' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'norm', norm)

@norecursion
def select_initial_step(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'select_initial_step'
    module_type_store = module_type_store.open_function_context('select_initial_step', 56, 0, False)
    
    # Passed parameters checking function
    select_initial_step.stypy_localization = localization
    select_initial_step.stypy_type_of_self = None
    select_initial_step.stypy_type_store = module_type_store
    select_initial_step.stypy_function_name = 'select_initial_step'
    select_initial_step.stypy_param_names_list = ['fun', 't0', 'y0', 'f0', 'direction', 'order', 'rtol', 'atol']
    select_initial_step.stypy_varargs_param_name = None
    select_initial_step.stypy_kwargs_param_name = None
    select_initial_step.stypy_call_defaults = defaults
    select_initial_step.stypy_call_varargs = varargs
    select_initial_step.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'select_initial_step', ['fun', 't0', 'y0', 'f0', 'direction', 'order', 'rtol', 'atol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'select_initial_step', localization, ['fun', 't0', 'y0', 'f0', 'direction', 'order', 'rtol', 'atol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'select_initial_step(...)' code ##################

    str_54133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', 'Empirically select a good initial step.\n\n    The algorithm is described in [1]_.\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system.\n    t0 : float\n        Initial value of the independent variable.\n    y0 : ndarray, shape (n,)\n        Initial value of the dependent variable.\n    f0 : ndarray, shape (n,)\n        Initial value of the derivative, i. e. ``fun(t0, y0)``.\n    direction : float\n        Integration direction.\n    order : float\n        Method order.\n    rtol : float\n        Desired relative tolerance.\n    atol : float\n        Desired absolute tolerance.\n\n    Returns\n    -------\n    h_abs : float\n        Absolute value of the suggested initial step.\n\n    References\n    ----------\n    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential\n           Equations I: Nonstiff Problems", Sec. II.4.\n    ')
    
    
    # Getting the type of 'y0' (line 90)
    y0_54134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 7), 'y0')
    # Obtaining the member 'size' of a type (line 90)
    size_54135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 7), y0_54134, 'size')
    int_54136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'int')
    # Applying the binary operator '==' (line 90)
    result_eq_54137 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 7), '==', size_54135, int_54136)
    
    # Testing the type of an if condition (line 90)
    if_condition_54138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 4), result_eq_54137)
    # Assigning a type to the variable 'if_condition_54138' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'if_condition_54138', if_condition_54138)
    # SSA begins for if statement (line 90)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'np' (line 91)
    np_54139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'np')
    # Obtaining the member 'inf' of a type (line 91)
    inf_54140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), np_54139, 'inf')
    # Assigning a type to the variable 'stypy_return_type' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'stypy_return_type', inf_54140)
    # SSA join for if statement (line 90)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 93):
    
    # Assigning a BinOp to a Name (line 93):
    # Getting the type of 'atol' (line 93)
    atol_54141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'atol')
    
    # Call to abs(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'y0' (line 93)
    y0_54144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), 'y0', False)
    # Processing the call keyword arguments (line 93)
    kwargs_54145 = {}
    # Getting the type of 'np' (line 93)
    np_54142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'np', False)
    # Obtaining the member 'abs' of a type (line 93)
    abs_54143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 19), np_54142, 'abs')
    # Calling abs(args, kwargs) (line 93)
    abs_call_result_54146 = invoke(stypy.reporting.localization.Localization(__file__, 93, 19), abs_54143, *[y0_54144], **kwargs_54145)
    
    # Getting the type of 'rtol' (line 93)
    rtol_54147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 32), 'rtol')
    # Applying the binary operator '*' (line 93)
    result_mul_54148 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 19), '*', abs_call_result_54146, rtol_54147)
    
    # Applying the binary operator '+' (line 93)
    result_add_54149 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '+', atol_54141, result_mul_54148)
    
    # Assigning a type to the variable 'scale' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'scale', result_add_54149)
    
    # Assigning a Call to a Name (line 94):
    
    # Assigning a Call to a Name (line 94):
    
    # Call to norm(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'y0' (line 94)
    y0_54151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'y0', False)
    # Getting the type of 'scale' (line 94)
    scale_54152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'scale', False)
    # Applying the binary operator 'div' (line 94)
    result_div_54153 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 14), 'div', y0_54151, scale_54152)
    
    # Processing the call keyword arguments (line 94)
    kwargs_54154 = {}
    # Getting the type of 'norm' (line 94)
    norm_54150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 9), 'norm', False)
    # Calling norm(args, kwargs) (line 94)
    norm_call_result_54155 = invoke(stypy.reporting.localization.Localization(__file__, 94, 9), norm_54150, *[result_div_54153], **kwargs_54154)
    
    # Assigning a type to the variable 'd0' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'd0', norm_call_result_54155)
    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to norm(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'f0' (line 95)
    f0_54157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 14), 'f0', False)
    # Getting the type of 'scale' (line 95)
    scale_54158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'scale', False)
    # Applying the binary operator 'div' (line 95)
    result_div_54159 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 14), 'div', f0_54157, scale_54158)
    
    # Processing the call keyword arguments (line 95)
    kwargs_54160 = {}
    # Getting the type of 'norm' (line 95)
    norm_54156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 9), 'norm', False)
    # Calling norm(args, kwargs) (line 95)
    norm_call_result_54161 = invoke(stypy.reporting.localization.Localization(__file__, 95, 9), norm_54156, *[result_div_54159], **kwargs_54160)
    
    # Assigning a type to the variable 'd1' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'd1', norm_call_result_54161)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'd0' (line 96)
    d0_54162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'd0')
    float_54163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'float')
    # Applying the binary operator '<' (line 96)
    result_lt_54164 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), '<', d0_54162, float_54163)
    
    
    # Getting the type of 'd1' (line 96)
    d1_54165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'd1')
    float_54166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 25), 'float')
    # Applying the binary operator '<' (line 96)
    result_lt_54167 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 20), '<', d1_54165, float_54166)
    
    # Applying the binary operator 'or' (line 96)
    result_or_keyword_54168 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), 'or', result_lt_54164, result_lt_54167)
    
    # Testing the type of an if condition (line 96)
    if_condition_54169 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), result_or_keyword_54168)
    # Assigning a type to the variable 'if_condition_54169' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_54169', if_condition_54169)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 97):
    
    # Assigning a Num to a Name (line 97):
    float_54170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 13), 'float')
    # Assigning a type to the variable 'h0' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'h0', float_54170)
    # SSA branch for the else part of an if statement (line 96)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 99):
    
    # Assigning a BinOp to a Name (line 99):
    float_54171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 13), 'float')
    # Getting the type of 'd0' (line 99)
    d0_54172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'd0')
    # Applying the binary operator '*' (line 99)
    result_mul_54173 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 13), '*', float_54171, d0_54172)
    
    # Getting the type of 'd1' (line 99)
    d1_54174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'd1')
    # Applying the binary operator 'div' (line 99)
    result_div_54175 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 23), 'div', result_mul_54173, d1_54174)
    
    # Assigning a type to the variable 'h0' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'h0', result_div_54175)
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 101):
    
    # Assigning a BinOp to a Name (line 101):
    # Getting the type of 'y0' (line 101)
    y0_54176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 9), 'y0')
    # Getting the type of 'h0' (line 101)
    h0_54177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), 'h0')
    # Getting the type of 'direction' (line 101)
    direction_54178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'direction')
    # Applying the binary operator '*' (line 101)
    result_mul_54179 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 14), '*', h0_54177, direction_54178)
    
    # Getting the type of 'f0' (line 101)
    f0_54180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'f0')
    # Applying the binary operator '*' (line 101)
    result_mul_54181 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 29), '*', result_mul_54179, f0_54180)
    
    # Applying the binary operator '+' (line 101)
    result_add_54182 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 9), '+', y0_54176, result_mul_54181)
    
    # Assigning a type to the variable 'y1' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'y1', result_add_54182)
    
    # Assigning a Call to a Name (line 102):
    
    # Assigning a Call to a Name (line 102):
    
    # Call to fun(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 't0' (line 102)
    t0_54184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 't0', False)
    # Getting the type of 'h0' (line 102)
    h0_54185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 18), 'h0', False)
    # Getting the type of 'direction' (line 102)
    direction_54186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'direction', False)
    # Applying the binary operator '*' (line 102)
    result_mul_54187 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 18), '*', h0_54185, direction_54186)
    
    # Applying the binary operator '+' (line 102)
    result_add_54188 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 13), '+', t0_54184, result_mul_54187)
    
    # Getting the type of 'y1' (line 102)
    y1_54189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'y1', False)
    # Processing the call keyword arguments (line 102)
    kwargs_54190 = {}
    # Getting the type of 'fun' (line 102)
    fun_54183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 9), 'fun', False)
    # Calling fun(args, kwargs) (line 102)
    fun_call_result_54191 = invoke(stypy.reporting.localization.Localization(__file__, 102, 9), fun_54183, *[result_add_54188, y1_54189], **kwargs_54190)
    
    # Assigning a type to the variable 'f1' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'f1', fun_call_result_54191)
    
    # Assigning a BinOp to a Name (line 103):
    
    # Assigning a BinOp to a Name (line 103):
    
    # Call to norm(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'f1' (line 103)
    f1_54193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'f1', False)
    # Getting the type of 'f0' (line 103)
    f0_54194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'f0', False)
    # Applying the binary operator '-' (line 103)
    result_sub_54195 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 15), '-', f1_54193, f0_54194)
    
    # Getting the type of 'scale' (line 103)
    scale_54196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'scale', False)
    # Applying the binary operator 'div' (line 103)
    result_div_54197 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 14), 'div', result_sub_54195, scale_54196)
    
    # Processing the call keyword arguments (line 103)
    kwargs_54198 = {}
    # Getting the type of 'norm' (line 103)
    norm_54192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 9), 'norm', False)
    # Calling norm(args, kwargs) (line 103)
    norm_call_result_54199 = invoke(stypy.reporting.localization.Localization(__file__, 103, 9), norm_54192, *[result_div_54197], **kwargs_54198)
    
    # Getting the type of 'h0' (line 103)
    h0_54200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 35), 'h0')
    # Applying the binary operator 'div' (line 103)
    result_div_54201 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 9), 'div', norm_call_result_54199, h0_54200)
    
    # Assigning a type to the variable 'd2' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'd2', result_div_54201)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'd1' (line 105)
    d1_54202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 7), 'd1')
    float_54203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 13), 'float')
    # Applying the binary operator '<=' (line 105)
    result_le_54204 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 7), '<=', d1_54202, float_54203)
    
    
    # Getting the type of 'd2' (line 105)
    d2_54205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 23), 'd2')
    float_54206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 29), 'float')
    # Applying the binary operator '<=' (line 105)
    result_le_54207 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 23), '<=', d2_54205, float_54206)
    
    # Applying the binary operator 'and' (line 105)
    result_and_keyword_54208 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 7), 'and', result_le_54204, result_le_54207)
    
    # Testing the type of an if condition (line 105)
    if_condition_54209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 4), result_and_keyword_54208)
    # Assigning a type to the variable 'if_condition_54209' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'if_condition_54209', if_condition_54209)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 106):
    
    # Call to max(...): (line 106)
    # Processing the call arguments (line 106)
    float_54211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 17), 'float')
    # Getting the type of 'h0' (line 106)
    h0_54212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'h0', False)
    float_54213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 28), 'float')
    # Applying the binary operator '*' (line 106)
    result_mul_54214 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 23), '*', h0_54212, float_54213)
    
    # Processing the call keyword arguments (line 106)
    kwargs_54215 = {}
    # Getting the type of 'max' (line 106)
    max_54210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'max', False)
    # Calling max(args, kwargs) (line 106)
    max_call_result_54216 = invoke(stypy.reporting.localization.Localization(__file__, 106, 13), max_54210, *[float_54211, result_mul_54214], **kwargs_54215)
    
    # Assigning a type to the variable 'h1' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'h1', max_call_result_54216)
    # SSA branch for the else part of an if statement (line 105)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 108):
    
    # Assigning a BinOp to a Name (line 108):
    float_54217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 14), 'float')
    
    # Call to max(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'd1' (line 108)
    d1_54219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'd1', False)
    # Getting the type of 'd2' (line 108)
    d2_54220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 'd2', False)
    # Processing the call keyword arguments (line 108)
    kwargs_54221 = {}
    # Getting the type of 'max' (line 108)
    max_54218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'max', False)
    # Calling max(args, kwargs) (line 108)
    max_call_result_54222 = invoke(stypy.reporting.localization.Localization(__file__, 108, 21), max_54218, *[d1_54219, d2_54220], **kwargs_54221)
    
    # Applying the binary operator 'div' (line 108)
    result_div_54223 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'div', float_54217, max_call_result_54222)
    
    int_54224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 38), 'int')
    # Getting the type of 'order' (line 108)
    order_54225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 43), 'order')
    int_54226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 51), 'int')
    # Applying the binary operator '+' (line 108)
    result_add_54227 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 43), '+', order_54225, int_54226)
    
    # Applying the binary operator 'div' (line 108)
    result_div_54228 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 38), 'div', int_54224, result_add_54227)
    
    # Applying the binary operator '**' (line 108)
    result_pow_54229 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 13), '**', result_div_54223, result_div_54228)
    
    # Assigning a type to the variable 'h1' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'h1', result_pow_54229)
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to min(...): (line 110)
    # Processing the call arguments (line 110)
    int_54231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 15), 'int')
    # Getting the type of 'h0' (line 110)
    h0_54232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'h0', False)
    # Applying the binary operator '*' (line 110)
    result_mul_54233 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 15), '*', int_54231, h0_54232)
    
    # Getting the type of 'h1' (line 110)
    h1_54234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'h1', False)
    # Processing the call keyword arguments (line 110)
    kwargs_54235 = {}
    # Getting the type of 'min' (line 110)
    min_54230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'min', False)
    # Calling min(args, kwargs) (line 110)
    min_call_result_54236 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), min_54230, *[result_mul_54233, h1_54234], **kwargs_54235)
    
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type', min_call_result_54236)
    
    # ################# End of 'select_initial_step(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'select_initial_step' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_54237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_54237)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'select_initial_step'
    return stypy_return_type_54237

# Assigning a type to the variable 'select_initial_step' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'select_initial_step', select_initial_step)
# Declaration of the 'OdeSolution' class

class OdeSolution(object, ):
    str_54238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, (-1)), 'str', 'Continuous ODE solution.\n\n    It is organized as a collection of `DenseOutput` objects which represent\n    local interpolants. It provides an algorithm to select a right interpolant\n    for each given point.\n\n    The interpolants cover the range between `t_min` and `t_max` (see\n    Attributes below). Evaluation outside this interval is not forbidden, but\n    the accuracy is not guaranteed.\n\n    When evaluating at a breakpoint (one of the values in `ts`) a segment with\n    the lower index is selected.\n\n    Parameters\n    ----------\n    ts : array_like, shape (n_segments + 1,)\n        Time instants between which local interpolants are defined. Must\n        be strictly increasing or decreasing (zero segment with two points is\n        also allowed).\n    interpolants : list of DenseOutput with n_segments elements\n        Local interpolants. An i-th interpolant is assumed to be defined\n        between ``ts[i]`` and ``ts[i + 1]``.\n\n    Attributes\n    ----------\n    t_min, t_max : float\n        Time range of the interpolation.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 142, 4, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdeSolution.__init__', ['ts', 'interpolants'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['ts', 'interpolants'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Name (line 143):
        
        # Assigning a Call to a Name (line 143):
        
        # Call to asarray(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'ts' (line 143)
        ts_54241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'ts', False)
        # Processing the call keyword arguments (line 143)
        kwargs_54242 = {}
        # Getting the type of 'np' (line 143)
        np_54239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 13), 'np', False)
        # Obtaining the member 'asarray' of a type (line 143)
        asarray_54240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 13), np_54239, 'asarray')
        # Calling asarray(args, kwargs) (line 143)
        asarray_call_result_54243 = invoke(stypy.reporting.localization.Localization(__file__, 143, 13), asarray_54240, *[ts_54241], **kwargs_54242)
        
        # Assigning a type to the variable 'ts' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'ts', asarray_call_result_54243)
        
        # Assigning a Call to a Name (line 144):
        
        # Assigning a Call to a Name (line 144):
        
        # Call to diff(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'ts' (line 144)
        ts_54246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'ts', False)
        # Processing the call keyword arguments (line 144)
        kwargs_54247 = {}
        # Getting the type of 'np' (line 144)
        np_54244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'np', False)
        # Obtaining the member 'diff' of a type (line 144)
        diff_54245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), np_54244, 'diff')
        # Calling diff(args, kwargs) (line 144)
        diff_call_result_54248 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), diff_54245, *[ts_54246], **kwargs_54247)
        
        # Assigning a type to the variable 'd' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'd', diff_call_result_54248)
        
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ts' (line 146)
        ts_54249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'ts')
        # Obtaining the member 'size' of a type (line 146)
        size_54250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 17), ts_54249, 'size')
        int_54251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 28), 'int')
        # Applying the binary operator '==' (line 146)
        result_eq_54252 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 17), '==', size_54250, int_54251)
        
        
        
        # Obtaining the type of the subscript
        int_54253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 37), 'int')
        # Getting the type of 'ts' (line 146)
        ts_54254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'ts')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___54255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 34), ts_54254, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_54256 = invoke(stypy.reporting.localization.Localization(__file__, 146, 34), getitem___54255, int_54253)
        
        
        # Obtaining the type of the subscript
        int_54257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 46), 'int')
        # Getting the type of 'ts' (line 146)
        ts_54258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 43), 'ts')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___54259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 43), ts_54258, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_54260 = invoke(stypy.reporting.localization.Localization(__file__, 146, 43), getitem___54259, int_54257)
        
        # Applying the binary operator '==' (line 146)
        result_eq_54261 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 34), '==', subscript_call_result_54256, subscript_call_result_54260)
        
        # Applying the binary operator 'and' (line 146)
        result_and_keyword_54262 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 17), 'and', result_eq_54252, result_eq_54261)
        
        
        # Call to all(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Getting the type of 'd' (line 147)
        d_54265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 26), 'd', False)
        int_54266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 30), 'int')
        # Applying the binary operator '>' (line 147)
        result_gt_54267 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 26), '>', d_54265, int_54266)
        
        # Processing the call keyword arguments (line 147)
        kwargs_54268 = {}
        # Getting the type of 'np' (line 147)
        np_54263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'np', False)
        # Obtaining the member 'all' of a type (line 147)
        all_54264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 19), np_54263, 'all')
        # Calling all(args, kwargs) (line 147)
        all_call_result_54269 = invoke(stypy.reporting.localization.Localization(__file__, 147, 19), all_54264, *[result_gt_54267], **kwargs_54268)
        
        # Applying the binary operator 'or' (line 146)
        result_or_keyword_54270 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 16), 'or', result_and_keyword_54262, all_call_result_54269)
        
        # Call to all(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Getting the type of 'd' (line 147)
        d_54273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 43), 'd', False)
        int_54274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 47), 'int')
        # Applying the binary operator '<' (line 147)
        result_lt_54275 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 43), '<', d_54273, int_54274)
        
        # Processing the call keyword arguments (line 147)
        kwargs_54276 = {}
        # Getting the type of 'np' (line 147)
        np_54271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 36), 'np', False)
        # Obtaining the member 'all' of a type (line 147)
        all_54272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 36), np_54271, 'all')
        # Calling all(args, kwargs) (line 147)
        all_call_result_54277 = invoke(stypy.reporting.localization.Localization(__file__, 147, 36), all_54272, *[result_lt_54275], **kwargs_54276)
        
        # Applying the binary operator 'or' (line 146)
        result_or_keyword_54278 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 16), 'or', result_or_keyword_54270, all_call_result_54277)
        
        # Applying the 'not' unary operator (line 146)
        result_not__54279 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 11), 'not', result_or_keyword_54278)
        
        # Testing the type of an if condition (line 146)
        if_condition_54280 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), result_not__54279)
        # Assigning a type to the variable 'if_condition_54280' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_54280', if_condition_54280)
        # SSA begins for if statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 148)
        # Processing the call arguments (line 148)
        str_54282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 29), 'str', '`ts` must be strictly increasing or decreasing.')
        # Processing the call keyword arguments (line 148)
        kwargs_54283 = {}
        # Getting the type of 'ValueError' (line 148)
        ValueError_54281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 148)
        ValueError_call_result_54284 = invoke(stypy.reporting.localization.Localization(__file__, 148, 18), ValueError_54281, *[str_54282], **kwargs_54283)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 148, 12), ValueError_call_result_54284, 'raise parameter', BaseException)
        # SSA join for if statement (line 146)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 150):
        
        # Assigning a Call to a Attribute (line 150):
        
        # Call to len(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'interpolants' (line 150)
        interpolants_54286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'interpolants', False)
        # Processing the call keyword arguments (line 150)
        kwargs_54287 = {}
        # Getting the type of 'len' (line 150)
        len_54285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'len', False)
        # Calling len(args, kwargs) (line 150)
        len_call_result_54288 = invoke(stypy.reporting.localization.Localization(__file__, 150, 26), len_54285, *[interpolants_54286], **kwargs_54287)
        
        # Getting the type of 'self' (line 150)
        self_54289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'n_segments' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_54289, 'n_segments', len_call_result_54288)
        
        
        # Getting the type of 'ts' (line 151)
        ts_54290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'ts')
        # Obtaining the member 'shape' of a type (line 151)
        shape_54291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), ts_54290, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 151)
        tuple_54292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 151)
        # Adding element type (line 151)
        # Getting the type of 'self' (line 151)
        self_54293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'self')
        # Obtaining the member 'n_segments' of a type (line 151)
        n_segments_54294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 24), self_54293, 'n_segments')
        int_54295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 42), 'int')
        # Applying the binary operator '+' (line 151)
        result_add_54296 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 24), '+', n_segments_54294, int_54295)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 24), tuple_54292, result_add_54296)
        
        # Applying the binary operator '!=' (line 151)
        result_ne_54297 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), '!=', shape_54291, tuple_54292)
        
        # Testing the type of an if condition (line 151)
        if_condition_54298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_ne_54297)
        # Assigning a type to the variable 'if_condition_54298' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_54298', if_condition_54298)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 152)
        # Processing the call arguments (line 152)
        str_54300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 29), 'str', "Numbers of time stamps and interpolants don't match.")
        # Processing the call keyword arguments (line 152)
        kwargs_54301 = {}
        # Getting the type of 'ValueError' (line 152)
        ValueError_54299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 152)
        ValueError_call_result_54302 = invoke(stypy.reporting.localization.Localization(__file__, 152, 18), ValueError_54299, *[str_54300], **kwargs_54301)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 152, 12), ValueError_call_result_54302, 'raise parameter', BaseException)
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 155):
        
        # Assigning a Name to a Attribute (line 155):
        # Getting the type of 'ts' (line 155)
        ts_54303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 18), 'ts')
        # Getting the type of 'self' (line 155)
        self_54304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self')
        # Setting the type of the member 'ts' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_54304, 'ts', ts_54303)
        
        # Assigning a Name to a Attribute (line 156):
        
        # Assigning a Name to a Attribute (line 156):
        # Getting the type of 'interpolants' (line 156)
        interpolants_54305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 28), 'interpolants')
        # Getting the type of 'self' (line 156)
        self_54306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'self')
        # Setting the type of the member 'interpolants' of a type (line 156)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), self_54306, 'interpolants', interpolants_54305)
        
        
        
        # Obtaining the type of the subscript
        int_54307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 14), 'int')
        # Getting the type of 'ts' (line 157)
        ts_54308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'ts')
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___54309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 11), ts_54308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_54310 = invoke(stypy.reporting.localization.Localization(__file__, 157, 11), getitem___54309, int_54307)
        
        
        # Obtaining the type of the subscript
        int_54311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 24), 'int')
        # Getting the type of 'ts' (line 157)
        ts_54312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'ts')
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___54313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 21), ts_54312, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_54314 = invoke(stypy.reporting.localization.Localization(__file__, 157, 21), getitem___54313, int_54311)
        
        # Applying the binary operator '>=' (line 157)
        result_ge_54315 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 11), '>=', subscript_call_result_54310, subscript_call_result_54314)
        
        # Testing the type of an if condition (line 157)
        if_condition_54316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 8), result_ge_54315)
        # Assigning a type to the variable 'if_condition_54316' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'if_condition_54316', if_condition_54316)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Attribute (line 158):
        
        # Assigning a Subscript to a Attribute (line 158):
        
        # Obtaining the type of the subscript
        int_54317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 28), 'int')
        # Getting the type of 'ts' (line 158)
        ts_54318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 25), 'ts')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___54319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 25), ts_54318, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_54320 = invoke(stypy.reporting.localization.Localization(__file__, 158, 25), getitem___54319, int_54317)
        
        # Getting the type of 'self' (line 158)
        self_54321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'self')
        # Setting the type of the member 't_min' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), self_54321, 't_min', subscript_call_result_54320)
        
        # Assigning a Subscript to a Attribute (line 159):
        
        # Assigning a Subscript to a Attribute (line 159):
        
        # Obtaining the type of the subscript
        int_54322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 28), 'int')
        # Getting the type of 'ts' (line 159)
        ts_54323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'ts')
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___54324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 25), ts_54323, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_54325 = invoke(stypy.reporting.localization.Localization(__file__, 159, 25), getitem___54324, int_54322)
        
        # Getting the type of 'self' (line 159)
        self_54326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'self')
        # Setting the type of the member 't_max' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), self_54326, 't_max', subscript_call_result_54325)
        
        # Assigning a Name to a Attribute (line 160):
        
        # Assigning a Name to a Attribute (line 160):
        # Getting the type of 'True' (line 160)
        True_54327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 29), 'True')
        # Getting the type of 'self' (line 160)
        self_54328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'self')
        # Setting the type of the member 'ascending' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), self_54328, 'ascending', True_54327)
        
        # Assigning a Name to a Attribute (line 161):
        
        # Assigning a Name to a Attribute (line 161):
        # Getting the type of 'ts' (line 161)
        ts_54329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'ts')
        # Getting the type of 'self' (line 161)
        self_54330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'self')
        # Setting the type of the member 'ts_sorted' of a type (line 161)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), self_54330, 'ts_sorted', ts_54329)
        # SSA branch for the else part of an if statement (line 157)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Attribute (line 163):
        
        # Assigning a Subscript to a Attribute (line 163):
        
        # Obtaining the type of the subscript
        int_54331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 28), 'int')
        # Getting the type of 'ts' (line 163)
        ts_54332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 25), 'ts')
        # Obtaining the member '__getitem__' of a type (line 163)
        getitem___54333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 25), ts_54332, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 163)
        subscript_call_result_54334 = invoke(stypy.reporting.localization.Localization(__file__, 163, 25), getitem___54333, int_54331)
        
        # Getting the type of 'self' (line 163)
        self_54335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'self')
        # Setting the type of the member 't_min' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), self_54335, 't_min', subscript_call_result_54334)
        
        # Assigning a Subscript to a Attribute (line 164):
        
        # Assigning a Subscript to a Attribute (line 164):
        
        # Obtaining the type of the subscript
        int_54336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 28), 'int')
        # Getting the type of 'ts' (line 164)
        ts_54337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'ts')
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___54338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 25), ts_54337, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_54339 = invoke(stypy.reporting.localization.Localization(__file__, 164, 25), getitem___54338, int_54336)
        
        # Getting the type of 'self' (line 164)
        self_54340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'self')
        # Setting the type of the member 't_max' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), self_54340, 't_max', subscript_call_result_54339)
        
        # Assigning a Name to a Attribute (line 165):
        
        # Assigning a Name to a Attribute (line 165):
        # Getting the type of 'False' (line 165)
        False_54341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 29), 'False')
        # Getting the type of 'self' (line 165)
        self_54342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'self')
        # Setting the type of the member 'ascending' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), self_54342, 'ascending', False_54341)
        
        # Assigning a Subscript to a Attribute (line 166):
        
        # Assigning a Subscript to a Attribute (line 166):
        
        # Obtaining the type of the subscript
        int_54343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 34), 'int')
        slice_54344 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 166, 29), None, None, int_54343)
        # Getting the type of 'ts' (line 166)
        ts_54345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 29), 'ts')
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___54346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 29), ts_54345, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_54347 = invoke(stypy.reporting.localization.Localization(__file__, 166, 29), getitem___54346, slice_54344)
        
        # Getting the type of 'self' (line 166)
        self_54348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'self')
        # Setting the type of the member 'ts_sorted' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), self_54348, 'ts_sorted', subscript_call_result_54347)
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _call_single(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_call_single'
        module_type_store = module_type_store.open_function_context('_call_single', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        OdeSolution._call_single.__dict__.__setitem__('stypy_localization', localization)
        OdeSolution._call_single.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        OdeSolution._call_single.__dict__.__setitem__('stypy_type_store', module_type_store)
        OdeSolution._call_single.__dict__.__setitem__('stypy_function_name', 'OdeSolution._call_single')
        OdeSolution._call_single.__dict__.__setitem__('stypy_param_names_list', ['t'])
        OdeSolution._call_single.__dict__.__setitem__('stypy_varargs_param_name', None)
        OdeSolution._call_single.__dict__.__setitem__('stypy_kwargs_param_name', None)
        OdeSolution._call_single.__dict__.__setitem__('stypy_call_defaults', defaults)
        OdeSolution._call_single.__dict__.__setitem__('stypy_call_varargs', varargs)
        OdeSolution._call_single.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        OdeSolution._call_single.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdeSolution._call_single', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_call_single', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_call_single(...)' code ##################

        
        # Getting the type of 'self' (line 171)
        self_54349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'self')
        # Obtaining the member 'ascending' of a type (line 171)
        ascending_54350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 11), self_54349, 'ascending')
        # Testing the type of an if condition (line 171)
        if_condition_54351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 8), ascending_54350)
        # Assigning a type to the variable 'if_condition_54351' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'if_condition_54351', if_condition_54351)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 172):
        
        # Assigning a Call to a Name (line 172):
        
        # Call to searchsorted(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'self' (line 172)
        self_54354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'self', False)
        # Obtaining the member 'ts_sorted' of a type (line 172)
        ts_sorted_54355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 34), self_54354, 'ts_sorted')
        # Getting the type of 't' (line 172)
        t_54356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 50), 't', False)
        # Processing the call keyword arguments (line 172)
        str_54357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 58), 'str', 'left')
        keyword_54358 = str_54357
        kwargs_54359 = {'side': keyword_54358}
        # Getting the type of 'np' (line 172)
        np_54352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'np', False)
        # Obtaining the member 'searchsorted' of a type (line 172)
        searchsorted_54353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 18), np_54352, 'searchsorted')
        # Calling searchsorted(args, kwargs) (line 172)
        searchsorted_call_result_54360 = invoke(stypy.reporting.localization.Localization(__file__, 172, 18), searchsorted_54353, *[ts_sorted_54355, t_54356], **kwargs_54359)
        
        # Assigning a type to the variable 'ind' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'ind', searchsorted_call_result_54360)
        # SSA branch for the else part of an if statement (line 171)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to searchsorted(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'self' (line 174)
        self_54363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 34), 'self', False)
        # Obtaining the member 'ts_sorted' of a type (line 174)
        ts_sorted_54364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 34), self_54363, 'ts_sorted')
        # Getting the type of 't' (line 174)
        t_54365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 50), 't', False)
        # Processing the call keyword arguments (line 174)
        str_54366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 58), 'str', 'right')
        keyword_54367 = str_54366
        kwargs_54368 = {'side': keyword_54367}
        # Getting the type of 'np' (line 174)
        np_54361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'np', False)
        # Obtaining the member 'searchsorted' of a type (line 174)
        searchsorted_54362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 18), np_54361, 'searchsorted')
        # Calling searchsorted(args, kwargs) (line 174)
        searchsorted_call_result_54369 = invoke(stypy.reporting.localization.Localization(__file__, 174, 18), searchsorted_54362, *[ts_sorted_54364, t_54365], **kwargs_54368)
        
        # Assigning a type to the variable 'ind' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'ind', searchsorted_call_result_54369)
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to min(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Call to max(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'ind' (line 176)
        ind_54372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'ind', False)
        int_54373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 32), 'int')
        # Applying the binary operator '-' (line 176)
        result_sub_54374 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 26), '-', ind_54372, int_54373)
        
        int_54375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 35), 'int')
        # Processing the call keyword arguments (line 176)
        kwargs_54376 = {}
        # Getting the type of 'max' (line 176)
        max_54371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 22), 'max', False)
        # Calling max(args, kwargs) (line 176)
        max_call_result_54377 = invoke(stypy.reporting.localization.Localization(__file__, 176, 22), max_54371, *[result_sub_54374, int_54375], **kwargs_54376)
        
        # Getting the type of 'self' (line 176)
        self_54378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 39), 'self', False)
        # Obtaining the member 'n_segments' of a type (line 176)
        n_segments_54379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 39), self_54378, 'n_segments')
        int_54380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 57), 'int')
        # Applying the binary operator '-' (line 176)
        result_sub_54381 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 39), '-', n_segments_54379, int_54380)
        
        # Processing the call keyword arguments (line 176)
        kwargs_54382 = {}
        # Getting the type of 'min' (line 176)
        min_54370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 18), 'min', False)
        # Calling min(args, kwargs) (line 176)
        min_call_result_54383 = invoke(stypy.reporting.localization.Localization(__file__, 176, 18), min_54370, *[max_call_result_54377, result_sub_54381], **kwargs_54382)
        
        # Assigning a type to the variable 'segment' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'segment', min_call_result_54383)
        
        
        # Getting the type of 'self' (line 177)
        self_54384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'self')
        # Obtaining the member 'ascending' of a type (line 177)
        ascending_54385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 15), self_54384, 'ascending')
        # Applying the 'not' unary operator (line 177)
        result_not__54386 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 11), 'not', ascending_54385)
        
        # Testing the type of an if condition (line 177)
        if_condition_54387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 8), result_not__54386)
        # Assigning a type to the variable 'if_condition_54387' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'if_condition_54387', if_condition_54387)
        # SSA begins for if statement (line 177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 178):
        
        # Assigning a BinOp to a Name (line 178):
        # Getting the type of 'self' (line 178)
        self_54388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'self')
        # Obtaining the member 'n_segments' of a type (line 178)
        n_segments_54389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 22), self_54388, 'n_segments')
        int_54390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 40), 'int')
        # Applying the binary operator '-' (line 178)
        result_sub_54391 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 22), '-', n_segments_54389, int_54390)
        
        # Getting the type of 'segment' (line 178)
        segment_54392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 44), 'segment')
        # Applying the binary operator '-' (line 178)
        result_sub_54393 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 42), '-', result_sub_54391, segment_54392)
        
        # Assigning a type to the variable 'segment' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'segment', result_sub_54393)
        # SSA join for if statement (line 177)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to (...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 't' (line 180)
        t_54399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 42), 't', False)
        # Processing the call keyword arguments (line 180)
        kwargs_54400 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'segment' (line 180)
        segment_54394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 33), 'segment', False)
        # Getting the type of 'self' (line 180)
        self_54395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'self', False)
        # Obtaining the member 'interpolants' of a type (line 180)
        interpolants_54396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 15), self_54395, 'interpolants')
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___54397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 15), interpolants_54396, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_54398 = invoke(stypy.reporting.localization.Localization(__file__, 180, 15), getitem___54397, segment_54394)
        
        # Calling (args, kwargs) (line 180)
        _call_result_54401 = invoke(stypy.reporting.localization.Localization(__file__, 180, 15), subscript_call_result_54398, *[t_54399], **kwargs_54400)
        
        # Assigning a type to the variable 'stypy_return_type' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'stypy_return_type', _call_result_54401)
        
        # ################# End of '_call_single(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_call_single' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_54402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_54402)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_call_single'
        return stypy_return_type_54402


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        OdeSolution.__call__.__dict__.__setitem__('stypy_localization', localization)
        OdeSolution.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        OdeSolution.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        OdeSolution.__call__.__dict__.__setitem__('stypy_function_name', 'OdeSolution.__call__')
        OdeSolution.__call__.__dict__.__setitem__('stypy_param_names_list', ['t'])
        OdeSolution.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        OdeSolution.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        OdeSolution.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        OdeSolution.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        OdeSolution.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        OdeSolution.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OdeSolution.__call__', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_54403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, (-1)), 'str', 'Evaluate the solution.\n\n        Parameters\n        ----------\n        t : float or array_like with shape (n_points,)\n            Points to evaluate at.\n\n        Returns\n        -------\n        y : ndarray, shape (n_states,) or (n_states, n_points)\n            Computed values. Shape depends on whether `t` is a scalar or a\n            1-d array.\n        ')
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to asarray(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 't' (line 196)
        t_54406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 't', False)
        # Processing the call keyword arguments (line 196)
        kwargs_54407 = {}
        # Getting the type of 'np' (line 196)
        np_54404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 196)
        asarray_54405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), np_54404, 'asarray')
        # Calling asarray(args, kwargs) (line 196)
        asarray_call_result_54408 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), asarray_54405, *[t_54406], **kwargs_54407)
        
        # Assigning a type to the variable 't' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 't', asarray_call_result_54408)
        
        
        # Getting the type of 't' (line 198)
        t_54409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 't')
        # Obtaining the member 'ndim' of a type (line 198)
        ndim_54410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), t_54409, 'ndim')
        int_54411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 21), 'int')
        # Applying the binary operator '==' (line 198)
        result_eq_54412 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), '==', ndim_54410, int_54411)
        
        # Testing the type of an if condition (line 198)
        if_condition_54413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_eq_54412)
        # Assigning a type to the variable 'if_condition_54413' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_54413', if_condition_54413)
        # SSA begins for if statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _call_single(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 't' (line 199)
        t_54416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 37), 't', False)
        # Processing the call keyword arguments (line 199)
        kwargs_54417 = {}
        # Getting the type of 'self' (line 199)
        self_54414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'self', False)
        # Obtaining the member '_call_single' of a type (line 199)
        _call_single_54415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 19), self_54414, '_call_single')
        # Calling _call_single(args, kwargs) (line 199)
        _call_single_call_result_54418 = invoke(stypy.reporting.localization.Localization(__file__, 199, 19), _call_single_54415, *[t_54416], **kwargs_54417)
        
        # Assigning a type to the variable 'stypy_return_type' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'stypy_return_type', _call_single_call_result_54418)
        # SSA join for if statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to argsort(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 't' (line 201)
        t_54421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 't', False)
        # Processing the call keyword arguments (line 201)
        kwargs_54422 = {}
        # Getting the type of 'np' (line 201)
        np_54419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'np', False)
        # Obtaining the member 'argsort' of a type (line 201)
        argsort_54420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), np_54419, 'argsort')
        # Calling argsort(args, kwargs) (line 201)
        argsort_call_result_54423 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), argsort_54420, *[t_54421], **kwargs_54422)
        
        # Assigning a type to the variable 'order' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'order', argsort_call_result_54423)
        
        # Assigning a Call to a Name (line 202):
        
        # Assigning a Call to a Name (line 202):
        
        # Call to empty_like(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'order' (line 202)
        order_54426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 32), 'order', False)
        # Processing the call keyword arguments (line 202)
        kwargs_54427 = {}
        # Getting the type of 'np' (line 202)
        np_54424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 202)
        empty_like_54425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 18), np_54424, 'empty_like')
        # Calling empty_like(args, kwargs) (line 202)
        empty_like_call_result_54428 = invoke(stypy.reporting.localization.Localization(__file__, 202, 18), empty_like_54425, *[order_54426], **kwargs_54427)
        
        # Assigning a type to the variable 'reverse' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'reverse', empty_like_call_result_54428)
        
        # Assigning a Call to a Subscript (line 203):
        
        # Assigning a Call to a Subscript (line 203):
        
        # Call to arange(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Obtaining the type of the subscript
        int_54431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 47), 'int')
        # Getting the type of 'order' (line 203)
        order_54432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 35), 'order', False)
        # Obtaining the member 'shape' of a type (line 203)
        shape_54433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 35), order_54432, 'shape')
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___54434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 35), shape_54433, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_54435 = invoke(stypy.reporting.localization.Localization(__file__, 203, 35), getitem___54434, int_54431)
        
        # Processing the call keyword arguments (line 203)
        kwargs_54436 = {}
        # Getting the type of 'np' (line 203)
        np_54429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'np', False)
        # Obtaining the member 'arange' of a type (line 203)
        arange_54430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 25), np_54429, 'arange')
        # Calling arange(args, kwargs) (line 203)
        arange_call_result_54437 = invoke(stypy.reporting.localization.Localization(__file__, 203, 25), arange_54430, *[subscript_call_result_54435], **kwargs_54436)
        
        # Getting the type of 'reverse' (line 203)
        reverse_54438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'reverse')
        # Getting the type of 'order' (line 203)
        order_54439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'order')
        # Storing an element on a container (line 203)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 8), reverse_54438, (order_54439, arange_call_result_54437))
        
        # Assigning a Subscript to a Name (line 204):
        
        # Assigning a Subscript to a Name (line 204):
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 204)
        order_54440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'order')
        # Getting the type of 't' (line 204)
        t_54441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 't')
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___54442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 19), t_54441, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_54443 = invoke(stypy.reporting.localization.Localization(__file__, 204, 19), getitem___54442, order_54440)
        
        # Assigning a type to the variable 't_sorted' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 't_sorted', subscript_call_result_54443)
        
        # Getting the type of 'self' (line 207)
        self_54444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'self')
        # Obtaining the member 'ascending' of a type (line 207)
        ascending_54445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 11), self_54444, 'ascending')
        # Testing the type of an if condition (line 207)
        if_condition_54446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), ascending_54445)
        # Assigning a type to the variable 'if_condition_54446' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_54446', if_condition_54446)
        # SSA begins for if statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Call to searchsorted(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'self' (line 208)
        self_54449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 39), 'self', False)
        # Obtaining the member 'ts_sorted' of a type (line 208)
        ts_sorted_54450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 39), self_54449, 'ts_sorted')
        # Getting the type of 't_sorted' (line 208)
        t_sorted_54451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 55), 't_sorted', False)
        # Processing the call keyword arguments (line 208)
        str_54452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 70), 'str', 'left')
        keyword_54453 = str_54452
        kwargs_54454 = {'side': keyword_54453}
        # Getting the type of 'np' (line 208)
        np_54447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'np', False)
        # Obtaining the member 'searchsorted' of a type (line 208)
        searchsorted_54448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 23), np_54447, 'searchsorted')
        # Calling searchsorted(args, kwargs) (line 208)
        searchsorted_call_result_54455 = invoke(stypy.reporting.localization.Localization(__file__, 208, 23), searchsorted_54448, *[ts_sorted_54450, t_sorted_54451], **kwargs_54454)
        
        # Assigning a type to the variable 'segments' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'segments', searchsorted_call_result_54455)
        # SSA branch for the else part of an if statement (line 207)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 210):
        
        # Assigning a Call to a Name (line 210):
        
        # Call to searchsorted(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'self' (line 210)
        self_54458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 39), 'self', False)
        # Obtaining the member 'ts_sorted' of a type (line 210)
        ts_sorted_54459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 39), self_54458, 'ts_sorted')
        # Getting the type of 't_sorted' (line 210)
        t_sorted_54460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 55), 't_sorted', False)
        # Processing the call keyword arguments (line 210)
        str_54461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 70), 'str', 'right')
        keyword_54462 = str_54461
        kwargs_54463 = {'side': keyword_54462}
        # Getting the type of 'np' (line 210)
        np_54456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), 'np', False)
        # Obtaining the member 'searchsorted' of a type (line 210)
        searchsorted_54457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 23), np_54456, 'searchsorted')
        # Calling searchsorted(args, kwargs) (line 210)
        searchsorted_call_result_54464 = invoke(stypy.reporting.localization.Localization(__file__, 210, 23), searchsorted_54457, *[ts_sorted_54459, t_sorted_54460], **kwargs_54463)
        
        # Assigning a type to the variable 'segments' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'segments', searchsorted_call_result_54464)
        # SSA join for if statement (line 207)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'segments' (line 211)
        segments_54465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'segments')
        int_54466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 20), 'int')
        # Applying the binary operator '-=' (line 211)
        result_isub_54467 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 8), '-=', segments_54465, int_54466)
        # Assigning a type to the variable 'segments' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'segments', result_isub_54467)
        
        
        # Assigning a Num to a Subscript (line 212):
        
        # Assigning a Num to a Subscript (line 212):
        int_54468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 33), 'int')
        # Getting the type of 'segments' (line 212)
        segments_54469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'segments')
        
        # Getting the type of 'segments' (line 212)
        segments_54470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 17), 'segments')
        int_54471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 28), 'int')
        # Applying the binary operator '<' (line 212)
        result_lt_54472 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 17), '<', segments_54470, int_54471)
        
        # Storing an element on a container (line 212)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 8), segments_54469, (result_lt_54472, int_54468))
        
        # Assigning a BinOp to a Subscript (line 213):
        
        # Assigning a BinOp to a Subscript (line 213):
        # Getting the type of 'self' (line 213)
        self_54473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 51), 'self')
        # Obtaining the member 'n_segments' of a type (line 213)
        n_segments_54474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 51), self_54473, 'n_segments')
        int_54475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 69), 'int')
        # Applying the binary operator '-' (line 213)
        result_sub_54476 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 51), '-', n_segments_54474, int_54475)
        
        # Getting the type of 'segments' (line 213)
        segments_54477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'segments')
        
        # Getting the type of 'segments' (line 213)
        segments_54478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 17), 'segments')
        # Getting the type of 'self' (line 213)
        self_54479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'self')
        # Obtaining the member 'n_segments' of a type (line 213)
        n_segments_54480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 28), self_54479, 'n_segments')
        int_54481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 46), 'int')
        # Applying the binary operator '-' (line 213)
        result_sub_54482 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 28), '-', n_segments_54480, int_54481)
        
        # Applying the binary operator '>' (line 213)
        result_gt_54483 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 17), '>', segments_54478, result_sub_54482)
        
        # Storing an element on a container (line 213)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 8), segments_54477, (result_gt_54483, result_sub_54476))
        
        
        # Getting the type of 'self' (line 214)
        self_54484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'self')
        # Obtaining the member 'ascending' of a type (line 214)
        ascending_54485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 15), self_54484, 'ascending')
        # Applying the 'not' unary operator (line 214)
        result_not__54486 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 11), 'not', ascending_54485)
        
        # Testing the type of an if condition (line 214)
        if_condition_54487 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 8), result_not__54486)
        # Assigning a type to the variable 'if_condition_54487' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'if_condition_54487', if_condition_54487)
        # SSA begins for if statement (line 214)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 215):
        
        # Assigning a BinOp to a Name (line 215):
        # Getting the type of 'self' (line 215)
        self_54488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'self')
        # Obtaining the member 'n_segments' of a type (line 215)
        n_segments_54489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 23), self_54488, 'n_segments')
        int_54490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 41), 'int')
        # Applying the binary operator '-' (line 215)
        result_sub_54491 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 23), '-', n_segments_54489, int_54490)
        
        # Getting the type of 'segments' (line 215)
        segments_54492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 45), 'segments')
        # Applying the binary operator '-' (line 215)
        result_sub_54493 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 43), '-', result_sub_54491, segments_54492)
        
        # Assigning a type to the variable 'segments' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'segments', result_sub_54493)
        # SSA join for if statement (line 214)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 217):
        
        # Assigning a List to a Name (line 217):
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_54494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        
        # Assigning a type to the variable 'ys' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'ys', list_54494)
        
        # Assigning a Num to a Name (line 218):
        
        # Assigning a Num to a Name (line 218):
        int_54495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 22), 'int')
        # Assigning a type to the variable 'group_start' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'group_start', int_54495)
        
        
        # Call to groupby(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'segments' (line 219)
        segments_54497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 38), 'segments', False)
        # Processing the call keyword arguments (line 219)
        kwargs_54498 = {}
        # Getting the type of 'groupby' (line 219)
        groupby_54496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'groupby', False)
        # Calling groupby(args, kwargs) (line 219)
        groupby_call_result_54499 = invoke(stypy.reporting.localization.Localization(__file__, 219, 30), groupby_54496, *[segments_54497], **kwargs_54498)
        
        # Testing the type of a for loop iterable (line 219)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 219, 8), groupby_call_result_54499)
        # Getting the type of the for loop variable (line 219)
        for_loop_var_54500 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 219, 8), groupby_call_result_54499)
        # Assigning a type to the variable 'segment' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'segment', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 8), for_loop_var_54500))
        # Assigning a type to the variable 'group' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'group', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 8), for_loop_var_54500))
        # SSA begins for a for statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 220):
        
        # Assigning a BinOp to a Name (line 220):
        # Getting the type of 'group_start' (line 220)
        group_start_54501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'group_start')
        
        # Call to len(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Call to list(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'group' (line 220)
        group_54504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 47), 'group', False)
        # Processing the call keyword arguments (line 220)
        kwargs_54505 = {}
        # Getting the type of 'list' (line 220)
        list_54503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 42), 'list', False)
        # Calling list(args, kwargs) (line 220)
        list_call_result_54506 = invoke(stypy.reporting.localization.Localization(__file__, 220, 42), list_54503, *[group_54504], **kwargs_54505)
        
        # Processing the call keyword arguments (line 220)
        kwargs_54507 = {}
        # Getting the type of 'len' (line 220)
        len_54502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'len', False)
        # Calling len(args, kwargs) (line 220)
        len_call_result_54508 = invoke(stypy.reporting.localization.Localization(__file__, 220, 38), len_54502, *[list_call_result_54506], **kwargs_54507)
        
        # Applying the binary operator '+' (line 220)
        result_add_54509 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 24), '+', group_start_54501, len_call_result_54508)
        
        # Assigning a type to the variable 'group_end' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'group_end', result_add_54509)
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to (...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining the type of the subscript
        # Getting the type of 'group_start' (line 221)
        group_start_54515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 52), 'group_start', False)
        # Getting the type of 'group_end' (line 221)
        group_end_54516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 64), 'group_end', False)
        slice_54517 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 221, 43), group_start_54515, group_end_54516, None)
        # Getting the type of 't_sorted' (line 221)
        t_sorted_54518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), 't_sorted', False)
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___54519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 43), t_sorted_54518, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_54520 = invoke(stypy.reporting.localization.Localization(__file__, 221, 43), getitem___54519, slice_54517)
        
        # Processing the call keyword arguments (line 221)
        kwargs_54521 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'segment' (line 221)
        segment_54510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 34), 'segment', False)
        # Getting the type of 'self' (line 221)
        self_54511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'self', False)
        # Obtaining the member 'interpolants' of a type (line 221)
        interpolants_54512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), self_54511, 'interpolants')
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___54513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), interpolants_54512, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_54514 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), getitem___54513, segment_54510)
        
        # Calling (args, kwargs) (line 221)
        _call_result_54522 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), subscript_call_result_54514, *[subscript_call_result_54520], **kwargs_54521)
        
        # Assigning a type to the variable 'y' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'y', _call_result_54522)
        
        # Call to append(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'y' (line 222)
        y_54525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 22), 'y', False)
        # Processing the call keyword arguments (line 222)
        kwargs_54526 = {}
        # Getting the type of 'ys' (line 222)
        ys_54523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'ys', False)
        # Obtaining the member 'append' of a type (line 222)
        append_54524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), ys_54523, 'append')
        # Calling append(args, kwargs) (line 222)
        append_call_result_54527 = invoke(stypy.reporting.localization.Localization(__file__, 222, 12), append_54524, *[y_54525], **kwargs_54526)
        
        
        # Assigning a Name to a Name (line 223):
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'group_end' (line 223)
        group_end_54528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 26), 'group_end')
        # Assigning a type to the variable 'group_start' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'group_start', group_end_54528)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 225):
        
        # Assigning a Call to a Name (line 225):
        
        # Call to hstack(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'ys' (line 225)
        ys_54531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 23), 'ys', False)
        # Processing the call keyword arguments (line 225)
        kwargs_54532 = {}
        # Getting the type of 'np' (line 225)
        np_54529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 13), 'np', False)
        # Obtaining the member 'hstack' of a type (line 225)
        hstack_54530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 13), np_54529, 'hstack')
        # Calling hstack(args, kwargs) (line 225)
        hstack_call_result_54533 = invoke(stypy.reporting.localization.Localization(__file__, 225, 13), hstack_54530, *[ys_54531], **kwargs_54532)
        
        # Assigning a type to the variable 'ys' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'ys', hstack_call_result_54533)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        slice_54534 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 226, 13), None, None, None)
        # Getting the type of 'reverse' (line 226)
        reverse_54535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'reverse')
        # Getting the type of 'ys' (line 226)
        ys_54536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 13), 'ys')
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___54537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 13), ys_54536, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_54538 = invoke(stypy.reporting.localization.Localization(__file__, 226, 13), getitem___54537, (slice_54534, reverse_54535))
        
        # Assigning a type to the variable 'ys' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'ys', subscript_call_result_54538)
        # Getting the type of 'ys' (line 228)
        ys_54539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'ys')
        # Assigning a type to the variable 'stypy_return_type' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stypy_return_type', ys_54539)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_54540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_54540)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_54540


# Assigning a type to the variable 'OdeSolution' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'OdeSolution', OdeSolution)

# Assigning a BinOp to a Name (line 231):

# Assigning a BinOp to a Name (line 231):
# Getting the type of 'EPS' (line 231)
EPS_54541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 22), 'EPS')
float_54542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 29), 'float')
# Applying the binary operator '**' (line 231)
result_pow_54543 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 22), '**', EPS_54541, float_54542)

# Assigning a type to the variable 'NUM_JAC_DIFF_REJECT' (line 231)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 0), 'NUM_JAC_DIFF_REJECT', result_pow_54543)

# Assigning a BinOp to a Name (line 232):

# Assigning a BinOp to a Name (line 232):
# Getting the type of 'EPS' (line 232)
EPS_54544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 21), 'EPS')
float_54545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 28), 'float')
# Applying the binary operator '**' (line 232)
result_pow_54546 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 21), '**', EPS_54544, float_54545)

# Assigning a type to the variable 'NUM_JAC_DIFF_SMALL' (line 232)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 0), 'NUM_JAC_DIFF_SMALL', result_pow_54546)

# Assigning a BinOp to a Name (line 233):

# Assigning a BinOp to a Name (line 233):
# Getting the type of 'EPS' (line 233)
EPS_54547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 19), 'EPS')
float_54548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 26), 'float')
# Applying the binary operator '**' (line 233)
result_pow_54549 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 19), '**', EPS_54547, float_54548)

# Assigning a type to the variable 'NUM_JAC_DIFF_BIG' (line 233)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 0), 'NUM_JAC_DIFF_BIG', result_pow_54549)

# Assigning a BinOp to a Name (line 234):

# Assigning a BinOp to a Name (line 234):
float_54550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 21), 'float')
# Getting the type of 'EPS' (line 234)
EPS_54551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 27), 'EPS')
# Applying the binary operator '*' (line 234)
result_mul_54552 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 21), '*', float_54550, EPS_54551)

# Assigning a type to the variable 'NUM_JAC_MIN_FACTOR' (line 234)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'NUM_JAC_MIN_FACTOR', result_mul_54552)

# Assigning a Num to a Name (line 235):

# Assigning a Num to a Name (line 235):
int_54553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 26), 'int')
# Assigning a type to the variable 'NUM_JAC_FACTOR_INCREASE' (line 235)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'NUM_JAC_FACTOR_INCREASE', int_54553)

# Assigning a Num to a Name (line 236):

# Assigning a Num to a Name (line 236):
float_54554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 26), 'float')
# Assigning a type to the variable 'NUM_JAC_FACTOR_DECREASE' (line 236)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'NUM_JAC_FACTOR_DECREASE', float_54554)

@norecursion
def num_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 239)
    None_54555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 54), 'None')
    defaults = [None_54555]
    # Create a new context for function 'num_jac'
    module_type_store = module_type_store.open_function_context('num_jac', 239, 0, False)
    
    # Passed parameters checking function
    num_jac.stypy_localization = localization
    num_jac.stypy_type_of_self = None
    num_jac.stypy_type_store = module_type_store
    num_jac.stypy_function_name = 'num_jac'
    num_jac.stypy_param_names_list = ['fun', 't', 'y', 'f', 'threshold', 'factor', 'sparsity']
    num_jac.stypy_varargs_param_name = None
    num_jac.stypy_kwargs_param_name = None
    num_jac.stypy_call_defaults = defaults
    num_jac.stypy_call_varargs = varargs
    num_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'num_jac', ['fun', 't', 'y', 'f', 'threshold', 'factor', 'sparsity'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'num_jac', localization, ['fun', 't', 'y', 'f', 'threshold', 'factor', 'sparsity'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'num_jac(...)' code ##################

    str_54556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, (-1)), 'str', 'Finite differences Jacobian approximation tailored for ODE solvers.\n\n    This function computes finite difference approximation to the Jacobian\n    matrix of `fun` with respect to `y` using forward differences.\n    The Jacobian matrix has shape (n, n) and its element (i, j) is equal to\n    ``d f_i / d y_j``.\n\n    A special feature of this function is the ability to correct the step\n    size from iteration to iteration. The main idea is to keep the finite\n    difference significantly separated from its round-off error which\n    approximately equals ``EPS * np.abs(f)``. It reduces a possibility of a\n    huge error and assures that the estimated derivative are reasonably close\n    to the true values (i.e. the finite difference approximation is at least\n    qualitatively reflects the structure of the true Jacobian).\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system implemented in a vectorized fashion.\n    t : float\n        Current time.\n    y : ndarray, shape (n,)\n        Current state.\n    f : ndarray, shape (n,)\n        Value of the right hand side at (t, y).\n    threshold : float\n        Threshold for `y` value used for computing the step size as\n        ``factor * np.maximum(np.abs(y), threshold)``. Typically the value of\n        absolute tolerance (atol) for a solver should be passed as `threshold`.\n    factor : ndarray with shape (n,) or None\n        Factor to use for computing the step size. Pass None for the very\n        evaluation, then use the value returned from this function.\n    sparsity : tuple (structure, groups) or None\n        Sparsity structure of the Jacobian, `structure` must be csc_matrix.\n\n    Returns\n    -------\n    J : ndarray or csc_matrix, shape (n, n)\n        Jacobian matrix.\n    factor : ndarray, shape (n,)\n        Suggested `factor` for the next evaluation.\n    ')
    
    # Assigning a Call to a Name (line 282):
    
    # Assigning a Call to a Name (line 282):
    
    # Call to asarray(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'y' (line 282)
    y_54559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'y', False)
    # Processing the call keyword arguments (line 282)
    kwargs_54560 = {}
    # Getting the type of 'np' (line 282)
    np_54557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 282)
    asarray_54558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), np_54557, 'asarray')
    # Calling asarray(args, kwargs) (line 282)
    asarray_call_result_54561 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), asarray_54558, *[y_54559], **kwargs_54560)
    
    # Assigning a type to the variable 'y' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'y', asarray_call_result_54561)
    
    # Assigning a Subscript to a Name (line 283):
    
    # Assigning a Subscript to a Name (line 283):
    
    # Obtaining the type of the subscript
    int_54562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 16), 'int')
    # Getting the type of 'y' (line 283)
    y_54563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'y')
    # Obtaining the member 'shape' of a type (line 283)
    shape_54564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), y_54563, 'shape')
    # Obtaining the member '__getitem__' of a type (line 283)
    getitem___54565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), shape_54564, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 283)
    subscript_call_result_54566 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), getitem___54565, int_54562)
    
    # Assigning a type to the variable 'n' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'n', subscript_call_result_54566)
    
    
    # Getting the type of 'n' (line 284)
    n_54567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 7), 'n')
    int_54568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 12), 'int')
    # Applying the binary operator '==' (line 284)
    result_eq_54569 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 7), '==', n_54567, int_54568)
    
    # Testing the type of an if condition (line 284)
    if_condition_54570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 4), result_eq_54569)
    # Assigning a type to the variable 'if_condition_54570' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'if_condition_54570', if_condition_54570)
    # SSA begins for if statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 285)
    tuple_54571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 285)
    # Adding element type (line 285)
    
    # Call to empty(...): (line 285)
    # Processing the call arguments (line 285)
    
    # Obtaining an instance of the builtin type 'tuple' (line 285)
    tuple_54574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 285)
    # Adding element type (line 285)
    int_54575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 25), tuple_54574, int_54575)
    # Adding element type (line 285)
    int_54576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 25), tuple_54574, int_54576)
    
    # Processing the call keyword arguments (line 285)
    kwargs_54577 = {}
    # Getting the type of 'np' (line 285)
    np_54572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'np', False)
    # Obtaining the member 'empty' of a type (line 285)
    empty_54573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 15), np_54572, 'empty')
    # Calling empty(args, kwargs) (line 285)
    empty_call_result_54578 = invoke(stypy.reporting.localization.Localization(__file__, 285, 15), empty_54573, *[tuple_54574], **kwargs_54577)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 15), tuple_54571, empty_call_result_54578)
    # Adding element type (line 285)
    # Getting the type of 'factor' (line 285)
    factor_54579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 33), 'factor')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 15), tuple_54571, factor_54579)
    
    # Assigning a type to the variable 'stypy_return_type' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'stypy_return_type', tuple_54571)
    # SSA join for if statement (line 284)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 287)
    # Getting the type of 'factor' (line 287)
    factor_54580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 7), 'factor')
    # Getting the type of 'None' (line 287)
    None_54581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 17), 'None')
    
    (may_be_54582, more_types_in_union_54583) = may_be_none(factor_54580, None_54581)

    if may_be_54582:

        if more_types_in_union_54583:
            # Runtime conditional SSA (line 287)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 288):
        
        # Assigning a BinOp to a Name (line 288):
        
        # Call to ones(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'n' (line 288)
        n_54586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 25), 'n', False)
        # Processing the call keyword arguments (line 288)
        kwargs_54587 = {}
        # Getting the type of 'np' (line 288)
        np_54584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 17), 'np', False)
        # Obtaining the member 'ones' of a type (line 288)
        ones_54585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 17), np_54584, 'ones')
        # Calling ones(args, kwargs) (line 288)
        ones_call_result_54588 = invoke(stypy.reporting.localization.Localization(__file__, 288, 17), ones_54585, *[n_54586], **kwargs_54587)
        
        # Getting the type of 'EPS' (line 288)
        EPS_54589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'EPS')
        float_54590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 37), 'float')
        # Applying the binary operator '**' (line 288)
        result_pow_54591 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 30), '**', EPS_54589, float_54590)
        
        # Applying the binary operator '*' (line 288)
        result_mul_54592 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 17), '*', ones_call_result_54588, result_pow_54591)
        
        # Assigning a type to the variable 'factor' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'factor', result_mul_54592)

        if more_types_in_union_54583:
            # Runtime conditional SSA for else branch (line 287)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_54582) or more_types_in_union_54583):
        
        # Assigning a Call to a Name (line 290):
        
        # Assigning a Call to a Name (line 290):
        
        # Call to copy(...): (line 290)
        # Processing the call keyword arguments (line 290)
        kwargs_54595 = {}
        # Getting the type of 'factor' (line 290)
        factor_54593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 17), 'factor', False)
        # Obtaining the member 'copy' of a type (line 290)
        copy_54594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 17), factor_54593, 'copy')
        # Calling copy(args, kwargs) (line 290)
        copy_call_result_54596 = invoke(stypy.reporting.localization.Localization(__file__, 290, 17), copy_54594, *[], **kwargs_54595)
        
        # Assigning a type to the variable 'factor' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'factor', copy_call_result_54596)

        if (may_be_54582 and more_types_in_union_54583):
            # SSA join for if statement (line 287)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 295):
    
    # Assigning a BinOp to a Name (line 295):
    int_54597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 13), 'int')
    
    # Call to astype(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'float' (line 295)
    float_54606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 42), 'float', False)
    # Processing the call keyword arguments (line 295)
    kwargs_54607 = {}
    
    
    # Call to real(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'f' (line 295)
    f_54600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 26), 'f', False)
    # Processing the call keyword arguments (line 295)
    kwargs_54601 = {}
    # Getting the type of 'np' (line 295)
    np_54598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'np', False)
    # Obtaining the member 'real' of a type (line 295)
    real_54599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 18), np_54598, 'real')
    # Calling real(args, kwargs) (line 295)
    real_call_result_54602 = invoke(stypy.reporting.localization.Localization(__file__, 295, 18), real_54599, *[f_54600], **kwargs_54601)
    
    int_54603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 32), 'int')
    # Applying the binary operator '>=' (line 295)
    result_ge_54604 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 18), '>=', real_call_result_54602, int_54603)
    
    # Obtaining the member 'astype' of a type (line 295)
    astype_54605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 18), result_ge_54604, 'astype')
    # Calling astype(args, kwargs) (line 295)
    astype_call_result_54608 = invoke(stypy.reporting.localization.Localization(__file__, 295, 18), astype_54605, *[float_54606], **kwargs_54607)
    
    # Applying the binary operator '*' (line 295)
    result_mul_54609 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 13), '*', int_54597, astype_call_result_54608)
    
    int_54610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 51), 'int')
    # Applying the binary operator '-' (line 295)
    result_sub_54611 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 13), '-', result_mul_54609, int_54610)
    
    # Assigning a type to the variable 'f_sign' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'f_sign', result_sub_54611)
    
    # Assigning a BinOp to a Name (line 296):
    
    # Assigning a BinOp to a Name (line 296):
    # Getting the type of 'f_sign' (line 296)
    f_sign_54612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 14), 'f_sign')
    
    # Call to maximum(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 'threshold' (line 296)
    threshold_54615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 34), 'threshold', False)
    
    # Call to abs(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 'y' (line 296)
    y_54618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 52), 'y', False)
    # Processing the call keyword arguments (line 296)
    kwargs_54619 = {}
    # Getting the type of 'np' (line 296)
    np_54616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 45), 'np', False)
    # Obtaining the member 'abs' of a type (line 296)
    abs_54617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 45), np_54616, 'abs')
    # Calling abs(args, kwargs) (line 296)
    abs_call_result_54620 = invoke(stypy.reporting.localization.Localization(__file__, 296, 45), abs_54617, *[y_54618], **kwargs_54619)
    
    # Processing the call keyword arguments (line 296)
    kwargs_54621 = {}
    # Getting the type of 'np' (line 296)
    np_54613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 23), 'np', False)
    # Obtaining the member 'maximum' of a type (line 296)
    maximum_54614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 23), np_54613, 'maximum')
    # Calling maximum(args, kwargs) (line 296)
    maximum_call_result_54622 = invoke(stypy.reporting.localization.Localization(__file__, 296, 23), maximum_54614, *[threshold_54615, abs_call_result_54620], **kwargs_54621)
    
    # Applying the binary operator '*' (line 296)
    result_mul_54623 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 14), '*', f_sign_54612, maximum_call_result_54622)
    
    # Assigning a type to the variable 'y_scale' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'y_scale', result_mul_54623)
    
    # Assigning a BinOp to a Name (line 297):
    
    # Assigning a BinOp to a Name (line 297):
    # Getting the type of 'y' (line 297)
    y_54624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 9), 'y')
    # Getting the type of 'factor' (line 297)
    factor_54625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 13), 'factor')
    # Getting the type of 'y_scale' (line 297)
    y_scale_54626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 22), 'y_scale')
    # Applying the binary operator '*' (line 297)
    result_mul_54627 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 13), '*', factor_54625, y_scale_54626)
    
    # Applying the binary operator '+' (line 297)
    result_add_54628 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 9), '+', y_54624, result_mul_54627)
    
    # Getting the type of 'y' (line 297)
    y_54629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 33), 'y')
    # Applying the binary operator '-' (line 297)
    result_sub_54630 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 8), '-', result_add_54628, y_54629)
    
    # Assigning a type to the variable 'h' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'h', result_sub_54630)
    
    
    # Obtaining the type of the subscript
    int_54631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 32), 'int')
    
    # Call to nonzero(...): (line 301)
    # Processing the call arguments (line 301)
    
    # Getting the type of 'h' (line 301)
    h_54634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'h', False)
    int_54635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 29), 'int')
    # Applying the binary operator '==' (line 301)
    result_eq_54636 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 24), '==', h_54634, int_54635)
    
    # Processing the call keyword arguments (line 301)
    kwargs_54637 = {}
    # Getting the type of 'np' (line 301)
    np_54632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 13), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 301)
    nonzero_54633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 13), np_54632, 'nonzero')
    # Calling nonzero(args, kwargs) (line 301)
    nonzero_call_result_54638 = invoke(stypy.reporting.localization.Localization(__file__, 301, 13), nonzero_54633, *[result_eq_54636], **kwargs_54637)
    
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___54639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 13), nonzero_call_result_54638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_54640 = invoke(stypy.reporting.localization.Localization(__file__, 301, 13), getitem___54639, int_54631)
    
    # Testing the type of a for loop iterable (line 301)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 301, 4), subscript_call_result_54640)
    # Getting the type of the for loop variable (line 301)
    for_loop_var_54641 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 301, 4), subscript_call_result_54640)
    # Assigning a type to the variable 'i' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'i', for_loop_var_54641)
    # SSA begins for a for statement (line 301)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 302)
    i_54642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'i')
    # Getting the type of 'h' (line 302)
    h_54643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 14), 'h')
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___54644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 14), h_54643, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_54645 = invoke(stypy.reporting.localization.Localization(__file__, 302, 14), getitem___54644, i_54642)
    
    int_54646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 22), 'int')
    # Applying the binary operator '==' (line 302)
    result_eq_54647 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 14), '==', subscript_call_result_54645, int_54646)
    
    # Testing the type of an if condition (line 302)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 8), result_eq_54647)
    # SSA begins for while statement (line 302)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'factor' (line 303)
    factor_54648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'factor')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 303)
    i_54649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 19), 'i')
    # Getting the type of 'factor' (line 303)
    factor_54650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'factor')
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___54651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 12), factor_54650, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_54652 = invoke(stypy.reporting.localization.Localization(__file__, 303, 12), getitem___54651, i_54649)
    
    int_54653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 25), 'int')
    # Applying the binary operator '*=' (line 303)
    result_imul_54654 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 12), '*=', subscript_call_result_54652, int_54653)
    # Getting the type of 'factor' (line 303)
    factor_54655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'factor')
    # Getting the type of 'i' (line 303)
    i_54656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 19), 'i')
    # Storing an element on a container (line 303)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 12), factor_54655, (i_54656, result_imul_54654))
    
    
    # Assigning a BinOp to a Subscript (line 304):
    
    # Assigning a BinOp to a Subscript (line 304):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 304)
    i_54657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 22), 'i')
    # Getting the type of 'y' (line 304)
    y_54658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'y')
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___54659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 20), y_54658, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_54660 = invoke(stypy.reporting.localization.Localization(__file__, 304, 20), getitem___54659, i_54657)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 304)
    i_54661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 34), 'i')
    # Getting the type of 'factor' (line 304)
    factor_54662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'factor')
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___54663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 27), factor_54662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_54664 = invoke(stypy.reporting.localization.Localization(__file__, 304, 27), getitem___54663, i_54661)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 304)
    i_54665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 47), 'i')
    # Getting the type of 'y_scale' (line 304)
    y_scale_54666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 39), 'y_scale')
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___54667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 39), y_scale_54666, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_54668 = invoke(stypy.reporting.localization.Localization(__file__, 304, 39), getitem___54667, i_54665)
    
    # Applying the binary operator '*' (line 304)
    result_mul_54669 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 27), '*', subscript_call_result_54664, subscript_call_result_54668)
    
    # Applying the binary operator '+' (line 304)
    result_add_54670 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 20), '+', subscript_call_result_54660, result_mul_54669)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 304)
    i_54671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 55), 'i')
    # Getting the type of 'y' (line 304)
    y_54672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 53), 'y')
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___54673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 53), y_54672, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_54674 = invoke(stypy.reporting.localization.Localization(__file__, 304, 53), getitem___54673, i_54671)
    
    # Applying the binary operator '-' (line 304)
    result_sub_54675 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 19), '-', result_add_54670, subscript_call_result_54674)
    
    # Getting the type of 'h' (line 304)
    h_54676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'h')
    # Getting the type of 'i' (line 304)
    i_54677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 14), 'i')
    # Storing an element on a container (line 304)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), h_54676, (i_54677, result_sub_54675))
    # SSA join for while statement (line 302)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 306)
    # Getting the type of 'sparsity' (line 306)
    sparsity_54678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 7), 'sparsity')
    # Getting the type of 'None' (line 306)
    None_54679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'None')
    
    (may_be_54680, more_types_in_union_54681) = may_be_none(sparsity_54678, None_54679)

    if may_be_54680:

        if more_types_in_union_54681:
            # Runtime conditional SSA (line 306)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to _dense_num_jac(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'fun' (line 307)
        fun_54683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 30), 'fun', False)
        # Getting the type of 't' (line 307)
        t_54684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 35), 't', False)
        # Getting the type of 'y' (line 307)
        y_54685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 38), 'y', False)
        # Getting the type of 'f' (line 307)
        f_54686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 41), 'f', False)
        # Getting the type of 'h' (line 307)
        h_54687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 44), 'h', False)
        # Getting the type of 'factor' (line 307)
        factor_54688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 47), 'factor', False)
        # Getting the type of 'y_scale' (line 307)
        y_scale_54689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 55), 'y_scale', False)
        # Processing the call keyword arguments (line 307)
        kwargs_54690 = {}
        # Getting the type of '_dense_num_jac' (line 307)
        _dense_num_jac_54682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), '_dense_num_jac', False)
        # Calling _dense_num_jac(args, kwargs) (line 307)
        _dense_num_jac_call_result_54691 = invoke(stypy.reporting.localization.Localization(__file__, 307, 15), _dense_num_jac_54682, *[fun_54683, t_54684, y_54685, f_54686, h_54687, factor_54688, y_scale_54689], **kwargs_54690)
        
        # Assigning a type to the variable 'stypy_return_type' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'stypy_return_type', _dense_num_jac_call_result_54691)

        if more_types_in_union_54681:
            # Runtime conditional SSA for else branch (line 306)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_54680) or more_types_in_union_54681):
        
        # Assigning a Name to a Tuple (line 309):
        
        # Assigning a Subscript to a Name (line 309):
        
        # Obtaining the type of the subscript
        int_54692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 8), 'int')
        # Getting the type of 'sparsity' (line 309)
        sparsity_54693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'sparsity')
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___54694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), sparsity_54693, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_54695 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), getitem___54694, int_54692)
        
        # Assigning a type to the variable 'tuple_var_assignment_54008' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'tuple_var_assignment_54008', subscript_call_result_54695)
        
        # Assigning a Subscript to a Name (line 309):
        
        # Obtaining the type of the subscript
        int_54696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 8), 'int')
        # Getting the type of 'sparsity' (line 309)
        sparsity_54697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'sparsity')
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___54698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), sparsity_54697, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_54699 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), getitem___54698, int_54696)
        
        # Assigning a type to the variable 'tuple_var_assignment_54009' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'tuple_var_assignment_54009', subscript_call_result_54699)
        
        # Assigning a Name to a Name (line 309):
        # Getting the type of 'tuple_var_assignment_54008' (line 309)
        tuple_var_assignment_54008_54700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'tuple_var_assignment_54008')
        # Assigning a type to the variable 'structure' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'structure', tuple_var_assignment_54008_54700)
        
        # Assigning a Name to a Name (line 309):
        # Getting the type of 'tuple_var_assignment_54009' (line 309)
        tuple_var_assignment_54009_54701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'tuple_var_assignment_54009')
        # Assigning a type to the variable 'groups' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'groups', tuple_var_assignment_54009_54701)
        
        # Call to _sparse_num_jac(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'fun' (line 310)
        fun_54703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 31), 'fun', False)
        # Getting the type of 't' (line 310)
        t_54704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 36), 't', False)
        # Getting the type of 'y' (line 310)
        y_54705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 39), 'y', False)
        # Getting the type of 'f' (line 310)
        f_54706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 42), 'f', False)
        # Getting the type of 'h' (line 310)
        h_54707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 45), 'h', False)
        # Getting the type of 'factor' (line 310)
        factor_54708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 48), 'factor', False)
        # Getting the type of 'y_scale' (line 310)
        y_scale_54709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 56), 'y_scale', False)
        # Getting the type of 'structure' (line 311)
        structure_54710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 31), 'structure', False)
        # Getting the type of 'groups' (line 311)
        groups_54711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 42), 'groups', False)
        # Processing the call keyword arguments (line 310)
        kwargs_54712 = {}
        # Getting the type of '_sparse_num_jac' (line 310)
        _sparse_num_jac_54702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), '_sparse_num_jac', False)
        # Calling _sparse_num_jac(args, kwargs) (line 310)
        _sparse_num_jac_call_result_54713 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), _sparse_num_jac_54702, *[fun_54703, t_54704, y_54705, f_54706, h_54707, factor_54708, y_scale_54709, structure_54710, groups_54711], **kwargs_54712)
        
        # Assigning a type to the variable 'stypy_return_type' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'stypy_return_type', _sparse_num_jac_call_result_54713)

        if (may_be_54680 and more_types_in_union_54681):
            # SSA join for if statement (line 306)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'num_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'num_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 239)
    stypy_return_type_54714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_54714)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'num_jac'
    return stypy_return_type_54714

# Assigning a type to the variable 'num_jac' (line 239)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'num_jac', num_jac)

@norecursion
def _dense_num_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_dense_num_jac'
    module_type_store = module_type_store.open_function_context('_dense_num_jac', 314, 0, False)
    
    # Passed parameters checking function
    _dense_num_jac.stypy_localization = localization
    _dense_num_jac.stypy_type_of_self = None
    _dense_num_jac.stypy_type_store = module_type_store
    _dense_num_jac.stypy_function_name = '_dense_num_jac'
    _dense_num_jac.stypy_param_names_list = ['fun', 't', 'y', 'f', 'h', 'factor', 'y_scale']
    _dense_num_jac.stypy_varargs_param_name = None
    _dense_num_jac.stypy_kwargs_param_name = None
    _dense_num_jac.stypy_call_defaults = defaults
    _dense_num_jac.stypy_call_varargs = varargs
    _dense_num_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_dense_num_jac', ['fun', 't', 'y', 'f', 'h', 'factor', 'y_scale'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_dense_num_jac', localization, ['fun', 't', 'y', 'f', 'h', 'factor', 'y_scale'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_dense_num_jac(...)' code ##################

    
    # Assigning a Subscript to a Name (line 315):
    
    # Assigning a Subscript to a Name (line 315):
    
    # Obtaining the type of the subscript
    int_54715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 16), 'int')
    # Getting the type of 'y' (line 315)
    y_54716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'y')
    # Obtaining the member 'shape' of a type (line 315)
    shape_54717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), y_54716, 'shape')
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___54718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), shape_54717, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 315)
    subscript_call_result_54719 = invoke(stypy.reporting.localization.Localization(__file__, 315, 8), getitem___54718, int_54715)
    
    # Assigning a type to the variable 'n' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'n', subscript_call_result_54719)
    
    # Assigning a Call to a Name (line 316):
    
    # Assigning a Call to a Name (line 316):
    
    # Call to diag(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'h' (line 316)
    h_54722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 21), 'h', False)
    # Processing the call keyword arguments (line 316)
    kwargs_54723 = {}
    # Getting the type of 'np' (line 316)
    np_54720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 13), 'np', False)
    # Obtaining the member 'diag' of a type (line 316)
    diag_54721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 13), np_54720, 'diag')
    # Calling diag(args, kwargs) (line 316)
    diag_call_result_54724 = invoke(stypy.reporting.localization.Localization(__file__, 316, 13), diag_54721, *[h_54722], **kwargs_54723)
    
    # Assigning a type to the variable 'h_vecs' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'h_vecs', diag_call_result_54724)
    
    # Assigning a Call to a Name (line 317):
    
    # Assigning a Call to a Name (line 317):
    
    # Call to fun(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 't' (line 317)
    t_54726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 16), 't', False)
    
    # Obtaining the type of the subscript
    slice_54727 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 317, 19), None, None, None)
    # Getting the type of 'None' (line 317)
    None_54728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 24), 'None', False)
    # Getting the type of 'y' (line 317)
    y_54729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 317)
    getitem___54730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 19), y_54729, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 317)
    subscript_call_result_54731 = invoke(stypy.reporting.localization.Localization(__file__, 317, 19), getitem___54730, (slice_54727, None_54728))
    
    # Getting the type of 'h_vecs' (line 317)
    h_vecs_54732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'h_vecs', False)
    # Applying the binary operator '+' (line 317)
    result_add_54733 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 19), '+', subscript_call_result_54731, h_vecs_54732)
    
    # Processing the call keyword arguments (line 317)
    kwargs_54734 = {}
    # Getting the type of 'fun' (line 317)
    fun_54725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'fun', False)
    # Calling fun(args, kwargs) (line 317)
    fun_call_result_54735 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), fun_54725, *[t_54726, result_add_54733], **kwargs_54734)
    
    # Assigning a type to the variable 'f_new' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'f_new', fun_call_result_54735)
    
    # Assigning a BinOp to a Name (line 318):
    
    # Assigning a BinOp to a Name (line 318):
    # Getting the type of 'f_new' (line 318)
    f_new_54736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'f_new')
    
    # Obtaining the type of the subscript
    slice_54737 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 318, 19), None, None, None)
    # Getting the type of 'None' (line 318)
    None_54738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 24), 'None')
    # Getting the type of 'f' (line 318)
    f_54739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 19), 'f')
    # Obtaining the member '__getitem__' of a type (line 318)
    getitem___54740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 19), f_54739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 318)
    subscript_call_result_54741 = invoke(stypy.reporting.localization.Localization(__file__, 318, 19), getitem___54740, (slice_54737, None_54738))
    
    # Applying the binary operator '-' (line 318)
    result_sub_54742 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 11), '-', f_new_54736, subscript_call_result_54741)
    
    # Assigning a type to the variable 'diff' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'diff', result_sub_54742)
    
    # Assigning a Call to a Name (line 319):
    
    # Assigning a Call to a Name (line 319):
    
    # Call to argmax(...): (line 319)
    # Processing the call arguments (line 319)
    
    # Call to abs(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'diff' (line 319)
    diff_54747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 31), 'diff', False)
    # Processing the call keyword arguments (line 319)
    kwargs_54748 = {}
    # Getting the type of 'np' (line 319)
    np_54745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 24), 'np', False)
    # Obtaining the member 'abs' of a type (line 319)
    abs_54746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 24), np_54745, 'abs')
    # Calling abs(args, kwargs) (line 319)
    abs_call_result_54749 = invoke(stypy.reporting.localization.Localization(__file__, 319, 24), abs_54746, *[diff_54747], **kwargs_54748)
    
    # Processing the call keyword arguments (line 319)
    int_54750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 43), 'int')
    keyword_54751 = int_54750
    kwargs_54752 = {'axis': keyword_54751}
    # Getting the type of 'np' (line 319)
    np_54743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 14), 'np', False)
    # Obtaining the member 'argmax' of a type (line 319)
    argmax_54744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 14), np_54743, 'argmax')
    # Calling argmax(args, kwargs) (line 319)
    argmax_call_result_54753 = invoke(stypy.reporting.localization.Localization(__file__, 319, 14), argmax_54744, *[abs_call_result_54749], **kwargs_54752)
    
    # Assigning a type to the variable 'max_ind' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'max_ind', argmax_call_result_54753)
    
    # Assigning a Call to a Name (line 320):
    
    # Assigning a Call to a Name (line 320):
    
    # Call to arange(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'n' (line 320)
    n_54756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 18), 'n', False)
    # Processing the call keyword arguments (line 320)
    kwargs_54757 = {}
    # Getting the type of 'np' (line 320)
    np_54754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 320)
    arange_54755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), np_54754, 'arange')
    # Calling arange(args, kwargs) (line 320)
    arange_call_result_54758 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), arange_54755, *[n_54756], **kwargs_54757)
    
    # Assigning a type to the variable 'r' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'r', arange_call_result_54758)
    
    # Assigning a Call to a Name (line 321):
    
    # Assigning a Call to a Name (line 321):
    
    # Call to abs(...): (line 321)
    # Processing the call arguments (line 321)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 321)
    tuple_54761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 321)
    # Adding element type (line 321)
    # Getting the type of 'max_ind' (line 321)
    max_ind_54762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 27), 'max_ind', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 27), tuple_54761, max_ind_54762)
    # Adding element type (line 321)
    # Getting the type of 'r' (line 321)
    r_54763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 36), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 27), tuple_54761, r_54763)
    
    # Getting the type of 'diff' (line 321)
    diff_54764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 22), 'diff', False)
    # Obtaining the member '__getitem__' of a type (line 321)
    getitem___54765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 22), diff_54764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 321)
    subscript_call_result_54766 = invoke(stypy.reporting.localization.Localization(__file__, 321, 22), getitem___54765, tuple_54761)
    
    # Processing the call keyword arguments (line 321)
    kwargs_54767 = {}
    # Getting the type of 'np' (line 321)
    np_54759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 15), 'np', False)
    # Obtaining the member 'abs' of a type (line 321)
    abs_54760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 15), np_54759, 'abs')
    # Calling abs(args, kwargs) (line 321)
    abs_call_result_54768 = invoke(stypy.reporting.localization.Localization(__file__, 321, 15), abs_54760, *[subscript_call_result_54766], **kwargs_54767)
    
    # Assigning a type to the variable 'max_diff' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'max_diff', abs_call_result_54768)
    
    # Assigning a Call to a Name (line 322):
    
    # Assigning a Call to a Name (line 322):
    
    # Call to maximum(...): (line 322)
    # Processing the call arguments (line 322)
    
    # Call to abs(...): (line 322)
    # Processing the call arguments (line 322)
    
    # Obtaining the type of the subscript
    # Getting the type of 'max_ind' (line 322)
    max_ind_54773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 32), 'max_ind', False)
    # Getting the type of 'f' (line 322)
    f_54774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 30), 'f', False)
    # Obtaining the member '__getitem__' of a type (line 322)
    getitem___54775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 30), f_54774, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
    subscript_call_result_54776 = invoke(stypy.reporting.localization.Localization(__file__, 322, 30), getitem___54775, max_ind_54773)
    
    # Processing the call keyword arguments (line 322)
    kwargs_54777 = {}
    # Getting the type of 'np' (line 322)
    np_54771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 23), 'np', False)
    # Obtaining the member 'abs' of a type (line 322)
    abs_54772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 23), np_54771, 'abs')
    # Calling abs(args, kwargs) (line 322)
    abs_call_result_54778 = invoke(stypy.reporting.localization.Localization(__file__, 322, 23), abs_54772, *[subscript_call_result_54776], **kwargs_54777)
    
    
    # Call to abs(...): (line 322)
    # Processing the call arguments (line 322)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 322)
    tuple_54781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 322)
    # Adding element type (line 322)
    # Getting the type of 'max_ind' (line 322)
    max_ind_54782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 56), 'max_ind', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 56), tuple_54781, max_ind_54782)
    # Adding element type (line 322)
    # Getting the type of 'r' (line 322)
    r_54783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 65), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 56), tuple_54781, r_54783)
    
    # Getting the type of 'f_new' (line 322)
    f_new_54784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 50), 'f_new', False)
    # Obtaining the member '__getitem__' of a type (line 322)
    getitem___54785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 50), f_new_54784, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
    subscript_call_result_54786 = invoke(stypy.reporting.localization.Localization(__file__, 322, 50), getitem___54785, tuple_54781)
    
    # Processing the call keyword arguments (line 322)
    kwargs_54787 = {}
    # Getting the type of 'np' (line 322)
    np_54779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 43), 'np', False)
    # Obtaining the member 'abs' of a type (line 322)
    abs_54780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 43), np_54779, 'abs')
    # Calling abs(args, kwargs) (line 322)
    abs_call_result_54788 = invoke(stypy.reporting.localization.Localization(__file__, 322, 43), abs_54780, *[subscript_call_result_54786], **kwargs_54787)
    
    # Processing the call keyword arguments (line 322)
    kwargs_54789 = {}
    # Getting the type of 'np' (line 322)
    np_54769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'np', False)
    # Obtaining the member 'maximum' of a type (line 322)
    maximum_54770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 12), np_54769, 'maximum')
    # Calling maximum(args, kwargs) (line 322)
    maximum_call_result_54790 = invoke(stypy.reporting.localization.Localization(__file__, 322, 12), maximum_54770, *[abs_call_result_54778, abs_call_result_54788], **kwargs_54789)
    
    # Assigning a type to the variable 'scale' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'scale', maximum_call_result_54790)
    
    # Assigning a Compare to a Name (line 324):
    
    # Assigning a Compare to a Name (line 324):
    
    # Getting the type of 'max_diff' (line 324)
    max_diff_54791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 21), 'max_diff')
    # Getting the type of 'NUM_JAC_DIFF_REJECT' (line 324)
    NUM_JAC_DIFF_REJECT_54792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 32), 'NUM_JAC_DIFF_REJECT')
    # Getting the type of 'scale' (line 324)
    scale_54793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 54), 'scale')
    # Applying the binary operator '*' (line 324)
    result_mul_54794 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 32), '*', NUM_JAC_DIFF_REJECT_54792, scale_54793)
    
    # Applying the binary operator '<' (line 324)
    result_lt_54795 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 21), '<', max_diff_54791, result_mul_54794)
    
    # Assigning a type to the variable 'diff_too_small' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'diff_too_small', result_lt_54795)
    
    
    # Call to any(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'diff_too_small' (line 325)
    diff_too_small_54798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 14), 'diff_too_small', False)
    # Processing the call keyword arguments (line 325)
    kwargs_54799 = {}
    # Getting the type of 'np' (line 325)
    np_54796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 325)
    any_54797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 7), np_54796, 'any')
    # Calling any(args, kwargs) (line 325)
    any_call_result_54800 = invoke(stypy.reporting.localization.Localization(__file__, 325, 7), any_54797, *[diff_too_small_54798], **kwargs_54799)
    
    # Testing the type of an if condition (line 325)
    if_condition_54801 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 4), any_call_result_54800)
    # Assigning a type to the variable 'if_condition_54801' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'if_condition_54801', if_condition_54801)
    # SSA begins for if statement (line 325)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 326):
    
    # Assigning a Subscript to a Name (line 326):
    
    # Obtaining the type of the subscript
    int_54802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 8), 'int')
    
    # Call to nonzero(...): (line 326)
    # Processing the call arguments (line 326)
    # Getting the type of 'diff_too_small' (line 326)
    diff_too_small_54805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), 'diff_too_small', False)
    # Processing the call keyword arguments (line 326)
    kwargs_54806 = {}
    # Getting the type of 'np' (line 326)
    np_54803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 15), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 326)
    nonzero_54804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 15), np_54803, 'nonzero')
    # Calling nonzero(args, kwargs) (line 326)
    nonzero_call_result_54807 = invoke(stypy.reporting.localization.Localization(__file__, 326, 15), nonzero_54804, *[diff_too_small_54805], **kwargs_54806)
    
    # Obtaining the member '__getitem__' of a type (line 326)
    getitem___54808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), nonzero_call_result_54807, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 326)
    subscript_call_result_54809 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), getitem___54808, int_54802)
    
    # Assigning a type to the variable 'tuple_var_assignment_54010' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'tuple_var_assignment_54010', subscript_call_result_54809)
    
    # Assigning a Name to a Name (line 326):
    # Getting the type of 'tuple_var_assignment_54010' (line 326)
    tuple_var_assignment_54010_54810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'tuple_var_assignment_54010')
    # Assigning a type to the variable 'ind' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'ind', tuple_var_assignment_54010_54810)
    
    # Assigning a BinOp to a Name (line 327):
    
    # Assigning a BinOp to a Name (line 327):
    # Getting the type of 'NUM_JAC_FACTOR_INCREASE' (line 327)
    NUM_JAC_FACTOR_INCREASE_54811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 21), 'NUM_JAC_FACTOR_INCREASE')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 327)
    ind_54812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 54), 'ind')
    # Getting the type of 'factor' (line 327)
    factor_54813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 47), 'factor')
    # Obtaining the member '__getitem__' of a type (line 327)
    getitem___54814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 47), factor_54813, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 327)
    subscript_call_result_54815 = invoke(stypy.reporting.localization.Localization(__file__, 327, 47), getitem___54814, ind_54812)
    
    # Applying the binary operator '*' (line 327)
    result_mul_54816 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 21), '*', NUM_JAC_FACTOR_INCREASE_54811, subscript_call_result_54815)
    
    # Assigning a type to the variable 'new_factor' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'new_factor', result_mul_54816)
    
    # Assigning a BinOp to a Name (line 328):
    
    # Assigning a BinOp to a Name (line 328):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 328)
    ind_54817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 19), 'ind')
    # Getting the type of 'y' (line 328)
    y_54818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 17), 'y')
    # Obtaining the member '__getitem__' of a type (line 328)
    getitem___54819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 17), y_54818, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 328)
    subscript_call_result_54820 = invoke(stypy.reporting.localization.Localization(__file__, 328, 17), getitem___54819, ind_54817)
    
    # Getting the type of 'new_factor' (line 328)
    new_factor_54821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 26), 'new_factor')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 328)
    ind_54822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 47), 'ind')
    # Getting the type of 'y_scale' (line 328)
    y_scale_54823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 39), 'y_scale')
    # Obtaining the member '__getitem__' of a type (line 328)
    getitem___54824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 39), y_scale_54823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 328)
    subscript_call_result_54825 = invoke(stypy.reporting.localization.Localization(__file__, 328, 39), getitem___54824, ind_54822)
    
    # Applying the binary operator '*' (line 328)
    result_mul_54826 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 26), '*', new_factor_54821, subscript_call_result_54825)
    
    # Applying the binary operator '+' (line 328)
    result_add_54827 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 17), '+', subscript_call_result_54820, result_mul_54826)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 328)
    ind_54828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 57), 'ind')
    # Getting the type of 'y' (line 328)
    y_54829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 55), 'y')
    # Obtaining the member '__getitem__' of a type (line 328)
    getitem___54830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 55), y_54829, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 328)
    subscript_call_result_54831 = invoke(stypy.reporting.localization.Localization(__file__, 328, 55), getitem___54830, ind_54828)
    
    # Applying the binary operator '-' (line 328)
    result_sub_54832 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 16), '-', result_add_54827, subscript_call_result_54831)
    
    # Assigning a type to the variable 'h_new' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'h_new', result_sub_54832)
    
    # Assigning a Name to a Subscript (line 329):
    
    # Assigning a Name to a Subscript (line 329):
    # Getting the type of 'h_new' (line 329)
    h_new_54833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 27), 'h_new')
    # Getting the type of 'h_vecs' (line 329)
    h_vecs_54834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'h_vecs')
    
    # Obtaining an instance of the builtin type 'tuple' (line 329)
    tuple_54835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 329)
    # Adding element type (line 329)
    # Getting the type of 'ind' (line 329)
    ind_54836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'ind')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 15), tuple_54835, ind_54836)
    # Adding element type (line 329)
    # Getting the type of 'ind' (line 329)
    ind_54837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'ind')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 15), tuple_54835, ind_54837)
    
    # Storing an element on a container (line 329)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 8), h_vecs_54834, (tuple_54835, h_new_54833))
    
    # Assigning a Call to a Name (line 330):
    
    # Assigning a Call to a Name (line 330):
    
    # Call to fun(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 't' (line 330)
    t_54839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 20), 't', False)
    
    # Obtaining the type of the subscript
    slice_54840 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 330, 23), None, None, None)
    # Getting the type of 'None' (line 330)
    None_54841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 28), 'None', False)
    # Getting the type of 'y' (line 330)
    y_54842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 23), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___54843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 23), y_54842, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_54844 = invoke(stypy.reporting.localization.Localization(__file__, 330, 23), getitem___54843, (slice_54840, None_54841))
    
    
    # Obtaining the type of the subscript
    slice_54845 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 330, 36), None, None, None)
    # Getting the type of 'ind' (line 330)
    ind_54846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 46), 'ind', False)
    # Getting the type of 'h_vecs' (line 330)
    h_vecs_54847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 36), 'h_vecs', False)
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___54848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 36), h_vecs_54847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_54849 = invoke(stypy.reporting.localization.Localization(__file__, 330, 36), getitem___54848, (slice_54845, ind_54846))
    
    # Applying the binary operator '+' (line 330)
    result_add_54850 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 23), '+', subscript_call_result_54844, subscript_call_result_54849)
    
    # Processing the call keyword arguments (line 330)
    kwargs_54851 = {}
    # Getting the type of 'fun' (line 330)
    fun_54838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'fun', False)
    # Calling fun(args, kwargs) (line 330)
    fun_call_result_54852 = invoke(stypy.reporting.localization.Localization(__file__, 330, 16), fun_54838, *[t_54839, result_add_54850], **kwargs_54851)
    
    # Assigning a type to the variable 'f_new' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'f_new', fun_call_result_54852)
    
    # Assigning a BinOp to a Name (line 331):
    
    # Assigning a BinOp to a Name (line 331):
    # Getting the type of 'f_new' (line 331)
    f_new_54853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 19), 'f_new')
    
    # Obtaining the type of the subscript
    slice_54854 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 331, 27), None, None, None)
    # Getting the type of 'None' (line 331)
    None_54855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 32), 'None')
    # Getting the type of 'f' (line 331)
    f_54856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 27), 'f')
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___54857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 27), f_54856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 331)
    subscript_call_result_54858 = invoke(stypy.reporting.localization.Localization(__file__, 331, 27), getitem___54857, (slice_54854, None_54855))
    
    # Applying the binary operator '-' (line 331)
    result_sub_54859 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 19), '-', f_new_54853, subscript_call_result_54858)
    
    # Assigning a type to the variable 'diff_new' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'diff_new', result_sub_54859)
    
    # Assigning a Call to a Name (line 332):
    
    # Assigning a Call to a Name (line 332):
    
    # Call to argmax(...): (line 332)
    # Processing the call arguments (line 332)
    
    # Call to abs(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'diff_new' (line 332)
    diff_new_54864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 35), 'diff_new', False)
    # Processing the call keyword arguments (line 332)
    kwargs_54865 = {}
    # Getting the type of 'np' (line 332)
    np_54862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 28), 'np', False)
    # Obtaining the member 'abs' of a type (line 332)
    abs_54863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 28), np_54862, 'abs')
    # Calling abs(args, kwargs) (line 332)
    abs_call_result_54866 = invoke(stypy.reporting.localization.Localization(__file__, 332, 28), abs_54863, *[diff_new_54864], **kwargs_54865)
    
    # Processing the call keyword arguments (line 332)
    int_54867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 51), 'int')
    keyword_54868 = int_54867
    kwargs_54869 = {'axis': keyword_54868}
    # Getting the type of 'np' (line 332)
    np_54860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'np', False)
    # Obtaining the member 'argmax' of a type (line 332)
    argmax_54861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 18), np_54860, 'argmax')
    # Calling argmax(args, kwargs) (line 332)
    argmax_call_result_54870 = invoke(stypy.reporting.localization.Localization(__file__, 332, 18), argmax_54861, *[abs_call_result_54866], **kwargs_54869)
    
    # Assigning a type to the variable 'max_ind' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'max_ind', argmax_call_result_54870)
    
    # Assigning a Call to a Name (line 333):
    
    # Assigning a Call to a Name (line 333):
    
    # Call to arange(...): (line 333)
    # Processing the call arguments (line 333)
    
    # Obtaining the type of the subscript
    int_54873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 32), 'int')
    # Getting the type of 'ind' (line 333)
    ind_54874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 22), 'ind', False)
    # Obtaining the member 'shape' of a type (line 333)
    shape_54875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 22), ind_54874, 'shape')
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___54876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 22), shape_54875, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_54877 = invoke(stypy.reporting.localization.Localization(__file__, 333, 22), getitem___54876, int_54873)
    
    # Processing the call keyword arguments (line 333)
    kwargs_54878 = {}
    # Getting the type of 'np' (line 333)
    np_54871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'np', False)
    # Obtaining the member 'arange' of a type (line 333)
    arange_54872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), np_54871, 'arange')
    # Calling arange(args, kwargs) (line 333)
    arange_call_result_54879 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), arange_54872, *[subscript_call_result_54877], **kwargs_54878)
    
    # Assigning a type to the variable 'r' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'r', arange_call_result_54879)
    
    # Assigning a Call to a Name (line 334):
    
    # Assigning a Call to a Name (line 334):
    
    # Call to abs(...): (line 334)
    # Processing the call arguments (line 334)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 334)
    tuple_54882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 334)
    # Adding element type (line 334)
    # Getting the type of 'max_ind' (line 334)
    max_ind_54883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 39), 'max_ind', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 39), tuple_54882, max_ind_54883)
    # Adding element type (line 334)
    # Getting the type of 'r' (line 334)
    r_54884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 48), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 39), tuple_54882, r_54884)
    
    # Getting the type of 'diff_new' (line 334)
    diff_new_54885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 30), 'diff_new', False)
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___54886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 30), diff_new_54885, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_54887 = invoke(stypy.reporting.localization.Localization(__file__, 334, 30), getitem___54886, tuple_54882)
    
    # Processing the call keyword arguments (line 334)
    kwargs_54888 = {}
    # Getting the type of 'np' (line 334)
    np_54880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 23), 'np', False)
    # Obtaining the member 'abs' of a type (line 334)
    abs_54881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 23), np_54880, 'abs')
    # Calling abs(args, kwargs) (line 334)
    abs_call_result_54889 = invoke(stypy.reporting.localization.Localization(__file__, 334, 23), abs_54881, *[subscript_call_result_54887], **kwargs_54888)
    
    # Assigning a type to the variable 'max_diff_new' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'max_diff_new', abs_call_result_54889)
    
    # Assigning a Call to a Name (line 335):
    
    # Assigning a Call to a Name (line 335):
    
    # Call to maximum(...): (line 335)
    # Processing the call arguments (line 335)
    
    # Call to abs(...): (line 335)
    # Processing the call arguments (line 335)
    
    # Obtaining the type of the subscript
    # Getting the type of 'max_ind' (line 335)
    max_ind_54894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 40), 'max_ind', False)
    # Getting the type of 'f' (line 335)
    f_54895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 38), 'f', False)
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___54896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 38), f_54895, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_54897 = invoke(stypy.reporting.localization.Localization(__file__, 335, 38), getitem___54896, max_ind_54894)
    
    # Processing the call keyword arguments (line 335)
    kwargs_54898 = {}
    # Getting the type of 'np' (line 335)
    np_54892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 31), 'np', False)
    # Obtaining the member 'abs' of a type (line 335)
    abs_54893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 31), np_54892, 'abs')
    # Calling abs(args, kwargs) (line 335)
    abs_call_result_54899 = invoke(stypy.reporting.localization.Localization(__file__, 335, 31), abs_54893, *[subscript_call_result_54897], **kwargs_54898)
    
    
    # Call to abs(...): (line 335)
    # Processing the call arguments (line 335)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 335)
    tuple_54902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 64), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 335)
    # Adding element type (line 335)
    # Getting the type of 'max_ind' (line 335)
    max_ind_54903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 64), 'max_ind', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 64), tuple_54902, max_ind_54903)
    # Adding element type (line 335)
    # Getting the type of 'r' (line 335)
    r_54904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 73), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 64), tuple_54902, r_54904)
    
    # Getting the type of 'f_new' (line 335)
    f_new_54905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 58), 'f_new', False)
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___54906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 58), f_new_54905, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_54907 = invoke(stypy.reporting.localization.Localization(__file__, 335, 58), getitem___54906, tuple_54902)
    
    # Processing the call keyword arguments (line 335)
    kwargs_54908 = {}
    # Getting the type of 'np' (line 335)
    np_54900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 51), 'np', False)
    # Obtaining the member 'abs' of a type (line 335)
    abs_54901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 51), np_54900, 'abs')
    # Calling abs(args, kwargs) (line 335)
    abs_call_result_54909 = invoke(stypy.reporting.localization.Localization(__file__, 335, 51), abs_54901, *[subscript_call_result_54907], **kwargs_54908)
    
    # Processing the call keyword arguments (line 335)
    kwargs_54910 = {}
    # Getting the type of 'np' (line 335)
    np_54890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'np', False)
    # Obtaining the member 'maximum' of a type (line 335)
    maximum_54891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 20), np_54890, 'maximum')
    # Calling maximum(args, kwargs) (line 335)
    maximum_call_result_54911 = invoke(stypy.reporting.localization.Localization(__file__, 335, 20), maximum_54891, *[abs_call_result_54899, abs_call_result_54909], **kwargs_54910)
    
    # Assigning a type to the variable 'scale_new' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'scale_new', maximum_call_result_54911)
    
    # Assigning a Compare to a Name (line 337):
    
    # Assigning a Compare to a Name (line 337):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 337)
    ind_54912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 26), 'ind')
    # Getting the type of 'max_diff' (line 337)
    max_diff_54913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 17), 'max_diff')
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___54914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 17), max_diff_54913, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_54915 = invoke(stypy.reporting.localization.Localization(__file__, 337, 17), getitem___54914, ind_54912)
    
    # Getting the type of 'scale_new' (line 337)
    scale_new_54916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 33), 'scale_new')
    # Applying the binary operator '*' (line 337)
    result_mul_54917 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 17), '*', subscript_call_result_54915, scale_new_54916)
    
    # Getting the type of 'max_diff_new' (line 337)
    max_diff_new_54918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 45), 'max_diff_new')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 337)
    ind_54919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 66), 'ind')
    # Getting the type of 'scale' (line 337)
    scale_54920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 60), 'scale')
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___54921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 60), scale_54920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_54922 = invoke(stypy.reporting.localization.Localization(__file__, 337, 60), getitem___54921, ind_54919)
    
    # Applying the binary operator '*' (line 337)
    result_mul_54923 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 45), '*', max_diff_new_54918, subscript_call_result_54922)
    
    # Applying the binary operator '<' (line 337)
    result_lt_54924 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 17), '<', result_mul_54917, result_mul_54923)
    
    # Assigning a type to the variable 'update' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'update', result_lt_54924)
    
    
    # Call to any(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'update' (line 338)
    update_54927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'update', False)
    # Processing the call keyword arguments (line 338)
    kwargs_54928 = {}
    # Getting the type of 'np' (line 338)
    np_54925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 11), 'np', False)
    # Obtaining the member 'any' of a type (line 338)
    any_54926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 11), np_54925, 'any')
    # Calling any(args, kwargs) (line 338)
    any_call_result_54929 = invoke(stypy.reporting.localization.Localization(__file__, 338, 11), any_54926, *[update_54927], **kwargs_54928)
    
    # Testing the type of an if condition (line 338)
    if_condition_54930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 8), any_call_result_54929)
    # Assigning a type to the variable 'if_condition_54930' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'if_condition_54930', if_condition_54930)
    # SSA begins for if statement (line 338)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 339):
    
    # Assigning a Subscript to a Name (line 339):
    
    # Obtaining the type of the subscript
    int_54931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 12), 'int')
    
    # Call to where(...): (line 339)
    # Processing the call arguments (line 339)
    # Getting the type of 'update' (line 339)
    update_54934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'update', False)
    # Processing the call keyword arguments (line 339)
    kwargs_54935 = {}
    # Getting the type of 'np' (line 339)
    np_54932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 22), 'np', False)
    # Obtaining the member 'where' of a type (line 339)
    where_54933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 22), np_54932, 'where')
    # Calling where(args, kwargs) (line 339)
    where_call_result_54936 = invoke(stypy.reporting.localization.Localization(__file__, 339, 22), where_54933, *[update_54934], **kwargs_54935)
    
    # Obtaining the member '__getitem__' of a type (line 339)
    getitem___54937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 12), where_call_result_54936, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 339)
    subscript_call_result_54938 = invoke(stypy.reporting.localization.Localization(__file__, 339, 12), getitem___54937, int_54931)
    
    # Assigning a type to the variable 'tuple_var_assignment_54011' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'tuple_var_assignment_54011', subscript_call_result_54938)
    
    # Assigning a Name to a Name (line 339):
    # Getting the type of 'tuple_var_assignment_54011' (line 339)
    tuple_var_assignment_54011_54939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'tuple_var_assignment_54011')
    # Assigning a type to the variable 'update' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'update', tuple_var_assignment_54011_54939)
    
    # Assigning a Subscript to a Name (line 340):
    
    # Assigning a Subscript to a Name (line 340):
    
    # Obtaining the type of the subscript
    # Getting the type of 'update' (line 340)
    update_54940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 29), 'update')
    # Getting the type of 'ind' (line 340)
    ind_54941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 25), 'ind')
    # Obtaining the member '__getitem__' of a type (line 340)
    getitem___54942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 25), ind_54941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 340)
    subscript_call_result_54943 = invoke(stypy.reporting.localization.Localization(__file__, 340, 25), getitem___54942, update_54940)
    
    # Assigning a type to the variable 'update_ind' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'update_ind', subscript_call_result_54943)
    
    # Assigning a Subscript to a Subscript (line 341):
    
    # Assigning a Subscript to a Subscript (line 341):
    
    # Obtaining the type of the subscript
    # Getting the type of 'update' (line 341)
    update_54944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 44), 'update')
    # Getting the type of 'new_factor' (line 341)
    new_factor_54945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 33), 'new_factor')
    # Obtaining the member '__getitem__' of a type (line 341)
    getitem___54946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 33), new_factor_54945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 341)
    subscript_call_result_54947 = invoke(stypy.reporting.localization.Localization(__file__, 341, 33), getitem___54946, update_54944)
    
    # Getting the type of 'factor' (line 341)
    factor_54948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'factor')
    # Getting the type of 'update_ind' (line 341)
    update_ind_54949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 19), 'update_ind')
    # Storing an element on a container (line 341)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 12), factor_54948, (update_ind_54949, subscript_call_result_54947))
    
    # Assigning a Subscript to a Subscript (line 342):
    
    # Assigning a Subscript to a Subscript (line 342):
    
    # Obtaining the type of the subscript
    # Getting the type of 'update' (line 342)
    update_54950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 34), 'update')
    # Getting the type of 'h_new' (line 342)
    h_new_54951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 28), 'h_new')
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___54952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 28), h_new_54951, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 342)
    subscript_call_result_54953 = invoke(stypy.reporting.localization.Localization(__file__, 342, 28), getitem___54952, update_54950)
    
    # Getting the type of 'h' (line 342)
    h_54954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'h')
    # Getting the type of 'update_ind' (line 342)
    update_ind_54955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 14), 'update_ind')
    # Storing an element on a container (line 342)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 12), h_54954, (update_ind_54955, subscript_call_result_54953))
    
    # Assigning a Subscript to a Subscript (line 343):
    
    # Assigning a Subscript to a Subscript (line 343):
    
    # Obtaining the type of the subscript
    slice_54956 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 343, 34), None, None, None)
    # Getting the type of 'update' (line 343)
    update_54957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 46), 'update')
    # Getting the type of 'diff_new' (line 343)
    diff_new_54958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 34), 'diff_new')
    # Obtaining the member '__getitem__' of a type (line 343)
    getitem___54959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 34), diff_new_54958, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 343)
    subscript_call_result_54960 = invoke(stypy.reporting.localization.Localization(__file__, 343, 34), getitem___54959, (slice_54956, update_54957))
    
    # Getting the type of 'diff' (line 343)
    diff_54961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'diff')
    slice_54962 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 343, 12), None, None, None)
    # Getting the type of 'update_ind' (line 343)
    update_ind_54963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 20), 'update_ind')
    # Storing an element on a container (line 343)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 12), diff_54961, ((slice_54962, update_ind_54963), subscript_call_result_54960))
    
    # Assigning a Subscript to a Subscript (line 344):
    
    # Assigning a Subscript to a Subscript (line 344):
    
    # Obtaining the type of the subscript
    # Getting the type of 'update' (line 344)
    update_54964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 42), 'update')
    # Getting the type of 'scale_new' (line 344)
    scale_new_54965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 32), 'scale_new')
    # Obtaining the member '__getitem__' of a type (line 344)
    getitem___54966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 32), scale_new_54965, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 344)
    subscript_call_result_54967 = invoke(stypy.reporting.localization.Localization(__file__, 344, 32), getitem___54966, update_54964)
    
    # Getting the type of 'scale' (line 344)
    scale_54968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'scale')
    # Getting the type of 'update_ind' (line 344)
    update_ind_54969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 18), 'update_ind')
    # Storing an element on a container (line 344)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 12), scale_54968, (update_ind_54969, subscript_call_result_54967))
    
    # Assigning a Subscript to a Subscript (line 345):
    
    # Assigning a Subscript to a Subscript (line 345):
    
    # Obtaining the type of the subscript
    # Getting the type of 'update' (line 345)
    update_54970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 48), 'update')
    # Getting the type of 'max_diff_new' (line 345)
    max_diff_new_54971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 35), 'max_diff_new')
    # Obtaining the member '__getitem__' of a type (line 345)
    getitem___54972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 35), max_diff_new_54971, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 345)
    subscript_call_result_54973 = invoke(stypy.reporting.localization.Localization(__file__, 345, 35), getitem___54972, update_54970)
    
    # Getting the type of 'max_diff' (line 345)
    max_diff_54974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'max_diff')
    # Getting the type of 'update_ind' (line 345)
    update_ind_54975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 21), 'update_ind')
    # Storing an element on a container (line 345)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 12), max_diff_54974, (update_ind_54975, subscript_call_result_54973))
    # SSA join for if statement (line 338)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 325)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'diff' (line 347)
    diff_54976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'diff')
    # Getting the type of 'h' (line 347)
    h_54977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'h')
    # Applying the binary operator 'div=' (line 347)
    result_div_54978 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 4), 'div=', diff_54976, h_54977)
    # Assigning a type to the variable 'diff' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'diff', result_div_54978)
    
    
    # Getting the type of 'factor' (line 349)
    factor_54979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'factor')
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'max_diff' (line 349)
    max_diff_54980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 11), 'max_diff')
    # Getting the type of 'NUM_JAC_DIFF_SMALL' (line 349)
    NUM_JAC_DIFF_SMALL_54981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 22), 'NUM_JAC_DIFF_SMALL')
    # Getting the type of 'scale' (line 349)
    scale_54982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 43), 'scale')
    # Applying the binary operator '*' (line 349)
    result_mul_54983 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 22), '*', NUM_JAC_DIFF_SMALL_54981, scale_54982)
    
    # Applying the binary operator '<' (line 349)
    result_lt_54984 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 11), '<', max_diff_54980, result_mul_54983)
    
    # Getting the type of 'factor' (line 349)
    factor_54985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'factor')
    # Obtaining the member '__getitem__' of a type (line 349)
    getitem___54986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 4), factor_54985, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 349)
    subscript_call_result_54987 = invoke(stypy.reporting.localization.Localization(__file__, 349, 4), getitem___54986, result_lt_54984)
    
    # Getting the type of 'NUM_JAC_FACTOR_INCREASE' (line 349)
    NUM_JAC_FACTOR_INCREASE_54988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 53), 'NUM_JAC_FACTOR_INCREASE')
    # Applying the binary operator '*=' (line 349)
    result_imul_54989 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 4), '*=', subscript_call_result_54987, NUM_JAC_FACTOR_INCREASE_54988)
    # Getting the type of 'factor' (line 349)
    factor_54990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'factor')
    
    # Getting the type of 'max_diff' (line 349)
    max_diff_54991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 11), 'max_diff')
    # Getting the type of 'NUM_JAC_DIFF_SMALL' (line 349)
    NUM_JAC_DIFF_SMALL_54992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 22), 'NUM_JAC_DIFF_SMALL')
    # Getting the type of 'scale' (line 349)
    scale_54993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 43), 'scale')
    # Applying the binary operator '*' (line 349)
    result_mul_54994 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 22), '*', NUM_JAC_DIFF_SMALL_54992, scale_54993)
    
    # Applying the binary operator '<' (line 349)
    result_lt_54995 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 11), '<', max_diff_54991, result_mul_54994)
    
    # Storing an element on a container (line 349)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 4), factor_54990, (result_lt_54995, result_imul_54989))
    
    
    # Getting the type of 'factor' (line 350)
    factor_54996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'factor')
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'max_diff' (line 350)
    max_diff_54997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 11), 'max_diff')
    # Getting the type of 'NUM_JAC_DIFF_BIG' (line 350)
    NUM_JAC_DIFF_BIG_54998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 22), 'NUM_JAC_DIFF_BIG')
    # Getting the type of 'scale' (line 350)
    scale_54999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 41), 'scale')
    # Applying the binary operator '*' (line 350)
    result_mul_55000 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 22), '*', NUM_JAC_DIFF_BIG_54998, scale_54999)
    
    # Applying the binary operator '>' (line 350)
    result_gt_55001 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 11), '>', max_diff_54997, result_mul_55000)
    
    # Getting the type of 'factor' (line 350)
    factor_55002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'factor')
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___55003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 4), factor_55002, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_55004 = invoke(stypy.reporting.localization.Localization(__file__, 350, 4), getitem___55003, result_gt_55001)
    
    # Getting the type of 'NUM_JAC_FACTOR_DECREASE' (line 350)
    NUM_JAC_FACTOR_DECREASE_55005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 51), 'NUM_JAC_FACTOR_DECREASE')
    # Applying the binary operator '*=' (line 350)
    result_imul_55006 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 4), '*=', subscript_call_result_55004, NUM_JAC_FACTOR_DECREASE_55005)
    # Getting the type of 'factor' (line 350)
    factor_55007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'factor')
    
    # Getting the type of 'max_diff' (line 350)
    max_diff_55008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 11), 'max_diff')
    # Getting the type of 'NUM_JAC_DIFF_BIG' (line 350)
    NUM_JAC_DIFF_BIG_55009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 22), 'NUM_JAC_DIFF_BIG')
    # Getting the type of 'scale' (line 350)
    scale_55010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 41), 'scale')
    # Applying the binary operator '*' (line 350)
    result_mul_55011 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 22), '*', NUM_JAC_DIFF_BIG_55009, scale_55010)
    
    # Applying the binary operator '>' (line 350)
    result_gt_55012 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 11), '>', max_diff_55008, result_mul_55011)
    
    # Storing an element on a container (line 350)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 4), factor_55007, (result_gt_55012, result_imul_55006))
    
    
    # Assigning a Call to a Name (line 351):
    
    # Assigning a Call to a Name (line 351):
    
    # Call to maximum(...): (line 351)
    # Processing the call arguments (line 351)
    # Getting the type of 'factor' (line 351)
    factor_55015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'factor', False)
    # Getting the type of 'NUM_JAC_MIN_FACTOR' (line 351)
    NUM_JAC_MIN_FACTOR_55016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 32), 'NUM_JAC_MIN_FACTOR', False)
    # Processing the call keyword arguments (line 351)
    kwargs_55017 = {}
    # Getting the type of 'np' (line 351)
    np_55013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 13), 'np', False)
    # Obtaining the member 'maximum' of a type (line 351)
    maximum_55014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 13), np_55013, 'maximum')
    # Calling maximum(args, kwargs) (line 351)
    maximum_call_result_55018 = invoke(stypy.reporting.localization.Localization(__file__, 351, 13), maximum_55014, *[factor_55015, NUM_JAC_MIN_FACTOR_55016], **kwargs_55017)
    
    # Assigning a type to the variable 'factor' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'factor', maximum_call_result_55018)
    
    # Obtaining an instance of the builtin type 'tuple' (line 353)
    tuple_55019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 353)
    # Adding element type (line 353)
    # Getting the type of 'diff' (line 353)
    diff_55020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 11), 'diff')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 11), tuple_55019, diff_55020)
    # Adding element type (line 353)
    # Getting the type of 'factor' (line 353)
    factor_55021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 17), 'factor')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 11), tuple_55019, factor_55021)
    
    # Assigning a type to the variable 'stypy_return_type' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'stypy_return_type', tuple_55019)
    
    # ################# End of '_dense_num_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_dense_num_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 314)
    stypy_return_type_55022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55022)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_dense_num_jac'
    return stypy_return_type_55022

# Assigning a type to the variable '_dense_num_jac' (line 314)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 0), '_dense_num_jac', _dense_num_jac)

@norecursion
def _sparse_num_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_sparse_num_jac'
    module_type_store = module_type_store.open_function_context('_sparse_num_jac', 356, 0, False)
    
    # Passed parameters checking function
    _sparse_num_jac.stypy_localization = localization
    _sparse_num_jac.stypy_type_of_self = None
    _sparse_num_jac.stypy_type_store = module_type_store
    _sparse_num_jac.stypy_function_name = '_sparse_num_jac'
    _sparse_num_jac.stypy_param_names_list = ['fun', 't', 'y', 'f', 'h', 'factor', 'y_scale', 'structure', 'groups']
    _sparse_num_jac.stypy_varargs_param_name = None
    _sparse_num_jac.stypy_kwargs_param_name = None
    _sparse_num_jac.stypy_call_defaults = defaults
    _sparse_num_jac.stypy_call_varargs = varargs
    _sparse_num_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_sparse_num_jac', ['fun', 't', 'y', 'f', 'h', 'factor', 'y_scale', 'structure', 'groups'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_sparse_num_jac', localization, ['fun', 't', 'y', 'f', 'h', 'factor', 'y_scale', 'structure', 'groups'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_sparse_num_jac(...)' code ##################

    
    # Assigning a Subscript to a Name (line 357):
    
    # Assigning a Subscript to a Name (line 357):
    
    # Obtaining the type of the subscript
    int_55023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 16), 'int')
    # Getting the type of 'y' (line 357)
    y_55024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'y')
    # Obtaining the member 'shape' of a type (line 357)
    shape_55025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), y_55024, 'shape')
    # Obtaining the member '__getitem__' of a type (line 357)
    getitem___55026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), shape_55025, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 357)
    subscript_call_result_55027 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), getitem___55026, int_55023)
    
    # Assigning a type to the variable 'n' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'n', subscript_call_result_55027)
    
    # Assigning a BinOp to a Name (line 358):
    
    # Assigning a BinOp to a Name (line 358):
    
    # Call to max(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'groups' (line 358)
    groups_55030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 22), 'groups', False)
    # Processing the call keyword arguments (line 358)
    kwargs_55031 = {}
    # Getting the type of 'np' (line 358)
    np_55028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 15), 'np', False)
    # Obtaining the member 'max' of a type (line 358)
    max_55029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 15), np_55028, 'max')
    # Calling max(args, kwargs) (line 358)
    max_call_result_55032 = invoke(stypy.reporting.localization.Localization(__file__, 358, 15), max_55029, *[groups_55030], **kwargs_55031)
    
    int_55033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 32), 'int')
    # Applying the binary operator '+' (line 358)
    result_add_55034 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 15), '+', max_call_result_55032, int_55033)
    
    # Assigning a type to the variable 'n_groups' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'n_groups', result_add_55034)
    
    # Assigning a Call to a Name (line 359):
    
    # Assigning a Call to a Name (line 359):
    
    # Call to empty(...): (line 359)
    # Processing the call arguments (line 359)
    
    # Obtaining an instance of the builtin type 'tuple' (line 359)
    tuple_55037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 359)
    # Adding element type (line 359)
    # Getting the type of 'n_groups' (line 359)
    n_groups_55038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 23), 'n_groups', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 23), tuple_55037, n_groups_55038)
    # Adding element type (line 359)
    # Getting the type of 'n' (line 359)
    n_55039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 33), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 23), tuple_55037, n_55039)
    
    # Processing the call keyword arguments (line 359)
    kwargs_55040 = {}
    # Getting the type of 'np' (line 359)
    np_55035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'np', False)
    # Obtaining the member 'empty' of a type (line 359)
    empty_55036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), np_55035, 'empty')
    # Calling empty(args, kwargs) (line 359)
    empty_call_result_55041 = invoke(stypy.reporting.localization.Localization(__file__, 359, 13), empty_55036, *[tuple_55037], **kwargs_55040)
    
    # Assigning a type to the variable 'h_vecs' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'h_vecs', empty_call_result_55041)
    
    
    # Call to range(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'n_groups' (line 360)
    n_groups_55043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'n_groups', False)
    # Processing the call keyword arguments (line 360)
    kwargs_55044 = {}
    # Getting the type of 'range' (line 360)
    range_55042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 17), 'range', False)
    # Calling range(args, kwargs) (line 360)
    range_call_result_55045 = invoke(stypy.reporting.localization.Localization(__file__, 360, 17), range_55042, *[n_groups_55043], **kwargs_55044)
    
    # Testing the type of a for loop iterable (line 360)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 360, 4), range_call_result_55045)
    # Getting the type of the for loop variable (line 360)
    for_loop_var_55046 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 360, 4), range_call_result_55045)
    # Assigning a type to the variable 'group' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'group', for_loop_var_55046)
    # SSA begins for a for statement (line 360)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 361):
    
    # Assigning a Call to a Name (line 361):
    
    # Call to equal(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'group' (line 361)
    group_55049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 21), 'group', False)
    # Getting the type of 'groups' (line 361)
    groups_55050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 28), 'groups', False)
    # Processing the call keyword arguments (line 361)
    kwargs_55051 = {}
    # Getting the type of 'np' (line 361)
    np_55047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'np', False)
    # Obtaining the member 'equal' of a type (line 361)
    equal_55048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 12), np_55047, 'equal')
    # Calling equal(args, kwargs) (line 361)
    equal_call_result_55052 = invoke(stypy.reporting.localization.Localization(__file__, 361, 12), equal_55048, *[group_55049, groups_55050], **kwargs_55051)
    
    # Assigning a type to the variable 'e' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'e', equal_call_result_55052)
    
    # Assigning a BinOp to a Subscript (line 362):
    
    # Assigning a BinOp to a Subscript (line 362):
    # Getting the type of 'h' (line 362)
    h_55053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 24), 'h')
    # Getting the type of 'e' (line 362)
    e_55054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 28), 'e')
    # Applying the binary operator '*' (line 362)
    result_mul_55055 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 24), '*', h_55053, e_55054)
    
    # Getting the type of 'h_vecs' (line 362)
    h_vecs_55056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'h_vecs')
    # Getting the type of 'group' (line 362)
    group_55057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'group')
    # Storing an element on a container (line 362)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 8), h_vecs_55056, (group_55057, result_mul_55055))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 363):
    
    # Assigning a Attribute to a Name (line 363):
    # Getting the type of 'h_vecs' (line 363)
    h_vecs_55058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 13), 'h_vecs')
    # Obtaining the member 'T' of a type (line 363)
    T_55059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 13), h_vecs_55058, 'T')
    # Assigning a type to the variable 'h_vecs' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'h_vecs', T_55059)
    
    # Assigning a Call to a Name (line 365):
    
    # Assigning a Call to a Name (line 365):
    
    # Call to fun(...): (line 365)
    # Processing the call arguments (line 365)
    # Getting the type of 't' (line 365)
    t_55061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 't', False)
    
    # Obtaining the type of the subscript
    slice_55062 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 365, 19), None, None, None)
    # Getting the type of 'None' (line 365)
    None_55063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 24), 'None', False)
    # Getting the type of 'y' (line 365)
    y_55064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 19), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 365)
    getitem___55065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 19), y_55064, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 365)
    subscript_call_result_55066 = invoke(stypy.reporting.localization.Localization(__file__, 365, 19), getitem___55065, (slice_55062, None_55063))
    
    # Getting the type of 'h_vecs' (line 365)
    h_vecs_55067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 32), 'h_vecs', False)
    # Applying the binary operator '+' (line 365)
    result_add_55068 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 19), '+', subscript_call_result_55066, h_vecs_55067)
    
    # Processing the call keyword arguments (line 365)
    kwargs_55069 = {}
    # Getting the type of 'fun' (line 365)
    fun_55060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'fun', False)
    # Calling fun(args, kwargs) (line 365)
    fun_call_result_55070 = invoke(stypy.reporting.localization.Localization(__file__, 365, 12), fun_55060, *[t_55061, result_add_55068], **kwargs_55069)
    
    # Assigning a type to the variable 'f_new' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'f_new', fun_call_result_55070)
    
    # Assigning a BinOp to a Name (line 366):
    
    # Assigning a BinOp to a Name (line 366):
    # Getting the type of 'f_new' (line 366)
    f_new_55071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 9), 'f_new')
    
    # Obtaining the type of the subscript
    slice_55072 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 366, 17), None, None, None)
    # Getting the type of 'None' (line 366)
    None_55073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'None')
    # Getting the type of 'f' (line 366)
    f_55074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 17), 'f')
    # Obtaining the member '__getitem__' of a type (line 366)
    getitem___55075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 17), f_55074, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 366)
    subscript_call_result_55076 = invoke(stypy.reporting.localization.Localization(__file__, 366, 17), getitem___55075, (slice_55072, None_55073))
    
    # Applying the binary operator '-' (line 366)
    result_sub_55077 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 9), '-', f_new_55071, subscript_call_result_55076)
    
    # Assigning a type to the variable 'df' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'df', result_sub_55077)
    
    # Assigning a Call to a Tuple (line 368):
    
    # Assigning a Subscript to a Name (line 368):
    
    # Obtaining the type of the subscript
    int_55078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 4), 'int')
    
    # Call to find(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'structure' (line 368)
    structure_55080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 19), 'structure', False)
    # Processing the call keyword arguments (line 368)
    kwargs_55081 = {}
    # Getting the type of 'find' (line 368)
    find_55079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 14), 'find', False)
    # Calling find(args, kwargs) (line 368)
    find_call_result_55082 = invoke(stypy.reporting.localization.Localization(__file__, 368, 14), find_55079, *[structure_55080], **kwargs_55081)
    
    # Obtaining the member '__getitem__' of a type (line 368)
    getitem___55083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 4), find_call_result_55082, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 368)
    subscript_call_result_55084 = invoke(stypy.reporting.localization.Localization(__file__, 368, 4), getitem___55083, int_55078)
    
    # Assigning a type to the variable 'tuple_var_assignment_54012' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'tuple_var_assignment_54012', subscript_call_result_55084)
    
    # Assigning a Subscript to a Name (line 368):
    
    # Obtaining the type of the subscript
    int_55085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 4), 'int')
    
    # Call to find(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'structure' (line 368)
    structure_55087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 19), 'structure', False)
    # Processing the call keyword arguments (line 368)
    kwargs_55088 = {}
    # Getting the type of 'find' (line 368)
    find_55086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 14), 'find', False)
    # Calling find(args, kwargs) (line 368)
    find_call_result_55089 = invoke(stypy.reporting.localization.Localization(__file__, 368, 14), find_55086, *[structure_55087], **kwargs_55088)
    
    # Obtaining the member '__getitem__' of a type (line 368)
    getitem___55090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 4), find_call_result_55089, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 368)
    subscript_call_result_55091 = invoke(stypy.reporting.localization.Localization(__file__, 368, 4), getitem___55090, int_55085)
    
    # Assigning a type to the variable 'tuple_var_assignment_54013' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'tuple_var_assignment_54013', subscript_call_result_55091)
    
    # Assigning a Subscript to a Name (line 368):
    
    # Obtaining the type of the subscript
    int_55092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 4), 'int')
    
    # Call to find(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'structure' (line 368)
    structure_55094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 19), 'structure', False)
    # Processing the call keyword arguments (line 368)
    kwargs_55095 = {}
    # Getting the type of 'find' (line 368)
    find_55093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 14), 'find', False)
    # Calling find(args, kwargs) (line 368)
    find_call_result_55096 = invoke(stypy.reporting.localization.Localization(__file__, 368, 14), find_55093, *[structure_55094], **kwargs_55095)
    
    # Obtaining the member '__getitem__' of a type (line 368)
    getitem___55097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 4), find_call_result_55096, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 368)
    subscript_call_result_55098 = invoke(stypy.reporting.localization.Localization(__file__, 368, 4), getitem___55097, int_55092)
    
    # Assigning a type to the variable 'tuple_var_assignment_54014' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'tuple_var_assignment_54014', subscript_call_result_55098)
    
    # Assigning a Name to a Name (line 368):
    # Getting the type of 'tuple_var_assignment_54012' (line 368)
    tuple_var_assignment_54012_55099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'tuple_var_assignment_54012')
    # Assigning a type to the variable 'i' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'i', tuple_var_assignment_54012_55099)
    
    # Assigning a Name to a Name (line 368):
    # Getting the type of 'tuple_var_assignment_54013' (line 368)
    tuple_var_assignment_54013_55100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'tuple_var_assignment_54013')
    # Assigning a type to the variable 'j' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 7), 'j', tuple_var_assignment_54013_55100)
    
    # Assigning a Name to a Name (line 368):
    # Getting the type of 'tuple_var_assignment_54014' (line 368)
    tuple_var_assignment_54014_55101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'tuple_var_assignment_54014')
    # Assigning a type to the variable '_' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 10), '_', tuple_var_assignment_54014_55101)
    
    # Assigning a Call to a Name (line 369):
    
    # Assigning a Call to a Name (line 369):
    
    # Call to tocsc(...): (line 369)
    # Processing the call keyword arguments (line 369)
    kwargs_55123 = {}
    
    # Call to coo_matrix(...): (line 369)
    # Processing the call arguments (line 369)
    
    # Obtaining an instance of the builtin type 'tuple' (line 369)
    tuple_55103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 369)
    # Adding element type (line 369)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 369)
    tuple_55104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 369)
    # Adding element type (line 369)
    # Getting the type of 'i' (line 369)
    i_55105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 26), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 26), tuple_55104, i_55105)
    # Adding element type (line 369)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 369)
    j_55106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 36), 'j', False)
    # Getting the type of 'groups' (line 369)
    groups_55107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 29), 'groups', False)
    # Obtaining the member '__getitem__' of a type (line 369)
    getitem___55108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 29), groups_55107, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 369)
    subscript_call_result_55109 = invoke(stypy.reporting.localization.Localization(__file__, 369, 29), getitem___55108, j_55106)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 26), tuple_55104, subscript_call_result_55109)
    
    # Getting the type of 'df' (line 369)
    df_55110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 23), 'df', False)
    # Obtaining the member '__getitem__' of a type (line 369)
    getitem___55111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 23), df_55110, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 369)
    subscript_call_result_55112 = invoke(stypy.reporting.localization.Localization(__file__, 369, 23), getitem___55111, tuple_55104)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 23), tuple_55103, subscript_call_result_55112)
    # Adding element type (line 369)
    
    # Obtaining an instance of the builtin type 'tuple' (line 369)
    tuple_55113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 369)
    # Adding element type (line 369)
    # Getting the type of 'i' (line 369)
    i_55114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 42), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 42), tuple_55113, i_55114)
    # Adding element type (line 369)
    # Getting the type of 'j' (line 369)
    j_55115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 45), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 42), tuple_55113, j_55115)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 23), tuple_55103, tuple_55113)
    
    # Processing the call keyword arguments (line 369)
    
    # Obtaining an instance of the builtin type 'tuple' (line 369)
    tuple_55116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 369)
    # Adding element type (line 369)
    # Getting the type of 'n' (line 369)
    n_55117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 57), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 57), tuple_55116, n_55117)
    # Adding element type (line 369)
    # Getting the type of 'n' (line 369)
    n_55118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 60), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 57), tuple_55116, n_55118)
    
    keyword_55119 = tuple_55116
    kwargs_55120 = {'shape': keyword_55119}
    # Getting the type of 'coo_matrix' (line 369)
    coo_matrix_55102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 11), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 369)
    coo_matrix_call_result_55121 = invoke(stypy.reporting.localization.Localization(__file__, 369, 11), coo_matrix_55102, *[tuple_55103], **kwargs_55120)
    
    # Obtaining the member 'tocsc' of a type (line 369)
    tocsc_55122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 11), coo_matrix_call_result_55121, 'tocsc')
    # Calling tocsc(args, kwargs) (line 369)
    tocsc_call_result_55124 = invoke(stypy.reporting.localization.Localization(__file__, 369, 11), tocsc_55122, *[], **kwargs_55123)
    
    # Assigning a type to the variable 'diff' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'diff', tocsc_call_result_55124)
    
    # Assigning a Call to a Name (line 370):
    
    # Assigning a Call to a Name (line 370):
    
    # Call to ravel(...): (line 370)
    # Processing the call keyword arguments (line 370)
    kwargs_55139 = {}
    
    # Call to array(...): (line 370)
    # Processing the call arguments (line 370)
    
    # Call to argmax(...): (line 370)
    # Processing the call keyword arguments (line 370)
    int_55132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 45), 'int')
    keyword_55133 = int_55132
    kwargs_55134 = {'axis': keyword_55133}
    
    # Call to abs(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'diff' (line 370)
    diff_55128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 27), 'diff', False)
    # Processing the call keyword arguments (line 370)
    kwargs_55129 = {}
    # Getting the type of 'abs' (line 370)
    abs_55127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 23), 'abs', False)
    # Calling abs(args, kwargs) (line 370)
    abs_call_result_55130 = invoke(stypy.reporting.localization.Localization(__file__, 370, 23), abs_55127, *[diff_55128], **kwargs_55129)
    
    # Obtaining the member 'argmax' of a type (line 370)
    argmax_55131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 23), abs_call_result_55130, 'argmax')
    # Calling argmax(args, kwargs) (line 370)
    argmax_call_result_55135 = invoke(stypy.reporting.localization.Localization(__file__, 370, 23), argmax_55131, *[], **kwargs_55134)
    
    # Processing the call keyword arguments (line 370)
    kwargs_55136 = {}
    # Getting the type of 'np' (line 370)
    np_55125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 370)
    array_55126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 14), np_55125, 'array')
    # Calling array(args, kwargs) (line 370)
    array_call_result_55137 = invoke(stypy.reporting.localization.Localization(__file__, 370, 14), array_55126, *[argmax_call_result_55135], **kwargs_55136)
    
    # Obtaining the member 'ravel' of a type (line 370)
    ravel_55138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 14), array_call_result_55137, 'ravel')
    # Calling ravel(args, kwargs) (line 370)
    ravel_call_result_55140 = invoke(stypy.reporting.localization.Localization(__file__, 370, 14), ravel_55138, *[], **kwargs_55139)
    
    # Assigning a type to the variable 'max_ind' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'max_ind', ravel_call_result_55140)
    
    # Assigning a Call to a Name (line 371):
    
    # Assigning a Call to a Name (line 371):
    
    # Call to arange(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'n' (line 371)
    n_55143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 18), 'n', False)
    # Processing the call keyword arguments (line 371)
    kwargs_55144 = {}
    # Getting the type of 'np' (line 371)
    np_55141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 371)
    arange_55142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), np_55141, 'arange')
    # Calling arange(args, kwargs) (line 371)
    arange_call_result_55145 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), arange_55142, *[n_55143], **kwargs_55144)
    
    # Assigning a type to the variable 'r' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'r', arange_call_result_55145)
    
    # Assigning a Call to a Name (line 372):
    
    # Assigning a Call to a Name (line 372):
    
    # Call to ravel(...): (line 372)
    # Processing the call keyword arguments (line 372)
    kwargs_55161 = {}
    
    # Call to asarray(...): (line 372)
    # Processing the call arguments (line 372)
    
    # Call to abs(...): (line 372)
    # Processing the call arguments (line 372)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 372)
    tuple_55150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 372)
    # Adding element type (line 372)
    # Getting the type of 'max_ind' (line 372)
    max_ind_55151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 38), 'max_ind', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 38), tuple_55150, max_ind_55151)
    # Adding element type (line 372)
    # Getting the type of 'r' (line 372)
    r_55152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 47), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 38), tuple_55150, r_55152)
    
    # Getting the type of 'diff' (line 372)
    diff_55153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 33), 'diff', False)
    # Obtaining the member '__getitem__' of a type (line 372)
    getitem___55154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 33), diff_55153, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 372)
    subscript_call_result_55155 = invoke(stypy.reporting.localization.Localization(__file__, 372, 33), getitem___55154, tuple_55150)
    
    # Processing the call keyword arguments (line 372)
    kwargs_55156 = {}
    # Getting the type of 'np' (line 372)
    np_55148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 26), 'np', False)
    # Obtaining the member 'abs' of a type (line 372)
    abs_55149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 26), np_55148, 'abs')
    # Calling abs(args, kwargs) (line 372)
    abs_call_result_55157 = invoke(stypy.reporting.localization.Localization(__file__, 372, 26), abs_55149, *[subscript_call_result_55155], **kwargs_55156)
    
    # Processing the call keyword arguments (line 372)
    kwargs_55158 = {}
    # Getting the type of 'np' (line 372)
    np_55146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 15), 'np', False)
    # Obtaining the member 'asarray' of a type (line 372)
    asarray_55147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 15), np_55146, 'asarray')
    # Calling asarray(args, kwargs) (line 372)
    asarray_call_result_55159 = invoke(stypy.reporting.localization.Localization(__file__, 372, 15), asarray_55147, *[abs_call_result_55157], **kwargs_55158)
    
    # Obtaining the member 'ravel' of a type (line 372)
    ravel_55160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 15), asarray_call_result_55159, 'ravel')
    # Calling ravel(args, kwargs) (line 372)
    ravel_call_result_55162 = invoke(stypy.reporting.localization.Localization(__file__, 372, 15), ravel_55160, *[], **kwargs_55161)
    
    # Assigning a type to the variable 'max_diff' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'max_diff', ravel_call_result_55162)
    
    # Assigning a Call to a Name (line 373):
    
    # Assigning a Call to a Name (line 373):
    
    # Call to maximum(...): (line 373)
    # Processing the call arguments (line 373)
    
    # Call to abs(...): (line 373)
    # Processing the call arguments (line 373)
    
    # Obtaining the type of the subscript
    # Getting the type of 'max_ind' (line 373)
    max_ind_55167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 32), 'max_ind', False)
    # Getting the type of 'f' (line 373)
    f_55168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 30), 'f', False)
    # Obtaining the member '__getitem__' of a type (line 373)
    getitem___55169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 30), f_55168, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 373)
    subscript_call_result_55170 = invoke(stypy.reporting.localization.Localization(__file__, 373, 30), getitem___55169, max_ind_55167)
    
    # Processing the call keyword arguments (line 373)
    kwargs_55171 = {}
    # Getting the type of 'np' (line 373)
    np_55165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 23), 'np', False)
    # Obtaining the member 'abs' of a type (line 373)
    abs_55166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 23), np_55165, 'abs')
    # Calling abs(args, kwargs) (line 373)
    abs_call_result_55172 = invoke(stypy.reporting.localization.Localization(__file__, 373, 23), abs_55166, *[subscript_call_result_55170], **kwargs_55171)
    
    
    # Call to abs(...): (line 374)
    # Processing the call arguments (line 374)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 374)
    tuple_55175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 374)
    # Adding element type (line 374)
    # Getting the type of 'max_ind' (line 374)
    max_ind_55176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 36), 'max_ind', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 36), tuple_55175, max_ind_55176)
    # Adding element type (line 374)
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 374)
    r_55177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 52), 'r', False)
    # Getting the type of 'groups' (line 374)
    groups_55178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 45), 'groups', False)
    # Obtaining the member '__getitem__' of a type (line 374)
    getitem___55179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 45), groups_55178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 374)
    subscript_call_result_55180 = invoke(stypy.reporting.localization.Localization(__file__, 374, 45), getitem___55179, r_55177)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 36), tuple_55175, subscript_call_result_55180)
    
    # Getting the type of 'f_new' (line 374)
    f_new_55181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 30), 'f_new', False)
    # Obtaining the member '__getitem__' of a type (line 374)
    getitem___55182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 30), f_new_55181, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 374)
    subscript_call_result_55183 = invoke(stypy.reporting.localization.Localization(__file__, 374, 30), getitem___55182, tuple_55175)
    
    # Processing the call keyword arguments (line 374)
    kwargs_55184 = {}
    # Getting the type of 'np' (line 374)
    np_55173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'np', False)
    # Obtaining the member 'abs' of a type (line 374)
    abs_55174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), np_55173, 'abs')
    # Calling abs(args, kwargs) (line 374)
    abs_call_result_55185 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), abs_55174, *[subscript_call_result_55183], **kwargs_55184)
    
    # Processing the call keyword arguments (line 373)
    kwargs_55186 = {}
    # Getting the type of 'np' (line 373)
    np_55163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'np', False)
    # Obtaining the member 'maximum' of a type (line 373)
    maximum_55164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 12), np_55163, 'maximum')
    # Calling maximum(args, kwargs) (line 373)
    maximum_call_result_55187 = invoke(stypy.reporting.localization.Localization(__file__, 373, 12), maximum_55164, *[abs_call_result_55172, abs_call_result_55185], **kwargs_55186)
    
    # Assigning a type to the variable 'scale' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'scale', maximum_call_result_55187)
    
    # Assigning a Compare to a Name (line 376):
    
    # Assigning a Compare to a Name (line 376):
    
    # Getting the type of 'max_diff' (line 376)
    max_diff_55188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 21), 'max_diff')
    # Getting the type of 'NUM_JAC_DIFF_REJECT' (line 376)
    NUM_JAC_DIFF_REJECT_55189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 32), 'NUM_JAC_DIFF_REJECT')
    # Getting the type of 'scale' (line 376)
    scale_55190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 54), 'scale')
    # Applying the binary operator '*' (line 376)
    result_mul_55191 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 32), '*', NUM_JAC_DIFF_REJECT_55189, scale_55190)
    
    # Applying the binary operator '<' (line 376)
    result_lt_55192 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 21), '<', max_diff_55188, result_mul_55191)
    
    # Assigning a type to the variable 'diff_too_small' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'diff_too_small', result_lt_55192)
    
    
    # Call to any(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of 'diff_too_small' (line 377)
    diff_too_small_55195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 14), 'diff_too_small', False)
    # Processing the call keyword arguments (line 377)
    kwargs_55196 = {}
    # Getting the type of 'np' (line 377)
    np_55193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 377)
    any_55194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 7), np_55193, 'any')
    # Calling any(args, kwargs) (line 377)
    any_call_result_55197 = invoke(stypy.reporting.localization.Localization(__file__, 377, 7), any_55194, *[diff_too_small_55195], **kwargs_55196)
    
    # Testing the type of an if condition (line 377)
    if_condition_55198 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 377, 4), any_call_result_55197)
    # Assigning a type to the variable 'if_condition_55198' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'if_condition_55198', if_condition_55198)
    # SSA begins for if statement (line 377)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 378):
    
    # Assigning a Subscript to a Name (line 378):
    
    # Obtaining the type of the subscript
    int_55199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 8), 'int')
    
    # Call to nonzero(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 'diff_too_small' (line 378)
    diff_too_small_55202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 26), 'diff_too_small', False)
    # Processing the call keyword arguments (line 378)
    kwargs_55203 = {}
    # Getting the type of 'np' (line 378)
    np_55200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 15), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 378)
    nonzero_55201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 15), np_55200, 'nonzero')
    # Calling nonzero(args, kwargs) (line 378)
    nonzero_call_result_55204 = invoke(stypy.reporting.localization.Localization(__file__, 378, 15), nonzero_55201, *[diff_too_small_55202], **kwargs_55203)
    
    # Obtaining the member '__getitem__' of a type (line 378)
    getitem___55205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), nonzero_call_result_55204, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 378)
    subscript_call_result_55206 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), getitem___55205, int_55199)
    
    # Assigning a type to the variable 'tuple_var_assignment_54015' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'tuple_var_assignment_54015', subscript_call_result_55206)
    
    # Assigning a Name to a Name (line 378):
    # Getting the type of 'tuple_var_assignment_54015' (line 378)
    tuple_var_assignment_54015_55207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'tuple_var_assignment_54015')
    # Assigning a type to the variable 'ind' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'ind', tuple_var_assignment_54015_55207)
    
    # Assigning a BinOp to a Name (line 379):
    
    # Assigning a BinOp to a Name (line 379):
    # Getting the type of 'NUM_JAC_FACTOR_INCREASE' (line 379)
    NUM_JAC_FACTOR_INCREASE_55208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 21), 'NUM_JAC_FACTOR_INCREASE')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 379)
    ind_55209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 54), 'ind')
    # Getting the type of 'factor' (line 379)
    factor_55210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 47), 'factor')
    # Obtaining the member '__getitem__' of a type (line 379)
    getitem___55211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 47), factor_55210, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 379)
    subscript_call_result_55212 = invoke(stypy.reporting.localization.Localization(__file__, 379, 47), getitem___55211, ind_55209)
    
    # Applying the binary operator '*' (line 379)
    result_mul_55213 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 21), '*', NUM_JAC_FACTOR_INCREASE_55208, subscript_call_result_55212)
    
    # Assigning a type to the variable 'new_factor' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'new_factor', result_mul_55213)
    
    # Assigning a BinOp to a Name (line 380):
    
    # Assigning a BinOp to a Name (line 380):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 380)
    ind_55214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 19), 'ind')
    # Getting the type of 'y' (line 380)
    y_55215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 17), 'y')
    # Obtaining the member '__getitem__' of a type (line 380)
    getitem___55216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 17), y_55215, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 380)
    subscript_call_result_55217 = invoke(stypy.reporting.localization.Localization(__file__, 380, 17), getitem___55216, ind_55214)
    
    # Getting the type of 'new_factor' (line 380)
    new_factor_55218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 26), 'new_factor')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 380)
    ind_55219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 47), 'ind')
    # Getting the type of 'y_scale' (line 380)
    y_scale_55220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 39), 'y_scale')
    # Obtaining the member '__getitem__' of a type (line 380)
    getitem___55221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 39), y_scale_55220, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 380)
    subscript_call_result_55222 = invoke(stypy.reporting.localization.Localization(__file__, 380, 39), getitem___55221, ind_55219)
    
    # Applying the binary operator '*' (line 380)
    result_mul_55223 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 26), '*', new_factor_55218, subscript_call_result_55222)
    
    # Applying the binary operator '+' (line 380)
    result_add_55224 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 17), '+', subscript_call_result_55217, result_mul_55223)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 380)
    ind_55225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 57), 'ind')
    # Getting the type of 'y' (line 380)
    y_55226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 55), 'y')
    # Obtaining the member '__getitem__' of a type (line 380)
    getitem___55227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 55), y_55226, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 380)
    subscript_call_result_55228 = invoke(stypy.reporting.localization.Localization(__file__, 380, 55), getitem___55227, ind_55225)
    
    # Applying the binary operator '-' (line 380)
    result_sub_55229 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 16), '-', result_add_55224, subscript_call_result_55228)
    
    # Assigning a type to the variable 'h_new' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'h_new', result_sub_55229)
    
    # Assigning a Call to a Name (line 381):
    
    # Assigning a Call to a Name (line 381):
    
    # Call to zeros(...): (line 381)
    # Processing the call arguments (line 381)
    # Getting the type of 'n' (line 381)
    n_55232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 29), 'n', False)
    # Processing the call keyword arguments (line 381)
    kwargs_55233 = {}
    # Getting the type of 'np' (line 381)
    np_55230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 20), 'np', False)
    # Obtaining the member 'zeros' of a type (line 381)
    zeros_55231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 20), np_55230, 'zeros')
    # Calling zeros(args, kwargs) (line 381)
    zeros_call_result_55234 = invoke(stypy.reporting.localization.Localization(__file__, 381, 20), zeros_55231, *[n_55232], **kwargs_55233)
    
    # Assigning a type to the variable 'h_new_all' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'h_new_all', zeros_call_result_55234)
    
    # Assigning a Name to a Subscript (line 382):
    
    # Assigning a Name to a Subscript (line 382):
    # Getting the type of 'h_new' (line 382)
    h_new_55235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 25), 'h_new')
    # Getting the type of 'h_new_all' (line 382)
    h_new_all_55236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'h_new_all')
    # Getting the type of 'ind' (line 382)
    ind_55237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 18), 'ind')
    # Storing an element on a container (line 382)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 8), h_new_all_55236, (ind_55237, h_new_55235))
    
    # Assigning a Call to a Name (line 384):
    
    # Assigning a Call to a Name (line 384):
    
    # Call to unique(...): (line 384)
    # Processing the call arguments (line 384)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 384)
    ind_55240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 41), 'ind', False)
    # Getting the type of 'groups' (line 384)
    groups_55241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 34), 'groups', False)
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___55242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 34), groups_55241, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_55243 = invoke(stypy.reporting.localization.Localization(__file__, 384, 34), getitem___55242, ind_55240)
    
    # Processing the call keyword arguments (line 384)
    kwargs_55244 = {}
    # Getting the type of 'np' (line 384)
    np_55238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'np', False)
    # Obtaining the member 'unique' of a type (line 384)
    unique_55239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 24), np_55238, 'unique')
    # Calling unique(args, kwargs) (line 384)
    unique_call_result_55245 = invoke(stypy.reporting.localization.Localization(__file__, 384, 24), unique_55239, *[subscript_call_result_55243], **kwargs_55244)
    
    # Assigning a type to the variable 'groups_unique' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'groups_unique', unique_call_result_55245)
    
    # Assigning a Call to a Name (line 385):
    
    # Assigning a Call to a Name (line 385):
    
    # Call to empty(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'n_groups' (line 385)
    n_groups_55248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 30), 'n_groups', False)
    # Processing the call keyword arguments (line 385)
    # Getting the type of 'int' (line 385)
    int_55249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 46), 'int', False)
    keyword_55250 = int_55249
    kwargs_55251 = {'dtype': keyword_55250}
    # Getting the type of 'np' (line 385)
    np_55246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 21), 'np', False)
    # Obtaining the member 'empty' of a type (line 385)
    empty_55247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 21), np_55246, 'empty')
    # Calling empty(args, kwargs) (line 385)
    empty_call_result_55252 = invoke(stypy.reporting.localization.Localization(__file__, 385, 21), empty_55247, *[n_groups_55248], **kwargs_55251)
    
    # Assigning a type to the variable 'groups_map' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'groups_map', empty_call_result_55252)
    
    # Assigning a Call to a Name (line 386):
    
    # Assigning a Call to a Name (line 386):
    
    # Call to empty(...): (line 386)
    # Processing the call arguments (line 386)
    
    # Obtaining an instance of the builtin type 'tuple' (line 386)
    tuple_55255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 386)
    # Adding element type (line 386)
    
    # Obtaining the type of the subscript
    int_55256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 47), 'int')
    # Getting the type of 'groups_unique' (line 386)
    groups_unique_55257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 27), 'groups_unique', False)
    # Obtaining the member 'shape' of a type (line 386)
    shape_55258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 27), groups_unique_55257, 'shape')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___55259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 27), shape_55258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_55260 = invoke(stypy.reporting.localization.Localization(__file__, 386, 27), getitem___55259, int_55256)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 27), tuple_55255, subscript_call_result_55260)
    # Adding element type (line 386)
    # Getting the type of 'n' (line 386)
    n_55261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 51), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 27), tuple_55255, n_55261)
    
    # Processing the call keyword arguments (line 386)
    kwargs_55262 = {}
    # Getting the type of 'np' (line 386)
    np_55253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'np', False)
    # Obtaining the member 'empty' of a type (line 386)
    empty_55254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 17), np_55253, 'empty')
    # Calling empty(args, kwargs) (line 386)
    empty_call_result_55263 = invoke(stypy.reporting.localization.Localization(__file__, 386, 17), empty_55254, *[tuple_55255], **kwargs_55262)
    
    # Assigning a type to the variable 'h_vecs' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'h_vecs', empty_call_result_55263)
    
    
    # Call to enumerate(...): (line 387)
    # Processing the call arguments (line 387)
    # Getting the type of 'groups_unique' (line 387)
    groups_unique_55265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 34), 'groups_unique', False)
    # Processing the call keyword arguments (line 387)
    kwargs_55266 = {}
    # Getting the type of 'enumerate' (line 387)
    enumerate_55264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 24), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 387)
    enumerate_call_result_55267 = invoke(stypy.reporting.localization.Localization(__file__, 387, 24), enumerate_55264, *[groups_unique_55265], **kwargs_55266)
    
    # Testing the type of a for loop iterable (line 387)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 387, 8), enumerate_call_result_55267)
    # Getting the type of the for loop variable (line 387)
    for_loop_var_55268 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 387, 8), enumerate_call_result_55267)
    # Assigning a type to the variable 'k' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 8), for_loop_var_55268))
    # Assigning a type to the variable 'group' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'group', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 8), for_loop_var_55268))
    # SSA begins for a for statement (line 387)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 388):
    
    # Assigning a Call to a Name (line 388):
    
    # Call to equal(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'group' (line 388)
    group_55271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 25), 'group', False)
    # Getting the type of 'groups' (line 388)
    groups_55272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 32), 'groups', False)
    # Processing the call keyword arguments (line 388)
    kwargs_55273 = {}
    # Getting the type of 'np' (line 388)
    np_55269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 16), 'np', False)
    # Obtaining the member 'equal' of a type (line 388)
    equal_55270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 16), np_55269, 'equal')
    # Calling equal(args, kwargs) (line 388)
    equal_call_result_55274 = invoke(stypy.reporting.localization.Localization(__file__, 388, 16), equal_55270, *[group_55271, groups_55272], **kwargs_55273)
    
    # Assigning a type to the variable 'e' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'e', equal_call_result_55274)
    
    # Assigning a BinOp to a Subscript (line 389):
    
    # Assigning a BinOp to a Subscript (line 389):
    # Getting the type of 'h_new_all' (line 389)
    h_new_all_55275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 24), 'h_new_all')
    # Getting the type of 'e' (line 389)
    e_55276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 36), 'e')
    # Applying the binary operator '*' (line 389)
    result_mul_55277 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 24), '*', h_new_all_55275, e_55276)
    
    # Getting the type of 'h_vecs' (line 389)
    h_vecs_55278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'h_vecs')
    # Getting the type of 'k' (line 389)
    k_55279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'k')
    # Storing an element on a container (line 389)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 12), h_vecs_55278, (k_55279, result_mul_55277))
    
    # Assigning a Name to a Subscript (line 390):
    
    # Assigning a Name to a Subscript (line 390):
    # Getting the type of 'k' (line 390)
    k_55280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 32), 'k')
    # Getting the type of 'groups_map' (line 390)
    groups_map_55281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'groups_map')
    # Getting the type of 'group' (line 390)
    group_55282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 23), 'group')
    # Storing an element on a container (line 390)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), groups_map_55281, (group_55282, k_55280))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 391):
    
    # Assigning a Attribute to a Name (line 391):
    # Getting the type of 'h_vecs' (line 391)
    h_vecs_55283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 17), 'h_vecs')
    # Obtaining the member 'T' of a type (line 391)
    T_55284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 17), h_vecs_55283, 'T')
    # Assigning a type to the variable 'h_vecs' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'h_vecs', T_55284)
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to fun(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 't' (line 393)
    t_55286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 20), 't', False)
    
    # Obtaining the type of the subscript
    slice_55287 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 393, 23), None, None, None)
    # Getting the type of 'None' (line 393)
    None_55288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 28), 'None', False)
    # Getting the type of 'y' (line 393)
    y_55289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 23), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___55290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 23), y_55289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_55291 = invoke(stypy.reporting.localization.Localization(__file__, 393, 23), getitem___55290, (slice_55287, None_55288))
    
    # Getting the type of 'h_vecs' (line 393)
    h_vecs_55292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 36), 'h_vecs', False)
    # Applying the binary operator '+' (line 393)
    result_add_55293 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 23), '+', subscript_call_result_55291, h_vecs_55292)
    
    # Processing the call keyword arguments (line 393)
    kwargs_55294 = {}
    # Getting the type of 'fun' (line 393)
    fun_55285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 16), 'fun', False)
    # Calling fun(args, kwargs) (line 393)
    fun_call_result_55295 = invoke(stypy.reporting.localization.Localization(__file__, 393, 16), fun_55285, *[t_55286, result_add_55293], **kwargs_55294)
    
    # Assigning a type to the variable 'f_new' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'f_new', fun_call_result_55295)
    
    # Assigning a BinOp to a Name (line 394):
    
    # Assigning a BinOp to a Name (line 394):
    # Getting the type of 'f_new' (line 394)
    f_new_55296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 13), 'f_new')
    
    # Obtaining the type of the subscript
    slice_55297 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 394, 21), None, None, None)
    # Getting the type of 'None' (line 394)
    None_55298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 26), 'None')
    # Getting the type of 'f' (line 394)
    f_55299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 21), 'f')
    # Obtaining the member '__getitem__' of a type (line 394)
    getitem___55300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 21), f_55299, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 394)
    subscript_call_result_55301 = invoke(stypy.reporting.localization.Localization(__file__, 394, 21), getitem___55300, (slice_55297, None_55298))
    
    # Applying the binary operator '-' (line 394)
    result_sub_55302 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 13), '-', f_new_55296, subscript_call_result_55301)
    
    # Assigning a type to the variable 'df' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'df', result_sub_55302)
    
    # Assigning a Call to a Tuple (line 395):
    
    # Assigning a Subscript to a Name (line 395):
    
    # Obtaining the type of the subscript
    int_55303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 8), 'int')
    
    # Call to find(...): (line 395)
    # Processing the call arguments (line 395)
    
    # Obtaining the type of the subscript
    slice_55305 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 395, 23), None, None, None)
    # Getting the type of 'ind' (line 395)
    ind_55306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 36), 'ind', False)
    # Getting the type of 'structure' (line 395)
    structure_55307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 23), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___55308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 23), structure_55307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_55309 = invoke(stypy.reporting.localization.Localization(__file__, 395, 23), getitem___55308, (slice_55305, ind_55306))
    
    # Processing the call keyword arguments (line 395)
    kwargs_55310 = {}
    # Getting the type of 'find' (line 395)
    find_55304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 18), 'find', False)
    # Calling find(args, kwargs) (line 395)
    find_call_result_55311 = invoke(stypy.reporting.localization.Localization(__file__, 395, 18), find_55304, *[subscript_call_result_55309], **kwargs_55310)
    
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___55312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), find_call_result_55311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_55313 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), getitem___55312, int_55303)
    
    # Assigning a type to the variable 'tuple_var_assignment_54016' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'tuple_var_assignment_54016', subscript_call_result_55313)
    
    # Assigning a Subscript to a Name (line 395):
    
    # Obtaining the type of the subscript
    int_55314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 8), 'int')
    
    # Call to find(...): (line 395)
    # Processing the call arguments (line 395)
    
    # Obtaining the type of the subscript
    slice_55316 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 395, 23), None, None, None)
    # Getting the type of 'ind' (line 395)
    ind_55317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 36), 'ind', False)
    # Getting the type of 'structure' (line 395)
    structure_55318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 23), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___55319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 23), structure_55318, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_55320 = invoke(stypy.reporting.localization.Localization(__file__, 395, 23), getitem___55319, (slice_55316, ind_55317))
    
    # Processing the call keyword arguments (line 395)
    kwargs_55321 = {}
    # Getting the type of 'find' (line 395)
    find_55315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 18), 'find', False)
    # Calling find(args, kwargs) (line 395)
    find_call_result_55322 = invoke(stypy.reporting.localization.Localization(__file__, 395, 18), find_55315, *[subscript_call_result_55320], **kwargs_55321)
    
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___55323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), find_call_result_55322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_55324 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), getitem___55323, int_55314)
    
    # Assigning a type to the variable 'tuple_var_assignment_54017' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'tuple_var_assignment_54017', subscript_call_result_55324)
    
    # Assigning a Subscript to a Name (line 395):
    
    # Obtaining the type of the subscript
    int_55325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 8), 'int')
    
    # Call to find(...): (line 395)
    # Processing the call arguments (line 395)
    
    # Obtaining the type of the subscript
    slice_55327 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 395, 23), None, None, None)
    # Getting the type of 'ind' (line 395)
    ind_55328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 36), 'ind', False)
    # Getting the type of 'structure' (line 395)
    structure_55329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 23), 'structure', False)
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___55330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 23), structure_55329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_55331 = invoke(stypy.reporting.localization.Localization(__file__, 395, 23), getitem___55330, (slice_55327, ind_55328))
    
    # Processing the call keyword arguments (line 395)
    kwargs_55332 = {}
    # Getting the type of 'find' (line 395)
    find_55326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 18), 'find', False)
    # Calling find(args, kwargs) (line 395)
    find_call_result_55333 = invoke(stypy.reporting.localization.Localization(__file__, 395, 18), find_55326, *[subscript_call_result_55331], **kwargs_55332)
    
    # Obtaining the member '__getitem__' of a type (line 395)
    getitem___55334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), find_call_result_55333, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 395)
    subscript_call_result_55335 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), getitem___55334, int_55325)
    
    # Assigning a type to the variable 'tuple_var_assignment_54018' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'tuple_var_assignment_54018', subscript_call_result_55335)
    
    # Assigning a Name to a Name (line 395):
    # Getting the type of 'tuple_var_assignment_54016' (line 395)
    tuple_var_assignment_54016_55336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'tuple_var_assignment_54016')
    # Assigning a type to the variable 'i' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'i', tuple_var_assignment_54016_55336)
    
    # Assigning a Name to a Name (line 395):
    # Getting the type of 'tuple_var_assignment_54017' (line 395)
    tuple_var_assignment_54017_55337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'tuple_var_assignment_54017')
    # Assigning a type to the variable 'j' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 11), 'j', tuple_var_assignment_54017_55337)
    
    # Assigning a Name to a Name (line 395):
    # Getting the type of 'tuple_var_assignment_54018' (line 395)
    tuple_var_assignment_54018_55338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'tuple_var_assignment_54018')
    # Assigning a type to the variable '_' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 14), '_', tuple_var_assignment_54018_55338)
    
    # Assigning a Call to a Name (line 396):
    
    # Assigning a Call to a Name (line 396):
    
    # Call to tocsc(...): (line 396)
    # Processing the call keyword arguments (line 396)
    kwargs_55370 = {}
    
    # Call to coo_matrix(...): (line 396)
    # Processing the call arguments (line 396)
    
    # Obtaining an instance of the builtin type 'tuple' (line 396)
    tuple_55340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 396)
    # Adding element type (line 396)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 396)
    tuple_55341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 396)
    # Adding element type (line 396)
    # Getting the type of 'i' (line 396)
    i_55342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 34), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 34), tuple_55341, i_55342)
    # Adding element type (line 396)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 396)
    j_55343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 59), 'j', False)
    # Getting the type of 'ind' (line 396)
    ind_55344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 55), 'ind', False)
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___55345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 55), ind_55344, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_55346 = invoke(stypy.reporting.localization.Localization(__file__, 396, 55), getitem___55345, j_55343)
    
    # Getting the type of 'groups' (line 396)
    groups_55347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 48), 'groups', False)
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___55348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 48), groups_55347, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_55349 = invoke(stypy.reporting.localization.Localization(__file__, 396, 48), getitem___55348, subscript_call_result_55346)
    
    # Getting the type of 'groups_map' (line 396)
    groups_map_55350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 37), 'groups_map', False)
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___55351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 37), groups_map_55350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_55352 = invoke(stypy.reporting.localization.Localization(__file__, 396, 37), getitem___55351, subscript_call_result_55349)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 34), tuple_55341, subscript_call_result_55352)
    
    # Getting the type of 'df' (line 396)
    df_55353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 31), 'df', False)
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___55354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 31), df_55353, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_55355 = invoke(stypy.reporting.localization.Localization(__file__, 396, 31), getitem___55354, tuple_55341)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 31), tuple_55340, subscript_call_result_55355)
    # Adding element type (line 396)
    
    # Obtaining an instance of the builtin type 'tuple' (line 397)
    tuple_55356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 397)
    # Adding element type (line 397)
    # Getting the type of 'i' (line 397)
    i_55357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 32), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 32), tuple_55356, i_55357)
    # Adding element type (line 397)
    # Getting the type of 'j' (line 397)
    j_55358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 35), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 32), tuple_55356, j_55358)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 31), tuple_55340, tuple_55356)
    
    # Processing the call keyword arguments (line 396)
    
    # Obtaining an instance of the builtin type 'tuple' (line 397)
    tuple_55359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 397)
    # Adding element type (line 397)
    # Getting the type of 'n' (line 397)
    n_55360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 47), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 47), tuple_55359, n_55360)
    # Adding element type (line 397)
    
    # Obtaining the type of the subscript
    int_55361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 60), 'int')
    # Getting the type of 'ind' (line 397)
    ind_55362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 50), 'ind', False)
    # Obtaining the member 'shape' of a type (line 397)
    shape_55363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 50), ind_55362, 'shape')
    # Obtaining the member '__getitem__' of a type (line 397)
    getitem___55364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 50), shape_55363, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 397)
    subscript_call_result_55365 = invoke(stypy.reporting.localization.Localization(__file__, 397, 50), getitem___55364, int_55361)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 47), tuple_55359, subscript_call_result_55365)
    
    keyword_55366 = tuple_55359
    kwargs_55367 = {'shape': keyword_55366}
    # Getting the type of 'coo_matrix' (line 396)
    coo_matrix_55339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 396)
    coo_matrix_call_result_55368 = invoke(stypy.reporting.localization.Localization(__file__, 396, 19), coo_matrix_55339, *[tuple_55340], **kwargs_55367)
    
    # Obtaining the member 'tocsc' of a type (line 396)
    tocsc_55369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 19), coo_matrix_call_result_55368, 'tocsc')
    # Calling tocsc(args, kwargs) (line 396)
    tocsc_call_result_55371 = invoke(stypy.reporting.localization.Localization(__file__, 396, 19), tocsc_55369, *[], **kwargs_55370)
    
    # Assigning a type to the variable 'diff_new' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'diff_new', tocsc_call_result_55371)
    
    # Assigning a Call to a Name (line 399):
    
    # Assigning a Call to a Name (line 399):
    
    # Call to ravel(...): (line 399)
    # Processing the call keyword arguments (line 399)
    kwargs_55386 = {}
    
    # Call to array(...): (line 399)
    # Processing the call arguments (line 399)
    
    # Call to argmax(...): (line 399)
    # Processing the call keyword arguments (line 399)
    int_55379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 57), 'int')
    keyword_55380 = int_55379
    kwargs_55381 = {'axis': keyword_55380}
    
    # Call to abs(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'diff_new' (line 399)
    diff_new_55375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 35), 'diff_new', False)
    # Processing the call keyword arguments (line 399)
    kwargs_55376 = {}
    # Getting the type of 'abs' (line 399)
    abs_55374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'abs', False)
    # Calling abs(args, kwargs) (line 399)
    abs_call_result_55377 = invoke(stypy.reporting.localization.Localization(__file__, 399, 31), abs_55374, *[diff_new_55375], **kwargs_55376)
    
    # Obtaining the member 'argmax' of a type (line 399)
    argmax_55378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 31), abs_call_result_55377, 'argmax')
    # Calling argmax(args, kwargs) (line 399)
    argmax_call_result_55382 = invoke(stypy.reporting.localization.Localization(__file__, 399, 31), argmax_55378, *[], **kwargs_55381)
    
    # Processing the call keyword arguments (line 399)
    kwargs_55383 = {}
    # Getting the type of 'np' (line 399)
    np_55372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 22), 'np', False)
    # Obtaining the member 'array' of a type (line 399)
    array_55373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 22), np_55372, 'array')
    # Calling array(args, kwargs) (line 399)
    array_call_result_55384 = invoke(stypy.reporting.localization.Localization(__file__, 399, 22), array_55373, *[argmax_call_result_55382], **kwargs_55383)
    
    # Obtaining the member 'ravel' of a type (line 399)
    ravel_55385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 22), array_call_result_55384, 'ravel')
    # Calling ravel(args, kwargs) (line 399)
    ravel_call_result_55387 = invoke(stypy.reporting.localization.Localization(__file__, 399, 22), ravel_55385, *[], **kwargs_55386)
    
    # Assigning a type to the variable 'max_ind_new' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'max_ind_new', ravel_call_result_55387)
    
    # Assigning a Call to a Name (line 400):
    
    # Assigning a Call to a Name (line 400):
    
    # Call to arange(...): (line 400)
    # Processing the call arguments (line 400)
    
    # Obtaining the type of the subscript
    int_55390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 32), 'int')
    # Getting the type of 'ind' (line 400)
    ind_55391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 22), 'ind', False)
    # Obtaining the member 'shape' of a type (line 400)
    shape_55392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 22), ind_55391, 'shape')
    # Obtaining the member '__getitem__' of a type (line 400)
    getitem___55393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 22), shape_55392, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 400)
    subscript_call_result_55394 = invoke(stypy.reporting.localization.Localization(__file__, 400, 22), getitem___55393, int_55390)
    
    # Processing the call keyword arguments (line 400)
    kwargs_55395 = {}
    # Getting the type of 'np' (line 400)
    np_55388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'np', False)
    # Obtaining the member 'arange' of a type (line 400)
    arange_55389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 12), np_55388, 'arange')
    # Calling arange(args, kwargs) (line 400)
    arange_call_result_55396 = invoke(stypy.reporting.localization.Localization(__file__, 400, 12), arange_55389, *[subscript_call_result_55394], **kwargs_55395)
    
    # Assigning a type to the variable 'r' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'r', arange_call_result_55396)
    
    # Assigning a Call to a Name (line 401):
    
    # Assigning a Call to a Name (line 401):
    
    # Call to ravel(...): (line 401)
    # Processing the call keyword arguments (line 401)
    kwargs_55412 = {}
    
    # Call to asarray(...): (line 401)
    # Processing the call arguments (line 401)
    
    # Call to abs(...): (line 401)
    # Processing the call arguments (line 401)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 401)
    tuple_55401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 401)
    # Adding element type (line 401)
    # Getting the type of 'max_ind_new' (line 401)
    max_ind_new_55402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 50), 'max_ind_new', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 50), tuple_55401, max_ind_new_55402)
    # Adding element type (line 401)
    # Getting the type of 'r' (line 401)
    r_55403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 63), 'r', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 50), tuple_55401, r_55403)
    
    # Getting the type of 'diff_new' (line 401)
    diff_new_55404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 41), 'diff_new', False)
    # Obtaining the member '__getitem__' of a type (line 401)
    getitem___55405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 41), diff_new_55404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 401)
    subscript_call_result_55406 = invoke(stypy.reporting.localization.Localization(__file__, 401, 41), getitem___55405, tuple_55401)
    
    # Processing the call keyword arguments (line 401)
    kwargs_55407 = {}
    # Getting the type of 'np' (line 401)
    np_55399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 34), 'np', False)
    # Obtaining the member 'abs' of a type (line 401)
    abs_55400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 34), np_55399, 'abs')
    # Calling abs(args, kwargs) (line 401)
    abs_call_result_55408 = invoke(stypy.reporting.localization.Localization(__file__, 401, 34), abs_55400, *[subscript_call_result_55406], **kwargs_55407)
    
    # Processing the call keyword arguments (line 401)
    kwargs_55409 = {}
    # Getting the type of 'np' (line 401)
    np_55397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 23), 'np', False)
    # Obtaining the member 'asarray' of a type (line 401)
    asarray_55398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 23), np_55397, 'asarray')
    # Calling asarray(args, kwargs) (line 401)
    asarray_call_result_55410 = invoke(stypy.reporting.localization.Localization(__file__, 401, 23), asarray_55398, *[abs_call_result_55408], **kwargs_55409)
    
    # Obtaining the member 'ravel' of a type (line 401)
    ravel_55411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 23), asarray_call_result_55410, 'ravel')
    # Calling ravel(args, kwargs) (line 401)
    ravel_call_result_55413 = invoke(stypy.reporting.localization.Localization(__file__, 401, 23), ravel_55411, *[], **kwargs_55412)
    
    # Assigning a type to the variable 'max_diff_new' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'max_diff_new', ravel_call_result_55413)
    
    # Assigning a Call to a Name (line 402):
    
    # Assigning a Call to a Name (line 402):
    
    # Call to maximum(...): (line 402)
    # Processing the call arguments (line 402)
    
    # Call to abs(...): (line 403)
    # Processing the call arguments (line 403)
    
    # Obtaining the type of the subscript
    # Getting the type of 'max_ind_new' (line 403)
    max_ind_new_55418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 21), 'max_ind_new', False)
    # Getting the type of 'f' (line 403)
    f_55419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'f', False)
    # Obtaining the member '__getitem__' of a type (line 403)
    getitem___55420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 19), f_55419, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 403)
    subscript_call_result_55421 = invoke(stypy.reporting.localization.Localization(__file__, 403, 19), getitem___55420, max_ind_new_55418)
    
    # Processing the call keyword arguments (line 403)
    kwargs_55422 = {}
    # Getting the type of 'np' (line 403)
    np_55416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'np', False)
    # Obtaining the member 'abs' of a type (line 403)
    abs_55417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 12), np_55416, 'abs')
    # Calling abs(args, kwargs) (line 403)
    abs_call_result_55423 = invoke(stypy.reporting.localization.Localization(__file__, 403, 12), abs_55417, *[subscript_call_result_55421], **kwargs_55422)
    
    
    # Call to abs(...): (line 404)
    # Processing the call arguments (line 404)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 404)
    tuple_55426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 404)
    # Adding element type (line 404)
    # Getting the type of 'max_ind_new' (line 404)
    max_ind_new_55427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 25), 'max_ind_new', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 25), tuple_55426, max_ind_new_55427)
    # Adding element type (line 404)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 404)
    ind_55428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 56), 'ind', False)
    # Getting the type of 'groups' (line 404)
    groups_55429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 49), 'groups', False)
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___55430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 49), groups_55429, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_55431 = invoke(stypy.reporting.localization.Localization(__file__, 404, 49), getitem___55430, ind_55428)
    
    # Getting the type of 'groups_map' (line 404)
    groups_map_55432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 38), 'groups_map', False)
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___55433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 38), groups_map_55432, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_55434 = invoke(stypy.reporting.localization.Localization(__file__, 404, 38), getitem___55433, subscript_call_result_55431)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 25), tuple_55426, subscript_call_result_55434)
    
    # Getting the type of 'f_new' (line 404)
    f_new_55435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 19), 'f_new', False)
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___55436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 19), f_new_55435, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_55437 = invoke(stypy.reporting.localization.Localization(__file__, 404, 19), getitem___55436, tuple_55426)
    
    # Processing the call keyword arguments (line 404)
    kwargs_55438 = {}
    # Getting the type of 'np' (line 404)
    np_55424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'np', False)
    # Obtaining the member 'abs' of a type (line 404)
    abs_55425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 12), np_55424, 'abs')
    # Calling abs(args, kwargs) (line 404)
    abs_call_result_55439 = invoke(stypy.reporting.localization.Localization(__file__, 404, 12), abs_55425, *[subscript_call_result_55437], **kwargs_55438)
    
    # Processing the call keyword arguments (line 402)
    kwargs_55440 = {}
    # Getting the type of 'np' (line 402)
    np_55414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'np', False)
    # Obtaining the member 'maximum' of a type (line 402)
    maximum_55415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 20), np_55414, 'maximum')
    # Calling maximum(args, kwargs) (line 402)
    maximum_call_result_55441 = invoke(stypy.reporting.localization.Localization(__file__, 402, 20), maximum_55415, *[abs_call_result_55423, abs_call_result_55439], **kwargs_55440)
    
    # Assigning a type to the variable 'scale_new' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'scale_new', maximum_call_result_55441)
    
    # Assigning a Compare to a Name (line 406):
    
    # Assigning a Compare to a Name (line 406):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 406)
    ind_55442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 26), 'ind')
    # Getting the type of 'max_diff' (line 406)
    max_diff_55443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 17), 'max_diff')
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___55444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 17), max_diff_55443, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_55445 = invoke(stypy.reporting.localization.Localization(__file__, 406, 17), getitem___55444, ind_55442)
    
    # Getting the type of 'scale_new' (line 406)
    scale_new_55446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 33), 'scale_new')
    # Applying the binary operator '*' (line 406)
    result_mul_55447 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 17), '*', subscript_call_result_55445, scale_new_55446)
    
    # Getting the type of 'max_diff_new' (line 406)
    max_diff_new_55448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 45), 'max_diff_new')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 406)
    ind_55449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 66), 'ind')
    # Getting the type of 'scale' (line 406)
    scale_55450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 60), 'scale')
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___55451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 60), scale_55450, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_55452 = invoke(stypy.reporting.localization.Localization(__file__, 406, 60), getitem___55451, ind_55449)
    
    # Applying the binary operator '*' (line 406)
    result_mul_55453 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 45), '*', max_diff_new_55448, subscript_call_result_55452)
    
    # Applying the binary operator '<' (line 406)
    result_lt_55454 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 17), '<', result_mul_55447, result_mul_55453)
    
    # Assigning a type to the variable 'update' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'update', result_lt_55454)
    
    
    # Call to any(...): (line 407)
    # Processing the call arguments (line 407)
    # Getting the type of 'update' (line 407)
    update_55457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 18), 'update', False)
    # Processing the call keyword arguments (line 407)
    kwargs_55458 = {}
    # Getting the type of 'np' (line 407)
    np_55455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 11), 'np', False)
    # Obtaining the member 'any' of a type (line 407)
    any_55456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 11), np_55455, 'any')
    # Calling any(args, kwargs) (line 407)
    any_call_result_55459 = invoke(stypy.reporting.localization.Localization(__file__, 407, 11), any_55456, *[update_55457], **kwargs_55458)
    
    # Testing the type of an if condition (line 407)
    if_condition_55460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 407, 8), any_call_result_55459)
    # Assigning a type to the variable 'if_condition_55460' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'if_condition_55460', if_condition_55460)
    # SSA begins for if statement (line 407)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 408):
    
    # Assigning a Subscript to a Name (line 408):
    
    # Obtaining the type of the subscript
    int_55461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 12), 'int')
    
    # Call to where(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'update' (line 408)
    update_55464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 31), 'update', False)
    # Processing the call keyword arguments (line 408)
    kwargs_55465 = {}
    # Getting the type of 'np' (line 408)
    np_55462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 22), 'np', False)
    # Obtaining the member 'where' of a type (line 408)
    where_55463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 22), np_55462, 'where')
    # Calling where(args, kwargs) (line 408)
    where_call_result_55466 = invoke(stypy.reporting.localization.Localization(__file__, 408, 22), where_55463, *[update_55464], **kwargs_55465)
    
    # Obtaining the member '__getitem__' of a type (line 408)
    getitem___55467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 12), where_call_result_55466, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 408)
    subscript_call_result_55468 = invoke(stypy.reporting.localization.Localization(__file__, 408, 12), getitem___55467, int_55461)
    
    # Assigning a type to the variable 'tuple_var_assignment_54019' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'tuple_var_assignment_54019', subscript_call_result_55468)
    
    # Assigning a Name to a Name (line 408):
    # Getting the type of 'tuple_var_assignment_54019' (line 408)
    tuple_var_assignment_54019_55469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'tuple_var_assignment_54019')
    # Assigning a type to the variable 'update' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'update', tuple_var_assignment_54019_55469)
    
    # Assigning a Subscript to a Name (line 409):
    
    # Assigning a Subscript to a Name (line 409):
    
    # Obtaining the type of the subscript
    # Getting the type of 'update' (line 409)
    update_55470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 29), 'update')
    # Getting the type of 'ind' (line 409)
    ind_55471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 25), 'ind')
    # Obtaining the member '__getitem__' of a type (line 409)
    getitem___55472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 25), ind_55471, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 409)
    subscript_call_result_55473 = invoke(stypy.reporting.localization.Localization(__file__, 409, 25), getitem___55472, update_55470)
    
    # Assigning a type to the variable 'update_ind' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'update_ind', subscript_call_result_55473)
    
    # Assigning a Subscript to a Subscript (line 410):
    
    # Assigning a Subscript to a Subscript (line 410):
    
    # Obtaining the type of the subscript
    # Getting the type of 'update' (line 410)
    update_55474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 44), 'update')
    # Getting the type of 'new_factor' (line 410)
    new_factor_55475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 33), 'new_factor')
    # Obtaining the member '__getitem__' of a type (line 410)
    getitem___55476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 33), new_factor_55475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 410)
    subscript_call_result_55477 = invoke(stypy.reporting.localization.Localization(__file__, 410, 33), getitem___55476, update_55474)
    
    # Getting the type of 'factor' (line 410)
    factor_55478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'factor')
    # Getting the type of 'update_ind' (line 410)
    update_ind_55479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 19), 'update_ind')
    # Storing an element on a container (line 410)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 12), factor_55478, (update_ind_55479, subscript_call_result_55477))
    
    # Assigning a Subscript to a Subscript (line 411):
    
    # Assigning a Subscript to a Subscript (line 411):
    
    # Obtaining the type of the subscript
    # Getting the type of 'update' (line 411)
    update_55480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 34), 'update')
    # Getting the type of 'h_new' (line 411)
    h_new_55481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 28), 'h_new')
    # Obtaining the member '__getitem__' of a type (line 411)
    getitem___55482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 28), h_new_55481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 411)
    subscript_call_result_55483 = invoke(stypy.reporting.localization.Localization(__file__, 411, 28), getitem___55482, update_55480)
    
    # Getting the type of 'h' (line 411)
    h_55484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'h')
    # Getting the type of 'update_ind' (line 411)
    update_ind_55485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 14), 'update_ind')
    # Storing an element on a container (line 411)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), h_55484, (update_ind_55485, subscript_call_result_55483))
    
    # Assigning a Subscript to a Subscript (line 412):
    
    # Assigning a Subscript to a Subscript (line 412):
    
    # Obtaining the type of the subscript
    slice_55486 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 412, 34), None, None, None)
    # Getting the type of 'update' (line 412)
    update_55487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 46), 'update')
    # Getting the type of 'diff_new' (line 412)
    diff_new_55488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 34), 'diff_new')
    # Obtaining the member '__getitem__' of a type (line 412)
    getitem___55489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 34), diff_new_55488, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 412)
    subscript_call_result_55490 = invoke(stypy.reporting.localization.Localization(__file__, 412, 34), getitem___55489, (slice_55486, update_55487))
    
    # Getting the type of 'diff' (line 412)
    diff_55491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'diff')
    slice_55492 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 412, 12), None, None, None)
    # Getting the type of 'update_ind' (line 412)
    update_ind_55493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 20), 'update_ind')
    # Storing an element on a container (line 412)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), diff_55491, ((slice_55492, update_ind_55493), subscript_call_result_55490))
    
    # Assigning a Subscript to a Subscript (line 413):
    
    # Assigning a Subscript to a Subscript (line 413):
    
    # Obtaining the type of the subscript
    # Getting the type of 'update' (line 413)
    update_55494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 42), 'update')
    # Getting the type of 'scale_new' (line 413)
    scale_new_55495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 32), 'scale_new')
    # Obtaining the member '__getitem__' of a type (line 413)
    getitem___55496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 32), scale_new_55495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 413)
    subscript_call_result_55497 = invoke(stypy.reporting.localization.Localization(__file__, 413, 32), getitem___55496, update_55494)
    
    # Getting the type of 'scale' (line 413)
    scale_55498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'scale')
    # Getting the type of 'update_ind' (line 413)
    update_ind_55499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 18), 'update_ind')
    # Storing an element on a container (line 413)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), scale_55498, (update_ind_55499, subscript_call_result_55497))
    
    # Assigning a Subscript to a Subscript (line 414):
    
    # Assigning a Subscript to a Subscript (line 414):
    
    # Obtaining the type of the subscript
    # Getting the type of 'update' (line 414)
    update_55500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 48), 'update')
    # Getting the type of 'max_diff_new' (line 414)
    max_diff_new_55501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 35), 'max_diff_new')
    # Obtaining the member '__getitem__' of a type (line 414)
    getitem___55502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 35), max_diff_new_55501, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 414)
    subscript_call_result_55503 = invoke(stypy.reporting.localization.Localization(__file__, 414, 35), getitem___55502, update_55500)
    
    # Getting the type of 'max_diff' (line 414)
    max_diff_55504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'max_diff')
    # Getting the type of 'update_ind' (line 414)
    update_ind_55505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 21), 'update_ind')
    # Storing an element on a container (line 414)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 12), max_diff_55504, (update_ind_55505, subscript_call_result_55503))
    # SSA join for if statement (line 407)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 377)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'diff' (line 416)
    diff_55506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'diff')
    # Obtaining the member 'data' of a type (line 416)
    data_55507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 4), diff_55506, 'data')
    
    # Call to repeat(...): (line 416)
    # Processing the call arguments (line 416)
    # Getting the type of 'h' (line 416)
    h_55510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 27), 'h', False)
    
    # Call to diff(...): (line 416)
    # Processing the call arguments (line 416)
    # Getting the type of 'diff' (line 416)
    diff_55513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 38), 'diff', False)
    # Obtaining the member 'indptr' of a type (line 416)
    indptr_55514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 38), diff_55513, 'indptr')
    # Processing the call keyword arguments (line 416)
    kwargs_55515 = {}
    # Getting the type of 'np' (line 416)
    np_55511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 30), 'np', False)
    # Obtaining the member 'diff' of a type (line 416)
    diff_55512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 30), np_55511, 'diff')
    # Calling diff(args, kwargs) (line 416)
    diff_call_result_55516 = invoke(stypy.reporting.localization.Localization(__file__, 416, 30), diff_55512, *[indptr_55514], **kwargs_55515)
    
    # Processing the call keyword arguments (line 416)
    kwargs_55517 = {}
    # Getting the type of 'np' (line 416)
    np_55508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 17), 'np', False)
    # Obtaining the member 'repeat' of a type (line 416)
    repeat_55509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 17), np_55508, 'repeat')
    # Calling repeat(args, kwargs) (line 416)
    repeat_call_result_55518 = invoke(stypy.reporting.localization.Localization(__file__, 416, 17), repeat_55509, *[h_55510, diff_call_result_55516], **kwargs_55517)
    
    # Applying the binary operator 'div=' (line 416)
    result_div_55519 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 4), 'div=', data_55507, repeat_call_result_55518)
    # Getting the type of 'diff' (line 416)
    diff_55520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'diff')
    # Setting the type of the member 'data' of a type (line 416)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 4), diff_55520, 'data', result_div_55519)
    
    
    # Getting the type of 'factor' (line 418)
    factor_55521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'factor')
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'max_diff' (line 418)
    max_diff_55522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 11), 'max_diff')
    # Getting the type of 'NUM_JAC_DIFF_SMALL' (line 418)
    NUM_JAC_DIFF_SMALL_55523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 22), 'NUM_JAC_DIFF_SMALL')
    # Getting the type of 'scale' (line 418)
    scale_55524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 43), 'scale')
    # Applying the binary operator '*' (line 418)
    result_mul_55525 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 22), '*', NUM_JAC_DIFF_SMALL_55523, scale_55524)
    
    # Applying the binary operator '<' (line 418)
    result_lt_55526 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 11), '<', max_diff_55522, result_mul_55525)
    
    # Getting the type of 'factor' (line 418)
    factor_55527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'factor')
    # Obtaining the member '__getitem__' of a type (line 418)
    getitem___55528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 4), factor_55527, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 418)
    subscript_call_result_55529 = invoke(stypy.reporting.localization.Localization(__file__, 418, 4), getitem___55528, result_lt_55526)
    
    # Getting the type of 'NUM_JAC_FACTOR_INCREASE' (line 418)
    NUM_JAC_FACTOR_INCREASE_55530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 53), 'NUM_JAC_FACTOR_INCREASE')
    # Applying the binary operator '*=' (line 418)
    result_imul_55531 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 4), '*=', subscript_call_result_55529, NUM_JAC_FACTOR_INCREASE_55530)
    # Getting the type of 'factor' (line 418)
    factor_55532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'factor')
    
    # Getting the type of 'max_diff' (line 418)
    max_diff_55533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 11), 'max_diff')
    # Getting the type of 'NUM_JAC_DIFF_SMALL' (line 418)
    NUM_JAC_DIFF_SMALL_55534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 22), 'NUM_JAC_DIFF_SMALL')
    # Getting the type of 'scale' (line 418)
    scale_55535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 43), 'scale')
    # Applying the binary operator '*' (line 418)
    result_mul_55536 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 22), '*', NUM_JAC_DIFF_SMALL_55534, scale_55535)
    
    # Applying the binary operator '<' (line 418)
    result_lt_55537 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 11), '<', max_diff_55533, result_mul_55536)
    
    # Storing an element on a container (line 418)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 4), factor_55532, (result_lt_55537, result_imul_55531))
    
    
    # Getting the type of 'factor' (line 419)
    factor_55538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'factor')
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'max_diff' (line 419)
    max_diff_55539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 11), 'max_diff')
    # Getting the type of 'NUM_JAC_DIFF_BIG' (line 419)
    NUM_JAC_DIFF_BIG_55540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 22), 'NUM_JAC_DIFF_BIG')
    # Getting the type of 'scale' (line 419)
    scale_55541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 41), 'scale')
    # Applying the binary operator '*' (line 419)
    result_mul_55542 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 22), '*', NUM_JAC_DIFF_BIG_55540, scale_55541)
    
    # Applying the binary operator '>' (line 419)
    result_gt_55543 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 11), '>', max_diff_55539, result_mul_55542)
    
    # Getting the type of 'factor' (line 419)
    factor_55544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'factor')
    # Obtaining the member '__getitem__' of a type (line 419)
    getitem___55545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 4), factor_55544, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 419)
    subscript_call_result_55546 = invoke(stypy.reporting.localization.Localization(__file__, 419, 4), getitem___55545, result_gt_55543)
    
    # Getting the type of 'NUM_JAC_FACTOR_DECREASE' (line 419)
    NUM_JAC_FACTOR_DECREASE_55547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 51), 'NUM_JAC_FACTOR_DECREASE')
    # Applying the binary operator '*=' (line 419)
    result_imul_55548 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 4), '*=', subscript_call_result_55546, NUM_JAC_FACTOR_DECREASE_55547)
    # Getting the type of 'factor' (line 419)
    factor_55549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'factor')
    
    # Getting the type of 'max_diff' (line 419)
    max_diff_55550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 11), 'max_diff')
    # Getting the type of 'NUM_JAC_DIFF_BIG' (line 419)
    NUM_JAC_DIFF_BIG_55551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 22), 'NUM_JAC_DIFF_BIG')
    # Getting the type of 'scale' (line 419)
    scale_55552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 41), 'scale')
    # Applying the binary operator '*' (line 419)
    result_mul_55553 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 22), '*', NUM_JAC_DIFF_BIG_55551, scale_55552)
    
    # Applying the binary operator '>' (line 419)
    result_gt_55554 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 11), '>', max_diff_55550, result_mul_55553)
    
    # Storing an element on a container (line 419)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 4), factor_55549, (result_gt_55554, result_imul_55548))
    
    
    # Assigning a Call to a Name (line 420):
    
    # Assigning a Call to a Name (line 420):
    
    # Call to maximum(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'factor' (line 420)
    factor_55557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 24), 'factor', False)
    # Getting the type of 'NUM_JAC_MIN_FACTOR' (line 420)
    NUM_JAC_MIN_FACTOR_55558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 32), 'NUM_JAC_MIN_FACTOR', False)
    # Processing the call keyword arguments (line 420)
    kwargs_55559 = {}
    # Getting the type of 'np' (line 420)
    np_55555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 13), 'np', False)
    # Obtaining the member 'maximum' of a type (line 420)
    maximum_55556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 13), np_55555, 'maximum')
    # Calling maximum(args, kwargs) (line 420)
    maximum_call_result_55560 = invoke(stypy.reporting.localization.Localization(__file__, 420, 13), maximum_55556, *[factor_55557, NUM_JAC_MIN_FACTOR_55558], **kwargs_55559)
    
    # Assigning a type to the variable 'factor' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'factor', maximum_call_result_55560)
    
    # Obtaining an instance of the builtin type 'tuple' (line 422)
    tuple_55561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 422)
    # Adding element type (line 422)
    # Getting the type of 'diff' (line 422)
    diff_55562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 11), 'diff')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 11), tuple_55561, diff_55562)
    # Adding element type (line 422)
    # Getting the type of 'factor' (line 422)
    factor_55563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 17), 'factor')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 11), tuple_55561, factor_55563)
    
    # Assigning a type to the variable 'stypy_return_type' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type', tuple_55561)
    
    # ################# End of '_sparse_num_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_sparse_num_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 356)
    stypy_return_type_55564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55564)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_sparse_num_jac'
    return stypy_return_type_55564

# Assigning a type to the variable '_sparse_num_jac' (line 356)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 0), '_sparse_num_jac', _sparse_num_jac)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
