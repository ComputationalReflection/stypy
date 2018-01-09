
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: import numpy as np
3: from .base import OdeSolver, DenseOutput
4: from .common import (validate_max_step, validate_tol, select_initial_step,
5:                      norm, warn_extraneous)
6: 
7: 
8: # Multiply steps computed from asymptotic behaviour of errors by this.
9: SAFETY = 0.9
10: 
11: MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
12: MAX_FACTOR = 10  # Maximum allowed increase in a step size.
13: 
14: 
15: def rk_step(fun, t, y, f, h, A, B, C, E, K):
16:     '''Perform a single Runge-Kutta step.
17: 
18:     This function computes a prediction of an explicit Runge-Kutta method and
19:     also estimates the error of a less accurate method.
20: 
21:     Notation for Butcher tableau is as in [1]_.
22: 
23:     Parameters
24:     ----------
25:     fun : callable
26:         Right-hand side of the system.
27:     t : float
28:         Current time.
29:     y : ndarray, shape (n,)
30:         Current state.
31:     f : ndarray, shape (n,)
32:         Current value of the derivative, i.e. ``fun(x, y)``.
33:     h : float
34:         Step to use.
35:     A : list of ndarray, length n_stages - 1
36:         Coefficients for combining previous RK stages to compute the next
37:         stage. For explicit methods the coefficients above the main diagonal
38:         are zeros, so `A` is stored as a list of arrays of increasing lengths.
39:         The first stage is always just `f`, thus no coefficients for it
40:         are required.
41:     B : ndarray, shape (n_stages,)
42:         Coefficients for combining RK stages for computing the final
43:         prediction.
44:     C : ndarray, shape (n_stages - 1,)
45:         Coefficients for incrementing time for consecutive RK stages.
46:         The value for the first stage is always zero, thus it is not stored.
47:     E : ndarray, shape (n_stages + 1,)
48:         Coefficients for estimating the error of a less accurate method. They
49:         are computed as the difference between b's in an extended tableau.
50:     K : ndarray, shape (n_stages + 1, n)
51:         Storage array for putting RK stages here. Stages are stored in rows.
52: 
53:     Returns
54:     -------
55:     y_new : ndarray, shape (n,)
56:         Solution at t + h computed with a higher accuracy.
57:     f_new : ndarray, shape (n,)
58:         Derivative ``fun(t + h, y_new)``.
59:     error : ndarray, shape (n,)
60:         Error estimate of a less accurate method.
61: 
62:     References
63:     ----------
64:     .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
65:            Equations I: Nonstiff Problems", Sec. II.4.
66:     '''
67:     K[0] = f
68:     for s, (a, c) in enumerate(zip(A, C)):
69:         dy = np.dot(K[:s + 1].T, a) * h
70:         K[s + 1] = fun(t + c * h, y + dy)
71: 
72:     y_new = y + h * np.dot(K[:-1].T, B)
73:     f_new = fun(t + h, y_new)
74: 
75:     K[-1] = f_new
76:     error = np.dot(K.T, E) * h
77: 
78:     return y_new, f_new, error
79: 
80: 
81: class RungeKutta(OdeSolver):
82:     '''Base class for explicit Runge-Kutta methods.'''
83:     C = NotImplemented
84:     A = NotImplemented
85:     B = NotImplemented
86:     E = NotImplemented
87:     P = NotImplemented
88:     order = NotImplemented
89:     n_stages = NotImplemented
90: 
91:     def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
92:                  rtol=1e-3, atol=1e-6, vectorized=False, **extraneous):
93:         warn_extraneous(extraneous)
94:         super(RungeKutta, self).__init__(fun, t0, y0, t_bound, vectorized,
95:                                          support_complex=True)
96:         self.y_old = None
97:         self.max_step = validate_max_step(max_step)
98:         self.rtol, self.atol = validate_tol(rtol, atol, self.n)
99:         self.f = self.fun(self.t, self.y)
100:         self.h_abs = select_initial_step(
101:             self.fun, self.t, self.y, self.f, self.direction,
102:             self.order, self.rtol, self.atol)
103:         self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
104: 
105:     def _step_impl(self):
106:         t = self.t
107:         y = self.y
108: 
109:         max_step = self.max_step
110:         rtol = self.rtol
111:         atol = self.atol
112: 
113:         min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
114: 
115:         if self.h_abs > max_step:
116:             h_abs = max_step
117:         elif self.h_abs < min_step:
118:             h_abs = min_step
119:         else:
120:             h_abs = self.h_abs
121: 
122:         order = self.order
123:         step_accepted = False
124: 
125:         while not step_accepted:
126:             if h_abs < min_step:
127:                 return False, self.TOO_SMALL_STEP
128: 
129:             h = h_abs * self.direction
130:             t_new = t + h
131: 
132:             if self.direction * (t_new - self.t_bound) > 0:
133:                 t_new = self.t_bound
134: 
135:             h = t_new - t
136:             h_abs = np.abs(h)
137: 
138:             y_new, f_new, error = rk_step(self.fun, t, y, self.f, h, self.A,
139:                                           self.B, self.C, self.E, self.K)
140:             scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
141:             error_norm = norm(error / scale)
142: 
143:             if error_norm < 1:
144:                 h_abs *= min(MAX_FACTOR,
145:                              max(1, SAFETY * error_norm ** (-1 / (order + 1))))
146:                 step_accepted = True
147:             else:
148:                 h_abs *= max(MIN_FACTOR,
149:                              SAFETY * error_norm ** (-1 / (order + 1)))
150: 
151:         self.y_old = y
152: 
153:         self.t = t_new
154:         self.y = y_new
155: 
156:         self.h_abs = h_abs
157:         self.f = f_new
158: 
159:         return True, None
160: 
161:     def _dense_output_impl(self):
162:         Q = self.K.T.dot(self.P)
163:         return RkDenseOutput(self.t_old, self.t, self.y_old, Q)
164: 
165: 
166: class RK23(RungeKutta):
167:     '''Explicit Runge-Kutta method of order 3(2).
168: 
169:     The Bogacki-Shamping pair of formulas is used [1]_. The error is controlled
170:     assuming 2nd order accuracy, but steps are taken using a 3rd oder accurate
171:     formula (local extrapolation is done). A cubic Hermit polynomial is used
172:     for the dense output.
173: 
174:     Can be applied in a complex domain.
175: 
176:     Parameters
177:     ----------
178:     fun : callable
179:         Right-hand side of the system. The calling signature is ``fun(t, y)``.
180:         Here ``t`` is a scalar and there are two options for ndarray ``y``.
181:         It can either have shape (n,), then ``fun`` must return array_like with
182:         shape (n,). Or alternatively it can have shape (n, k), then ``fun``
183:         must return array_like with shape (n, k), i.e. each column
184:         corresponds to a single column in ``y``. The choice between the two
185:         options is determined by `vectorized` argument (see below). The
186:         vectorized implementation allows faster approximation of the Jacobian
187:         by finite differences.
188:     t0 : float
189:         Initial time.
190:     y0 : array_like, shape (n,)
191:         Initial state.
192:     t_bound : float
193:         Boundary time --- the integration won't continue beyond it. It also
194:         determines the direction of the integration.
195:     max_step : float, optional
196:         Maximum allowed step size. Default is np.inf, i.e. the step is not
197:         bounded and determined solely by the solver.
198:     rtol, atol : float and array_like, optional
199:         Relative and absolute tolerances. The solver keeps the local error
200:         estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
201:         relative accuracy (number of correct digits). But if a component of `y`
202:         is approximately below `atol` then the error only needs to fall within
203:         the same `atol` threshold, and the number of correct digits is not
204:         guaranteed. If components of y have different scales, it might be
205:         beneficial to set different `atol` values for different components by
206:         passing array_like with shape (n,) for `atol`. Default values are
207:         1e-3 for `rtol` and 1e-6 for `atol`.
208:     vectorized : bool, optional
209:         Whether `fun` is implemented in a vectorized fashion. Default is False.
210: 
211:     Attributes
212:     ----------
213:     n : int
214:         Number of equations.
215:     status : string
216:         Current status of the solver: 'running', 'finished' or 'failed'.
217:     t_bound : float
218:         Boundary time.
219:     direction : float
220:         Integration direction: +1 or -1.
221:     t : float
222:         Current time.
223:     y : ndarray
224:         Current state.
225:     t_old : float
226:         Previous time. None if no steps were made yet.
227:     step_size : float
228:         Size of the last successful step. None if no steps were made yet.
229:     nfev : int
230:         Number of the system's rhs evaluations.
231:     njev : int
232:         Number of the Jacobian evaluations.
233:     nlu : int
234:         Number of LU decompositions.
235: 
236:     References
237:     ----------
238:     .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
239:            Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
240:     '''
241:     order = 2
242:     n_stages = 3
243:     C = np.array([1/2, 3/4])
244:     A = [np.array([1/2]),
245:          np.array([0, 3/4])]
246:     B = np.array([2/9, 1/3, 4/9])
247:     E = np.array([5/72, -1/12, -1/9, 1/8])
248:     P = np.array([[1, -4 / 3, 5 / 9],
249:                   [0, 1, -2/3],
250:                   [0, 4/3, -8/9],
251:                   [0, -1, 1]])
252: 
253: 
254: class RK45(RungeKutta):
255:     '''Explicit Runge-Kutta method of order 5(4).
256: 
257:     The Dormand-Prince pair of formulas is used [1]_. The error is controlled
258:     assuming 4th order accuracy, but steps are taken using a 5th
259:     oder accurate formula (local extrapolation is done). A quartic
260:     interpolation polynomial is used for the dense output [2]_.
261: 
262:     Can be applied in a complex domain.
263: 
264:     Parameters
265:     ----------
266:     fun : callable
267:         Right-hand side of the system. The calling signature is ``fun(t, y)``.
268:         Here ``t`` is a scalar and there are two options for ndarray ``y``.
269:         It can either have shape (n,), then ``fun`` must return array_like with
270:         shape (n,). Or alternatively it can have shape (n, k), then ``fun``
271:         must return array_like with shape (n, k), i.e. each column
272:         corresponds to a single column in ``y``. The choice between the two
273:         options is determined by `vectorized` argument (see below). The
274:         vectorized implementation allows faster approximation of the Jacobian
275:         by finite differences.
276:     t0 : float
277:         Initial value of the independent variable.
278:     y0 : array_like, shape (n,)
279:         Initial values of the dependent variable.
280:     t_bound : float
281:         Boundary time --- the integration won't continue beyond it. It also
282:         determines the direction of the integration.
283:     max_step : float, optional
284:         Maximum allowed step size. Default is np.inf, i.e. the step is not
285:         bounded and determined solely by the solver.
286:     rtol, atol : float and array_like, optional
287:         Relative and absolute tolerances. The solver keeps the local error
288:         estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
289:         relative accuracy (number of correct digits). But if a component of `y`
290:         is approximately below `atol` then the error only needs to fall within
291:         the same `atol` threshold, and the number of correct digits is not
292:         guaranteed. If components of y have different scales, it might be
293:         beneficial to set different `atol` values for different components by
294:         passing array_like with shape (n,) for `atol`. Default values are
295:         1e-3 for `rtol` and 1e-6 for `atol`.
296:     vectorized : bool, optional
297:         Whether `fun` is implemented in a vectorized fashion. Default is False.
298: 
299:     Attributes
300:     ----------
301:     n : int
302:         Number of equations.
303:     status : string
304:         Current status of the solver: 'running', 'finished' or 'failed'.
305:     t_bound : float
306:         Boundary time.
307:     direction : float
308:         Integration direction: +1 or -1.
309:     t : float
310:         Current time.
311:     y : ndarray
312:         Current state.
313:     t_old : float
314:         Previous time. None if no steps were made yet.
315:     step_size : float
316:         Size of the last successful step. None if no steps were made yet.
317:     nfev : int
318:         Number of the system's rhs evaluations.
319:     njev : int
320:         Number of the Jacobian evaluations.
321:     nlu : int
322:         Number of LU decompositions.
323: 
324:     References
325:     ----------
326:     .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
327:            formulae", Journal of Computational and Applied Mathematics, Vol. 6,
328:            No. 1, pp. 19-26, 1980.
329:     .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
330:            of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
331:     '''
332:     order = 4
333:     n_stages = 6
334:     C = np.array([1/5, 3/10, 4/5, 8/9, 1])
335:     A = [np.array([1/5]),
336:          np.array([3/40, 9/40]),
337:          np.array([44/45, -56/15, 32/9]),
338:          np.array([19372/6561, -25360/2187, 64448/6561, -212/729]),
339:          np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656])]
340:     B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
341:     E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
342:                   1/40])
343:     # Corresponds to the optimum value of c_6 from [2]_.
344:     P = np.array([
345:         [1, -8048581381/2820520608, 8663915743/2820520608,
346:          -12715105075/11282082432],
347:         [0, 0, 0, 0],
348:         [0, 131558114200/32700410799, -68118460800/10900136933,
349:          87487479700/32700410799],
350:         [0, -1754552775/470086768, 14199869525/1410260304,
351:          -10690763975/1880347072],
352:         [0, 127303824393/49829197408, -318862633887/49829197408,
353:          701980252875 / 199316789632],
354:         [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
355:         [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])
356: 
357: 
358: class RkDenseOutput(DenseOutput):
359:     def __init__(self, t_old, t, y_old, Q):
360:         super(RkDenseOutput, self).__init__(t_old, t)
361:         self.h = t - t_old
362:         self.Q = Q
363:         self.order = Q.shape[1] - 1
364:         self.y_old = y_old
365: 
366:     def _call_impl(self, t):
367:         x = (t - self.t_old) / self.h
368:         if t.ndim == 0:
369:             p = np.tile(x, self.order + 1)
370:             p = np.cumprod(p)
371:         else:
372:             p = np.tile(x, (self.order + 1, 1))
373:             p = np.cumprod(p, axis=0)
374:         y = self.h * np.dot(self.Q, p)
375:         if y.ndim == 2:
376:             y += self.y_old[:, None]
377:         else:
378:             y += self.y_old
379: 
380:         return y
381: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_58181 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_58181) is not StypyTypeError):

    if (import_58181 != 'pyd_module'):
        __import__(import_58181)
        sys_modules_58182 = sys.modules[import_58181]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_58182.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_58181)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.integrate._ivp.base import OdeSolver, DenseOutput' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_58183 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.integrate._ivp.base')

if (type(import_58183) is not StypyTypeError):

    if (import_58183 != 'pyd_module'):
        __import__(import_58183)
        sys_modules_58184 = sys.modules[import_58183]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.integrate._ivp.base', sys_modules_58184.module_type_store, module_type_store, ['OdeSolver', 'DenseOutput'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_58184, sys_modules_58184.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.base import OdeSolver, DenseOutput

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.integrate._ivp.base', None, module_type_store, ['OdeSolver', 'DenseOutput'], [OdeSolver, DenseOutput])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.base' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.integrate._ivp.base', import_58183)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.integrate._ivp.common import validate_max_step, validate_tol, select_initial_step, norm, warn_extraneous' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_58185 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.common')

if (type(import_58185) is not StypyTypeError):

    if (import_58185 != 'pyd_module'):
        __import__(import_58185)
        sys_modules_58186 = sys.modules[import_58185]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.common', sys_modules_58186.module_type_store, module_type_store, ['validate_max_step', 'validate_tol', 'select_initial_step', 'norm', 'warn_extraneous'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_58186, sys_modules_58186.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.common import validate_max_step, validate_tol, select_initial_step, norm, warn_extraneous

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.common', None, module_type_store, ['validate_max_step', 'validate_tol', 'select_initial_step', 'norm', 'warn_extraneous'], [validate_max_step, validate_tol, select_initial_step, norm, warn_extraneous])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.common' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.common', import_58185)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')


# Assigning a Num to a Name (line 9):

# Assigning a Num to a Name (line 9):
float_58187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 9), 'float')
# Assigning a type to the variable 'SAFETY' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'SAFETY', float_58187)

# Assigning a Num to a Name (line 11):

# Assigning a Num to a Name (line 11):
float_58188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'float')
# Assigning a type to the variable 'MIN_FACTOR' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'MIN_FACTOR', float_58188)

# Assigning a Num to a Name (line 12):

# Assigning a Num to a Name (line 12):
int_58189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 13), 'int')
# Assigning a type to the variable 'MAX_FACTOR' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'MAX_FACTOR', int_58189)

@norecursion
def rk_step(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rk_step'
    module_type_store = module_type_store.open_function_context('rk_step', 15, 0, False)
    
    # Passed parameters checking function
    rk_step.stypy_localization = localization
    rk_step.stypy_type_of_self = None
    rk_step.stypy_type_store = module_type_store
    rk_step.stypy_function_name = 'rk_step'
    rk_step.stypy_param_names_list = ['fun', 't', 'y', 'f', 'h', 'A', 'B', 'C', 'E', 'K']
    rk_step.stypy_varargs_param_name = None
    rk_step.stypy_kwargs_param_name = None
    rk_step.stypy_call_defaults = defaults
    rk_step.stypy_call_varargs = varargs
    rk_step.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rk_step', ['fun', 't', 'y', 'f', 'h', 'A', 'B', 'C', 'E', 'K'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rk_step', localization, ['fun', 't', 'y', 'f', 'h', 'A', 'B', 'C', 'E', 'K'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rk_step(...)' code ##################

    str_58190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'str', 'Perform a single Runge-Kutta step.\n\n    This function computes a prediction of an explicit Runge-Kutta method and\n    also estimates the error of a less accurate method.\n\n    Notation for Butcher tableau is as in [1]_.\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system.\n    t : float\n        Current time.\n    y : ndarray, shape (n,)\n        Current state.\n    f : ndarray, shape (n,)\n        Current value of the derivative, i.e. ``fun(x, y)``.\n    h : float\n        Step to use.\n    A : list of ndarray, length n_stages - 1\n        Coefficients for combining previous RK stages to compute the next\n        stage. For explicit methods the coefficients above the main diagonal\n        are zeros, so `A` is stored as a list of arrays of increasing lengths.\n        The first stage is always just `f`, thus no coefficients for it\n        are required.\n    B : ndarray, shape (n_stages,)\n        Coefficients for combining RK stages for computing the final\n        prediction.\n    C : ndarray, shape (n_stages - 1,)\n        Coefficients for incrementing time for consecutive RK stages.\n        The value for the first stage is always zero, thus it is not stored.\n    E : ndarray, shape (n_stages + 1,)\n        Coefficients for estimating the error of a less accurate method. They\n        are computed as the difference between b\'s in an extended tableau.\n    K : ndarray, shape (n_stages + 1, n)\n        Storage array for putting RK stages here. Stages are stored in rows.\n\n    Returns\n    -------\n    y_new : ndarray, shape (n,)\n        Solution at t + h computed with a higher accuracy.\n    f_new : ndarray, shape (n,)\n        Derivative ``fun(t + h, y_new)``.\n    error : ndarray, shape (n,)\n        Error estimate of a less accurate method.\n\n    References\n    ----------\n    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential\n           Equations I: Nonstiff Problems", Sec. II.4.\n    ')
    
    # Assigning a Name to a Subscript (line 67):
    
    # Assigning a Name to a Subscript (line 67):
    # Getting the type of 'f' (line 67)
    f_58191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'f')
    # Getting the type of 'K' (line 67)
    K_58192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'K')
    int_58193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 6), 'int')
    # Storing an element on a container (line 67)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 4), K_58192, (int_58193, f_58191))
    
    
    # Call to enumerate(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Call to zip(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'A' (line 68)
    A_58196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 35), 'A', False)
    # Getting the type of 'C' (line 68)
    C_58197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 38), 'C', False)
    # Processing the call keyword arguments (line 68)
    kwargs_58198 = {}
    # Getting the type of 'zip' (line 68)
    zip_58195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 31), 'zip', False)
    # Calling zip(args, kwargs) (line 68)
    zip_call_result_58199 = invoke(stypy.reporting.localization.Localization(__file__, 68, 31), zip_58195, *[A_58196, C_58197], **kwargs_58198)
    
    # Processing the call keyword arguments (line 68)
    kwargs_58200 = {}
    # Getting the type of 'enumerate' (line 68)
    enumerate_58194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 68)
    enumerate_call_result_58201 = invoke(stypy.reporting.localization.Localization(__file__, 68, 21), enumerate_58194, *[zip_call_result_58199], **kwargs_58200)
    
    # Testing the type of a for loop iterable (line 68)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 68, 4), enumerate_call_result_58201)
    # Getting the type of the for loop variable (line 68)
    for_loop_var_58202 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 68, 4), enumerate_call_result_58201)
    # Assigning a type to the variable 's' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 's', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 4), for_loop_var_58202))
    # Assigning a type to the variable 'a' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 4), for_loop_var_58202))
    # Assigning a type to the variable 'c' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 4), for_loop_var_58202))
    # SSA begins for a for statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 69):
    
    # Assigning a BinOp to a Name (line 69):
    
    # Call to dot(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 69)
    s_58205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 's', False)
    int_58206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 27), 'int')
    # Applying the binary operator '+' (line 69)
    result_add_58207 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 23), '+', s_58205, int_58206)
    
    slice_58208 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 20), None, result_add_58207, None)
    # Getting the type of 'K' (line 69)
    K_58209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'K', False)
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___58210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 20), K_58209, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_58211 = invoke(stypy.reporting.localization.Localization(__file__, 69, 20), getitem___58210, slice_58208)
    
    # Obtaining the member 'T' of a type (line 69)
    T_58212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 20), subscript_call_result_58211, 'T')
    # Getting the type of 'a' (line 69)
    a_58213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 33), 'a', False)
    # Processing the call keyword arguments (line 69)
    kwargs_58214 = {}
    # Getting the type of 'np' (line 69)
    np_58203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'np', False)
    # Obtaining the member 'dot' of a type (line 69)
    dot_58204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), np_58203, 'dot')
    # Calling dot(args, kwargs) (line 69)
    dot_call_result_58215 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), dot_58204, *[T_58212, a_58213], **kwargs_58214)
    
    # Getting the type of 'h' (line 69)
    h_58216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 38), 'h')
    # Applying the binary operator '*' (line 69)
    result_mul_58217 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '*', dot_call_result_58215, h_58216)
    
    # Assigning a type to the variable 'dy' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'dy', result_mul_58217)
    
    # Assigning a Call to a Subscript (line 70):
    
    # Assigning a Call to a Subscript (line 70):
    
    # Call to fun(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 't' (line 70)
    t_58219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 't', False)
    # Getting the type of 'c' (line 70)
    c_58220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'c', False)
    # Getting the type of 'h' (line 70)
    h_58221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 31), 'h', False)
    # Applying the binary operator '*' (line 70)
    result_mul_58222 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 27), '*', c_58220, h_58221)
    
    # Applying the binary operator '+' (line 70)
    result_add_58223 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 23), '+', t_58219, result_mul_58222)
    
    # Getting the type of 'y' (line 70)
    y_58224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 34), 'y', False)
    # Getting the type of 'dy' (line 70)
    dy_58225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'dy', False)
    # Applying the binary operator '+' (line 70)
    result_add_58226 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 34), '+', y_58224, dy_58225)
    
    # Processing the call keyword arguments (line 70)
    kwargs_58227 = {}
    # Getting the type of 'fun' (line 70)
    fun_58218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'fun', False)
    # Calling fun(args, kwargs) (line 70)
    fun_call_result_58228 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), fun_58218, *[result_add_58223, result_add_58226], **kwargs_58227)
    
    # Getting the type of 'K' (line 70)
    K_58229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'K')
    # Getting the type of 's' (line 70)
    s_58230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 10), 's')
    int_58231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 14), 'int')
    # Applying the binary operator '+' (line 70)
    result_add_58232 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 10), '+', s_58230, int_58231)
    
    # Storing an element on a container (line 70)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 8), K_58229, (result_add_58232, fun_call_result_58228))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 72):
    
    # Assigning a BinOp to a Name (line 72):
    # Getting the type of 'y' (line 72)
    y_58233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'y')
    # Getting the type of 'h' (line 72)
    h_58234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'h')
    
    # Call to dot(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Obtaining the type of the subscript
    int_58237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 30), 'int')
    slice_58238 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 72, 27), None, int_58237, None)
    # Getting the type of 'K' (line 72)
    K_58239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'K', False)
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___58240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 27), K_58239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_58241 = invoke(stypy.reporting.localization.Localization(__file__, 72, 27), getitem___58240, slice_58238)
    
    # Obtaining the member 'T' of a type (line 72)
    T_58242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 27), subscript_call_result_58241, 'T')
    # Getting the type of 'B' (line 72)
    B_58243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 37), 'B', False)
    # Processing the call keyword arguments (line 72)
    kwargs_58244 = {}
    # Getting the type of 'np' (line 72)
    np_58235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'np', False)
    # Obtaining the member 'dot' of a type (line 72)
    dot_58236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 20), np_58235, 'dot')
    # Calling dot(args, kwargs) (line 72)
    dot_call_result_58245 = invoke(stypy.reporting.localization.Localization(__file__, 72, 20), dot_58236, *[T_58242, B_58243], **kwargs_58244)
    
    # Applying the binary operator '*' (line 72)
    result_mul_58246 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 16), '*', h_58234, dot_call_result_58245)
    
    # Applying the binary operator '+' (line 72)
    result_add_58247 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 12), '+', y_58233, result_mul_58246)
    
    # Assigning a type to the variable 'y_new' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'y_new', result_add_58247)
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to fun(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 't' (line 73)
    t_58249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 't', False)
    # Getting the type of 'h' (line 73)
    h_58250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'h', False)
    # Applying the binary operator '+' (line 73)
    result_add_58251 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 16), '+', t_58249, h_58250)
    
    # Getting the type of 'y_new' (line 73)
    y_new_58252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'y_new', False)
    # Processing the call keyword arguments (line 73)
    kwargs_58253 = {}
    # Getting the type of 'fun' (line 73)
    fun_58248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'fun', False)
    # Calling fun(args, kwargs) (line 73)
    fun_call_result_58254 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), fun_58248, *[result_add_58251, y_new_58252], **kwargs_58253)
    
    # Assigning a type to the variable 'f_new' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'f_new', fun_call_result_58254)
    
    # Assigning a Name to a Subscript (line 75):
    
    # Assigning a Name to a Subscript (line 75):
    # Getting the type of 'f_new' (line 75)
    f_new_58255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'f_new')
    # Getting the type of 'K' (line 75)
    K_58256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'K')
    int_58257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 6), 'int')
    # Storing an element on a container (line 75)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 4), K_58256, (int_58257, f_new_58255))
    
    # Assigning a BinOp to a Name (line 76):
    
    # Assigning a BinOp to a Name (line 76):
    
    # Call to dot(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'K' (line 76)
    K_58260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'K', False)
    # Obtaining the member 'T' of a type (line 76)
    T_58261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 19), K_58260, 'T')
    # Getting the type of 'E' (line 76)
    E_58262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'E', False)
    # Processing the call keyword arguments (line 76)
    kwargs_58263 = {}
    # Getting the type of 'np' (line 76)
    np_58258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'np', False)
    # Obtaining the member 'dot' of a type (line 76)
    dot_58259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), np_58258, 'dot')
    # Calling dot(args, kwargs) (line 76)
    dot_call_result_58264 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), dot_58259, *[T_58261, E_58262], **kwargs_58263)
    
    # Getting the type of 'h' (line 76)
    h_58265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 29), 'h')
    # Applying the binary operator '*' (line 76)
    result_mul_58266 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 12), '*', dot_call_result_58264, h_58265)
    
    # Assigning a type to the variable 'error' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'error', result_mul_58266)
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_58267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    # Getting the type of 'y_new' (line 78)
    y_new_58268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'y_new')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 11), tuple_58267, y_new_58268)
    # Adding element type (line 78)
    # Getting the type of 'f_new' (line 78)
    f_new_58269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 18), 'f_new')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 11), tuple_58267, f_new_58269)
    # Adding element type (line 78)
    # Getting the type of 'error' (line 78)
    error_58270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'error')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 11), tuple_58267, error_58270)
    
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type', tuple_58267)
    
    # ################# End of 'rk_step(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rk_step' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_58271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_58271)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rk_step'
    return stypy_return_type_58271

# Assigning a type to the variable 'rk_step' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'rk_step', rk_step)
# Declaration of the 'RungeKutta' class
# Getting the type of 'OdeSolver' (line 81)
OdeSolver_58272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'OdeSolver')

class RungeKutta(OdeSolver_58272, ):
    str_58273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'str', 'Base class for explicit Runge-Kutta methods.')
    
    # Assigning a Name to a Name (line 83):
    
    # Assigning a Name to a Name (line 84):
    
    # Assigning a Name to a Name (line 85):
    
    # Assigning a Name to a Name (line 86):
    
    # Assigning a Name to a Name (line 87):
    
    # Assigning a Name to a Name (line 88):
    
    # Assigning a Name to a Name (line 89):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'np' (line 91)
        np_58274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 54), 'np')
        # Obtaining the member 'inf' of a type (line 91)
        inf_58275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 54), np_58274, 'inf')
        float_58276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'float')
        float_58277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 33), 'float')
        # Getting the type of 'False' (line 92)
        False_58278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 50), 'False')
        defaults = [inf_58275, float_58276, float_58277, False_58278]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RungeKutta.__init__', ['fun', 't0', 'y0', 't_bound', 'max_step', 'rtol', 'atol', 'vectorized'], None, 'extraneous', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fun', 't0', 'y0', 't_bound', 'max_step', 'rtol', 'atol', 'vectorized'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to warn_extraneous(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'extraneous' (line 93)
        extraneous_58280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'extraneous', False)
        # Processing the call keyword arguments (line 93)
        kwargs_58281 = {}
        # Getting the type of 'warn_extraneous' (line 93)
        warn_extraneous_58279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'warn_extraneous', False)
        # Calling warn_extraneous(args, kwargs) (line 93)
        warn_extraneous_call_result_58282 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), warn_extraneous_58279, *[extraneous_58280], **kwargs_58281)
        
        
        # Call to __init__(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'fun' (line 94)
        fun_58289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 41), 'fun', False)
        # Getting the type of 't0' (line 94)
        t0_58290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 46), 't0', False)
        # Getting the type of 'y0' (line 94)
        y0_58291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 50), 'y0', False)
        # Getting the type of 't_bound' (line 94)
        t_bound_58292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 54), 't_bound', False)
        # Getting the type of 'vectorized' (line 94)
        vectorized_58293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 63), 'vectorized', False)
        # Processing the call keyword arguments (line 94)
        # Getting the type of 'True' (line 95)
        True_58294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 57), 'True', False)
        keyword_58295 = True_58294
        kwargs_58296 = {'support_complex': keyword_58295}
        
        # Call to super(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'RungeKutta' (line 94)
        RungeKutta_58284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'RungeKutta', False)
        # Getting the type of 'self' (line 94)
        self_58285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'self', False)
        # Processing the call keyword arguments (line 94)
        kwargs_58286 = {}
        # Getting the type of 'super' (line 94)
        super_58283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'super', False)
        # Calling super(args, kwargs) (line 94)
        super_call_result_58287 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), super_58283, *[RungeKutta_58284, self_58285], **kwargs_58286)
        
        # Obtaining the member '__init__' of a type (line 94)
        init___58288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), super_call_result_58287, '__init__')
        # Calling __init__(args, kwargs) (line 94)
        init___call_result_58297 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), init___58288, *[fun_58289, t0_58290, y0_58291, t_bound_58292, vectorized_58293], **kwargs_58296)
        
        
        # Assigning a Name to a Attribute (line 96):
        
        # Assigning a Name to a Attribute (line 96):
        # Getting the type of 'None' (line 96)
        None_58298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'None')
        # Getting the type of 'self' (line 96)
        self_58299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Setting the type of the member 'y_old' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_58299, 'y_old', None_58298)
        
        # Assigning a Call to a Attribute (line 97):
        
        # Assigning a Call to a Attribute (line 97):
        
        # Call to validate_max_step(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'max_step' (line 97)
        max_step_58301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 42), 'max_step', False)
        # Processing the call keyword arguments (line 97)
        kwargs_58302 = {}
        # Getting the type of 'validate_max_step' (line 97)
        validate_max_step_58300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'validate_max_step', False)
        # Calling validate_max_step(args, kwargs) (line 97)
        validate_max_step_call_result_58303 = invoke(stypy.reporting.localization.Localization(__file__, 97, 24), validate_max_step_58300, *[max_step_58301], **kwargs_58302)
        
        # Getting the type of 'self' (line 97)
        self_58304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self')
        # Setting the type of the member 'max_step' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_58304, 'max_step', validate_max_step_call_result_58303)
        
        # Assigning a Call to a Tuple (line 98):
        
        # Assigning a Subscript to a Name (line 98):
        
        # Obtaining the type of the subscript
        int_58305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 8), 'int')
        
        # Call to validate_tol(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'rtol' (line 98)
        rtol_58307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 44), 'rtol', False)
        # Getting the type of 'atol' (line 98)
        atol_58308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 50), 'atol', False)
        # Getting the type of 'self' (line 98)
        self_58309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 56), 'self', False)
        # Obtaining the member 'n' of a type (line 98)
        n_58310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 56), self_58309, 'n')
        # Processing the call keyword arguments (line 98)
        kwargs_58311 = {}
        # Getting the type of 'validate_tol' (line 98)
        validate_tol_58306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'validate_tol', False)
        # Calling validate_tol(args, kwargs) (line 98)
        validate_tol_call_result_58312 = invoke(stypy.reporting.localization.Localization(__file__, 98, 31), validate_tol_58306, *[rtol_58307, atol_58308, n_58310], **kwargs_58311)
        
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___58313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), validate_tol_call_result_58312, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_58314 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), getitem___58313, int_58305)
        
        # Assigning a type to the variable 'tuple_var_assignment_58176' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'tuple_var_assignment_58176', subscript_call_result_58314)
        
        # Assigning a Subscript to a Name (line 98):
        
        # Obtaining the type of the subscript
        int_58315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 8), 'int')
        
        # Call to validate_tol(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'rtol' (line 98)
        rtol_58317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 44), 'rtol', False)
        # Getting the type of 'atol' (line 98)
        atol_58318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 50), 'atol', False)
        # Getting the type of 'self' (line 98)
        self_58319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 56), 'self', False)
        # Obtaining the member 'n' of a type (line 98)
        n_58320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 56), self_58319, 'n')
        # Processing the call keyword arguments (line 98)
        kwargs_58321 = {}
        # Getting the type of 'validate_tol' (line 98)
        validate_tol_58316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'validate_tol', False)
        # Calling validate_tol(args, kwargs) (line 98)
        validate_tol_call_result_58322 = invoke(stypy.reporting.localization.Localization(__file__, 98, 31), validate_tol_58316, *[rtol_58317, atol_58318, n_58320], **kwargs_58321)
        
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___58323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), validate_tol_call_result_58322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_58324 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), getitem___58323, int_58315)
        
        # Assigning a type to the variable 'tuple_var_assignment_58177' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'tuple_var_assignment_58177', subscript_call_result_58324)
        
        # Assigning a Name to a Attribute (line 98):
        # Getting the type of 'tuple_var_assignment_58176' (line 98)
        tuple_var_assignment_58176_58325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'tuple_var_assignment_58176')
        # Getting the type of 'self' (line 98)
        self_58326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self')
        # Setting the type of the member 'rtol' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_58326, 'rtol', tuple_var_assignment_58176_58325)
        
        # Assigning a Name to a Attribute (line 98):
        # Getting the type of 'tuple_var_assignment_58177' (line 98)
        tuple_var_assignment_58177_58327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'tuple_var_assignment_58177')
        # Getting the type of 'self' (line 98)
        self_58328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 19), 'self')
        # Setting the type of the member 'atol' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 19), self_58328, 'atol', tuple_var_assignment_58177_58327)
        
        # Assigning a Call to a Attribute (line 99):
        
        # Assigning a Call to a Attribute (line 99):
        
        # Call to fun(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'self' (line 99)
        self_58331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'self', False)
        # Obtaining the member 't' of a type (line 99)
        t_58332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 26), self_58331, 't')
        # Getting the type of 'self' (line 99)
        self_58333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'self', False)
        # Obtaining the member 'y' of a type (line 99)
        y_58334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 34), self_58333, 'y')
        # Processing the call keyword arguments (line 99)
        kwargs_58335 = {}
        # Getting the type of 'self' (line 99)
        self_58329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'self', False)
        # Obtaining the member 'fun' of a type (line 99)
        fun_58330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 17), self_58329, 'fun')
        # Calling fun(args, kwargs) (line 99)
        fun_call_result_58336 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), fun_58330, *[t_58332, y_58334], **kwargs_58335)
        
        # Getting the type of 'self' (line 99)
        self_58337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Setting the type of the member 'f' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_58337, 'f', fun_call_result_58336)
        
        # Assigning a Call to a Attribute (line 100):
        
        # Assigning a Call to a Attribute (line 100):
        
        # Call to select_initial_step(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'self' (line 101)
        self_58339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self', False)
        # Obtaining the member 'fun' of a type (line 101)
        fun_58340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_58339, 'fun')
        # Getting the type of 'self' (line 101)
        self_58341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'self', False)
        # Obtaining the member 't' of a type (line 101)
        t_58342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 22), self_58341, 't')
        # Getting the type of 'self' (line 101)
        self_58343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'self', False)
        # Obtaining the member 'y' of a type (line 101)
        y_58344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 30), self_58343, 'y')
        # Getting the type of 'self' (line 101)
        self_58345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 38), 'self', False)
        # Obtaining the member 'f' of a type (line 101)
        f_58346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 38), self_58345, 'f')
        # Getting the type of 'self' (line 101)
        self_58347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 46), 'self', False)
        # Obtaining the member 'direction' of a type (line 101)
        direction_58348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 46), self_58347, 'direction')
        # Getting the type of 'self' (line 102)
        self_58349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self', False)
        # Obtaining the member 'order' of a type (line 102)
        order_58350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_58349, 'order')
        # Getting the type of 'self' (line 102)
        self_58351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'self', False)
        # Obtaining the member 'rtol' of a type (line 102)
        rtol_58352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 24), self_58351, 'rtol')
        # Getting the type of 'self' (line 102)
        self_58353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'self', False)
        # Obtaining the member 'atol' of a type (line 102)
        atol_58354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 35), self_58353, 'atol')
        # Processing the call keyword arguments (line 100)
        kwargs_58355 = {}
        # Getting the type of 'select_initial_step' (line 100)
        select_initial_step_58338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 21), 'select_initial_step', False)
        # Calling select_initial_step(args, kwargs) (line 100)
        select_initial_step_call_result_58356 = invoke(stypy.reporting.localization.Localization(__file__, 100, 21), select_initial_step_58338, *[fun_58340, t_58342, y_58344, f_58346, direction_58348, order_58350, rtol_58352, atol_58354], **kwargs_58355)
        
        # Getting the type of 'self' (line 100)
        self_58357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Setting the type of the member 'h_abs' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_58357, 'h_abs', select_initial_step_call_result_58356)
        
        # Assigning a Call to a Attribute (line 103):
        
        # Assigning a Call to a Attribute (line 103):
        
        # Call to empty(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_58360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        # Getting the type of 'self' (line 103)
        self_58361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'self', False)
        # Obtaining the member 'n_stages' of a type (line 103)
        n_stages_58362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), self_58361, 'n_stages')
        int_58363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 43), 'int')
        # Applying the binary operator '+' (line 103)
        result_add_58364 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 27), '+', n_stages_58362, int_58363)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 27), tuple_58360, result_add_58364)
        # Adding element type (line 103)
        # Getting the type of 'self' (line 103)
        self_58365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 46), 'self', False)
        # Obtaining the member 'n' of a type (line 103)
        n_58366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 46), self_58365, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 27), tuple_58360, n_58366)
        
        # Processing the call keyword arguments (line 103)
        # Getting the type of 'self' (line 103)
        self_58367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 61), 'self', False)
        # Obtaining the member 'y' of a type (line 103)
        y_58368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 61), self_58367, 'y')
        # Obtaining the member 'dtype' of a type (line 103)
        dtype_58369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 61), y_58368, 'dtype')
        keyword_58370 = dtype_58369
        kwargs_58371 = {'dtype': keyword_58370}
        # Getting the type of 'np' (line 103)
        np_58358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 17), 'np', False)
        # Obtaining the member 'empty' of a type (line 103)
        empty_58359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 17), np_58358, 'empty')
        # Calling empty(args, kwargs) (line 103)
        empty_call_result_58372 = invoke(stypy.reporting.localization.Localization(__file__, 103, 17), empty_58359, *[tuple_58360], **kwargs_58371)
        
        # Getting the type of 'self' (line 103)
        self_58373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self')
        # Setting the type of the member 'K' of a type (line 103)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_58373, 'K', empty_call_result_58372)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _step_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_step_impl'
        module_type_store = module_type_store.open_function_context('_step_impl', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RungeKutta._step_impl.__dict__.__setitem__('stypy_localization', localization)
        RungeKutta._step_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RungeKutta._step_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        RungeKutta._step_impl.__dict__.__setitem__('stypy_function_name', 'RungeKutta._step_impl')
        RungeKutta._step_impl.__dict__.__setitem__('stypy_param_names_list', [])
        RungeKutta._step_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        RungeKutta._step_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RungeKutta._step_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        RungeKutta._step_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        RungeKutta._step_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RungeKutta._step_impl.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RungeKutta._step_impl', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_step_impl', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_step_impl(...)' code ##################

        
        # Assigning a Attribute to a Name (line 106):
        
        # Assigning a Attribute to a Name (line 106):
        # Getting the type of 'self' (line 106)
        self_58374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'self')
        # Obtaining the member 't' of a type (line 106)
        t_58375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), self_58374, 't')
        # Assigning a type to the variable 't' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 't', t_58375)
        
        # Assigning a Attribute to a Name (line 107):
        
        # Assigning a Attribute to a Name (line 107):
        # Getting the type of 'self' (line 107)
        self_58376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'self')
        # Obtaining the member 'y' of a type (line 107)
        y_58377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), self_58376, 'y')
        # Assigning a type to the variable 'y' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'y', y_58377)
        
        # Assigning a Attribute to a Name (line 109):
        
        # Assigning a Attribute to a Name (line 109):
        # Getting the type of 'self' (line 109)
        self_58378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'self')
        # Obtaining the member 'max_step' of a type (line 109)
        max_step_58379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), self_58378, 'max_step')
        # Assigning a type to the variable 'max_step' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'max_step', max_step_58379)
        
        # Assigning a Attribute to a Name (line 110):
        
        # Assigning a Attribute to a Name (line 110):
        # Getting the type of 'self' (line 110)
        self_58380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'self')
        # Obtaining the member 'rtol' of a type (line 110)
        rtol_58381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), self_58380, 'rtol')
        # Assigning a type to the variable 'rtol' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'rtol', rtol_58381)
        
        # Assigning a Attribute to a Name (line 111):
        
        # Assigning a Attribute to a Name (line 111):
        # Getting the type of 'self' (line 111)
        self_58382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'self')
        # Obtaining the member 'atol' of a type (line 111)
        atol_58383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), self_58382, 'atol')
        # Assigning a type to the variable 'atol' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'atol', atol_58383)
        
        # Assigning a BinOp to a Name (line 113):
        
        # Assigning a BinOp to a Name (line 113):
        int_58384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 19), 'int')
        
        # Call to abs(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to nextafter(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 't' (line 113)
        t_58389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 44), 't', False)
        # Getting the type of 'self' (line 113)
        self_58390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 47), 'self', False)
        # Obtaining the member 'direction' of a type (line 113)
        direction_58391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 47), self_58390, 'direction')
        # Getting the type of 'np' (line 113)
        np_58392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 64), 'np', False)
        # Obtaining the member 'inf' of a type (line 113)
        inf_58393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 64), np_58392, 'inf')
        # Applying the binary operator '*' (line 113)
        result_mul_58394 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 47), '*', direction_58391, inf_58393)
        
        # Processing the call keyword arguments (line 113)
        kwargs_58395 = {}
        # Getting the type of 'np' (line 113)
        np_58387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'np', False)
        # Obtaining the member 'nextafter' of a type (line 113)
        nextafter_58388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 31), np_58387, 'nextafter')
        # Calling nextafter(args, kwargs) (line 113)
        nextafter_call_result_58396 = invoke(stypy.reporting.localization.Localization(__file__, 113, 31), nextafter_58388, *[t_58389, result_mul_58394], **kwargs_58395)
        
        # Getting the type of 't' (line 113)
        t_58397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 74), 't', False)
        # Applying the binary operator '-' (line 113)
        result_sub_58398 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 31), '-', nextafter_call_result_58396, t_58397)
        
        # Processing the call keyword arguments (line 113)
        kwargs_58399 = {}
        # Getting the type of 'np' (line 113)
        np_58385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 113)
        abs_58386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 24), np_58385, 'abs')
        # Calling abs(args, kwargs) (line 113)
        abs_call_result_58400 = invoke(stypy.reporting.localization.Localization(__file__, 113, 24), abs_58386, *[result_sub_58398], **kwargs_58399)
        
        # Applying the binary operator '*' (line 113)
        result_mul_58401 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 19), '*', int_58384, abs_call_result_58400)
        
        # Assigning a type to the variable 'min_step' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'min_step', result_mul_58401)
        
        
        # Getting the type of 'self' (line 115)
        self_58402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'self')
        # Obtaining the member 'h_abs' of a type (line 115)
        h_abs_58403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 11), self_58402, 'h_abs')
        # Getting the type of 'max_step' (line 115)
        max_step_58404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 24), 'max_step')
        # Applying the binary operator '>' (line 115)
        result_gt_58405 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 11), '>', h_abs_58403, max_step_58404)
        
        # Testing the type of an if condition (line 115)
        if_condition_58406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 8), result_gt_58405)
        # Assigning a type to the variable 'if_condition_58406' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'if_condition_58406', if_condition_58406)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 116):
        
        # Assigning a Name to a Name (line 116):
        # Getting the type of 'max_step' (line 116)
        max_step_58407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'max_step')
        # Assigning a type to the variable 'h_abs' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'h_abs', max_step_58407)
        # SSA branch for the else part of an if statement (line 115)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 117)
        self_58408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'self')
        # Obtaining the member 'h_abs' of a type (line 117)
        h_abs_58409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 13), self_58408, 'h_abs')
        # Getting the type of 'min_step' (line 117)
        min_step_58410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 26), 'min_step')
        # Applying the binary operator '<' (line 117)
        result_lt_58411 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 13), '<', h_abs_58409, min_step_58410)
        
        # Testing the type of an if condition (line 117)
        if_condition_58412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 13), result_lt_58411)
        # Assigning a type to the variable 'if_condition_58412' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'if_condition_58412', if_condition_58412)
        # SSA begins for if statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 118):
        
        # Assigning a Name to a Name (line 118):
        # Getting the type of 'min_step' (line 118)
        min_step_58413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'min_step')
        # Assigning a type to the variable 'h_abs' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'h_abs', min_step_58413)
        # SSA branch for the else part of an if statement (line 117)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 120):
        
        # Assigning a Attribute to a Name (line 120):
        # Getting the type of 'self' (line 120)
        self_58414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'self')
        # Obtaining the member 'h_abs' of a type (line 120)
        h_abs_58415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 20), self_58414, 'h_abs')
        # Assigning a type to the variable 'h_abs' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'h_abs', h_abs_58415)
        # SSA join for if statement (line 117)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 122):
        
        # Assigning a Attribute to a Name (line 122):
        # Getting the type of 'self' (line 122)
        self_58416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'self')
        # Obtaining the member 'order' of a type (line 122)
        order_58417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), self_58416, 'order')
        # Assigning a type to the variable 'order' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'order', order_58417)
        
        # Assigning a Name to a Name (line 123):
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'False' (line 123)
        False_58418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'False')
        # Assigning a type to the variable 'step_accepted' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'step_accepted', False_58418)
        
        
        # Getting the type of 'step_accepted' (line 125)
        step_accepted_58419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 18), 'step_accepted')
        # Applying the 'not' unary operator (line 125)
        result_not__58420 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 14), 'not', step_accepted_58419)
        
        # Testing the type of an if condition (line 125)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_not__58420)
        # SSA begins for while statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # Getting the type of 'h_abs' (line 126)
        h_abs_58421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'h_abs')
        # Getting the type of 'min_step' (line 126)
        min_step_58422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'min_step')
        # Applying the binary operator '<' (line 126)
        result_lt_58423 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 15), '<', h_abs_58421, min_step_58422)
        
        # Testing the type of an if condition (line 126)
        if_condition_58424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 12), result_lt_58423)
        # Assigning a type to the variable 'if_condition_58424' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'if_condition_58424', if_condition_58424)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_58425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        # Getting the type of 'False' (line 127)
        False_58426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 23), tuple_58425, False_58426)
        # Adding element type (line 127)
        # Getting the type of 'self' (line 127)
        self_58427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'self')
        # Obtaining the member 'TOO_SMALL_STEP' of a type (line 127)
        TOO_SMALL_STEP_58428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), self_58427, 'TOO_SMALL_STEP')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 23), tuple_58425, TOO_SMALL_STEP_58428)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'stypy_return_type', tuple_58425)
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 129):
        
        # Assigning a BinOp to a Name (line 129):
        # Getting the type of 'h_abs' (line 129)
        h_abs_58429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'h_abs')
        # Getting the type of 'self' (line 129)
        self_58430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'self')
        # Obtaining the member 'direction' of a type (line 129)
        direction_58431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 24), self_58430, 'direction')
        # Applying the binary operator '*' (line 129)
        result_mul_58432 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 16), '*', h_abs_58429, direction_58431)
        
        # Assigning a type to the variable 'h' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'h', result_mul_58432)
        
        # Assigning a BinOp to a Name (line 130):
        
        # Assigning a BinOp to a Name (line 130):
        # Getting the type of 't' (line 130)
        t_58433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 't')
        # Getting the type of 'h' (line 130)
        h_58434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'h')
        # Applying the binary operator '+' (line 130)
        result_add_58435 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 20), '+', t_58433, h_58434)
        
        # Assigning a type to the variable 't_new' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 't_new', result_add_58435)
        
        
        # Getting the type of 'self' (line 132)
        self_58436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'self')
        # Obtaining the member 'direction' of a type (line 132)
        direction_58437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 15), self_58436, 'direction')
        # Getting the type of 't_new' (line 132)
        t_new_58438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 33), 't_new')
        # Getting the type of 'self' (line 132)
        self_58439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 41), 'self')
        # Obtaining the member 't_bound' of a type (line 132)
        t_bound_58440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 41), self_58439, 't_bound')
        # Applying the binary operator '-' (line 132)
        result_sub_58441 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 33), '-', t_new_58438, t_bound_58440)
        
        # Applying the binary operator '*' (line 132)
        result_mul_58442 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 15), '*', direction_58437, result_sub_58441)
        
        int_58443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 57), 'int')
        # Applying the binary operator '>' (line 132)
        result_gt_58444 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 15), '>', result_mul_58442, int_58443)
        
        # Testing the type of an if condition (line 132)
        if_condition_58445 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 12), result_gt_58444)
        # Assigning a type to the variable 'if_condition_58445' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'if_condition_58445', if_condition_58445)
        # SSA begins for if statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 133):
        
        # Assigning a Attribute to a Name (line 133):
        # Getting the type of 'self' (line 133)
        self_58446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'self')
        # Obtaining the member 't_bound' of a type (line 133)
        t_bound_58447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 24), self_58446, 't_bound')
        # Assigning a type to the variable 't_new' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 't_new', t_bound_58447)
        # SSA join for if statement (line 132)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 135):
        
        # Assigning a BinOp to a Name (line 135):
        # Getting the type of 't_new' (line 135)
        t_new_58448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 't_new')
        # Getting the type of 't' (line 135)
        t_58449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 't')
        # Applying the binary operator '-' (line 135)
        result_sub_58450 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 16), '-', t_new_58448, t_58449)
        
        # Assigning a type to the variable 'h' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'h', result_sub_58450)
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to abs(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'h' (line 136)
        h_58453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'h', False)
        # Processing the call keyword arguments (line 136)
        kwargs_58454 = {}
        # Getting the type of 'np' (line 136)
        np_58451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'np', False)
        # Obtaining the member 'abs' of a type (line 136)
        abs_58452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 20), np_58451, 'abs')
        # Calling abs(args, kwargs) (line 136)
        abs_call_result_58455 = invoke(stypy.reporting.localization.Localization(__file__, 136, 20), abs_58452, *[h_58453], **kwargs_58454)
        
        # Assigning a type to the variable 'h_abs' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'h_abs', abs_call_result_58455)
        
        # Assigning a Call to a Tuple (line 138):
        
        # Assigning a Subscript to a Name (line 138):
        
        # Obtaining the type of the subscript
        int_58456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 12), 'int')
        
        # Call to rk_step(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'self' (line 138)
        self_58458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 42), 'self', False)
        # Obtaining the member 'fun' of a type (line 138)
        fun_58459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 42), self_58458, 'fun')
        # Getting the type of 't' (line 138)
        t_58460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 52), 't', False)
        # Getting the type of 'y' (line 138)
        y_58461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 55), 'y', False)
        # Getting the type of 'self' (line 138)
        self_58462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 58), 'self', False)
        # Obtaining the member 'f' of a type (line 138)
        f_58463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 58), self_58462, 'f')
        # Getting the type of 'h' (line 138)
        h_58464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 66), 'h', False)
        # Getting the type of 'self' (line 138)
        self_58465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 69), 'self', False)
        # Obtaining the member 'A' of a type (line 138)
        A_58466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 69), self_58465, 'A')
        # Getting the type of 'self' (line 139)
        self_58467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 42), 'self', False)
        # Obtaining the member 'B' of a type (line 139)
        B_58468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 42), self_58467, 'B')
        # Getting the type of 'self' (line 139)
        self_58469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'self', False)
        # Obtaining the member 'C' of a type (line 139)
        C_58470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 50), self_58469, 'C')
        # Getting the type of 'self' (line 139)
        self_58471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 58), 'self', False)
        # Obtaining the member 'E' of a type (line 139)
        E_58472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 58), self_58471, 'E')
        # Getting the type of 'self' (line 139)
        self_58473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 66), 'self', False)
        # Obtaining the member 'K' of a type (line 139)
        K_58474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 66), self_58473, 'K')
        # Processing the call keyword arguments (line 138)
        kwargs_58475 = {}
        # Getting the type of 'rk_step' (line 138)
        rk_step_58457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 34), 'rk_step', False)
        # Calling rk_step(args, kwargs) (line 138)
        rk_step_call_result_58476 = invoke(stypy.reporting.localization.Localization(__file__, 138, 34), rk_step_58457, *[fun_58459, t_58460, y_58461, f_58463, h_58464, A_58466, B_58468, C_58470, E_58472, K_58474], **kwargs_58475)
        
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___58477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), rk_step_call_result_58476, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_58478 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), getitem___58477, int_58456)
        
        # Assigning a type to the variable 'tuple_var_assignment_58178' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'tuple_var_assignment_58178', subscript_call_result_58478)
        
        # Assigning a Subscript to a Name (line 138):
        
        # Obtaining the type of the subscript
        int_58479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 12), 'int')
        
        # Call to rk_step(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'self' (line 138)
        self_58481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 42), 'self', False)
        # Obtaining the member 'fun' of a type (line 138)
        fun_58482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 42), self_58481, 'fun')
        # Getting the type of 't' (line 138)
        t_58483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 52), 't', False)
        # Getting the type of 'y' (line 138)
        y_58484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 55), 'y', False)
        # Getting the type of 'self' (line 138)
        self_58485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 58), 'self', False)
        # Obtaining the member 'f' of a type (line 138)
        f_58486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 58), self_58485, 'f')
        # Getting the type of 'h' (line 138)
        h_58487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 66), 'h', False)
        # Getting the type of 'self' (line 138)
        self_58488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 69), 'self', False)
        # Obtaining the member 'A' of a type (line 138)
        A_58489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 69), self_58488, 'A')
        # Getting the type of 'self' (line 139)
        self_58490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 42), 'self', False)
        # Obtaining the member 'B' of a type (line 139)
        B_58491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 42), self_58490, 'B')
        # Getting the type of 'self' (line 139)
        self_58492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'self', False)
        # Obtaining the member 'C' of a type (line 139)
        C_58493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 50), self_58492, 'C')
        # Getting the type of 'self' (line 139)
        self_58494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 58), 'self', False)
        # Obtaining the member 'E' of a type (line 139)
        E_58495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 58), self_58494, 'E')
        # Getting the type of 'self' (line 139)
        self_58496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 66), 'self', False)
        # Obtaining the member 'K' of a type (line 139)
        K_58497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 66), self_58496, 'K')
        # Processing the call keyword arguments (line 138)
        kwargs_58498 = {}
        # Getting the type of 'rk_step' (line 138)
        rk_step_58480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 34), 'rk_step', False)
        # Calling rk_step(args, kwargs) (line 138)
        rk_step_call_result_58499 = invoke(stypy.reporting.localization.Localization(__file__, 138, 34), rk_step_58480, *[fun_58482, t_58483, y_58484, f_58486, h_58487, A_58489, B_58491, C_58493, E_58495, K_58497], **kwargs_58498)
        
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___58500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), rk_step_call_result_58499, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_58501 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), getitem___58500, int_58479)
        
        # Assigning a type to the variable 'tuple_var_assignment_58179' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'tuple_var_assignment_58179', subscript_call_result_58501)
        
        # Assigning a Subscript to a Name (line 138):
        
        # Obtaining the type of the subscript
        int_58502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 12), 'int')
        
        # Call to rk_step(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'self' (line 138)
        self_58504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 42), 'self', False)
        # Obtaining the member 'fun' of a type (line 138)
        fun_58505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 42), self_58504, 'fun')
        # Getting the type of 't' (line 138)
        t_58506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 52), 't', False)
        # Getting the type of 'y' (line 138)
        y_58507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 55), 'y', False)
        # Getting the type of 'self' (line 138)
        self_58508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 58), 'self', False)
        # Obtaining the member 'f' of a type (line 138)
        f_58509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 58), self_58508, 'f')
        # Getting the type of 'h' (line 138)
        h_58510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 66), 'h', False)
        # Getting the type of 'self' (line 138)
        self_58511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 69), 'self', False)
        # Obtaining the member 'A' of a type (line 138)
        A_58512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 69), self_58511, 'A')
        # Getting the type of 'self' (line 139)
        self_58513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 42), 'self', False)
        # Obtaining the member 'B' of a type (line 139)
        B_58514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 42), self_58513, 'B')
        # Getting the type of 'self' (line 139)
        self_58515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'self', False)
        # Obtaining the member 'C' of a type (line 139)
        C_58516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 50), self_58515, 'C')
        # Getting the type of 'self' (line 139)
        self_58517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 58), 'self', False)
        # Obtaining the member 'E' of a type (line 139)
        E_58518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 58), self_58517, 'E')
        # Getting the type of 'self' (line 139)
        self_58519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 66), 'self', False)
        # Obtaining the member 'K' of a type (line 139)
        K_58520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 66), self_58519, 'K')
        # Processing the call keyword arguments (line 138)
        kwargs_58521 = {}
        # Getting the type of 'rk_step' (line 138)
        rk_step_58503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 34), 'rk_step', False)
        # Calling rk_step(args, kwargs) (line 138)
        rk_step_call_result_58522 = invoke(stypy.reporting.localization.Localization(__file__, 138, 34), rk_step_58503, *[fun_58505, t_58506, y_58507, f_58509, h_58510, A_58512, B_58514, C_58516, E_58518, K_58520], **kwargs_58521)
        
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___58523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), rk_step_call_result_58522, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_58524 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), getitem___58523, int_58502)
        
        # Assigning a type to the variable 'tuple_var_assignment_58180' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'tuple_var_assignment_58180', subscript_call_result_58524)
        
        # Assigning a Name to a Name (line 138):
        # Getting the type of 'tuple_var_assignment_58178' (line 138)
        tuple_var_assignment_58178_58525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'tuple_var_assignment_58178')
        # Assigning a type to the variable 'y_new' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'y_new', tuple_var_assignment_58178_58525)
        
        # Assigning a Name to a Name (line 138):
        # Getting the type of 'tuple_var_assignment_58179' (line 138)
        tuple_var_assignment_58179_58526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'tuple_var_assignment_58179')
        # Assigning a type to the variable 'f_new' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'f_new', tuple_var_assignment_58179_58526)
        
        # Assigning a Name to a Name (line 138):
        # Getting the type of 'tuple_var_assignment_58180' (line 138)
        tuple_var_assignment_58180_58527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'tuple_var_assignment_58180')
        # Assigning a type to the variable 'error' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 26), 'error', tuple_var_assignment_58180_58527)
        
        # Assigning a BinOp to a Name (line 140):
        
        # Assigning a BinOp to a Name (line 140):
        # Getting the type of 'atol' (line 140)
        atol_58528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'atol')
        
        # Call to maximum(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Call to abs(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'y' (line 140)
        y_58533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 45), 'y', False)
        # Processing the call keyword arguments (line 140)
        kwargs_58534 = {}
        # Getting the type of 'np' (line 140)
        np_58531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 38), 'np', False)
        # Obtaining the member 'abs' of a type (line 140)
        abs_58532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 38), np_58531, 'abs')
        # Calling abs(args, kwargs) (line 140)
        abs_call_result_58535 = invoke(stypy.reporting.localization.Localization(__file__, 140, 38), abs_58532, *[y_58533], **kwargs_58534)
        
        
        # Call to abs(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'y_new' (line 140)
        y_new_58538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 56), 'y_new', False)
        # Processing the call keyword arguments (line 140)
        kwargs_58539 = {}
        # Getting the type of 'np' (line 140)
        np_58536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 49), 'np', False)
        # Obtaining the member 'abs' of a type (line 140)
        abs_58537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 49), np_58536, 'abs')
        # Calling abs(args, kwargs) (line 140)
        abs_call_result_58540 = invoke(stypy.reporting.localization.Localization(__file__, 140, 49), abs_58537, *[y_new_58538], **kwargs_58539)
        
        # Processing the call keyword arguments (line 140)
        kwargs_58541 = {}
        # Getting the type of 'np' (line 140)
        np_58529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 27), 'np', False)
        # Obtaining the member 'maximum' of a type (line 140)
        maximum_58530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 27), np_58529, 'maximum')
        # Calling maximum(args, kwargs) (line 140)
        maximum_call_result_58542 = invoke(stypy.reporting.localization.Localization(__file__, 140, 27), maximum_58530, *[abs_call_result_58535, abs_call_result_58540], **kwargs_58541)
        
        # Getting the type of 'rtol' (line 140)
        rtol_58543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 66), 'rtol')
        # Applying the binary operator '*' (line 140)
        result_mul_58544 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 27), '*', maximum_call_result_58542, rtol_58543)
        
        # Applying the binary operator '+' (line 140)
        result_add_58545 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 20), '+', atol_58528, result_mul_58544)
        
        # Assigning a type to the variable 'scale' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'scale', result_add_58545)
        
        # Assigning a Call to a Name (line 141):
        
        # Assigning a Call to a Name (line 141):
        
        # Call to norm(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'error' (line 141)
        error_58547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 30), 'error', False)
        # Getting the type of 'scale' (line 141)
        scale_58548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 38), 'scale', False)
        # Applying the binary operator 'div' (line 141)
        result_div_58549 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 30), 'div', error_58547, scale_58548)
        
        # Processing the call keyword arguments (line 141)
        kwargs_58550 = {}
        # Getting the type of 'norm' (line 141)
        norm_58546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'norm', False)
        # Calling norm(args, kwargs) (line 141)
        norm_call_result_58551 = invoke(stypy.reporting.localization.Localization(__file__, 141, 25), norm_58546, *[result_div_58549], **kwargs_58550)
        
        # Assigning a type to the variable 'error_norm' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'error_norm', norm_call_result_58551)
        
        
        # Getting the type of 'error_norm' (line 143)
        error_norm_58552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'error_norm')
        int_58553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 28), 'int')
        # Applying the binary operator '<' (line 143)
        result_lt_58554 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 15), '<', error_norm_58552, int_58553)
        
        # Testing the type of an if condition (line 143)
        if_condition_58555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 12), result_lt_58554)
        # Assigning a type to the variable 'if_condition_58555' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'if_condition_58555', if_condition_58555)
        # SSA begins for if statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'h_abs' (line 144)
        h_abs_58556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'h_abs')
        
        # Call to min(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'MAX_FACTOR' (line 144)
        MAX_FACTOR_58558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 29), 'MAX_FACTOR', False)
        
        # Call to max(...): (line 145)
        # Processing the call arguments (line 145)
        int_58560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 33), 'int')
        # Getting the type of 'SAFETY' (line 145)
        SAFETY_58561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 36), 'SAFETY', False)
        # Getting the type of 'error_norm' (line 145)
        error_norm_58562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 45), 'error_norm', False)
        int_58563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 60), 'int')
        # Getting the type of 'order' (line 145)
        order_58564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 66), 'order', False)
        int_58565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 74), 'int')
        # Applying the binary operator '+' (line 145)
        result_add_58566 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 66), '+', order_58564, int_58565)
        
        # Applying the binary operator 'div' (line 145)
        result_div_58567 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 60), 'div', int_58563, result_add_58566)
        
        # Applying the binary operator '**' (line 145)
        result_pow_58568 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 45), '**', error_norm_58562, result_div_58567)
        
        # Applying the binary operator '*' (line 145)
        result_mul_58569 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 36), '*', SAFETY_58561, result_pow_58568)
        
        # Processing the call keyword arguments (line 145)
        kwargs_58570 = {}
        # Getting the type of 'max' (line 145)
        max_58559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 29), 'max', False)
        # Calling max(args, kwargs) (line 145)
        max_call_result_58571 = invoke(stypy.reporting.localization.Localization(__file__, 145, 29), max_58559, *[int_58560, result_mul_58569], **kwargs_58570)
        
        # Processing the call keyword arguments (line 144)
        kwargs_58572 = {}
        # Getting the type of 'min' (line 144)
        min_58557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'min', False)
        # Calling min(args, kwargs) (line 144)
        min_call_result_58573 = invoke(stypy.reporting.localization.Localization(__file__, 144, 25), min_58557, *[MAX_FACTOR_58558, max_call_result_58571], **kwargs_58572)
        
        # Applying the binary operator '*=' (line 144)
        result_imul_58574 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 16), '*=', h_abs_58556, min_call_result_58573)
        # Assigning a type to the variable 'h_abs' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'h_abs', result_imul_58574)
        
        
        # Assigning a Name to a Name (line 146):
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'True' (line 146)
        True_58575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 32), 'True')
        # Assigning a type to the variable 'step_accepted' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'step_accepted', True_58575)
        # SSA branch for the else part of an if statement (line 143)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'h_abs' (line 148)
        h_abs_58576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'h_abs')
        
        # Call to max(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'MIN_FACTOR' (line 148)
        MIN_FACTOR_58578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 29), 'MIN_FACTOR', False)
        # Getting the type of 'SAFETY' (line 149)
        SAFETY_58579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 29), 'SAFETY', False)
        # Getting the type of 'error_norm' (line 149)
        error_norm_58580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'error_norm', False)
        int_58581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 53), 'int')
        # Getting the type of 'order' (line 149)
        order_58582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 59), 'order', False)
        int_58583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 67), 'int')
        # Applying the binary operator '+' (line 149)
        result_add_58584 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 59), '+', order_58582, int_58583)
        
        # Applying the binary operator 'div' (line 149)
        result_div_58585 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 53), 'div', int_58581, result_add_58584)
        
        # Applying the binary operator '**' (line 149)
        result_pow_58586 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 38), '**', error_norm_58580, result_div_58585)
        
        # Applying the binary operator '*' (line 149)
        result_mul_58587 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 29), '*', SAFETY_58579, result_pow_58586)
        
        # Processing the call keyword arguments (line 148)
        kwargs_58588 = {}
        # Getting the type of 'max' (line 148)
        max_58577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'max', False)
        # Calling max(args, kwargs) (line 148)
        max_call_result_58589 = invoke(stypy.reporting.localization.Localization(__file__, 148, 25), max_58577, *[MIN_FACTOR_58578, result_mul_58587], **kwargs_58588)
        
        # Applying the binary operator '*=' (line 148)
        result_imul_58590 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 16), '*=', h_abs_58576, max_call_result_58589)
        # Assigning a type to the variable 'h_abs' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'h_abs', result_imul_58590)
        
        # SSA join for if statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 151):
        
        # Assigning a Name to a Attribute (line 151):
        # Getting the type of 'y' (line 151)
        y_58591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'y')
        # Getting the type of 'self' (line 151)
        self_58592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self')
        # Setting the type of the member 'y_old' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_58592, 'y_old', y_58591)
        
        # Assigning a Name to a Attribute (line 153):
        
        # Assigning a Name to a Attribute (line 153):
        # Getting the type of 't_new' (line 153)
        t_new_58593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 't_new')
        # Getting the type of 'self' (line 153)
        self_58594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'self')
        # Setting the type of the member 't' of a type (line 153)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), self_58594, 't', t_new_58593)
        
        # Assigning a Name to a Attribute (line 154):
        
        # Assigning a Name to a Attribute (line 154):
        # Getting the type of 'y_new' (line 154)
        y_new_58595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'y_new')
        # Getting the type of 'self' (line 154)
        self_58596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self')
        # Setting the type of the member 'y' of a type (line 154)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_58596, 'y', y_new_58595)
        
        # Assigning a Name to a Attribute (line 156):
        
        # Assigning a Name to a Attribute (line 156):
        # Getting the type of 'h_abs' (line 156)
        h_abs_58597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'h_abs')
        # Getting the type of 'self' (line 156)
        self_58598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'self')
        # Setting the type of the member 'h_abs' of a type (line 156)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), self_58598, 'h_abs', h_abs_58597)
        
        # Assigning a Name to a Attribute (line 157):
        
        # Assigning a Name to a Attribute (line 157):
        # Getting the type of 'f_new' (line 157)
        f_new_58599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'f_new')
        # Getting the type of 'self' (line 157)
        self_58600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member 'f' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_58600, 'f', f_new_58599)
        
        # Obtaining an instance of the builtin type 'tuple' (line 159)
        tuple_58601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 159)
        # Adding element type (line 159)
        # Getting the type of 'True' (line 159)
        True_58602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 15), tuple_58601, True_58602)
        # Adding element type (line 159)
        # Getting the type of 'None' (line 159)
        None_58603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 21), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 15), tuple_58601, None_58603)
        
        # Assigning a type to the variable 'stypy_return_type' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type', tuple_58601)
        
        # ################# End of '_step_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_step_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_58604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58604)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_step_impl'
        return stypy_return_type_58604


    @norecursion
    def _dense_output_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dense_output_impl'
        module_type_store = module_type_store.open_function_context('_dense_output_impl', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RungeKutta._dense_output_impl.__dict__.__setitem__('stypy_localization', localization)
        RungeKutta._dense_output_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RungeKutta._dense_output_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        RungeKutta._dense_output_impl.__dict__.__setitem__('stypy_function_name', 'RungeKutta._dense_output_impl')
        RungeKutta._dense_output_impl.__dict__.__setitem__('stypy_param_names_list', [])
        RungeKutta._dense_output_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        RungeKutta._dense_output_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RungeKutta._dense_output_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        RungeKutta._dense_output_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        RungeKutta._dense_output_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RungeKutta._dense_output_impl.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RungeKutta._dense_output_impl', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dense_output_impl', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dense_output_impl(...)' code ##################

        
        # Assigning a Call to a Name (line 162):
        
        # Assigning a Call to a Name (line 162):
        
        # Call to dot(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'self' (line 162)
        self_58609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'self', False)
        # Obtaining the member 'P' of a type (line 162)
        P_58610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 25), self_58609, 'P')
        # Processing the call keyword arguments (line 162)
        kwargs_58611 = {}
        # Getting the type of 'self' (line 162)
        self_58605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'self', False)
        # Obtaining the member 'K' of a type (line 162)
        K_58606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), self_58605, 'K')
        # Obtaining the member 'T' of a type (line 162)
        T_58607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), K_58606, 'T')
        # Obtaining the member 'dot' of a type (line 162)
        dot_58608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), T_58607, 'dot')
        # Calling dot(args, kwargs) (line 162)
        dot_call_result_58612 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), dot_58608, *[P_58610], **kwargs_58611)
        
        # Assigning a type to the variable 'Q' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'Q', dot_call_result_58612)
        
        # Call to RkDenseOutput(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'self' (line 163)
        self_58614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'self', False)
        # Obtaining the member 't_old' of a type (line 163)
        t_old_58615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 29), self_58614, 't_old')
        # Getting the type of 'self' (line 163)
        self_58616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 41), 'self', False)
        # Obtaining the member 't' of a type (line 163)
        t_58617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 41), self_58616, 't')
        # Getting the type of 'self' (line 163)
        self_58618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 49), 'self', False)
        # Obtaining the member 'y_old' of a type (line 163)
        y_old_58619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 49), self_58618, 'y_old')
        # Getting the type of 'Q' (line 163)
        Q_58620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 61), 'Q', False)
        # Processing the call keyword arguments (line 163)
        kwargs_58621 = {}
        # Getting the type of 'RkDenseOutput' (line 163)
        RkDenseOutput_58613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'RkDenseOutput', False)
        # Calling RkDenseOutput(args, kwargs) (line 163)
        RkDenseOutput_call_result_58622 = invoke(stypy.reporting.localization.Localization(__file__, 163, 15), RkDenseOutput_58613, *[t_old_58615, t_58617, y_old_58619, Q_58620], **kwargs_58621)
        
        # Assigning a type to the variable 'stypy_return_type' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'stypy_return_type', RkDenseOutput_call_result_58622)
        
        # ################# End of '_dense_output_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dense_output_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_58623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_58623)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dense_output_impl'
        return stypy_return_type_58623


# Assigning a type to the variable 'RungeKutta' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'RungeKutta', RungeKutta)

# Assigning a Name to a Name (line 83):
# Getting the type of 'NotImplemented' (line 83)
NotImplemented_58624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'NotImplemented')
# Getting the type of 'RungeKutta'
RungeKutta_58625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RungeKutta')
# Setting the type of the member 'C' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RungeKutta_58625, 'C', NotImplemented_58624)

# Assigning a Name to a Name (line 84):
# Getting the type of 'NotImplemented' (line 84)
NotImplemented_58626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'NotImplemented')
# Getting the type of 'RungeKutta'
RungeKutta_58627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RungeKutta')
# Setting the type of the member 'A' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RungeKutta_58627, 'A', NotImplemented_58626)

# Assigning a Name to a Name (line 85):
# Getting the type of 'NotImplemented' (line 85)
NotImplemented_58628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'NotImplemented')
# Getting the type of 'RungeKutta'
RungeKutta_58629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RungeKutta')
# Setting the type of the member 'B' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RungeKutta_58629, 'B', NotImplemented_58628)

# Assigning a Name to a Name (line 86):
# Getting the type of 'NotImplemented' (line 86)
NotImplemented_58630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'NotImplemented')
# Getting the type of 'RungeKutta'
RungeKutta_58631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RungeKutta')
# Setting the type of the member 'E' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RungeKutta_58631, 'E', NotImplemented_58630)

# Assigning a Name to a Name (line 87):
# Getting the type of 'NotImplemented' (line 87)
NotImplemented_58632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'NotImplemented')
# Getting the type of 'RungeKutta'
RungeKutta_58633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RungeKutta')
# Setting the type of the member 'P' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RungeKutta_58633, 'P', NotImplemented_58632)

# Assigning a Name to a Name (line 88):
# Getting the type of 'NotImplemented' (line 88)
NotImplemented_58634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'NotImplemented')
# Getting the type of 'RungeKutta'
RungeKutta_58635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RungeKutta')
# Setting the type of the member 'order' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RungeKutta_58635, 'order', NotImplemented_58634)

# Assigning a Name to a Name (line 89):
# Getting the type of 'NotImplemented' (line 89)
NotImplemented_58636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'NotImplemented')
# Getting the type of 'RungeKutta'
RungeKutta_58637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RungeKutta')
# Setting the type of the member 'n_stages' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RungeKutta_58637, 'n_stages', NotImplemented_58636)
# Declaration of the 'RK23' class
# Getting the type of 'RungeKutta' (line 166)
RungeKutta_58638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'RungeKutta')

class RK23(RungeKutta_58638, ):
    str_58639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, (-1)), 'str', 'Explicit Runge-Kutta method of order 3(2).\n\n    The Bogacki-Shamping pair of formulas is used [1]_. The error is controlled\n    assuming 2nd order accuracy, but steps are taken using a 3rd oder accurate\n    formula (local extrapolation is done). A cubic Hermit polynomial is used\n    for the dense output.\n\n    Can be applied in a complex domain.\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system. The calling signature is ``fun(t, y)``.\n        Here ``t`` is a scalar and there are two options for ndarray ``y``.\n        It can either have shape (n,), then ``fun`` must return array_like with\n        shape (n,). Or alternatively it can have shape (n, k), then ``fun``\n        must return array_like with shape (n, k), i.e. each column\n        corresponds to a single column in ``y``. The choice between the two\n        options is determined by `vectorized` argument (see below). The\n        vectorized implementation allows faster approximation of the Jacobian\n        by finite differences.\n    t0 : float\n        Initial time.\n    y0 : array_like, shape (n,)\n        Initial state.\n    t_bound : float\n        Boundary time --- the integration won\'t continue beyond it. It also\n        determines the direction of the integration.\n    max_step : float, optional\n        Maximum allowed step size. Default is np.inf, i.e. the step is not\n        bounded and determined solely by the solver.\n    rtol, atol : float and array_like, optional\n        Relative and absolute tolerances. The solver keeps the local error\n        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a\n        relative accuracy (number of correct digits). But if a component of `y`\n        is approximately below `atol` then the error only needs to fall within\n        the same `atol` threshold, and the number of correct digits is not\n        guaranteed. If components of y have different scales, it might be\n        beneficial to set different `atol` values for different components by\n        passing array_like with shape (n,) for `atol`. Default values are\n        1e-3 for `rtol` and 1e-6 for `atol`.\n    vectorized : bool, optional\n        Whether `fun` is implemented in a vectorized fashion. Default is False.\n\n    Attributes\n    ----------\n    n : int\n        Number of equations.\n    status : string\n        Current status of the solver: \'running\', \'finished\' or \'failed\'.\n    t_bound : float\n        Boundary time.\n    direction : float\n        Integration direction: +1 or -1.\n    t : float\n        Current time.\n    y : ndarray\n        Current state.\n    t_old : float\n        Previous time. None if no steps were made yet.\n    step_size : float\n        Size of the last successful step. None if no steps were made yet.\n    nfev : int\n        Number of the system\'s rhs evaluations.\n    njev : int\n        Number of the Jacobian evaluations.\n    nlu : int\n        Number of LU decompositions.\n\n    References\n    ----------\n    .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",\n           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.\n    ')
    
    # Assigning a Num to a Name (line 241):
    
    # Assigning a Num to a Name (line 242):
    
    # Assigning a Call to a Name (line 243):
    
    # Assigning a List to a Name (line 244):
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 247):
    
    # Assigning a Call to a Name (line 248):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 166, 0, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RK23.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'RK23' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'RK23', RK23)

# Assigning a Num to a Name (line 241):
int_58640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 12), 'int')
# Getting the type of 'RK23'
RK23_58641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK23')
# Setting the type of the member 'order' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK23_58641, 'order', int_58640)

# Assigning a Num to a Name (line 242):
int_58642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 15), 'int')
# Getting the type of 'RK23'
RK23_58643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK23')
# Setting the type of the member 'n_stages' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK23_58643, 'n_stages', int_58642)

# Assigning a Call to a Name (line 243):

# Call to array(...): (line 243)
# Processing the call arguments (line 243)

# Obtaining an instance of the builtin type 'list' (line 243)
list_58646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 243)
# Adding element type (line 243)
int_58647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 18), 'int')
int_58648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 20), 'int')
# Applying the binary operator 'div' (line 243)
result_div_58649 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 18), 'div', int_58647, int_58648)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 17), list_58646, result_div_58649)
# Adding element type (line 243)
int_58650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 23), 'int')
int_58651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 25), 'int')
# Applying the binary operator 'div' (line 243)
result_div_58652 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 23), 'div', int_58650, int_58651)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 17), list_58646, result_div_58652)

# Processing the call keyword arguments (line 243)
kwargs_58653 = {}
# Getting the type of 'np' (line 243)
np_58644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'np', False)
# Obtaining the member 'array' of a type (line 243)
array_58645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), np_58644, 'array')
# Calling array(args, kwargs) (line 243)
array_call_result_58654 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), array_58645, *[list_58646], **kwargs_58653)

# Getting the type of 'RK23'
RK23_58655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK23')
# Setting the type of the member 'C' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK23_58655, 'C', array_call_result_58654)

# Assigning a List to a Name (line 244):

# Obtaining an instance of the builtin type 'list' (line 244)
list_58656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 244)
# Adding element type (line 244)

# Call to array(...): (line 244)
# Processing the call arguments (line 244)

# Obtaining an instance of the builtin type 'list' (line 244)
list_58659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 244)
# Adding element type (line 244)
int_58660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 19), 'int')
int_58661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 21), 'int')
# Applying the binary operator 'div' (line 244)
result_div_58662 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 19), 'div', int_58660, int_58661)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 18), list_58659, result_div_58662)

# Processing the call keyword arguments (line 244)
kwargs_58663 = {}
# Getting the type of 'np' (line 244)
np_58657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 9), 'np', False)
# Obtaining the member 'array' of a type (line 244)
array_58658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 9), np_58657, 'array')
# Calling array(args, kwargs) (line 244)
array_call_result_58664 = invoke(stypy.reporting.localization.Localization(__file__, 244, 9), array_58658, *[list_58659], **kwargs_58663)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 8), list_58656, array_call_result_58664)
# Adding element type (line 244)

# Call to array(...): (line 245)
# Processing the call arguments (line 245)

# Obtaining an instance of the builtin type 'list' (line 245)
list_58667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 245)
# Adding element type (line 245)
int_58668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 18), list_58667, int_58668)
# Adding element type (line 245)
int_58669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 22), 'int')
int_58670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 24), 'int')
# Applying the binary operator 'div' (line 245)
result_div_58671 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 22), 'div', int_58669, int_58670)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 18), list_58667, result_div_58671)

# Processing the call keyword arguments (line 245)
kwargs_58672 = {}
# Getting the type of 'np' (line 245)
np_58665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 9), 'np', False)
# Obtaining the member 'array' of a type (line 245)
array_58666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 9), np_58665, 'array')
# Calling array(args, kwargs) (line 245)
array_call_result_58673 = invoke(stypy.reporting.localization.Localization(__file__, 245, 9), array_58666, *[list_58667], **kwargs_58672)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 8), list_58656, array_call_result_58673)

# Getting the type of 'RK23'
RK23_58674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK23')
# Setting the type of the member 'A' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK23_58674, 'A', list_58656)

# Assigning a Call to a Name (line 246):

# Call to array(...): (line 246)
# Processing the call arguments (line 246)

# Obtaining an instance of the builtin type 'list' (line 246)
list_58677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 246)
# Adding element type (line 246)
int_58678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 18), 'int')
int_58679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 20), 'int')
# Applying the binary operator 'div' (line 246)
result_div_58680 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 18), 'div', int_58678, int_58679)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 17), list_58677, result_div_58680)
# Adding element type (line 246)
int_58681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 23), 'int')
int_58682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 25), 'int')
# Applying the binary operator 'div' (line 246)
result_div_58683 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 23), 'div', int_58681, int_58682)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 17), list_58677, result_div_58683)
# Adding element type (line 246)
int_58684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 28), 'int')
int_58685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 30), 'int')
# Applying the binary operator 'div' (line 246)
result_div_58686 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 28), 'div', int_58684, int_58685)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 17), list_58677, result_div_58686)

# Processing the call keyword arguments (line 246)
kwargs_58687 = {}
# Getting the type of 'np' (line 246)
np_58675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'np', False)
# Obtaining the member 'array' of a type (line 246)
array_58676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), np_58675, 'array')
# Calling array(args, kwargs) (line 246)
array_call_result_58688 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), array_58676, *[list_58677], **kwargs_58687)

# Getting the type of 'RK23'
RK23_58689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK23')
# Setting the type of the member 'B' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK23_58689, 'B', array_call_result_58688)

# Assigning a Call to a Name (line 247):

# Call to array(...): (line 247)
# Processing the call arguments (line 247)

# Obtaining an instance of the builtin type 'list' (line 247)
list_58692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 247)
# Adding element type (line 247)
int_58693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 18), 'int')
int_58694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 20), 'int')
# Applying the binary operator 'div' (line 247)
result_div_58695 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 18), 'div', int_58693, int_58694)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 17), list_58692, result_div_58695)
# Adding element type (line 247)
int_58696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 24), 'int')
int_58697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 27), 'int')
# Applying the binary operator 'div' (line 247)
result_div_58698 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 24), 'div', int_58696, int_58697)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 17), list_58692, result_div_58698)
# Adding element type (line 247)
int_58699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 31), 'int')
int_58700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 34), 'int')
# Applying the binary operator 'div' (line 247)
result_div_58701 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 31), 'div', int_58699, int_58700)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 17), list_58692, result_div_58701)
# Adding element type (line 247)
int_58702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 37), 'int')
int_58703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 39), 'int')
# Applying the binary operator 'div' (line 247)
result_div_58704 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 37), 'div', int_58702, int_58703)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 17), list_58692, result_div_58704)

# Processing the call keyword arguments (line 247)
kwargs_58705 = {}
# Getting the type of 'np' (line 247)
np_58690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'np', False)
# Obtaining the member 'array' of a type (line 247)
array_58691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), np_58690, 'array')
# Calling array(args, kwargs) (line 247)
array_call_result_58706 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), array_58691, *[list_58692], **kwargs_58705)

# Getting the type of 'RK23'
RK23_58707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK23')
# Setting the type of the member 'E' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK23_58707, 'E', array_call_result_58706)

# Assigning a Call to a Name (line 248):

# Call to array(...): (line 248)
# Processing the call arguments (line 248)

# Obtaining an instance of the builtin type 'list' (line 248)
list_58710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 248)
# Adding element type (line 248)

# Obtaining an instance of the builtin type 'list' (line 248)
list_58711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 248)
# Adding element type (line 248)
int_58712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 18), list_58711, int_58712)
# Adding element type (line 248)
int_58713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 22), 'int')
int_58714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 27), 'int')
# Applying the binary operator 'div' (line 248)
result_div_58715 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 22), 'div', int_58713, int_58714)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 18), list_58711, result_div_58715)
# Adding element type (line 248)
int_58716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 30), 'int')
int_58717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 34), 'int')
# Applying the binary operator 'div' (line 248)
result_div_58718 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 30), 'div', int_58716, int_58717)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 18), list_58711, result_div_58718)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 17), list_58710, list_58711)
# Adding element type (line 248)

# Obtaining an instance of the builtin type 'list' (line 249)
list_58719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 249)
# Adding element type (line 249)
int_58720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), list_58719, int_58720)
# Adding element type (line 249)
int_58721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), list_58719, int_58721)
# Adding element type (line 249)
int_58722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 25), 'int')
int_58723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 28), 'int')
# Applying the binary operator 'div' (line 249)
result_div_58724 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 25), 'div', int_58722, int_58723)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), list_58719, result_div_58724)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 17), list_58710, list_58719)
# Adding element type (line 248)

# Obtaining an instance of the builtin type 'list' (line 250)
list_58725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 250)
# Adding element type (line 250)
int_58726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 18), list_58725, int_58726)
# Adding element type (line 250)
int_58727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 22), 'int')
int_58728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 24), 'int')
# Applying the binary operator 'div' (line 250)
result_div_58729 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 22), 'div', int_58727, int_58728)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 18), list_58725, result_div_58729)
# Adding element type (line 250)
int_58730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 27), 'int')
int_58731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 30), 'int')
# Applying the binary operator 'div' (line 250)
result_div_58732 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 27), 'div', int_58730, int_58731)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 18), list_58725, result_div_58732)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 17), list_58710, list_58725)
# Adding element type (line 248)

# Obtaining an instance of the builtin type 'list' (line 251)
list_58733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 251)
# Adding element type (line 251)
int_58734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 18), list_58733, int_58734)
# Adding element type (line 251)
int_58735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 18), list_58733, int_58735)
# Adding element type (line 251)
int_58736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 18), list_58733, int_58736)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 17), list_58710, list_58733)

# Processing the call keyword arguments (line 248)
kwargs_58737 = {}
# Getting the type of 'np' (line 248)
np_58708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'np', False)
# Obtaining the member 'array' of a type (line 248)
array_58709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), np_58708, 'array')
# Calling array(args, kwargs) (line 248)
array_call_result_58738 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), array_58709, *[list_58710], **kwargs_58737)

# Getting the type of 'RK23'
RK23_58739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK23')
# Setting the type of the member 'P' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK23_58739, 'P', array_call_result_58738)
# Declaration of the 'RK45' class
# Getting the type of 'RungeKutta' (line 254)
RungeKutta_58740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), 'RungeKutta')

class RK45(RungeKutta_58740, ):
    str_58741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, (-1)), 'str', 'Explicit Runge-Kutta method of order 5(4).\n\n    The Dormand-Prince pair of formulas is used [1]_. The error is controlled\n    assuming 4th order accuracy, but steps are taken using a 5th\n    oder accurate formula (local extrapolation is done). A quartic\n    interpolation polynomial is used for the dense output [2]_.\n\n    Can be applied in a complex domain.\n\n    Parameters\n    ----------\n    fun : callable\n        Right-hand side of the system. The calling signature is ``fun(t, y)``.\n        Here ``t`` is a scalar and there are two options for ndarray ``y``.\n        It can either have shape (n,), then ``fun`` must return array_like with\n        shape (n,). Or alternatively it can have shape (n, k), then ``fun``\n        must return array_like with shape (n, k), i.e. each column\n        corresponds to a single column in ``y``. The choice between the two\n        options is determined by `vectorized` argument (see below). The\n        vectorized implementation allows faster approximation of the Jacobian\n        by finite differences.\n    t0 : float\n        Initial value of the independent variable.\n    y0 : array_like, shape (n,)\n        Initial values of the dependent variable.\n    t_bound : float\n        Boundary time --- the integration won\'t continue beyond it. It also\n        determines the direction of the integration.\n    max_step : float, optional\n        Maximum allowed step size. Default is np.inf, i.e. the step is not\n        bounded and determined solely by the solver.\n    rtol, atol : float and array_like, optional\n        Relative and absolute tolerances. The solver keeps the local error\n        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a\n        relative accuracy (number of correct digits). But if a component of `y`\n        is approximately below `atol` then the error only needs to fall within\n        the same `atol` threshold, and the number of correct digits is not\n        guaranteed. If components of y have different scales, it might be\n        beneficial to set different `atol` values for different components by\n        passing array_like with shape (n,) for `atol`. Default values are\n        1e-3 for `rtol` and 1e-6 for `atol`.\n    vectorized : bool, optional\n        Whether `fun` is implemented in a vectorized fashion. Default is False.\n\n    Attributes\n    ----------\n    n : int\n        Number of equations.\n    status : string\n        Current status of the solver: \'running\', \'finished\' or \'failed\'.\n    t_bound : float\n        Boundary time.\n    direction : float\n        Integration direction: +1 or -1.\n    t : float\n        Current time.\n    y : ndarray\n        Current state.\n    t_old : float\n        Previous time. None if no steps were made yet.\n    step_size : float\n        Size of the last successful step. None if no steps were made yet.\n    nfev : int\n        Number of the system\'s rhs evaluations.\n    njev : int\n        Number of the Jacobian evaluations.\n    nlu : int\n        Number of LU decompositions.\n\n    References\n    ----------\n    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta\n           formulae", Journal of Computational and Applied Mathematics, Vol. 6,\n           No. 1, pp. 19-26, 1980.\n    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics\n           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.\n    ')
    
    # Assigning a Num to a Name (line 332):
    
    # Assigning a Num to a Name (line 333):
    
    # Assigning a Call to a Name (line 334):
    
    # Assigning a List to a Name (line 335):
    
    # Assigning a Call to a Name (line 340):
    
    # Assigning a Call to a Name (line 341):
    
    # Assigning a Call to a Name (line 344):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 254, 0, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RK45.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'RK45' (line 254)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), 'RK45', RK45)

# Assigning a Num to a Name (line 332):
int_58742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 12), 'int')
# Getting the type of 'RK45'
RK45_58743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK45')
# Setting the type of the member 'order' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK45_58743, 'order', int_58742)

# Assigning a Num to a Name (line 333):
int_58744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 15), 'int')
# Getting the type of 'RK45'
RK45_58745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK45')
# Setting the type of the member 'n_stages' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK45_58745, 'n_stages', int_58744)

# Assigning a Call to a Name (line 334):

# Call to array(...): (line 334)
# Processing the call arguments (line 334)

# Obtaining an instance of the builtin type 'list' (line 334)
list_58748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 334)
# Adding element type (line 334)
int_58749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 18), 'int')
int_58750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 20), 'int')
# Applying the binary operator 'div' (line 334)
result_div_58751 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 18), 'div', int_58749, int_58750)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 17), list_58748, result_div_58751)
# Adding element type (line 334)
int_58752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 23), 'int')
int_58753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 25), 'int')
# Applying the binary operator 'div' (line 334)
result_div_58754 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 23), 'div', int_58752, int_58753)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 17), list_58748, result_div_58754)
# Adding element type (line 334)
int_58755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 29), 'int')
int_58756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 31), 'int')
# Applying the binary operator 'div' (line 334)
result_div_58757 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 29), 'div', int_58755, int_58756)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 17), list_58748, result_div_58757)
# Adding element type (line 334)
int_58758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 34), 'int')
int_58759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 36), 'int')
# Applying the binary operator 'div' (line 334)
result_div_58760 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 34), 'div', int_58758, int_58759)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 17), list_58748, result_div_58760)
# Adding element type (line 334)
int_58761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 17), list_58748, int_58761)

# Processing the call keyword arguments (line 334)
kwargs_58762 = {}
# Getting the type of 'np' (line 334)
np_58746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'np', False)
# Obtaining the member 'array' of a type (line 334)
array_58747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), np_58746, 'array')
# Calling array(args, kwargs) (line 334)
array_call_result_58763 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), array_58747, *[list_58748], **kwargs_58762)

# Getting the type of 'RK45'
RK45_58764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK45')
# Setting the type of the member 'C' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK45_58764, 'C', array_call_result_58763)

# Assigning a List to a Name (line 335):

# Obtaining an instance of the builtin type 'list' (line 335)
list_58765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 335)
# Adding element type (line 335)

# Call to array(...): (line 335)
# Processing the call arguments (line 335)

# Obtaining an instance of the builtin type 'list' (line 335)
list_58768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 335)
# Adding element type (line 335)
int_58769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 19), 'int')
int_58770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 21), 'int')
# Applying the binary operator 'div' (line 335)
result_div_58771 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 19), 'div', int_58769, int_58770)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 18), list_58768, result_div_58771)

# Processing the call keyword arguments (line 335)
kwargs_58772 = {}
# Getting the type of 'np' (line 335)
np_58766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 9), 'np', False)
# Obtaining the member 'array' of a type (line 335)
array_58767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 9), np_58766, 'array')
# Calling array(args, kwargs) (line 335)
array_call_result_58773 = invoke(stypy.reporting.localization.Localization(__file__, 335, 9), array_58767, *[list_58768], **kwargs_58772)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 8), list_58765, array_call_result_58773)
# Adding element type (line 335)

# Call to array(...): (line 336)
# Processing the call arguments (line 336)

# Obtaining an instance of the builtin type 'list' (line 336)
list_58776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 336)
# Adding element type (line 336)
int_58777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 19), 'int')
int_58778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 21), 'int')
# Applying the binary operator 'div' (line 336)
result_div_58779 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 19), 'div', int_58777, int_58778)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 18), list_58776, result_div_58779)
# Adding element type (line 336)
int_58780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 25), 'int')
int_58781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 27), 'int')
# Applying the binary operator 'div' (line 336)
result_div_58782 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 25), 'div', int_58780, int_58781)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 18), list_58776, result_div_58782)

# Processing the call keyword arguments (line 336)
kwargs_58783 = {}
# Getting the type of 'np' (line 336)
np_58774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 9), 'np', False)
# Obtaining the member 'array' of a type (line 336)
array_58775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 9), np_58774, 'array')
# Calling array(args, kwargs) (line 336)
array_call_result_58784 = invoke(stypy.reporting.localization.Localization(__file__, 336, 9), array_58775, *[list_58776], **kwargs_58783)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 8), list_58765, array_call_result_58784)
# Adding element type (line 335)

# Call to array(...): (line 337)
# Processing the call arguments (line 337)

# Obtaining an instance of the builtin type 'list' (line 337)
list_58787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 337)
# Adding element type (line 337)
int_58788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 19), 'int')
int_58789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 22), 'int')
# Applying the binary operator 'div' (line 337)
result_div_58790 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 19), 'div', int_58788, int_58789)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 18), list_58787, result_div_58790)
# Adding element type (line 337)
int_58791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 26), 'int')
int_58792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 30), 'int')
# Applying the binary operator 'div' (line 337)
result_div_58793 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 26), 'div', int_58791, int_58792)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 18), list_58787, result_div_58793)
# Adding element type (line 337)
int_58794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 34), 'int')
int_58795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 37), 'int')
# Applying the binary operator 'div' (line 337)
result_div_58796 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 34), 'div', int_58794, int_58795)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 18), list_58787, result_div_58796)

# Processing the call keyword arguments (line 337)
kwargs_58797 = {}
# Getting the type of 'np' (line 337)
np_58785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 9), 'np', False)
# Obtaining the member 'array' of a type (line 337)
array_58786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 9), np_58785, 'array')
# Calling array(args, kwargs) (line 337)
array_call_result_58798 = invoke(stypy.reporting.localization.Localization(__file__, 337, 9), array_58786, *[list_58787], **kwargs_58797)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 8), list_58765, array_call_result_58798)
# Adding element type (line 335)

# Call to array(...): (line 338)
# Processing the call arguments (line 338)

# Obtaining an instance of the builtin type 'list' (line 338)
list_58801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 338)
# Adding element type (line 338)
int_58802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 19), 'int')
int_58803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 25), 'int')
# Applying the binary operator 'div' (line 338)
result_div_58804 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 19), 'div', int_58802, int_58803)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 18), list_58801, result_div_58804)
# Adding element type (line 338)
int_58805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 31), 'int')
int_58806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 38), 'int')
# Applying the binary operator 'div' (line 338)
result_div_58807 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 31), 'div', int_58805, int_58806)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 18), list_58801, result_div_58807)
# Adding element type (line 338)
int_58808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 44), 'int')
int_58809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 50), 'int')
# Applying the binary operator 'div' (line 338)
result_div_58810 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 44), 'div', int_58808, int_58809)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 18), list_58801, result_div_58810)
# Adding element type (line 338)
int_58811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 56), 'int')
int_58812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 61), 'int')
# Applying the binary operator 'div' (line 338)
result_div_58813 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 56), 'div', int_58811, int_58812)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 18), list_58801, result_div_58813)

# Processing the call keyword arguments (line 338)
kwargs_58814 = {}
# Getting the type of 'np' (line 338)
np_58799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 9), 'np', False)
# Obtaining the member 'array' of a type (line 338)
array_58800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 9), np_58799, 'array')
# Calling array(args, kwargs) (line 338)
array_call_result_58815 = invoke(stypy.reporting.localization.Localization(__file__, 338, 9), array_58800, *[list_58801], **kwargs_58814)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 8), list_58765, array_call_result_58815)
# Adding element type (line 335)

# Call to array(...): (line 339)
# Processing the call arguments (line 339)

# Obtaining an instance of the builtin type 'list' (line 339)
list_58818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 339)
# Adding element type (line 339)
int_58819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 19), 'int')
int_58820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 24), 'int')
# Applying the binary operator 'div' (line 339)
result_div_58821 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 19), 'div', int_58819, int_58820)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 18), list_58818, result_div_58821)
# Adding element type (line 339)
int_58822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 30), 'int')
int_58823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 35), 'int')
# Applying the binary operator 'div' (line 339)
result_div_58824 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 30), 'div', int_58822, int_58823)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 18), list_58818, result_div_58824)
# Adding element type (line 339)
int_58825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 39), 'int')
int_58826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 45), 'int')
# Applying the binary operator 'div' (line 339)
result_div_58827 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 39), 'div', int_58825, int_58826)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 18), list_58818, result_div_58827)
# Adding element type (line 339)
int_58828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 51), 'int')
int_58829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 54), 'int')
# Applying the binary operator 'div' (line 339)
result_div_58830 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 51), 'div', int_58828, int_58829)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 18), list_58818, result_div_58830)
# Adding element type (line 339)
int_58831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 59), 'int')
int_58832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 65), 'int')
# Applying the binary operator 'div' (line 339)
result_div_58833 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 59), 'div', int_58831, int_58832)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 18), list_58818, result_div_58833)

# Processing the call keyword arguments (line 339)
kwargs_58834 = {}
# Getting the type of 'np' (line 339)
np_58816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 9), 'np', False)
# Obtaining the member 'array' of a type (line 339)
array_58817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 9), np_58816, 'array')
# Calling array(args, kwargs) (line 339)
array_call_result_58835 = invoke(stypy.reporting.localization.Localization(__file__, 339, 9), array_58817, *[list_58818], **kwargs_58834)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 8), list_58765, array_call_result_58835)

# Getting the type of 'RK45'
RK45_58836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK45')
# Setting the type of the member 'A' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK45_58836, 'A', list_58765)

# Assigning a Call to a Name (line 340):

# Call to array(...): (line 340)
# Processing the call arguments (line 340)

# Obtaining an instance of the builtin type 'list' (line 340)
list_58839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 340)
# Adding element type (line 340)
int_58840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 18), 'int')
int_58841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 21), 'int')
# Applying the binary operator 'div' (line 340)
result_div_58842 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 18), 'div', int_58840, int_58841)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 17), list_58839, result_div_58842)
# Adding element type (line 340)
int_58843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 17), list_58839, int_58843)
# Adding element type (line 340)
int_58844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 29), 'int')
int_58845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 33), 'int')
# Applying the binary operator 'div' (line 340)
result_div_58846 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 29), 'div', int_58844, int_58845)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 17), list_58839, result_div_58846)
# Adding element type (line 340)
int_58847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 39), 'int')
int_58848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 43), 'int')
# Applying the binary operator 'div' (line 340)
result_div_58849 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 39), 'div', int_58847, int_58848)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 17), list_58839, result_div_58849)
# Adding element type (line 340)
int_58850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 48), 'int')
int_58851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 54), 'int')
# Applying the binary operator 'div' (line 340)
result_div_58852 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 48), 'div', int_58850, int_58851)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 17), list_58839, result_div_58852)
# Adding element type (line 340)
int_58853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 60), 'int')
int_58854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 63), 'int')
# Applying the binary operator 'div' (line 340)
result_div_58855 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 60), 'div', int_58853, int_58854)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 17), list_58839, result_div_58855)

# Processing the call keyword arguments (line 340)
kwargs_58856 = {}
# Getting the type of 'np' (line 340)
np_58837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'np', False)
# Obtaining the member 'array' of a type (line 340)
array_58838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), np_58837, 'array')
# Calling array(args, kwargs) (line 340)
array_call_result_58857 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), array_58838, *[list_58839], **kwargs_58856)

# Getting the type of 'RK45'
RK45_58858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK45')
# Setting the type of the member 'B' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK45_58858, 'B', array_call_result_58857)

# Assigning a Call to a Name (line 341):

# Call to array(...): (line 341)
# Processing the call arguments (line 341)

# Obtaining an instance of the builtin type 'list' (line 341)
list_58861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 341)
# Adding element type (line 341)
int_58862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 18), 'int')
int_58863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 22), 'int')
# Applying the binary operator 'div' (line 341)
result_div_58864 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 18), 'div', int_58862, int_58863)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 17), list_58861, result_div_58864)
# Adding element type (line 341)
int_58865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 17), list_58861, int_58865)
# Adding element type (line 341)
int_58866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 32), 'int')
int_58867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 35), 'int')
# Applying the binary operator 'div' (line 341)
result_div_58868 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 32), 'div', int_58866, int_58867)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 17), list_58861, result_div_58868)
# Adding element type (line 341)
int_58869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 42), 'int')
int_58870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 46), 'int')
# Applying the binary operator 'div' (line 341)
result_div_58871 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 42), 'div', int_58869, int_58870)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 17), list_58861, result_div_58871)
# Adding element type (line 341)
int_58872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 52), 'int')
int_58873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 58), 'int')
# Applying the binary operator 'div' (line 341)
result_div_58874 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 52), 'div', int_58872, int_58873)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 17), list_58861, result_div_58874)
# Adding element type (line 341)
int_58875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 66), 'int')
int_58876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 70), 'int')
# Applying the binary operator 'div' (line 341)
result_div_58877 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 66), 'div', int_58875, int_58876)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 17), list_58861, result_div_58877)
# Adding element type (line 341)
int_58878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 18), 'int')
int_58879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 20), 'int')
# Applying the binary operator 'div' (line 342)
result_div_58880 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 18), 'div', int_58878, int_58879)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 17), list_58861, result_div_58880)

# Processing the call keyword arguments (line 341)
kwargs_58881 = {}
# Getting the type of 'np' (line 341)
np_58859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'np', False)
# Obtaining the member 'array' of a type (line 341)
array_58860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), np_58859, 'array')
# Calling array(args, kwargs) (line 341)
array_call_result_58882 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), array_58860, *[list_58861], **kwargs_58881)

# Getting the type of 'RK45'
RK45_58883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK45')
# Setting the type of the member 'E' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK45_58883, 'E', array_call_result_58882)

# Assigning a Call to a Name (line 344):

# Call to array(...): (line 344)
# Processing the call arguments (line 344)

# Obtaining an instance of the builtin type 'list' (line 344)
list_58886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 344)
# Adding element type (line 344)

# Obtaining an instance of the builtin type 'list' (line 345)
list_58887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 345)
# Adding element type (line 345)
int_58888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 8), list_58887, int_58888)
# Adding element type (line 345)
long_58889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 12), 'long')
long_58890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 24), 'long')
# Applying the binary operator 'div' (line 345)
result_div_58891 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 12), 'div', long_58889, long_58890)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 8), list_58887, result_div_58891)
# Adding element type (line 345)
long_58892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 36), 'long')
long_58893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 47), 'long')
# Applying the binary operator 'div' (line 345)
result_div_58894 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 36), 'div', long_58892, long_58893)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 8), list_58887, result_div_58894)
# Adding element type (line 345)
long_58895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 9), 'long')
long_58896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 22), 'long')
# Applying the binary operator 'div' (line 346)
result_div_58897 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 9), 'div', long_58895, long_58896)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 8), list_58887, result_div_58897)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 17), list_58886, list_58887)
# Adding element type (line 344)

# Obtaining an instance of the builtin type 'list' (line 347)
list_58898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 347)
# Adding element type (line 347)
int_58899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 8), list_58898, int_58899)
# Adding element type (line 347)
int_58900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 8), list_58898, int_58900)
# Adding element type (line 347)
int_58901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 8), list_58898, int_58901)
# Adding element type (line 347)
int_58902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 8), list_58898, int_58902)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 17), list_58886, list_58898)
# Adding element type (line 344)

# Obtaining an instance of the builtin type 'list' (line 348)
list_58903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 348)
# Adding element type (line 348)
int_58904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 8), list_58903, int_58904)
# Adding element type (line 348)
long_58905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 12), 'long')
long_58906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 25), 'long')
# Applying the binary operator 'div' (line 348)
result_div_58907 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 12), 'div', long_58905, long_58906)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 8), list_58903, result_div_58907)
# Adding element type (line 348)
long_58908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 38), 'long')
long_58909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 51), 'long')
# Applying the binary operator 'div' (line 348)
result_div_58910 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 38), 'div', long_58908, long_58909)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 8), list_58903, result_div_58910)
# Adding element type (line 348)
long_58911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 9), 'long')
long_58912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 21), 'long')
# Applying the binary operator 'div' (line 349)
result_div_58913 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 9), 'div', long_58911, long_58912)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 8), list_58903, result_div_58913)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 17), list_58886, list_58903)
# Adding element type (line 344)

# Obtaining an instance of the builtin type 'list' (line 350)
list_58914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 350)
# Adding element type (line 350)
int_58915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 8), list_58914, int_58915)
# Adding element type (line 350)
int_58916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 12), 'int')
int_58917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 24), 'int')
# Applying the binary operator 'div' (line 350)
result_div_58918 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 12), 'div', int_58916, int_58917)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 8), list_58914, result_div_58918)
# Adding element type (line 350)
long_58919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 35), 'long')
int_58920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 47), 'int')
# Applying the binary operator 'div' (line 350)
result_div_58921 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 35), 'div', long_58919, int_58920)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 8), list_58914, result_div_58921)
# Adding element type (line 350)
long_58922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 9), 'long')
int_58923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 22), 'int')
# Applying the binary operator 'div' (line 351)
result_div_58924 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 9), 'div', long_58922, int_58923)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 8), list_58914, result_div_58924)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 17), list_58886, list_58914)
# Adding element type (line 344)

# Obtaining an instance of the builtin type 'list' (line 352)
list_58925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 352)
# Adding element type (line 352)
int_58926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 8), list_58925, int_58926)
# Adding element type (line 352)
long_58927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 12), 'long')
long_58928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 25), 'long')
# Applying the binary operator 'div' (line 352)
result_div_58929 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 12), 'div', long_58927, long_58928)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 8), list_58925, result_div_58929)
# Adding element type (line 352)
long_58930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 38), 'long')
long_58931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 52), 'long')
# Applying the binary operator 'div' (line 352)
result_div_58932 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 38), 'div', long_58930, long_58931)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 8), list_58925, result_div_58932)
# Adding element type (line 352)
long_58933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 9), 'long')
long_58934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 24), 'long')
# Applying the binary operator 'div' (line 353)
result_div_58935 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 9), 'div', long_58933, long_58934)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 8), list_58925, result_div_58935)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 17), list_58886, list_58925)
# Adding element type (line 344)

# Obtaining an instance of the builtin type 'list' (line 354)
list_58936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 354)
# Adding element type (line 354)
int_58937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 8), list_58936, int_58937)
# Adding element type (line 354)
int_58938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 12), 'int')
int_58939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 23), 'int')
# Applying the binary operator 'div' (line 354)
result_div_58940 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 12), 'div', int_58938, int_58939)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 8), list_58936, result_div_58940)
# Adding element type (line 354)
int_58941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 34), 'int')
int_58942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 45), 'int')
# Applying the binary operator 'div' (line 354)
result_div_58943 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 34), 'div', int_58941, int_58942)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 8), list_58936, result_div_58943)
# Adding element type (line 354)
int_58944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 56), 'int')
int_58945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 68), 'int')
# Applying the binary operator 'div' (line 354)
result_div_58946 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 56), 'div', int_58944, int_58945)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 8), list_58936, result_div_58946)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 17), list_58886, list_58936)
# Adding element type (line 344)

# Obtaining an instance of the builtin type 'list' (line 355)
list_58947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 355)
# Adding element type (line 355)
int_58948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 8), list_58947, int_58948)
# Adding element type (line 355)
int_58949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 12), 'int')
int_58950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 21), 'int')
# Applying the binary operator 'div' (line 355)
result_div_58951 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 12), 'div', int_58949, int_58950)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 8), list_58947, result_div_58951)
# Adding element type (line 355)
int_58952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 31), 'int')
int_58953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 42), 'int')
# Applying the binary operator 'div' (line 355)
result_div_58954 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 31), 'div', int_58952, int_58953)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 8), list_58947, result_div_58954)
# Adding element type (line 355)
int_58955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 52), 'int')
int_58956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 61), 'int')
# Applying the binary operator 'div' (line 355)
result_div_58957 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 52), 'div', int_58955, int_58956)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 8), list_58947, result_div_58957)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 17), list_58886, list_58947)

# Processing the call keyword arguments (line 344)
kwargs_58958 = {}
# Getting the type of 'np' (line 344)
np_58884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'np', False)
# Obtaining the member 'array' of a type (line 344)
array_58885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), np_58884, 'array')
# Calling array(args, kwargs) (line 344)
array_call_result_58959 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), array_58885, *[list_58886], **kwargs_58958)

# Getting the type of 'RK45'
RK45_58960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RK45')
# Setting the type of the member 'P' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RK45_58960, 'P', array_call_result_58959)
# Declaration of the 'RkDenseOutput' class
# Getting the type of 'DenseOutput' (line 358)
DenseOutput_58961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 20), 'DenseOutput')

class RkDenseOutput(DenseOutput_58961, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 359, 4, False)
        # Assigning a type to the variable 'self' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RkDenseOutput.__init__', ['t_old', 't', 'y_old', 'Q'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['t_old', 't', 'y_old', 'Q'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 't_old' (line 360)
        t_old_58968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 44), 't_old', False)
        # Getting the type of 't' (line 360)
        t_58969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 51), 't', False)
        # Processing the call keyword arguments (line 360)
        kwargs_58970 = {}
        
        # Call to super(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'RkDenseOutput' (line 360)
        RkDenseOutput_58963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 14), 'RkDenseOutput', False)
        # Getting the type of 'self' (line 360)
        self_58964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 29), 'self', False)
        # Processing the call keyword arguments (line 360)
        kwargs_58965 = {}
        # Getting the type of 'super' (line 360)
        super_58962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'super', False)
        # Calling super(args, kwargs) (line 360)
        super_call_result_58966 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), super_58962, *[RkDenseOutput_58963, self_58964], **kwargs_58965)
        
        # Obtaining the member '__init__' of a type (line 360)
        init___58967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 8), super_call_result_58966, '__init__')
        # Calling __init__(args, kwargs) (line 360)
        init___call_result_58971 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), init___58967, *[t_old_58968, t_58969], **kwargs_58970)
        
        
        # Assigning a BinOp to a Attribute (line 361):
        
        # Assigning a BinOp to a Attribute (line 361):
        # Getting the type of 't' (line 361)
        t_58972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 17), 't')
        # Getting the type of 't_old' (line 361)
        t_old_58973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 21), 't_old')
        # Applying the binary operator '-' (line 361)
        result_sub_58974 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 17), '-', t_58972, t_old_58973)
        
        # Getting the type of 'self' (line 361)
        self_58975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'self')
        # Setting the type of the member 'h' of a type (line 361)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), self_58975, 'h', result_sub_58974)
        
        # Assigning a Name to a Attribute (line 362):
        
        # Assigning a Name to a Attribute (line 362):
        # Getting the type of 'Q' (line 362)
        Q_58976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 17), 'Q')
        # Getting the type of 'self' (line 362)
        self_58977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'self')
        # Setting the type of the member 'Q' of a type (line 362)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), self_58977, 'Q', Q_58976)
        
        # Assigning a BinOp to a Attribute (line 363):
        
        # Assigning a BinOp to a Attribute (line 363):
        
        # Obtaining the type of the subscript
        int_58978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 29), 'int')
        # Getting the type of 'Q' (line 363)
        Q_58979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 21), 'Q')
        # Obtaining the member 'shape' of a type (line 363)
        shape_58980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 21), Q_58979, 'shape')
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___58981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 21), shape_58980, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_58982 = invoke(stypy.reporting.localization.Localization(__file__, 363, 21), getitem___58981, int_58978)
        
        int_58983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 34), 'int')
        # Applying the binary operator '-' (line 363)
        result_sub_58984 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 21), '-', subscript_call_result_58982, int_58983)
        
        # Getting the type of 'self' (line 363)
        self_58985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'self')
        # Setting the type of the member 'order' of a type (line 363)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), self_58985, 'order', result_sub_58984)
        
        # Assigning a Name to a Attribute (line 364):
        
        # Assigning a Name to a Attribute (line 364):
        # Getting the type of 'y_old' (line 364)
        y_old_58986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 21), 'y_old')
        # Getting the type of 'self' (line 364)
        self_58987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'self')
        # Setting the type of the member 'y_old' of a type (line 364)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 8), self_58987, 'y_old', y_old_58986)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _call_impl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_call_impl'
        module_type_store = module_type_store.open_function_context('_call_impl', 366, 4, False)
        # Assigning a type to the variable 'self' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RkDenseOutput._call_impl.__dict__.__setitem__('stypy_localization', localization)
        RkDenseOutput._call_impl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RkDenseOutput._call_impl.__dict__.__setitem__('stypy_type_store', module_type_store)
        RkDenseOutput._call_impl.__dict__.__setitem__('stypy_function_name', 'RkDenseOutput._call_impl')
        RkDenseOutput._call_impl.__dict__.__setitem__('stypy_param_names_list', ['t'])
        RkDenseOutput._call_impl.__dict__.__setitem__('stypy_varargs_param_name', None)
        RkDenseOutput._call_impl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RkDenseOutput._call_impl.__dict__.__setitem__('stypy_call_defaults', defaults)
        RkDenseOutput._call_impl.__dict__.__setitem__('stypy_call_varargs', varargs)
        RkDenseOutput._call_impl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RkDenseOutput._call_impl.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RkDenseOutput._call_impl', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_call_impl', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_call_impl(...)' code ##################

        
        # Assigning a BinOp to a Name (line 367):
        
        # Assigning a BinOp to a Name (line 367):
        # Getting the type of 't' (line 367)
        t_58988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 13), 't')
        # Getting the type of 'self' (line 367)
        self_58989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 17), 'self')
        # Obtaining the member 't_old' of a type (line 367)
        t_old_58990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 17), self_58989, 't_old')
        # Applying the binary operator '-' (line 367)
        result_sub_58991 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 13), '-', t_58988, t_old_58990)
        
        # Getting the type of 'self' (line 367)
        self_58992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 31), 'self')
        # Obtaining the member 'h' of a type (line 367)
        h_58993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 31), self_58992, 'h')
        # Applying the binary operator 'div' (line 367)
        result_div_58994 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 12), 'div', result_sub_58991, h_58993)
        
        # Assigning a type to the variable 'x' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'x', result_div_58994)
        
        
        # Getting the type of 't' (line 368)
        t_58995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 11), 't')
        # Obtaining the member 'ndim' of a type (line 368)
        ndim_58996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 11), t_58995, 'ndim')
        int_58997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 21), 'int')
        # Applying the binary operator '==' (line 368)
        result_eq_58998 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 11), '==', ndim_58996, int_58997)
        
        # Testing the type of an if condition (line 368)
        if_condition_58999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 8), result_eq_58998)
        # Assigning a type to the variable 'if_condition_58999' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'if_condition_58999', if_condition_58999)
        # SSA begins for if statement (line 368)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 369):
        
        # Assigning a Call to a Name (line 369):
        
        # Call to tile(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'x' (line 369)
        x_59002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 24), 'x', False)
        # Getting the type of 'self' (line 369)
        self_59003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 27), 'self', False)
        # Obtaining the member 'order' of a type (line 369)
        order_59004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 27), self_59003, 'order')
        int_59005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 40), 'int')
        # Applying the binary operator '+' (line 369)
        result_add_59006 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 27), '+', order_59004, int_59005)
        
        # Processing the call keyword arguments (line 369)
        kwargs_59007 = {}
        # Getting the type of 'np' (line 369)
        np_59000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'np', False)
        # Obtaining the member 'tile' of a type (line 369)
        tile_59001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 16), np_59000, 'tile')
        # Calling tile(args, kwargs) (line 369)
        tile_call_result_59008 = invoke(stypy.reporting.localization.Localization(__file__, 369, 16), tile_59001, *[x_59002, result_add_59006], **kwargs_59007)
        
        # Assigning a type to the variable 'p' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'p', tile_call_result_59008)
        
        # Assigning a Call to a Name (line 370):
        
        # Assigning a Call to a Name (line 370):
        
        # Call to cumprod(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'p' (line 370)
        p_59011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 27), 'p', False)
        # Processing the call keyword arguments (line 370)
        kwargs_59012 = {}
        # Getting the type of 'np' (line 370)
        np_59009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'np', False)
        # Obtaining the member 'cumprod' of a type (line 370)
        cumprod_59010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 16), np_59009, 'cumprod')
        # Calling cumprod(args, kwargs) (line 370)
        cumprod_call_result_59013 = invoke(stypy.reporting.localization.Localization(__file__, 370, 16), cumprod_59010, *[p_59011], **kwargs_59012)
        
        # Assigning a type to the variable 'p' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'p', cumprod_call_result_59013)
        # SSA branch for the else part of an if statement (line 368)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 372):
        
        # Assigning a Call to a Name (line 372):
        
        # Call to tile(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'x' (line 372)
        x_59016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 24), 'x', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 372)
        tuple_59017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 372)
        # Adding element type (line 372)
        # Getting the type of 'self' (line 372)
        self_59018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 28), 'self', False)
        # Obtaining the member 'order' of a type (line 372)
        order_59019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 28), self_59018, 'order')
        int_59020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 41), 'int')
        # Applying the binary operator '+' (line 372)
        result_add_59021 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 28), '+', order_59019, int_59020)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 28), tuple_59017, result_add_59021)
        # Adding element type (line 372)
        int_59022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 28), tuple_59017, int_59022)
        
        # Processing the call keyword arguments (line 372)
        kwargs_59023 = {}
        # Getting the type of 'np' (line 372)
        np_59014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'np', False)
        # Obtaining the member 'tile' of a type (line 372)
        tile_59015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 16), np_59014, 'tile')
        # Calling tile(args, kwargs) (line 372)
        tile_call_result_59024 = invoke(stypy.reporting.localization.Localization(__file__, 372, 16), tile_59015, *[x_59016, tuple_59017], **kwargs_59023)
        
        # Assigning a type to the variable 'p' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'p', tile_call_result_59024)
        
        # Assigning a Call to a Name (line 373):
        
        # Assigning a Call to a Name (line 373):
        
        # Call to cumprod(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'p' (line 373)
        p_59027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 27), 'p', False)
        # Processing the call keyword arguments (line 373)
        int_59028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 35), 'int')
        keyword_59029 = int_59028
        kwargs_59030 = {'axis': keyword_59029}
        # Getting the type of 'np' (line 373)
        np_59025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'np', False)
        # Obtaining the member 'cumprod' of a type (line 373)
        cumprod_59026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 16), np_59025, 'cumprod')
        # Calling cumprod(args, kwargs) (line 373)
        cumprod_call_result_59031 = invoke(stypy.reporting.localization.Localization(__file__, 373, 16), cumprod_59026, *[p_59027], **kwargs_59030)
        
        # Assigning a type to the variable 'p' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'p', cumprod_call_result_59031)
        # SSA join for if statement (line 368)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 374):
        
        # Assigning a BinOp to a Name (line 374):
        # Getting the type of 'self' (line 374)
        self_59032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'self')
        # Obtaining the member 'h' of a type (line 374)
        h_59033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), self_59032, 'h')
        
        # Call to dot(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'self' (line 374)
        self_59036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 28), 'self', False)
        # Obtaining the member 'Q' of a type (line 374)
        Q_59037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 28), self_59036, 'Q')
        # Getting the type of 'p' (line 374)
        p_59038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 36), 'p', False)
        # Processing the call keyword arguments (line 374)
        kwargs_59039 = {}
        # Getting the type of 'np' (line 374)
        np_59034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 21), 'np', False)
        # Obtaining the member 'dot' of a type (line 374)
        dot_59035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 21), np_59034, 'dot')
        # Calling dot(args, kwargs) (line 374)
        dot_call_result_59040 = invoke(stypy.reporting.localization.Localization(__file__, 374, 21), dot_59035, *[Q_59037, p_59038], **kwargs_59039)
        
        # Applying the binary operator '*' (line 374)
        result_mul_59041 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 12), '*', h_59033, dot_call_result_59040)
        
        # Assigning a type to the variable 'y' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'y', result_mul_59041)
        
        
        # Getting the type of 'y' (line 375)
        y_59042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'y')
        # Obtaining the member 'ndim' of a type (line 375)
        ndim_59043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 11), y_59042, 'ndim')
        int_59044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 21), 'int')
        # Applying the binary operator '==' (line 375)
        result_eq_59045 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 11), '==', ndim_59043, int_59044)
        
        # Testing the type of an if condition (line 375)
        if_condition_59046 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 8), result_eq_59045)
        # Assigning a type to the variable 'if_condition_59046' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'if_condition_59046', if_condition_59046)
        # SSA begins for if statement (line 375)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'y' (line 376)
        y_59047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'y')
        
        # Obtaining the type of the subscript
        slice_59048 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 376, 17), None, None, None)
        # Getting the type of 'None' (line 376)
        None_59049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 31), 'None')
        # Getting the type of 'self' (line 376)
        self_59050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 17), 'self')
        # Obtaining the member 'y_old' of a type (line 376)
        y_old_59051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 17), self_59050, 'y_old')
        # Obtaining the member '__getitem__' of a type (line 376)
        getitem___59052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 17), y_old_59051, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 376)
        subscript_call_result_59053 = invoke(stypy.reporting.localization.Localization(__file__, 376, 17), getitem___59052, (slice_59048, None_59049))
        
        # Applying the binary operator '+=' (line 376)
        result_iadd_59054 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 12), '+=', y_59047, subscript_call_result_59053)
        # Assigning a type to the variable 'y' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'y', result_iadd_59054)
        
        # SSA branch for the else part of an if statement (line 375)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'y' (line 378)
        y_59055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'y')
        # Getting the type of 'self' (line 378)
        self_59056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), 'self')
        # Obtaining the member 'y_old' of a type (line 378)
        y_old_59057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 17), self_59056, 'y_old')
        # Applying the binary operator '+=' (line 378)
        result_iadd_59058 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 12), '+=', y_59055, y_old_59057)
        # Assigning a type to the variable 'y' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'y', result_iadd_59058)
        
        # SSA join for if statement (line 375)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'y' (line 380)
        y_59059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'stypy_return_type', y_59059)
        
        # ################# End of '_call_impl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_call_impl' in the type store
        # Getting the type of 'stypy_return_type' (line 366)
        stypy_return_type_59060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59060)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_call_impl'
        return stypy_return_type_59060


# Assigning a type to the variable 'RkDenseOutput' (line 358)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), 'RkDenseOutput', RkDenseOutput)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
