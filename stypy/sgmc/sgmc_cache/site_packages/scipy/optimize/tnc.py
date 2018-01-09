
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # TNC Python interface
2: # @(#) $Jeannot: tnc.py,v 1.11 2005/01/28 18:27:31 js Exp $
3: 
4: # Copyright (c) 2004-2005, Jean-Sebastien Roy (js@jeannot.org)
5: 
6: # Permission is hereby granted, free of charge, to any person obtaining a
7: # copy of this software and associated documentation files (the
8: # "Software"), to deal in the Software without restriction, including
9: # without limitation the rights to use, copy, modify, merge, publish,
10: # distribute, sublicense, and/or sell copies of the Software, and to
11: # permit persons to whom the Software is furnished to do so, subject to
12: # the following conditions:
13: 
14: # The above copyright notice and this permission notice shall be included
15: # in all copies or substantial portions of the Software.
16: 
17: # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
18: # OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
19: # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
20: # IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
21: # CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
22: # TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
23: # SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
24: 
25: '''
26: TNC: A python interface to the TNC non-linear optimizer
27: 
28: TNC is a non-linear optimizer. To use it, you must provide a function to
29: minimize. The function must take one argument: the list of coordinates where to
30: evaluate the function; and it must return either a tuple, whose first element is the
31: value of the function, and whose second argument is the gradient of the function
32: (as a list of values); or None, to abort the minimization.
33: '''
34: 
35: from __future__ import division, print_function, absolute_import
36: 
37: from scipy.optimize import moduleTNC, approx_fprime
38: from .optimize import MemoizeJac, OptimizeResult, _check_unknown_options
39: from numpy import inf, array, zeros, asfarray
40: 
41: __all__ = ['fmin_tnc']
42: 
43: 
44: MSG_NONE = 0  # No messages
45: MSG_ITER = 1  # One line per iteration
46: MSG_INFO = 2  # Informational messages
47: MSG_VERS = 4  # Version info
48: MSG_EXIT = 8  # Exit reasons
49: MSG_ALL = MSG_ITER + MSG_INFO + MSG_VERS + MSG_EXIT
50: 
51: MSGS = {
52:         MSG_NONE: "No messages",
53:         MSG_ITER: "One line per iteration",
54:         MSG_INFO: "Informational messages",
55:         MSG_VERS: "Version info",
56:         MSG_EXIT: "Exit reasons",
57:         MSG_ALL: "All messages"
58: }
59: 
60: INFEASIBLE = -1  # Infeasible (lower bound > upper bound)
61: LOCALMINIMUM = 0  # Local minimum reached (|pg| ~= 0)
62: FCONVERGED = 1  # Converged (|f_n-f_(n-1)| ~= 0)
63: XCONVERGED = 2  # Converged (|x_n-x_(n-1)| ~= 0)
64: MAXFUN = 3  # Max. number of function evaluations reached
65: LSFAIL = 4  # Linear search failed
66: CONSTANT = 5  # All lower bounds are equal to the upper bounds
67: NOPROGRESS = 6  # Unable to progress
68: USERABORT = 7  # User requested end of minimization
69: 
70: RCSTRINGS = {
71:         INFEASIBLE: "Infeasible (lower bound > upper bound)",
72:         LOCALMINIMUM: "Local minimum reached (|pg| ~= 0)",
73:         FCONVERGED: "Converged (|f_n-f_(n-1)| ~= 0)",
74:         XCONVERGED: "Converged (|x_n-x_(n-1)| ~= 0)",
75:         MAXFUN: "Max. number of function evaluations reached",
76:         LSFAIL: "Linear search failed",
77:         CONSTANT: "All lower bounds are equal to the upper bounds",
78:         NOPROGRESS: "Unable to progress",
79:         USERABORT: "User requested end of minimization"
80: }
81: 
82: # Changes to interface made by Travis Oliphant, Apr. 2004 for inclusion in
83: #  SciPy
84: 
85: 
86: def fmin_tnc(func, x0, fprime=None, args=(), approx_grad=0,
87:              bounds=None, epsilon=1e-8, scale=None, offset=None,
88:              messages=MSG_ALL, maxCGit=-1, maxfun=None, eta=-1,
89:              stepmx=0, accuracy=0, fmin=0, ftol=-1, xtol=-1, pgtol=-1,
90:              rescale=-1, disp=None, callback=None):
91:     '''
92:     Minimize a function with variables subject to bounds, using
93:     gradient information in a truncated Newton algorithm. This
94:     method wraps a C implementation of the algorithm.
95: 
96:     Parameters
97:     ----------
98:     func : callable ``func(x, *args)``
99:         Function to minimize.  Must do one of:
100: 
101:         1. Return f and g, where f is the value of the function and g its
102:            gradient (a list of floats).
103: 
104:         2. Return the function value but supply gradient function
105:            separately as `fprime`.
106: 
107:         3. Return the function value and set ``approx_grad=True``.
108: 
109:         If the function returns None, the minimization
110:         is aborted.
111:     x0 : array_like
112:         Initial estimate of minimum.
113:     fprime : callable ``fprime(x, *args)``, optional
114:         Gradient of `func`. If None, then either `func` must return the
115:         function value and the gradient (``f,g = func(x, *args)``)
116:         or `approx_grad` must be True.
117:     args : tuple, optional
118:         Arguments to pass to function.
119:     approx_grad : bool, optional
120:         If true, approximate the gradient numerically.
121:     bounds : list, optional
122:         (min, max) pairs for each element in x0, defining the
123:         bounds on that parameter. Use None or +/-inf for one of
124:         min or max when there is no bound in that direction.
125:     epsilon : float, optional
126:         Used if approx_grad is True. The stepsize in a finite
127:         difference approximation for fprime.
128:     scale : array_like, optional
129:         Scaling factors to apply to each variable.  If None, the
130:         factors are up-low for interval bounded variables and
131:         1+|x| for the others.  Defaults to None.
132:     offset : array_like, optional
133:         Value to subtract from each variable.  If None, the
134:         offsets are (up+low)/2 for interval bounded variables
135:         and x for the others.
136:     messages : int, optional
137:         Bit mask used to select messages display during
138:         minimization values defined in the MSGS dict.  Defaults to
139:         MGS_ALL.
140:     disp : int, optional
141:         Integer interface to messages.  0 = no message, 5 = all messages
142:     maxCGit : int, optional
143:         Maximum number of hessian*vector evaluations per main
144:         iteration.  If maxCGit == 0, the direction chosen is
145:         -gradient if maxCGit < 0, maxCGit is set to
146:         max(1,min(50,n/2)).  Defaults to -1.
147:     maxfun : int, optional
148:         Maximum number of function evaluation.  if None, maxfun is
149:         set to max(100, 10*len(x0)).  Defaults to None.
150:     eta : float, optional
151:         Severity of the line search. if < 0 or > 1, set to 0.25.
152:         Defaults to -1.
153:     stepmx : float, optional
154:         Maximum step for the line search.  May be increased during
155:         call.  If too small, it will be set to 10.0.  Defaults to 0.
156:     accuracy : float, optional
157:         Relative precision for finite difference calculations.  If
158:         <= machine_precision, set to sqrt(machine_precision).
159:         Defaults to 0.
160:     fmin : float, optional
161:         Minimum function value estimate.  Defaults to 0.
162:     ftol : float, optional
163:         Precision goal for the value of f in the stoping criterion.
164:         If ftol < 0.0, ftol is set to 0.0 defaults to -1.
165:     xtol : float, optional
166:         Precision goal for the value of x in the stopping
167:         criterion (after applying x scaling factors).  If xtol <
168:         0.0, xtol is set to sqrt(machine_precision).  Defaults to
169:         -1.
170:     pgtol : float, optional
171:         Precision goal for the value of the projected gradient in
172:         the stopping criterion (after applying x scaling factors).
173:         If pgtol < 0.0, pgtol is set to 1e-2 * sqrt(accuracy).
174:         Setting it to 0.0 is not recommended.  Defaults to -1.
175:     rescale : float, optional
176:         Scaling factor (in log10) used to trigger f value
177:         rescaling.  If 0, rescale at each iteration.  If a large
178:         value, never rescale.  If < 0, rescale is set to 1.3.
179:     callback : callable, optional
180:         Called after each iteration, as callback(xk), where xk is the
181:         current parameter vector.
182: 
183:     Returns
184:     -------
185:     x : ndarray
186:         The solution.
187:     nfeval : int
188:         The number of function evaluations.
189:     rc : int
190:         Return code, see below
191: 
192:     See also
193:     --------
194:     minimize: Interface to minimization algorithms for multivariate
195:         functions. See the 'TNC' `method` in particular.
196: 
197:     Notes
198:     -----
199:     The underlying algorithm is truncated Newton, also called
200:     Newton Conjugate-Gradient. This method differs from
201:     scipy.optimize.fmin_ncg in that
202: 
203:     1. It wraps a C implementation of the algorithm
204:     2. It allows each variable to be given an upper and lower bound.
205: 
206:     The algorithm incoporates the bound constraints by determining
207:     the descent direction as in an unconstrained truncated Newton,
208:     but never taking a step-size large enough to leave the space
209:     of feasible x's. The algorithm keeps track of a set of
210:     currently active constraints, and ignores them when computing
211:     the minimum allowable step size. (The x's associated with the
212:     active constraint are kept fixed.) If the maximum allowable
213:     step size is zero then a new constraint is added. At the end
214:     of each iteration one of the constraints may be deemed no
215:     longer active and removed. A constraint is considered
216:     no longer active is if it is currently active
217:     but the gradient for that variable points inward from the
218:     constraint. The specific constraint removed is the one
219:     associated with the variable of largest index whose
220:     constraint is no longer active.
221: 
222:     Return codes are defined as follows::
223: 
224:         -1 : Infeasible (lower bound > upper bound)
225:          0 : Local minimum reached (|pg| ~= 0)
226:          1 : Converged (|f_n-f_(n-1)| ~= 0)
227:          2 : Converged (|x_n-x_(n-1)| ~= 0)
228:          3 : Max. number of function evaluations reached
229:          4 : Linear search failed
230:          5 : All lower bounds are equal to the upper bounds
231:          6 : Unable to progress
232:          7 : User requested end of minimization
233: 
234:     References
235:     ----------
236:     Wright S., Nocedal J. (2006), 'Numerical Optimization'
237: 
238:     Nash S.G. (1984), "Newton-Type Minimization Via the Lanczos Method",
239:     SIAM Journal of Numerical Analysis 21, pp. 770-778
240: 
241:     '''
242:     # handle fprime/approx_grad
243:     if approx_grad:
244:         fun = func
245:         jac = None
246:     elif fprime is None:
247:         fun = MemoizeJac(func)
248:         jac = fun.derivative
249:     else:
250:         fun = func
251:         jac = fprime
252: 
253:     if disp is not None:  # disp takes precedence over messages
254:         mesg_num = disp
255:     else:
256:         mesg_num = {0:MSG_NONE, 1:MSG_ITER, 2:MSG_INFO, 3:MSG_VERS,
257:                     4:MSG_EXIT, 5:MSG_ALL}.get(messages, MSG_ALL)
258:     # build options
259:     opts = {'eps': epsilon,
260:             'scale': scale,
261:             'offset': offset,
262:             'mesg_num': mesg_num,
263:             'maxCGit': maxCGit,
264:             'maxiter': maxfun,
265:             'eta': eta,
266:             'stepmx': stepmx,
267:             'accuracy': accuracy,
268:             'minfev': fmin,
269:             'ftol': ftol,
270:             'xtol': xtol,
271:             'gtol': pgtol,
272:             'rescale': rescale,
273:             'disp': False}
274: 
275:     res = _minimize_tnc(fun, x0, args, jac, bounds, callback=callback, **opts)
276: 
277:     return res['x'], res['nfev'], res['status']
278: 
279: 
280: def _minimize_tnc(fun, x0, args=(), jac=None, bounds=None,
281:                   eps=1e-8, scale=None, offset=None, mesg_num=None,
282:                   maxCGit=-1, maxiter=None, eta=-1, stepmx=0, accuracy=0,
283:                   minfev=0, ftol=-1, xtol=-1, gtol=-1, rescale=-1, disp=False,
284:                   callback=None, **unknown_options):
285:     '''
286:     Minimize a scalar function of one or more variables using a truncated
287:     Newton (TNC) algorithm.
288: 
289:     Options
290:     -------
291:     eps : float
292:         Step size used for numerical approximation of the jacobian.
293:     scale : list of floats
294:         Scaling factors to apply to each variable.  If None, the
295:         factors are up-low for interval bounded variables and
296:         1+|x] fo the others.  Defaults to None
297:     offset : float
298:         Value to subtract from each variable.  If None, the
299:         offsets are (up+low)/2 for interval bounded variables
300:         and x for the others.
301:     disp : bool
302:        Set to True to print convergence messages.
303:     maxCGit : int
304:         Maximum number of hessian*vector evaluations per main
305:         iteration.  If maxCGit == 0, the direction chosen is
306:         -gradient if maxCGit < 0, maxCGit is set to
307:         max(1,min(50,n/2)).  Defaults to -1.
308:     maxiter : int
309:         Maximum number of function evaluation.  if None, `maxiter` is
310:         set to max(100, 10*len(x0)).  Defaults to None.
311:     eta : float
312:         Severity of the line search. if < 0 or > 1, set to 0.25.
313:         Defaults to -1.
314:     stepmx : float
315:         Maximum step for the line search.  May be increased during
316:         call.  If too small, it will be set to 10.0.  Defaults to 0.
317:     accuracy : float
318:         Relative precision for finite difference calculations.  If
319:         <= machine_precision, set to sqrt(machine_precision).
320:         Defaults to 0.
321:     minfev : float
322:         Minimum function value estimate.  Defaults to 0.
323:     ftol : float
324:         Precision goal for the value of f in the stoping criterion.
325:         If ftol < 0.0, ftol is set to 0.0 defaults to -1.
326:     xtol : float
327:         Precision goal for the value of x in the stopping
328:         criterion (after applying x scaling factors).  If xtol <
329:         0.0, xtol is set to sqrt(machine_precision).  Defaults to
330:         -1.
331:     gtol : float
332:         Precision goal for the value of the projected gradient in
333:         the stopping criterion (after applying x scaling factors).
334:         If gtol < 0.0, gtol is set to 1e-2 * sqrt(accuracy).
335:         Setting it to 0.0 is not recommended.  Defaults to -1.
336:     rescale : float
337:         Scaling factor (in log10) used to trigger f value
338:         rescaling.  If 0, rescale at each iteration.  If a large
339:         value, never rescale.  If < 0, rescale is set to 1.3.
340: 
341:     '''
342:     _check_unknown_options(unknown_options)
343:     epsilon = eps
344:     maxfun = maxiter
345:     fmin = minfev
346:     pgtol = gtol
347: 
348:     x0 = asfarray(x0).flatten()
349:     n = len(x0)
350: 
351:     if bounds is None:
352:         bounds = [(None,None)] * n
353:     if len(bounds) != n:
354:         raise ValueError('length of x0 != length of bounds')
355: 
356:     if mesg_num is not None:
357:         messages = {0:MSG_NONE, 1:MSG_ITER, 2:MSG_INFO, 3:MSG_VERS,
358:                     4:MSG_EXIT, 5:MSG_ALL}.get(mesg_num, MSG_ALL)
359:     elif disp:
360:         messages = MSG_ALL
361:     else:
362:         messages = MSG_NONE
363: 
364:     if jac is None:
365:         def func_and_grad(x):
366:             f = fun(x, *args)
367:             g = approx_fprime(x, fun, epsilon, *args)
368:             return f, g
369:     else:
370:         def func_and_grad(x):
371:             f = fun(x, *args)
372:             g = jac(x, *args)
373:             return f, g
374: 
375:     '''
376:     low, up   : the bounds (lists of floats)
377:                 if low is None, the lower bounds are removed.
378:                 if up is None, the upper bounds are removed.
379:                 low and up defaults to None
380:     '''
381:     low = zeros(n)
382:     up = zeros(n)
383:     for i in range(n):
384:         if bounds[i] is None:
385:             l, u = -inf, inf
386:         else:
387:             l,u = bounds[i]
388:             if l is None:
389:                 low[i] = -inf
390:             else:
391:                 low[i] = l
392:             if u is None:
393:                 up[i] = inf
394:             else:
395:                 up[i] = u
396: 
397:     if scale is None:
398:         scale = array([])
399: 
400:     if offset is None:
401:         offset = array([])
402: 
403:     if maxfun is None:
404:         maxfun = max(100, 10*len(x0))
405: 
406:     rc, nf, nit, x = moduleTNC.minimize(func_and_grad, x0, low, up, scale,
407:                                         offset, messages, maxCGit, maxfun,
408:                                         eta, stepmx, accuracy, fmin, ftol,
409:                                         xtol, pgtol, rescale, callback)
410: 
411:     funv, jacv = func_and_grad(x)
412: 
413:     return OptimizeResult(x=x, fun=funv, jac=jacv, nfev=nf, nit=nit, status=rc,
414:                           message=RCSTRINGS[rc], success=(-1 < rc < 3))
415: 
416: if __name__ == '__main__':
417:     # Examples for TNC
418: 
419:     def example():
420:         print("Example")
421: 
422:         # A function to minimize
423:         def function(x):
424:             f = pow(x[0],2.0)+pow(abs(x[1]),3.0)
425:             g = [0,0]
426:             g[0] = 2.0*x[0]
427:             g[1] = 3.0*pow(abs(x[1]),2.0)
428:             if x[1] < 0:
429:                 g[1] = -g[1]
430:             return f, g
431: 
432:         # Optimizer call
433:         x, nf, rc = fmin_tnc(function, [-7, 3], bounds=([-10, 1], [10, 10]))
434: 
435:         print("After", nf, "function evaluations, TNC returned:", RCSTRINGS[rc])
436:         print("x =", x)
437:         print("exact value = [0, 1]")
438:         print()
439: 
440:     example()
441: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_186073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', '\nTNC: A python interface to the TNC non-linear optimizer\n\nTNC is a non-linear optimizer. To use it, you must provide a function to\nminimize. The function must take one argument: the list of coordinates where to\nevaluate the function; and it must return either a tuple, whose first element is the\nvalue of the function, and whose second argument is the gradient of the function\n(as a list of values); or None, to abort the minimization.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from scipy.optimize import moduleTNC, approx_fprime' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_186074 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.optimize')

if (type(import_186074) is not StypyTypeError):

    if (import_186074 != 'pyd_module'):
        __import__(import_186074)
        sys_modules_186075 = sys.modules[import_186074]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.optimize', sys_modules_186075.module_type_store, module_type_store, ['moduleTNC', 'approx_fprime'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_186075, sys_modules_186075.module_type_store, module_type_store)
    else:
        from scipy.optimize import moduleTNC, approx_fprime

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.optimize', None, module_type_store, ['moduleTNC', 'approx_fprime'], [moduleTNC, approx_fprime])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.optimize', import_186074)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'from scipy.optimize.optimize import MemoizeJac, OptimizeResult, _check_unknown_options' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_186076 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.optimize.optimize')

if (type(import_186076) is not StypyTypeError):

    if (import_186076 != 'pyd_module'):
        __import__(import_186076)
        sys_modules_186077 = sys.modules[import_186076]
        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.optimize.optimize', sys_modules_186077.module_type_store, module_type_store, ['MemoizeJac', 'OptimizeResult', '_check_unknown_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 38, 0), __file__, sys_modules_186077, sys_modules_186077.module_type_store, module_type_store)
    else:
        from scipy.optimize.optimize import MemoizeJac, OptimizeResult, _check_unknown_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.optimize.optimize', None, module_type_store, ['MemoizeJac', 'OptimizeResult', '_check_unknown_options'], [MemoizeJac, OptimizeResult, _check_unknown_options])

else:
    # Assigning a type to the variable 'scipy.optimize.optimize' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.optimize.optimize', import_186076)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'from numpy import inf, array, zeros, asfarray' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_186078 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy')

if (type(import_186078) is not StypyTypeError):

    if (import_186078 != 'pyd_module'):
        __import__(import_186078)
        sys_modules_186079 = sys.modules[import_186078]
        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy', sys_modules_186079.module_type_store, module_type_store, ['inf', 'array', 'zeros', 'asfarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 39, 0), __file__, sys_modules_186079, sys_modules_186079.module_type_store, module_type_store)
    else:
        from numpy import inf, array, zeros, asfarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy', None, module_type_store, ['inf', 'array', 'zeros', 'asfarray'], [inf, array, zeros, asfarray])

else:
    # Assigning a type to the variable 'numpy' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy', import_186078)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a List to a Name (line 41):

# Assigning a List to a Name (line 41):
__all__ = ['fmin_tnc']
module_type_store.set_exportable_members(['fmin_tnc'])

# Obtaining an instance of the builtin type 'list' (line 41)
list_186080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)
str_186081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'str', 'fmin_tnc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_186080, str_186081)

# Assigning a type to the variable '__all__' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '__all__', list_186080)

# Assigning a Num to a Name (line 44):

# Assigning a Num to a Name (line 44):
int_186082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 11), 'int')
# Assigning a type to the variable 'MSG_NONE' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'MSG_NONE', int_186082)

# Assigning a Num to a Name (line 45):

# Assigning a Num to a Name (line 45):
int_186083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 11), 'int')
# Assigning a type to the variable 'MSG_ITER' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'MSG_ITER', int_186083)

# Assigning a Num to a Name (line 46):

# Assigning a Num to a Name (line 46):
int_186084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 11), 'int')
# Assigning a type to the variable 'MSG_INFO' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'MSG_INFO', int_186084)

# Assigning a Num to a Name (line 47):

# Assigning a Num to a Name (line 47):
int_186085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'int')
# Assigning a type to the variable 'MSG_VERS' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'MSG_VERS', int_186085)

# Assigning a Num to a Name (line 48):

# Assigning a Num to a Name (line 48):
int_186086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 11), 'int')
# Assigning a type to the variable 'MSG_EXIT' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'MSG_EXIT', int_186086)

# Assigning a BinOp to a Name (line 49):

# Assigning a BinOp to a Name (line 49):
# Getting the type of 'MSG_ITER' (line 49)
MSG_ITER_186087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'MSG_ITER')
# Getting the type of 'MSG_INFO' (line 49)
MSG_INFO_186088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'MSG_INFO')
# Applying the binary operator '+' (line 49)
result_add_186089 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 10), '+', MSG_ITER_186087, MSG_INFO_186088)

# Getting the type of 'MSG_VERS' (line 49)
MSG_VERS_186090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 32), 'MSG_VERS')
# Applying the binary operator '+' (line 49)
result_add_186091 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 30), '+', result_add_186089, MSG_VERS_186090)

# Getting the type of 'MSG_EXIT' (line 49)
MSG_EXIT_186092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 43), 'MSG_EXIT')
# Applying the binary operator '+' (line 49)
result_add_186093 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 41), '+', result_add_186091, MSG_EXIT_186092)

# Assigning a type to the variable 'MSG_ALL' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'MSG_ALL', result_add_186093)

# Assigning a Dict to a Name (line 51):

# Assigning a Dict to a Name (line 51):

# Obtaining an instance of the builtin type 'dict' (line 51)
dict_186094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 7), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 51)
# Adding element type (key, value) (line 51)
# Getting the type of 'MSG_NONE' (line 52)
MSG_NONE_186095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'MSG_NONE')
str_186096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 18), 'str', 'No messages')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_186094, (MSG_NONE_186095, str_186096))
# Adding element type (key, value) (line 51)
# Getting the type of 'MSG_ITER' (line 53)
MSG_ITER_186097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'MSG_ITER')
str_186098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 18), 'str', 'One line per iteration')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_186094, (MSG_ITER_186097, str_186098))
# Adding element type (key, value) (line 51)
# Getting the type of 'MSG_INFO' (line 54)
MSG_INFO_186099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'MSG_INFO')
str_186100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'str', 'Informational messages')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_186094, (MSG_INFO_186099, str_186100))
# Adding element type (key, value) (line 51)
# Getting the type of 'MSG_VERS' (line 55)
MSG_VERS_186101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'MSG_VERS')
str_186102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 18), 'str', 'Version info')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_186094, (MSG_VERS_186101, str_186102))
# Adding element type (key, value) (line 51)
# Getting the type of 'MSG_EXIT' (line 56)
MSG_EXIT_186103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'MSG_EXIT')
str_186104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 18), 'str', 'Exit reasons')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_186094, (MSG_EXIT_186103, str_186104))
# Adding element type (key, value) (line 51)
# Getting the type of 'MSG_ALL' (line 57)
MSG_ALL_186105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'MSG_ALL')
str_186106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 17), 'str', 'All messages')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_186094, (MSG_ALL_186105, str_186106))

# Assigning a type to the variable 'MSGS' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'MSGS', dict_186094)

# Assigning a Num to a Name (line 60):

# Assigning a Num to a Name (line 60):
int_186107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 13), 'int')
# Assigning a type to the variable 'INFEASIBLE' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'INFEASIBLE', int_186107)

# Assigning a Num to a Name (line 61):

# Assigning a Num to a Name (line 61):
int_186108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'int')
# Assigning a type to the variable 'LOCALMINIMUM' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'LOCALMINIMUM', int_186108)

# Assigning a Num to a Name (line 62):

# Assigning a Num to a Name (line 62):
int_186109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 13), 'int')
# Assigning a type to the variable 'FCONVERGED' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'FCONVERGED', int_186109)

# Assigning a Num to a Name (line 63):

# Assigning a Num to a Name (line 63):
int_186110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 13), 'int')
# Assigning a type to the variable 'XCONVERGED' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'XCONVERGED', int_186110)

# Assigning a Num to a Name (line 64):

# Assigning a Num to a Name (line 64):
int_186111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 9), 'int')
# Assigning a type to the variable 'MAXFUN' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'MAXFUN', int_186111)

# Assigning a Num to a Name (line 65):

# Assigning a Num to a Name (line 65):
int_186112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 9), 'int')
# Assigning a type to the variable 'LSFAIL' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'LSFAIL', int_186112)

# Assigning a Num to a Name (line 66):

# Assigning a Num to a Name (line 66):
int_186113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 11), 'int')
# Assigning a type to the variable 'CONSTANT' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'CONSTANT', int_186113)

# Assigning a Num to a Name (line 67):

# Assigning a Num to a Name (line 67):
int_186114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 13), 'int')
# Assigning a type to the variable 'NOPROGRESS' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'NOPROGRESS', int_186114)

# Assigning a Num to a Name (line 68):

# Assigning a Num to a Name (line 68):
int_186115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 12), 'int')
# Assigning a type to the variable 'USERABORT' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'USERABORT', int_186115)

# Assigning a Dict to a Name (line 70):

# Assigning a Dict to a Name (line 70):

# Obtaining an instance of the builtin type 'dict' (line 70)
dict_186116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 70)
# Adding element type (key, value) (line 70)
# Getting the type of 'INFEASIBLE' (line 71)
INFEASIBLE_186117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'INFEASIBLE')
str_186118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'str', 'Infeasible (lower bound > upper bound)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_186116, (INFEASIBLE_186117, str_186118))
# Adding element type (key, value) (line 70)
# Getting the type of 'LOCALMINIMUM' (line 72)
LOCALMINIMUM_186119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'LOCALMINIMUM')
str_186120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 22), 'str', 'Local minimum reached (|pg| ~= 0)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_186116, (LOCALMINIMUM_186119, str_186120))
# Adding element type (key, value) (line 70)
# Getting the type of 'FCONVERGED' (line 73)
FCONVERGED_186121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'FCONVERGED')
str_186122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'str', 'Converged (|f_n-f_(n-1)| ~= 0)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_186116, (FCONVERGED_186121, str_186122))
# Adding element type (key, value) (line 70)
# Getting the type of 'XCONVERGED' (line 74)
XCONVERGED_186123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'XCONVERGED')
str_186124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 20), 'str', 'Converged (|x_n-x_(n-1)| ~= 0)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_186116, (XCONVERGED_186123, str_186124))
# Adding element type (key, value) (line 70)
# Getting the type of 'MAXFUN' (line 75)
MAXFUN_186125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'MAXFUN')
str_186126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 16), 'str', 'Max. number of function evaluations reached')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_186116, (MAXFUN_186125, str_186126))
# Adding element type (key, value) (line 70)
# Getting the type of 'LSFAIL' (line 76)
LSFAIL_186127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'LSFAIL')
str_186128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 16), 'str', 'Linear search failed')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_186116, (LSFAIL_186127, str_186128))
# Adding element type (key, value) (line 70)
# Getting the type of 'CONSTANT' (line 77)
CONSTANT_186129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'CONSTANT')
str_186130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'str', 'All lower bounds are equal to the upper bounds')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_186116, (CONSTANT_186129, str_186130))
# Adding element type (key, value) (line 70)
# Getting the type of 'NOPROGRESS' (line 78)
NOPROGRESS_186131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'NOPROGRESS')
str_186132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'str', 'Unable to progress')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_186116, (NOPROGRESS_186131, str_186132))
# Adding element type (key, value) (line 70)
# Getting the type of 'USERABORT' (line 79)
USERABORT_186133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'USERABORT')
str_186134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'str', 'User requested end of minimization')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), dict_186116, (USERABORT_186133, str_186134))

# Assigning a type to the variable 'RCSTRINGS' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'RCSTRINGS', dict_186116)

@norecursion
def fmin_tnc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 86)
    None_186135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 86)
    tuple_186136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 86)
    
    int_186137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 57), 'int')
    # Getting the type of 'None' (line 87)
    None_186138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'None')
    float_186139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 34), 'float')
    # Getting the type of 'None' (line 87)
    None_186140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 46), 'None')
    # Getting the type of 'None' (line 87)
    None_186141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 59), 'None')
    # Getting the type of 'MSG_ALL' (line 88)
    MSG_ALL_186142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'MSG_ALL')
    int_186143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 39), 'int')
    # Getting the type of 'None' (line 88)
    None_186144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 50), 'None')
    int_186145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 60), 'int')
    int_186146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 20), 'int')
    int_186147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 32), 'int')
    int_186148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 40), 'int')
    int_186149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 48), 'int')
    int_186150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 57), 'int')
    int_186151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 67), 'int')
    int_186152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'int')
    # Getting the type of 'None' (line 90)
    None_186153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'None')
    # Getting the type of 'None' (line 90)
    None_186154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 45), 'None')
    defaults = [None_186135, tuple_186136, int_186137, None_186138, float_186139, None_186140, None_186141, MSG_ALL_186142, int_186143, None_186144, int_186145, int_186146, int_186147, int_186148, int_186149, int_186150, int_186151, int_186152, None_186153, None_186154]
    # Create a new context for function 'fmin_tnc'
    module_type_store = module_type_store.open_function_context('fmin_tnc', 86, 0, False)
    
    # Passed parameters checking function
    fmin_tnc.stypy_localization = localization
    fmin_tnc.stypy_type_of_self = None
    fmin_tnc.stypy_type_store = module_type_store
    fmin_tnc.stypy_function_name = 'fmin_tnc'
    fmin_tnc.stypy_param_names_list = ['func', 'x0', 'fprime', 'args', 'approx_grad', 'bounds', 'epsilon', 'scale', 'offset', 'messages', 'maxCGit', 'maxfun', 'eta', 'stepmx', 'accuracy', 'fmin', 'ftol', 'xtol', 'pgtol', 'rescale', 'disp', 'callback']
    fmin_tnc.stypy_varargs_param_name = None
    fmin_tnc.stypy_kwargs_param_name = None
    fmin_tnc.stypy_call_defaults = defaults
    fmin_tnc.stypy_call_varargs = varargs
    fmin_tnc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fmin_tnc', ['func', 'x0', 'fprime', 'args', 'approx_grad', 'bounds', 'epsilon', 'scale', 'offset', 'messages', 'maxCGit', 'maxfun', 'eta', 'stepmx', 'accuracy', 'fmin', 'ftol', 'xtol', 'pgtol', 'rescale', 'disp', 'callback'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fmin_tnc', localization, ['func', 'x0', 'fprime', 'args', 'approx_grad', 'bounds', 'epsilon', 'scale', 'offset', 'messages', 'maxCGit', 'maxfun', 'eta', 'stepmx', 'accuracy', 'fmin', 'ftol', 'xtol', 'pgtol', 'rescale', 'disp', 'callback'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fmin_tnc(...)' code ##################

    str_186155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, (-1)), 'str', '\n    Minimize a function with variables subject to bounds, using\n    gradient information in a truncated Newton algorithm. This\n    method wraps a C implementation of the algorithm.\n\n    Parameters\n    ----------\n    func : callable ``func(x, *args)``\n        Function to minimize.  Must do one of:\n\n        1. Return f and g, where f is the value of the function and g its\n           gradient (a list of floats).\n\n        2. Return the function value but supply gradient function\n           separately as `fprime`.\n\n        3. Return the function value and set ``approx_grad=True``.\n\n        If the function returns None, the minimization\n        is aborted.\n    x0 : array_like\n        Initial estimate of minimum.\n    fprime : callable ``fprime(x, *args)``, optional\n        Gradient of `func`. If None, then either `func` must return the\n        function value and the gradient (``f,g = func(x, *args)``)\n        or `approx_grad` must be True.\n    args : tuple, optional\n        Arguments to pass to function.\n    approx_grad : bool, optional\n        If true, approximate the gradient numerically.\n    bounds : list, optional\n        (min, max) pairs for each element in x0, defining the\n        bounds on that parameter. Use None or +/-inf for one of\n        min or max when there is no bound in that direction.\n    epsilon : float, optional\n        Used if approx_grad is True. The stepsize in a finite\n        difference approximation for fprime.\n    scale : array_like, optional\n        Scaling factors to apply to each variable.  If None, the\n        factors are up-low for interval bounded variables and\n        1+|x| for the others.  Defaults to None.\n    offset : array_like, optional\n        Value to subtract from each variable.  If None, the\n        offsets are (up+low)/2 for interval bounded variables\n        and x for the others.\n    messages : int, optional\n        Bit mask used to select messages display during\n        minimization values defined in the MSGS dict.  Defaults to\n        MGS_ALL.\n    disp : int, optional\n        Integer interface to messages.  0 = no message, 5 = all messages\n    maxCGit : int, optional\n        Maximum number of hessian*vector evaluations per main\n        iteration.  If maxCGit == 0, the direction chosen is\n        -gradient if maxCGit < 0, maxCGit is set to\n        max(1,min(50,n/2)).  Defaults to -1.\n    maxfun : int, optional\n        Maximum number of function evaluation.  if None, maxfun is\n        set to max(100, 10*len(x0)).  Defaults to None.\n    eta : float, optional\n        Severity of the line search. if < 0 or > 1, set to 0.25.\n        Defaults to -1.\n    stepmx : float, optional\n        Maximum step for the line search.  May be increased during\n        call.  If too small, it will be set to 10.0.  Defaults to 0.\n    accuracy : float, optional\n        Relative precision for finite difference calculations.  If\n        <= machine_precision, set to sqrt(machine_precision).\n        Defaults to 0.\n    fmin : float, optional\n        Minimum function value estimate.  Defaults to 0.\n    ftol : float, optional\n        Precision goal for the value of f in the stoping criterion.\n        If ftol < 0.0, ftol is set to 0.0 defaults to -1.\n    xtol : float, optional\n        Precision goal for the value of x in the stopping\n        criterion (after applying x scaling factors).  If xtol <\n        0.0, xtol is set to sqrt(machine_precision).  Defaults to\n        -1.\n    pgtol : float, optional\n        Precision goal for the value of the projected gradient in\n        the stopping criterion (after applying x scaling factors).\n        If pgtol < 0.0, pgtol is set to 1e-2 * sqrt(accuracy).\n        Setting it to 0.0 is not recommended.  Defaults to -1.\n    rescale : float, optional\n        Scaling factor (in log10) used to trigger f value\n        rescaling.  If 0, rescale at each iteration.  If a large\n        value, never rescale.  If < 0, rescale is set to 1.3.\n    callback : callable, optional\n        Called after each iteration, as callback(xk), where xk is the\n        current parameter vector.\n\n    Returns\n    -------\n    x : ndarray\n        The solution.\n    nfeval : int\n        The number of function evaluations.\n    rc : int\n        Return code, see below\n\n    See also\n    --------\n    minimize: Interface to minimization algorithms for multivariate\n        functions. See the \'TNC\' `method` in particular.\n\n    Notes\n    -----\n    The underlying algorithm is truncated Newton, also called\n    Newton Conjugate-Gradient. This method differs from\n    scipy.optimize.fmin_ncg in that\n\n    1. It wraps a C implementation of the algorithm\n    2. It allows each variable to be given an upper and lower bound.\n\n    The algorithm incoporates the bound constraints by determining\n    the descent direction as in an unconstrained truncated Newton,\n    but never taking a step-size large enough to leave the space\n    of feasible x\'s. The algorithm keeps track of a set of\n    currently active constraints, and ignores them when computing\n    the minimum allowable step size. (The x\'s associated with the\n    active constraint are kept fixed.) If the maximum allowable\n    step size is zero then a new constraint is added. At the end\n    of each iteration one of the constraints may be deemed no\n    longer active and removed. A constraint is considered\n    no longer active is if it is currently active\n    but the gradient for that variable points inward from the\n    constraint. The specific constraint removed is the one\n    associated with the variable of largest index whose\n    constraint is no longer active.\n\n    Return codes are defined as follows::\n\n        -1 : Infeasible (lower bound > upper bound)\n         0 : Local minimum reached (|pg| ~= 0)\n         1 : Converged (|f_n-f_(n-1)| ~= 0)\n         2 : Converged (|x_n-x_(n-1)| ~= 0)\n         3 : Max. number of function evaluations reached\n         4 : Linear search failed\n         5 : All lower bounds are equal to the upper bounds\n         6 : Unable to progress\n         7 : User requested end of minimization\n\n    References\n    ----------\n    Wright S., Nocedal J. (2006), \'Numerical Optimization\'\n\n    Nash S.G. (1984), "Newton-Type Minimization Via the Lanczos Method",\n    SIAM Journal of Numerical Analysis 21, pp. 770-778\n\n    ')
    
    # Getting the type of 'approx_grad' (line 243)
    approx_grad_186156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 7), 'approx_grad')
    # Testing the type of an if condition (line 243)
    if_condition_186157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 4), approx_grad_186156)
    # Assigning a type to the variable 'if_condition_186157' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'if_condition_186157', if_condition_186157)
    # SSA begins for if statement (line 243)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 244):
    
    # Assigning a Name to a Name (line 244):
    # Getting the type of 'func' (line 244)
    func_186158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 14), 'func')
    # Assigning a type to the variable 'fun' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'fun', func_186158)
    
    # Assigning a Name to a Name (line 245):
    
    # Assigning a Name to a Name (line 245):
    # Getting the type of 'None' (line 245)
    None_186159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 14), 'None')
    # Assigning a type to the variable 'jac' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'jac', None_186159)
    # SSA branch for the else part of an if statement (line 243)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 246)
    # Getting the type of 'fprime' (line 246)
    fprime_186160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 9), 'fprime')
    # Getting the type of 'None' (line 246)
    None_186161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'None')
    
    (may_be_186162, more_types_in_union_186163) = may_be_none(fprime_186160, None_186161)

    if may_be_186162:

        if more_types_in_union_186163:
            # Runtime conditional SSA (line 246)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to MemoizeJac(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'func' (line 247)
        func_186165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 25), 'func', False)
        # Processing the call keyword arguments (line 247)
        kwargs_186166 = {}
        # Getting the type of 'MemoizeJac' (line 247)
        MemoizeJac_186164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 14), 'MemoizeJac', False)
        # Calling MemoizeJac(args, kwargs) (line 247)
        MemoizeJac_call_result_186167 = invoke(stypy.reporting.localization.Localization(__file__, 247, 14), MemoizeJac_186164, *[func_186165], **kwargs_186166)
        
        # Assigning a type to the variable 'fun' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'fun', MemoizeJac_call_result_186167)
        
        # Assigning a Attribute to a Name (line 248):
        
        # Assigning a Attribute to a Name (line 248):
        # Getting the type of 'fun' (line 248)
        fun_186168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 14), 'fun')
        # Obtaining the member 'derivative' of a type (line 248)
        derivative_186169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 14), fun_186168, 'derivative')
        # Assigning a type to the variable 'jac' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'jac', derivative_186169)

        if more_types_in_union_186163:
            # Runtime conditional SSA for else branch (line 246)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_186162) or more_types_in_union_186163):
        
        # Assigning a Name to a Name (line 250):
        
        # Assigning a Name to a Name (line 250):
        # Getting the type of 'func' (line 250)
        func_186170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 14), 'func')
        # Assigning a type to the variable 'fun' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'fun', func_186170)
        
        # Assigning a Name to a Name (line 251):
        
        # Assigning a Name to a Name (line 251):
        # Getting the type of 'fprime' (line 251)
        fprime_186171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 14), 'fprime')
        # Assigning a type to the variable 'jac' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'jac', fprime_186171)

        if (may_be_186162 and more_types_in_union_186163):
            # SSA join for if statement (line 246)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 243)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 253)
    # Getting the type of 'disp' (line 253)
    disp_186172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'disp')
    # Getting the type of 'None' (line 253)
    None_186173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'None')
    
    (may_be_186174, more_types_in_union_186175) = may_not_be_none(disp_186172, None_186173)

    if may_be_186174:

        if more_types_in_union_186175:
            # Runtime conditional SSA (line 253)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 254):
        
        # Assigning a Name to a Name (line 254):
        # Getting the type of 'disp' (line 254)
        disp_186176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 'disp')
        # Assigning a type to the variable 'mesg_num' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'mesg_num', disp_186176)

        if more_types_in_union_186175:
            # Runtime conditional SSA for else branch (line 253)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_186174) or more_types_in_union_186175):
        
        # Assigning a Call to a Name (line 256):
        
        # Assigning a Call to a Name (line 256):
        
        # Call to get(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'messages' (line 257)
        messages_186191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 47), 'messages', False)
        # Getting the type of 'MSG_ALL' (line 257)
        MSG_ALL_186192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 57), 'MSG_ALL', False)
        # Processing the call keyword arguments (line 256)
        kwargs_186193 = {}
        
        # Obtaining an instance of the builtin type 'dict' (line 256)
        dict_186177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 256)
        # Adding element type (key, value) (line 256)
        int_186178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 20), 'int')
        # Getting the type of 'MSG_NONE' (line 256)
        MSG_NONE_186179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 22), 'MSG_NONE', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), dict_186177, (int_186178, MSG_NONE_186179))
        # Adding element type (key, value) (line 256)
        int_186180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 32), 'int')
        # Getting the type of 'MSG_ITER' (line 256)
        MSG_ITER_186181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 34), 'MSG_ITER', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), dict_186177, (int_186180, MSG_ITER_186181))
        # Adding element type (key, value) (line 256)
        int_186182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 44), 'int')
        # Getting the type of 'MSG_INFO' (line 256)
        MSG_INFO_186183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 46), 'MSG_INFO', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), dict_186177, (int_186182, MSG_INFO_186183))
        # Adding element type (key, value) (line 256)
        int_186184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 56), 'int')
        # Getting the type of 'MSG_VERS' (line 256)
        MSG_VERS_186185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 58), 'MSG_VERS', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), dict_186177, (int_186184, MSG_VERS_186185))
        # Adding element type (key, value) (line 256)
        int_186186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 20), 'int')
        # Getting the type of 'MSG_EXIT' (line 257)
        MSG_EXIT_186187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 22), 'MSG_EXIT', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), dict_186177, (int_186186, MSG_EXIT_186187))
        # Adding element type (key, value) (line 256)
        int_186188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 32), 'int')
        # Getting the type of 'MSG_ALL' (line 257)
        MSG_ALL_186189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 34), 'MSG_ALL', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), dict_186177, (int_186188, MSG_ALL_186189))
        
        # Obtaining the member 'get' of a type (line 256)
        get_186190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 19), dict_186177, 'get')
        # Calling get(args, kwargs) (line 256)
        get_call_result_186194 = invoke(stypy.reporting.localization.Localization(__file__, 256, 19), get_186190, *[messages_186191, MSG_ALL_186192], **kwargs_186193)
        
        # Assigning a type to the variable 'mesg_num' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'mesg_num', get_call_result_186194)

        if (may_be_186174 and more_types_in_union_186175):
            # SSA join for if statement (line 253)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Dict to a Name (line 259):
    
    # Assigning a Dict to a Name (line 259):
    
    # Obtaining an instance of the builtin type 'dict' (line 259)
    dict_186195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 259)
    # Adding element type (key, value) (line 259)
    str_186196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 12), 'str', 'eps')
    # Getting the type of 'epsilon' (line 259)
    epsilon_186197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 19), 'epsilon')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186196, epsilon_186197))
    # Adding element type (key, value) (line 259)
    str_186198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 12), 'str', 'scale')
    # Getting the type of 'scale' (line 260)
    scale_186199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 21), 'scale')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186198, scale_186199))
    # Adding element type (key, value) (line 259)
    str_186200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 12), 'str', 'offset')
    # Getting the type of 'offset' (line 261)
    offset_186201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 22), 'offset')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186200, offset_186201))
    # Adding element type (key, value) (line 259)
    str_186202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 12), 'str', 'mesg_num')
    # Getting the type of 'mesg_num' (line 262)
    mesg_num_186203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'mesg_num')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186202, mesg_num_186203))
    # Adding element type (key, value) (line 259)
    str_186204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 12), 'str', 'maxCGit')
    # Getting the type of 'maxCGit' (line 263)
    maxCGit_186205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 23), 'maxCGit')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186204, maxCGit_186205))
    # Adding element type (key, value) (line 259)
    str_186206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 12), 'str', 'maxiter')
    # Getting the type of 'maxfun' (line 264)
    maxfun_186207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 23), 'maxfun')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186206, maxfun_186207))
    # Adding element type (key, value) (line 259)
    str_186208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 12), 'str', 'eta')
    # Getting the type of 'eta' (line 265)
    eta_186209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 19), 'eta')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186208, eta_186209))
    # Adding element type (key, value) (line 259)
    str_186210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 12), 'str', 'stepmx')
    # Getting the type of 'stepmx' (line 266)
    stepmx_186211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'stepmx')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186210, stepmx_186211))
    # Adding element type (key, value) (line 259)
    str_186212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 12), 'str', 'accuracy')
    # Getting the type of 'accuracy' (line 267)
    accuracy_186213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 24), 'accuracy')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186212, accuracy_186213))
    # Adding element type (key, value) (line 259)
    str_186214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 12), 'str', 'minfev')
    # Getting the type of 'fmin' (line 268)
    fmin_186215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'fmin')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186214, fmin_186215))
    # Adding element type (key, value) (line 259)
    str_186216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 12), 'str', 'ftol')
    # Getting the type of 'ftol' (line 269)
    ftol_186217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'ftol')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186216, ftol_186217))
    # Adding element type (key, value) (line 259)
    str_186218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 12), 'str', 'xtol')
    # Getting the type of 'xtol' (line 270)
    xtol_186219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'xtol')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186218, xtol_186219))
    # Adding element type (key, value) (line 259)
    str_186220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 12), 'str', 'gtol')
    # Getting the type of 'pgtol' (line 271)
    pgtol_186221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 20), 'pgtol')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186220, pgtol_186221))
    # Adding element type (key, value) (line 259)
    str_186222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 12), 'str', 'rescale')
    # Getting the type of 'rescale' (line 272)
    rescale_186223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 23), 'rescale')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186222, rescale_186223))
    # Adding element type (key, value) (line 259)
    str_186224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 12), 'str', 'disp')
    # Getting the type of 'False' (line 273)
    False_186225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'False')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), dict_186195, (str_186224, False_186225))
    
    # Assigning a type to the variable 'opts' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'opts', dict_186195)
    
    # Assigning a Call to a Name (line 275):
    
    # Assigning a Call to a Name (line 275):
    
    # Call to _minimize_tnc(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of 'fun' (line 275)
    fun_186227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'fun', False)
    # Getting the type of 'x0' (line 275)
    x0_186228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 29), 'x0', False)
    # Getting the type of 'args' (line 275)
    args_186229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 33), 'args', False)
    # Getting the type of 'jac' (line 275)
    jac_186230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 39), 'jac', False)
    # Getting the type of 'bounds' (line 275)
    bounds_186231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 44), 'bounds', False)
    # Processing the call keyword arguments (line 275)
    # Getting the type of 'callback' (line 275)
    callback_186232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 61), 'callback', False)
    keyword_186233 = callback_186232
    # Getting the type of 'opts' (line 275)
    opts_186234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 73), 'opts', False)
    kwargs_186235 = {'callback': keyword_186233, 'opts_186234': opts_186234}
    # Getting the type of '_minimize_tnc' (line 275)
    _minimize_tnc_186226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 10), '_minimize_tnc', False)
    # Calling _minimize_tnc(args, kwargs) (line 275)
    _minimize_tnc_call_result_186236 = invoke(stypy.reporting.localization.Localization(__file__, 275, 10), _minimize_tnc_186226, *[fun_186227, x0_186228, args_186229, jac_186230, bounds_186231], **kwargs_186235)
    
    # Assigning a type to the variable 'res' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'res', _minimize_tnc_call_result_186236)
    
    # Obtaining an instance of the builtin type 'tuple' (line 277)
    tuple_186237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 277)
    # Adding element type (line 277)
    
    # Obtaining the type of the subscript
    str_186238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 15), 'str', 'x')
    # Getting the type of 'res' (line 277)
    res_186239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 11), 'res')
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___186240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 11), res_186239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_186241 = invoke(stypy.reporting.localization.Localization(__file__, 277, 11), getitem___186240, str_186238)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 11), tuple_186237, subscript_call_result_186241)
    # Adding element type (line 277)
    
    # Obtaining the type of the subscript
    str_186242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 25), 'str', 'nfev')
    # Getting the type of 'res' (line 277)
    res_186243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'res')
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___186244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 21), res_186243, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_186245 = invoke(stypy.reporting.localization.Localization(__file__, 277, 21), getitem___186244, str_186242)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 11), tuple_186237, subscript_call_result_186245)
    # Adding element type (line 277)
    
    # Obtaining the type of the subscript
    str_186246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 38), 'str', 'status')
    # Getting the type of 'res' (line 277)
    res_186247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 34), 'res')
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___186248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 34), res_186247, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_186249 = invoke(stypy.reporting.localization.Localization(__file__, 277, 34), getitem___186248, str_186246)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 11), tuple_186237, subscript_call_result_186249)
    
    # Assigning a type to the variable 'stypy_return_type' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'stypy_return_type', tuple_186237)
    
    # ################# End of 'fmin_tnc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fmin_tnc' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_186250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_186250)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fmin_tnc'
    return stypy_return_type_186250

# Assigning a type to the variable 'fmin_tnc' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'fmin_tnc', fmin_tnc)

@norecursion
def _minimize_tnc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 280)
    tuple_186251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 280)
    
    # Getting the type of 'None' (line 280)
    None_186252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 40), 'None')
    # Getting the type of 'None' (line 280)
    None_186253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 53), 'None')
    float_186254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 22), 'float')
    # Getting the type of 'None' (line 281)
    None_186255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 34), 'None')
    # Getting the type of 'None' (line 281)
    None_186256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 47), 'None')
    # Getting the type of 'None' (line 281)
    None_186257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 62), 'None')
    int_186258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 26), 'int')
    # Getting the type of 'None' (line 282)
    None_186259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 38), 'None')
    int_186260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 48), 'int')
    int_186261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 59), 'int')
    int_186262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 71), 'int')
    int_186263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 25), 'int')
    int_186264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 33), 'int')
    int_186265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 42), 'int')
    int_186266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 51), 'int')
    int_186267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 63), 'int')
    # Getting the type of 'False' (line 283)
    False_186268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 72), 'False')
    # Getting the type of 'None' (line 284)
    None_186269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 27), 'None')
    defaults = [tuple_186251, None_186252, None_186253, float_186254, None_186255, None_186256, None_186257, int_186258, None_186259, int_186260, int_186261, int_186262, int_186263, int_186264, int_186265, int_186266, int_186267, False_186268, None_186269]
    # Create a new context for function '_minimize_tnc'
    module_type_store = module_type_store.open_function_context('_minimize_tnc', 280, 0, False)
    
    # Passed parameters checking function
    _minimize_tnc.stypy_localization = localization
    _minimize_tnc.stypy_type_of_self = None
    _minimize_tnc.stypy_type_store = module_type_store
    _minimize_tnc.stypy_function_name = '_minimize_tnc'
    _minimize_tnc.stypy_param_names_list = ['fun', 'x0', 'args', 'jac', 'bounds', 'eps', 'scale', 'offset', 'mesg_num', 'maxCGit', 'maxiter', 'eta', 'stepmx', 'accuracy', 'minfev', 'ftol', 'xtol', 'gtol', 'rescale', 'disp', 'callback']
    _minimize_tnc.stypy_varargs_param_name = None
    _minimize_tnc.stypy_kwargs_param_name = 'unknown_options'
    _minimize_tnc.stypy_call_defaults = defaults
    _minimize_tnc.stypy_call_varargs = varargs
    _minimize_tnc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_minimize_tnc', ['fun', 'x0', 'args', 'jac', 'bounds', 'eps', 'scale', 'offset', 'mesg_num', 'maxCGit', 'maxiter', 'eta', 'stepmx', 'accuracy', 'minfev', 'ftol', 'xtol', 'gtol', 'rescale', 'disp', 'callback'], None, 'unknown_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_minimize_tnc', localization, ['fun', 'x0', 'args', 'jac', 'bounds', 'eps', 'scale', 'offset', 'mesg_num', 'maxCGit', 'maxiter', 'eta', 'stepmx', 'accuracy', 'minfev', 'ftol', 'xtol', 'gtol', 'rescale', 'disp', 'callback'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_minimize_tnc(...)' code ##################

    str_186270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, (-1)), 'str', '\n    Minimize a scalar function of one or more variables using a truncated\n    Newton (TNC) algorithm.\n\n    Options\n    -------\n    eps : float\n        Step size used for numerical approximation of the jacobian.\n    scale : list of floats\n        Scaling factors to apply to each variable.  If None, the\n        factors are up-low for interval bounded variables and\n        1+|x] fo the others.  Defaults to None\n    offset : float\n        Value to subtract from each variable.  If None, the\n        offsets are (up+low)/2 for interval bounded variables\n        and x for the others.\n    disp : bool\n       Set to True to print convergence messages.\n    maxCGit : int\n        Maximum number of hessian*vector evaluations per main\n        iteration.  If maxCGit == 0, the direction chosen is\n        -gradient if maxCGit < 0, maxCGit is set to\n        max(1,min(50,n/2)).  Defaults to -1.\n    maxiter : int\n        Maximum number of function evaluation.  if None, `maxiter` is\n        set to max(100, 10*len(x0)).  Defaults to None.\n    eta : float\n        Severity of the line search. if < 0 or > 1, set to 0.25.\n        Defaults to -1.\n    stepmx : float\n        Maximum step for the line search.  May be increased during\n        call.  If too small, it will be set to 10.0.  Defaults to 0.\n    accuracy : float\n        Relative precision for finite difference calculations.  If\n        <= machine_precision, set to sqrt(machine_precision).\n        Defaults to 0.\n    minfev : float\n        Minimum function value estimate.  Defaults to 0.\n    ftol : float\n        Precision goal for the value of f in the stoping criterion.\n        If ftol < 0.0, ftol is set to 0.0 defaults to -1.\n    xtol : float\n        Precision goal for the value of x in the stopping\n        criterion (after applying x scaling factors).  If xtol <\n        0.0, xtol is set to sqrt(machine_precision).  Defaults to\n        -1.\n    gtol : float\n        Precision goal for the value of the projected gradient in\n        the stopping criterion (after applying x scaling factors).\n        If gtol < 0.0, gtol is set to 1e-2 * sqrt(accuracy).\n        Setting it to 0.0 is not recommended.  Defaults to -1.\n    rescale : float\n        Scaling factor (in log10) used to trigger f value\n        rescaling.  If 0, rescale at each iteration.  If a large\n        value, never rescale.  If < 0, rescale is set to 1.3.\n\n    ')
    
    # Call to _check_unknown_options(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'unknown_options' (line 342)
    unknown_options_186272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 27), 'unknown_options', False)
    # Processing the call keyword arguments (line 342)
    kwargs_186273 = {}
    # Getting the type of '_check_unknown_options' (line 342)
    _check_unknown_options_186271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), '_check_unknown_options', False)
    # Calling _check_unknown_options(args, kwargs) (line 342)
    _check_unknown_options_call_result_186274 = invoke(stypy.reporting.localization.Localization(__file__, 342, 4), _check_unknown_options_186271, *[unknown_options_186272], **kwargs_186273)
    
    
    # Assigning a Name to a Name (line 343):
    
    # Assigning a Name to a Name (line 343):
    # Getting the type of 'eps' (line 343)
    eps_186275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 14), 'eps')
    # Assigning a type to the variable 'epsilon' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'epsilon', eps_186275)
    
    # Assigning a Name to a Name (line 344):
    
    # Assigning a Name to a Name (line 344):
    # Getting the type of 'maxiter' (line 344)
    maxiter_186276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 13), 'maxiter')
    # Assigning a type to the variable 'maxfun' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'maxfun', maxiter_186276)
    
    # Assigning a Name to a Name (line 345):
    
    # Assigning a Name to a Name (line 345):
    # Getting the type of 'minfev' (line 345)
    minfev_186277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 11), 'minfev')
    # Assigning a type to the variable 'fmin' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'fmin', minfev_186277)
    
    # Assigning a Name to a Name (line 346):
    
    # Assigning a Name to a Name (line 346):
    # Getting the type of 'gtol' (line 346)
    gtol_186278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'gtol')
    # Assigning a type to the variable 'pgtol' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'pgtol', gtol_186278)
    
    # Assigning a Call to a Name (line 348):
    
    # Assigning a Call to a Name (line 348):
    
    # Call to flatten(...): (line 348)
    # Processing the call keyword arguments (line 348)
    kwargs_186284 = {}
    
    # Call to asfarray(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'x0' (line 348)
    x0_186280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 18), 'x0', False)
    # Processing the call keyword arguments (line 348)
    kwargs_186281 = {}
    # Getting the type of 'asfarray' (line 348)
    asfarray_186279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 9), 'asfarray', False)
    # Calling asfarray(args, kwargs) (line 348)
    asfarray_call_result_186282 = invoke(stypy.reporting.localization.Localization(__file__, 348, 9), asfarray_186279, *[x0_186280], **kwargs_186281)
    
    # Obtaining the member 'flatten' of a type (line 348)
    flatten_186283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 9), asfarray_call_result_186282, 'flatten')
    # Calling flatten(args, kwargs) (line 348)
    flatten_call_result_186285 = invoke(stypy.reporting.localization.Localization(__file__, 348, 9), flatten_186283, *[], **kwargs_186284)
    
    # Assigning a type to the variable 'x0' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'x0', flatten_call_result_186285)
    
    # Assigning a Call to a Name (line 349):
    
    # Assigning a Call to a Name (line 349):
    
    # Call to len(...): (line 349)
    # Processing the call arguments (line 349)
    # Getting the type of 'x0' (line 349)
    x0_186287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'x0', False)
    # Processing the call keyword arguments (line 349)
    kwargs_186288 = {}
    # Getting the type of 'len' (line 349)
    len_186286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'len', False)
    # Calling len(args, kwargs) (line 349)
    len_call_result_186289 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), len_186286, *[x0_186287], **kwargs_186288)
    
    # Assigning a type to the variable 'n' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'n', len_call_result_186289)
    
    # Type idiom detected: calculating its left and rigth part (line 351)
    # Getting the type of 'bounds' (line 351)
    bounds_186290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 7), 'bounds')
    # Getting the type of 'None' (line 351)
    None_186291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 17), 'None')
    
    (may_be_186292, more_types_in_union_186293) = may_be_none(bounds_186290, None_186291)

    if may_be_186292:

        if more_types_in_union_186293:
            # Runtime conditional SSA (line 351)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 352):
        
        # Assigning a BinOp to a Name (line 352):
        
        # Obtaining an instance of the builtin type 'list' (line 352)
        list_186294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 352)
        # Adding element type (line 352)
        
        # Obtaining an instance of the builtin type 'tuple' (line 352)
        tuple_186295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 352)
        # Adding element type (line 352)
        # Getting the type of 'None' (line 352)
        None_186296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 19), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 19), tuple_186295, None_186296)
        # Adding element type (line 352)
        # Getting the type of 'None' (line 352)
        None_186297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 24), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 19), tuple_186295, None_186297)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 17), list_186294, tuple_186295)
        
        # Getting the type of 'n' (line 352)
        n_186298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 33), 'n')
        # Applying the binary operator '*' (line 352)
        result_mul_186299 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 17), '*', list_186294, n_186298)
        
        # Assigning a type to the variable 'bounds' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'bounds', result_mul_186299)

        if more_types_in_union_186293:
            # SSA join for if statement (line 351)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to len(...): (line 353)
    # Processing the call arguments (line 353)
    # Getting the type of 'bounds' (line 353)
    bounds_186301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 11), 'bounds', False)
    # Processing the call keyword arguments (line 353)
    kwargs_186302 = {}
    # Getting the type of 'len' (line 353)
    len_186300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 7), 'len', False)
    # Calling len(args, kwargs) (line 353)
    len_call_result_186303 = invoke(stypy.reporting.localization.Localization(__file__, 353, 7), len_186300, *[bounds_186301], **kwargs_186302)
    
    # Getting the type of 'n' (line 353)
    n_186304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 22), 'n')
    # Applying the binary operator '!=' (line 353)
    result_ne_186305 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 7), '!=', len_call_result_186303, n_186304)
    
    # Testing the type of an if condition (line 353)
    if_condition_186306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 353, 4), result_ne_186305)
    # Assigning a type to the variable 'if_condition_186306' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'if_condition_186306', if_condition_186306)
    # SSA begins for if statement (line 353)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 354)
    # Processing the call arguments (line 354)
    str_186308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 25), 'str', 'length of x0 != length of bounds')
    # Processing the call keyword arguments (line 354)
    kwargs_186309 = {}
    # Getting the type of 'ValueError' (line 354)
    ValueError_186307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 354)
    ValueError_call_result_186310 = invoke(stypy.reporting.localization.Localization(__file__, 354, 14), ValueError_186307, *[str_186308], **kwargs_186309)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 354, 8), ValueError_call_result_186310, 'raise parameter', BaseException)
    # SSA join for if statement (line 353)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 356)
    # Getting the type of 'mesg_num' (line 356)
    mesg_num_186311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'mesg_num')
    # Getting the type of 'None' (line 356)
    None_186312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 23), 'None')
    
    (may_be_186313, more_types_in_union_186314) = may_not_be_none(mesg_num_186311, None_186312)

    if may_be_186313:

        if more_types_in_union_186314:
            # Runtime conditional SSA (line 356)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 357):
        
        # Assigning a Call to a Name (line 357):
        
        # Call to get(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'mesg_num' (line 358)
        mesg_num_186329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 47), 'mesg_num', False)
        # Getting the type of 'MSG_ALL' (line 358)
        MSG_ALL_186330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 57), 'MSG_ALL', False)
        # Processing the call keyword arguments (line 357)
        kwargs_186331 = {}
        
        # Obtaining an instance of the builtin type 'dict' (line 357)
        dict_186315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 357)
        # Adding element type (key, value) (line 357)
        int_186316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 20), 'int')
        # Getting the type of 'MSG_NONE' (line 357)
        MSG_NONE_186317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 22), 'MSG_NONE', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 19), dict_186315, (int_186316, MSG_NONE_186317))
        # Adding element type (key, value) (line 357)
        int_186318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 32), 'int')
        # Getting the type of 'MSG_ITER' (line 357)
        MSG_ITER_186319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 34), 'MSG_ITER', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 19), dict_186315, (int_186318, MSG_ITER_186319))
        # Adding element type (key, value) (line 357)
        int_186320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 44), 'int')
        # Getting the type of 'MSG_INFO' (line 357)
        MSG_INFO_186321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 46), 'MSG_INFO', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 19), dict_186315, (int_186320, MSG_INFO_186321))
        # Adding element type (key, value) (line 357)
        int_186322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 56), 'int')
        # Getting the type of 'MSG_VERS' (line 357)
        MSG_VERS_186323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 58), 'MSG_VERS', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 19), dict_186315, (int_186322, MSG_VERS_186323))
        # Adding element type (key, value) (line 357)
        int_186324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 20), 'int')
        # Getting the type of 'MSG_EXIT' (line 358)
        MSG_EXIT_186325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 22), 'MSG_EXIT', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 19), dict_186315, (int_186324, MSG_EXIT_186325))
        # Adding element type (key, value) (line 357)
        int_186326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 32), 'int')
        # Getting the type of 'MSG_ALL' (line 358)
        MSG_ALL_186327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 34), 'MSG_ALL', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 19), dict_186315, (int_186326, MSG_ALL_186327))
        
        # Obtaining the member 'get' of a type (line 357)
        get_186328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 19), dict_186315, 'get')
        # Calling get(args, kwargs) (line 357)
        get_call_result_186332 = invoke(stypy.reporting.localization.Localization(__file__, 357, 19), get_186328, *[mesg_num_186329, MSG_ALL_186330], **kwargs_186331)
        
        # Assigning a type to the variable 'messages' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'messages', get_call_result_186332)

        if more_types_in_union_186314:
            # Runtime conditional SSA for else branch (line 356)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_186313) or more_types_in_union_186314):
        
        # Getting the type of 'disp' (line 359)
        disp_186333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 9), 'disp')
        # Testing the type of an if condition (line 359)
        if_condition_186334 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 9), disp_186333)
        # Assigning a type to the variable 'if_condition_186334' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 9), 'if_condition_186334', if_condition_186334)
        # SSA begins for if statement (line 359)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 360):
        
        # Assigning a Name to a Name (line 360):
        # Getting the type of 'MSG_ALL' (line 360)
        MSG_ALL_186335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 19), 'MSG_ALL')
        # Assigning a type to the variable 'messages' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'messages', MSG_ALL_186335)
        # SSA branch for the else part of an if statement (line 359)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 362):
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'MSG_NONE' (line 362)
        MSG_NONE_186336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'MSG_NONE')
        # Assigning a type to the variable 'messages' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'messages', MSG_NONE_186336)
        # SSA join for if statement (line 359)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_186313 and more_types_in_union_186314):
            # SSA join for if statement (line 356)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 364)
    # Getting the type of 'jac' (line 364)
    jac_186337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 7), 'jac')
    # Getting the type of 'None' (line 364)
    None_186338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 14), 'None')
    
    (may_be_186339, more_types_in_union_186340) = may_be_none(jac_186337, None_186338)

    if may_be_186339:

        if more_types_in_union_186340:
            # Runtime conditional SSA (line 364)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def func_and_grad(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func_and_grad'
            module_type_store = module_type_store.open_function_context('func_and_grad', 365, 8, False)
            
            # Passed parameters checking function
            func_and_grad.stypy_localization = localization
            func_and_grad.stypy_type_of_self = None
            func_and_grad.stypy_type_store = module_type_store
            func_and_grad.stypy_function_name = 'func_and_grad'
            func_and_grad.stypy_param_names_list = ['x']
            func_and_grad.stypy_varargs_param_name = None
            func_and_grad.stypy_kwargs_param_name = None
            func_and_grad.stypy_call_defaults = defaults
            func_and_grad.stypy_call_varargs = varargs
            func_and_grad.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func_and_grad', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func_and_grad', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func_and_grad(...)' code ##################

            
            # Assigning a Call to a Name (line 366):
            
            # Assigning a Call to a Name (line 366):
            
            # Call to fun(...): (line 366)
            # Processing the call arguments (line 366)
            # Getting the type of 'x' (line 366)
            x_186342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 20), 'x', False)
            # Getting the type of 'args' (line 366)
            args_186343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 24), 'args', False)
            # Processing the call keyword arguments (line 366)
            kwargs_186344 = {}
            # Getting the type of 'fun' (line 366)
            fun_186341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'fun', False)
            # Calling fun(args, kwargs) (line 366)
            fun_call_result_186345 = invoke(stypy.reporting.localization.Localization(__file__, 366, 16), fun_186341, *[x_186342, args_186343], **kwargs_186344)
            
            # Assigning a type to the variable 'f' (line 366)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'f', fun_call_result_186345)
            
            # Assigning a Call to a Name (line 367):
            
            # Assigning a Call to a Name (line 367):
            
            # Call to approx_fprime(...): (line 367)
            # Processing the call arguments (line 367)
            # Getting the type of 'x' (line 367)
            x_186347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 30), 'x', False)
            # Getting the type of 'fun' (line 367)
            fun_186348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 33), 'fun', False)
            # Getting the type of 'epsilon' (line 367)
            epsilon_186349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 38), 'epsilon', False)
            # Getting the type of 'args' (line 367)
            args_186350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 48), 'args', False)
            # Processing the call keyword arguments (line 367)
            kwargs_186351 = {}
            # Getting the type of 'approx_fprime' (line 367)
            approx_fprime_186346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'approx_fprime', False)
            # Calling approx_fprime(args, kwargs) (line 367)
            approx_fprime_call_result_186352 = invoke(stypy.reporting.localization.Localization(__file__, 367, 16), approx_fprime_186346, *[x_186347, fun_186348, epsilon_186349, args_186350], **kwargs_186351)
            
            # Assigning a type to the variable 'g' (line 367)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'g', approx_fprime_call_result_186352)
            
            # Obtaining an instance of the builtin type 'tuple' (line 368)
            tuple_186353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 368)
            # Adding element type (line 368)
            # Getting the type of 'f' (line 368)
            f_186354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 19), 'f')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 19), tuple_186353, f_186354)
            # Adding element type (line 368)
            # Getting the type of 'g' (line 368)
            g_186355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 22), 'g')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 19), tuple_186353, g_186355)
            
            # Assigning a type to the variable 'stypy_return_type' (line 368)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'stypy_return_type', tuple_186353)
            
            # ################# End of 'func_and_grad(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func_and_grad' in the type store
            # Getting the type of 'stypy_return_type' (line 365)
            stypy_return_type_186356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_186356)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func_and_grad'
            return stypy_return_type_186356

        # Assigning a type to the variable 'func_and_grad' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'func_and_grad', func_and_grad)

        if more_types_in_union_186340:
            # Runtime conditional SSA for else branch (line 364)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_186339) or more_types_in_union_186340):

        @norecursion
        def func_and_grad(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func_and_grad'
            module_type_store = module_type_store.open_function_context('func_and_grad', 370, 8, False)
            
            # Passed parameters checking function
            func_and_grad.stypy_localization = localization
            func_and_grad.stypy_type_of_self = None
            func_and_grad.stypy_type_store = module_type_store
            func_and_grad.stypy_function_name = 'func_and_grad'
            func_and_grad.stypy_param_names_list = ['x']
            func_and_grad.stypy_varargs_param_name = None
            func_and_grad.stypy_kwargs_param_name = None
            func_and_grad.stypy_call_defaults = defaults
            func_and_grad.stypy_call_varargs = varargs
            func_and_grad.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func_and_grad', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func_and_grad', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func_and_grad(...)' code ##################

            
            # Assigning a Call to a Name (line 371):
            
            # Assigning a Call to a Name (line 371):
            
            # Call to fun(...): (line 371)
            # Processing the call arguments (line 371)
            # Getting the type of 'x' (line 371)
            x_186358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 20), 'x', False)
            # Getting the type of 'args' (line 371)
            args_186359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'args', False)
            # Processing the call keyword arguments (line 371)
            kwargs_186360 = {}
            # Getting the type of 'fun' (line 371)
            fun_186357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'fun', False)
            # Calling fun(args, kwargs) (line 371)
            fun_call_result_186361 = invoke(stypy.reporting.localization.Localization(__file__, 371, 16), fun_186357, *[x_186358, args_186359], **kwargs_186360)
            
            # Assigning a type to the variable 'f' (line 371)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'f', fun_call_result_186361)
            
            # Assigning a Call to a Name (line 372):
            
            # Assigning a Call to a Name (line 372):
            
            # Call to jac(...): (line 372)
            # Processing the call arguments (line 372)
            # Getting the type of 'x' (line 372)
            x_186363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'x', False)
            # Getting the type of 'args' (line 372)
            args_186364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 24), 'args', False)
            # Processing the call keyword arguments (line 372)
            kwargs_186365 = {}
            # Getting the type of 'jac' (line 372)
            jac_186362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'jac', False)
            # Calling jac(args, kwargs) (line 372)
            jac_call_result_186366 = invoke(stypy.reporting.localization.Localization(__file__, 372, 16), jac_186362, *[x_186363, args_186364], **kwargs_186365)
            
            # Assigning a type to the variable 'g' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'g', jac_call_result_186366)
            
            # Obtaining an instance of the builtin type 'tuple' (line 373)
            tuple_186367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 373)
            # Adding element type (line 373)
            # Getting the type of 'f' (line 373)
            f_186368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 19), 'f')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 19), tuple_186367, f_186368)
            # Adding element type (line 373)
            # Getting the type of 'g' (line 373)
            g_186369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), 'g')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 19), tuple_186367, g_186369)
            
            # Assigning a type to the variable 'stypy_return_type' (line 373)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'stypy_return_type', tuple_186367)
            
            # ################# End of 'func_and_grad(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func_and_grad' in the type store
            # Getting the type of 'stypy_return_type' (line 370)
            stypy_return_type_186370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_186370)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func_and_grad'
            return stypy_return_type_186370

        # Assigning a type to the variable 'func_and_grad' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'func_and_grad', func_and_grad)

        if (may_be_186339 and more_types_in_union_186340):
            # SSA join for if statement (line 364)
            module_type_store = module_type_store.join_ssa_context()


    
    str_186371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, (-1)), 'str', '\n    low, up   : the bounds (lists of floats)\n                if low is None, the lower bounds are removed.\n                if up is None, the upper bounds are removed.\n                low and up defaults to None\n    ')
    
    # Assigning a Call to a Name (line 381):
    
    # Assigning a Call to a Name (line 381):
    
    # Call to zeros(...): (line 381)
    # Processing the call arguments (line 381)
    # Getting the type of 'n' (line 381)
    n_186373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'n', False)
    # Processing the call keyword arguments (line 381)
    kwargs_186374 = {}
    # Getting the type of 'zeros' (line 381)
    zeros_186372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 10), 'zeros', False)
    # Calling zeros(args, kwargs) (line 381)
    zeros_call_result_186375 = invoke(stypy.reporting.localization.Localization(__file__, 381, 10), zeros_186372, *[n_186373], **kwargs_186374)
    
    # Assigning a type to the variable 'low' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'low', zeros_call_result_186375)
    
    # Assigning a Call to a Name (line 382):
    
    # Assigning a Call to a Name (line 382):
    
    # Call to zeros(...): (line 382)
    # Processing the call arguments (line 382)
    # Getting the type of 'n' (line 382)
    n_186377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 15), 'n', False)
    # Processing the call keyword arguments (line 382)
    kwargs_186378 = {}
    # Getting the type of 'zeros' (line 382)
    zeros_186376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 9), 'zeros', False)
    # Calling zeros(args, kwargs) (line 382)
    zeros_call_result_186379 = invoke(stypy.reporting.localization.Localization(__file__, 382, 9), zeros_186376, *[n_186377], **kwargs_186378)
    
    # Assigning a type to the variable 'up' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'up', zeros_call_result_186379)
    
    
    # Call to range(...): (line 383)
    # Processing the call arguments (line 383)
    # Getting the type of 'n' (line 383)
    n_186381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'n', False)
    # Processing the call keyword arguments (line 383)
    kwargs_186382 = {}
    # Getting the type of 'range' (line 383)
    range_186380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 13), 'range', False)
    # Calling range(args, kwargs) (line 383)
    range_call_result_186383 = invoke(stypy.reporting.localization.Localization(__file__, 383, 13), range_186380, *[n_186381], **kwargs_186382)
    
    # Testing the type of a for loop iterable (line 383)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 383, 4), range_call_result_186383)
    # Getting the type of the for loop variable (line 383)
    for_loop_var_186384 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 383, 4), range_call_result_186383)
    # Assigning a type to the variable 'i' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'i', for_loop_var_186384)
    # SSA begins for a for statement (line 383)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 384)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 384)
    i_186385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 18), 'i')
    # Getting the type of 'bounds' (line 384)
    bounds_186386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 11), 'bounds')
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___186387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 11), bounds_186386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_186388 = invoke(stypy.reporting.localization.Localization(__file__, 384, 11), getitem___186387, i_186385)
    
    # Getting the type of 'None' (line 384)
    None_186389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'None')
    
    (may_be_186390, more_types_in_union_186391) = may_be_none(subscript_call_result_186388, None_186389)

    if may_be_186390:

        if more_types_in_union_186391:
            # Runtime conditional SSA (line 384)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Tuple to a Tuple (line 385):
        
        # Assigning a UnaryOp to a Name (line 385):
        
        # Getting the type of 'inf' (line 385)
        inf_186392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'inf')
        # Applying the 'usub' unary operator (line 385)
        result___neg___186393 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 19), 'usub', inf_186392)
        
        # Assigning a type to the variable 'tuple_assignment_186060' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'tuple_assignment_186060', result___neg___186393)
        
        # Assigning a Name to a Name (line 385):
        # Getting the type of 'inf' (line 385)
        inf_186394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 25), 'inf')
        # Assigning a type to the variable 'tuple_assignment_186061' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'tuple_assignment_186061', inf_186394)
        
        # Assigning a Name to a Name (line 385):
        # Getting the type of 'tuple_assignment_186060' (line 385)
        tuple_assignment_186060_186395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'tuple_assignment_186060')
        # Assigning a type to the variable 'l' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'l', tuple_assignment_186060_186395)
        
        # Assigning a Name to a Name (line 385):
        # Getting the type of 'tuple_assignment_186061' (line 385)
        tuple_assignment_186061_186396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'tuple_assignment_186061')
        # Assigning a type to the variable 'u' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 15), 'u', tuple_assignment_186061_186396)

        if more_types_in_union_186391:
            # Runtime conditional SSA for else branch (line 384)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_186390) or more_types_in_union_186391):
        
        # Assigning a Subscript to a Tuple (line 387):
        
        # Assigning a Subscript to a Name (line 387):
        
        # Obtaining the type of the subscript
        int_186397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 387)
        i_186398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 25), 'i')
        # Getting the type of 'bounds' (line 387)
        bounds_186399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 18), 'bounds')
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___186400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 18), bounds_186399, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_186401 = invoke(stypy.reporting.localization.Localization(__file__, 387, 18), getitem___186400, i_186398)
        
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___186402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), subscript_call_result_186401, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_186403 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), getitem___186402, int_186397)
        
        # Assigning a type to the variable 'tuple_var_assignment_186062' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'tuple_var_assignment_186062', subscript_call_result_186403)
        
        # Assigning a Subscript to a Name (line 387):
        
        # Obtaining the type of the subscript
        int_186404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 387)
        i_186405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 25), 'i')
        # Getting the type of 'bounds' (line 387)
        bounds_186406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 18), 'bounds')
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___186407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 18), bounds_186406, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_186408 = invoke(stypy.reporting.localization.Localization(__file__, 387, 18), getitem___186407, i_186405)
        
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___186409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), subscript_call_result_186408, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_186410 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), getitem___186409, int_186404)
        
        # Assigning a type to the variable 'tuple_var_assignment_186063' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'tuple_var_assignment_186063', subscript_call_result_186410)
        
        # Assigning a Name to a Name (line 387):
        # Getting the type of 'tuple_var_assignment_186062' (line 387)
        tuple_var_assignment_186062_186411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'tuple_var_assignment_186062')
        # Assigning a type to the variable 'l' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'l', tuple_var_assignment_186062_186411)
        
        # Assigning a Name to a Name (line 387):
        # Getting the type of 'tuple_var_assignment_186063' (line 387)
        tuple_var_assignment_186063_186412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'tuple_var_assignment_186063')
        # Assigning a type to the variable 'u' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 14), 'u', tuple_var_assignment_186063_186412)
        
        # Type idiom detected: calculating its left and rigth part (line 388)
        # Getting the type of 'l' (line 388)
        l_186413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'l')
        # Getting the type of 'None' (line 388)
        None_186414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'None')
        
        (may_be_186415, more_types_in_union_186416) = may_be_none(l_186413, None_186414)

        if may_be_186415:

            if more_types_in_union_186416:
                # Runtime conditional SSA (line 388)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a UnaryOp to a Subscript (line 389):
            
            # Assigning a UnaryOp to a Subscript (line 389):
            
            # Getting the type of 'inf' (line 389)
            inf_186417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 26), 'inf')
            # Applying the 'usub' unary operator (line 389)
            result___neg___186418 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 25), 'usub', inf_186417)
            
            # Getting the type of 'low' (line 389)
            low_186419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'low')
            # Getting the type of 'i' (line 389)
            i_186420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 20), 'i')
            # Storing an element on a container (line 389)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 16), low_186419, (i_186420, result___neg___186418))

            if more_types_in_union_186416:
                # Runtime conditional SSA for else branch (line 388)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_186415) or more_types_in_union_186416):
            
            # Assigning a Name to a Subscript (line 391):
            
            # Assigning a Name to a Subscript (line 391):
            # Getting the type of 'l' (line 391)
            l_186421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 25), 'l')
            # Getting the type of 'low' (line 391)
            low_186422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'low')
            # Getting the type of 'i' (line 391)
            i_186423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'i')
            # Storing an element on a container (line 391)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 16), low_186422, (i_186423, l_186421))

            if (may_be_186415 and more_types_in_union_186416):
                # SSA join for if statement (line 388)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 392)
        # Getting the type of 'u' (line 392)
        u_186424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 15), 'u')
        # Getting the type of 'None' (line 392)
        None_186425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 20), 'None')
        
        (may_be_186426, more_types_in_union_186427) = may_be_none(u_186424, None_186425)

        if may_be_186426:

            if more_types_in_union_186427:
                # Runtime conditional SSA (line 392)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 393):
            
            # Assigning a Name to a Subscript (line 393):
            # Getting the type of 'inf' (line 393)
            inf_186428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 24), 'inf')
            # Getting the type of 'up' (line 393)
            up_186429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 16), 'up')
            # Getting the type of 'i' (line 393)
            i_186430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 19), 'i')
            # Storing an element on a container (line 393)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 16), up_186429, (i_186430, inf_186428))

            if more_types_in_union_186427:
                # Runtime conditional SSA for else branch (line 392)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_186426) or more_types_in_union_186427):
            
            # Assigning a Name to a Subscript (line 395):
            
            # Assigning a Name to a Subscript (line 395):
            # Getting the type of 'u' (line 395)
            u_186431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 24), 'u')
            # Getting the type of 'up' (line 395)
            up_186432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'up')
            # Getting the type of 'i' (line 395)
            i_186433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'i')
            # Storing an element on a container (line 395)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 16), up_186432, (i_186433, u_186431))

            if (may_be_186426 and more_types_in_union_186427):
                # SSA join for if statement (line 392)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_186390 and more_types_in_union_186391):
            # SSA join for if statement (line 384)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 397)
    # Getting the type of 'scale' (line 397)
    scale_186434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 7), 'scale')
    # Getting the type of 'None' (line 397)
    None_186435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'None')
    
    (may_be_186436, more_types_in_union_186437) = may_be_none(scale_186434, None_186435)

    if may_be_186436:

        if more_types_in_union_186437:
            # Runtime conditional SSA (line 397)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 398):
        
        # Assigning a Call to a Name (line 398):
        
        # Call to array(...): (line 398)
        # Processing the call arguments (line 398)
        
        # Obtaining an instance of the builtin type 'list' (line 398)
        list_186439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 398)
        
        # Processing the call keyword arguments (line 398)
        kwargs_186440 = {}
        # Getting the type of 'array' (line 398)
        array_186438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 16), 'array', False)
        # Calling array(args, kwargs) (line 398)
        array_call_result_186441 = invoke(stypy.reporting.localization.Localization(__file__, 398, 16), array_186438, *[list_186439], **kwargs_186440)
        
        # Assigning a type to the variable 'scale' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'scale', array_call_result_186441)

        if more_types_in_union_186437:
            # SSA join for if statement (line 397)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 400)
    # Getting the type of 'offset' (line 400)
    offset_186442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 7), 'offset')
    # Getting the type of 'None' (line 400)
    None_186443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 17), 'None')
    
    (may_be_186444, more_types_in_union_186445) = may_be_none(offset_186442, None_186443)

    if may_be_186444:

        if more_types_in_union_186445:
            # Runtime conditional SSA (line 400)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 401):
        
        # Assigning a Call to a Name (line 401):
        
        # Call to array(...): (line 401)
        # Processing the call arguments (line 401)
        
        # Obtaining an instance of the builtin type 'list' (line 401)
        list_186447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 401)
        
        # Processing the call keyword arguments (line 401)
        kwargs_186448 = {}
        # Getting the type of 'array' (line 401)
        array_186446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 17), 'array', False)
        # Calling array(args, kwargs) (line 401)
        array_call_result_186449 = invoke(stypy.reporting.localization.Localization(__file__, 401, 17), array_186446, *[list_186447], **kwargs_186448)
        
        # Assigning a type to the variable 'offset' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'offset', array_call_result_186449)

        if more_types_in_union_186445:
            # SSA join for if statement (line 400)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 403)
    # Getting the type of 'maxfun' (line 403)
    maxfun_186450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 7), 'maxfun')
    # Getting the type of 'None' (line 403)
    None_186451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 17), 'None')
    
    (may_be_186452, more_types_in_union_186453) = may_be_none(maxfun_186450, None_186451)

    if may_be_186452:

        if more_types_in_union_186453:
            # Runtime conditional SSA (line 403)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 404):
        
        # Assigning a Call to a Name (line 404):
        
        # Call to max(...): (line 404)
        # Processing the call arguments (line 404)
        int_186455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 21), 'int')
        int_186456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 26), 'int')
        
        # Call to len(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'x0' (line 404)
        x0_186458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 33), 'x0', False)
        # Processing the call keyword arguments (line 404)
        kwargs_186459 = {}
        # Getting the type of 'len' (line 404)
        len_186457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 29), 'len', False)
        # Calling len(args, kwargs) (line 404)
        len_call_result_186460 = invoke(stypy.reporting.localization.Localization(__file__, 404, 29), len_186457, *[x0_186458], **kwargs_186459)
        
        # Applying the binary operator '*' (line 404)
        result_mul_186461 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 26), '*', int_186456, len_call_result_186460)
        
        # Processing the call keyword arguments (line 404)
        kwargs_186462 = {}
        # Getting the type of 'max' (line 404)
        max_186454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 17), 'max', False)
        # Calling max(args, kwargs) (line 404)
        max_call_result_186463 = invoke(stypy.reporting.localization.Localization(__file__, 404, 17), max_186454, *[int_186455, result_mul_186461], **kwargs_186462)
        
        # Assigning a type to the variable 'maxfun' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'maxfun', max_call_result_186463)

        if more_types_in_union_186453:
            # SSA join for if statement (line 403)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 406):
    
    # Assigning a Subscript to a Name (line 406):
    
    # Obtaining the type of the subscript
    int_186464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 4), 'int')
    
    # Call to minimize(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'func_and_grad' (line 406)
    func_and_grad_186467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 40), 'func_and_grad', False)
    # Getting the type of 'x0' (line 406)
    x0_186468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 55), 'x0', False)
    # Getting the type of 'low' (line 406)
    low_186469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 59), 'low', False)
    # Getting the type of 'up' (line 406)
    up_186470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 64), 'up', False)
    # Getting the type of 'scale' (line 406)
    scale_186471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 68), 'scale', False)
    # Getting the type of 'offset' (line 407)
    offset_186472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 40), 'offset', False)
    # Getting the type of 'messages' (line 407)
    messages_186473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 48), 'messages', False)
    # Getting the type of 'maxCGit' (line 407)
    maxCGit_186474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 58), 'maxCGit', False)
    # Getting the type of 'maxfun' (line 407)
    maxfun_186475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 67), 'maxfun', False)
    # Getting the type of 'eta' (line 408)
    eta_186476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 40), 'eta', False)
    # Getting the type of 'stepmx' (line 408)
    stepmx_186477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 45), 'stepmx', False)
    # Getting the type of 'accuracy' (line 408)
    accuracy_186478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 53), 'accuracy', False)
    # Getting the type of 'fmin' (line 408)
    fmin_186479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 63), 'fmin', False)
    # Getting the type of 'ftol' (line 408)
    ftol_186480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 69), 'ftol', False)
    # Getting the type of 'xtol' (line 409)
    xtol_186481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 40), 'xtol', False)
    # Getting the type of 'pgtol' (line 409)
    pgtol_186482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 46), 'pgtol', False)
    # Getting the type of 'rescale' (line 409)
    rescale_186483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 53), 'rescale', False)
    # Getting the type of 'callback' (line 409)
    callback_186484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 62), 'callback', False)
    # Processing the call keyword arguments (line 406)
    kwargs_186485 = {}
    # Getting the type of 'moduleTNC' (line 406)
    moduleTNC_186465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 21), 'moduleTNC', False)
    # Obtaining the member 'minimize' of a type (line 406)
    minimize_186466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 21), moduleTNC_186465, 'minimize')
    # Calling minimize(args, kwargs) (line 406)
    minimize_call_result_186486 = invoke(stypy.reporting.localization.Localization(__file__, 406, 21), minimize_186466, *[func_and_grad_186467, x0_186468, low_186469, up_186470, scale_186471, offset_186472, messages_186473, maxCGit_186474, maxfun_186475, eta_186476, stepmx_186477, accuracy_186478, fmin_186479, ftol_186480, xtol_186481, pgtol_186482, rescale_186483, callback_186484], **kwargs_186485)
    
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___186487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 4), minimize_call_result_186486, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_186488 = invoke(stypy.reporting.localization.Localization(__file__, 406, 4), getitem___186487, int_186464)
    
    # Assigning a type to the variable 'tuple_var_assignment_186064' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'tuple_var_assignment_186064', subscript_call_result_186488)
    
    # Assigning a Subscript to a Name (line 406):
    
    # Obtaining the type of the subscript
    int_186489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 4), 'int')
    
    # Call to minimize(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'func_and_grad' (line 406)
    func_and_grad_186492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 40), 'func_and_grad', False)
    # Getting the type of 'x0' (line 406)
    x0_186493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 55), 'x0', False)
    # Getting the type of 'low' (line 406)
    low_186494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 59), 'low', False)
    # Getting the type of 'up' (line 406)
    up_186495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 64), 'up', False)
    # Getting the type of 'scale' (line 406)
    scale_186496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 68), 'scale', False)
    # Getting the type of 'offset' (line 407)
    offset_186497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 40), 'offset', False)
    # Getting the type of 'messages' (line 407)
    messages_186498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 48), 'messages', False)
    # Getting the type of 'maxCGit' (line 407)
    maxCGit_186499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 58), 'maxCGit', False)
    # Getting the type of 'maxfun' (line 407)
    maxfun_186500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 67), 'maxfun', False)
    # Getting the type of 'eta' (line 408)
    eta_186501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 40), 'eta', False)
    # Getting the type of 'stepmx' (line 408)
    stepmx_186502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 45), 'stepmx', False)
    # Getting the type of 'accuracy' (line 408)
    accuracy_186503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 53), 'accuracy', False)
    # Getting the type of 'fmin' (line 408)
    fmin_186504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 63), 'fmin', False)
    # Getting the type of 'ftol' (line 408)
    ftol_186505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 69), 'ftol', False)
    # Getting the type of 'xtol' (line 409)
    xtol_186506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 40), 'xtol', False)
    # Getting the type of 'pgtol' (line 409)
    pgtol_186507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 46), 'pgtol', False)
    # Getting the type of 'rescale' (line 409)
    rescale_186508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 53), 'rescale', False)
    # Getting the type of 'callback' (line 409)
    callback_186509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 62), 'callback', False)
    # Processing the call keyword arguments (line 406)
    kwargs_186510 = {}
    # Getting the type of 'moduleTNC' (line 406)
    moduleTNC_186490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 21), 'moduleTNC', False)
    # Obtaining the member 'minimize' of a type (line 406)
    minimize_186491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 21), moduleTNC_186490, 'minimize')
    # Calling minimize(args, kwargs) (line 406)
    minimize_call_result_186511 = invoke(stypy.reporting.localization.Localization(__file__, 406, 21), minimize_186491, *[func_and_grad_186492, x0_186493, low_186494, up_186495, scale_186496, offset_186497, messages_186498, maxCGit_186499, maxfun_186500, eta_186501, stepmx_186502, accuracy_186503, fmin_186504, ftol_186505, xtol_186506, pgtol_186507, rescale_186508, callback_186509], **kwargs_186510)
    
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___186512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 4), minimize_call_result_186511, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_186513 = invoke(stypy.reporting.localization.Localization(__file__, 406, 4), getitem___186512, int_186489)
    
    # Assigning a type to the variable 'tuple_var_assignment_186065' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'tuple_var_assignment_186065', subscript_call_result_186513)
    
    # Assigning a Subscript to a Name (line 406):
    
    # Obtaining the type of the subscript
    int_186514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 4), 'int')
    
    # Call to minimize(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'func_and_grad' (line 406)
    func_and_grad_186517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 40), 'func_and_grad', False)
    # Getting the type of 'x0' (line 406)
    x0_186518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 55), 'x0', False)
    # Getting the type of 'low' (line 406)
    low_186519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 59), 'low', False)
    # Getting the type of 'up' (line 406)
    up_186520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 64), 'up', False)
    # Getting the type of 'scale' (line 406)
    scale_186521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 68), 'scale', False)
    # Getting the type of 'offset' (line 407)
    offset_186522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 40), 'offset', False)
    # Getting the type of 'messages' (line 407)
    messages_186523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 48), 'messages', False)
    # Getting the type of 'maxCGit' (line 407)
    maxCGit_186524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 58), 'maxCGit', False)
    # Getting the type of 'maxfun' (line 407)
    maxfun_186525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 67), 'maxfun', False)
    # Getting the type of 'eta' (line 408)
    eta_186526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 40), 'eta', False)
    # Getting the type of 'stepmx' (line 408)
    stepmx_186527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 45), 'stepmx', False)
    # Getting the type of 'accuracy' (line 408)
    accuracy_186528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 53), 'accuracy', False)
    # Getting the type of 'fmin' (line 408)
    fmin_186529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 63), 'fmin', False)
    # Getting the type of 'ftol' (line 408)
    ftol_186530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 69), 'ftol', False)
    # Getting the type of 'xtol' (line 409)
    xtol_186531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 40), 'xtol', False)
    # Getting the type of 'pgtol' (line 409)
    pgtol_186532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 46), 'pgtol', False)
    # Getting the type of 'rescale' (line 409)
    rescale_186533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 53), 'rescale', False)
    # Getting the type of 'callback' (line 409)
    callback_186534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 62), 'callback', False)
    # Processing the call keyword arguments (line 406)
    kwargs_186535 = {}
    # Getting the type of 'moduleTNC' (line 406)
    moduleTNC_186515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 21), 'moduleTNC', False)
    # Obtaining the member 'minimize' of a type (line 406)
    minimize_186516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 21), moduleTNC_186515, 'minimize')
    # Calling minimize(args, kwargs) (line 406)
    minimize_call_result_186536 = invoke(stypy.reporting.localization.Localization(__file__, 406, 21), minimize_186516, *[func_and_grad_186517, x0_186518, low_186519, up_186520, scale_186521, offset_186522, messages_186523, maxCGit_186524, maxfun_186525, eta_186526, stepmx_186527, accuracy_186528, fmin_186529, ftol_186530, xtol_186531, pgtol_186532, rescale_186533, callback_186534], **kwargs_186535)
    
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___186537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 4), minimize_call_result_186536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_186538 = invoke(stypy.reporting.localization.Localization(__file__, 406, 4), getitem___186537, int_186514)
    
    # Assigning a type to the variable 'tuple_var_assignment_186066' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'tuple_var_assignment_186066', subscript_call_result_186538)
    
    # Assigning a Subscript to a Name (line 406):
    
    # Obtaining the type of the subscript
    int_186539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 4), 'int')
    
    # Call to minimize(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'func_and_grad' (line 406)
    func_and_grad_186542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 40), 'func_and_grad', False)
    # Getting the type of 'x0' (line 406)
    x0_186543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 55), 'x0', False)
    # Getting the type of 'low' (line 406)
    low_186544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 59), 'low', False)
    # Getting the type of 'up' (line 406)
    up_186545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 64), 'up', False)
    # Getting the type of 'scale' (line 406)
    scale_186546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 68), 'scale', False)
    # Getting the type of 'offset' (line 407)
    offset_186547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 40), 'offset', False)
    # Getting the type of 'messages' (line 407)
    messages_186548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 48), 'messages', False)
    # Getting the type of 'maxCGit' (line 407)
    maxCGit_186549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 58), 'maxCGit', False)
    # Getting the type of 'maxfun' (line 407)
    maxfun_186550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 67), 'maxfun', False)
    # Getting the type of 'eta' (line 408)
    eta_186551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 40), 'eta', False)
    # Getting the type of 'stepmx' (line 408)
    stepmx_186552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 45), 'stepmx', False)
    # Getting the type of 'accuracy' (line 408)
    accuracy_186553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 53), 'accuracy', False)
    # Getting the type of 'fmin' (line 408)
    fmin_186554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 63), 'fmin', False)
    # Getting the type of 'ftol' (line 408)
    ftol_186555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 69), 'ftol', False)
    # Getting the type of 'xtol' (line 409)
    xtol_186556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 40), 'xtol', False)
    # Getting the type of 'pgtol' (line 409)
    pgtol_186557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 46), 'pgtol', False)
    # Getting the type of 'rescale' (line 409)
    rescale_186558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 53), 'rescale', False)
    # Getting the type of 'callback' (line 409)
    callback_186559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 62), 'callback', False)
    # Processing the call keyword arguments (line 406)
    kwargs_186560 = {}
    # Getting the type of 'moduleTNC' (line 406)
    moduleTNC_186540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 21), 'moduleTNC', False)
    # Obtaining the member 'minimize' of a type (line 406)
    minimize_186541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 21), moduleTNC_186540, 'minimize')
    # Calling minimize(args, kwargs) (line 406)
    minimize_call_result_186561 = invoke(stypy.reporting.localization.Localization(__file__, 406, 21), minimize_186541, *[func_and_grad_186542, x0_186543, low_186544, up_186545, scale_186546, offset_186547, messages_186548, maxCGit_186549, maxfun_186550, eta_186551, stepmx_186552, accuracy_186553, fmin_186554, ftol_186555, xtol_186556, pgtol_186557, rescale_186558, callback_186559], **kwargs_186560)
    
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___186562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 4), minimize_call_result_186561, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_186563 = invoke(stypy.reporting.localization.Localization(__file__, 406, 4), getitem___186562, int_186539)
    
    # Assigning a type to the variable 'tuple_var_assignment_186067' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'tuple_var_assignment_186067', subscript_call_result_186563)
    
    # Assigning a Name to a Name (line 406):
    # Getting the type of 'tuple_var_assignment_186064' (line 406)
    tuple_var_assignment_186064_186564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'tuple_var_assignment_186064')
    # Assigning a type to the variable 'rc' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'rc', tuple_var_assignment_186064_186564)
    
    # Assigning a Name to a Name (line 406):
    # Getting the type of 'tuple_var_assignment_186065' (line 406)
    tuple_var_assignment_186065_186565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'tuple_var_assignment_186065')
    # Assigning a type to the variable 'nf' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'nf', tuple_var_assignment_186065_186565)
    
    # Assigning a Name to a Name (line 406):
    # Getting the type of 'tuple_var_assignment_186066' (line 406)
    tuple_var_assignment_186066_186566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'tuple_var_assignment_186066')
    # Assigning a type to the variable 'nit' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'nit', tuple_var_assignment_186066_186566)
    
    # Assigning a Name to a Name (line 406):
    # Getting the type of 'tuple_var_assignment_186067' (line 406)
    tuple_var_assignment_186067_186567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'tuple_var_assignment_186067')
    # Assigning a type to the variable 'x' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 17), 'x', tuple_var_assignment_186067_186567)
    
    # Assigning a Call to a Tuple (line 411):
    
    # Assigning a Subscript to a Name (line 411):
    
    # Obtaining the type of the subscript
    int_186568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 4), 'int')
    
    # Call to func_and_grad(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'x' (line 411)
    x_186570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'x', False)
    # Processing the call keyword arguments (line 411)
    kwargs_186571 = {}
    # Getting the type of 'func_and_grad' (line 411)
    func_and_grad_186569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 17), 'func_and_grad', False)
    # Calling func_and_grad(args, kwargs) (line 411)
    func_and_grad_call_result_186572 = invoke(stypy.reporting.localization.Localization(__file__, 411, 17), func_and_grad_186569, *[x_186570], **kwargs_186571)
    
    # Obtaining the member '__getitem__' of a type (line 411)
    getitem___186573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 4), func_and_grad_call_result_186572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 411)
    subscript_call_result_186574 = invoke(stypy.reporting.localization.Localization(__file__, 411, 4), getitem___186573, int_186568)
    
    # Assigning a type to the variable 'tuple_var_assignment_186068' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'tuple_var_assignment_186068', subscript_call_result_186574)
    
    # Assigning a Subscript to a Name (line 411):
    
    # Obtaining the type of the subscript
    int_186575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 4), 'int')
    
    # Call to func_and_grad(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'x' (line 411)
    x_186577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'x', False)
    # Processing the call keyword arguments (line 411)
    kwargs_186578 = {}
    # Getting the type of 'func_and_grad' (line 411)
    func_and_grad_186576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 17), 'func_and_grad', False)
    # Calling func_and_grad(args, kwargs) (line 411)
    func_and_grad_call_result_186579 = invoke(stypy.reporting.localization.Localization(__file__, 411, 17), func_and_grad_186576, *[x_186577], **kwargs_186578)
    
    # Obtaining the member '__getitem__' of a type (line 411)
    getitem___186580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 4), func_and_grad_call_result_186579, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 411)
    subscript_call_result_186581 = invoke(stypy.reporting.localization.Localization(__file__, 411, 4), getitem___186580, int_186575)
    
    # Assigning a type to the variable 'tuple_var_assignment_186069' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'tuple_var_assignment_186069', subscript_call_result_186581)
    
    # Assigning a Name to a Name (line 411):
    # Getting the type of 'tuple_var_assignment_186068' (line 411)
    tuple_var_assignment_186068_186582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'tuple_var_assignment_186068')
    # Assigning a type to the variable 'funv' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'funv', tuple_var_assignment_186068_186582)
    
    # Assigning a Name to a Name (line 411):
    # Getting the type of 'tuple_var_assignment_186069' (line 411)
    tuple_var_assignment_186069_186583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'tuple_var_assignment_186069')
    # Assigning a type to the variable 'jacv' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 10), 'jacv', tuple_var_assignment_186069_186583)
    
    # Call to OptimizeResult(...): (line 413)
    # Processing the call keyword arguments (line 413)
    # Getting the type of 'x' (line 413)
    x_186585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 28), 'x', False)
    keyword_186586 = x_186585
    # Getting the type of 'funv' (line 413)
    funv_186587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 35), 'funv', False)
    keyword_186588 = funv_186587
    # Getting the type of 'jacv' (line 413)
    jacv_186589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 45), 'jacv', False)
    keyword_186590 = jacv_186589
    # Getting the type of 'nf' (line 413)
    nf_186591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 56), 'nf', False)
    keyword_186592 = nf_186591
    # Getting the type of 'nit' (line 413)
    nit_186593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 64), 'nit', False)
    keyword_186594 = nit_186593
    # Getting the type of 'rc' (line 413)
    rc_186595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 76), 'rc', False)
    keyword_186596 = rc_186595
    
    # Obtaining the type of the subscript
    # Getting the type of 'rc' (line 414)
    rc_186597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 44), 'rc', False)
    # Getting the type of 'RCSTRINGS' (line 414)
    RCSTRINGS_186598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 34), 'RCSTRINGS', False)
    # Obtaining the member '__getitem__' of a type (line 414)
    getitem___186599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 34), RCSTRINGS_186598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 414)
    subscript_call_result_186600 = invoke(stypy.reporting.localization.Localization(__file__, 414, 34), getitem___186599, rc_186597)
    
    keyword_186601 = subscript_call_result_186600
    
    int_186602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 58), 'int')
    # Getting the type of 'rc' (line 414)
    rc_186603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 63), 'rc', False)
    # Applying the binary operator '<' (line 414)
    result_lt_186604 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 58), '<', int_186602, rc_186603)
    int_186605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 68), 'int')
    # Applying the binary operator '<' (line 414)
    result_lt_186606 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 58), '<', rc_186603, int_186605)
    # Applying the binary operator '&' (line 414)
    result_and__186607 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 58), '&', result_lt_186604, result_lt_186606)
    
    keyword_186608 = result_and__186607
    kwargs_186609 = {'status': keyword_186596, 'success': keyword_186608, 'nfev': keyword_186592, 'fun': keyword_186588, 'x': keyword_186586, 'message': keyword_186601, 'jac': keyword_186590, 'nit': keyword_186594}
    # Getting the type of 'OptimizeResult' (line 413)
    OptimizeResult_186584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 11), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 413)
    OptimizeResult_call_result_186610 = invoke(stypy.reporting.localization.Localization(__file__, 413, 11), OptimizeResult_186584, *[], **kwargs_186609)
    
    # Assigning a type to the variable 'stypy_return_type' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'stypy_return_type', OptimizeResult_call_result_186610)
    
    # ################# End of '_minimize_tnc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_minimize_tnc' in the type store
    # Getting the type of 'stypy_return_type' (line 280)
    stypy_return_type_186611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_186611)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_minimize_tnc'
    return stypy_return_type_186611

# Assigning a type to the variable '_minimize_tnc' (line 280)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 0), '_minimize_tnc', _minimize_tnc)

if (__name__ == '__main__'):

    @norecursion
    def example(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'example'
        module_type_store = module_type_store.open_function_context('example', 419, 4, False)
        
        # Passed parameters checking function
        example.stypy_localization = localization
        example.stypy_type_of_self = None
        example.stypy_type_store = module_type_store
        example.stypy_function_name = 'example'
        example.stypy_param_names_list = []
        example.stypy_varargs_param_name = None
        example.stypy_kwargs_param_name = None
        example.stypy_call_defaults = defaults
        example.stypy_call_varargs = varargs
        example.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'example', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'example', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'example(...)' code ##################

        
        # Call to print(...): (line 420)
        # Processing the call arguments (line 420)
        str_186613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 14), 'str', 'Example')
        # Processing the call keyword arguments (line 420)
        kwargs_186614 = {}
        # Getting the type of 'print' (line 420)
        print_186612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'print', False)
        # Calling print(args, kwargs) (line 420)
        print_call_result_186615 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), print_186612, *[str_186613], **kwargs_186614)
        

        @norecursion
        def function(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'function'
            module_type_store = module_type_store.open_function_context('function', 423, 8, False)
            
            # Passed parameters checking function
            function.stypy_localization = localization
            function.stypy_type_of_self = None
            function.stypy_type_store = module_type_store
            function.stypy_function_name = 'function'
            function.stypy_param_names_list = ['x']
            function.stypy_varargs_param_name = None
            function.stypy_kwargs_param_name = None
            function.stypy_call_defaults = defaults
            function.stypy_call_varargs = varargs
            function.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'function', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'function', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'function(...)' code ##################

            
            # Assigning a BinOp to a Name (line 424):
            
            # Assigning a BinOp to a Name (line 424):
            
            # Call to pow(...): (line 424)
            # Processing the call arguments (line 424)
            
            # Obtaining the type of the subscript
            int_186617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 22), 'int')
            # Getting the type of 'x' (line 424)
            x_186618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 20), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 424)
            getitem___186619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 20), x_186618, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 424)
            subscript_call_result_186620 = invoke(stypy.reporting.localization.Localization(__file__, 424, 20), getitem___186619, int_186617)
            
            float_186621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 25), 'float')
            # Processing the call keyword arguments (line 424)
            kwargs_186622 = {}
            # Getting the type of 'pow' (line 424)
            pow_186616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 16), 'pow', False)
            # Calling pow(args, kwargs) (line 424)
            pow_call_result_186623 = invoke(stypy.reporting.localization.Localization(__file__, 424, 16), pow_186616, *[subscript_call_result_186620, float_186621], **kwargs_186622)
            
            
            # Call to pow(...): (line 424)
            # Processing the call arguments (line 424)
            
            # Call to abs(...): (line 424)
            # Processing the call arguments (line 424)
            
            # Obtaining the type of the subscript
            int_186626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 40), 'int')
            # Getting the type of 'x' (line 424)
            x_186627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 38), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 424)
            getitem___186628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 38), x_186627, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 424)
            subscript_call_result_186629 = invoke(stypy.reporting.localization.Localization(__file__, 424, 38), getitem___186628, int_186626)
            
            # Processing the call keyword arguments (line 424)
            kwargs_186630 = {}
            # Getting the type of 'abs' (line 424)
            abs_186625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 34), 'abs', False)
            # Calling abs(args, kwargs) (line 424)
            abs_call_result_186631 = invoke(stypy.reporting.localization.Localization(__file__, 424, 34), abs_186625, *[subscript_call_result_186629], **kwargs_186630)
            
            float_186632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 44), 'float')
            # Processing the call keyword arguments (line 424)
            kwargs_186633 = {}
            # Getting the type of 'pow' (line 424)
            pow_186624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 30), 'pow', False)
            # Calling pow(args, kwargs) (line 424)
            pow_call_result_186634 = invoke(stypy.reporting.localization.Localization(__file__, 424, 30), pow_186624, *[abs_call_result_186631, float_186632], **kwargs_186633)
            
            # Applying the binary operator '+' (line 424)
            result_add_186635 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 16), '+', pow_call_result_186623, pow_call_result_186634)
            
            # Assigning a type to the variable 'f' (line 424)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'f', result_add_186635)
            
            # Assigning a List to a Name (line 425):
            
            # Assigning a List to a Name (line 425):
            
            # Obtaining an instance of the builtin type 'list' (line 425)
            list_186636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 16), 'list')
            # Adding type elements to the builtin type 'list' instance (line 425)
            # Adding element type (line 425)
            int_186637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 17), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 16), list_186636, int_186637)
            # Adding element type (line 425)
            int_186638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 19), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 16), list_186636, int_186638)
            
            # Assigning a type to the variable 'g' (line 425)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'g', list_186636)
            
            # Assigning a BinOp to a Subscript (line 426):
            
            # Assigning a BinOp to a Subscript (line 426):
            float_186639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 19), 'float')
            
            # Obtaining the type of the subscript
            int_186640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 25), 'int')
            # Getting the type of 'x' (line 426)
            x_186641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 23), 'x')
            # Obtaining the member '__getitem__' of a type (line 426)
            getitem___186642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 23), x_186641, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 426)
            subscript_call_result_186643 = invoke(stypy.reporting.localization.Localization(__file__, 426, 23), getitem___186642, int_186640)
            
            # Applying the binary operator '*' (line 426)
            result_mul_186644 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 19), '*', float_186639, subscript_call_result_186643)
            
            # Getting the type of 'g' (line 426)
            g_186645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'g')
            int_186646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 14), 'int')
            # Storing an element on a container (line 426)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 12), g_186645, (int_186646, result_mul_186644))
            
            # Assigning a BinOp to a Subscript (line 427):
            
            # Assigning a BinOp to a Subscript (line 427):
            float_186647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 19), 'float')
            
            # Call to pow(...): (line 427)
            # Processing the call arguments (line 427)
            
            # Call to abs(...): (line 427)
            # Processing the call arguments (line 427)
            
            # Obtaining the type of the subscript
            int_186650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 33), 'int')
            # Getting the type of 'x' (line 427)
            x_186651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 31), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 427)
            getitem___186652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 31), x_186651, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 427)
            subscript_call_result_186653 = invoke(stypy.reporting.localization.Localization(__file__, 427, 31), getitem___186652, int_186650)
            
            # Processing the call keyword arguments (line 427)
            kwargs_186654 = {}
            # Getting the type of 'abs' (line 427)
            abs_186649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 27), 'abs', False)
            # Calling abs(args, kwargs) (line 427)
            abs_call_result_186655 = invoke(stypy.reporting.localization.Localization(__file__, 427, 27), abs_186649, *[subscript_call_result_186653], **kwargs_186654)
            
            float_186656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 37), 'float')
            # Processing the call keyword arguments (line 427)
            kwargs_186657 = {}
            # Getting the type of 'pow' (line 427)
            pow_186648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 23), 'pow', False)
            # Calling pow(args, kwargs) (line 427)
            pow_call_result_186658 = invoke(stypy.reporting.localization.Localization(__file__, 427, 23), pow_186648, *[abs_call_result_186655, float_186656], **kwargs_186657)
            
            # Applying the binary operator '*' (line 427)
            result_mul_186659 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 19), '*', float_186647, pow_call_result_186658)
            
            # Getting the type of 'g' (line 427)
            g_186660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'g')
            int_186661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 14), 'int')
            # Storing an element on a container (line 427)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 12), g_186660, (int_186661, result_mul_186659))
            
            
            
            # Obtaining the type of the subscript
            int_186662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 17), 'int')
            # Getting the type of 'x' (line 428)
            x_186663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 15), 'x')
            # Obtaining the member '__getitem__' of a type (line 428)
            getitem___186664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 15), x_186663, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 428)
            subscript_call_result_186665 = invoke(stypy.reporting.localization.Localization(__file__, 428, 15), getitem___186664, int_186662)
            
            int_186666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 22), 'int')
            # Applying the binary operator '<' (line 428)
            result_lt_186667 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 15), '<', subscript_call_result_186665, int_186666)
            
            # Testing the type of an if condition (line 428)
            if_condition_186668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 12), result_lt_186667)
            # Assigning a type to the variable 'if_condition_186668' (line 428)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'if_condition_186668', if_condition_186668)
            # SSA begins for if statement (line 428)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a UnaryOp to a Subscript (line 429):
            
            # Assigning a UnaryOp to a Subscript (line 429):
            
            
            # Obtaining the type of the subscript
            int_186669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 26), 'int')
            # Getting the type of 'g' (line 429)
            g_186670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 24), 'g')
            # Obtaining the member '__getitem__' of a type (line 429)
            getitem___186671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 24), g_186670, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 429)
            subscript_call_result_186672 = invoke(stypy.reporting.localization.Localization(__file__, 429, 24), getitem___186671, int_186669)
            
            # Applying the 'usub' unary operator (line 429)
            result___neg___186673 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 23), 'usub', subscript_call_result_186672)
            
            # Getting the type of 'g' (line 429)
            g_186674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'g')
            int_186675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 18), 'int')
            # Storing an element on a container (line 429)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 16), g_186674, (int_186675, result___neg___186673))
            # SSA join for if statement (line 428)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 430)
            tuple_186676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 430)
            # Adding element type (line 430)
            # Getting the type of 'f' (line 430)
            f_186677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 'f')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 19), tuple_186676, f_186677)
            # Adding element type (line 430)
            # Getting the type of 'g' (line 430)
            g_186678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 22), 'g')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 19), tuple_186676, g_186678)
            
            # Assigning a type to the variable 'stypy_return_type' (line 430)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'stypy_return_type', tuple_186676)
            
            # ################# End of 'function(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'function' in the type store
            # Getting the type of 'stypy_return_type' (line 423)
            stypy_return_type_186679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_186679)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'function'
            return stypy_return_type_186679

        # Assigning a type to the variable 'function' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'function', function)
        
        # Assigning a Call to a Tuple (line 433):
        
        # Assigning a Subscript to a Name (line 433):
        
        # Obtaining the type of the subscript
        int_186680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 8), 'int')
        
        # Call to fmin_tnc(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'function' (line 433)
        function_186682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 29), 'function', False)
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_186683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        int_186684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 39), list_186683, int_186684)
        # Adding element type (line 433)
        int_186685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 39), list_186683, int_186685)
        
        # Processing the call keyword arguments (line 433)
        
        # Obtaining an instance of the builtin type 'tuple' (line 433)
        tuple_186686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 433)
        # Adding element type (line 433)
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_186687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        int_186688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), list_186687, int_186688)
        # Adding element type (line 433)
        int_186689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), list_186687, int_186689)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), tuple_186686, list_186687)
        # Adding element type (line 433)
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_186690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        int_186691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 66), list_186690, int_186691)
        # Adding element type (line 433)
        int_186692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 71), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 66), list_186690, int_186692)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), tuple_186686, list_186690)
        
        keyword_186693 = tuple_186686
        kwargs_186694 = {'bounds': keyword_186693}
        # Getting the type of 'fmin_tnc' (line 433)
        fmin_tnc_186681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'fmin_tnc', False)
        # Calling fmin_tnc(args, kwargs) (line 433)
        fmin_tnc_call_result_186695 = invoke(stypy.reporting.localization.Localization(__file__, 433, 20), fmin_tnc_186681, *[function_186682, list_186683], **kwargs_186694)
        
        # Obtaining the member '__getitem__' of a type (line 433)
        getitem___186696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), fmin_tnc_call_result_186695, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 433)
        subscript_call_result_186697 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), getitem___186696, int_186680)
        
        # Assigning a type to the variable 'tuple_var_assignment_186070' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'tuple_var_assignment_186070', subscript_call_result_186697)
        
        # Assigning a Subscript to a Name (line 433):
        
        # Obtaining the type of the subscript
        int_186698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 8), 'int')
        
        # Call to fmin_tnc(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'function' (line 433)
        function_186700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 29), 'function', False)
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_186701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        int_186702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 39), list_186701, int_186702)
        # Adding element type (line 433)
        int_186703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 39), list_186701, int_186703)
        
        # Processing the call keyword arguments (line 433)
        
        # Obtaining an instance of the builtin type 'tuple' (line 433)
        tuple_186704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 433)
        # Adding element type (line 433)
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_186705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        int_186706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), list_186705, int_186706)
        # Adding element type (line 433)
        int_186707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), list_186705, int_186707)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), tuple_186704, list_186705)
        # Adding element type (line 433)
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_186708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        int_186709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 66), list_186708, int_186709)
        # Adding element type (line 433)
        int_186710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 71), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 66), list_186708, int_186710)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), tuple_186704, list_186708)
        
        keyword_186711 = tuple_186704
        kwargs_186712 = {'bounds': keyword_186711}
        # Getting the type of 'fmin_tnc' (line 433)
        fmin_tnc_186699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'fmin_tnc', False)
        # Calling fmin_tnc(args, kwargs) (line 433)
        fmin_tnc_call_result_186713 = invoke(stypy.reporting.localization.Localization(__file__, 433, 20), fmin_tnc_186699, *[function_186700, list_186701], **kwargs_186712)
        
        # Obtaining the member '__getitem__' of a type (line 433)
        getitem___186714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), fmin_tnc_call_result_186713, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 433)
        subscript_call_result_186715 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), getitem___186714, int_186698)
        
        # Assigning a type to the variable 'tuple_var_assignment_186071' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'tuple_var_assignment_186071', subscript_call_result_186715)
        
        # Assigning a Subscript to a Name (line 433):
        
        # Obtaining the type of the subscript
        int_186716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 8), 'int')
        
        # Call to fmin_tnc(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'function' (line 433)
        function_186718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 29), 'function', False)
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_186719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        int_186720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 39), list_186719, int_186720)
        # Adding element type (line 433)
        int_186721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 39), list_186719, int_186721)
        
        # Processing the call keyword arguments (line 433)
        
        # Obtaining an instance of the builtin type 'tuple' (line 433)
        tuple_186722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 433)
        # Adding element type (line 433)
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_186723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        int_186724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), list_186723, int_186724)
        # Adding element type (line 433)
        int_186725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), list_186723, int_186725)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), tuple_186722, list_186723)
        # Adding element type (line 433)
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_186726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        int_186727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 66), list_186726, int_186727)
        # Adding element type (line 433)
        int_186728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 71), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 66), list_186726, int_186728)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 56), tuple_186722, list_186726)
        
        keyword_186729 = tuple_186722
        kwargs_186730 = {'bounds': keyword_186729}
        # Getting the type of 'fmin_tnc' (line 433)
        fmin_tnc_186717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'fmin_tnc', False)
        # Calling fmin_tnc(args, kwargs) (line 433)
        fmin_tnc_call_result_186731 = invoke(stypy.reporting.localization.Localization(__file__, 433, 20), fmin_tnc_186717, *[function_186718, list_186719], **kwargs_186730)
        
        # Obtaining the member '__getitem__' of a type (line 433)
        getitem___186732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), fmin_tnc_call_result_186731, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 433)
        subscript_call_result_186733 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), getitem___186732, int_186716)
        
        # Assigning a type to the variable 'tuple_var_assignment_186072' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'tuple_var_assignment_186072', subscript_call_result_186733)
        
        # Assigning a Name to a Name (line 433):
        # Getting the type of 'tuple_var_assignment_186070' (line 433)
        tuple_var_assignment_186070_186734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'tuple_var_assignment_186070')
        # Assigning a type to the variable 'x' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'x', tuple_var_assignment_186070_186734)
        
        # Assigning a Name to a Name (line 433):
        # Getting the type of 'tuple_var_assignment_186071' (line 433)
        tuple_var_assignment_186071_186735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'tuple_var_assignment_186071')
        # Assigning a type to the variable 'nf' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'nf', tuple_var_assignment_186071_186735)
        
        # Assigning a Name to a Name (line 433):
        # Getting the type of 'tuple_var_assignment_186072' (line 433)
        tuple_var_assignment_186072_186736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'tuple_var_assignment_186072')
        # Assigning a type to the variable 'rc' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'rc', tuple_var_assignment_186072_186736)
        
        # Call to print(...): (line 435)
        # Processing the call arguments (line 435)
        str_186738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 14), 'str', 'After')
        # Getting the type of 'nf' (line 435)
        nf_186739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 23), 'nf', False)
        str_186740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 27), 'str', 'function evaluations, TNC returned:')
        
        # Obtaining the type of the subscript
        # Getting the type of 'rc' (line 435)
        rc_186741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 76), 'rc', False)
        # Getting the type of 'RCSTRINGS' (line 435)
        RCSTRINGS_186742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 66), 'RCSTRINGS', False)
        # Obtaining the member '__getitem__' of a type (line 435)
        getitem___186743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 66), RCSTRINGS_186742, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 435)
        subscript_call_result_186744 = invoke(stypy.reporting.localization.Localization(__file__, 435, 66), getitem___186743, rc_186741)
        
        # Processing the call keyword arguments (line 435)
        kwargs_186745 = {}
        # Getting the type of 'print' (line 435)
        print_186737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'print', False)
        # Calling print(args, kwargs) (line 435)
        print_call_result_186746 = invoke(stypy.reporting.localization.Localization(__file__, 435, 8), print_186737, *[str_186738, nf_186739, str_186740, subscript_call_result_186744], **kwargs_186745)
        
        
        # Call to print(...): (line 436)
        # Processing the call arguments (line 436)
        str_186748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 14), 'str', 'x =')
        # Getting the type of 'x' (line 436)
        x_186749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 21), 'x', False)
        # Processing the call keyword arguments (line 436)
        kwargs_186750 = {}
        # Getting the type of 'print' (line 436)
        print_186747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'print', False)
        # Calling print(args, kwargs) (line 436)
        print_call_result_186751 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), print_186747, *[str_186748, x_186749], **kwargs_186750)
        
        
        # Call to print(...): (line 437)
        # Processing the call arguments (line 437)
        str_186753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 14), 'str', 'exact value = [0, 1]')
        # Processing the call keyword arguments (line 437)
        kwargs_186754 = {}
        # Getting the type of 'print' (line 437)
        print_186752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'print', False)
        # Calling print(args, kwargs) (line 437)
        print_call_result_186755 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), print_186752, *[str_186753], **kwargs_186754)
        
        
        # Call to print(...): (line 438)
        # Processing the call keyword arguments (line 438)
        kwargs_186757 = {}
        # Getting the type of 'print' (line 438)
        print_186756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'print', False)
        # Calling print(args, kwargs) (line 438)
        print_call_result_186758 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), print_186756, *[], **kwargs_186757)
        
        
        # ################# End of 'example(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'example' in the type store
        # Getting the type of 'stypy_return_type' (line 419)
        stypy_return_type_186759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186759)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'example'
        return stypy_return_type_186759

    # Assigning a type to the variable 'example' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'example', example)
    
    # Call to example(...): (line 440)
    # Processing the call keyword arguments (line 440)
    kwargs_186761 = {}
    # Getting the type of 'example' (line 440)
    example_186760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'example', False)
    # Calling example(args, kwargs) (line 440)
    example_call_result_186762 = invoke(stypy.reporting.localization.Localization(__file__, 440, 4), example_186760, *[], **kwargs_186761)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
