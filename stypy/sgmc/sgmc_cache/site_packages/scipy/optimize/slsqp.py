
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This module implements the Sequential Least SQuares Programming optimization
3: algorithm (SLSQP), originally developed by Dieter Kraft.
4: See http://www.netlib.org/toms/733
5: 
6: Functions
7: ---------
8: .. autosummary::
9:    :toctree: generated/
10: 
11:     approx_jacobian
12:     fmin_slsqp
13: 
14: '''
15: 
16: from __future__ import division, print_function, absolute_import
17: 
18: __all__ = ['approx_jacobian', 'fmin_slsqp']
19: 
20: import numpy as np
21: from scipy.optimize._slsqp import slsqp
22: from numpy import (zeros, array, linalg, append, asfarray, concatenate, finfo,
23:                    sqrt, vstack, exp, inf, isfinite, atleast_1d)
24: from .optimize import wrap_function, OptimizeResult, _check_unknown_options
25: 
26: __docformat__ = "restructuredtext en"
27: 
28: _epsilon = sqrt(finfo(float).eps)
29: 
30: 
31: def approx_jacobian(x, func, epsilon, *args):
32:     '''
33:     Approximate the Jacobian matrix of a callable function.
34: 
35:     Parameters
36:     ----------
37:     x : array_like
38:         The state vector at which to compute the Jacobian matrix.
39:     func : callable f(x,*args)
40:         The vector-valued function.
41:     epsilon : float
42:         The perturbation used to determine the partial derivatives.
43:     args : sequence
44:         Additional arguments passed to func.
45: 
46:     Returns
47:     -------
48:     An array of dimensions ``(lenf, lenx)`` where ``lenf`` is the length
49:     of the outputs of `func`, and ``lenx`` is the number of elements in
50:     `x`.
51: 
52:     Notes
53:     -----
54:     The approximation is done using forward differences.
55: 
56:     '''
57:     x0 = asfarray(x)
58:     f0 = atleast_1d(func(*((x0,)+args)))
59:     jac = zeros([len(x0), len(f0)])
60:     dx = zeros(len(x0))
61:     for i in range(len(x0)):
62:         dx[i] = epsilon
63:         jac[i] = (func(*((x0+dx,)+args)) - f0)/epsilon
64:         dx[i] = 0.0
65: 
66:     return jac.transpose()
67: 
68: 
69: def fmin_slsqp(func, x0, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None,
70:                bounds=(), fprime=None, fprime_eqcons=None,
71:                fprime_ieqcons=None, args=(), iter=100, acc=1.0E-6,
72:                iprint=1, disp=None, full_output=0, epsilon=_epsilon,
73:                callback=None):
74:     '''
75:     Minimize a function using Sequential Least SQuares Programming
76: 
77:     Python interface function for the SLSQP Optimization subroutine
78:     originally implemented by Dieter Kraft.
79: 
80:     Parameters
81:     ----------
82:     func : callable f(x,*args)
83:         Objective function.  Must return a scalar.
84:     x0 : 1-D ndarray of float
85:         Initial guess for the independent variable(s).
86:     eqcons : list, optional
87:         A list of functions of length n such that
88:         eqcons[j](x,*args) == 0.0 in a successfully optimized
89:         problem.
90:     f_eqcons : callable f(x,*args), optional
91:         Returns a 1-D array in which each element must equal 0.0 in a
92:         successfully optimized problem.  If f_eqcons is specified,
93:         eqcons is ignored.
94:     ieqcons : list, optional
95:         A list of functions of length n such that
96:         ieqcons[j](x,*args) >= 0.0 in a successfully optimized
97:         problem.
98:     f_ieqcons : callable f(x,*args), optional
99:         Returns a 1-D ndarray in which each element must be greater or
100:         equal to 0.0 in a successfully optimized problem.  If
101:         f_ieqcons is specified, ieqcons is ignored.
102:     bounds : list, optional
103:         A list of tuples specifying the lower and upper bound
104:         for each independent variable [(xl0, xu0),(xl1, xu1),...]
105:         Infinite values will be interpreted as large floating values.
106:     fprime : callable `f(x,*args)`, optional
107:         A function that evaluates the partial derivatives of func.
108:     fprime_eqcons : callable `f(x,*args)`, optional
109:         A function of the form `f(x, *args)` that returns the m by n
110:         array of equality constraint normals.  If not provided,
111:         the normals will be approximated. The array returned by
112:         fprime_eqcons should be sized as ( len(eqcons), len(x0) ).
113:     fprime_ieqcons : callable `f(x,*args)`, optional
114:         A function of the form `f(x, *args)` that returns the m by n
115:         array of inequality constraint normals.  If not provided,
116:         the normals will be approximated. The array returned by
117:         fprime_ieqcons should be sized as ( len(ieqcons), len(x0) ).
118:     args : sequence, optional
119:         Additional arguments passed to func and fprime.
120:     iter : int, optional
121:         The maximum number of iterations.
122:     acc : float, optional
123:         Requested accuracy.
124:     iprint : int, optional
125:         The verbosity of fmin_slsqp :
126: 
127:         * iprint <= 0 : Silent operation
128:         * iprint == 1 : Print summary upon completion (default)
129:         * iprint >= 2 : Print status of each iterate and summary
130:     disp : int, optional
131:         Over-rides the iprint interface (preferred).
132:     full_output : bool, optional
133:         If False, return only the minimizer of func (default).
134:         Otherwise, output final objective function and summary
135:         information.
136:     epsilon : float, optional
137:         The step size for finite-difference derivative estimates.
138:     callback : callable, optional
139:         Called after each iteration, as ``callback(x)``, where ``x`` is the
140:         current parameter vector.
141: 
142:     Returns
143:     -------
144:     out : ndarray of float
145:         The final minimizer of func.
146:     fx : ndarray of float, if full_output is true
147:         The final value of the objective function.
148:     its : int, if full_output is true
149:         The number of iterations.
150:     imode : int, if full_output is true
151:         The exit mode from the optimizer (see below).
152:     smode : string, if full_output is true
153:         Message describing the exit mode from the optimizer.
154: 
155:     See also
156:     --------
157:     minimize: Interface to minimization algorithms for multivariate
158:         functions. See the 'SLSQP' `method` in particular.
159: 
160:     Notes
161:     -----
162:     Exit modes are defined as follows ::
163: 
164:         -1 : Gradient evaluation required (g & a)
165:          0 : Optimization terminated successfully.
166:          1 : Function evaluation required (f & c)
167:          2 : More equality constraints than independent variables
168:          3 : More than 3*n iterations in LSQ subproblem
169:          4 : Inequality constraints incompatible
170:          5 : Singular matrix E in LSQ subproblem
171:          6 : Singular matrix C in LSQ subproblem
172:          7 : Rank-deficient equality constraint subproblem HFTI
173:          8 : Positive directional derivative for linesearch
174:          9 : Iteration limit exceeded
175: 
176:     Examples
177:     --------
178:     Examples are given :ref:`in the tutorial <tutorial-sqlsp>`.
179: 
180:     '''
181:     if disp is not None:
182:         iprint = disp
183:     opts = {'maxiter': iter,
184:             'ftol': acc,
185:             'iprint': iprint,
186:             'disp': iprint != 0,
187:             'eps': epsilon,
188:             'callback': callback}
189: 
190:     # Build the constraints as a tuple of dictionaries
191:     cons = ()
192:     # 1. constraints of the 1st kind (eqcons, ieqcons); no Jacobian; take
193:     #    the same extra arguments as the objective function.
194:     cons += tuple({'type': 'eq', 'fun': c, 'args': args} for c in eqcons)
195:     cons += tuple({'type': 'ineq', 'fun': c, 'args': args} for c in ieqcons)
196:     # 2. constraints of the 2nd kind (f_eqcons, f_ieqcons) and their Jacobian
197:     #    (fprime_eqcons, fprime_ieqcons); also take the same extra arguments
198:     #    as the objective function.
199:     if f_eqcons:
200:         cons += ({'type': 'eq', 'fun': f_eqcons, 'jac': fprime_eqcons,
201:                   'args': args}, )
202:     if f_ieqcons:
203:         cons += ({'type': 'ineq', 'fun': f_ieqcons, 'jac': fprime_ieqcons,
204:                   'args': args}, )
205: 
206:     res = _minimize_slsqp(func, x0, args, jac=fprime, bounds=bounds,
207:                           constraints=cons, **opts)
208:     if full_output:
209:         return res['x'], res['fun'], res['nit'], res['status'], res['message']
210:     else:
211:         return res['x']
212: 
213: 
214: def _minimize_slsqp(func, x0, args=(), jac=None, bounds=None,
215:                     constraints=(),
216:                     maxiter=100, ftol=1.0E-6, iprint=1, disp=False,
217:                     eps=_epsilon, callback=None,
218:                     **unknown_options):
219:     '''
220:     Minimize a scalar function of one or more variables using Sequential
221:     Least SQuares Programming (SLSQP).
222: 
223:     Options
224:     -------
225:     ftol : float
226:         Precision goal for the value of f in the stopping criterion.
227:     eps : float
228:         Step size used for numerical approximation of the Jacobian.
229:     disp : bool
230:         Set to True to print convergence messages. If False,
231:         `verbosity` is ignored and set to 0.
232:     maxiter : int
233:         Maximum number of iterations.
234: 
235:     '''
236:     _check_unknown_options(unknown_options)
237:     fprime = jac
238:     iter = maxiter
239:     acc = ftol
240:     epsilon = eps
241: 
242:     if not disp:
243:         iprint = 0
244: 
245:     # Constraints are triaged per type into a dictionary of tuples
246:     if isinstance(constraints, dict):
247:         constraints = (constraints, )
248: 
249:     cons = {'eq': (), 'ineq': ()}
250:     for ic, con in enumerate(constraints):
251:         # check type
252:         try:
253:             ctype = con['type'].lower()
254:         except KeyError:
255:             raise KeyError('Constraint %d has no type defined.' % ic)
256:         except TypeError:
257:             raise TypeError('Constraints must be defined using a '
258:                             'dictionary.')
259:         except AttributeError:
260:             raise TypeError("Constraint's type must be a string.")
261:         else:
262:             if ctype not in ['eq', 'ineq']:
263:                 raise ValueError("Unknown constraint type '%s'." % con['type'])
264: 
265:         # check function
266:         if 'fun' not in con:
267:             raise ValueError('Constraint %d has no function defined.' % ic)
268: 
269:         # check Jacobian
270:         cjac = con.get('jac')
271:         if cjac is None:
272:             # approximate Jacobian function.  The factory function is needed
273:             # to keep a reference to `fun`, see gh-4240.
274:             def cjac_factory(fun):
275:                 def cjac(x, *args):
276:                     return approx_jacobian(x, fun, epsilon, *args)
277:                 return cjac
278:             cjac = cjac_factory(con['fun'])
279: 
280:         # update constraints' dictionary
281:         cons[ctype] += ({'fun': con['fun'],
282:                          'jac': cjac,
283:                          'args': con.get('args', ())}, )
284: 
285:     exit_modes = {-1: "Gradient evaluation required (g & a)",
286:                    0: "Optimization terminated successfully.",
287:                    1: "Function evaluation required (f & c)",
288:                    2: "More equality constraints than independent variables",
289:                    3: "More than 3*n iterations in LSQ subproblem",
290:                    4: "Inequality constraints incompatible",
291:                    5: "Singular matrix E in LSQ subproblem",
292:                    6: "Singular matrix C in LSQ subproblem",
293:                    7: "Rank-deficient equality constraint subproblem HFTI",
294:                    8: "Positive directional derivative for linesearch",
295:                    9: "Iteration limit exceeded"}
296: 
297:     # Wrap func
298:     feval, func = wrap_function(func, args)
299: 
300:     # Wrap fprime, if provided, or approx_jacobian if not
301:     if fprime:
302:         geval, fprime = wrap_function(fprime, args)
303:     else:
304:         geval, fprime = wrap_function(approx_jacobian, (func, epsilon))
305: 
306:     # Transform x0 into an array.
307:     x = asfarray(x0).flatten()
308: 
309:     # Set the parameters that SLSQP will need
310:     # meq, mieq: number of equality and inequality constraints
311:     meq = sum(map(len, [atleast_1d(c['fun'](x, *c['args']))
312:               for c in cons['eq']]))
313:     mieq = sum(map(len, [atleast_1d(c['fun'](x, *c['args']))
314:                for c in cons['ineq']]))
315:     # m = The total number of constraints
316:     m = meq + mieq
317:     # la = The number of constraints, or 1 if there are no constraints
318:     la = array([1, m]).max()
319:     # n = The number of independent variables
320:     n = len(x)
321: 
322:     # Define the workspaces for SLSQP
323:     n1 = n + 1
324:     mineq = m - meq + n1 + n1
325:     len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
326:             + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
327:     len_jw = mineq
328:     w = zeros(len_w)
329:     jw = zeros(len_jw)
330: 
331:     # Decompose bounds into xl and xu
332:     if bounds is None or len(bounds) == 0:
333:         xl = np.empty(n, dtype=float)
334:         xu = np.empty(n, dtype=float)
335:         xl.fill(np.nan)
336:         xu.fill(np.nan)
337:     else:
338:         bnds = array(bounds, float)
339:         if bnds.shape[0] != n:
340:             raise IndexError('SLSQP Error: the length of bounds is not '
341:                              'compatible with that of x0.')
342: 
343:         with np.errstate(invalid='ignore'):
344:             bnderr = bnds[:, 0] > bnds[:, 1]
345: 
346:         if bnderr.any():
347:             raise ValueError('SLSQP Error: lb > ub in bounds %s.' %
348:                              ', '.join(str(b) for b in bnderr))
349:         xl, xu = bnds[:, 0], bnds[:, 1]
350: 
351:         # Mark infinite bounds with nans; the Fortran code understands this
352:         infbnd = ~isfinite(bnds)
353:         xl[infbnd[:, 0]] = np.nan
354:         xu[infbnd[:, 1]] = np.nan
355: 
356:     # Clip initial guess to bounds (SLSQP may fail with bounds-infeasible
357:     # initial point)
358:     have_bound = np.isfinite(xl)
359:     x[have_bound] = np.clip(x[have_bound], xl[have_bound], np.inf)
360:     have_bound = np.isfinite(xu)
361:     x[have_bound] = np.clip(x[have_bound], -np.inf, xu[have_bound])
362: 
363:     # Initialize the iteration counter and the mode value
364:     mode = array(0, int)
365:     acc = array(acc, float)
366:     majiter = array(iter, int)
367:     majiter_prev = 0
368: 
369:     # Print the header if iprint >= 2
370:     if iprint >= 2:
371:         print("%5s %5s %16s %16s" % ("NIT", "FC", "OBJFUN", "GNORM"))
372: 
373:     while 1:
374: 
375:         if mode == 0 or mode == 1:  # objective and constraint evaluation required
376: 
377:             # Compute objective function
378:             fx = func(x)
379:             try:
380:                 fx = float(np.asarray(fx))
381:             except (TypeError, ValueError):
382:                 raise ValueError("Objective function must return a scalar")
383:             # Compute the constraints
384:             if cons['eq']:
385:                 c_eq = concatenate([atleast_1d(con['fun'](x, *con['args']))
386:                                     for con in cons['eq']])
387:             else:
388:                 c_eq = zeros(0)
389:             if cons['ineq']:
390:                 c_ieq = concatenate([atleast_1d(con['fun'](x, *con['args']))
391:                                      for con in cons['ineq']])
392:             else:
393:                 c_ieq = zeros(0)
394: 
395:             # Now combine c_eq and c_ieq into a single matrix
396:             c = concatenate((c_eq, c_ieq))
397: 
398:         if mode == 0 or mode == -1:  # gradient evaluation required
399: 
400:             # Compute the derivatives of the objective function
401:             # For some reason SLSQP wants g dimensioned to n+1
402:             g = append(fprime(x), 0.0)
403: 
404:             # Compute the normals of the constraints
405:             if cons['eq']:
406:                 a_eq = vstack([con['jac'](x, *con['args'])
407:                                for con in cons['eq']])
408:             else:  # no equality constraint
409:                 a_eq = zeros((meq, n))
410: 
411:             if cons['ineq']:
412:                 a_ieq = vstack([con['jac'](x, *con['args'])
413:                                 for con in cons['ineq']])
414:             else:  # no inequality constraint
415:                 a_ieq = zeros((mieq, n))
416: 
417:             # Now combine a_eq and a_ieq into a single a matrix
418:             if m == 0:  # no constraints
419:                 a = zeros((la, n))
420:             else:
421:                 a = vstack((a_eq, a_ieq))
422:             a = concatenate((a, zeros([la, 1])), 1)
423: 
424:         # Call SLSQP
425:         slsqp(m, meq, x, xl, xu, fx, c, g, a, acc, majiter, mode, w, jw)
426: 
427:         # call callback if major iteration has incremented
428:         if callback is not None and majiter > majiter_prev:
429:             callback(x)
430: 
431:         # Print the status of the current iterate if iprint > 2 and the
432:         # major iteration has incremented
433:         if iprint >= 2 and majiter > majiter_prev:
434:             print("%5i %5i % 16.6E % 16.6E" % (majiter, feval[0],
435:                                                fx, linalg.norm(g)))
436: 
437:         # If exit mode is not -1 or 1, slsqp has completed
438:         if abs(mode) != 1:
439:             break
440: 
441:         majiter_prev = int(majiter)
442: 
443:     # Optimization loop complete.  Print status if requested
444:     if iprint >= 1:
445:         print(exit_modes[int(mode)] + "    (Exit mode " + str(mode) + ')')
446:         print("            Current function value:", fx)
447:         print("            Iterations:", majiter)
448:         print("            Function evaluations:", feval[0])
449:         print("            Gradient evaluations:", geval[0])
450: 
451:     return OptimizeResult(x=x, fun=fx, jac=g[:-1], nit=int(majiter),
452:                           nfev=feval[0], njev=geval[0], status=int(mode),
453:                           message=exit_modes[int(mode)], success=(mode == 0))
454: 
455: 
456: if __name__ == '__main__':
457: 
458:     # objective function
459:     def fun(x, r=[4, 2, 4, 2, 1]):
460:         ''' Objective function '''
461:         return exp(x[0]) * (r[0] * x[0]**2 + r[1] * x[1]**2 +
462:                             r[2] * x[0] * x[1] + r[3] * x[1] +
463:                             r[4])
464: 
465:     # bounds
466:     bnds = array([[-inf]*2, [inf]*2]).T
467:     bnds[:, 0] = [0.1, 0.2]
468: 
469:     # constraints
470:     def feqcon(x, b=1):
471:         ''' Equality constraint '''
472:         return array([x[0]**2 + x[1] - b])
473: 
474:     def jeqcon(x, b=1):
475:         ''' Jacobian of equality constraint '''
476:         return array([[2*x[0], 1]])
477: 
478:     def fieqcon(x, c=10):
479:         ''' Inequality constraint '''
480:         return array([x[0] * x[1] + c])
481: 
482:     def jieqcon(x, c=10):
483:         ''' Jacobian of Inequality constraint '''
484:         return array([[1, 1]])
485: 
486:     # constraints dictionaries
487:     cons = ({'type': 'eq', 'fun': feqcon, 'jac': jeqcon, 'args': (1, )},
488:             {'type': 'ineq', 'fun': fieqcon, 'jac': jieqcon, 'args': (10,)})
489: 
490:     # Bounds constraint problem
491:     print(' Bounds constraints '.center(72, '-'))
492:     print(' * fmin_slsqp')
493:     x, f = fmin_slsqp(fun, array([-1, 1]), bounds=bnds, disp=1,
494:                       full_output=True)[:2]
495:     print(' * _minimize_slsqp')
496:     res = _minimize_slsqp(fun, array([-1, 1]), bounds=bnds,
497:                           **{'disp': True})
498: 
499:     # Equality and inequality constraints problem
500:     print(' Equality and inequality constraints '.center(72, '-'))
501:     print(' * fmin_slsqp')
502:     x, f = fmin_slsqp(fun, array([-1, 1]),
503:                       f_eqcons=feqcon, fprime_eqcons=jeqcon,
504:                       f_ieqcons=fieqcon, fprime_ieqcons=jieqcon,
505:                       disp=1, full_output=True)[:2]
506:     print(' * _minimize_slsqp')
507:     res = _minimize_slsqp(fun, array([-1, 1]), constraints=cons,
508:                           **{'disp': True})
509: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_184552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, (-1)), 'str', '\nThis module implements the Sequential Least SQuares Programming optimization\nalgorithm (SLSQP), originally developed by Dieter Kraft.\nSee http://www.netlib.org/toms/733\n\nFunctions\n---------\n.. autosummary::\n   :toctree: generated/\n\n    approx_jacobian\n    fmin_slsqp\n\n')

# Assigning a List to a Name (line 18):

# Assigning a List to a Name (line 18):
__all__ = ['approx_jacobian', 'fmin_slsqp']
module_type_store.set_exportable_members(['approx_jacobian', 'fmin_slsqp'])

# Obtaining an instance of the builtin type 'list' (line 18)
list_184553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
str_184554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'str', 'approx_jacobian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), list_184553, str_184554)
# Adding element type (line 18)
str_184555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'str', 'fmin_slsqp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), list_184553, str_184555)

# Assigning a type to the variable '__all__' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '__all__', list_184553)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import numpy' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_184556 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy')

if (type(import_184556) is not StypyTypeError):

    if (import_184556 != 'pyd_module'):
        __import__(import_184556)
        sys_modules_184557 = sys.modules[import_184556]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'np', sys_modules_184557.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy', import_184556)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from scipy.optimize._slsqp import slsqp' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_184558 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize._slsqp')

if (type(import_184558) is not StypyTypeError):

    if (import_184558 != 'pyd_module'):
        __import__(import_184558)
        sys_modules_184559 = sys.modules[import_184558]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize._slsqp', sys_modules_184559.module_type_store, module_type_store, ['slsqp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_184559, sys_modules_184559.module_type_store, module_type_store)
    else:
        from scipy.optimize._slsqp import slsqp

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize._slsqp', None, module_type_store, ['slsqp'], [slsqp])

else:
    # Assigning a type to the variable 'scipy.optimize._slsqp' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.optimize._slsqp', import_184558)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from numpy import zeros, array, linalg, append, asfarray, concatenate, finfo, sqrt, vstack, exp, inf, isfinite, atleast_1d' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_184560 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy')

if (type(import_184560) is not StypyTypeError):

    if (import_184560 != 'pyd_module'):
        __import__(import_184560)
        sys_modules_184561 = sys.modules[import_184560]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy', sys_modules_184561.module_type_store, module_type_store, ['zeros', 'array', 'linalg', 'append', 'asfarray', 'concatenate', 'finfo', 'sqrt', 'vstack', 'exp', 'inf', 'isfinite', 'atleast_1d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_184561, sys_modules_184561.module_type_store, module_type_store)
    else:
        from numpy import zeros, array, linalg, append, asfarray, concatenate, finfo, sqrt, vstack, exp, inf, isfinite, atleast_1d

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy', None, module_type_store, ['zeros', 'array', 'linalg', 'append', 'asfarray', 'concatenate', 'finfo', 'sqrt', 'vstack', 'exp', 'inf', 'isfinite', 'atleast_1d'], [zeros, array, linalg, append, asfarray, concatenate, finfo, sqrt, vstack, exp, inf, isfinite, atleast_1d])

else:
    # Assigning a type to the variable 'numpy' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy', import_184560)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from scipy.optimize.optimize import wrap_function, OptimizeResult, _check_unknown_options' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_184562 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.optimize.optimize')

if (type(import_184562) is not StypyTypeError):

    if (import_184562 != 'pyd_module'):
        __import__(import_184562)
        sys_modules_184563 = sys.modules[import_184562]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.optimize.optimize', sys_modules_184563.module_type_store, module_type_store, ['wrap_function', 'OptimizeResult', '_check_unknown_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_184563, sys_modules_184563.module_type_store, module_type_store)
    else:
        from scipy.optimize.optimize import wrap_function, OptimizeResult, _check_unknown_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.optimize.optimize', None, module_type_store, ['wrap_function', 'OptimizeResult', '_check_unknown_options'], [wrap_function, OptimizeResult, _check_unknown_options])

else:
    # Assigning a type to the variable 'scipy.optimize.optimize' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.optimize.optimize', import_184562)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a Str to a Name (line 26):

# Assigning a Str to a Name (line 26):
str_184564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__docformat__', str_184564)

# Assigning a Call to a Name (line 28):

# Assigning a Call to a Name (line 28):

# Call to sqrt(...): (line 28)
# Processing the call arguments (line 28)

# Call to finfo(...): (line 28)
# Processing the call arguments (line 28)
# Getting the type of 'float' (line 28)
float_184567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'float', False)
# Processing the call keyword arguments (line 28)
kwargs_184568 = {}
# Getting the type of 'finfo' (line 28)
finfo_184566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'finfo', False)
# Calling finfo(args, kwargs) (line 28)
finfo_call_result_184569 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), finfo_184566, *[float_184567], **kwargs_184568)

# Obtaining the member 'eps' of a type (line 28)
eps_184570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), finfo_call_result_184569, 'eps')
# Processing the call keyword arguments (line 28)
kwargs_184571 = {}
# Getting the type of 'sqrt' (line 28)
sqrt_184565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'sqrt', False)
# Calling sqrt(args, kwargs) (line 28)
sqrt_call_result_184572 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), sqrt_184565, *[eps_184570], **kwargs_184571)

# Assigning a type to the variable '_epsilon' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), '_epsilon', sqrt_call_result_184572)

@norecursion
def approx_jacobian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'approx_jacobian'
    module_type_store = module_type_store.open_function_context('approx_jacobian', 31, 0, False)
    
    # Passed parameters checking function
    approx_jacobian.stypy_localization = localization
    approx_jacobian.stypy_type_of_self = None
    approx_jacobian.stypy_type_store = module_type_store
    approx_jacobian.stypy_function_name = 'approx_jacobian'
    approx_jacobian.stypy_param_names_list = ['x', 'func', 'epsilon']
    approx_jacobian.stypy_varargs_param_name = 'args'
    approx_jacobian.stypy_kwargs_param_name = None
    approx_jacobian.stypy_call_defaults = defaults
    approx_jacobian.stypy_call_varargs = varargs
    approx_jacobian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'approx_jacobian', ['x', 'func', 'epsilon'], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'approx_jacobian', localization, ['x', 'func', 'epsilon'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'approx_jacobian(...)' code ##################

    str_184573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n    Approximate the Jacobian matrix of a callable function.\n\n    Parameters\n    ----------\n    x : array_like\n        The state vector at which to compute the Jacobian matrix.\n    func : callable f(x,*args)\n        The vector-valued function.\n    epsilon : float\n        The perturbation used to determine the partial derivatives.\n    args : sequence\n        Additional arguments passed to func.\n\n    Returns\n    -------\n    An array of dimensions ``(lenf, lenx)`` where ``lenf`` is the length\n    of the outputs of `func`, and ``lenx`` is the number of elements in\n    `x`.\n\n    Notes\n    -----\n    The approximation is done using forward differences.\n\n    ')
    
    # Assigning a Call to a Name (line 57):
    
    # Assigning a Call to a Name (line 57):
    
    # Call to asfarray(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'x' (line 57)
    x_184575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'x', False)
    # Processing the call keyword arguments (line 57)
    kwargs_184576 = {}
    # Getting the type of 'asfarray' (line 57)
    asfarray_184574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 9), 'asfarray', False)
    # Calling asfarray(args, kwargs) (line 57)
    asfarray_call_result_184577 = invoke(stypy.reporting.localization.Localization(__file__, 57, 9), asfarray_184574, *[x_184575], **kwargs_184576)
    
    # Assigning a type to the variable 'x0' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'x0', asfarray_call_result_184577)
    
    # Assigning a Call to a Name (line 58):
    
    # Assigning a Call to a Name (line 58):
    
    # Call to atleast_1d(...): (line 58)
    # Processing the call arguments (line 58)
    
    # Call to func(...): (line 58)
    
    # Obtaining an instance of the builtin type 'tuple' (line 58)
    tuple_184580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 58)
    # Adding element type (line 58)
    # Getting the type of 'x0' (line 58)
    x0_184581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'x0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 28), tuple_184580, x0_184581)
    
    # Getting the type of 'args' (line 58)
    args_184582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 33), 'args', False)
    # Applying the binary operator '+' (line 58)
    result_add_184583 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 27), '+', tuple_184580, args_184582)
    
    # Processing the call keyword arguments (line 58)
    kwargs_184584 = {}
    # Getting the type of 'func' (line 58)
    func_184579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'func', False)
    # Calling func(args, kwargs) (line 58)
    func_call_result_184585 = invoke(stypy.reporting.localization.Localization(__file__, 58, 20), func_184579, *[result_add_184583], **kwargs_184584)
    
    # Processing the call keyword arguments (line 58)
    kwargs_184586 = {}
    # Getting the type of 'atleast_1d' (line 58)
    atleast_1d_184578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 9), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 58)
    atleast_1d_call_result_184587 = invoke(stypy.reporting.localization.Localization(__file__, 58, 9), atleast_1d_184578, *[func_call_result_184585], **kwargs_184586)
    
    # Assigning a type to the variable 'f0' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'f0', atleast_1d_call_result_184587)
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to zeros(...): (line 59)
    # Processing the call arguments (line 59)
    
    # Obtaining an instance of the builtin type 'list' (line 59)
    list_184589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 59)
    # Adding element type (line 59)
    
    # Call to len(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'x0' (line 59)
    x0_184591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'x0', False)
    # Processing the call keyword arguments (line 59)
    kwargs_184592 = {}
    # Getting the type of 'len' (line 59)
    len_184590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 17), 'len', False)
    # Calling len(args, kwargs) (line 59)
    len_call_result_184593 = invoke(stypy.reporting.localization.Localization(__file__, 59, 17), len_184590, *[x0_184591], **kwargs_184592)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 16), list_184589, len_call_result_184593)
    # Adding element type (line 59)
    
    # Call to len(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'f0' (line 59)
    f0_184595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'f0', False)
    # Processing the call keyword arguments (line 59)
    kwargs_184596 = {}
    # Getting the type of 'len' (line 59)
    len_184594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'len', False)
    # Calling len(args, kwargs) (line 59)
    len_call_result_184597 = invoke(stypy.reporting.localization.Localization(__file__, 59, 26), len_184594, *[f0_184595], **kwargs_184596)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 16), list_184589, len_call_result_184597)
    
    # Processing the call keyword arguments (line 59)
    kwargs_184598 = {}
    # Getting the type of 'zeros' (line 59)
    zeros_184588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 10), 'zeros', False)
    # Calling zeros(args, kwargs) (line 59)
    zeros_call_result_184599 = invoke(stypy.reporting.localization.Localization(__file__, 59, 10), zeros_184588, *[list_184589], **kwargs_184598)
    
    # Assigning a type to the variable 'jac' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'jac', zeros_call_result_184599)
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to zeros(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Call to len(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'x0' (line 60)
    x0_184602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'x0', False)
    # Processing the call keyword arguments (line 60)
    kwargs_184603 = {}
    # Getting the type of 'len' (line 60)
    len_184601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'len', False)
    # Calling len(args, kwargs) (line 60)
    len_call_result_184604 = invoke(stypy.reporting.localization.Localization(__file__, 60, 15), len_184601, *[x0_184602], **kwargs_184603)
    
    # Processing the call keyword arguments (line 60)
    kwargs_184605 = {}
    # Getting the type of 'zeros' (line 60)
    zeros_184600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 9), 'zeros', False)
    # Calling zeros(args, kwargs) (line 60)
    zeros_call_result_184606 = invoke(stypy.reporting.localization.Localization(__file__, 60, 9), zeros_184600, *[len_call_result_184604], **kwargs_184605)
    
    # Assigning a type to the variable 'dx' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'dx', zeros_call_result_184606)
    
    
    # Call to range(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Call to len(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'x0' (line 61)
    x0_184609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'x0', False)
    # Processing the call keyword arguments (line 61)
    kwargs_184610 = {}
    # Getting the type of 'len' (line 61)
    len_184608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'len', False)
    # Calling len(args, kwargs) (line 61)
    len_call_result_184611 = invoke(stypy.reporting.localization.Localization(__file__, 61, 19), len_184608, *[x0_184609], **kwargs_184610)
    
    # Processing the call keyword arguments (line 61)
    kwargs_184612 = {}
    # Getting the type of 'range' (line 61)
    range_184607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'range', False)
    # Calling range(args, kwargs) (line 61)
    range_call_result_184613 = invoke(stypy.reporting.localization.Localization(__file__, 61, 13), range_184607, *[len_call_result_184611], **kwargs_184612)
    
    # Testing the type of a for loop iterable (line 61)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 4), range_call_result_184613)
    # Getting the type of the for loop variable (line 61)
    for_loop_var_184614 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 4), range_call_result_184613)
    # Assigning a type to the variable 'i' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'i', for_loop_var_184614)
    # SSA begins for a for statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Subscript (line 62):
    
    # Assigning a Name to a Subscript (line 62):
    # Getting the type of 'epsilon' (line 62)
    epsilon_184615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'epsilon')
    # Getting the type of 'dx' (line 62)
    dx_184616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'dx')
    # Getting the type of 'i' (line 62)
    i_184617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'i')
    # Storing an element on a container (line 62)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 8), dx_184616, (i_184617, epsilon_184615))
    
    # Assigning a BinOp to a Subscript (line 63):
    
    # Assigning a BinOp to a Subscript (line 63):
    
    # Call to func(...): (line 63)
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_184619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    # Getting the type of 'x0' (line 63)
    x0_184620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 26), 'x0', False)
    # Getting the type of 'dx' (line 63)
    dx_184621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'dx', False)
    # Applying the binary operator '+' (line 63)
    result_add_184622 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 26), '+', x0_184620, dx_184621)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 26), tuple_184619, result_add_184622)
    
    # Getting the type of 'args' (line 63)
    args_184623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 34), 'args', False)
    # Applying the binary operator '+' (line 63)
    result_add_184624 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 25), '+', tuple_184619, args_184623)
    
    # Processing the call keyword arguments (line 63)
    kwargs_184625 = {}
    # Getting the type of 'func' (line 63)
    func_184618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'func', False)
    # Calling func(args, kwargs) (line 63)
    func_call_result_184626 = invoke(stypy.reporting.localization.Localization(__file__, 63, 18), func_184618, *[result_add_184624], **kwargs_184625)
    
    # Getting the type of 'f0' (line 63)
    f0_184627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 43), 'f0')
    # Applying the binary operator '-' (line 63)
    result_sub_184628 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 18), '-', func_call_result_184626, f0_184627)
    
    # Getting the type of 'epsilon' (line 63)
    epsilon_184629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'epsilon')
    # Applying the binary operator 'div' (line 63)
    result_div_184630 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 17), 'div', result_sub_184628, epsilon_184629)
    
    # Getting the type of 'jac' (line 63)
    jac_184631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'jac')
    # Getting the type of 'i' (line 63)
    i_184632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'i')
    # Storing an element on a container (line 63)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 8), jac_184631, (i_184632, result_div_184630))
    
    # Assigning a Num to a Subscript (line 64):
    
    # Assigning a Num to a Subscript (line 64):
    float_184633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'float')
    # Getting the type of 'dx' (line 64)
    dx_184634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'dx')
    # Getting the type of 'i' (line 64)
    i_184635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'i')
    # Storing an element on a container (line 64)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 8), dx_184634, (i_184635, float_184633))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to transpose(...): (line 66)
    # Processing the call keyword arguments (line 66)
    kwargs_184638 = {}
    # Getting the type of 'jac' (line 66)
    jac_184636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'jac', False)
    # Obtaining the member 'transpose' of a type (line 66)
    transpose_184637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), jac_184636, 'transpose')
    # Calling transpose(args, kwargs) (line 66)
    transpose_call_result_184639 = invoke(stypy.reporting.localization.Localization(__file__, 66, 11), transpose_184637, *[], **kwargs_184638)
    
    # Assigning a type to the variable 'stypy_return_type' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type', transpose_call_result_184639)
    
    # ################# End of 'approx_jacobian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'approx_jacobian' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_184640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_184640)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'approx_jacobian'
    return stypy_return_type_184640

# Assigning a type to the variable 'approx_jacobian' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'approx_jacobian', approx_jacobian)

@norecursion
def fmin_slsqp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_184641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    
    # Getting the type of 'None' (line 69)
    None_184642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 45), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_184643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    
    # Getting the type of 'None' (line 69)
    None_184644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 73), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 70)
    tuple_184645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 70)
    
    # Getting the type of 'None' (line 70)
    None_184646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 33), 'None')
    # Getting the type of 'None' (line 70)
    None_184647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 53), 'None')
    # Getting the type of 'None' (line 71)
    None_184648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 30), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 71)
    tuple_184649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 71)
    
    int_184650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 50), 'int')
    float_184651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 59), 'float')
    int_184652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 22), 'int')
    # Getting the type of 'None' (line 72)
    None_184653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'None')
    int_184654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 48), 'int')
    # Getting the type of '_epsilon' (line 72)
    _epsilon_184655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 59), '_epsilon')
    # Getting the type of 'None' (line 73)
    None_184656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'None')
    defaults = [tuple_184641, None_184642, tuple_184643, None_184644, tuple_184645, None_184646, None_184647, None_184648, tuple_184649, int_184650, float_184651, int_184652, None_184653, int_184654, _epsilon_184655, None_184656]
    # Create a new context for function 'fmin_slsqp'
    module_type_store = module_type_store.open_function_context('fmin_slsqp', 69, 0, False)
    
    # Passed parameters checking function
    fmin_slsqp.stypy_localization = localization
    fmin_slsqp.stypy_type_of_self = None
    fmin_slsqp.stypy_type_store = module_type_store
    fmin_slsqp.stypy_function_name = 'fmin_slsqp'
    fmin_slsqp.stypy_param_names_list = ['func', 'x0', 'eqcons', 'f_eqcons', 'ieqcons', 'f_ieqcons', 'bounds', 'fprime', 'fprime_eqcons', 'fprime_ieqcons', 'args', 'iter', 'acc', 'iprint', 'disp', 'full_output', 'epsilon', 'callback']
    fmin_slsqp.stypy_varargs_param_name = None
    fmin_slsqp.stypy_kwargs_param_name = None
    fmin_slsqp.stypy_call_defaults = defaults
    fmin_slsqp.stypy_call_varargs = varargs
    fmin_slsqp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fmin_slsqp', ['func', 'x0', 'eqcons', 'f_eqcons', 'ieqcons', 'f_ieqcons', 'bounds', 'fprime', 'fprime_eqcons', 'fprime_ieqcons', 'args', 'iter', 'acc', 'iprint', 'disp', 'full_output', 'epsilon', 'callback'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fmin_slsqp', localization, ['func', 'x0', 'eqcons', 'f_eqcons', 'ieqcons', 'f_ieqcons', 'bounds', 'fprime', 'fprime_eqcons', 'fprime_ieqcons', 'args', 'iter', 'acc', 'iprint', 'disp', 'full_output', 'epsilon', 'callback'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fmin_slsqp(...)' code ##################

    str_184657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, (-1)), 'str', "\n    Minimize a function using Sequential Least SQuares Programming\n\n    Python interface function for the SLSQP Optimization subroutine\n    originally implemented by Dieter Kraft.\n\n    Parameters\n    ----------\n    func : callable f(x,*args)\n        Objective function.  Must return a scalar.\n    x0 : 1-D ndarray of float\n        Initial guess for the independent variable(s).\n    eqcons : list, optional\n        A list of functions of length n such that\n        eqcons[j](x,*args) == 0.0 in a successfully optimized\n        problem.\n    f_eqcons : callable f(x,*args), optional\n        Returns a 1-D array in which each element must equal 0.0 in a\n        successfully optimized problem.  If f_eqcons is specified,\n        eqcons is ignored.\n    ieqcons : list, optional\n        A list of functions of length n such that\n        ieqcons[j](x,*args) >= 0.0 in a successfully optimized\n        problem.\n    f_ieqcons : callable f(x,*args), optional\n        Returns a 1-D ndarray in which each element must be greater or\n        equal to 0.0 in a successfully optimized problem.  If\n        f_ieqcons is specified, ieqcons is ignored.\n    bounds : list, optional\n        A list of tuples specifying the lower and upper bound\n        for each independent variable [(xl0, xu0),(xl1, xu1),...]\n        Infinite values will be interpreted as large floating values.\n    fprime : callable `f(x,*args)`, optional\n        A function that evaluates the partial derivatives of func.\n    fprime_eqcons : callable `f(x,*args)`, optional\n        A function of the form `f(x, *args)` that returns the m by n\n        array of equality constraint normals.  If not provided,\n        the normals will be approximated. The array returned by\n        fprime_eqcons should be sized as ( len(eqcons), len(x0) ).\n    fprime_ieqcons : callable `f(x,*args)`, optional\n        A function of the form `f(x, *args)` that returns the m by n\n        array of inequality constraint normals.  If not provided,\n        the normals will be approximated. The array returned by\n        fprime_ieqcons should be sized as ( len(ieqcons), len(x0) ).\n    args : sequence, optional\n        Additional arguments passed to func and fprime.\n    iter : int, optional\n        The maximum number of iterations.\n    acc : float, optional\n        Requested accuracy.\n    iprint : int, optional\n        The verbosity of fmin_slsqp :\n\n        * iprint <= 0 : Silent operation\n        * iprint == 1 : Print summary upon completion (default)\n        * iprint >= 2 : Print status of each iterate and summary\n    disp : int, optional\n        Over-rides the iprint interface (preferred).\n    full_output : bool, optional\n        If False, return only the minimizer of func (default).\n        Otherwise, output final objective function and summary\n        information.\n    epsilon : float, optional\n        The step size for finite-difference derivative estimates.\n    callback : callable, optional\n        Called after each iteration, as ``callback(x)``, where ``x`` is the\n        current parameter vector.\n\n    Returns\n    -------\n    out : ndarray of float\n        The final minimizer of func.\n    fx : ndarray of float, if full_output is true\n        The final value of the objective function.\n    its : int, if full_output is true\n        The number of iterations.\n    imode : int, if full_output is true\n        The exit mode from the optimizer (see below).\n    smode : string, if full_output is true\n        Message describing the exit mode from the optimizer.\n\n    See also\n    --------\n    minimize: Interface to minimization algorithms for multivariate\n        functions. See the 'SLSQP' `method` in particular.\n\n    Notes\n    -----\n    Exit modes are defined as follows ::\n\n        -1 : Gradient evaluation required (g & a)\n         0 : Optimization terminated successfully.\n         1 : Function evaluation required (f & c)\n         2 : More equality constraints than independent variables\n         3 : More than 3*n iterations in LSQ subproblem\n         4 : Inequality constraints incompatible\n         5 : Singular matrix E in LSQ subproblem\n         6 : Singular matrix C in LSQ subproblem\n         7 : Rank-deficient equality constraint subproblem HFTI\n         8 : Positive directional derivative for linesearch\n         9 : Iteration limit exceeded\n\n    Examples\n    --------\n    Examples are given :ref:`in the tutorial <tutorial-sqlsp>`.\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 181)
    # Getting the type of 'disp' (line 181)
    disp_184658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'disp')
    # Getting the type of 'None' (line 181)
    None_184659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'None')
    
    (may_be_184660, more_types_in_union_184661) = may_not_be_none(disp_184658, None_184659)

    if may_be_184660:

        if more_types_in_union_184661:
            # Runtime conditional SSA (line 181)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 182):
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'disp' (line 182)
        disp_184662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'disp')
        # Assigning a type to the variable 'iprint' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'iprint', disp_184662)

        if more_types_in_union_184661:
            # SSA join for if statement (line 181)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Dict to a Name (line 183):
    
    # Assigning a Dict to a Name (line 183):
    
    # Obtaining an instance of the builtin type 'dict' (line 183)
    dict_184663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 183)
    # Adding element type (key, value) (line 183)
    str_184664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 12), 'str', 'maxiter')
    # Getting the type of 'iter' (line 183)
    iter_184665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 23), 'iter')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 11), dict_184663, (str_184664, iter_184665))
    # Adding element type (key, value) (line 183)
    str_184666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 12), 'str', 'ftol')
    # Getting the type of 'acc' (line 184)
    acc_184667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'acc')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 11), dict_184663, (str_184666, acc_184667))
    # Adding element type (key, value) (line 183)
    str_184668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 12), 'str', 'iprint')
    # Getting the type of 'iprint' (line 185)
    iprint_184669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 22), 'iprint')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 11), dict_184663, (str_184668, iprint_184669))
    # Adding element type (key, value) (line 183)
    str_184670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 12), 'str', 'disp')
    
    # Getting the type of 'iprint' (line 186)
    iprint_184671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'iprint')
    int_184672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 30), 'int')
    # Applying the binary operator '!=' (line 186)
    result_ne_184673 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 20), '!=', iprint_184671, int_184672)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 11), dict_184663, (str_184670, result_ne_184673))
    # Adding element type (key, value) (line 183)
    str_184674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 12), 'str', 'eps')
    # Getting the type of 'epsilon' (line 187)
    epsilon_184675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'epsilon')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 11), dict_184663, (str_184674, epsilon_184675))
    # Adding element type (key, value) (line 183)
    str_184676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 12), 'str', 'callback')
    # Getting the type of 'callback' (line 188)
    callback_184677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'callback')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 11), dict_184663, (str_184676, callback_184677))
    
    # Assigning a type to the variable 'opts' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'opts', dict_184663)
    
    # Assigning a Tuple to a Name (line 191):
    
    # Assigning a Tuple to a Name (line 191):
    
    # Obtaining an instance of the builtin type 'tuple' (line 191)
    tuple_184678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 191)
    
    # Assigning a type to the variable 'cons' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'cons', tuple_184678)
    
    # Getting the type of 'cons' (line 194)
    cons_184679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'cons')
    
    # Call to tuple(...): (line 194)
    # Processing the call arguments (line 194)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 194, 18, True)
    # Calculating comprehension expression
    # Getting the type of 'eqcons' (line 194)
    eqcons_184688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 66), 'eqcons', False)
    comprehension_184689 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 18), eqcons_184688)
    # Assigning a type to the variable 'c' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 18), 'c', comprehension_184689)
    
    # Obtaining an instance of the builtin type 'dict' (line 194)
    dict_184681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 18), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 194)
    # Adding element type (key, value) (line 194)
    str_184682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 19), 'str', 'type')
    str_184683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 27), 'str', 'eq')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 18), dict_184681, (str_184682, str_184683))
    # Adding element type (key, value) (line 194)
    str_184684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 33), 'str', 'fun')
    # Getting the type of 'c' (line 194)
    c_184685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 40), 'c', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 18), dict_184681, (str_184684, c_184685))
    # Adding element type (key, value) (line 194)
    str_184686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 43), 'str', 'args')
    # Getting the type of 'args' (line 194)
    args_184687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 51), 'args', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 18), dict_184681, (str_184686, args_184687))
    
    list_184690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 18), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 18), list_184690, dict_184681)
    # Processing the call keyword arguments (line 194)
    kwargs_184691 = {}
    # Getting the type of 'tuple' (line 194)
    tuple_184680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'tuple', False)
    # Calling tuple(args, kwargs) (line 194)
    tuple_call_result_184692 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), tuple_184680, *[list_184690], **kwargs_184691)
    
    # Applying the binary operator '+=' (line 194)
    result_iadd_184693 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 4), '+=', cons_184679, tuple_call_result_184692)
    # Assigning a type to the variable 'cons' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'cons', result_iadd_184693)
    
    
    # Getting the type of 'cons' (line 195)
    cons_184694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'cons')
    
    # Call to tuple(...): (line 195)
    # Processing the call arguments (line 195)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 195, 18, True)
    # Calculating comprehension expression
    # Getting the type of 'ieqcons' (line 195)
    ieqcons_184703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 68), 'ieqcons', False)
    comprehension_184704 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), ieqcons_184703)
    # Assigning a type to the variable 'c' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 18), 'c', comprehension_184704)
    
    # Obtaining an instance of the builtin type 'dict' (line 195)
    dict_184696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 195)
    # Adding element type (key, value) (line 195)
    str_184697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 19), 'str', 'type')
    str_184698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 27), 'str', 'ineq')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), dict_184696, (str_184697, str_184698))
    # Adding element type (key, value) (line 195)
    str_184699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 35), 'str', 'fun')
    # Getting the type of 'c' (line 195)
    c_184700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 42), 'c', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), dict_184696, (str_184699, c_184700))
    # Adding element type (key, value) (line 195)
    str_184701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 45), 'str', 'args')
    # Getting the type of 'args' (line 195)
    args_184702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 53), 'args', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), dict_184696, (str_184701, args_184702))
    
    list_184705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), list_184705, dict_184696)
    # Processing the call keyword arguments (line 195)
    kwargs_184706 = {}
    # Getting the type of 'tuple' (line 195)
    tuple_184695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'tuple', False)
    # Calling tuple(args, kwargs) (line 195)
    tuple_call_result_184707 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), tuple_184695, *[list_184705], **kwargs_184706)
    
    # Applying the binary operator '+=' (line 195)
    result_iadd_184708 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 4), '+=', cons_184694, tuple_call_result_184707)
    # Assigning a type to the variable 'cons' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'cons', result_iadd_184708)
    
    
    # Getting the type of 'f_eqcons' (line 199)
    f_eqcons_184709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 7), 'f_eqcons')
    # Testing the type of an if condition (line 199)
    if_condition_184710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 4), f_eqcons_184709)
    # Assigning a type to the variable 'if_condition_184710' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'if_condition_184710', if_condition_184710)
    # SSA begins for if statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'cons' (line 200)
    cons_184711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'cons')
    
    # Obtaining an instance of the builtin type 'tuple' (line 200)
    tuple_184712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 200)
    # Adding element type (line 200)
    
    # Obtaining an instance of the builtin type 'dict' (line 200)
    dict_184713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 200)
    # Adding element type (key, value) (line 200)
    str_184714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 18), 'str', 'type')
    str_184715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 26), 'str', 'eq')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 17), dict_184713, (str_184714, str_184715))
    # Adding element type (key, value) (line 200)
    str_184716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 32), 'str', 'fun')
    # Getting the type of 'f_eqcons' (line 200)
    f_eqcons_184717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 39), 'f_eqcons')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 17), dict_184713, (str_184716, f_eqcons_184717))
    # Adding element type (key, value) (line 200)
    str_184718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 49), 'str', 'jac')
    # Getting the type of 'fprime_eqcons' (line 200)
    fprime_eqcons_184719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 56), 'fprime_eqcons')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 17), dict_184713, (str_184718, fprime_eqcons_184719))
    # Adding element type (key, value) (line 200)
    str_184720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 18), 'str', 'args')
    # Getting the type of 'args' (line 201)
    args_184721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 26), 'args')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 17), dict_184713, (str_184720, args_184721))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 17), tuple_184712, dict_184713)
    
    # Applying the binary operator '+=' (line 200)
    result_iadd_184722 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 8), '+=', cons_184711, tuple_184712)
    # Assigning a type to the variable 'cons' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'cons', result_iadd_184722)
    
    # SSA join for if statement (line 199)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'f_ieqcons' (line 202)
    f_ieqcons_184723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 7), 'f_ieqcons')
    # Testing the type of an if condition (line 202)
    if_condition_184724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 4), f_ieqcons_184723)
    # Assigning a type to the variable 'if_condition_184724' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'if_condition_184724', if_condition_184724)
    # SSA begins for if statement (line 202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'cons' (line 203)
    cons_184725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'cons')
    
    # Obtaining an instance of the builtin type 'tuple' (line 203)
    tuple_184726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 203)
    # Adding element type (line 203)
    
    # Obtaining an instance of the builtin type 'dict' (line 203)
    dict_184727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 203)
    # Adding element type (key, value) (line 203)
    str_184728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 18), 'str', 'type')
    str_184729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 26), 'str', 'ineq')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 17), dict_184727, (str_184728, str_184729))
    # Adding element type (key, value) (line 203)
    str_184730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 34), 'str', 'fun')
    # Getting the type of 'f_ieqcons' (line 203)
    f_ieqcons_184731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 41), 'f_ieqcons')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 17), dict_184727, (str_184730, f_ieqcons_184731))
    # Adding element type (key, value) (line 203)
    str_184732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 52), 'str', 'jac')
    # Getting the type of 'fprime_ieqcons' (line 203)
    fprime_ieqcons_184733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 59), 'fprime_ieqcons')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 17), dict_184727, (str_184732, fprime_ieqcons_184733))
    # Adding element type (key, value) (line 203)
    str_184734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 18), 'str', 'args')
    # Getting the type of 'args' (line 204)
    args_184735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 26), 'args')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 17), dict_184727, (str_184734, args_184735))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 17), tuple_184726, dict_184727)
    
    # Applying the binary operator '+=' (line 203)
    result_iadd_184736 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 8), '+=', cons_184725, tuple_184726)
    # Assigning a type to the variable 'cons' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'cons', result_iadd_184736)
    
    # SSA join for if statement (line 202)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 206):
    
    # Assigning a Call to a Name (line 206):
    
    # Call to _minimize_slsqp(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'func' (line 206)
    func_184738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'func', False)
    # Getting the type of 'x0' (line 206)
    x0_184739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 32), 'x0', False)
    # Getting the type of 'args' (line 206)
    args_184740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 36), 'args', False)
    # Processing the call keyword arguments (line 206)
    # Getting the type of 'fprime' (line 206)
    fprime_184741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 46), 'fprime', False)
    keyword_184742 = fprime_184741
    # Getting the type of 'bounds' (line 206)
    bounds_184743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 61), 'bounds', False)
    keyword_184744 = bounds_184743
    # Getting the type of 'cons' (line 207)
    cons_184745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 38), 'cons', False)
    keyword_184746 = cons_184745
    # Getting the type of 'opts' (line 207)
    opts_184747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 46), 'opts', False)
    kwargs_184748 = {'opts_184747': opts_184747, 'constraints': keyword_184746, 'jac': keyword_184742, 'bounds': keyword_184744}
    # Getting the type of '_minimize_slsqp' (line 206)
    _minimize_slsqp_184737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 10), '_minimize_slsqp', False)
    # Calling _minimize_slsqp(args, kwargs) (line 206)
    _minimize_slsqp_call_result_184749 = invoke(stypy.reporting.localization.Localization(__file__, 206, 10), _minimize_slsqp_184737, *[func_184738, x0_184739, args_184740], **kwargs_184748)
    
    # Assigning a type to the variable 'res' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'res', _minimize_slsqp_call_result_184749)
    
    # Getting the type of 'full_output' (line 208)
    full_output_184750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 7), 'full_output')
    # Testing the type of an if condition (line 208)
    if_condition_184751 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 4), full_output_184750)
    # Assigning a type to the variable 'if_condition_184751' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'if_condition_184751', if_condition_184751)
    # SSA begins for if statement (line 208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 209)
    tuple_184752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 209)
    # Adding element type (line 209)
    
    # Obtaining the type of the subscript
    str_184753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 19), 'str', 'x')
    # Getting the type of 'res' (line 209)
    res_184754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'res')
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___184755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 15), res_184754, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_184756 = invoke(stypy.reporting.localization.Localization(__file__, 209, 15), getitem___184755, str_184753)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), tuple_184752, subscript_call_result_184756)
    # Adding element type (line 209)
    
    # Obtaining the type of the subscript
    str_184757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 29), 'str', 'fun')
    # Getting the type of 'res' (line 209)
    res_184758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 25), 'res')
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___184759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 25), res_184758, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_184760 = invoke(stypy.reporting.localization.Localization(__file__, 209, 25), getitem___184759, str_184757)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), tuple_184752, subscript_call_result_184760)
    # Adding element type (line 209)
    
    # Obtaining the type of the subscript
    str_184761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 41), 'str', 'nit')
    # Getting the type of 'res' (line 209)
    res_184762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 37), 'res')
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___184763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 37), res_184762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_184764 = invoke(stypy.reporting.localization.Localization(__file__, 209, 37), getitem___184763, str_184761)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), tuple_184752, subscript_call_result_184764)
    # Adding element type (line 209)
    
    # Obtaining the type of the subscript
    str_184765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 53), 'str', 'status')
    # Getting the type of 'res' (line 209)
    res_184766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 49), 'res')
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___184767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 49), res_184766, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_184768 = invoke(stypy.reporting.localization.Localization(__file__, 209, 49), getitem___184767, str_184765)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), tuple_184752, subscript_call_result_184768)
    # Adding element type (line 209)
    
    # Obtaining the type of the subscript
    str_184769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 68), 'str', 'message')
    # Getting the type of 'res' (line 209)
    res_184770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 64), 'res')
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___184771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 64), res_184770, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_184772 = invoke(stypy.reporting.localization.Localization(__file__, 209, 64), getitem___184771, str_184769)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), tuple_184752, subscript_call_result_184772)
    
    # Assigning a type to the variable 'stypy_return_type' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'stypy_return_type', tuple_184752)
    # SSA branch for the else part of an if statement (line 208)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining the type of the subscript
    str_184773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 19), 'str', 'x')
    # Getting the type of 'res' (line 211)
    res_184774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'res')
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___184775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 15), res_184774, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_184776 = invoke(stypy.reporting.localization.Localization(__file__, 211, 15), getitem___184775, str_184773)
    
    # Assigning a type to the variable 'stypy_return_type' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'stypy_return_type', subscript_call_result_184776)
    # SSA join for if statement (line 208)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'fmin_slsqp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fmin_slsqp' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_184777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_184777)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fmin_slsqp'
    return stypy_return_type_184777

# Assigning a type to the variable 'fmin_slsqp' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'fmin_slsqp', fmin_slsqp)

@norecursion
def _minimize_slsqp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 214)
    tuple_184778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 214)
    
    # Getting the type of 'None' (line 214)
    None_184779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 43), 'None')
    # Getting the type of 'None' (line 214)
    None_184780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 56), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 215)
    tuple_184781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 215)
    
    int_184782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 28), 'int')
    float_184783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 38), 'float')
    int_184784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 53), 'int')
    # Getting the type of 'False' (line 216)
    False_184785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 61), 'False')
    # Getting the type of '_epsilon' (line 217)
    _epsilon_184786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), '_epsilon')
    # Getting the type of 'None' (line 217)
    None_184787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 43), 'None')
    defaults = [tuple_184778, None_184779, None_184780, tuple_184781, int_184782, float_184783, int_184784, False_184785, _epsilon_184786, None_184787]
    # Create a new context for function '_minimize_slsqp'
    module_type_store = module_type_store.open_function_context('_minimize_slsqp', 214, 0, False)
    
    # Passed parameters checking function
    _minimize_slsqp.stypy_localization = localization
    _minimize_slsqp.stypy_type_of_self = None
    _minimize_slsqp.stypy_type_store = module_type_store
    _minimize_slsqp.stypy_function_name = '_minimize_slsqp'
    _minimize_slsqp.stypy_param_names_list = ['func', 'x0', 'args', 'jac', 'bounds', 'constraints', 'maxiter', 'ftol', 'iprint', 'disp', 'eps', 'callback']
    _minimize_slsqp.stypy_varargs_param_name = None
    _minimize_slsqp.stypy_kwargs_param_name = 'unknown_options'
    _minimize_slsqp.stypy_call_defaults = defaults
    _minimize_slsqp.stypy_call_varargs = varargs
    _minimize_slsqp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_minimize_slsqp', ['func', 'x0', 'args', 'jac', 'bounds', 'constraints', 'maxiter', 'ftol', 'iprint', 'disp', 'eps', 'callback'], None, 'unknown_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_minimize_slsqp', localization, ['func', 'x0', 'args', 'jac', 'bounds', 'constraints', 'maxiter', 'ftol', 'iprint', 'disp', 'eps', 'callback'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_minimize_slsqp(...)' code ##################

    str_184788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, (-1)), 'str', '\n    Minimize a scalar function of one or more variables using Sequential\n    Least SQuares Programming (SLSQP).\n\n    Options\n    -------\n    ftol : float\n        Precision goal for the value of f in the stopping criterion.\n    eps : float\n        Step size used for numerical approximation of the Jacobian.\n    disp : bool\n        Set to True to print convergence messages. If False,\n        `verbosity` is ignored and set to 0.\n    maxiter : int\n        Maximum number of iterations.\n\n    ')
    
    # Call to _check_unknown_options(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'unknown_options' (line 236)
    unknown_options_184790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 27), 'unknown_options', False)
    # Processing the call keyword arguments (line 236)
    kwargs_184791 = {}
    # Getting the type of '_check_unknown_options' (line 236)
    _check_unknown_options_184789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), '_check_unknown_options', False)
    # Calling _check_unknown_options(args, kwargs) (line 236)
    _check_unknown_options_call_result_184792 = invoke(stypy.reporting.localization.Localization(__file__, 236, 4), _check_unknown_options_184789, *[unknown_options_184790], **kwargs_184791)
    
    
    # Assigning a Name to a Name (line 237):
    
    # Assigning a Name to a Name (line 237):
    # Getting the type of 'jac' (line 237)
    jac_184793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 13), 'jac')
    # Assigning a type to the variable 'fprime' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'fprime', jac_184793)
    
    # Assigning a Name to a Name (line 238):
    
    # Assigning a Name to a Name (line 238):
    # Getting the type of 'maxiter' (line 238)
    maxiter_184794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'maxiter')
    # Assigning a type to the variable 'iter' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'iter', maxiter_184794)
    
    # Assigning a Name to a Name (line 239):
    
    # Assigning a Name to a Name (line 239):
    # Getting the type of 'ftol' (line 239)
    ftol_184795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 10), 'ftol')
    # Assigning a type to the variable 'acc' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'acc', ftol_184795)
    
    # Assigning a Name to a Name (line 240):
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'eps' (line 240)
    eps_184796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 14), 'eps')
    # Assigning a type to the variable 'epsilon' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'epsilon', eps_184796)
    
    
    # Getting the type of 'disp' (line 242)
    disp_184797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'disp')
    # Applying the 'not' unary operator (line 242)
    result_not__184798 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 7), 'not', disp_184797)
    
    # Testing the type of an if condition (line 242)
    if_condition_184799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 4), result_not__184798)
    # Assigning a type to the variable 'if_condition_184799' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'if_condition_184799', if_condition_184799)
    # SSA begins for if statement (line 242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 243):
    
    # Assigning a Num to a Name (line 243):
    int_184800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 17), 'int')
    # Assigning a type to the variable 'iprint' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'iprint', int_184800)
    # SSA join for if statement (line 242)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 246)
    # Getting the type of 'dict' (line 246)
    dict_184801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'dict')
    # Getting the type of 'constraints' (line 246)
    constraints_184802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 18), 'constraints')
    
    (may_be_184803, more_types_in_union_184804) = may_be_subtype(dict_184801, constraints_184802)

    if may_be_184803:

        if more_types_in_union_184804:
            # Runtime conditional SSA (line 246)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'constraints' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'constraints', remove_not_subtype_from_union(constraints_184802, dict))
        
        # Assigning a Tuple to a Name (line 247):
        
        # Assigning a Tuple to a Name (line 247):
        
        # Obtaining an instance of the builtin type 'tuple' (line 247)
        tuple_184805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 247)
        # Adding element type (line 247)
        # Getting the type of 'constraints' (line 247)
        constraints_184806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'constraints')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 23), tuple_184805, constraints_184806)
        
        # Assigning a type to the variable 'constraints' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'constraints', tuple_184805)

        if more_types_in_union_184804:
            # SSA join for if statement (line 246)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Dict to a Name (line 249):
    
    # Assigning a Dict to a Name (line 249):
    
    # Obtaining an instance of the builtin type 'dict' (line 249)
    dict_184807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 249)
    # Adding element type (key, value) (line 249)
    str_184808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'str', 'eq')
    
    # Obtaining an instance of the builtin type 'tuple' (line 249)
    tuple_184809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 249)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 11), dict_184807, (str_184808, tuple_184809))
    # Adding element type (key, value) (line 249)
    str_184810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 22), 'str', 'ineq')
    
    # Obtaining an instance of the builtin type 'tuple' (line 249)
    tuple_184811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 249)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 11), dict_184807, (str_184810, tuple_184811))
    
    # Assigning a type to the variable 'cons' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'cons', dict_184807)
    
    
    # Call to enumerate(...): (line 250)
    # Processing the call arguments (line 250)
    # Getting the type of 'constraints' (line 250)
    constraints_184813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 29), 'constraints', False)
    # Processing the call keyword arguments (line 250)
    kwargs_184814 = {}
    # Getting the type of 'enumerate' (line 250)
    enumerate_184812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 250)
    enumerate_call_result_184815 = invoke(stypy.reporting.localization.Localization(__file__, 250, 19), enumerate_184812, *[constraints_184813], **kwargs_184814)
    
    # Testing the type of a for loop iterable (line 250)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 250, 4), enumerate_call_result_184815)
    # Getting the type of the for loop variable (line 250)
    for_loop_var_184816 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 250, 4), enumerate_call_result_184815)
    # Assigning a type to the variable 'ic' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'ic', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 4), for_loop_var_184816))
    # Assigning a type to the variable 'con' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'con', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 4), for_loop_var_184816))
    # SSA begins for a for statement (line 250)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 252)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 253):
    
    # Assigning a Call to a Name (line 253):
    
    # Call to lower(...): (line 253)
    # Processing the call keyword arguments (line 253)
    kwargs_184822 = {}
    
    # Obtaining the type of the subscript
    str_184817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 24), 'str', 'type')
    # Getting the type of 'con' (line 253)
    con_184818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___184819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), con_184818, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_184820 = invoke(stypy.reporting.localization.Localization(__file__, 253, 20), getitem___184819, str_184817)
    
    # Obtaining the member 'lower' of a type (line 253)
    lower_184821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), subscript_call_result_184820, 'lower')
    # Calling lower(args, kwargs) (line 253)
    lower_call_result_184823 = invoke(stypy.reporting.localization.Localization(__file__, 253, 20), lower_184821, *[], **kwargs_184822)
    
    # Assigning a type to the variable 'ctype' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'ctype', lower_call_result_184823)
    # SSA branch for the except part of a try statement (line 252)
    # SSA branch for the except 'KeyError' branch of a try statement (line 252)
    module_type_store.open_ssa_branch('except')
    
    # Call to KeyError(...): (line 255)
    # Processing the call arguments (line 255)
    str_184825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 27), 'str', 'Constraint %d has no type defined.')
    # Getting the type of 'ic' (line 255)
    ic_184826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 66), 'ic', False)
    # Applying the binary operator '%' (line 255)
    result_mod_184827 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 27), '%', str_184825, ic_184826)
    
    # Processing the call keyword arguments (line 255)
    kwargs_184828 = {}
    # Getting the type of 'KeyError' (line 255)
    KeyError_184824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 18), 'KeyError', False)
    # Calling KeyError(args, kwargs) (line 255)
    KeyError_call_result_184829 = invoke(stypy.reporting.localization.Localization(__file__, 255, 18), KeyError_184824, *[result_mod_184827], **kwargs_184828)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 255, 12), KeyError_call_result_184829, 'raise parameter', BaseException)
    # SSA branch for the except 'TypeError' branch of a try statement (line 252)
    module_type_store.open_ssa_branch('except')
    
    # Call to TypeError(...): (line 257)
    # Processing the call arguments (line 257)
    str_184831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 28), 'str', 'Constraints must be defined using a dictionary.')
    # Processing the call keyword arguments (line 257)
    kwargs_184832 = {}
    # Getting the type of 'TypeError' (line 257)
    TypeError_184830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 257)
    TypeError_call_result_184833 = invoke(stypy.reporting.localization.Localization(__file__, 257, 18), TypeError_184830, *[str_184831], **kwargs_184832)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 257, 12), TypeError_call_result_184833, 'raise parameter', BaseException)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 252)
    module_type_store.open_ssa_branch('except')
    
    # Call to TypeError(...): (line 260)
    # Processing the call arguments (line 260)
    str_184835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 28), 'str', "Constraint's type must be a string.")
    # Processing the call keyword arguments (line 260)
    kwargs_184836 = {}
    # Getting the type of 'TypeError' (line 260)
    TypeError_184834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 260)
    TypeError_call_result_184837 = invoke(stypy.reporting.localization.Localization(__file__, 260, 18), TypeError_184834, *[str_184835], **kwargs_184836)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 260, 12), TypeError_call_result_184837, 'raise parameter', BaseException)
    # SSA branch for the else branch of a try statement (line 252)
    module_type_store.open_ssa_branch('except else')
    
    
    # Getting the type of 'ctype' (line 262)
    ctype_184838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'ctype')
    
    # Obtaining an instance of the builtin type 'list' (line 262)
    list_184839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 262)
    # Adding element type (line 262)
    str_184840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 29), 'str', 'eq')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 28), list_184839, str_184840)
    # Adding element type (line 262)
    str_184841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 35), 'str', 'ineq')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 28), list_184839, str_184841)
    
    # Applying the binary operator 'notin' (line 262)
    result_contains_184842 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 15), 'notin', ctype_184838, list_184839)
    
    # Testing the type of an if condition (line 262)
    if_condition_184843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 12), result_contains_184842)
    # Assigning a type to the variable 'if_condition_184843' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'if_condition_184843', if_condition_184843)
    # SSA begins for if statement (line 262)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 263)
    # Processing the call arguments (line 263)
    str_184845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 33), 'str', "Unknown constraint type '%s'.")
    
    # Obtaining the type of the subscript
    str_184846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 71), 'str', 'type')
    # Getting the type of 'con' (line 263)
    con_184847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 67), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___184848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 67), con_184847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_184849 = invoke(stypy.reporting.localization.Localization(__file__, 263, 67), getitem___184848, str_184846)
    
    # Applying the binary operator '%' (line 263)
    result_mod_184850 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 33), '%', str_184845, subscript_call_result_184849)
    
    # Processing the call keyword arguments (line 263)
    kwargs_184851 = {}
    # Getting the type of 'ValueError' (line 263)
    ValueError_184844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 263)
    ValueError_call_result_184852 = invoke(stypy.reporting.localization.Localization(__file__, 263, 22), ValueError_184844, *[result_mod_184850], **kwargs_184851)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 263, 16), ValueError_call_result_184852, 'raise parameter', BaseException)
    # SSA join for if statement (line 262)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 252)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_184853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 11), 'str', 'fun')
    # Getting the type of 'con' (line 266)
    con_184854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 24), 'con')
    # Applying the binary operator 'notin' (line 266)
    result_contains_184855 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 11), 'notin', str_184853, con_184854)
    
    # Testing the type of an if condition (line 266)
    if_condition_184856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 8), result_contains_184855)
    # Assigning a type to the variable 'if_condition_184856' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'if_condition_184856', if_condition_184856)
    # SSA begins for if statement (line 266)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 267)
    # Processing the call arguments (line 267)
    str_184858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 29), 'str', 'Constraint %d has no function defined.')
    # Getting the type of 'ic' (line 267)
    ic_184859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 72), 'ic', False)
    # Applying the binary operator '%' (line 267)
    result_mod_184860 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 29), '%', str_184858, ic_184859)
    
    # Processing the call keyword arguments (line 267)
    kwargs_184861 = {}
    # Getting the type of 'ValueError' (line 267)
    ValueError_184857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 267)
    ValueError_call_result_184862 = invoke(stypy.reporting.localization.Localization(__file__, 267, 18), ValueError_184857, *[result_mod_184860], **kwargs_184861)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 267, 12), ValueError_call_result_184862, 'raise parameter', BaseException)
    # SSA join for if statement (line 266)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 270):
    
    # Assigning a Call to a Name (line 270):
    
    # Call to get(...): (line 270)
    # Processing the call arguments (line 270)
    str_184865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 23), 'str', 'jac')
    # Processing the call keyword arguments (line 270)
    kwargs_184866 = {}
    # Getting the type of 'con' (line 270)
    con_184863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'con', False)
    # Obtaining the member 'get' of a type (line 270)
    get_184864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 15), con_184863, 'get')
    # Calling get(args, kwargs) (line 270)
    get_call_result_184867 = invoke(stypy.reporting.localization.Localization(__file__, 270, 15), get_184864, *[str_184865], **kwargs_184866)
    
    # Assigning a type to the variable 'cjac' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'cjac', get_call_result_184867)
    
    # Type idiom detected: calculating its left and rigth part (line 271)
    # Getting the type of 'cjac' (line 271)
    cjac_184868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 11), 'cjac')
    # Getting the type of 'None' (line 271)
    None_184869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), 'None')
    
    (may_be_184870, more_types_in_union_184871) = may_be_none(cjac_184868, None_184869)

    if may_be_184870:

        if more_types_in_union_184871:
            # Runtime conditional SSA (line 271)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def cjac_factory(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cjac_factory'
            module_type_store = module_type_store.open_function_context('cjac_factory', 274, 12, False)
            
            # Passed parameters checking function
            cjac_factory.stypy_localization = localization
            cjac_factory.stypy_type_of_self = None
            cjac_factory.stypy_type_store = module_type_store
            cjac_factory.stypy_function_name = 'cjac_factory'
            cjac_factory.stypy_param_names_list = ['fun']
            cjac_factory.stypy_varargs_param_name = None
            cjac_factory.stypy_kwargs_param_name = None
            cjac_factory.stypy_call_defaults = defaults
            cjac_factory.stypy_call_varargs = varargs
            cjac_factory.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cjac_factory', ['fun'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cjac_factory', localization, ['fun'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cjac_factory(...)' code ##################


            @norecursion
            def cjac(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'cjac'
                module_type_store = module_type_store.open_function_context('cjac', 275, 16, False)
                
                # Passed parameters checking function
                cjac.stypy_localization = localization
                cjac.stypy_type_of_self = None
                cjac.stypy_type_store = module_type_store
                cjac.stypy_function_name = 'cjac'
                cjac.stypy_param_names_list = ['x']
                cjac.stypy_varargs_param_name = 'args'
                cjac.stypy_kwargs_param_name = None
                cjac.stypy_call_defaults = defaults
                cjac.stypy_call_varargs = varargs
                cjac.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'cjac', ['x'], 'args', None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'cjac', localization, ['x'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'cjac(...)' code ##################

                
                # Call to approx_jacobian(...): (line 276)
                # Processing the call arguments (line 276)
                # Getting the type of 'x' (line 276)
                x_184873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 43), 'x', False)
                # Getting the type of 'fun' (line 276)
                fun_184874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 46), 'fun', False)
                # Getting the type of 'epsilon' (line 276)
                epsilon_184875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 51), 'epsilon', False)
                # Getting the type of 'args' (line 276)
                args_184876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 61), 'args', False)
                # Processing the call keyword arguments (line 276)
                kwargs_184877 = {}
                # Getting the type of 'approx_jacobian' (line 276)
                approx_jacobian_184872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 27), 'approx_jacobian', False)
                # Calling approx_jacobian(args, kwargs) (line 276)
                approx_jacobian_call_result_184878 = invoke(stypy.reporting.localization.Localization(__file__, 276, 27), approx_jacobian_184872, *[x_184873, fun_184874, epsilon_184875, args_184876], **kwargs_184877)
                
                # Assigning a type to the variable 'stypy_return_type' (line 276)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'stypy_return_type', approx_jacobian_call_result_184878)
                
                # ################# End of 'cjac(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'cjac' in the type store
                # Getting the type of 'stypy_return_type' (line 275)
                stypy_return_type_184879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_184879)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'cjac'
                return stypy_return_type_184879

            # Assigning a type to the variable 'cjac' (line 275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'cjac', cjac)
            # Getting the type of 'cjac' (line 277)
            cjac_184880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 23), 'cjac')
            # Assigning a type to the variable 'stypy_return_type' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'stypy_return_type', cjac_184880)
            
            # ################# End of 'cjac_factory(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cjac_factory' in the type store
            # Getting the type of 'stypy_return_type' (line 274)
            stypy_return_type_184881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_184881)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cjac_factory'
            return stypy_return_type_184881

        # Assigning a type to the variable 'cjac_factory' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'cjac_factory', cjac_factory)
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to cjac_factory(...): (line 278)
        # Processing the call arguments (line 278)
        
        # Obtaining the type of the subscript
        str_184883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 36), 'str', 'fun')
        # Getting the type of 'con' (line 278)
        con_184884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 32), 'con', False)
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___184885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 32), con_184884, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_184886 = invoke(stypy.reporting.localization.Localization(__file__, 278, 32), getitem___184885, str_184883)
        
        # Processing the call keyword arguments (line 278)
        kwargs_184887 = {}
        # Getting the type of 'cjac_factory' (line 278)
        cjac_factory_184882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'cjac_factory', False)
        # Calling cjac_factory(args, kwargs) (line 278)
        cjac_factory_call_result_184888 = invoke(stypy.reporting.localization.Localization(__file__, 278, 19), cjac_factory_184882, *[subscript_call_result_184886], **kwargs_184887)
        
        # Assigning a type to the variable 'cjac' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'cjac', cjac_factory_call_result_184888)

        if more_types_in_union_184871:
            # SSA join for if statement (line 271)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'cons' (line 281)
    cons_184889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'cons')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ctype' (line 281)
    ctype_184890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 13), 'ctype')
    # Getting the type of 'cons' (line 281)
    cons_184891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'cons')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___184892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), cons_184891, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_184893 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), getitem___184892, ctype_184890)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 281)
    tuple_184894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 281)
    # Adding element type (line 281)
    
    # Obtaining an instance of the builtin type 'dict' (line 281)
    dict_184895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 24), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 281)
    # Adding element type (key, value) (line 281)
    str_184896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 25), 'str', 'fun')
    
    # Obtaining the type of the subscript
    str_184897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 36), 'str', 'fun')
    # Getting the type of 'con' (line 281)
    con_184898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 32), 'con')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___184899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 32), con_184898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_184900 = invoke(stypy.reporting.localization.Localization(__file__, 281, 32), getitem___184899, str_184897)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 24), dict_184895, (str_184896, subscript_call_result_184900))
    # Adding element type (key, value) (line 281)
    str_184901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 25), 'str', 'jac')
    # Getting the type of 'cjac' (line 282)
    cjac_184902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 32), 'cjac')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 24), dict_184895, (str_184901, cjac_184902))
    # Adding element type (key, value) (line 281)
    str_184903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 25), 'str', 'args')
    
    # Call to get(...): (line 283)
    # Processing the call arguments (line 283)
    str_184906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 41), 'str', 'args')
    
    # Obtaining an instance of the builtin type 'tuple' (line 283)
    tuple_184907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 283)
    
    # Processing the call keyword arguments (line 283)
    kwargs_184908 = {}
    # Getting the type of 'con' (line 283)
    con_184904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 33), 'con', False)
    # Obtaining the member 'get' of a type (line 283)
    get_184905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 33), con_184904, 'get')
    # Calling get(args, kwargs) (line 283)
    get_call_result_184909 = invoke(stypy.reporting.localization.Localization(__file__, 283, 33), get_184905, *[str_184906, tuple_184907], **kwargs_184908)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 24), dict_184895, (str_184903, get_call_result_184909))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 24), tuple_184894, dict_184895)
    
    # Applying the binary operator '+=' (line 281)
    result_iadd_184910 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 8), '+=', subscript_call_result_184893, tuple_184894)
    # Getting the type of 'cons' (line 281)
    cons_184911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'cons')
    # Getting the type of 'ctype' (line 281)
    ctype_184912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 13), 'ctype')
    # Storing an element on a container (line 281)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 8), cons_184911, (ctype_184912, result_iadd_184910))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 285):
    
    # Assigning a Dict to a Name (line 285):
    
    # Obtaining an instance of the builtin type 'dict' (line 285)
    dict_184913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 285)
    # Adding element type (key, value) (line 285)
    int_184914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 18), 'int')
    str_184915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 22), 'str', 'Gradient evaluation required (g & a)')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), dict_184913, (int_184914, str_184915))
    # Adding element type (key, value) (line 285)
    int_184916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 19), 'int')
    str_184917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 22), 'str', 'Optimization terminated successfully.')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), dict_184913, (int_184916, str_184917))
    # Adding element type (key, value) (line 285)
    int_184918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 19), 'int')
    str_184919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 22), 'str', 'Function evaluation required (f & c)')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), dict_184913, (int_184918, str_184919))
    # Adding element type (key, value) (line 285)
    int_184920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 19), 'int')
    str_184921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 22), 'str', 'More equality constraints than independent variables')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), dict_184913, (int_184920, str_184921))
    # Adding element type (key, value) (line 285)
    int_184922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 19), 'int')
    str_184923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 22), 'str', 'More than 3*n iterations in LSQ subproblem')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), dict_184913, (int_184922, str_184923))
    # Adding element type (key, value) (line 285)
    int_184924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 19), 'int')
    str_184925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 22), 'str', 'Inequality constraints incompatible')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), dict_184913, (int_184924, str_184925))
    # Adding element type (key, value) (line 285)
    int_184926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 19), 'int')
    str_184927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 22), 'str', 'Singular matrix E in LSQ subproblem')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), dict_184913, (int_184926, str_184927))
    # Adding element type (key, value) (line 285)
    int_184928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 19), 'int')
    str_184929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 22), 'str', 'Singular matrix C in LSQ subproblem')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), dict_184913, (int_184928, str_184929))
    # Adding element type (key, value) (line 285)
    int_184930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 19), 'int')
    str_184931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 22), 'str', 'Rank-deficient equality constraint subproblem HFTI')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), dict_184913, (int_184930, str_184931))
    # Adding element type (key, value) (line 285)
    int_184932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 19), 'int')
    str_184933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 22), 'str', 'Positive directional derivative for linesearch')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), dict_184913, (int_184932, str_184933))
    # Adding element type (key, value) (line 285)
    int_184934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 19), 'int')
    str_184935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 22), 'str', 'Iteration limit exceeded')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), dict_184913, (int_184934, str_184935))
    
    # Assigning a type to the variable 'exit_modes' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'exit_modes', dict_184913)
    
    # Assigning a Call to a Tuple (line 298):
    
    # Assigning a Subscript to a Name (line 298):
    
    # Obtaining the type of the subscript
    int_184936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 4), 'int')
    
    # Call to wrap_function(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'func' (line 298)
    func_184938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 32), 'func', False)
    # Getting the type of 'args' (line 298)
    args_184939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 38), 'args', False)
    # Processing the call keyword arguments (line 298)
    kwargs_184940 = {}
    # Getting the type of 'wrap_function' (line 298)
    wrap_function_184937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 18), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 298)
    wrap_function_call_result_184941 = invoke(stypy.reporting.localization.Localization(__file__, 298, 18), wrap_function_184937, *[func_184938, args_184939], **kwargs_184940)
    
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___184942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 4), wrap_function_call_result_184941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_184943 = invoke(stypy.reporting.localization.Localization(__file__, 298, 4), getitem___184942, int_184936)
    
    # Assigning a type to the variable 'tuple_var_assignment_184540' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'tuple_var_assignment_184540', subscript_call_result_184943)
    
    # Assigning a Subscript to a Name (line 298):
    
    # Obtaining the type of the subscript
    int_184944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 4), 'int')
    
    # Call to wrap_function(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'func' (line 298)
    func_184946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 32), 'func', False)
    # Getting the type of 'args' (line 298)
    args_184947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 38), 'args', False)
    # Processing the call keyword arguments (line 298)
    kwargs_184948 = {}
    # Getting the type of 'wrap_function' (line 298)
    wrap_function_184945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 18), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 298)
    wrap_function_call_result_184949 = invoke(stypy.reporting.localization.Localization(__file__, 298, 18), wrap_function_184945, *[func_184946, args_184947], **kwargs_184948)
    
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___184950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 4), wrap_function_call_result_184949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_184951 = invoke(stypy.reporting.localization.Localization(__file__, 298, 4), getitem___184950, int_184944)
    
    # Assigning a type to the variable 'tuple_var_assignment_184541' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'tuple_var_assignment_184541', subscript_call_result_184951)
    
    # Assigning a Name to a Name (line 298):
    # Getting the type of 'tuple_var_assignment_184540' (line 298)
    tuple_var_assignment_184540_184952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'tuple_var_assignment_184540')
    # Assigning a type to the variable 'feval' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'feval', tuple_var_assignment_184540_184952)
    
    # Assigning a Name to a Name (line 298):
    # Getting the type of 'tuple_var_assignment_184541' (line 298)
    tuple_var_assignment_184541_184953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'tuple_var_assignment_184541')
    # Assigning a type to the variable 'func' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 'func', tuple_var_assignment_184541_184953)
    
    # Getting the type of 'fprime' (line 301)
    fprime_184954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 7), 'fprime')
    # Testing the type of an if condition (line 301)
    if_condition_184955 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 4), fprime_184954)
    # Assigning a type to the variable 'if_condition_184955' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'if_condition_184955', if_condition_184955)
    # SSA begins for if statement (line 301)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 302):
    
    # Assigning a Subscript to a Name (line 302):
    
    # Obtaining the type of the subscript
    int_184956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 8), 'int')
    
    # Call to wrap_function(...): (line 302)
    # Processing the call arguments (line 302)
    # Getting the type of 'fprime' (line 302)
    fprime_184958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 38), 'fprime', False)
    # Getting the type of 'args' (line 302)
    args_184959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 46), 'args', False)
    # Processing the call keyword arguments (line 302)
    kwargs_184960 = {}
    # Getting the type of 'wrap_function' (line 302)
    wrap_function_184957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 24), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 302)
    wrap_function_call_result_184961 = invoke(stypy.reporting.localization.Localization(__file__, 302, 24), wrap_function_184957, *[fprime_184958, args_184959], **kwargs_184960)
    
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___184962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), wrap_function_call_result_184961, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_184963 = invoke(stypy.reporting.localization.Localization(__file__, 302, 8), getitem___184962, int_184956)
    
    # Assigning a type to the variable 'tuple_var_assignment_184542' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'tuple_var_assignment_184542', subscript_call_result_184963)
    
    # Assigning a Subscript to a Name (line 302):
    
    # Obtaining the type of the subscript
    int_184964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 8), 'int')
    
    # Call to wrap_function(...): (line 302)
    # Processing the call arguments (line 302)
    # Getting the type of 'fprime' (line 302)
    fprime_184966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 38), 'fprime', False)
    # Getting the type of 'args' (line 302)
    args_184967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 46), 'args', False)
    # Processing the call keyword arguments (line 302)
    kwargs_184968 = {}
    # Getting the type of 'wrap_function' (line 302)
    wrap_function_184965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 24), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 302)
    wrap_function_call_result_184969 = invoke(stypy.reporting.localization.Localization(__file__, 302, 24), wrap_function_184965, *[fprime_184966, args_184967], **kwargs_184968)
    
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___184970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), wrap_function_call_result_184969, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_184971 = invoke(stypy.reporting.localization.Localization(__file__, 302, 8), getitem___184970, int_184964)
    
    # Assigning a type to the variable 'tuple_var_assignment_184543' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'tuple_var_assignment_184543', subscript_call_result_184971)
    
    # Assigning a Name to a Name (line 302):
    # Getting the type of 'tuple_var_assignment_184542' (line 302)
    tuple_var_assignment_184542_184972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'tuple_var_assignment_184542')
    # Assigning a type to the variable 'geval' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'geval', tuple_var_assignment_184542_184972)
    
    # Assigning a Name to a Name (line 302):
    # Getting the type of 'tuple_var_assignment_184543' (line 302)
    tuple_var_assignment_184543_184973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'tuple_var_assignment_184543')
    # Assigning a type to the variable 'fprime' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'fprime', tuple_var_assignment_184543_184973)
    # SSA branch for the else part of an if statement (line 301)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 304):
    
    # Assigning a Subscript to a Name (line 304):
    
    # Obtaining the type of the subscript
    int_184974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 8), 'int')
    
    # Call to wrap_function(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'approx_jacobian' (line 304)
    approx_jacobian_184976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 38), 'approx_jacobian', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 304)
    tuple_184977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 304)
    # Adding element type (line 304)
    # Getting the type of 'func' (line 304)
    func_184978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 56), 'func', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 56), tuple_184977, func_184978)
    # Adding element type (line 304)
    # Getting the type of 'epsilon' (line 304)
    epsilon_184979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 62), 'epsilon', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 56), tuple_184977, epsilon_184979)
    
    # Processing the call keyword arguments (line 304)
    kwargs_184980 = {}
    # Getting the type of 'wrap_function' (line 304)
    wrap_function_184975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 24), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 304)
    wrap_function_call_result_184981 = invoke(stypy.reporting.localization.Localization(__file__, 304, 24), wrap_function_184975, *[approx_jacobian_184976, tuple_184977], **kwargs_184980)
    
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___184982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), wrap_function_call_result_184981, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_184983 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), getitem___184982, int_184974)
    
    # Assigning a type to the variable 'tuple_var_assignment_184544' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'tuple_var_assignment_184544', subscript_call_result_184983)
    
    # Assigning a Subscript to a Name (line 304):
    
    # Obtaining the type of the subscript
    int_184984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 8), 'int')
    
    # Call to wrap_function(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'approx_jacobian' (line 304)
    approx_jacobian_184986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 38), 'approx_jacobian', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 304)
    tuple_184987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 304)
    # Adding element type (line 304)
    # Getting the type of 'func' (line 304)
    func_184988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 56), 'func', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 56), tuple_184987, func_184988)
    # Adding element type (line 304)
    # Getting the type of 'epsilon' (line 304)
    epsilon_184989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 62), 'epsilon', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 56), tuple_184987, epsilon_184989)
    
    # Processing the call keyword arguments (line 304)
    kwargs_184990 = {}
    # Getting the type of 'wrap_function' (line 304)
    wrap_function_184985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 24), 'wrap_function', False)
    # Calling wrap_function(args, kwargs) (line 304)
    wrap_function_call_result_184991 = invoke(stypy.reporting.localization.Localization(__file__, 304, 24), wrap_function_184985, *[approx_jacobian_184986, tuple_184987], **kwargs_184990)
    
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___184992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), wrap_function_call_result_184991, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_184993 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), getitem___184992, int_184984)
    
    # Assigning a type to the variable 'tuple_var_assignment_184545' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'tuple_var_assignment_184545', subscript_call_result_184993)
    
    # Assigning a Name to a Name (line 304):
    # Getting the type of 'tuple_var_assignment_184544' (line 304)
    tuple_var_assignment_184544_184994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'tuple_var_assignment_184544')
    # Assigning a type to the variable 'geval' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'geval', tuple_var_assignment_184544_184994)
    
    # Assigning a Name to a Name (line 304):
    # Getting the type of 'tuple_var_assignment_184545' (line 304)
    tuple_var_assignment_184545_184995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'tuple_var_assignment_184545')
    # Assigning a type to the variable 'fprime' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'fprime', tuple_var_assignment_184545_184995)
    # SSA join for if statement (line 301)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 307):
    
    # Assigning a Call to a Name (line 307):
    
    # Call to flatten(...): (line 307)
    # Processing the call keyword arguments (line 307)
    kwargs_185001 = {}
    
    # Call to asfarray(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'x0' (line 307)
    x0_184997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 17), 'x0', False)
    # Processing the call keyword arguments (line 307)
    kwargs_184998 = {}
    # Getting the type of 'asfarray' (line 307)
    asfarray_184996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'asfarray', False)
    # Calling asfarray(args, kwargs) (line 307)
    asfarray_call_result_184999 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), asfarray_184996, *[x0_184997], **kwargs_184998)
    
    # Obtaining the member 'flatten' of a type (line 307)
    flatten_185000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), asfarray_call_result_184999, 'flatten')
    # Calling flatten(args, kwargs) (line 307)
    flatten_call_result_185002 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), flatten_185000, *[], **kwargs_185001)
    
    # Assigning a type to the variable 'x' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'x', flatten_call_result_185002)
    
    # Assigning a Call to a Name (line 311):
    
    # Assigning a Call to a Name (line 311):
    
    # Call to sum(...): (line 311)
    # Processing the call arguments (line 311)
    
    # Call to map(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'len' (line 311)
    len_185005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 18), 'len', False)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    str_185020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 28), 'str', 'eq')
    # Getting the type of 'cons' (line 312)
    cons_185021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 23), 'cons', False)
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___185022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 23), cons_185021, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_185023 = invoke(stypy.reporting.localization.Localization(__file__, 312, 23), getitem___185022, str_185020)
    
    comprehension_185024 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 24), subscript_call_result_185023)
    # Assigning a type to the variable 'c' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 24), 'c', comprehension_185024)
    
    # Call to atleast_1d(...): (line 311)
    # Processing the call arguments (line 311)
    
    # Call to (...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'x' (line 311)
    x_185011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 44), 'x', False)
    
    # Obtaining the type of the subscript
    str_185012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 50), 'str', 'args')
    # Getting the type of 'c' (line 311)
    c_185013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 48), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___185014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 48), c_185013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_185015 = invoke(stypy.reporting.localization.Localization(__file__, 311, 48), getitem___185014, str_185012)
    
    # Processing the call keyword arguments (line 311)
    kwargs_185016 = {}
    
    # Obtaining the type of the subscript
    str_185007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 37), 'str', 'fun')
    # Getting the type of 'c' (line 311)
    c_185008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 35), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___185009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 35), c_185008, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_185010 = invoke(stypy.reporting.localization.Localization(__file__, 311, 35), getitem___185009, str_185007)
    
    # Calling (args, kwargs) (line 311)
    _call_result_185017 = invoke(stypy.reporting.localization.Localization(__file__, 311, 35), subscript_call_result_185010, *[x_185011, subscript_call_result_185015], **kwargs_185016)
    
    # Processing the call keyword arguments (line 311)
    kwargs_185018 = {}
    # Getting the type of 'atleast_1d' (line 311)
    atleast_1d_185006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 24), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 311)
    atleast_1d_call_result_185019 = invoke(stypy.reporting.localization.Localization(__file__, 311, 24), atleast_1d_185006, *[_call_result_185017], **kwargs_185018)
    
    list_185025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 24), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 24), list_185025, atleast_1d_call_result_185019)
    # Processing the call keyword arguments (line 311)
    kwargs_185026 = {}
    # Getting the type of 'map' (line 311)
    map_185004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 14), 'map', False)
    # Calling map(args, kwargs) (line 311)
    map_call_result_185027 = invoke(stypy.reporting.localization.Localization(__file__, 311, 14), map_185004, *[len_185005, list_185025], **kwargs_185026)
    
    # Processing the call keyword arguments (line 311)
    kwargs_185028 = {}
    # Getting the type of 'sum' (line 311)
    sum_185003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 10), 'sum', False)
    # Calling sum(args, kwargs) (line 311)
    sum_call_result_185029 = invoke(stypy.reporting.localization.Localization(__file__, 311, 10), sum_185003, *[map_call_result_185027], **kwargs_185028)
    
    # Assigning a type to the variable 'meq' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'meq', sum_call_result_185029)
    
    # Assigning a Call to a Name (line 313):
    
    # Assigning a Call to a Name (line 313):
    
    # Call to sum(...): (line 313)
    # Processing the call arguments (line 313)
    
    # Call to map(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'len' (line 313)
    len_185032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'len', False)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    str_185047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 29), 'str', 'ineq')
    # Getting the type of 'cons' (line 314)
    cons_185048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 24), 'cons', False)
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___185049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 24), cons_185048, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_185050 = invoke(stypy.reporting.localization.Localization(__file__, 314, 24), getitem___185049, str_185047)
    
    comprehension_185051 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 25), subscript_call_result_185050)
    # Assigning a type to the variable 'c' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 25), 'c', comprehension_185051)
    
    # Call to atleast_1d(...): (line 313)
    # Processing the call arguments (line 313)
    
    # Call to (...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'x' (line 313)
    x_185038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 45), 'x', False)
    
    # Obtaining the type of the subscript
    str_185039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 51), 'str', 'args')
    # Getting the type of 'c' (line 313)
    c_185040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 49), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___185041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 49), c_185040, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_185042 = invoke(stypy.reporting.localization.Localization(__file__, 313, 49), getitem___185041, str_185039)
    
    # Processing the call keyword arguments (line 313)
    kwargs_185043 = {}
    
    # Obtaining the type of the subscript
    str_185034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 38), 'str', 'fun')
    # Getting the type of 'c' (line 313)
    c_185035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 36), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___185036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 36), c_185035, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_185037 = invoke(stypy.reporting.localization.Localization(__file__, 313, 36), getitem___185036, str_185034)
    
    # Calling (args, kwargs) (line 313)
    _call_result_185044 = invoke(stypy.reporting.localization.Localization(__file__, 313, 36), subscript_call_result_185037, *[x_185038, subscript_call_result_185042], **kwargs_185043)
    
    # Processing the call keyword arguments (line 313)
    kwargs_185045 = {}
    # Getting the type of 'atleast_1d' (line 313)
    atleast_1d_185033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 25), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 313)
    atleast_1d_call_result_185046 = invoke(stypy.reporting.localization.Localization(__file__, 313, 25), atleast_1d_185033, *[_call_result_185044], **kwargs_185045)
    
    list_185052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 25), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 25), list_185052, atleast_1d_call_result_185046)
    # Processing the call keyword arguments (line 313)
    kwargs_185053 = {}
    # Getting the type of 'map' (line 313)
    map_185031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'map', False)
    # Calling map(args, kwargs) (line 313)
    map_call_result_185054 = invoke(stypy.reporting.localization.Localization(__file__, 313, 15), map_185031, *[len_185032, list_185052], **kwargs_185053)
    
    # Processing the call keyword arguments (line 313)
    kwargs_185055 = {}
    # Getting the type of 'sum' (line 313)
    sum_185030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 11), 'sum', False)
    # Calling sum(args, kwargs) (line 313)
    sum_call_result_185056 = invoke(stypy.reporting.localization.Localization(__file__, 313, 11), sum_185030, *[map_call_result_185054], **kwargs_185055)
    
    # Assigning a type to the variable 'mieq' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'mieq', sum_call_result_185056)
    
    # Assigning a BinOp to a Name (line 316):
    
    # Assigning a BinOp to a Name (line 316):
    # Getting the type of 'meq' (line 316)
    meq_185057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'meq')
    # Getting the type of 'mieq' (line 316)
    mieq_185058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 14), 'mieq')
    # Applying the binary operator '+' (line 316)
    result_add_185059 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 8), '+', meq_185057, mieq_185058)
    
    # Assigning a type to the variable 'm' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'm', result_add_185059)
    
    # Assigning a Call to a Name (line 318):
    
    # Assigning a Call to a Name (line 318):
    
    # Call to max(...): (line 318)
    # Processing the call keyword arguments (line 318)
    kwargs_185067 = {}
    
    # Call to array(...): (line 318)
    # Processing the call arguments (line 318)
    
    # Obtaining an instance of the builtin type 'list' (line 318)
    list_185061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 318)
    # Adding element type (line 318)
    int_185062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 15), list_185061, int_185062)
    # Adding element type (line 318)
    # Getting the type of 'm' (line 318)
    m_185063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 19), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 15), list_185061, m_185063)
    
    # Processing the call keyword arguments (line 318)
    kwargs_185064 = {}
    # Getting the type of 'array' (line 318)
    array_185060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 9), 'array', False)
    # Calling array(args, kwargs) (line 318)
    array_call_result_185065 = invoke(stypy.reporting.localization.Localization(__file__, 318, 9), array_185060, *[list_185061], **kwargs_185064)
    
    # Obtaining the member 'max' of a type (line 318)
    max_185066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 9), array_call_result_185065, 'max')
    # Calling max(args, kwargs) (line 318)
    max_call_result_185068 = invoke(stypy.reporting.localization.Localization(__file__, 318, 9), max_185066, *[], **kwargs_185067)
    
    # Assigning a type to the variable 'la' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'la', max_call_result_185068)
    
    # Assigning a Call to a Name (line 320):
    
    # Assigning a Call to a Name (line 320):
    
    # Call to len(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'x' (line 320)
    x_185070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'x', False)
    # Processing the call keyword arguments (line 320)
    kwargs_185071 = {}
    # Getting the type of 'len' (line 320)
    len_185069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'len', False)
    # Calling len(args, kwargs) (line 320)
    len_call_result_185072 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), len_185069, *[x_185070], **kwargs_185071)
    
    # Assigning a type to the variable 'n' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'n', len_call_result_185072)
    
    # Assigning a BinOp to a Name (line 323):
    
    # Assigning a BinOp to a Name (line 323):
    # Getting the type of 'n' (line 323)
    n_185073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 9), 'n')
    int_185074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 13), 'int')
    # Applying the binary operator '+' (line 323)
    result_add_185075 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 9), '+', n_185073, int_185074)
    
    # Assigning a type to the variable 'n1' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'n1', result_add_185075)
    
    # Assigning a BinOp to a Name (line 324):
    
    # Assigning a BinOp to a Name (line 324):
    # Getting the type of 'm' (line 324)
    m_185076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'm')
    # Getting the type of 'meq' (line 324)
    meq_185077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'meq')
    # Applying the binary operator '-' (line 324)
    result_sub_185078 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 12), '-', m_185076, meq_185077)
    
    # Getting the type of 'n1' (line 324)
    n1_185079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 22), 'n1')
    # Applying the binary operator '+' (line 324)
    result_add_185080 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 20), '+', result_sub_185078, n1_185079)
    
    # Getting the type of 'n1' (line 324)
    n1_185081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 27), 'n1')
    # Applying the binary operator '+' (line 324)
    result_add_185082 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 25), '+', result_add_185080, n1_185081)
    
    # Assigning a type to the variable 'mineq' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'mineq', result_add_185082)
    
    # Assigning a BinOp to a Name (line 325):
    
    # Assigning a BinOp to a Name (line 325):
    int_185083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 13), 'int')
    # Getting the type of 'n1' (line 325)
    n1_185084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'n1')
    # Applying the binary operator '*' (line 325)
    result_mul_185085 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 13), '*', int_185083, n1_185084)
    
    # Getting the type of 'm' (line 325)
    m_185086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 18), 'm')
    # Applying the binary operator '+' (line 325)
    result_add_185087 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 13), '+', result_mul_185085, m_185086)
    
    # Getting the type of 'n1' (line 325)
    n1_185088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 22), 'n1')
    int_185089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 25), 'int')
    # Applying the binary operator '+' (line 325)
    result_add_185090 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 22), '+', n1_185088, int_185089)
    
    # Applying the binary operator '*' (line 325)
    result_mul_185091 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 12), '*', result_add_185087, result_add_185090)
    
    # Getting the type of 'n1' (line 325)
    n1_185092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 29), 'n1')
    # Getting the type of 'meq' (line 325)
    meq_185093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 32), 'meq')
    # Applying the binary operator '-' (line 325)
    result_sub_185094 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 29), '-', n1_185092, meq_185093)
    
    int_185095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 36), 'int')
    # Applying the binary operator '+' (line 325)
    result_add_185096 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 35), '+', result_sub_185094, int_185095)
    
    # Getting the type of 'mineq' (line 325)
    mineq_185097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 40), 'mineq')
    int_185098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 46), 'int')
    # Applying the binary operator '+' (line 325)
    result_add_185099 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 40), '+', mineq_185097, int_185098)
    
    # Applying the binary operator '*' (line 325)
    result_mul_185100 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 28), '*', result_add_185096, result_add_185099)
    
    # Applying the binary operator '+' (line 325)
    result_add_185101 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 12), '+', result_mul_185091, result_mul_185100)
    
    int_185102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 51), 'int')
    # Getting the type of 'mineq' (line 325)
    mineq_185103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 53), 'mineq')
    # Applying the binary operator '*' (line 325)
    result_mul_185104 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 51), '*', int_185102, mineq_185103)
    
    # Applying the binary operator '+' (line 325)
    result_add_185105 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 49), '+', result_add_185101, result_mul_185104)
    
    # Getting the type of 'n1' (line 325)
    n1_185106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 60), 'n1')
    # Getting the type of 'mineq' (line 325)
    mineq_185107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 63), 'mineq')
    # Applying the binary operator '+' (line 325)
    result_add_185108 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 60), '+', n1_185106, mineq_185107)
    
    # Getting the type of 'n1' (line 325)
    n1_185109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 71), 'n1')
    # Getting the type of 'meq' (line 325)
    meq_185110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 74), 'meq')
    # Applying the binary operator '-' (line 325)
    result_sub_185111 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 71), '-', n1_185109, meq_185110)
    
    # Applying the binary operator '*' (line 325)
    result_mul_185112 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 59), '*', result_add_185108, result_sub_185111)
    
    # Applying the binary operator '+' (line 325)
    result_add_185113 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 58), '+', result_add_185105, result_mul_185112)
    
    int_185114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 14), 'int')
    # Getting the type of 'meq' (line 326)
    meq_185115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'meq')
    # Applying the binary operator '*' (line 326)
    result_mul_185116 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 14), '*', int_185114, meq_185115)
    
    # Applying the binary operator '+' (line 326)
    result_add_185117 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 12), '+', result_add_185113, result_mul_185116)
    
    # Getting the type of 'n1' (line 326)
    n1_185118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 22), 'n1')
    # Applying the binary operator '+' (line 326)
    result_add_185119 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 20), '+', result_add_185117, n1_185118)
    
    # Getting the type of 'n' (line 326)
    n_185120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 29), 'n')
    int_185121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 31), 'int')
    # Applying the binary operator '+' (line 326)
    result_add_185122 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 29), '+', n_185120, int_185121)
    
    # Getting the type of 'n' (line 326)
    n_185123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 34), 'n')
    # Applying the binary operator '*' (line 326)
    result_mul_185124 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 28), '*', result_add_185122, n_185123)
    
    int_185125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 38), 'int')
    # Applying the binary operator '//' (line 326)
    result_floordiv_185126 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 27), '//', result_mul_185124, int_185125)
    
    # Applying the binary operator '+' (line 326)
    result_add_185127 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 25), '+', result_add_185119, result_floordiv_185126)
    
    int_185128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 42), 'int')
    # Getting the type of 'm' (line 326)
    m_185129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 44), 'm')
    # Applying the binary operator '*' (line 326)
    result_mul_185130 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 42), '*', int_185128, m_185129)
    
    # Applying the binary operator '+' (line 326)
    result_add_185131 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 40), '+', result_add_185127, result_mul_185130)
    
    int_185132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 48), 'int')
    # Getting the type of 'n' (line 326)
    n_185133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 50), 'n')
    # Applying the binary operator '*' (line 326)
    result_mul_185134 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 48), '*', int_185132, n_185133)
    
    # Applying the binary operator '+' (line 326)
    result_add_185135 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 46), '+', result_add_185131, result_mul_185134)
    
    int_185136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 54), 'int')
    # Getting the type of 'n1' (line 326)
    n1_185137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 56), 'n1')
    # Applying the binary operator '*' (line 326)
    result_mul_185138 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 54), '*', int_185136, n1_185137)
    
    # Applying the binary operator '+' (line 326)
    result_add_185139 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 52), '+', result_add_185135, result_mul_185138)
    
    int_185140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 61), 'int')
    # Applying the binary operator '+' (line 326)
    result_add_185141 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 59), '+', result_add_185139, int_185140)
    
    # Assigning a type to the variable 'len_w' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'len_w', result_add_185141)
    
    # Assigning a Name to a Name (line 327):
    
    # Assigning a Name to a Name (line 327):
    # Getting the type of 'mineq' (line 327)
    mineq_185142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 13), 'mineq')
    # Assigning a type to the variable 'len_jw' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'len_jw', mineq_185142)
    
    # Assigning a Call to a Name (line 328):
    
    # Assigning a Call to a Name (line 328):
    
    # Call to zeros(...): (line 328)
    # Processing the call arguments (line 328)
    # Getting the type of 'len_w' (line 328)
    len_w_185144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 14), 'len_w', False)
    # Processing the call keyword arguments (line 328)
    kwargs_185145 = {}
    # Getting the type of 'zeros' (line 328)
    zeros_185143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'zeros', False)
    # Calling zeros(args, kwargs) (line 328)
    zeros_call_result_185146 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), zeros_185143, *[len_w_185144], **kwargs_185145)
    
    # Assigning a type to the variable 'w' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'w', zeros_call_result_185146)
    
    # Assigning a Call to a Name (line 329):
    
    # Assigning a Call to a Name (line 329):
    
    # Call to zeros(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'len_jw' (line 329)
    len_jw_185148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'len_jw', False)
    # Processing the call keyword arguments (line 329)
    kwargs_185149 = {}
    # Getting the type of 'zeros' (line 329)
    zeros_185147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 9), 'zeros', False)
    # Calling zeros(args, kwargs) (line 329)
    zeros_call_result_185150 = invoke(stypy.reporting.localization.Localization(__file__, 329, 9), zeros_185147, *[len_jw_185148], **kwargs_185149)
    
    # Assigning a type to the variable 'jw' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'jw', zeros_call_result_185150)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'bounds' (line 332)
    bounds_185151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 7), 'bounds')
    # Getting the type of 'None' (line 332)
    None_185152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 17), 'None')
    # Applying the binary operator 'is' (line 332)
    result_is__185153 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 7), 'is', bounds_185151, None_185152)
    
    
    
    # Call to len(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'bounds' (line 332)
    bounds_185155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 29), 'bounds', False)
    # Processing the call keyword arguments (line 332)
    kwargs_185156 = {}
    # Getting the type of 'len' (line 332)
    len_185154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 25), 'len', False)
    # Calling len(args, kwargs) (line 332)
    len_call_result_185157 = invoke(stypy.reporting.localization.Localization(__file__, 332, 25), len_185154, *[bounds_185155], **kwargs_185156)
    
    int_185158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 40), 'int')
    # Applying the binary operator '==' (line 332)
    result_eq_185159 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 25), '==', len_call_result_185157, int_185158)
    
    # Applying the binary operator 'or' (line 332)
    result_or_keyword_185160 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 7), 'or', result_is__185153, result_eq_185159)
    
    # Testing the type of an if condition (line 332)
    if_condition_185161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 4), result_or_keyword_185160)
    # Assigning a type to the variable 'if_condition_185161' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'if_condition_185161', if_condition_185161)
    # SSA begins for if statement (line 332)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 333):
    
    # Assigning a Call to a Name (line 333):
    
    # Call to empty(...): (line 333)
    # Processing the call arguments (line 333)
    # Getting the type of 'n' (line 333)
    n_185164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 22), 'n', False)
    # Processing the call keyword arguments (line 333)
    # Getting the type of 'float' (line 333)
    float_185165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 31), 'float', False)
    keyword_185166 = float_185165
    kwargs_185167 = {'dtype': keyword_185166}
    # Getting the type of 'np' (line 333)
    np_185162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 13), 'np', False)
    # Obtaining the member 'empty' of a type (line 333)
    empty_185163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 13), np_185162, 'empty')
    # Calling empty(args, kwargs) (line 333)
    empty_call_result_185168 = invoke(stypy.reporting.localization.Localization(__file__, 333, 13), empty_185163, *[n_185164], **kwargs_185167)
    
    # Assigning a type to the variable 'xl' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'xl', empty_call_result_185168)
    
    # Assigning a Call to a Name (line 334):
    
    # Assigning a Call to a Name (line 334):
    
    # Call to empty(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'n' (line 334)
    n_185171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 22), 'n', False)
    # Processing the call keyword arguments (line 334)
    # Getting the type of 'float' (line 334)
    float_185172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 31), 'float', False)
    keyword_185173 = float_185172
    kwargs_185174 = {'dtype': keyword_185173}
    # Getting the type of 'np' (line 334)
    np_185169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 13), 'np', False)
    # Obtaining the member 'empty' of a type (line 334)
    empty_185170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 13), np_185169, 'empty')
    # Calling empty(args, kwargs) (line 334)
    empty_call_result_185175 = invoke(stypy.reporting.localization.Localization(__file__, 334, 13), empty_185170, *[n_185171], **kwargs_185174)
    
    # Assigning a type to the variable 'xu' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'xu', empty_call_result_185175)
    
    # Call to fill(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'np' (line 335)
    np_185178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'np', False)
    # Obtaining the member 'nan' of a type (line 335)
    nan_185179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 16), np_185178, 'nan')
    # Processing the call keyword arguments (line 335)
    kwargs_185180 = {}
    # Getting the type of 'xl' (line 335)
    xl_185176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'xl', False)
    # Obtaining the member 'fill' of a type (line 335)
    fill_185177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), xl_185176, 'fill')
    # Calling fill(args, kwargs) (line 335)
    fill_call_result_185181 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), fill_185177, *[nan_185179], **kwargs_185180)
    
    
    # Call to fill(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'np' (line 336)
    np_185184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'np', False)
    # Obtaining the member 'nan' of a type (line 336)
    nan_185185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 16), np_185184, 'nan')
    # Processing the call keyword arguments (line 336)
    kwargs_185186 = {}
    # Getting the type of 'xu' (line 336)
    xu_185182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'xu', False)
    # Obtaining the member 'fill' of a type (line 336)
    fill_185183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), xu_185182, 'fill')
    # Calling fill(args, kwargs) (line 336)
    fill_call_result_185187 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), fill_185183, *[nan_185185], **kwargs_185186)
    
    # SSA branch for the else part of an if statement (line 332)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 338):
    
    # Assigning a Call to a Name (line 338):
    
    # Call to array(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'bounds' (line 338)
    bounds_185189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 21), 'bounds', False)
    # Getting the type of 'float' (line 338)
    float_185190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 29), 'float', False)
    # Processing the call keyword arguments (line 338)
    kwargs_185191 = {}
    # Getting the type of 'array' (line 338)
    array_185188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 15), 'array', False)
    # Calling array(args, kwargs) (line 338)
    array_call_result_185192 = invoke(stypy.reporting.localization.Localization(__file__, 338, 15), array_185188, *[bounds_185189, float_185190], **kwargs_185191)
    
    # Assigning a type to the variable 'bnds' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'bnds', array_call_result_185192)
    
    
    
    # Obtaining the type of the subscript
    int_185193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 22), 'int')
    # Getting the type of 'bnds' (line 339)
    bnds_185194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 11), 'bnds')
    # Obtaining the member 'shape' of a type (line 339)
    shape_185195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 11), bnds_185194, 'shape')
    # Obtaining the member '__getitem__' of a type (line 339)
    getitem___185196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 11), shape_185195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 339)
    subscript_call_result_185197 = invoke(stypy.reporting.localization.Localization(__file__, 339, 11), getitem___185196, int_185193)
    
    # Getting the type of 'n' (line 339)
    n_185198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 28), 'n')
    # Applying the binary operator '!=' (line 339)
    result_ne_185199 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 11), '!=', subscript_call_result_185197, n_185198)
    
    # Testing the type of an if condition (line 339)
    if_condition_185200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 339, 8), result_ne_185199)
    # Assigning a type to the variable 'if_condition_185200' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'if_condition_185200', if_condition_185200)
    # SSA begins for if statement (line 339)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to IndexError(...): (line 340)
    # Processing the call arguments (line 340)
    str_185202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 29), 'str', 'SLSQP Error: the length of bounds is not compatible with that of x0.')
    # Processing the call keyword arguments (line 340)
    kwargs_185203 = {}
    # Getting the type of 'IndexError' (line 340)
    IndexError_185201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 18), 'IndexError', False)
    # Calling IndexError(args, kwargs) (line 340)
    IndexError_call_result_185204 = invoke(stypy.reporting.localization.Localization(__file__, 340, 18), IndexError_185201, *[str_185202], **kwargs_185203)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 340, 12), IndexError_call_result_185204, 'raise parameter', BaseException)
    # SSA join for if statement (line 339)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to errstate(...): (line 343)
    # Processing the call keyword arguments (line 343)
    str_185207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 33), 'str', 'ignore')
    keyword_185208 = str_185207
    kwargs_185209 = {'invalid': keyword_185208}
    # Getting the type of 'np' (line 343)
    np_185205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 13), 'np', False)
    # Obtaining the member 'errstate' of a type (line 343)
    errstate_185206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 13), np_185205, 'errstate')
    # Calling errstate(args, kwargs) (line 343)
    errstate_call_result_185210 = invoke(stypy.reporting.localization.Localization(__file__, 343, 13), errstate_185206, *[], **kwargs_185209)
    
    with_185211 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 343, 13), errstate_call_result_185210, 'with parameter', '__enter__', '__exit__')

    if with_185211:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 343)
        enter___185212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 13), errstate_call_result_185210, '__enter__')
        with_enter_185213 = invoke(stypy.reporting.localization.Localization(__file__, 343, 13), enter___185212)
        
        # Assigning a Compare to a Name (line 344):
        
        # Assigning a Compare to a Name (line 344):
        
        
        # Obtaining the type of the subscript
        slice_185214 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 344, 21), None, None, None)
        int_185215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 29), 'int')
        # Getting the type of 'bnds' (line 344)
        bnds_185216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 21), 'bnds')
        # Obtaining the member '__getitem__' of a type (line 344)
        getitem___185217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 21), bnds_185216, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 344)
        subscript_call_result_185218 = invoke(stypy.reporting.localization.Localization(__file__, 344, 21), getitem___185217, (slice_185214, int_185215))
        
        
        # Obtaining the type of the subscript
        slice_185219 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 344, 34), None, None, None)
        int_185220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 42), 'int')
        # Getting the type of 'bnds' (line 344)
        bnds_185221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 34), 'bnds')
        # Obtaining the member '__getitem__' of a type (line 344)
        getitem___185222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 34), bnds_185221, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 344)
        subscript_call_result_185223 = invoke(stypy.reporting.localization.Localization(__file__, 344, 34), getitem___185222, (slice_185219, int_185220))
        
        # Applying the binary operator '>' (line 344)
        result_gt_185224 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 21), '>', subscript_call_result_185218, subscript_call_result_185223)
        
        # Assigning a type to the variable 'bnderr' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'bnderr', result_gt_185224)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 343)
        exit___185225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 13), errstate_call_result_185210, '__exit__')
        with_exit_185226 = invoke(stypy.reporting.localization.Localization(__file__, 343, 13), exit___185225, None, None, None)

    
    
    # Call to any(...): (line 346)
    # Processing the call keyword arguments (line 346)
    kwargs_185229 = {}
    # Getting the type of 'bnderr' (line 346)
    bnderr_185227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 11), 'bnderr', False)
    # Obtaining the member 'any' of a type (line 346)
    any_185228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 11), bnderr_185227, 'any')
    # Calling any(args, kwargs) (line 346)
    any_call_result_185230 = invoke(stypy.reporting.localization.Localization(__file__, 346, 11), any_185228, *[], **kwargs_185229)
    
    # Testing the type of an if condition (line 346)
    if_condition_185231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 8), any_call_result_185230)
    # Assigning a type to the variable 'if_condition_185231' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'if_condition_185231', if_condition_185231)
    # SSA begins for if statement (line 346)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 347)
    # Processing the call arguments (line 347)
    str_185233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 29), 'str', 'SLSQP Error: lb > ub in bounds %s.')
    
    # Call to join(...): (line 348)
    # Processing the call arguments (line 348)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 348, 39, True)
    # Calculating comprehension expression
    # Getting the type of 'bnderr' (line 348)
    bnderr_185240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 55), 'bnderr', False)
    comprehension_185241 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 39), bnderr_185240)
    # Assigning a type to the variable 'b' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 39), 'b', comprehension_185241)
    
    # Call to str(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'b' (line 348)
    b_185237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 43), 'b', False)
    # Processing the call keyword arguments (line 348)
    kwargs_185238 = {}
    # Getting the type of 'str' (line 348)
    str_185236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 39), 'str', False)
    # Calling str(args, kwargs) (line 348)
    str_call_result_185239 = invoke(stypy.reporting.localization.Localization(__file__, 348, 39), str_185236, *[b_185237], **kwargs_185238)
    
    list_185242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 39), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 39), list_185242, str_call_result_185239)
    # Processing the call keyword arguments (line 348)
    kwargs_185243 = {}
    str_185234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 29), 'str', ', ')
    # Obtaining the member 'join' of a type (line 348)
    join_185235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 29), str_185234, 'join')
    # Calling join(args, kwargs) (line 348)
    join_call_result_185244 = invoke(stypy.reporting.localization.Localization(__file__, 348, 29), join_185235, *[list_185242], **kwargs_185243)
    
    # Applying the binary operator '%' (line 347)
    result_mod_185245 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 29), '%', str_185233, join_call_result_185244)
    
    # Processing the call keyword arguments (line 347)
    kwargs_185246 = {}
    # Getting the type of 'ValueError' (line 347)
    ValueError_185232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 347)
    ValueError_call_result_185247 = invoke(stypy.reporting.localization.Localization(__file__, 347, 18), ValueError_185232, *[result_mod_185245], **kwargs_185246)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 347, 12), ValueError_call_result_185247, 'raise parameter', BaseException)
    # SSA join for if statement (line 346)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 349):
    
    # Assigning a Subscript to a Name (line 349):
    
    # Obtaining the type of the subscript
    slice_185248 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 349, 17), None, None, None)
    int_185249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 25), 'int')
    # Getting the type of 'bnds' (line 349)
    bnds_185250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 17), 'bnds')
    # Obtaining the member '__getitem__' of a type (line 349)
    getitem___185251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 17), bnds_185250, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 349)
    subscript_call_result_185252 = invoke(stypy.reporting.localization.Localization(__file__, 349, 17), getitem___185251, (slice_185248, int_185249))
    
    # Assigning a type to the variable 'tuple_assignment_184546' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_assignment_184546', subscript_call_result_185252)
    
    # Assigning a Subscript to a Name (line 349):
    
    # Obtaining the type of the subscript
    slice_185253 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 349, 29), None, None, None)
    int_185254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 37), 'int')
    # Getting the type of 'bnds' (line 349)
    bnds_185255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 29), 'bnds')
    # Obtaining the member '__getitem__' of a type (line 349)
    getitem___185256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 29), bnds_185255, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 349)
    subscript_call_result_185257 = invoke(stypy.reporting.localization.Localization(__file__, 349, 29), getitem___185256, (slice_185253, int_185254))
    
    # Assigning a type to the variable 'tuple_assignment_184547' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_assignment_184547', subscript_call_result_185257)
    
    # Assigning a Name to a Name (line 349):
    # Getting the type of 'tuple_assignment_184546' (line 349)
    tuple_assignment_184546_185258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_assignment_184546')
    # Assigning a type to the variable 'xl' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'xl', tuple_assignment_184546_185258)
    
    # Assigning a Name to a Name (line 349):
    # Getting the type of 'tuple_assignment_184547' (line 349)
    tuple_assignment_184547_185259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_assignment_184547')
    # Assigning a type to the variable 'xu' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'xu', tuple_assignment_184547_185259)
    
    # Assigning a UnaryOp to a Name (line 352):
    
    # Assigning a UnaryOp to a Name (line 352):
    
    
    # Call to isfinite(...): (line 352)
    # Processing the call arguments (line 352)
    # Getting the type of 'bnds' (line 352)
    bnds_185261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 27), 'bnds', False)
    # Processing the call keyword arguments (line 352)
    kwargs_185262 = {}
    # Getting the type of 'isfinite' (line 352)
    isfinite_185260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 18), 'isfinite', False)
    # Calling isfinite(args, kwargs) (line 352)
    isfinite_call_result_185263 = invoke(stypy.reporting.localization.Localization(__file__, 352, 18), isfinite_185260, *[bnds_185261], **kwargs_185262)
    
    # Applying the '~' unary operator (line 352)
    result_inv_185264 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 17), '~', isfinite_call_result_185263)
    
    # Assigning a type to the variable 'infbnd' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'infbnd', result_inv_185264)
    
    # Assigning a Attribute to a Subscript (line 353):
    
    # Assigning a Attribute to a Subscript (line 353):
    # Getting the type of 'np' (line 353)
    np_185265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 27), 'np')
    # Obtaining the member 'nan' of a type (line 353)
    nan_185266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 27), np_185265, 'nan')
    # Getting the type of 'xl' (line 353)
    xl_185267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'xl')
    
    # Obtaining the type of the subscript
    slice_185268 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 353, 11), None, None, None)
    int_185269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 21), 'int')
    # Getting the type of 'infbnd' (line 353)
    infbnd_185270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 11), 'infbnd')
    # Obtaining the member '__getitem__' of a type (line 353)
    getitem___185271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 11), infbnd_185270, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 353)
    subscript_call_result_185272 = invoke(stypy.reporting.localization.Localization(__file__, 353, 11), getitem___185271, (slice_185268, int_185269))
    
    # Storing an element on a container (line 353)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 8), xl_185267, (subscript_call_result_185272, nan_185266))
    
    # Assigning a Attribute to a Subscript (line 354):
    
    # Assigning a Attribute to a Subscript (line 354):
    # Getting the type of 'np' (line 354)
    np_185273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'np')
    # Obtaining the member 'nan' of a type (line 354)
    nan_185274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 27), np_185273, 'nan')
    # Getting the type of 'xu' (line 354)
    xu_185275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'xu')
    
    # Obtaining the type of the subscript
    slice_185276 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 354, 11), None, None, None)
    int_185277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 21), 'int')
    # Getting the type of 'infbnd' (line 354)
    infbnd_185278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'infbnd')
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___185279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 11), infbnd_185278, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_185280 = invoke(stypy.reporting.localization.Localization(__file__, 354, 11), getitem___185279, (slice_185276, int_185277))
    
    # Storing an element on a container (line 354)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 8), xu_185275, (subscript_call_result_185280, nan_185274))
    # SSA join for if statement (line 332)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 358):
    
    # Assigning a Call to a Name (line 358):
    
    # Call to isfinite(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'xl' (line 358)
    xl_185283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'xl', False)
    # Processing the call keyword arguments (line 358)
    kwargs_185284 = {}
    # Getting the type of 'np' (line 358)
    np_185281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 17), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 358)
    isfinite_185282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 17), np_185281, 'isfinite')
    # Calling isfinite(args, kwargs) (line 358)
    isfinite_call_result_185285 = invoke(stypy.reporting.localization.Localization(__file__, 358, 17), isfinite_185282, *[xl_185283], **kwargs_185284)
    
    # Assigning a type to the variable 'have_bound' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'have_bound', isfinite_call_result_185285)
    
    # Assigning a Call to a Subscript (line 359):
    
    # Assigning a Call to a Subscript (line 359):
    
    # Call to clip(...): (line 359)
    # Processing the call arguments (line 359)
    
    # Obtaining the type of the subscript
    # Getting the type of 'have_bound' (line 359)
    have_bound_185288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 30), 'have_bound', False)
    # Getting the type of 'x' (line 359)
    x_185289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 28), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___185290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 28), x_185289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_185291 = invoke(stypy.reporting.localization.Localization(__file__, 359, 28), getitem___185290, have_bound_185288)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'have_bound' (line 359)
    have_bound_185292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 46), 'have_bound', False)
    # Getting the type of 'xl' (line 359)
    xl_185293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 43), 'xl', False)
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___185294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 43), xl_185293, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_185295 = invoke(stypy.reporting.localization.Localization(__file__, 359, 43), getitem___185294, have_bound_185292)
    
    # Getting the type of 'np' (line 359)
    np_185296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 59), 'np', False)
    # Obtaining the member 'inf' of a type (line 359)
    inf_185297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 59), np_185296, 'inf')
    # Processing the call keyword arguments (line 359)
    kwargs_185298 = {}
    # Getting the type of 'np' (line 359)
    np_185286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 20), 'np', False)
    # Obtaining the member 'clip' of a type (line 359)
    clip_185287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 20), np_185286, 'clip')
    # Calling clip(args, kwargs) (line 359)
    clip_call_result_185299 = invoke(stypy.reporting.localization.Localization(__file__, 359, 20), clip_185287, *[subscript_call_result_185291, subscript_call_result_185295, inf_185297], **kwargs_185298)
    
    # Getting the type of 'x' (line 359)
    x_185300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'x')
    # Getting the type of 'have_bound' (line 359)
    have_bound_185301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 6), 'have_bound')
    # Storing an element on a container (line 359)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 4), x_185300, (have_bound_185301, clip_call_result_185299))
    
    # Assigning a Call to a Name (line 360):
    
    # Assigning a Call to a Name (line 360):
    
    # Call to isfinite(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'xu' (line 360)
    xu_185304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 29), 'xu', False)
    # Processing the call keyword arguments (line 360)
    kwargs_185305 = {}
    # Getting the type of 'np' (line 360)
    np_185302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 17), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 360)
    isfinite_185303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 17), np_185302, 'isfinite')
    # Calling isfinite(args, kwargs) (line 360)
    isfinite_call_result_185306 = invoke(stypy.reporting.localization.Localization(__file__, 360, 17), isfinite_185303, *[xu_185304], **kwargs_185305)
    
    # Assigning a type to the variable 'have_bound' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'have_bound', isfinite_call_result_185306)
    
    # Assigning a Call to a Subscript (line 361):
    
    # Assigning a Call to a Subscript (line 361):
    
    # Call to clip(...): (line 361)
    # Processing the call arguments (line 361)
    
    # Obtaining the type of the subscript
    # Getting the type of 'have_bound' (line 361)
    have_bound_185309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 30), 'have_bound', False)
    # Getting the type of 'x' (line 361)
    x_185310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 28), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 361)
    getitem___185311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 28), x_185310, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 361)
    subscript_call_result_185312 = invoke(stypy.reporting.localization.Localization(__file__, 361, 28), getitem___185311, have_bound_185309)
    
    
    # Getting the type of 'np' (line 361)
    np_185313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 44), 'np', False)
    # Obtaining the member 'inf' of a type (line 361)
    inf_185314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 44), np_185313, 'inf')
    # Applying the 'usub' unary operator (line 361)
    result___neg___185315 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 43), 'usub', inf_185314)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'have_bound' (line 361)
    have_bound_185316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 55), 'have_bound', False)
    # Getting the type of 'xu' (line 361)
    xu_185317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 52), 'xu', False)
    # Obtaining the member '__getitem__' of a type (line 361)
    getitem___185318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 52), xu_185317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 361)
    subscript_call_result_185319 = invoke(stypy.reporting.localization.Localization(__file__, 361, 52), getitem___185318, have_bound_185316)
    
    # Processing the call keyword arguments (line 361)
    kwargs_185320 = {}
    # Getting the type of 'np' (line 361)
    np_185307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'np', False)
    # Obtaining the member 'clip' of a type (line 361)
    clip_185308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 20), np_185307, 'clip')
    # Calling clip(args, kwargs) (line 361)
    clip_call_result_185321 = invoke(stypy.reporting.localization.Localization(__file__, 361, 20), clip_185308, *[subscript_call_result_185312, result___neg___185315, subscript_call_result_185319], **kwargs_185320)
    
    # Getting the type of 'x' (line 361)
    x_185322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'x')
    # Getting the type of 'have_bound' (line 361)
    have_bound_185323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 6), 'have_bound')
    # Storing an element on a container (line 361)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 4), x_185322, (have_bound_185323, clip_call_result_185321))
    
    # Assigning a Call to a Name (line 364):
    
    # Assigning a Call to a Name (line 364):
    
    # Call to array(...): (line 364)
    # Processing the call arguments (line 364)
    int_185325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 17), 'int')
    # Getting the type of 'int' (line 364)
    int_185326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 20), 'int', False)
    # Processing the call keyword arguments (line 364)
    kwargs_185327 = {}
    # Getting the type of 'array' (line 364)
    array_185324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 11), 'array', False)
    # Calling array(args, kwargs) (line 364)
    array_call_result_185328 = invoke(stypy.reporting.localization.Localization(__file__, 364, 11), array_185324, *[int_185325, int_185326], **kwargs_185327)
    
    # Assigning a type to the variable 'mode' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'mode', array_call_result_185328)
    
    # Assigning a Call to a Name (line 365):
    
    # Assigning a Call to a Name (line 365):
    
    # Call to array(...): (line 365)
    # Processing the call arguments (line 365)
    # Getting the type of 'acc' (line 365)
    acc_185330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'acc', False)
    # Getting the type of 'float' (line 365)
    float_185331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 21), 'float', False)
    # Processing the call keyword arguments (line 365)
    kwargs_185332 = {}
    # Getting the type of 'array' (line 365)
    array_185329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 10), 'array', False)
    # Calling array(args, kwargs) (line 365)
    array_call_result_185333 = invoke(stypy.reporting.localization.Localization(__file__, 365, 10), array_185329, *[acc_185330, float_185331], **kwargs_185332)
    
    # Assigning a type to the variable 'acc' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'acc', array_call_result_185333)
    
    # Assigning a Call to a Name (line 366):
    
    # Assigning a Call to a Name (line 366):
    
    # Call to array(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'iter' (line 366)
    iter_185335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 20), 'iter', False)
    # Getting the type of 'int' (line 366)
    int_185336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 26), 'int', False)
    # Processing the call keyword arguments (line 366)
    kwargs_185337 = {}
    # Getting the type of 'array' (line 366)
    array_185334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 14), 'array', False)
    # Calling array(args, kwargs) (line 366)
    array_call_result_185338 = invoke(stypy.reporting.localization.Localization(__file__, 366, 14), array_185334, *[iter_185335, int_185336], **kwargs_185337)
    
    # Assigning a type to the variable 'majiter' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'majiter', array_call_result_185338)
    
    # Assigning a Num to a Name (line 367):
    
    # Assigning a Num to a Name (line 367):
    int_185339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 19), 'int')
    # Assigning a type to the variable 'majiter_prev' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'majiter_prev', int_185339)
    
    
    # Getting the type of 'iprint' (line 370)
    iprint_185340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 7), 'iprint')
    int_185341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 17), 'int')
    # Applying the binary operator '>=' (line 370)
    result_ge_185342 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 7), '>=', iprint_185340, int_185341)
    
    # Testing the type of an if condition (line 370)
    if_condition_185343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 4), result_ge_185342)
    # Assigning a type to the variable 'if_condition_185343' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'if_condition_185343', if_condition_185343)
    # SSA begins for if statement (line 370)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 371)
    # Processing the call arguments (line 371)
    str_185345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 14), 'str', '%5s %5s %16s %16s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 371)
    tuple_185346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 371)
    # Adding element type (line 371)
    str_185347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 37), 'str', 'NIT')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 37), tuple_185346, str_185347)
    # Adding element type (line 371)
    str_185348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 44), 'str', 'FC')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 37), tuple_185346, str_185348)
    # Adding element type (line 371)
    str_185349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 50), 'str', 'OBJFUN')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 37), tuple_185346, str_185349)
    # Adding element type (line 371)
    str_185350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 60), 'str', 'GNORM')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 37), tuple_185346, str_185350)
    
    # Applying the binary operator '%' (line 371)
    result_mod_185351 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 14), '%', str_185345, tuple_185346)
    
    # Processing the call keyword arguments (line 371)
    kwargs_185352 = {}
    # Getting the type of 'print' (line 371)
    print_185344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'print', False)
    # Calling print(args, kwargs) (line 371)
    print_call_result_185353 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), print_185344, *[result_mod_185351], **kwargs_185352)
    
    # SSA join for if statement (line 370)
    module_type_store = module_type_store.join_ssa_context()
    
    
    int_185354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 10), 'int')
    # Testing the type of an if condition (line 373)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 4), int_185354)
    # SSA begins for while statement (line 373)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'mode' (line 375)
    mode_185355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'mode')
    int_185356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 19), 'int')
    # Applying the binary operator '==' (line 375)
    result_eq_185357 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 11), '==', mode_185355, int_185356)
    
    
    # Getting the type of 'mode' (line 375)
    mode_185358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 24), 'mode')
    int_185359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 32), 'int')
    # Applying the binary operator '==' (line 375)
    result_eq_185360 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 24), '==', mode_185358, int_185359)
    
    # Applying the binary operator 'or' (line 375)
    result_or_keyword_185361 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 11), 'or', result_eq_185357, result_eq_185360)
    
    # Testing the type of an if condition (line 375)
    if_condition_185362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 8), result_or_keyword_185361)
    # Assigning a type to the variable 'if_condition_185362' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'if_condition_185362', if_condition_185362)
    # SSA begins for if statement (line 375)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 378):
    
    # Assigning a Call to a Name (line 378):
    
    # Call to func(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 'x' (line 378)
    x_185364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 22), 'x', False)
    # Processing the call keyword arguments (line 378)
    kwargs_185365 = {}
    # Getting the type of 'func' (line 378)
    func_185363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), 'func', False)
    # Calling func(args, kwargs) (line 378)
    func_call_result_185366 = invoke(stypy.reporting.localization.Localization(__file__, 378, 17), func_185363, *[x_185364], **kwargs_185365)
    
    # Assigning a type to the variable 'fx' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'fx', func_call_result_185366)
    
    
    # SSA begins for try-except statement (line 379)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 380):
    
    # Assigning a Call to a Name (line 380):
    
    # Call to float(...): (line 380)
    # Processing the call arguments (line 380)
    
    # Call to asarray(...): (line 380)
    # Processing the call arguments (line 380)
    # Getting the type of 'fx' (line 380)
    fx_185370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 38), 'fx', False)
    # Processing the call keyword arguments (line 380)
    kwargs_185371 = {}
    # Getting the type of 'np' (line 380)
    np_185368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 27), 'np', False)
    # Obtaining the member 'asarray' of a type (line 380)
    asarray_185369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 27), np_185368, 'asarray')
    # Calling asarray(args, kwargs) (line 380)
    asarray_call_result_185372 = invoke(stypy.reporting.localization.Localization(__file__, 380, 27), asarray_185369, *[fx_185370], **kwargs_185371)
    
    # Processing the call keyword arguments (line 380)
    kwargs_185373 = {}
    # Getting the type of 'float' (line 380)
    float_185367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), 'float', False)
    # Calling float(args, kwargs) (line 380)
    float_call_result_185374 = invoke(stypy.reporting.localization.Localization(__file__, 380, 21), float_185367, *[asarray_call_result_185372], **kwargs_185373)
    
    # Assigning a type to the variable 'fx' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'fx', float_call_result_185374)
    # SSA branch for the except part of a try statement (line 379)
    # SSA branch for the except 'Tuple' branch of a try statement (line 379)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 382)
    # Processing the call arguments (line 382)
    str_185376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 33), 'str', 'Objective function must return a scalar')
    # Processing the call keyword arguments (line 382)
    kwargs_185377 = {}
    # Getting the type of 'ValueError' (line 382)
    ValueError_185375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 382)
    ValueError_call_result_185378 = invoke(stypy.reporting.localization.Localization(__file__, 382, 22), ValueError_185375, *[str_185376], **kwargs_185377)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 382, 16), ValueError_call_result_185378, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 379)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    str_185379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 20), 'str', 'eq')
    # Getting the type of 'cons' (line 384)
    cons_185380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 15), 'cons')
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___185381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 15), cons_185380, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_185382 = invoke(stypy.reporting.localization.Localization(__file__, 384, 15), getitem___185381, str_185379)
    
    # Testing the type of an if condition (line 384)
    if_condition_185383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 12), subscript_call_result_185382)
    # Assigning a type to the variable 'if_condition_185383' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'if_condition_185383', if_condition_185383)
    # SSA begins for if statement (line 384)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 385):
    
    # Assigning a Call to a Name (line 385):
    
    # Call to concatenate(...): (line 385)
    # Processing the call arguments (line 385)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    str_185399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 52), 'str', 'eq')
    # Getting the type of 'cons' (line 386)
    cons_185400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 47), 'cons', False)
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___185401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 47), cons_185400, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_185402 = invoke(stypy.reporting.localization.Localization(__file__, 386, 47), getitem___185401, str_185399)
    
    comprehension_185403 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 36), subscript_call_result_185402)
    # Assigning a type to the variable 'con' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 36), 'con', comprehension_185403)
    
    # Call to atleast_1d(...): (line 385)
    # Processing the call arguments (line 385)
    
    # Call to (...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'x' (line 385)
    x_185390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 58), 'x', False)
    
    # Obtaining the type of the subscript
    str_185391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 66), 'str', 'args')
    # Getting the type of 'con' (line 385)
    con_185392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 62), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 385)
    getitem___185393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 62), con_185392, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 385)
    subscript_call_result_185394 = invoke(stypy.reporting.localization.Localization(__file__, 385, 62), getitem___185393, str_185391)
    
    # Processing the call keyword arguments (line 385)
    kwargs_185395 = {}
    
    # Obtaining the type of the subscript
    str_185386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 51), 'str', 'fun')
    # Getting the type of 'con' (line 385)
    con_185387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 47), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 385)
    getitem___185388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 47), con_185387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 385)
    subscript_call_result_185389 = invoke(stypy.reporting.localization.Localization(__file__, 385, 47), getitem___185388, str_185386)
    
    # Calling (args, kwargs) (line 385)
    _call_result_185396 = invoke(stypy.reporting.localization.Localization(__file__, 385, 47), subscript_call_result_185389, *[x_185390, subscript_call_result_185394], **kwargs_185395)
    
    # Processing the call keyword arguments (line 385)
    kwargs_185397 = {}
    # Getting the type of 'atleast_1d' (line 385)
    atleast_1d_185385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 36), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 385)
    atleast_1d_call_result_185398 = invoke(stypy.reporting.localization.Localization(__file__, 385, 36), atleast_1d_185385, *[_call_result_185396], **kwargs_185397)
    
    list_185404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 36), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 36), list_185404, atleast_1d_call_result_185398)
    # Processing the call keyword arguments (line 385)
    kwargs_185405 = {}
    # Getting the type of 'concatenate' (line 385)
    concatenate_185384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 23), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 385)
    concatenate_call_result_185406 = invoke(stypy.reporting.localization.Localization(__file__, 385, 23), concatenate_185384, *[list_185404], **kwargs_185405)
    
    # Assigning a type to the variable 'c_eq' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'c_eq', concatenate_call_result_185406)
    # SSA branch for the else part of an if statement (line 384)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 388):
    
    # Assigning a Call to a Name (line 388):
    
    # Call to zeros(...): (line 388)
    # Processing the call arguments (line 388)
    int_185408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 29), 'int')
    # Processing the call keyword arguments (line 388)
    kwargs_185409 = {}
    # Getting the type of 'zeros' (line 388)
    zeros_185407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 23), 'zeros', False)
    # Calling zeros(args, kwargs) (line 388)
    zeros_call_result_185410 = invoke(stypy.reporting.localization.Localization(__file__, 388, 23), zeros_185407, *[int_185408], **kwargs_185409)
    
    # Assigning a type to the variable 'c_eq' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 16), 'c_eq', zeros_call_result_185410)
    # SSA join for if statement (line 384)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    str_185411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 20), 'str', 'ineq')
    # Getting the type of 'cons' (line 389)
    cons_185412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'cons')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___185413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 15), cons_185412, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_185414 = invoke(stypy.reporting.localization.Localization(__file__, 389, 15), getitem___185413, str_185411)
    
    # Testing the type of an if condition (line 389)
    if_condition_185415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 12), subscript_call_result_185414)
    # Assigning a type to the variable 'if_condition_185415' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'if_condition_185415', if_condition_185415)
    # SSA begins for if statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 390):
    
    # Assigning a Call to a Name (line 390):
    
    # Call to concatenate(...): (line 390)
    # Processing the call arguments (line 390)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    str_185431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 53), 'str', 'ineq')
    # Getting the type of 'cons' (line 391)
    cons_185432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 48), 'cons', False)
    # Obtaining the member '__getitem__' of a type (line 391)
    getitem___185433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 48), cons_185432, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 391)
    subscript_call_result_185434 = invoke(stypy.reporting.localization.Localization(__file__, 391, 48), getitem___185433, str_185431)
    
    comprehension_185435 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 37), subscript_call_result_185434)
    # Assigning a type to the variable 'con' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 37), 'con', comprehension_185435)
    
    # Call to atleast_1d(...): (line 390)
    # Processing the call arguments (line 390)
    
    # Call to (...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'x' (line 390)
    x_185422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 59), 'x', False)
    
    # Obtaining the type of the subscript
    str_185423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 67), 'str', 'args')
    # Getting the type of 'con' (line 390)
    con_185424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 63), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 390)
    getitem___185425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 63), con_185424, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 390)
    subscript_call_result_185426 = invoke(stypy.reporting.localization.Localization(__file__, 390, 63), getitem___185425, str_185423)
    
    # Processing the call keyword arguments (line 390)
    kwargs_185427 = {}
    
    # Obtaining the type of the subscript
    str_185418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 52), 'str', 'fun')
    # Getting the type of 'con' (line 390)
    con_185419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 48), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 390)
    getitem___185420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 48), con_185419, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 390)
    subscript_call_result_185421 = invoke(stypy.reporting.localization.Localization(__file__, 390, 48), getitem___185420, str_185418)
    
    # Calling (args, kwargs) (line 390)
    _call_result_185428 = invoke(stypy.reporting.localization.Localization(__file__, 390, 48), subscript_call_result_185421, *[x_185422, subscript_call_result_185426], **kwargs_185427)
    
    # Processing the call keyword arguments (line 390)
    kwargs_185429 = {}
    # Getting the type of 'atleast_1d' (line 390)
    atleast_1d_185417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 37), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 390)
    atleast_1d_call_result_185430 = invoke(stypy.reporting.localization.Localization(__file__, 390, 37), atleast_1d_185417, *[_call_result_185428], **kwargs_185429)
    
    list_185436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 37), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 37), list_185436, atleast_1d_call_result_185430)
    # Processing the call keyword arguments (line 390)
    kwargs_185437 = {}
    # Getting the type of 'concatenate' (line 390)
    concatenate_185416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 24), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 390)
    concatenate_call_result_185438 = invoke(stypy.reporting.localization.Localization(__file__, 390, 24), concatenate_185416, *[list_185436], **kwargs_185437)
    
    # Assigning a type to the variable 'c_ieq' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'c_ieq', concatenate_call_result_185438)
    # SSA branch for the else part of an if statement (line 389)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to zeros(...): (line 393)
    # Processing the call arguments (line 393)
    int_185440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 30), 'int')
    # Processing the call keyword arguments (line 393)
    kwargs_185441 = {}
    # Getting the type of 'zeros' (line 393)
    zeros_185439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 24), 'zeros', False)
    # Calling zeros(args, kwargs) (line 393)
    zeros_call_result_185442 = invoke(stypy.reporting.localization.Localization(__file__, 393, 24), zeros_185439, *[int_185440], **kwargs_185441)
    
    # Assigning a type to the variable 'c_ieq' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 16), 'c_ieq', zeros_call_result_185442)
    # SSA join for if statement (line 389)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 396):
    
    # Assigning a Call to a Name (line 396):
    
    # Call to concatenate(...): (line 396)
    # Processing the call arguments (line 396)
    
    # Obtaining an instance of the builtin type 'tuple' (line 396)
    tuple_185444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 396)
    # Adding element type (line 396)
    # Getting the type of 'c_eq' (line 396)
    c_eq_185445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 29), 'c_eq', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 29), tuple_185444, c_eq_185445)
    # Adding element type (line 396)
    # Getting the type of 'c_ieq' (line 396)
    c_ieq_185446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 35), 'c_ieq', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 29), tuple_185444, c_ieq_185446)
    
    # Processing the call keyword arguments (line 396)
    kwargs_185447 = {}
    # Getting the type of 'concatenate' (line 396)
    concatenate_185443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 16), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 396)
    concatenate_call_result_185448 = invoke(stypy.reporting.localization.Localization(__file__, 396, 16), concatenate_185443, *[tuple_185444], **kwargs_185447)
    
    # Assigning a type to the variable 'c' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'c', concatenate_call_result_185448)
    # SSA join for if statement (line 375)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'mode' (line 398)
    mode_185449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 11), 'mode')
    int_185450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 19), 'int')
    # Applying the binary operator '==' (line 398)
    result_eq_185451 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 11), '==', mode_185449, int_185450)
    
    
    # Getting the type of 'mode' (line 398)
    mode_185452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 24), 'mode')
    int_185453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 32), 'int')
    # Applying the binary operator '==' (line 398)
    result_eq_185454 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 24), '==', mode_185452, int_185453)
    
    # Applying the binary operator 'or' (line 398)
    result_or_keyword_185455 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 11), 'or', result_eq_185451, result_eq_185454)
    
    # Testing the type of an if condition (line 398)
    if_condition_185456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 398, 8), result_or_keyword_185455)
    # Assigning a type to the variable 'if_condition_185456' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'if_condition_185456', if_condition_185456)
    # SSA begins for if statement (line 398)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 402):
    
    # Assigning a Call to a Name (line 402):
    
    # Call to append(...): (line 402)
    # Processing the call arguments (line 402)
    
    # Call to fprime(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'x' (line 402)
    x_185459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 30), 'x', False)
    # Processing the call keyword arguments (line 402)
    kwargs_185460 = {}
    # Getting the type of 'fprime' (line 402)
    fprime_185458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 23), 'fprime', False)
    # Calling fprime(args, kwargs) (line 402)
    fprime_call_result_185461 = invoke(stypy.reporting.localization.Localization(__file__, 402, 23), fprime_185458, *[x_185459], **kwargs_185460)
    
    float_185462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 34), 'float')
    # Processing the call keyword arguments (line 402)
    kwargs_185463 = {}
    # Getting the type of 'append' (line 402)
    append_185457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 16), 'append', False)
    # Calling append(args, kwargs) (line 402)
    append_call_result_185464 = invoke(stypy.reporting.localization.Localization(__file__, 402, 16), append_185457, *[fprime_call_result_185461, float_185462], **kwargs_185463)
    
    # Assigning a type to the variable 'g' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'g', append_call_result_185464)
    
    
    # Obtaining the type of the subscript
    str_185465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 20), 'str', 'eq')
    # Getting the type of 'cons' (line 405)
    cons_185466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 15), 'cons')
    # Obtaining the member '__getitem__' of a type (line 405)
    getitem___185467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 15), cons_185466, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 405)
    subscript_call_result_185468 = invoke(stypy.reporting.localization.Localization(__file__, 405, 15), getitem___185467, str_185465)
    
    # Testing the type of an if condition (line 405)
    if_condition_185469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 12), subscript_call_result_185468)
    # Assigning a type to the variable 'if_condition_185469' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'if_condition_185469', if_condition_185469)
    # SSA begins for if statement (line 405)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 406):
    
    # Assigning a Call to a Name (line 406):
    
    # Call to vstack(...): (line 406)
    # Processing the call arguments (line 406)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    str_185482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 47), 'str', 'eq')
    # Getting the type of 'cons' (line 407)
    cons_185483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 42), 'cons', False)
    # Obtaining the member '__getitem__' of a type (line 407)
    getitem___185484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 42), cons_185483, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 407)
    subscript_call_result_185485 = invoke(stypy.reporting.localization.Localization(__file__, 407, 42), getitem___185484, str_185482)
    
    comprehension_185486 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 31), subscript_call_result_185485)
    # Assigning a type to the variable 'con' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 31), 'con', comprehension_185486)
    
    # Call to (...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'x' (line 406)
    x_185475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 42), 'x', False)
    
    # Obtaining the type of the subscript
    str_185476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 50), 'str', 'args')
    # Getting the type of 'con' (line 406)
    con_185477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 46), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___185478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 46), con_185477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_185479 = invoke(stypy.reporting.localization.Localization(__file__, 406, 46), getitem___185478, str_185476)
    
    # Processing the call keyword arguments (line 406)
    kwargs_185480 = {}
    
    # Obtaining the type of the subscript
    str_185471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 35), 'str', 'jac')
    # Getting the type of 'con' (line 406)
    con_185472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 31), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___185473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 31), con_185472, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_185474 = invoke(stypy.reporting.localization.Localization(__file__, 406, 31), getitem___185473, str_185471)
    
    # Calling (args, kwargs) (line 406)
    _call_result_185481 = invoke(stypy.reporting.localization.Localization(__file__, 406, 31), subscript_call_result_185474, *[x_185475, subscript_call_result_185479], **kwargs_185480)
    
    list_185487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 31), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 31), list_185487, _call_result_185481)
    # Processing the call keyword arguments (line 406)
    kwargs_185488 = {}
    # Getting the type of 'vstack' (line 406)
    vstack_185470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 23), 'vstack', False)
    # Calling vstack(args, kwargs) (line 406)
    vstack_call_result_185489 = invoke(stypy.reporting.localization.Localization(__file__, 406, 23), vstack_185470, *[list_185487], **kwargs_185488)
    
    # Assigning a type to the variable 'a_eq' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'a_eq', vstack_call_result_185489)
    # SSA branch for the else part of an if statement (line 405)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 409):
    
    # Assigning a Call to a Name (line 409):
    
    # Call to zeros(...): (line 409)
    # Processing the call arguments (line 409)
    
    # Obtaining an instance of the builtin type 'tuple' (line 409)
    tuple_185491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 409)
    # Adding element type (line 409)
    # Getting the type of 'meq' (line 409)
    meq_185492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 30), 'meq', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 30), tuple_185491, meq_185492)
    # Adding element type (line 409)
    # Getting the type of 'n' (line 409)
    n_185493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 35), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 30), tuple_185491, n_185493)
    
    # Processing the call keyword arguments (line 409)
    kwargs_185494 = {}
    # Getting the type of 'zeros' (line 409)
    zeros_185490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 23), 'zeros', False)
    # Calling zeros(args, kwargs) (line 409)
    zeros_call_result_185495 = invoke(stypy.reporting.localization.Localization(__file__, 409, 23), zeros_185490, *[tuple_185491], **kwargs_185494)
    
    # Assigning a type to the variable 'a_eq' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 16), 'a_eq', zeros_call_result_185495)
    # SSA join for if statement (line 405)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    str_185496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 20), 'str', 'ineq')
    # Getting the type of 'cons' (line 411)
    cons_185497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 15), 'cons')
    # Obtaining the member '__getitem__' of a type (line 411)
    getitem___185498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 15), cons_185497, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 411)
    subscript_call_result_185499 = invoke(stypy.reporting.localization.Localization(__file__, 411, 15), getitem___185498, str_185496)
    
    # Testing the type of an if condition (line 411)
    if_condition_185500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 12), subscript_call_result_185499)
    # Assigning a type to the variable 'if_condition_185500' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'if_condition_185500', if_condition_185500)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 412):
    
    # Assigning a Call to a Name (line 412):
    
    # Call to vstack(...): (line 412)
    # Processing the call arguments (line 412)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    str_185513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 48), 'str', 'ineq')
    # Getting the type of 'cons' (line 413)
    cons_185514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 43), 'cons', False)
    # Obtaining the member '__getitem__' of a type (line 413)
    getitem___185515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 43), cons_185514, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 413)
    subscript_call_result_185516 = invoke(stypy.reporting.localization.Localization(__file__, 413, 43), getitem___185515, str_185513)
    
    comprehension_185517 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 32), subscript_call_result_185516)
    # Assigning a type to the variable 'con' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 32), 'con', comprehension_185517)
    
    # Call to (...): (line 412)
    # Processing the call arguments (line 412)
    # Getting the type of 'x' (line 412)
    x_185506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 43), 'x', False)
    
    # Obtaining the type of the subscript
    str_185507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 51), 'str', 'args')
    # Getting the type of 'con' (line 412)
    con_185508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 47), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 412)
    getitem___185509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 47), con_185508, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 412)
    subscript_call_result_185510 = invoke(stypy.reporting.localization.Localization(__file__, 412, 47), getitem___185509, str_185507)
    
    # Processing the call keyword arguments (line 412)
    kwargs_185511 = {}
    
    # Obtaining the type of the subscript
    str_185502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 36), 'str', 'jac')
    # Getting the type of 'con' (line 412)
    con_185503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 32), 'con', False)
    # Obtaining the member '__getitem__' of a type (line 412)
    getitem___185504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 32), con_185503, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 412)
    subscript_call_result_185505 = invoke(stypy.reporting.localization.Localization(__file__, 412, 32), getitem___185504, str_185502)
    
    # Calling (args, kwargs) (line 412)
    _call_result_185512 = invoke(stypy.reporting.localization.Localization(__file__, 412, 32), subscript_call_result_185505, *[x_185506, subscript_call_result_185510], **kwargs_185511)
    
    list_185518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 32), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 32), list_185518, _call_result_185512)
    # Processing the call keyword arguments (line 412)
    kwargs_185519 = {}
    # Getting the type of 'vstack' (line 412)
    vstack_185501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 24), 'vstack', False)
    # Calling vstack(args, kwargs) (line 412)
    vstack_call_result_185520 = invoke(stypy.reporting.localization.Localization(__file__, 412, 24), vstack_185501, *[list_185518], **kwargs_185519)
    
    # Assigning a type to the variable 'a_ieq' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 16), 'a_ieq', vstack_call_result_185520)
    # SSA branch for the else part of an if statement (line 411)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 415):
    
    # Assigning a Call to a Name (line 415):
    
    # Call to zeros(...): (line 415)
    # Processing the call arguments (line 415)
    
    # Obtaining an instance of the builtin type 'tuple' (line 415)
    tuple_185522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 415)
    # Adding element type (line 415)
    # Getting the type of 'mieq' (line 415)
    mieq_185523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 31), 'mieq', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 31), tuple_185522, mieq_185523)
    # Adding element type (line 415)
    # Getting the type of 'n' (line 415)
    n_185524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 37), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 31), tuple_185522, n_185524)
    
    # Processing the call keyword arguments (line 415)
    kwargs_185525 = {}
    # Getting the type of 'zeros' (line 415)
    zeros_185521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 24), 'zeros', False)
    # Calling zeros(args, kwargs) (line 415)
    zeros_call_result_185526 = invoke(stypy.reporting.localization.Localization(__file__, 415, 24), zeros_185521, *[tuple_185522], **kwargs_185525)
    
    # Assigning a type to the variable 'a_ieq' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), 'a_ieq', zeros_call_result_185526)
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'm' (line 418)
    m_185527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'm')
    int_185528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 20), 'int')
    # Applying the binary operator '==' (line 418)
    result_eq_185529 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 15), '==', m_185527, int_185528)
    
    # Testing the type of an if condition (line 418)
    if_condition_185530 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 12), result_eq_185529)
    # Assigning a type to the variable 'if_condition_185530' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'if_condition_185530', if_condition_185530)
    # SSA begins for if statement (line 418)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 419):
    
    # Assigning a Call to a Name (line 419):
    
    # Call to zeros(...): (line 419)
    # Processing the call arguments (line 419)
    
    # Obtaining an instance of the builtin type 'tuple' (line 419)
    tuple_185532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 419)
    # Adding element type (line 419)
    # Getting the type of 'la' (line 419)
    la_185533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 27), 'la', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 27), tuple_185532, la_185533)
    # Adding element type (line 419)
    # Getting the type of 'n' (line 419)
    n_185534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 31), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 27), tuple_185532, n_185534)
    
    # Processing the call keyword arguments (line 419)
    kwargs_185535 = {}
    # Getting the type of 'zeros' (line 419)
    zeros_185531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 20), 'zeros', False)
    # Calling zeros(args, kwargs) (line 419)
    zeros_call_result_185536 = invoke(stypy.reporting.localization.Localization(__file__, 419, 20), zeros_185531, *[tuple_185532], **kwargs_185535)
    
    # Assigning a type to the variable 'a' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'a', zeros_call_result_185536)
    # SSA branch for the else part of an if statement (line 418)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 421):
    
    # Assigning a Call to a Name (line 421):
    
    # Call to vstack(...): (line 421)
    # Processing the call arguments (line 421)
    
    # Obtaining an instance of the builtin type 'tuple' (line 421)
    tuple_185538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 421)
    # Adding element type (line 421)
    # Getting the type of 'a_eq' (line 421)
    a_eq_185539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 28), 'a_eq', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 28), tuple_185538, a_eq_185539)
    # Adding element type (line 421)
    # Getting the type of 'a_ieq' (line 421)
    a_ieq_185540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 34), 'a_ieq', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 28), tuple_185538, a_ieq_185540)
    
    # Processing the call keyword arguments (line 421)
    kwargs_185541 = {}
    # Getting the type of 'vstack' (line 421)
    vstack_185537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 20), 'vstack', False)
    # Calling vstack(args, kwargs) (line 421)
    vstack_call_result_185542 = invoke(stypy.reporting.localization.Localization(__file__, 421, 20), vstack_185537, *[tuple_185538], **kwargs_185541)
    
    # Assigning a type to the variable 'a' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 16), 'a', vstack_call_result_185542)
    # SSA join for if statement (line 418)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 422):
    
    # Assigning a Call to a Name (line 422):
    
    # Call to concatenate(...): (line 422)
    # Processing the call arguments (line 422)
    
    # Obtaining an instance of the builtin type 'tuple' (line 422)
    tuple_185544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 422)
    # Adding element type (line 422)
    # Getting the type of 'a' (line 422)
    a_185545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 29), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 29), tuple_185544, a_185545)
    # Adding element type (line 422)
    
    # Call to zeros(...): (line 422)
    # Processing the call arguments (line 422)
    
    # Obtaining an instance of the builtin type 'list' (line 422)
    list_185547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 422)
    # Adding element type (line 422)
    # Getting the type of 'la' (line 422)
    la_185548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 39), 'la', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 38), list_185547, la_185548)
    # Adding element type (line 422)
    int_185549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 38), list_185547, int_185549)
    
    # Processing the call keyword arguments (line 422)
    kwargs_185550 = {}
    # Getting the type of 'zeros' (line 422)
    zeros_185546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 32), 'zeros', False)
    # Calling zeros(args, kwargs) (line 422)
    zeros_call_result_185551 = invoke(stypy.reporting.localization.Localization(__file__, 422, 32), zeros_185546, *[list_185547], **kwargs_185550)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 29), tuple_185544, zeros_call_result_185551)
    
    int_185552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 49), 'int')
    # Processing the call keyword arguments (line 422)
    kwargs_185553 = {}
    # Getting the type of 'concatenate' (line 422)
    concatenate_185543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 16), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 422)
    concatenate_call_result_185554 = invoke(stypy.reporting.localization.Localization(__file__, 422, 16), concatenate_185543, *[tuple_185544, int_185552], **kwargs_185553)
    
    # Assigning a type to the variable 'a' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'a', concatenate_call_result_185554)
    # SSA join for if statement (line 398)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to slsqp(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'm' (line 425)
    m_185556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 14), 'm', False)
    # Getting the type of 'meq' (line 425)
    meq_185557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 17), 'meq', False)
    # Getting the type of 'x' (line 425)
    x_185558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 22), 'x', False)
    # Getting the type of 'xl' (line 425)
    xl_185559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 25), 'xl', False)
    # Getting the type of 'xu' (line 425)
    xu_185560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 29), 'xu', False)
    # Getting the type of 'fx' (line 425)
    fx_185561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 33), 'fx', False)
    # Getting the type of 'c' (line 425)
    c_185562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 37), 'c', False)
    # Getting the type of 'g' (line 425)
    g_185563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 40), 'g', False)
    # Getting the type of 'a' (line 425)
    a_185564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 43), 'a', False)
    # Getting the type of 'acc' (line 425)
    acc_185565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 46), 'acc', False)
    # Getting the type of 'majiter' (line 425)
    majiter_185566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 51), 'majiter', False)
    # Getting the type of 'mode' (line 425)
    mode_185567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 60), 'mode', False)
    # Getting the type of 'w' (line 425)
    w_185568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 66), 'w', False)
    # Getting the type of 'jw' (line 425)
    jw_185569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 69), 'jw', False)
    # Processing the call keyword arguments (line 425)
    kwargs_185570 = {}
    # Getting the type of 'slsqp' (line 425)
    slsqp_185555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'slsqp', False)
    # Calling slsqp(args, kwargs) (line 425)
    slsqp_call_result_185571 = invoke(stypy.reporting.localization.Localization(__file__, 425, 8), slsqp_185555, *[m_185556, meq_185557, x_185558, xl_185559, xu_185560, fx_185561, c_185562, g_185563, a_185564, acc_185565, majiter_185566, mode_185567, w_185568, jw_185569], **kwargs_185570)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'callback' (line 428)
    callback_185572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 11), 'callback')
    # Getting the type of 'None' (line 428)
    None_185573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 27), 'None')
    # Applying the binary operator 'isnot' (line 428)
    result_is_not_185574 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 11), 'isnot', callback_185572, None_185573)
    
    
    # Getting the type of 'majiter' (line 428)
    majiter_185575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 36), 'majiter')
    # Getting the type of 'majiter_prev' (line 428)
    majiter_prev_185576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 46), 'majiter_prev')
    # Applying the binary operator '>' (line 428)
    result_gt_185577 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 36), '>', majiter_185575, majiter_prev_185576)
    
    # Applying the binary operator 'and' (line 428)
    result_and_keyword_185578 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 11), 'and', result_is_not_185574, result_gt_185577)
    
    # Testing the type of an if condition (line 428)
    if_condition_185579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 8), result_and_keyword_185578)
    # Assigning a type to the variable 'if_condition_185579' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'if_condition_185579', if_condition_185579)
    # SSA begins for if statement (line 428)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to callback(...): (line 429)
    # Processing the call arguments (line 429)
    # Getting the type of 'x' (line 429)
    x_185581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 21), 'x', False)
    # Processing the call keyword arguments (line 429)
    kwargs_185582 = {}
    # Getting the type of 'callback' (line 429)
    callback_185580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'callback', False)
    # Calling callback(args, kwargs) (line 429)
    callback_call_result_185583 = invoke(stypy.reporting.localization.Localization(__file__, 429, 12), callback_185580, *[x_185581], **kwargs_185582)
    
    # SSA join for if statement (line 428)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'iprint' (line 433)
    iprint_185584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'iprint')
    int_185585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 21), 'int')
    # Applying the binary operator '>=' (line 433)
    result_ge_185586 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 11), '>=', iprint_185584, int_185585)
    
    
    # Getting the type of 'majiter' (line 433)
    majiter_185587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 27), 'majiter')
    # Getting the type of 'majiter_prev' (line 433)
    majiter_prev_185588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 37), 'majiter_prev')
    # Applying the binary operator '>' (line 433)
    result_gt_185589 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 27), '>', majiter_185587, majiter_prev_185588)
    
    # Applying the binary operator 'and' (line 433)
    result_and_keyword_185590 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 11), 'and', result_ge_185586, result_gt_185589)
    
    # Testing the type of an if condition (line 433)
    if_condition_185591 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 8), result_and_keyword_185590)
    # Assigning a type to the variable 'if_condition_185591' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'if_condition_185591', if_condition_185591)
    # SSA begins for if statement (line 433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 434)
    # Processing the call arguments (line 434)
    str_185593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 18), 'str', '%5i %5i % 16.6E % 16.6E')
    
    # Obtaining an instance of the builtin type 'tuple' (line 434)
    tuple_185594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 434)
    # Adding element type (line 434)
    # Getting the type of 'majiter' (line 434)
    majiter_185595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 47), 'majiter', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 47), tuple_185594, majiter_185595)
    # Adding element type (line 434)
    
    # Obtaining the type of the subscript
    int_185596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 62), 'int')
    # Getting the type of 'feval' (line 434)
    feval_185597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 56), 'feval', False)
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___185598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 56), feval_185597, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 434)
    subscript_call_result_185599 = invoke(stypy.reporting.localization.Localization(__file__, 434, 56), getitem___185598, int_185596)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 47), tuple_185594, subscript_call_result_185599)
    # Adding element type (line 434)
    # Getting the type of 'fx' (line 435)
    fx_185600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 47), 'fx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 47), tuple_185594, fx_185600)
    # Adding element type (line 434)
    
    # Call to norm(...): (line 435)
    # Processing the call arguments (line 435)
    # Getting the type of 'g' (line 435)
    g_185603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 63), 'g', False)
    # Processing the call keyword arguments (line 435)
    kwargs_185604 = {}
    # Getting the type of 'linalg' (line 435)
    linalg_185601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 51), 'linalg', False)
    # Obtaining the member 'norm' of a type (line 435)
    norm_185602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 51), linalg_185601, 'norm')
    # Calling norm(args, kwargs) (line 435)
    norm_call_result_185605 = invoke(stypy.reporting.localization.Localization(__file__, 435, 51), norm_185602, *[g_185603], **kwargs_185604)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 47), tuple_185594, norm_call_result_185605)
    
    # Applying the binary operator '%' (line 434)
    result_mod_185606 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 18), '%', str_185593, tuple_185594)
    
    # Processing the call keyword arguments (line 434)
    kwargs_185607 = {}
    # Getting the type of 'print' (line 434)
    print_185592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'print', False)
    # Calling print(args, kwargs) (line 434)
    print_call_result_185608 = invoke(stypy.reporting.localization.Localization(__file__, 434, 12), print_185592, *[result_mod_185606], **kwargs_185607)
    
    # SSA join for if statement (line 433)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to abs(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'mode' (line 438)
    mode_185610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'mode', False)
    # Processing the call keyword arguments (line 438)
    kwargs_185611 = {}
    # Getting the type of 'abs' (line 438)
    abs_185609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 438)
    abs_call_result_185612 = invoke(stypy.reporting.localization.Localization(__file__, 438, 11), abs_185609, *[mode_185610], **kwargs_185611)
    
    int_185613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 24), 'int')
    # Applying the binary operator '!=' (line 438)
    result_ne_185614 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 11), '!=', abs_call_result_185612, int_185613)
    
    # Testing the type of an if condition (line 438)
    if_condition_185615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 8), result_ne_185614)
    # Assigning a type to the variable 'if_condition_185615' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'if_condition_185615', if_condition_185615)
    # SSA begins for if statement (line 438)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 438)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 441):
    
    # Assigning a Call to a Name (line 441):
    
    # Call to int(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'majiter' (line 441)
    majiter_185617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 27), 'majiter', False)
    # Processing the call keyword arguments (line 441)
    kwargs_185618 = {}
    # Getting the type of 'int' (line 441)
    int_185616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 23), 'int', False)
    # Calling int(args, kwargs) (line 441)
    int_call_result_185619 = invoke(stypy.reporting.localization.Localization(__file__, 441, 23), int_185616, *[majiter_185617], **kwargs_185618)
    
    # Assigning a type to the variable 'majiter_prev' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'majiter_prev', int_call_result_185619)
    # SSA join for while statement (line 373)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iprint' (line 444)
    iprint_185620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 7), 'iprint')
    int_185621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 17), 'int')
    # Applying the binary operator '>=' (line 444)
    result_ge_185622 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 7), '>=', iprint_185620, int_185621)
    
    # Testing the type of an if condition (line 444)
    if_condition_185623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 444, 4), result_ge_185622)
    # Assigning a type to the variable 'if_condition_185623' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'if_condition_185623', if_condition_185623)
    # SSA begins for if statement (line 444)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 445)
    # Processing the call arguments (line 445)
    
    # Obtaining the type of the subscript
    
    # Call to int(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'mode' (line 445)
    mode_185626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 29), 'mode', False)
    # Processing the call keyword arguments (line 445)
    kwargs_185627 = {}
    # Getting the type of 'int' (line 445)
    int_185625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 25), 'int', False)
    # Calling int(args, kwargs) (line 445)
    int_call_result_185628 = invoke(stypy.reporting.localization.Localization(__file__, 445, 25), int_185625, *[mode_185626], **kwargs_185627)
    
    # Getting the type of 'exit_modes' (line 445)
    exit_modes_185629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 14), 'exit_modes', False)
    # Obtaining the member '__getitem__' of a type (line 445)
    getitem___185630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 14), exit_modes_185629, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 445)
    subscript_call_result_185631 = invoke(stypy.reporting.localization.Localization(__file__, 445, 14), getitem___185630, int_call_result_185628)
    
    str_185632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 38), 'str', '    (Exit mode ')
    # Applying the binary operator '+' (line 445)
    result_add_185633 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 14), '+', subscript_call_result_185631, str_185632)
    
    
    # Call to str(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'mode' (line 445)
    mode_185635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 62), 'mode', False)
    # Processing the call keyword arguments (line 445)
    kwargs_185636 = {}
    # Getting the type of 'str' (line 445)
    str_185634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 58), 'str', False)
    # Calling str(args, kwargs) (line 445)
    str_call_result_185637 = invoke(stypy.reporting.localization.Localization(__file__, 445, 58), str_185634, *[mode_185635], **kwargs_185636)
    
    # Applying the binary operator '+' (line 445)
    result_add_185638 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 56), '+', result_add_185633, str_call_result_185637)
    
    str_185639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 70), 'str', ')')
    # Applying the binary operator '+' (line 445)
    result_add_185640 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 68), '+', result_add_185638, str_185639)
    
    # Processing the call keyword arguments (line 445)
    kwargs_185641 = {}
    # Getting the type of 'print' (line 445)
    print_185624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'print', False)
    # Calling print(args, kwargs) (line 445)
    print_call_result_185642 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), print_185624, *[result_add_185640], **kwargs_185641)
    
    
    # Call to print(...): (line 446)
    # Processing the call arguments (line 446)
    str_185644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 14), 'str', '            Current function value:')
    # Getting the type of 'fx' (line 446)
    fx_185645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 53), 'fx', False)
    # Processing the call keyword arguments (line 446)
    kwargs_185646 = {}
    # Getting the type of 'print' (line 446)
    print_185643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'print', False)
    # Calling print(args, kwargs) (line 446)
    print_call_result_185647 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), print_185643, *[str_185644, fx_185645], **kwargs_185646)
    
    
    # Call to print(...): (line 447)
    # Processing the call arguments (line 447)
    str_185649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 14), 'str', '            Iterations:')
    # Getting the type of 'majiter' (line 447)
    majiter_185650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 41), 'majiter', False)
    # Processing the call keyword arguments (line 447)
    kwargs_185651 = {}
    # Getting the type of 'print' (line 447)
    print_185648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'print', False)
    # Calling print(args, kwargs) (line 447)
    print_call_result_185652 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), print_185648, *[str_185649, majiter_185650], **kwargs_185651)
    
    
    # Call to print(...): (line 448)
    # Processing the call arguments (line 448)
    str_185654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 14), 'str', '            Function evaluations:')
    
    # Obtaining the type of the subscript
    int_185655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 57), 'int')
    # Getting the type of 'feval' (line 448)
    feval_185656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 51), 'feval', False)
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___185657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 51), feval_185656, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_185658 = invoke(stypy.reporting.localization.Localization(__file__, 448, 51), getitem___185657, int_185655)
    
    # Processing the call keyword arguments (line 448)
    kwargs_185659 = {}
    # Getting the type of 'print' (line 448)
    print_185653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'print', False)
    # Calling print(args, kwargs) (line 448)
    print_call_result_185660 = invoke(stypy.reporting.localization.Localization(__file__, 448, 8), print_185653, *[str_185654, subscript_call_result_185658], **kwargs_185659)
    
    
    # Call to print(...): (line 449)
    # Processing the call arguments (line 449)
    str_185662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 14), 'str', '            Gradient evaluations:')
    
    # Obtaining the type of the subscript
    int_185663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 57), 'int')
    # Getting the type of 'geval' (line 449)
    geval_185664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 51), 'geval', False)
    # Obtaining the member '__getitem__' of a type (line 449)
    getitem___185665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 51), geval_185664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 449)
    subscript_call_result_185666 = invoke(stypy.reporting.localization.Localization(__file__, 449, 51), getitem___185665, int_185663)
    
    # Processing the call keyword arguments (line 449)
    kwargs_185667 = {}
    # Getting the type of 'print' (line 449)
    print_185661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'print', False)
    # Calling print(args, kwargs) (line 449)
    print_call_result_185668 = invoke(stypy.reporting.localization.Localization(__file__, 449, 8), print_185661, *[str_185662, subscript_call_result_185666], **kwargs_185667)
    
    # SSA join for if statement (line 444)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to OptimizeResult(...): (line 451)
    # Processing the call keyword arguments (line 451)
    # Getting the type of 'x' (line 451)
    x_185670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 28), 'x', False)
    keyword_185671 = x_185670
    # Getting the type of 'fx' (line 451)
    fx_185672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 35), 'fx', False)
    keyword_185673 = fx_185672
    
    # Obtaining the type of the subscript
    int_185674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 46), 'int')
    slice_185675 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 451, 43), None, int_185674, None)
    # Getting the type of 'g' (line 451)
    g_185676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 43), 'g', False)
    # Obtaining the member '__getitem__' of a type (line 451)
    getitem___185677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 43), g_185676, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 451)
    subscript_call_result_185678 = invoke(stypy.reporting.localization.Localization(__file__, 451, 43), getitem___185677, slice_185675)
    
    keyword_185679 = subscript_call_result_185678
    
    # Call to int(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'majiter' (line 451)
    majiter_185681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 59), 'majiter', False)
    # Processing the call keyword arguments (line 451)
    kwargs_185682 = {}
    # Getting the type of 'int' (line 451)
    int_185680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 55), 'int', False)
    # Calling int(args, kwargs) (line 451)
    int_call_result_185683 = invoke(stypy.reporting.localization.Localization(__file__, 451, 55), int_185680, *[majiter_185681], **kwargs_185682)
    
    keyword_185684 = int_call_result_185683
    
    # Obtaining the type of the subscript
    int_185685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 37), 'int')
    # Getting the type of 'feval' (line 452)
    feval_185686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 31), 'feval', False)
    # Obtaining the member '__getitem__' of a type (line 452)
    getitem___185687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 31), feval_185686, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 452)
    subscript_call_result_185688 = invoke(stypy.reporting.localization.Localization(__file__, 452, 31), getitem___185687, int_185685)
    
    keyword_185689 = subscript_call_result_185688
    
    # Obtaining the type of the subscript
    int_185690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 52), 'int')
    # Getting the type of 'geval' (line 452)
    geval_185691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 46), 'geval', False)
    # Obtaining the member '__getitem__' of a type (line 452)
    getitem___185692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 46), geval_185691, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 452)
    subscript_call_result_185693 = invoke(stypy.reporting.localization.Localization(__file__, 452, 46), getitem___185692, int_185690)
    
    keyword_185694 = subscript_call_result_185693
    
    # Call to int(...): (line 452)
    # Processing the call arguments (line 452)
    # Getting the type of 'mode' (line 452)
    mode_185696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 67), 'mode', False)
    # Processing the call keyword arguments (line 452)
    kwargs_185697 = {}
    # Getting the type of 'int' (line 452)
    int_185695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 63), 'int', False)
    # Calling int(args, kwargs) (line 452)
    int_call_result_185698 = invoke(stypy.reporting.localization.Localization(__file__, 452, 63), int_185695, *[mode_185696], **kwargs_185697)
    
    keyword_185699 = int_call_result_185698
    
    # Obtaining the type of the subscript
    
    # Call to int(...): (line 453)
    # Processing the call arguments (line 453)
    # Getting the type of 'mode' (line 453)
    mode_185701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 49), 'mode', False)
    # Processing the call keyword arguments (line 453)
    kwargs_185702 = {}
    # Getting the type of 'int' (line 453)
    int_185700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 45), 'int', False)
    # Calling int(args, kwargs) (line 453)
    int_call_result_185703 = invoke(stypy.reporting.localization.Localization(__file__, 453, 45), int_185700, *[mode_185701], **kwargs_185702)
    
    # Getting the type of 'exit_modes' (line 453)
    exit_modes_185704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 34), 'exit_modes', False)
    # Obtaining the member '__getitem__' of a type (line 453)
    getitem___185705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 34), exit_modes_185704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 453)
    subscript_call_result_185706 = invoke(stypy.reporting.localization.Localization(__file__, 453, 34), getitem___185705, int_call_result_185703)
    
    keyword_185707 = subscript_call_result_185706
    
    # Getting the type of 'mode' (line 453)
    mode_185708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 66), 'mode', False)
    int_185709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 74), 'int')
    # Applying the binary operator '==' (line 453)
    result_eq_185710 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 66), '==', mode_185708, int_185709)
    
    keyword_185711 = result_eq_185710
    kwargs_185712 = {'status': keyword_185699, 'success': keyword_185711, 'jac': keyword_185679, 'nfev': keyword_185689, 'fun': keyword_185673, 'x': keyword_185671, 'message': keyword_185707, 'njev': keyword_185694, 'nit': keyword_185684}
    # Getting the type of 'OptimizeResult' (line 451)
    OptimizeResult_185669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 11), 'OptimizeResult', False)
    # Calling OptimizeResult(args, kwargs) (line 451)
    OptimizeResult_call_result_185713 = invoke(stypy.reporting.localization.Localization(__file__, 451, 11), OptimizeResult_185669, *[], **kwargs_185712)
    
    # Assigning a type to the variable 'stypy_return_type' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type', OptimizeResult_call_result_185713)
    
    # ################# End of '_minimize_slsqp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_minimize_slsqp' in the type store
    # Getting the type of 'stypy_return_type' (line 214)
    stypy_return_type_185714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_185714)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_minimize_slsqp'
    return stypy_return_type_185714

# Assigning a type to the variable '_minimize_slsqp' (line 214)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), '_minimize_slsqp', _minimize_slsqp)

if (__name__ == '__main__'):

    @norecursion
    def fun(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'list' (line 459)
        list_185715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 459)
        # Adding element type (line 459)
        int_185716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 17), list_185715, int_185716)
        # Adding element type (line 459)
        int_185717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 17), list_185715, int_185717)
        # Adding element type (line 459)
        int_185718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 17), list_185715, int_185718)
        # Adding element type (line 459)
        int_185719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 17), list_185715, int_185719)
        # Adding element type (line 459)
        int_185720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 17), list_185715, int_185720)
        
        defaults = [list_185715]
        # Create a new context for function 'fun'
        module_type_store = module_type_store.open_function_context('fun', 459, 4, False)
        
        # Passed parameters checking function
        fun.stypy_localization = localization
        fun.stypy_type_of_self = None
        fun.stypy_type_store = module_type_store
        fun.stypy_function_name = 'fun'
        fun.stypy_param_names_list = ['x', 'r']
        fun.stypy_varargs_param_name = None
        fun.stypy_kwargs_param_name = None
        fun.stypy_call_defaults = defaults
        fun.stypy_call_varargs = varargs
        fun.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun', ['x', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun', localization, ['x', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun(...)' code ##################

        str_185721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 8), 'str', ' Objective function ')
        
        # Call to exp(...): (line 461)
        # Processing the call arguments (line 461)
        
        # Obtaining the type of the subscript
        int_185723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 21), 'int')
        # Getting the type of 'x' (line 461)
        x_185724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 461)
        getitem___185725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 19), x_185724, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 461)
        subscript_call_result_185726 = invoke(stypy.reporting.localization.Localization(__file__, 461, 19), getitem___185725, int_185723)
        
        # Processing the call keyword arguments (line 461)
        kwargs_185727 = {}
        # Getting the type of 'exp' (line 461)
        exp_185722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 15), 'exp', False)
        # Calling exp(args, kwargs) (line 461)
        exp_call_result_185728 = invoke(stypy.reporting.localization.Localization(__file__, 461, 15), exp_185722, *[subscript_call_result_185726], **kwargs_185727)
        
        
        # Obtaining the type of the subscript
        int_185729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 30), 'int')
        # Getting the type of 'r' (line 461)
        r_185730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 28), 'r')
        # Obtaining the member '__getitem__' of a type (line 461)
        getitem___185731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 28), r_185730, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 461)
        subscript_call_result_185732 = invoke(stypy.reporting.localization.Localization(__file__, 461, 28), getitem___185731, int_185729)
        
        
        # Obtaining the type of the subscript
        int_185733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 37), 'int')
        # Getting the type of 'x' (line 461)
        x_185734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 35), 'x')
        # Obtaining the member '__getitem__' of a type (line 461)
        getitem___185735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 35), x_185734, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 461)
        subscript_call_result_185736 = invoke(stypy.reporting.localization.Localization(__file__, 461, 35), getitem___185735, int_185733)
        
        int_185737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 41), 'int')
        # Applying the binary operator '**' (line 461)
        result_pow_185738 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 35), '**', subscript_call_result_185736, int_185737)
        
        # Applying the binary operator '*' (line 461)
        result_mul_185739 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 28), '*', subscript_call_result_185732, result_pow_185738)
        
        
        # Obtaining the type of the subscript
        int_185740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 47), 'int')
        # Getting the type of 'r' (line 461)
        r_185741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 45), 'r')
        # Obtaining the member '__getitem__' of a type (line 461)
        getitem___185742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 45), r_185741, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 461)
        subscript_call_result_185743 = invoke(stypy.reporting.localization.Localization(__file__, 461, 45), getitem___185742, int_185740)
        
        
        # Obtaining the type of the subscript
        int_185744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 54), 'int')
        # Getting the type of 'x' (line 461)
        x_185745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 52), 'x')
        # Obtaining the member '__getitem__' of a type (line 461)
        getitem___185746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 52), x_185745, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 461)
        subscript_call_result_185747 = invoke(stypy.reporting.localization.Localization(__file__, 461, 52), getitem___185746, int_185744)
        
        int_185748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 58), 'int')
        # Applying the binary operator '**' (line 461)
        result_pow_185749 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 52), '**', subscript_call_result_185747, int_185748)
        
        # Applying the binary operator '*' (line 461)
        result_mul_185750 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 45), '*', subscript_call_result_185743, result_pow_185749)
        
        # Applying the binary operator '+' (line 461)
        result_add_185751 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 28), '+', result_mul_185739, result_mul_185750)
        
        
        # Obtaining the type of the subscript
        int_185752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 30), 'int')
        # Getting the type of 'r' (line 462)
        r_185753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 28), 'r')
        # Obtaining the member '__getitem__' of a type (line 462)
        getitem___185754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 28), r_185753, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 462)
        subscript_call_result_185755 = invoke(stypy.reporting.localization.Localization(__file__, 462, 28), getitem___185754, int_185752)
        
        
        # Obtaining the type of the subscript
        int_185756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 37), 'int')
        # Getting the type of 'x' (line 462)
        x_185757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 35), 'x')
        # Obtaining the member '__getitem__' of a type (line 462)
        getitem___185758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 35), x_185757, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 462)
        subscript_call_result_185759 = invoke(stypy.reporting.localization.Localization(__file__, 462, 35), getitem___185758, int_185756)
        
        # Applying the binary operator '*' (line 462)
        result_mul_185760 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 28), '*', subscript_call_result_185755, subscript_call_result_185759)
        
        
        # Obtaining the type of the subscript
        int_185761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 44), 'int')
        # Getting the type of 'x' (line 462)
        x_185762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 42), 'x')
        # Obtaining the member '__getitem__' of a type (line 462)
        getitem___185763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 42), x_185762, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 462)
        subscript_call_result_185764 = invoke(stypy.reporting.localization.Localization(__file__, 462, 42), getitem___185763, int_185761)
        
        # Applying the binary operator '*' (line 462)
        result_mul_185765 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 40), '*', result_mul_185760, subscript_call_result_185764)
        
        # Applying the binary operator '+' (line 461)
        result_add_185766 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 60), '+', result_add_185751, result_mul_185765)
        
        
        # Obtaining the type of the subscript
        int_185767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 51), 'int')
        # Getting the type of 'r' (line 462)
        r_185768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 49), 'r')
        # Obtaining the member '__getitem__' of a type (line 462)
        getitem___185769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 49), r_185768, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 462)
        subscript_call_result_185770 = invoke(stypy.reporting.localization.Localization(__file__, 462, 49), getitem___185769, int_185767)
        
        
        # Obtaining the type of the subscript
        int_185771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 58), 'int')
        # Getting the type of 'x' (line 462)
        x_185772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 56), 'x')
        # Obtaining the member '__getitem__' of a type (line 462)
        getitem___185773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 56), x_185772, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 462)
        subscript_call_result_185774 = invoke(stypy.reporting.localization.Localization(__file__, 462, 56), getitem___185773, int_185771)
        
        # Applying the binary operator '*' (line 462)
        result_mul_185775 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 49), '*', subscript_call_result_185770, subscript_call_result_185774)
        
        # Applying the binary operator '+' (line 462)
        result_add_185776 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 47), '+', result_add_185766, result_mul_185775)
        
        
        # Obtaining the type of the subscript
        int_185777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 30), 'int')
        # Getting the type of 'r' (line 463)
        r_185778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 28), 'r')
        # Obtaining the member '__getitem__' of a type (line 463)
        getitem___185779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 28), r_185778, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 463)
        subscript_call_result_185780 = invoke(stypy.reporting.localization.Localization(__file__, 463, 28), getitem___185779, int_185777)
        
        # Applying the binary operator '+' (line 462)
        result_add_185781 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 61), '+', result_add_185776, subscript_call_result_185780)
        
        # Applying the binary operator '*' (line 461)
        result_mul_185782 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 15), '*', exp_call_result_185728, result_add_185781)
        
        # Assigning a type to the variable 'stypy_return_type' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'stypy_return_type', result_mul_185782)
        
        # ################# End of 'fun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun' in the type store
        # Getting the type of 'stypy_return_type' (line 459)
        stypy_return_type_185783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_185783)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun'
        return stypy_return_type_185783

    # Assigning a type to the variable 'fun' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'fun', fun)
    
    # Assigning a Attribute to a Name (line 466):
    
    # Assigning a Attribute to a Name (line 466):
    
    # Call to array(...): (line 466)
    # Processing the call arguments (line 466)
    
    # Obtaining an instance of the builtin type 'list' (line 466)
    list_185785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 466)
    # Adding element type (line 466)
    
    # Obtaining an instance of the builtin type 'list' (line 466)
    list_185786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 466)
    # Adding element type (line 466)
    
    # Getting the type of 'inf' (line 466)
    inf_185787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 20), 'inf', False)
    # Applying the 'usub' unary operator (line 466)
    result___neg___185788 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 19), 'usub', inf_185787)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 18), list_185786, result___neg___185788)
    
    int_185789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 25), 'int')
    # Applying the binary operator '*' (line 466)
    result_mul_185790 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 18), '*', list_185786, int_185789)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 17), list_185785, result_mul_185790)
    # Adding element type (line 466)
    
    # Obtaining an instance of the builtin type 'list' (line 466)
    list_185791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 466)
    # Adding element type (line 466)
    # Getting the type of 'inf' (line 466)
    inf_185792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 29), 'inf', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 28), list_185791, inf_185792)
    
    int_185793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 34), 'int')
    # Applying the binary operator '*' (line 466)
    result_mul_185794 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 28), '*', list_185791, int_185793)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 17), list_185785, result_mul_185794)
    
    # Processing the call keyword arguments (line 466)
    kwargs_185795 = {}
    # Getting the type of 'array' (line 466)
    array_185784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 11), 'array', False)
    # Calling array(args, kwargs) (line 466)
    array_call_result_185796 = invoke(stypy.reporting.localization.Localization(__file__, 466, 11), array_185784, *[list_185785], **kwargs_185795)
    
    # Obtaining the member 'T' of a type (line 466)
    T_185797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 11), array_call_result_185796, 'T')
    # Assigning a type to the variable 'bnds' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'bnds', T_185797)
    
    # Assigning a List to a Subscript (line 467):
    
    # Assigning a List to a Subscript (line 467):
    
    # Obtaining an instance of the builtin type 'list' (line 467)
    list_185798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 467)
    # Adding element type (line 467)
    float_185799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 17), list_185798, float_185799)
    # Adding element type (line 467)
    float_185800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 17), list_185798, float_185800)
    
    # Getting the type of 'bnds' (line 467)
    bnds_185801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'bnds')
    slice_185802 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 467, 4), None, None, None)
    int_185803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 12), 'int')
    # Storing an element on a container (line 467)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 4), bnds_185801, ((slice_185802, int_185803), list_185798))

    @norecursion
    def feqcon(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_185804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 20), 'int')
        defaults = [int_185804]
        # Create a new context for function 'feqcon'
        module_type_store = module_type_store.open_function_context('feqcon', 470, 4, False)
        
        # Passed parameters checking function
        feqcon.stypy_localization = localization
        feqcon.stypy_type_of_self = None
        feqcon.stypy_type_store = module_type_store
        feqcon.stypy_function_name = 'feqcon'
        feqcon.stypy_param_names_list = ['x', 'b']
        feqcon.stypy_varargs_param_name = None
        feqcon.stypy_kwargs_param_name = None
        feqcon.stypy_call_defaults = defaults
        feqcon.stypy_call_varargs = varargs
        feqcon.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'feqcon', ['x', 'b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'feqcon', localization, ['x', 'b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'feqcon(...)' code ##################

        str_185805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 8), 'str', ' Equality constraint ')
        
        # Call to array(...): (line 472)
        # Processing the call arguments (line 472)
        
        # Obtaining an instance of the builtin type 'list' (line 472)
        list_185807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 472)
        # Adding element type (line 472)
        
        # Obtaining the type of the subscript
        int_185808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 24), 'int')
        # Getting the type of 'x' (line 472)
        x_185809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 22), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 472)
        getitem___185810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 22), x_185809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 472)
        subscript_call_result_185811 = invoke(stypy.reporting.localization.Localization(__file__, 472, 22), getitem___185810, int_185808)
        
        int_185812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 28), 'int')
        # Applying the binary operator '**' (line 472)
        result_pow_185813 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 22), '**', subscript_call_result_185811, int_185812)
        
        
        # Obtaining the type of the subscript
        int_185814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 34), 'int')
        # Getting the type of 'x' (line 472)
        x_185815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 32), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 472)
        getitem___185816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 32), x_185815, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 472)
        subscript_call_result_185817 = invoke(stypy.reporting.localization.Localization(__file__, 472, 32), getitem___185816, int_185814)
        
        # Applying the binary operator '+' (line 472)
        result_add_185818 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 22), '+', result_pow_185813, subscript_call_result_185817)
        
        # Getting the type of 'b' (line 472)
        b_185819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 39), 'b', False)
        # Applying the binary operator '-' (line 472)
        result_sub_185820 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 37), '-', result_add_185818, b_185819)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 21), list_185807, result_sub_185820)
        
        # Processing the call keyword arguments (line 472)
        kwargs_185821 = {}
        # Getting the type of 'array' (line 472)
        array_185806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'array', False)
        # Calling array(args, kwargs) (line 472)
        array_call_result_185822 = invoke(stypy.reporting.localization.Localization(__file__, 472, 15), array_185806, *[list_185807], **kwargs_185821)
        
        # Assigning a type to the variable 'stypy_return_type' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'stypy_return_type', array_call_result_185822)
        
        # ################# End of 'feqcon(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'feqcon' in the type store
        # Getting the type of 'stypy_return_type' (line 470)
        stypy_return_type_185823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_185823)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'feqcon'
        return stypy_return_type_185823

    # Assigning a type to the variable 'feqcon' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'feqcon', feqcon)

    @norecursion
    def jeqcon(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_185824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 20), 'int')
        defaults = [int_185824]
        # Create a new context for function 'jeqcon'
        module_type_store = module_type_store.open_function_context('jeqcon', 474, 4, False)
        
        # Passed parameters checking function
        jeqcon.stypy_localization = localization
        jeqcon.stypy_type_of_self = None
        jeqcon.stypy_type_store = module_type_store
        jeqcon.stypy_function_name = 'jeqcon'
        jeqcon.stypy_param_names_list = ['x', 'b']
        jeqcon.stypy_varargs_param_name = None
        jeqcon.stypy_kwargs_param_name = None
        jeqcon.stypy_call_defaults = defaults
        jeqcon.stypy_call_varargs = varargs
        jeqcon.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'jeqcon', ['x', 'b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jeqcon', localization, ['x', 'b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jeqcon(...)' code ##################

        str_185825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 8), 'str', ' Jacobian of equality constraint ')
        
        # Call to array(...): (line 476)
        # Processing the call arguments (line 476)
        
        # Obtaining an instance of the builtin type 'list' (line 476)
        list_185827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 476)
        # Adding element type (line 476)
        
        # Obtaining an instance of the builtin type 'list' (line 476)
        list_185828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 476)
        # Adding element type (line 476)
        int_185829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 23), 'int')
        
        # Obtaining the type of the subscript
        int_185830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 27), 'int')
        # Getting the type of 'x' (line 476)
        x_185831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 25), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 476)
        getitem___185832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 25), x_185831, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 476)
        subscript_call_result_185833 = invoke(stypy.reporting.localization.Localization(__file__, 476, 25), getitem___185832, int_185830)
        
        # Applying the binary operator '*' (line 476)
        result_mul_185834 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 23), '*', int_185829, subscript_call_result_185833)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 22), list_185828, result_mul_185834)
        # Adding element type (line 476)
        int_185835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 22), list_185828, int_185835)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 21), list_185827, list_185828)
        
        # Processing the call keyword arguments (line 476)
        kwargs_185836 = {}
        # Getting the type of 'array' (line 476)
        array_185826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'array', False)
        # Calling array(args, kwargs) (line 476)
        array_call_result_185837 = invoke(stypy.reporting.localization.Localization(__file__, 476, 15), array_185826, *[list_185827], **kwargs_185836)
        
        # Assigning a type to the variable 'stypy_return_type' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'stypy_return_type', array_call_result_185837)
        
        # ################# End of 'jeqcon(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jeqcon' in the type store
        # Getting the type of 'stypy_return_type' (line 474)
        stypy_return_type_185838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_185838)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jeqcon'
        return stypy_return_type_185838

    # Assigning a type to the variable 'jeqcon' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'jeqcon', jeqcon)

    @norecursion
    def fieqcon(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_185839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 21), 'int')
        defaults = [int_185839]
        # Create a new context for function 'fieqcon'
        module_type_store = module_type_store.open_function_context('fieqcon', 478, 4, False)
        
        # Passed parameters checking function
        fieqcon.stypy_localization = localization
        fieqcon.stypy_type_of_self = None
        fieqcon.stypy_type_store = module_type_store
        fieqcon.stypy_function_name = 'fieqcon'
        fieqcon.stypy_param_names_list = ['x', 'c']
        fieqcon.stypy_varargs_param_name = None
        fieqcon.stypy_kwargs_param_name = None
        fieqcon.stypy_call_defaults = defaults
        fieqcon.stypy_call_varargs = varargs
        fieqcon.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fieqcon', ['x', 'c'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fieqcon', localization, ['x', 'c'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fieqcon(...)' code ##################

        str_185840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 8), 'str', ' Inequality constraint ')
        
        # Call to array(...): (line 480)
        # Processing the call arguments (line 480)
        
        # Obtaining an instance of the builtin type 'list' (line 480)
        list_185842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 480)
        # Adding element type (line 480)
        
        # Obtaining the type of the subscript
        int_185843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 24), 'int')
        # Getting the type of 'x' (line 480)
        x_185844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 22), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 480)
        getitem___185845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 22), x_185844, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 480)
        subscript_call_result_185846 = invoke(stypy.reporting.localization.Localization(__file__, 480, 22), getitem___185845, int_185843)
        
        
        # Obtaining the type of the subscript
        int_185847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 31), 'int')
        # Getting the type of 'x' (line 480)
        x_185848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 29), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 480)
        getitem___185849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 29), x_185848, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 480)
        subscript_call_result_185850 = invoke(stypy.reporting.localization.Localization(__file__, 480, 29), getitem___185849, int_185847)
        
        # Applying the binary operator '*' (line 480)
        result_mul_185851 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 22), '*', subscript_call_result_185846, subscript_call_result_185850)
        
        # Getting the type of 'c' (line 480)
        c_185852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 36), 'c', False)
        # Applying the binary operator '+' (line 480)
        result_add_185853 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 22), '+', result_mul_185851, c_185852)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 21), list_185842, result_add_185853)
        
        # Processing the call keyword arguments (line 480)
        kwargs_185854 = {}
        # Getting the type of 'array' (line 480)
        array_185841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 15), 'array', False)
        # Calling array(args, kwargs) (line 480)
        array_call_result_185855 = invoke(stypy.reporting.localization.Localization(__file__, 480, 15), array_185841, *[list_185842], **kwargs_185854)
        
        # Assigning a type to the variable 'stypy_return_type' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'stypy_return_type', array_call_result_185855)
        
        # ################# End of 'fieqcon(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fieqcon' in the type store
        # Getting the type of 'stypy_return_type' (line 478)
        stypy_return_type_185856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_185856)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fieqcon'
        return stypy_return_type_185856

    # Assigning a type to the variable 'fieqcon' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'fieqcon', fieqcon)

    @norecursion
    def jieqcon(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_185857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 21), 'int')
        defaults = [int_185857]
        # Create a new context for function 'jieqcon'
        module_type_store = module_type_store.open_function_context('jieqcon', 482, 4, False)
        
        # Passed parameters checking function
        jieqcon.stypy_localization = localization
        jieqcon.stypy_type_of_self = None
        jieqcon.stypy_type_store = module_type_store
        jieqcon.stypy_function_name = 'jieqcon'
        jieqcon.stypy_param_names_list = ['x', 'c']
        jieqcon.stypy_varargs_param_name = None
        jieqcon.stypy_kwargs_param_name = None
        jieqcon.stypy_call_defaults = defaults
        jieqcon.stypy_call_varargs = varargs
        jieqcon.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'jieqcon', ['x', 'c'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'jieqcon', localization, ['x', 'c'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'jieqcon(...)' code ##################

        str_185858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 8), 'str', ' Jacobian of Inequality constraint ')
        
        # Call to array(...): (line 484)
        # Processing the call arguments (line 484)
        
        # Obtaining an instance of the builtin type 'list' (line 484)
        list_185860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 484)
        # Adding element type (line 484)
        
        # Obtaining an instance of the builtin type 'list' (line 484)
        list_185861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 484)
        # Adding element type (line 484)
        int_185862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 22), list_185861, int_185862)
        # Adding element type (line 484)
        int_185863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 22), list_185861, int_185863)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 21), list_185860, list_185861)
        
        # Processing the call keyword arguments (line 484)
        kwargs_185864 = {}
        # Getting the type of 'array' (line 484)
        array_185859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 15), 'array', False)
        # Calling array(args, kwargs) (line 484)
        array_call_result_185865 = invoke(stypy.reporting.localization.Localization(__file__, 484, 15), array_185859, *[list_185860], **kwargs_185864)
        
        # Assigning a type to the variable 'stypy_return_type' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'stypy_return_type', array_call_result_185865)
        
        # ################# End of 'jieqcon(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'jieqcon' in the type store
        # Getting the type of 'stypy_return_type' (line 482)
        stypy_return_type_185866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_185866)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'jieqcon'
        return stypy_return_type_185866

    # Assigning a type to the variable 'jieqcon' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'jieqcon', jieqcon)
    
    # Assigning a Tuple to a Name (line 487):
    
    # Assigning a Tuple to a Name (line 487):
    
    # Obtaining an instance of the builtin type 'tuple' (line 487)
    tuple_185867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 487)
    # Adding element type (line 487)
    
    # Obtaining an instance of the builtin type 'dict' (line 487)
    dict_185868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 487)
    # Adding element type (key, value) (line 487)
    str_185869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 13), 'str', 'type')
    str_185870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 21), 'str', 'eq')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), dict_185868, (str_185869, str_185870))
    # Adding element type (key, value) (line 487)
    str_185871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 27), 'str', 'fun')
    # Getting the type of 'feqcon' (line 487)
    feqcon_185872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 34), 'feqcon')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), dict_185868, (str_185871, feqcon_185872))
    # Adding element type (key, value) (line 487)
    str_185873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 42), 'str', 'jac')
    # Getting the type of 'jeqcon' (line 487)
    jeqcon_185874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 49), 'jeqcon')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), dict_185868, (str_185873, jeqcon_185874))
    # Adding element type (key, value) (line 487)
    str_185875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 57), 'str', 'args')
    
    # Obtaining an instance of the builtin type 'tuple' (line 487)
    tuple_185876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 66), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 487)
    # Adding element type (line 487)
    int_185877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 66), tuple_185876, int_185877)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), dict_185868, (str_185875, tuple_185876))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), tuple_185867, dict_185868)
    # Adding element type (line 487)
    
    # Obtaining an instance of the builtin type 'dict' (line 488)
    dict_185878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 488)
    # Adding element type (key, value) (line 488)
    str_185879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 13), 'str', 'type')
    str_185880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 21), 'str', 'ineq')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 12), dict_185878, (str_185879, str_185880))
    # Adding element type (key, value) (line 488)
    str_185881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 29), 'str', 'fun')
    # Getting the type of 'fieqcon' (line 488)
    fieqcon_185882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 36), 'fieqcon')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 12), dict_185878, (str_185881, fieqcon_185882))
    # Adding element type (key, value) (line 488)
    str_185883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 45), 'str', 'jac')
    # Getting the type of 'jieqcon' (line 488)
    jieqcon_185884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 52), 'jieqcon')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 12), dict_185878, (str_185883, jieqcon_185884))
    # Adding element type (key, value) (line 488)
    str_185885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 61), 'str', 'args')
    
    # Obtaining an instance of the builtin type 'tuple' (line 488)
    tuple_185886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 70), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 488)
    # Adding element type (line 488)
    int_185887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 70), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 70), tuple_185886, int_185887)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 12), dict_185878, (str_185885, tuple_185886))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), tuple_185867, dict_185878)
    
    # Assigning a type to the variable 'cons' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'cons', tuple_185867)
    
    # Call to print(...): (line 491)
    # Processing the call arguments (line 491)
    
    # Call to center(...): (line 491)
    # Processing the call arguments (line 491)
    int_185891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 40), 'int')
    str_185892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 44), 'str', '-')
    # Processing the call keyword arguments (line 491)
    kwargs_185893 = {}
    str_185889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 10), 'str', ' Bounds constraints ')
    # Obtaining the member 'center' of a type (line 491)
    center_185890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 10), str_185889, 'center')
    # Calling center(args, kwargs) (line 491)
    center_call_result_185894 = invoke(stypy.reporting.localization.Localization(__file__, 491, 10), center_185890, *[int_185891, str_185892], **kwargs_185893)
    
    # Processing the call keyword arguments (line 491)
    kwargs_185895 = {}
    # Getting the type of 'print' (line 491)
    print_185888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'print', False)
    # Calling print(args, kwargs) (line 491)
    print_call_result_185896 = invoke(stypy.reporting.localization.Localization(__file__, 491, 4), print_185888, *[center_call_result_185894], **kwargs_185895)
    
    
    # Call to print(...): (line 492)
    # Processing the call arguments (line 492)
    str_185898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 10), 'str', ' * fmin_slsqp')
    # Processing the call keyword arguments (line 492)
    kwargs_185899 = {}
    # Getting the type of 'print' (line 492)
    print_185897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'print', False)
    # Calling print(args, kwargs) (line 492)
    print_call_result_185900 = invoke(stypy.reporting.localization.Localization(__file__, 492, 4), print_185897, *[str_185898], **kwargs_185899)
    
    
    # Assigning a Subscript to a Tuple (line 493):
    
    # Assigning a Subscript to a Name (line 493):
    
    # Obtaining the type of the subscript
    int_185901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 4), 'int')
    
    # Obtaining the type of the subscript
    int_185902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 41), 'int')
    slice_185903 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 493, 11), None, int_185902, None)
    
    # Call to fmin_slsqp(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of 'fun' (line 493)
    fun_185905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 22), 'fun', False)
    
    # Call to array(...): (line 493)
    # Processing the call arguments (line 493)
    
    # Obtaining an instance of the builtin type 'list' (line 493)
    list_185907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 493)
    # Adding element type (line 493)
    int_185908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 33), list_185907, int_185908)
    # Adding element type (line 493)
    int_185909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 33), list_185907, int_185909)
    
    # Processing the call keyword arguments (line 493)
    kwargs_185910 = {}
    # Getting the type of 'array' (line 493)
    array_185906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 27), 'array', False)
    # Calling array(args, kwargs) (line 493)
    array_call_result_185911 = invoke(stypy.reporting.localization.Localization(__file__, 493, 27), array_185906, *[list_185907], **kwargs_185910)
    
    # Processing the call keyword arguments (line 493)
    # Getting the type of 'bnds' (line 493)
    bnds_185912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 50), 'bnds', False)
    keyword_185913 = bnds_185912
    int_185914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 61), 'int')
    keyword_185915 = int_185914
    # Getting the type of 'True' (line 494)
    True_185916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 34), 'True', False)
    keyword_185917 = True_185916
    kwargs_185918 = {'disp': keyword_185915, 'bounds': keyword_185913, 'full_output': keyword_185917}
    # Getting the type of 'fmin_slsqp' (line 493)
    fmin_slsqp_185904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 11), 'fmin_slsqp', False)
    # Calling fmin_slsqp(args, kwargs) (line 493)
    fmin_slsqp_call_result_185919 = invoke(stypy.reporting.localization.Localization(__file__, 493, 11), fmin_slsqp_185904, *[fun_185905, array_call_result_185911], **kwargs_185918)
    
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___185920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 11), fmin_slsqp_call_result_185919, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 493)
    subscript_call_result_185921 = invoke(stypy.reporting.localization.Localization(__file__, 493, 11), getitem___185920, slice_185903)
    
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___185922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 4), subscript_call_result_185921, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 493)
    subscript_call_result_185923 = invoke(stypy.reporting.localization.Localization(__file__, 493, 4), getitem___185922, int_185901)
    
    # Assigning a type to the variable 'tuple_var_assignment_184548' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'tuple_var_assignment_184548', subscript_call_result_185923)
    
    # Assigning a Subscript to a Name (line 493):
    
    # Obtaining the type of the subscript
    int_185924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 4), 'int')
    
    # Obtaining the type of the subscript
    int_185925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 41), 'int')
    slice_185926 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 493, 11), None, int_185925, None)
    
    # Call to fmin_slsqp(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of 'fun' (line 493)
    fun_185928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 22), 'fun', False)
    
    # Call to array(...): (line 493)
    # Processing the call arguments (line 493)
    
    # Obtaining an instance of the builtin type 'list' (line 493)
    list_185930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 493)
    # Adding element type (line 493)
    int_185931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 33), list_185930, int_185931)
    # Adding element type (line 493)
    int_185932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 33), list_185930, int_185932)
    
    # Processing the call keyword arguments (line 493)
    kwargs_185933 = {}
    # Getting the type of 'array' (line 493)
    array_185929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 27), 'array', False)
    # Calling array(args, kwargs) (line 493)
    array_call_result_185934 = invoke(stypy.reporting.localization.Localization(__file__, 493, 27), array_185929, *[list_185930], **kwargs_185933)
    
    # Processing the call keyword arguments (line 493)
    # Getting the type of 'bnds' (line 493)
    bnds_185935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 50), 'bnds', False)
    keyword_185936 = bnds_185935
    int_185937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 61), 'int')
    keyword_185938 = int_185937
    # Getting the type of 'True' (line 494)
    True_185939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 34), 'True', False)
    keyword_185940 = True_185939
    kwargs_185941 = {'disp': keyword_185938, 'bounds': keyword_185936, 'full_output': keyword_185940}
    # Getting the type of 'fmin_slsqp' (line 493)
    fmin_slsqp_185927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 11), 'fmin_slsqp', False)
    # Calling fmin_slsqp(args, kwargs) (line 493)
    fmin_slsqp_call_result_185942 = invoke(stypy.reporting.localization.Localization(__file__, 493, 11), fmin_slsqp_185927, *[fun_185928, array_call_result_185934], **kwargs_185941)
    
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___185943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 11), fmin_slsqp_call_result_185942, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 493)
    subscript_call_result_185944 = invoke(stypy.reporting.localization.Localization(__file__, 493, 11), getitem___185943, slice_185926)
    
    # Obtaining the member '__getitem__' of a type (line 493)
    getitem___185945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 4), subscript_call_result_185944, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 493)
    subscript_call_result_185946 = invoke(stypy.reporting.localization.Localization(__file__, 493, 4), getitem___185945, int_185924)
    
    # Assigning a type to the variable 'tuple_var_assignment_184549' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'tuple_var_assignment_184549', subscript_call_result_185946)
    
    # Assigning a Name to a Name (line 493):
    # Getting the type of 'tuple_var_assignment_184548' (line 493)
    tuple_var_assignment_184548_185947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'tuple_var_assignment_184548')
    # Assigning a type to the variable 'x' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'x', tuple_var_assignment_184548_185947)
    
    # Assigning a Name to a Name (line 493):
    # Getting the type of 'tuple_var_assignment_184549' (line 493)
    tuple_var_assignment_184549_185948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'tuple_var_assignment_184549')
    # Assigning a type to the variable 'f' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 7), 'f', tuple_var_assignment_184549_185948)
    
    # Call to print(...): (line 495)
    # Processing the call arguments (line 495)
    str_185950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 10), 'str', ' * _minimize_slsqp')
    # Processing the call keyword arguments (line 495)
    kwargs_185951 = {}
    # Getting the type of 'print' (line 495)
    print_185949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'print', False)
    # Calling print(args, kwargs) (line 495)
    print_call_result_185952 = invoke(stypy.reporting.localization.Localization(__file__, 495, 4), print_185949, *[str_185950], **kwargs_185951)
    
    
    # Assigning a Call to a Name (line 496):
    
    # Assigning a Call to a Name (line 496):
    
    # Call to _minimize_slsqp(...): (line 496)
    # Processing the call arguments (line 496)
    # Getting the type of 'fun' (line 496)
    fun_185954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 26), 'fun', False)
    
    # Call to array(...): (line 496)
    # Processing the call arguments (line 496)
    
    # Obtaining an instance of the builtin type 'list' (line 496)
    list_185956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 496)
    # Adding element type (line 496)
    int_185957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 37), list_185956, int_185957)
    # Adding element type (line 496)
    int_185958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 37), list_185956, int_185958)
    
    # Processing the call keyword arguments (line 496)
    kwargs_185959 = {}
    # Getting the type of 'array' (line 496)
    array_185955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 31), 'array', False)
    # Calling array(args, kwargs) (line 496)
    array_call_result_185960 = invoke(stypy.reporting.localization.Localization(__file__, 496, 31), array_185955, *[list_185956], **kwargs_185959)
    
    # Processing the call keyword arguments (line 496)
    # Getting the type of 'bnds' (line 496)
    bnds_185961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 54), 'bnds', False)
    keyword_185962 = bnds_185961
    
    # Obtaining an instance of the builtin type 'dict' (line 497)
    dict_185963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 28), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 497)
    # Adding element type (key, value) (line 497)
    str_185964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 29), 'str', 'disp')
    # Getting the type of 'True' (line 497)
    True_185965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 37), 'True', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 28), dict_185963, (str_185964, True_185965))
    
    kwargs_185966 = {'dict_185963': dict_185963, 'bounds': keyword_185962}
    # Getting the type of '_minimize_slsqp' (line 496)
    _minimize_slsqp_185953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 10), '_minimize_slsqp', False)
    # Calling _minimize_slsqp(args, kwargs) (line 496)
    _minimize_slsqp_call_result_185967 = invoke(stypy.reporting.localization.Localization(__file__, 496, 10), _minimize_slsqp_185953, *[fun_185954, array_call_result_185960], **kwargs_185966)
    
    # Assigning a type to the variable 'res' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'res', _minimize_slsqp_call_result_185967)
    
    # Call to print(...): (line 500)
    # Processing the call arguments (line 500)
    
    # Call to center(...): (line 500)
    # Processing the call arguments (line 500)
    int_185971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 57), 'int')
    str_185972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 61), 'str', '-')
    # Processing the call keyword arguments (line 500)
    kwargs_185973 = {}
    str_185969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 10), 'str', ' Equality and inequality constraints ')
    # Obtaining the member 'center' of a type (line 500)
    center_185970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 10), str_185969, 'center')
    # Calling center(args, kwargs) (line 500)
    center_call_result_185974 = invoke(stypy.reporting.localization.Localization(__file__, 500, 10), center_185970, *[int_185971, str_185972], **kwargs_185973)
    
    # Processing the call keyword arguments (line 500)
    kwargs_185975 = {}
    # Getting the type of 'print' (line 500)
    print_185968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'print', False)
    # Calling print(args, kwargs) (line 500)
    print_call_result_185976 = invoke(stypy.reporting.localization.Localization(__file__, 500, 4), print_185968, *[center_call_result_185974], **kwargs_185975)
    
    
    # Call to print(...): (line 501)
    # Processing the call arguments (line 501)
    str_185978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 10), 'str', ' * fmin_slsqp')
    # Processing the call keyword arguments (line 501)
    kwargs_185979 = {}
    # Getting the type of 'print' (line 501)
    print_185977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'print', False)
    # Calling print(args, kwargs) (line 501)
    print_call_result_185980 = invoke(stypy.reporting.localization.Localization(__file__, 501, 4), print_185977, *[str_185978], **kwargs_185979)
    
    
    # Assigning a Subscript to a Tuple (line 502):
    
    # Assigning a Subscript to a Name (line 502):
    
    # Obtaining the type of the subscript
    int_185981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 4), 'int')
    
    # Obtaining the type of the subscript
    int_185982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 49), 'int')
    slice_185983 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 502, 11), None, int_185982, None)
    
    # Call to fmin_slsqp(...): (line 502)
    # Processing the call arguments (line 502)
    # Getting the type of 'fun' (line 502)
    fun_185985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 22), 'fun', False)
    
    # Call to array(...): (line 502)
    # Processing the call arguments (line 502)
    
    # Obtaining an instance of the builtin type 'list' (line 502)
    list_185987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 502)
    # Adding element type (line 502)
    int_185988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 33), list_185987, int_185988)
    # Adding element type (line 502)
    int_185989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 33), list_185987, int_185989)
    
    # Processing the call keyword arguments (line 502)
    kwargs_185990 = {}
    # Getting the type of 'array' (line 502)
    array_185986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 27), 'array', False)
    # Calling array(args, kwargs) (line 502)
    array_call_result_185991 = invoke(stypy.reporting.localization.Localization(__file__, 502, 27), array_185986, *[list_185987], **kwargs_185990)
    
    # Processing the call keyword arguments (line 502)
    # Getting the type of 'feqcon' (line 503)
    feqcon_185992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 31), 'feqcon', False)
    keyword_185993 = feqcon_185992
    # Getting the type of 'jeqcon' (line 503)
    jeqcon_185994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 53), 'jeqcon', False)
    keyword_185995 = jeqcon_185994
    # Getting the type of 'fieqcon' (line 504)
    fieqcon_185996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 32), 'fieqcon', False)
    keyword_185997 = fieqcon_185996
    # Getting the type of 'jieqcon' (line 504)
    jieqcon_185998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 56), 'jieqcon', False)
    keyword_185999 = jieqcon_185998
    int_186000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 27), 'int')
    keyword_186001 = int_186000
    # Getting the type of 'True' (line 505)
    True_186002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 42), 'True', False)
    keyword_186003 = True_186002
    kwargs_186004 = {'disp': keyword_186001, 'full_output': keyword_186003, 'fprime_ieqcons': keyword_185999, 'f_ieqcons': keyword_185997, 'f_eqcons': keyword_185993, 'fprime_eqcons': keyword_185995}
    # Getting the type of 'fmin_slsqp' (line 502)
    fmin_slsqp_185984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 11), 'fmin_slsqp', False)
    # Calling fmin_slsqp(args, kwargs) (line 502)
    fmin_slsqp_call_result_186005 = invoke(stypy.reporting.localization.Localization(__file__, 502, 11), fmin_slsqp_185984, *[fun_185985, array_call_result_185991], **kwargs_186004)
    
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___186006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 11), fmin_slsqp_call_result_186005, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_186007 = invoke(stypy.reporting.localization.Localization(__file__, 502, 11), getitem___186006, slice_185983)
    
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___186008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 4), subscript_call_result_186007, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_186009 = invoke(stypy.reporting.localization.Localization(__file__, 502, 4), getitem___186008, int_185981)
    
    # Assigning a type to the variable 'tuple_var_assignment_184550' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'tuple_var_assignment_184550', subscript_call_result_186009)
    
    # Assigning a Subscript to a Name (line 502):
    
    # Obtaining the type of the subscript
    int_186010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 4), 'int')
    
    # Obtaining the type of the subscript
    int_186011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 49), 'int')
    slice_186012 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 502, 11), None, int_186011, None)
    
    # Call to fmin_slsqp(...): (line 502)
    # Processing the call arguments (line 502)
    # Getting the type of 'fun' (line 502)
    fun_186014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 22), 'fun', False)
    
    # Call to array(...): (line 502)
    # Processing the call arguments (line 502)
    
    # Obtaining an instance of the builtin type 'list' (line 502)
    list_186016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 502)
    # Adding element type (line 502)
    int_186017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 33), list_186016, int_186017)
    # Adding element type (line 502)
    int_186018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 33), list_186016, int_186018)
    
    # Processing the call keyword arguments (line 502)
    kwargs_186019 = {}
    # Getting the type of 'array' (line 502)
    array_186015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 27), 'array', False)
    # Calling array(args, kwargs) (line 502)
    array_call_result_186020 = invoke(stypy.reporting.localization.Localization(__file__, 502, 27), array_186015, *[list_186016], **kwargs_186019)
    
    # Processing the call keyword arguments (line 502)
    # Getting the type of 'feqcon' (line 503)
    feqcon_186021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 31), 'feqcon', False)
    keyword_186022 = feqcon_186021
    # Getting the type of 'jeqcon' (line 503)
    jeqcon_186023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 53), 'jeqcon', False)
    keyword_186024 = jeqcon_186023
    # Getting the type of 'fieqcon' (line 504)
    fieqcon_186025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 32), 'fieqcon', False)
    keyword_186026 = fieqcon_186025
    # Getting the type of 'jieqcon' (line 504)
    jieqcon_186027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 56), 'jieqcon', False)
    keyword_186028 = jieqcon_186027
    int_186029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 27), 'int')
    keyword_186030 = int_186029
    # Getting the type of 'True' (line 505)
    True_186031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 42), 'True', False)
    keyword_186032 = True_186031
    kwargs_186033 = {'disp': keyword_186030, 'full_output': keyword_186032, 'fprime_ieqcons': keyword_186028, 'f_ieqcons': keyword_186026, 'f_eqcons': keyword_186022, 'fprime_eqcons': keyword_186024}
    # Getting the type of 'fmin_slsqp' (line 502)
    fmin_slsqp_186013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 11), 'fmin_slsqp', False)
    # Calling fmin_slsqp(args, kwargs) (line 502)
    fmin_slsqp_call_result_186034 = invoke(stypy.reporting.localization.Localization(__file__, 502, 11), fmin_slsqp_186013, *[fun_186014, array_call_result_186020], **kwargs_186033)
    
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___186035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 11), fmin_slsqp_call_result_186034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_186036 = invoke(stypy.reporting.localization.Localization(__file__, 502, 11), getitem___186035, slice_186012)
    
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___186037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 4), subscript_call_result_186036, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_186038 = invoke(stypy.reporting.localization.Localization(__file__, 502, 4), getitem___186037, int_186010)
    
    # Assigning a type to the variable 'tuple_var_assignment_184551' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'tuple_var_assignment_184551', subscript_call_result_186038)
    
    # Assigning a Name to a Name (line 502):
    # Getting the type of 'tuple_var_assignment_184550' (line 502)
    tuple_var_assignment_184550_186039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'tuple_var_assignment_184550')
    # Assigning a type to the variable 'x' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'x', tuple_var_assignment_184550_186039)
    
    # Assigning a Name to a Name (line 502):
    # Getting the type of 'tuple_var_assignment_184551' (line 502)
    tuple_var_assignment_184551_186040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'tuple_var_assignment_184551')
    # Assigning a type to the variable 'f' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 7), 'f', tuple_var_assignment_184551_186040)
    
    # Call to print(...): (line 506)
    # Processing the call arguments (line 506)
    str_186042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 10), 'str', ' * _minimize_slsqp')
    # Processing the call keyword arguments (line 506)
    kwargs_186043 = {}
    # Getting the type of 'print' (line 506)
    print_186041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'print', False)
    # Calling print(args, kwargs) (line 506)
    print_call_result_186044 = invoke(stypy.reporting.localization.Localization(__file__, 506, 4), print_186041, *[str_186042], **kwargs_186043)
    
    
    # Assigning a Call to a Name (line 507):
    
    # Assigning a Call to a Name (line 507):
    
    # Call to _minimize_slsqp(...): (line 507)
    # Processing the call arguments (line 507)
    # Getting the type of 'fun' (line 507)
    fun_186046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 26), 'fun', False)
    
    # Call to array(...): (line 507)
    # Processing the call arguments (line 507)
    
    # Obtaining an instance of the builtin type 'list' (line 507)
    list_186048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 507)
    # Adding element type (line 507)
    int_186049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 37), list_186048, int_186049)
    # Adding element type (line 507)
    int_186050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 37), list_186048, int_186050)
    
    # Processing the call keyword arguments (line 507)
    kwargs_186051 = {}
    # Getting the type of 'array' (line 507)
    array_186047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 31), 'array', False)
    # Calling array(args, kwargs) (line 507)
    array_call_result_186052 = invoke(stypy.reporting.localization.Localization(__file__, 507, 31), array_186047, *[list_186048], **kwargs_186051)
    
    # Processing the call keyword arguments (line 507)
    # Getting the type of 'cons' (line 507)
    cons_186053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 59), 'cons', False)
    keyword_186054 = cons_186053
    
    # Obtaining an instance of the builtin type 'dict' (line 508)
    dict_186055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 28), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 508)
    # Adding element type (key, value) (line 508)
    str_186056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 29), 'str', 'disp')
    # Getting the type of 'True' (line 508)
    True_186057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 37), 'True', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 28), dict_186055, (str_186056, True_186057))
    
    kwargs_186058 = {'dict_186055': dict_186055, 'constraints': keyword_186054}
    # Getting the type of '_minimize_slsqp' (line 507)
    _minimize_slsqp_186045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 10), '_minimize_slsqp', False)
    # Calling _minimize_slsqp(args, kwargs) (line 507)
    _minimize_slsqp_call_result_186059 = invoke(stypy.reporting.localization.Localization(__file__, 507, 10), _minimize_slsqp_186045, *[fun_186046, array_call_result_186052], **kwargs_186058)
    
    # Assigning a type to the variable 'res' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'res', _minimize_slsqp_call_result_186059)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
