
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Copyright (C) 2010 David Fong and Michael Saunders
3: 
4: LSMR uses an iterative method.
5: 
6: 07 Jun 2010: Documentation updated
7: 03 Jun 2010: First release version in Python
8: 
9: David Chin-lung Fong            clfong@stanford.edu
10: Institute for Computational and Mathematical Engineering
11: Stanford University
12: 
13: Michael Saunders                saunders@stanford.edu
14: Systems Optimization Laboratory
15: Dept of MS&E, Stanford University.
16: 
17: '''
18: 
19: from __future__ import division, print_function, absolute_import
20: 
21: __all__ = ['lsmr']
22: 
23: from numpy import zeros, infty, atleast_1d
24: from numpy.linalg import norm
25: from math import sqrt
26: from scipy.sparse.linalg.interface import aslinearoperator
27: 
28: from .lsqr import _sym_ortho
29: 
30: 
31: def lsmr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
32:          maxiter=None, show=False, x0=None):
33:     '''Iterative solver for least-squares problems.
34: 
35:     lsmr solves the system of linear equations ``Ax = b``. If the system
36:     is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
37:     A is a rectangular matrix of dimension m-by-n, where all cases are
38:     allowed: m = n, m > n, or m < n. B is a vector of length m.
39:     The matrix A may be dense or sparse (usually sparse).
40: 
41:     Parameters
42:     ----------
43:     A : {matrix, sparse matrix, ndarray, LinearOperator}
44:         Matrix A in the linear system.
45:     b : array_like, shape (m,)
46:         Vector b in the linear system.
47:     damp : float
48:         Damping factor for regularized least-squares. `lsmr` solves
49:         the regularized least-squares problem::
50: 
51:          min ||(b) - (  A   )x||
52:              ||(0)   (damp*I) ||_2
53: 
54:         where damp is a scalar.  If damp is None or 0, the system
55:         is solved without regularization.
56:     atol, btol : float, optional
57:         Stopping tolerances. `lsmr` continues iterations until a
58:         certain backward error estimate is smaller than some quantity
59:         depending on atol and btol.  Let ``r = b - Ax`` be the
60:         residual vector for the current approximate solution ``x``.
61:         If ``Ax = b`` seems to be consistent, ``lsmr`` terminates
62:         when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
63:         Otherwise, lsmr terminates when ``norm(A^{T} r) <=
64:         atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (say),
65:         the final ``norm(r)`` should be accurate to about 6
66:         digits. (The final x will usually have fewer correct digits,
67:         depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
68:         or `btol` is None, a default value of 1.0e-6 will be used.
69:         Ideally, they should be estimates of the relative error in the
70:         entries of A and B respectively.  For example, if the entries
71:         of `A` have 7 correct digits, set atol = 1e-7. This prevents
72:         the algorithm from doing unnecessary work beyond the
73:         uncertainty of the input data.
74:     conlim : float, optional
75:         `lsmr` terminates if an estimate of ``cond(A)`` exceeds
76:         `conlim`.  For compatible systems ``Ax = b``, conlim could be
77:         as large as 1.0e+12 (say).  For least-squares problems,
78:         `conlim` should be less than 1.0e+8. If `conlim` is None, the
79:         default value is 1e+8.  Maximum precision can be obtained by
80:         setting ``atol = btol = conlim = 0``, but the number of
81:         iterations may then be excessive.
82:     maxiter : int, optional
83:         `lsmr` terminates if the number of iterations reaches
84:         `maxiter`.  The default is ``maxiter = min(m, n)``.  For
85:         ill-conditioned systems, a larger value of `maxiter` may be
86:         needed.
87:     show : bool, optional
88:         Print iterations logs if ``show=True``.
89:     x0 : array_like, shape (n,), optional
90:         Initial guess of x, if None zeros are used.
91: 
92:         .. versionadded:: 1.0.0
93:     Returns
94:     -------
95:     x : ndarray of float
96:         Least-square solution returned.
97:     istop : int
98:         istop gives the reason for stopping::
99: 
100:           istop   = 0 means x=0 is a solution.  If x0 was given, then x=x0 is a
101:                       solution.
102:                   = 1 means x is an approximate solution to A*x = B,
103:                       according to atol and btol.
104:                   = 2 means x approximately solves the least-squares problem
105:                       according to atol.
106:                   = 3 means COND(A) seems to be greater than CONLIM.
107:                   = 4 is the same as 1 with atol = btol = eps (machine
108:                       precision)
109:                   = 5 is the same as 2 with atol = eps.
110:                   = 6 is the same as 3 with CONLIM = 1/eps.
111:                   = 7 means ITN reached maxiter before the other stopping
112:                       conditions were satisfied.
113: 
114:     itn : int
115:         Number of iterations used.
116:     normr : float
117:         ``norm(b-Ax)``
118:     normar : float
119:         ``norm(A^T (b - Ax))``
120:     norma : float
121:         ``norm(A)``
122:     conda : float
123:         Condition number of A.
124:     normx : float
125:         ``norm(x)``
126: 
127:     Notes
128:     -----
129: 
130:     .. versionadded:: 0.11.0
131: 
132:     References
133:     ----------
134:     .. [1] D. C.-L. Fong and M. A. Saunders,
135:            "LSMR: An iterative algorithm for sparse least-squares problems",
136:            SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
137:            http://arxiv.org/abs/1006.0758
138:     .. [2] LSMR Software, http://web.stanford.edu/group/SOL/software/lsmr/
139: 
140:     Examples
141:     --------
142:     >>> from scipy.sparse import csc_matrix
143:     >>> from scipy.sparse.linalg import lsmr
144:     >>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)
145: 
146:     The first example has the trivial solution `[0, 0]`
147: 
148:     >>> b = np.array([0., 0., 0.], dtype=float)
149:     >>> x, istop, itn, normr = lsmr(A, b)[:4]
150:     >>> istop
151:     0
152:     >>> x
153:     array([ 0.,  0.])
154: 
155:     The stopping code `istop=0` returned indicates that a vector of zeros was
156:     found as a solution. The returned solution `x` indeed contains `[0., 0.]`.
157:     The next example has a non-trivial solution:
158: 
159:     >>> b = np.array([1., 0., -1.], dtype=float)
160:     >>> x, istop, itn, normr = lsmr(A, b)[:4]
161:     >>> istop
162:     1
163:     >>> x
164:     array([ 1., -1.])
165:     >>> itn
166:     1
167:     >>> normr
168:     4.440892098500627e-16
169: 
170:     As indicated by `istop=1`, `lsmr` found a solution obeying the tolerance
171:     limits. The given solution `[1., -1.]` obviously solves the equation. The
172:     remaining return values include information about the number of iterations
173:     (`itn=1`) and the remaining difference of left and right side of the solved
174:     equation.
175:     The final example demonstrates the behavior in the case where there is no
176:     solution for the equation:
177: 
178:     >>> b = np.array([1., 0.01, -1.], dtype=float)
179:     >>> x, istop, itn, normr = lsmr(A, b)[:4]
180:     >>> istop
181:     2
182:     >>> x
183:     array([ 1.00333333, -0.99666667])
184:     >>> A.dot(x)-b
185:     array([ 0.00333333, -0.00333333,  0.00333333])
186:     >>> normr
187:     0.005773502691896255
188: 
189:     `istop` indicates that the system is inconsistent and thus `x` is rather an
190:     approximate solution to the corresponding least-squares problem. `normr`
191:     contains the minimal distance that was found.
192:     '''
193: 
194:     A = aslinearoperator(A)
195:     b = atleast_1d(b)
196:     if b.ndim > 1:
197:         b = b.squeeze()
198: 
199:     msg = ('The exact solution is x = 0, or x = x0, if x0 was given  ',
200:          'Ax - b is small enough, given atol, btol                  ',
201:          'The least-squares solution is good enough, given atol     ',
202:          'The estimate of cond(Abar) has exceeded conlim            ',
203:          'Ax - b is small enough for this machine                   ',
204:          'The least-squares solution is good enough for this machine',
205:          'Cond(Abar) seems to be too large for this machine         ',
206:          'The iteration limit has been reached                      ')
207: 
208:     hdg1 = '   itn      x(1)       norm r    norm A''r'
209:     hdg2 = ' compatible   LS      norm A   cond A'
210:     pfreq = 20   # print frequency (for repeating the heading)
211:     pcount = 0   # print counter
212: 
213:     m, n = A.shape
214: 
215:     # stores the num of singular values
216:     minDim = min([m, n])
217: 
218:     if maxiter is None:
219:         maxiter = minDim
220: 
221:     if show:
222:         print(' ')
223:         print('LSMR            Least-squares solution of  Ax = b\n')
224:         print('The matrix A has %8g rows  and %8g cols' % (m, n))
225:         print('damp = %20.14e\n' % (damp))
226:         print('atol = %8.2e                 conlim = %8.2e\n' % (atol, conlim))
227:         print('btol = %8.2e             maxiter = %8g\n' % (btol, maxiter))
228: 
229:     u = b
230:     normb = norm(b)
231:     if x0 is None:
232:         x = zeros(n)
233:         beta = normb.copy()
234:     else:
235:         x = atleast_1d(x0)
236:         u = u - A.matvec(x)
237:         beta = norm(u)
238: 
239:     if beta > 0:
240:         u = (1 / beta) * u
241:         v = A.rmatvec(u)
242:         alpha = norm(v)
243:     else:
244:         v = zeros(n)
245:         alpha = 0
246: 
247:     if alpha > 0:
248:         v = (1 / alpha) * v
249: 
250:     # Initialize variables for 1st iteration.
251: 
252:     itn = 0
253:     zetabar = alpha * beta
254:     alphabar = alpha
255:     rho = 1
256:     rhobar = 1
257:     cbar = 1
258:     sbar = 0
259: 
260:     h = v.copy()
261:     hbar = zeros(n)
262: 
263:     # Initialize variables for estimation of ||r||.
264: 
265:     betadd = beta
266:     betad = 0
267:     rhodold = 1
268:     tautildeold = 0
269:     thetatilde = 0
270:     zeta = 0
271:     d = 0
272: 
273:     # Initialize variables for estimation of ||A|| and cond(A)
274: 
275:     normA2 = alpha * alpha
276:     maxrbar = 0
277:     minrbar = 1e+100
278:     normA = sqrt(normA2)
279:     condA = 1
280:     normx = 0
281: 
282:     # Items for use in stopping rules, normb set earlier
283:     istop = 0
284:     ctol = 0
285:     if conlim > 0:
286:         ctol = 1 / conlim
287:     normr = beta
288: 
289:     # Reverse the order here from the original matlab code because
290:     # there was an error on return when arnorm==0
291:     normar = alpha * beta
292:     if normar == 0:
293:         if show:
294:             print(msg[0])
295:         return x, istop, itn, normr, normar, normA, condA, normx
296: 
297:     if show:
298:         print(' ')
299:         print(hdg1, hdg2)
300:         test1 = 1
301:         test2 = alpha / beta
302:         str1 = '%6g %12.5e' % (itn, x[0])
303:         str2 = ' %10.3e %10.3e' % (normr, normar)
304:         str3 = '  %8.1e %8.1e' % (test1, test2)
305:         print(''.join([str1, str2, str3]))
306: 
307:     # Main iteration loop.
308:     while itn < maxiter:
309:         itn = itn + 1
310: 
311:         # Perform the next step of the bidiagonalization to obtain the
312:         # next  beta, u, alpha, v.  These satisfy the relations
313:         #         beta*u  =  a*v   -  alpha*u,
314:         #        alpha*v  =  A'*u  -  beta*v.
315: 
316:         u = A.matvec(v) - alpha * u
317:         beta = norm(u)
318: 
319:         if beta > 0:
320:             u = (1 / beta) * u
321:             v = A.rmatvec(u) - beta * v
322:             alpha = norm(v)
323:             if alpha > 0:
324:                 v = (1 / alpha) * v
325: 
326:         # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.
327: 
328:         # Construct rotation Qhat_{k,2k+1}.
329: 
330:         chat, shat, alphahat = _sym_ortho(alphabar, damp)
331: 
332:         # Use a plane rotation (Q_i) to turn B_i to R_i
333: 
334:         rhoold = rho
335:         c, s, rho = _sym_ortho(alphahat, beta)
336:         thetanew = s*alpha
337:         alphabar = c*alpha
338: 
339:         # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
340: 
341:         rhobarold = rhobar
342:         zetaold = zeta
343:         thetabar = sbar * rho
344:         rhotemp = cbar * rho
345:         cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
346:         zeta = cbar * zetabar
347:         zetabar = - sbar * zetabar
348: 
349:         # Update h, h_hat, x.
350: 
351:         hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar
352:         x = x + (zeta / (rho * rhobar)) * hbar
353:         h = v - (thetanew / rho) * h
354: 
355:         # Estimate of ||r||.
356: 
357:         # Apply rotation Qhat_{k,2k+1}.
358:         betaacute = chat * betadd
359:         betacheck = -shat * betadd
360: 
361:         # Apply rotation Q_{k,k+1}.
362:         betahat = c * betaacute
363:         betadd = -s * betaacute
364: 
365:         # Apply rotation Qtilde_{k-1}.
366:         # betad = betad_{k-1} here.
367: 
368:         thetatildeold = thetatilde
369:         ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
370:         thetatilde = stildeold * rhobar
371:         rhodold = ctildeold * rhobar
372:         betad = - stildeold * betad + ctildeold * betahat
373: 
374:         # betad   = betad_k here.
375:         # rhodold = rhod_k  here.
376: 
377:         tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
378:         taud = (zeta - thetatilde * tautildeold) / rhodold
379:         d = d + betacheck * betacheck
380:         normr = sqrt(d + (betad - taud)**2 + betadd * betadd)
381: 
382:         # Estimate ||A||.
383:         normA2 = normA2 + beta * beta
384:         normA = sqrt(normA2)
385:         normA2 = normA2 + alpha * alpha
386: 
387:         # Estimate cond(A).
388:         maxrbar = max(maxrbar, rhobarold)
389:         if itn > 1:
390:             minrbar = min(minrbar, rhobarold)
391:         condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
392: 
393:         # Test for convergence.
394: 
395:         # Compute norms for convergence testing.
396:         normar = abs(zetabar)
397:         normx = norm(x)
398: 
399:         # Now use these norms to estimate certain other quantities,
400:         # some of which will be small near a solution.
401: 
402:         test1 = normr / normb
403:         if (normA * normr) != 0:
404:             test2 = normar / (normA * normr)
405:         else:
406:             test2 = infty
407:         test3 = 1 / condA
408:         t1 = test1 / (1 + normA * normx / normb)
409:         rtol = btol + atol * normA * normx / normb
410: 
411:         # The following tests guard against extremely small values of
412:         # atol, btol or ctol.  (The user may have set any or all of
413:         # the parameters atol, btol, conlim  to 0.)
414:         # The effect is equivalent to the normAl tests using
415:         # atol = eps,  btol = eps,  conlim = 1/eps.
416: 
417:         if itn >= maxiter:
418:             istop = 7
419:         if 1 + test3 <= 1:
420:             istop = 6
421:         if 1 + test2 <= 1:
422:             istop = 5
423:         if 1 + t1 <= 1:
424:             istop = 4
425: 
426:         # Allow for tolerances set by the user.
427: 
428:         if test3 <= ctol:
429:             istop = 3
430:         if test2 <= atol:
431:             istop = 2
432:         if test1 <= rtol:
433:             istop = 1
434: 
435:         # See if it is time to print something.
436: 
437:         if show:
438:             if (n <= 40) or (itn <= 10) or (itn >= maxiter - 10) or \
439:                (itn % 10 == 0) or (test3 <= 1.1 * ctol) or \
440:                (test2 <= 1.1 * atol) or (test1 <= 1.1 * rtol) or \
441:                (istop != 0):
442: 
443:                 if pcount >= pfreq:
444:                     pcount = 0
445:                     print(' ')
446:                     print(hdg1, hdg2)
447:                 pcount = pcount + 1
448:                 str1 = '%6g %12.5e' % (itn, x[0])
449:                 str2 = ' %10.3e %10.3e' % (normr, normar)
450:                 str3 = '  %8.1e %8.1e' % (test1, test2)
451:                 str4 = ' %8.1e %8.1e' % (normA, condA)
452:                 print(''.join([str1, str2, str3, str4]))
453: 
454:         if istop > 0:
455:             break
456: 
457:     # Print the stopping condition.
458: 
459:     if show:
460:         print(' ')
461:         print('LSMR finished')
462:         print(msg[istop])
463:         print('istop =%8g    normr =%8.1e' % (istop, normr))
464:         print('    normA =%8.1e    normAr =%8.1e' % (normA, normar))
465:         print('itn   =%8g    condA =%8.1e' % (itn, condA))
466:         print('    normx =%8.1e' % (normx))
467:         print(str1, str2)
468:         print(str3, str4)
469: 
470:     return x, istop, itn, normr, normar, normA, condA, normx
471: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_411617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\nCopyright (C) 2010 David Fong and Michael Saunders\n\nLSMR uses an iterative method.\n\n07 Jun 2010: Documentation updated\n03 Jun 2010: First release version in Python\n\nDavid Chin-lung Fong            clfong@stanford.edu\nInstitute for Computational and Mathematical Engineering\nStanford University\n\nMichael Saunders                saunders@stanford.edu\nSystems Optimization Laboratory\nDept of MS&E, Stanford University.\n\n')

# Assigning a List to a Name (line 21):

# Assigning a List to a Name (line 21):
__all__ = ['lsmr']
module_type_store.set_exportable_members(['lsmr'])

# Obtaining an instance of the builtin type 'list' (line 21)
list_411618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_411619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'str', 'lsmr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_411618, str_411619)

# Assigning a type to the variable '__all__' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '__all__', list_411618)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy import zeros, infty, atleast_1d' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_411620 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy')

if (type(import_411620) is not StypyTypeError):

    if (import_411620 != 'pyd_module'):
        __import__(import_411620)
        sys_modules_411621 = sys.modules[import_411620]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy', sys_modules_411621.module_type_store, module_type_store, ['zeros', 'infty', 'atleast_1d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_411621, sys_modules_411621.module_type_store, module_type_store)
    else:
        from numpy import zeros, infty, atleast_1d

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy', None, module_type_store, ['zeros', 'infty', 'atleast_1d'], [zeros, infty, atleast_1d])

else:
    # Assigning a type to the variable 'numpy' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy', import_411620)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.linalg import norm' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_411622 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.linalg')

if (type(import_411622) is not StypyTypeError):

    if (import_411622 != 'pyd_module'):
        __import__(import_411622)
        sys_modules_411623 = sys.modules[import_411622]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.linalg', sys_modules_411623.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_411623, sys_modules_411623.module_type_store, module_type_store)
    else:
        from numpy.linalg import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.linalg', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.linalg', import_411622)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from math import sqrt' statement (line 25)
try:
    from math import sqrt

except:
    sqrt = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'math', None, module_type_store, ['sqrt'], [sqrt])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from scipy.sparse.linalg.interface import aslinearoperator' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_411624 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse.linalg.interface')

if (type(import_411624) is not StypyTypeError):

    if (import_411624 != 'pyd_module'):
        __import__(import_411624)
        sys_modules_411625 = sys.modules[import_411624]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse.linalg.interface', sys_modules_411625.module_type_store, module_type_store, ['aslinearoperator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_411625, sys_modules_411625.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.interface import aslinearoperator

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse.linalg.interface', None, module_type_store, ['aslinearoperator'], [aslinearoperator])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.interface' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse.linalg.interface', import_411624)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from scipy.sparse.linalg.isolve.lsqr import _sym_ortho' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_411626 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg.isolve.lsqr')

if (type(import_411626) is not StypyTypeError):

    if (import_411626 != 'pyd_module'):
        __import__(import_411626)
        sys_modules_411627 = sys.modules[import_411626]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg.isolve.lsqr', sys_modules_411627.module_type_store, module_type_store, ['_sym_ortho'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_411627, sys_modules_411627.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.lsqr import _sym_ortho

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg.isolve.lsqr', None, module_type_store, ['_sym_ortho'], [_sym_ortho])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.lsqr' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.sparse.linalg.isolve.lsqr', import_411626)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')


@norecursion
def lsmr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_411628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'float')
    float_411629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 30), 'float')
    float_411630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 41), 'float')
    float_411631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 54), 'float')
    # Getting the type of 'None' (line 32)
    None_411632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'None')
    # Getting the type of 'False' (line 32)
    False_411633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 28), 'False')
    # Getting the type of 'None' (line 32)
    None_411634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 38), 'None')
    defaults = [float_411628, float_411629, float_411630, float_411631, None_411632, False_411633, None_411634]
    # Create a new context for function 'lsmr'
    module_type_store = module_type_store.open_function_context('lsmr', 31, 0, False)
    
    # Passed parameters checking function
    lsmr.stypy_localization = localization
    lsmr.stypy_type_of_self = None
    lsmr.stypy_type_store = module_type_store
    lsmr.stypy_function_name = 'lsmr'
    lsmr.stypy_param_names_list = ['A', 'b', 'damp', 'atol', 'btol', 'conlim', 'maxiter', 'show', 'x0']
    lsmr.stypy_varargs_param_name = None
    lsmr.stypy_kwargs_param_name = None
    lsmr.stypy_call_defaults = defaults
    lsmr.stypy_call_varargs = varargs
    lsmr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lsmr', ['A', 'b', 'damp', 'atol', 'btol', 'conlim', 'maxiter', 'show', 'x0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lsmr', localization, ['A', 'b', 'damp', 'atol', 'btol', 'conlim', 'maxiter', 'show', 'x0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lsmr(...)' code ##################

    str_411635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, (-1)), 'str', 'Iterative solver for least-squares problems.\n\n    lsmr solves the system of linear equations ``Ax = b``. If the system\n    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.\n    A is a rectangular matrix of dimension m-by-n, where all cases are\n    allowed: m = n, m > n, or m < n. B is a vector of length m.\n    The matrix A may be dense or sparse (usually sparse).\n\n    Parameters\n    ----------\n    A : {matrix, sparse matrix, ndarray, LinearOperator}\n        Matrix A in the linear system.\n    b : array_like, shape (m,)\n        Vector b in the linear system.\n    damp : float\n        Damping factor for regularized least-squares. `lsmr` solves\n        the regularized least-squares problem::\n\n         min ||(b) - (  A   )x||\n             ||(0)   (damp*I) ||_2\n\n        where damp is a scalar.  If damp is None or 0, the system\n        is solved without regularization.\n    atol, btol : float, optional\n        Stopping tolerances. `lsmr` continues iterations until a\n        certain backward error estimate is smaller than some quantity\n        depending on atol and btol.  Let ``r = b - Ax`` be the\n        residual vector for the current approximate solution ``x``.\n        If ``Ax = b`` seems to be consistent, ``lsmr`` terminates\n        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.\n        Otherwise, lsmr terminates when ``norm(A^{T} r) <=\n        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (say),\n        the final ``norm(r)`` should be accurate to about 6\n        digits. (The final x will usually have fewer correct digits,\n        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`\n        or `btol` is None, a default value of 1.0e-6 will be used.\n        Ideally, they should be estimates of the relative error in the\n        entries of A and B respectively.  For example, if the entries\n        of `A` have 7 correct digits, set atol = 1e-7. This prevents\n        the algorithm from doing unnecessary work beyond the\n        uncertainty of the input data.\n    conlim : float, optional\n        `lsmr` terminates if an estimate of ``cond(A)`` exceeds\n        `conlim`.  For compatible systems ``Ax = b``, conlim could be\n        as large as 1.0e+12 (say).  For least-squares problems,\n        `conlim` should be less than 1.0e+8. If `conlim` is None, the\n        default value is 1e+8.  Maximum precision can be obtained by\n        setting ``atol = btol = conlim = 0``, but the number of\n        iterations may then be excessive.\n    maxiter : int, optional\n        `lsmr` terminates if the number of iterations reaches\n        `maxiter`.  The default is ``maxiter = min(m, n)``.  For\n        ill-conditioned systems, a larger value of `maxiter` may be\n        needed.\n    show : bool, optional\n        Print iterations logs if ``show=True``.\n    x0 : array_like, shape (n,), optional\n        Initial guess of x, if None zeros are used.\n\n        .. versionadded:: 1.0.0\n    Returns\n    -------\n    x : ndarray of float\n        Least-square solution returned.\n    istop : int\n        istop gives the reason for stopping::\n\n          istop   = 0 means x=0 is a solution.  If x0 was given, then x=x0 is a\n                      solution.\n                  = 1 means x is an approximate solution to A*x = B,\n                      according to atol and btol.\n                  = 2 means x approximately solves the least-squares problem\n                      according to atol.\n                  = 3 means COND(A) seems to be greater than CONLIM.\n                  = 4 is the same as 1 with atol = btol = eps (machine\n                      precision)\n                  = 5 is the same as 2 with atol = eps.\n                  = 6 is the same as 3 with CONLIM = 1/eps.\n                  = 7 means ITN reached maxiter before the other stopping\n                      conditions were satisfied.\n\n    itn : int\n        Number of iterations used.\n    normr : float\n        ``norm(b-Ax)``\n    normar : float\n        ``norm(A^T (b - Ax))``\n    norma : float\n        ``norm(A)``\n    conda : float\n        Condition number of A.\n    normx : float\n        ``norm(x)``\n\n    Notes\n    -----\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1] D. C.-L. Fong and M. A. Saunders,\n           "LSMR: An iterative algorithm for sparse least-squares problems",\n           SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.\n           http://arxiv.org/abs/1006.0758\n    .. [2] LSMR Software, http://web.stanford.edu/group/SOL/software/lsmr/\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import lsmr\n    >>> A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)\n\n    The first example has the trivial solution `[0, 0]`\n\n    >>> b = np.array([0., 0., 0.], dtype=float)\n    >>> x, istop, itn, normr = lsmr(A, b)[:4]\n    >>> istop\n    0\n    >>> x\n    array([ 0.,  0.])\n\n    The stopping code `istop=0` returned indicates that a vector of zeros was\n    found as a solution. The returned solution `x` indeed contains `[0., 0.]`.\n    The next example has a non-trivial solution:\n\n    >>> b = np.array([1., 0., -1.], dtype=float)\n    >>> x, istop, itn, normr = lsmr(A, b)[:4]\n    >>> istop\n    1\n    >>> x\n    array([ 1., -1.])\n    >>> itn\n    1\n    >>> normr\n    4.440892098500627e-16\n\n    As indicated by `istop=1`, `lsmr` found a solution obeying the tolerance\n    limits. The given solution `[1., -1.]` obviously solves the equation. The\n    remaining return values include information about the number of iterations\n    (`itn=1`) and the remaining difference of left and right side of the solved\n    equation.\n    The final example demonstrates the behavior in the case where there is no\n    solution for the equation:\n\n    >>> b = np.array([1., 0.01, -1.], dtype=float)\n    >>> x, istop, itn, normr = lsmr(A, b)[:4]\n    >>> istop\n    2\n    >>> x\n    array([ 1.00333333, -0.99666667])\n    >>> A.dot(x)-b\n    array([ 0.00333333, -0.00333333,  0.00333333])\n    >>> normr\n    0.005773502691896255\n\n    `istop` indicates that the system is inconsistent and thus `x` is rather an\n    approximate solution to the corresponding least-squares problem. `normr`\n    contains the minimal distance that was found.\n    ')
    
    # Assigning a Call to a Name (line 194):
    
    # Assigning a Call to a Name (line 194):
    
    # Call to aslinearoperator(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'A' (line 194)
    A_411637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 25), 'A', False)
    # Processing the call keyword arguments (line 194)
    kwargs_411638 = {}
    # Getting the type of 'aslinearoperator' (line 194)
    aslinearoperator_411636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 194)
    aslinearoperator_call_result_411639 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), aslinearoperator_411636, *[A_411637], **kwargs_411638)
    
    # Assigning a type to the variable 'A' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'A', aslinearoperator_call_result_411639)
    
    # Assigning a Call to a Name (line 195):
    
    # Assigning a Call to a Name (line 195):
    
    # Call to atleast_1d(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'b' (line 195)
    b_411641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'b', False)
    # Processing the call keyword arguments (line 195)
    kwargs_411642 = {}
    # Getting the type of 'atleast_1d' (line 195)
    atleast_1d_411640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 195)
    atleast_1d_call_result_411643 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), atleast_1d_411640, *[b_411641], **kwargs_411642)
    
    # Assigning a type to the variable 'b' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'b', atleast_1d_call_result_411643)
    
    
    # Getting the type of 'b' (line 196)
    b_411644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 7), 'b')
    # Obtaining the member 'ndim' of a type (line 196)
    ndim_411645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 7), b_411644, 'ndim')
    int_411646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 16), 'int')
    # Applying the binary operator '>' (line 196)
    result_gt_411647 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 7), '>', ndim_411645, int_411646)
    
    # Testing the type of an if condition (line 196)
    if_condition_411648 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 4), result_gt_411647)
    # Assigning a type to the variable 'if_condition_411648' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'if_condition_411648', if_condition_411648)
    # SSA begins for if statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 197):
    
    # Assigning a Call to a Name (line 197):
    
    # Call to squeeze(...): (line 197)
    # Processing the call keyword arguments (line 197)
    kwargs_411651 = {}
    # Getting the type of 'b' (line 197)
    b_411649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'b', False)
    # Obtaining the member 'squeeze' of a type (line 197)
    squeeze_411650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), b_411649, 'squeeze')
    # Calling squeeze(args, kwargs) (line 197)
    squeeze_call_result_411652 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), squeeze_411650, *[], **kwargs_411651)
    
    # Assigning a type to the variable 'b' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'b', squeeze_call_result_411652)
    # SSA join for if statement (line 196)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Name (line 199):
    
    # Assigning a Tuple to a Name (line 199):
    
    # Obtaining an instance of the builtin type 'tuple' (line 199)
    tuple_411653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 199)
    # Adding element type (line 199)
    str_411654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 11), 'str', 'The exact solution is x = 0, or x = x0, if x0 was given  ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 11), tuple_411653, str_411654)
    # Adding element type (line 199)
    str_411655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 9), 'str', 'Ax - b is small enough, given atol, btol                  ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 11), tuple_411653, str_411655)
    # Adding element type (line 199)
    str_411656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 9), 'str', 'The least-squares solution is good enough, given atol     ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 11), tuple_411653, str_411656)
    # Adding element type (line 199)
    str_411657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 9), 'str', 'The estimate of cond(Abar) has exceeded conlim            ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 11), tuple_411653, str_411657)
    # Adding element type (line 199)
    str_411658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 9), 'str', 'Ax - b is small enough for this machine                   ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 11), tuple_411653, str_411658)
    # Adding element type (line 199)
    str_411659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 9), 'str', 'The least-squares solution is good enough for this machine')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 11), tuple_411653, str_411659)
    # Adding element type (line 199)
    str_411660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 9), 'str', 'Cond(Abar) seems to be too large for this machine         ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 11), tuple_411653, str_411660)
    # Adding element type (line 199)
    str_411661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 9), 'str', 'The iteration limit has been reached                      ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 11), tuple_411653, str_411661)
    
    # Assigning a type to the variable 'msg' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'msg', tuple_411653)
    
    # Assigning a Str to a Name (line 208):
    
    # Assigning a Str to a Name (line 208):
    str_411662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 11), 'str', '   itn      x(1)       norm r    norm Ar')
    # Assigning a type to the variable 'hdg1' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'hdg1', str_411662)
    
    # Assigning a Str to a Name (line 209):
    
    # Assigning a Str to a Name (line 209):
    str_411663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 11), 'str', ' compatible   LS      norm A   cond A')
    # Assigning a type to the variable 'hdg2' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'hdg2', str_411663)
    
    # Assigning a Num to a Name (line 210):
    
    # Assigning a Num to a Name (line 210):
    int_411664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 12), 'int')
    # Assigning a type to the variable 'pfreq' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'pfreq', int_411664)
    
    # Assigning a Num to a Name (line 211):
    
    # Assigning a Num to a Name (line 211):
    int_411665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 13), 'int')
    # Assigning a type to the variable 'pcount' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'pcount', int_411665)
    
    # Assigning a Attribute to a Tuple (line 213):
    
    # Assigning a Subscript to a Name (line 213):
    
    # Obtaining the type of the subscript
    int_411666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 4), 'int')
    # Getting the type of 'A' (line 213)
    A_411667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'A')
    # Obtaining the member 'shape' of a type (line 213)
    shape_411668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 11), A_411667, 'shape')
    # Obtaining the member '__getitem__' of a type (line 213)
    getitem___411669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 4), shape_411668, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 213)
    subscript_call_result_411670 = invoke(stypy.reporting.localization.Localization(__file__, 213, 4), getitem___411669, int_411666)
    
    # Assigning a type to the variable 'tuple_var_assignment_411603' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_var_assignment_411603', subscript_call_result_411670)
    
    # Assigning a Subscript to a Name (line 213):
    
    # Obtaining the type of the subscript
    int_411671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 4), 'int')
    # Getting the type of 'A' (line 213)
    A_411672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'A')
    # Obtaining the member 'shape' of a type (line 213)
    shape_411673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 11), A_411672, 'shape')
    # Obtaining the member '__getitem__' of a type (line 213)
    getitem___411674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 4), shape_411673, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 213)
    subscript_call_result_411675 = invoke(stypy.reporting.localization.Localization(__file__, 213, 4), getitem___411674, int_411671)
    
    # Assigning a type to the variable 'tuple_var_assignment_411604' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_var_assignment_411604', subscript_call_result_411675)
    
    # Assigning a Name to a Name (line 213):
    # Getting the type of 'tuple_var_assignment_411603' (line 213)
    tuple_var_assignment_411603_411676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_var_assignment_411603')
    # Assigning a type to the variable 'm' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'm', tuple_var_assignment_411603_411676)
    
    # Assigning a Name to a Name (line 213):
    # Getting the type of 'tuple_var_assignment_411604' (line 213)
    tuple_var_assignment_411604_411677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_var_assignment_411604')
    # Assigning a type to the variable 'n' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 7), 'n', tuple_var_assignment_411604_411677)
    
    # Assigning a Call to a Name (line 216):
    
    # Assigning a Call to a Name (line 216):
    
    # Call to min(...): (line 216)
    # Processing the call arguments (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_411679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    # Adding element type (line 216)
    # Getting the type of 'm' (line 216)
    m_411680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 18), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 17), list_411679, m_411680)
    # Adding element type (line 216)
    # Getting the type of 'n' (line 216)
    n_411681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 17), list_411679, n_411681)
    
    # Processing the call keyword arguments (line 216)
    kwargs_411682 = {}
    # Getting the type of 'min' (line 216)
    min_411678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 13), 'min', False)
    # Calling min(args, kwargs) (line 216)
    min_call_result_411683 = invoke(stypy.reporting.localization.Localization(__file__, 216, 13), min_411678, *[list_411679], **kwargs_411682)
    
    # Assigning a type to the variable 'minDim' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'minDim', min_call_result_411683)
    
    # Type idiom detected: calculating its left and rigth part (line 218)
    # Getting the type of 'maxiter' (line 218)
    maxiter_411684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 7), 'maxiter')
    # Getting the type of 'None' (line 218)
    None_411685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'None')
    
    (may_be_411686, more_types_in_union_411687) = may_be_none(maxiter_411684, None_411685)

    if may_be_411686:

        if more_types_in_union_411687:
            # Runtime conditional SSA (line 218)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 219):
        
        # Assigning a Name to a Name (line 219):
        # Getting the type of 'minDim' (line 219)
        minDim_411688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 18), 'minDim')
        # Assigning a type to the variable 'maxiter' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'maxiter', minDim_411688)

        if more_types_in_union_411687:
            # SSA join for if statement (line 218)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'show' (line 221)
    show_411689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 7), 'show')
    # Testing the type of an if condition (line 221)
    if_condition_411690 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 4), show_411689)
    # Assigning a type to the variable 'if_condition_411690' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'if_condition_411690', if_condition_411690)
    # SSA begins for if statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 222)
    # Processing the call arguments (line 222)
    str_411692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 14), 'str', ' ')
    # Processing the call keyword arguments (line 222)
    kwargs_411693 = {}
    # Getting the type of 'print' (line 222)
    print_411691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'print', False)
    # Calling print(args, kwargs) (line 222)
    print_call_result_411694 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), print_411691, *[str_411692], **kwargs_411693)
    
    
    # Call to print(...): (line 223)
    # Processing the call arguments (line 223)
    str_411696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 14), 'str', 'LSMR            Least-squares solution of  Ax = b\n')
    # Processing the call keyword arguments (line 223)
    kwargs_411697 = {}
    # Getting the type of 'print' (line 223)
    print_411695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'print', False)
    # Calling print(args, kwargs) (line 223)
    print_call_result_411698 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), print_411695, *[str_411696], **kwargs_411697)
    
    
    # Call to print(...): (line 224)
    # Processing the call arguments (line 224)
    str_411700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 14), 'str', 'The matrix A has %8g rows  and %8g cols')
    
    # Obtaining an instance of the builtin type 'tuple' (line 224)
    tuple_411701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 224)
    # Adding element type (line 224)
    # Getting the type of 'm' (line 224)
    m_411702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 59), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 59), tuple_411701, m_411702)
    # Adding element type (line 224)
    # Getting the type of 'n' (line 224)
    n_411703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 62), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 59), tuple_411701, n_411703)
    
    # Applying the binary operator '%' (line 224)
    result_mod_411704 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 14), '%', str_411700, tuple_411701)
    
    # Processing the call keyword arguments (line 224)
    kwargs_411705 = {}
    # Getting the type of 'print' (line 224)
    print_411699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'print', False)
    # Calling print(args, kwargs) (line 224)
    print_call_result_411706 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), print_411699, *[result_mod_411704], **kwargs_411705)
    
    
    # Call to print(...): (line 225)
    # Processing the call arguments (line 225)
    str_411708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 14), 'str', 'damp = %20.14e\n')
    # Getting the type of 'damp' (line 225)
    damp_411709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 36), 'damp', False)
    # Applying the binary operator '%' (line 225)
    result_mod_411710 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 14), '%', str_411708, damp_411709)
    
    # Processing the call keyword arguments (line 225)
    kwargs_411711 = {}
    # Getting the type of 'print' (line 225)
    print_411707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'print', False)
    # Calling print(args, kwargs) (line 225)
    print_call_result_411712 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), print_411707, *[result_mod_411710], **kwargs_411711)
    
    
    # Call to print(...): (line 226)
    # Processing the call arguments (line 226)
    str_411714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 14), 'str', 'atol = %8.2e                 conlim = %8.2e\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 226)
    tuple_411715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 65), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 226)
    # Adding element type (line 226)
    # Getting the type of 'atol' (line 226)
    atol_411716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 65), 'atol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 65), tuple_411715, atol_411716)
    # Adding element type (line 226)
    # Getting the type of 'conlim' (line 226)
    conlim_411717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 71), 'conlim', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 65), tuple_411715, conlim_411717)
    
    # Applying the binary operator '%' (line 226)
    result_mod_411718 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 14), '%', str_411714, tuple_411715)
    
    # Processing the call keyword arguments (line 226)
    kwargs_411719 = {}
    # Getting the type of 'print' (line 226)
    print_411713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'print', False)
    # Calling print(args, kwargs) (line 226)
    print_call_result_411720 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), print_411713, *[result_mod_411718], **kwargs_411719)
    
    
    # Call to print(...): (line 227)
    # Processing the call arguments (line 227)
    str_411722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 14), 'str', 'btol = %8.2e             maxiter = %8g\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 227)
    tuple_411723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 227)
    # Adding element type (line 227)
    # Getting the type of 'btol' (line 227)
    btol_411724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 60), 'btol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 60), tuple_411723, btol_411724)
    # Adding element type (line 227)
    # Getting the type of 'maxiter' (line 227)
    maxiter_411725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 66), 'maxiter', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 60), tuple_411723, maxiter_411725)
    
    # Applying the binary operator '%' (line 227)
    result_mod_411726 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 14), '%', str_411722, tuple_411723)
    
    # Processing the call keyword arguments (line 227)
    kwargs_411727 = {}
    # Getting the type of 'print' (line 227)
    print_411721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'print', False)
    # Calling print(args, kwargs) (line 227)
    print_call_result_411728 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), print_411721, *[result_mod_411726], **kwargs_411727)
    
    # SSA join for if statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 229):
    
    # Assigning a Name to a Name (line 229):
    # Getting the type of 'b' (line 229)
    b_411729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'b')
    # Assigning a type to the variable 'u' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'u', b_411729)
    
    # Assigning a Call to a Name (line 230):
    
    # Assigning a Call to a Name (line 230):
    
    # Call to norm(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'b' (line 230)
    b_411731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 17), 'b', False)
    # Processing the call keyword arguments (line 230)
    kwargs_411732 = {}
    # Getting the type of 'norm' (line 230)
    norm_411730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'norm', False)
    # Calling norm(args, kwargs) (line 230)
    norm_call_result_411733 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), norm_411730, *[b_411731], **kwargs_411732)
    
    # Assigning a type to the variable 'normb' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'normb', norm_call_result_411733)
    
    # Type idiom detected: calculating its left and rigth part (line 231)
    # Getting the type of 'x0' (line 231)
    x0_411734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 7), 'x0')
    # Getting the type of 'None' (line 231)
    None_411735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 13), 'None')
    
    (may_be_411736, more_types_in_union_411737) = may_be_none(x0_411734, None_411735)

    if may_be_411736:

        if more_types_in_union_411737:
            # Runtime conditional SSA (line 231)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 232):
        
        # Assigning a Call to a Name (line 232):
        
        # Call to zeros(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'n' (line 232)
        n_411739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 18), 'n', False)
        # Processing the call keyword arguments (line 232)
        kwargs_411740 = {}
        # Getting the type of 'zeros' (line 232)
        zeros_411738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 232)
        zeros_call_result_411741 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), zeros_411738, *[n_411739], **kwargs_411740)
        
        # Assigning a type to the variable 'x' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'x', zeros_call_result_411741)
        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to copy(...): (line 233)
        # Processing the call keyword arguments (line 233)
        kwargs_411744 = {}
        # Getting the type of 'normb' (line 233)
        normb_411742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'normb', False)
        # Obtaining the member 'copy' of a type (line 233)
        copy_411743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 15), normb_411742, 'copy')
        # Calling copy(args, kwargs) (line 233)
        copy_call_result_411745 = invoke(stypy.reporting.localization.Localization(__file__, 233, 15), copy_411743, *[], **kwargs_411744)
        
        # Assigning a type to the variable 'beta' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'beta', copy_call_result_411745)

        if more_types_in_union_411737:
            # Runtime conditional SSA for else branch (line 231)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_411736) or more_types_in_union_411737):
        
        # Assigning a Call to a Name (line 235):
        
        # Assigning a Call to a Name (line 235):
        
        # Call to atleast_1d(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'x0' (line 235)
        x0_411747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'x0', False)
        # Processing the call keyword arguments (line 235)
        kwargs_411748 = {}
        # Getting the type of 'atleast_1d' (line 235)
        atleast_1d_411746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'atleast_1d', False)
        # Calling atleast_1d(args, kwargs) (line 235)
        atleast_1d_call_result_411749 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), atleast_1d_411746, *[x0_411747], **kwargs_411748)
        
        # Assigning a type to the variable 'x' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'x', atleast_1d_call_result_411749)
        
        # Assigning a BinOp to a Name (line 236):
        
        # Assigning a BinOp to a Name (line 236):
        # Getting the type of 'u' (line 236)
        u_411750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'u')
        
        # Call to matvec(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'x' (line 236)
        x_411753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 25), 'x', False)
        # Processing the call keyword arguments (line 236)
        kwargs_411754 = {}
        # Getting the type of 'A' (line 236)
        A_411751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'A', False)
        # Obtaining the member 'matvec' of a type (line 236)
        matvec_411752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), A_411751, 'matvec')
        # Calling matvec(args, kwargs) (line 236)
        matvec_call_result_411755 = invoke(stypy.reporting.localization.Localization(__file__, 236, 16), matvec_411752, *[x_411753], **kwargs_411754)
        
        # Applying the binary operator '-' (line 236)
        result_sub_411756 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 12), '-', u_411750, matvec_call_result_411755)
        
        # Assigning a type to the variable 'u' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'u', result_sub_411756)
        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to norm(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'u' (line 237)
        u_411758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'u', False)
        # Processing the call keyword arguments (line 237)
        kwargs_411759 = {}
        # Getting the type of 'norm' (line 237)
        norm_411757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'norm', False)
        # Calling norm(args, kwargs) (line 237)
        norm_call_result_411760 = invoke(stypy.reporting.localization.Localization(__file__, 237, 15), norm_411757, *[u_411758], **kwargs_411759)
        
        # Assigning a type to the variable 'beta' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'beta', norm_call_result_411760)

        if (may_be_411736 and more_types_in_union_411737):
            # SSA join for if statement (line 231)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'beta' (line 239)
    beta_411761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 7), 'beta')
    int_411762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 14), 'int')
    # Applying the binary operator '>' (line 239)
    result_gt_411763 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 7), '>', beta_411761, int_411762)
    
    # Testing the type of an if condition (line 239)
    if_condition_411764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 4), result_gt_411763)
    # Assigning a type to the variable 'if_condition_411764' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'if_condition_411764', if_condition_411764)
    # SSA begins for if statement (line 239)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 240):
    
    # Assigning a BinOp to a Name (line 240):
    int_411765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 13), 'int')
    # Getting the type of 'beta' (line 240)
    beta_411766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 17), 'beta')
    # Applying the binary operator 'div' (line 240)
    result_div_411767 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 13), 'div', int_411765, beta_411766)
    
    # Getting the type of 'u' (line 240)
    u_411768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 25), 'u')
    # Applying the binary operator '*' (line 240)
    result_mul_411769 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 12), '*', result_div_411767, u_411768)
    
    # Assigning a type to the variable 'u' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'u', result_mul_411769)
    
    # Assigning a Call to a Name (line 241):
    
    # Assigning a Call to a Name (line 241):
    
    # Call to rmatvec(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'u' (line 241)
    u_411772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'u', False)
    # Processing the call keyword arguments (line 241)
    kwargs_411773 = {}
    # Getting the type of 'A' (line 241)
    A_411770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'A', False)
    # Obtaining the member 'rmatvec' of a type (line 241)
    rmatvec_411771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), A_411770, 'rmatvec')
    # Calling rmatvec(args, kwargs) (line 241)
    rmatvec_call_result_411774 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), rmatvec_411771, *[u_411772], **kwargs_411773)
    
    # Assigning a type to the variable 'v' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'v', rmatvec_call_result_411774)
    
    # Assigning a Call to a Name (line 242):
    
    # Assigning a Call to a Name (line 242):
    
    # Call to norm(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'v' (line 242)
    v_411776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 'v', False)
    # Processing the call keyword arguments (line 242)
    kwargs_411777 = {}
    # Getting the type of 'norm' (line 242)
    norm_411775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'norm', False)
    # Calling norm(args, kwargs) (line 242)
    norm_call_result_411778 = invoke(stypy.reporting.localization.Localization(__file__, 242, 16), norm_411775, *[v_411776], **kwargs_411777)
    
    # Assigning a type to the variable 'alpha' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'alpha', norm_call_result_411778)
    # SSA branch for the else part of an if statement (line 239)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 244):
    
    # Assigning a Call to a Name (line 244):
    
    # Call to zeros(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'n' (line 244)
    n_411780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 18), 'n', False)
    # Processing the call keyword arguments (line 244)
    kwargs_411781 = {}
    # Getting the type of 'zeros' (line 244)
    zeros_411779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'zeros', False)
    # Calling zeros(args, kwargs) (line 244)
    zeros_call_result_411782 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), zeros_411779, *[n_411780], **kwargs_411781)
    
    # Assigning a type to the variable 'v' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'v', zeros_call_result_411782)
    
    # Assigning a Num to a Name (line 245):
    
    # Assigning a Num to a Name (line 245):
    int_411783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 16), 'int')
    # Assigning a type to the variable 'alpha' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'alpha', int_411783)
    # SSA join for if statement (line 239)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'alpha' (line 247)
    alpha_411784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 7), 'alpha')
    int_411785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 15), 'int')
    # Applying the binary operator '>' (line 247)
    result_gt_411786 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 7), '>', alpha_411784, int_411785)
    
    # Testing the type of an if condition (line 247)
    if_condition_411787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 4), result_gt_411786)
    # Assigning a type to the variable 'if_condition_411787' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'if_condition_411787', if_condition_411787)
    # SSA begins for if statement (line 247)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 248):
    
    # Assigning a BinOp to a Name (line 248):
    int_411788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 13), 'int')
    # Getting the type of 'alpha' (line 248)
    alpha_411789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 17), 'alpha')
    # Applying the binary operator 'div' (line 248)
    result_div_411790 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 13), 'div', int_411788, alpha_411789)
    
    # Getting the type of 'v' (line 248)
    v_411791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 26), 'v')
    # Applying the binary operator '*' (line 248)
    result_mul_411792 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 12), '*', result_div_411790, v_411791)
    
    # Assigning a type to the variable 'v' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'v', result_mul_411792)
    # SSA join for if statement (line 247)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 252):
    
    # Assigning a Num to a Name (line 252):
    int_411793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 10), 'int')
    # Assigning a type to the variable 'itn' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'itn', int_411793)
    
    # Assigning a BinOp to a Name (line 253):
    
    # Assigning a BinOp to a Name (line 253):
    # Getting the type of 'alpha' (line 253)
    alpha_411794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 14), 'alpha')
    # Getting the type of 'beta' (line 253)
    beta_411795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 22), 'beta')
    # Applying the binary operator '*' (line 253)
    result_mul_411796 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 14), '*', alpha_411794, beta_411795)
    
    # Assigning a type to the variable 'zetabar' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'zetabar', result_mul_411796)
    
    # Assigning a Name to a Name (line 254):
    
    # Assigning a Name to a Name (line 254):
    # Getting the type of 'alpha' (line 254)
    alpha_411797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'alpha')
    # Assigning a type to the variable 'alphabar' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'alphabar', alpha_411797)
    
    # Assigning a Num to a Name (line 255):
    
    # Assigning a Num to a Name (line 255):
    int_411798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 10), 'int')
    # Assigning a type to the variable 'rho' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'rho', int_411798)
    
    # Assigning a Num to a Name (line 256):
    
    # Assigning a Num to a Name (line 256):
    int_411799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 13), 'int')
    # Assigning a type to the variable 'rhobar' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'rhobar', int_411799)
    
    # Assigning a Num to a Name (line 257):
    
    # Assigning a Num to a Name (line 257):
    int_411800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 11), 'int')
    # Assigning a type to the variable 'cbar' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'cbar', int_411800)
    
    # Assigning a Num to a Name (line 258):
    
    # Assigning a Num to a Name (line 258):
    int_411801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 11), 'int')
    # Assigning a type to the variable 'sbar' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'sbar', int_411801)
    
    # Assigning a Call to a Name (line 260):
    
    # Assigning a Call to a Name (line 260):
    
    # Call to copy(...): (line 260)
    # Processing the call keyword arguments (line 260)
    kwargs_411804 = {}
    # Getting the type of 'v' (line 260)
    v_411802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'v', False)
    # Obtaining the member 'copy' of a type (line 260)
    copy_411803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), v_411802, 'copy')
    # Calling copy(args, kwargs) (line 260)
    copy_call_result_411805 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), copy_411803, *[], **kwargs_411804)
    
    # Assigning a type to the variable 'h' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'h', copy_call_result_411805)
    
    # Assigning a Call to a Name (line 261):
    
    # Assigning a Call to a Name (line 261):
    
    # Call to zeros(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'n' (line 261)
    n_411807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 17), 'n', False)
    # Processing the call keyword arguments (line 261)
    kwargs_411808 = {}
    # Getting the type of 'zeros' (line 261)
    zeros_411806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'zeros', False)
    # Calling zeros(args, kwargs) (line 261)
    zeros_call_result_411809 = invoke(stypy.reporting.localization.Localization(__file__, 261, 11), zeros_411806, *[n_411807], **kwargs_411808)
    
    # Assigning a type to the variable 'hbar' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'hbar', zeros_call_result_411809)
    
    # Assigning a Name to a Name (line 265):
    
    # Assigning a Name to a Name (line 265):
    # Getting the type of 'beta' (line 265)
    beta_411810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'beta')
    # Assigning a type to the variable 'betadd' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'betadd', beta_411810)
    
    # Assigning a Num to a Name (line 266):
    
    # Assigning a Num to a Name (line 266):
    int_411811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 12), 'int')
    # Assigning a type to the variable 'betad' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'betad', int_411811)
    
    # Assigning a Num to a Name (line 267):
    
    # Assigning a Num to a Name (line 267):
    int_411812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 14), 'int')
    # Assigning a type to the variable 'rhodold' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'rhodold', int_411812)
    
    # Assigning a Num to a Name (line 268):
    
    # Assigning a Num to a Name (line 268):
    int_411813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 18), 'int')
    # Assigning a type to the variable 'tautildeold' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'tautildeold', int_411813)
    
    # Assigning a Num to a Name (line 269):
    
    # Assigning a Num to a Name (line 269):
    int_411814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 17), 'int')
    # Assigning a type to the variable 'thetatilde' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'thetatilde', int_411814)
    
    # Assigning a Num to a Name (line 270):
    
    # Assigning a Num to a Name (line 270):
    int_411815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 11), 'int')
    # Assigning a type to the variable 'zeta' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'zeta', int_411815)
    
    # Assigning a Num to a Name (line 271):
    
    # Assigning a Num to a Name (line 271):
    int_411816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 8), 'int')
    # Assigning a type to the variable 'd' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'd', int_411816)
    
    # Assigning a BinOp to a Name (line 275):
    
    # Assigning a BinOp to a Name (line 275):
    # Getting the type of 'alpha' (line 275)
    alpha_411817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'alpha')
    # Getting the type of 'alpha' (line 275)
    alpha_411818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 21), 'alpha')
    # Applying the binary operator '*' (line 275)
    result_mul_411819 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 13), '*', alpha_411817, alpha_411818)
    
    # Assigning a type to the variable 'normA2' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'normA2', result_mul_411819)
    
    # Assigning a Num to a Name (line 276):
    
    # Assigning a Num to a Name (line 276):
    int_411820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 14), 'int')
    # Assigning a type to the variable 'maxrbar' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'maxrbar', int_411820)
    
    # Assigning a Num to a Name (line 277):
    
    # Assigning a Num to a Name (line 277):
    float_411821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 14), 'float')
    # Assigning a type to the variable 'minrbar' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'minrbar', float_411821)
    
    # Assigning a Call to a Name (line 278):
    
    # Assigning a Call to a Name (line 278):
    
    # Call to sqrt(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'normA2' (line 278)
    normA2_411823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 17), 'normA2', False)
    # Processing the call keyword arguments (line 278)
    kwargs_411824 = {}
    # Getting the type of 'sqrt' (line 278)
    sqrt_411822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 278)
    sqrt_call_result_411825 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), sqrt_411822, *[normA2_411823], **kwargs_411824)
    
    # Assigning a type to the variable 'normA' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'normA', sqrt_call_result_411825)
    
    # Assigning a Num to a Name (line 279):
    
    # Assigning a Num to a Name (line 279):
    int_411826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 12), 'int')
    # Assigning a type to the variable 'condA' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'condA', int_411826)
    
    # Assigning a Num to a Name (line 280):
    
    # Assigning a Num to a Name (line 280):
    int_411827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 12), 'int')
    # Assigning a type to the variable 'normx' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'normx', int_411827)
    
    # Assigning a Num to a Name (line 283):
    
    # Assigning a Num to a Name (line 283):
    int_411828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 12), 'int')
    # Assigning a type to the variable 'istop' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'istop', int_411828)
    
    # Assigning a Num to a Name (line 284):
    
    # Assigning a Num to a Name (line 284):
    int_411829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 11), 'int')
    # Assigning a type to the variable 'ctol' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'ctol', int_411829)
    
    
    # Getting the type of 'conlim' (line 285)
    conlim_411830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 7), 'conlim')
    int_411831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 16), 'int')
    # Applying the binary operator '>' (line 285)
    result_gt_411832 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 7), '>', conlim_411830, int_411831)
    
    # Testing the type of an if condition (line 285)
    if_condition_411833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 4), result_gt_411832)
    # Assigning a type to the variable 'if_condition_411833' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'if_condition_411833', if_condition_411833)
    # SSA begins for if statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 286):
    
    # Assigning a BinOp to a Name (line 286):
    int_411834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 15), 'int')
    # Getting the type of 'conlim' (line 286)
    conlim_411835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'conlim')
    # Applying the binary operator 'div' (line 286)
    result_div_411836 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 15), 'div', int_411834, conlim_411835)
    
    # Assigning a type to the variable 'ctol' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'ctol', result_div_411836)
    # SSA join for if statement (line 285)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 287):
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'beta' (line 287)
    beta_411837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'beta')
    # Assigning a type to the variable 'normr' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'normr', beta_411837)
    
    # Assigning a BinOp to a Name (line 291):
    
    # Assigning a BinOp to a Name (line 291):
    # Getting the type of 'alpha' (line 291)
    alpha_411838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 13), 'alpha')
    # Getting the type of 'beta' (line 291)
    beta_411839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 21), 'beta')
    # Applying the binary operator '*' (line 291)
    result_mul_411840 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 13), '*', alpha_411838, beta_411839)
    
    # Assigning a type to the variable 'normar' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'normar', result_mul_411840)
    
    
    # Getting the type of 'normar' (line 292)
    normar_411841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 7), 'normar')
    int_411842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 17), 'int')
    # Applying the binary operator '==' (line 292)
    result_eq_411843 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 7), '==', normar_411841, int_411842)
    
    # Testing the type of an if condition (line 292)
    if_condition_411844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 4), result_eq_411843)
    # Assigning a type to the variable 'if_condition_411844' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'if_condition_411844', if_condition_411844)
    # SSA begins for if statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'show' (line 293)
    show_411845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 11), 'show')
    # Testing the type of an if condition (line 293)
    if_condition_411846 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 8), show_411845)
    # Assigning a type to the variable 'if_condition_411846' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'if_condition_411846', if_condition_411846)
    # SSA begins for if statement (line 293)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 294)
    # Processing the call arguments (line 294)
    
    # Obtaining the type of the subscript
    int_411848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 22), 'int')
    # Getting the type of 'msg' (line 294)
    msg_411849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 18), 'msg', False)
    # Obtaining the member '__getitem__' of a type (line 294)
    getitem___411850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 18), msg_411849, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 294)
    subscript_call_result_411851 = invoke(stypy.reporting.localization.Localization(__file__, 294, 18), getitem___411850, int_411848)
    
    # Processing the call keyword arguments (line 294)
    kwargs_411852 = {}
    # Getting the type of 'print' (line 294)
    print_411847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'print', False)
    # Calling print(args, kwargs) (line 294)
    print_call_result_411853 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), print_411847, *[subscript_call_result_411851], **kwargs_411852)
    
    # SSA join for if statement (line 293)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 295)
    tuple_411854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 295)
    # Adding element type (line 295)
    # Getting the type of 'x' (line 295)
    x_411855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 15), tuple_411854, x_411855)
    # Adding element type (line 295)
    # Getting the type of 'istop' (line 295)
    istop_411856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'istop')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 15), tuple_411854, istop_411856)
    # Adding element type (line 295)
    # Getting the type of 'itn' (line 295)
    itn_411857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 25), 'itn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 15), tuple_411854, itn_411857)
    # Adding element type (line 295)
    # Getting the type of 'normr' (line 295)
    normr_411858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 30), 'normr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 15), tuple_411854, normr_411858)
    # Adding element type (line 295)
    # Getting the type of 'normar' (line 295)
    normar_411859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 37), 'normar')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 15), tuple_411854, normar_411859)
    # Adding element type (line 295)
    # Getting the type of 'normA' (line 295)
    normA_411860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 45), 'normA')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 15), tuple_411854, normA_411860)
    # Adding element type (line 295)
    # Getting the type of 'condA' (line 295)
    condA_411861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'condA')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 15), tuple_411854, condA_411861)
    # Adding element type (line 295)
    # Getting the type of 'normx' (line 295)
    normx_411862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 59), 'normx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 15), tuple_411854, normx_411862)
    
    # Assigning a type to the variable 'stypy_return_type' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'stypy_return_type', tuple_411854)
    # SSA join for if statement (line 292)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'show' (line 297)
    show_411863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 7), 'show')
    # Testing the type of an if condition (line 297)
    if_condition_411864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 4), show_411863)
    # Assigning a type to the variable 'if_condition_411864' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'if_condition_411864', if_condition_411864)
    # SSA begins for if statement (line 297)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 298)
    # Processing the call arguments (line 298)
    str_411866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 14), 'str', ' ')
    # Processing the call keyword arguments (line 298)
    kwargs_411867 = {}
    # Getting the type of 'print' (line 298)
    print_411865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'print', False)
    # Calling print(args, kwargs) (line 298)
    print_call_result_411868 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), print_411865, *[str_411866], **kwargs_411867)
    
    
    # Call to print(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'hdg1' (line 299)
    hdg1_411870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 14), 'hdg1', False)
    # Getting the type of 'hdg2' (line 299)
    hdg2_411871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 20), 'hdg2', False)
    # Processing the call keyword arguments (line 299)
    kwargs_411872 = {}
    # Getting the type of 'print' (line 299)
    print_411869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'print', False)
    # Calling print(args, kwargs) (line 299)
    print_call_result_411873 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), print_411869, *[hdg1_411870, hdg2_411871], **kwargs_411872)
    
    
    # Assigning a Num to a Name (line 300):
    
    # Assigning a Num to a Name (line 300):
    int_411874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 16), 'int')
    # Assigning a type to the variable 'test1' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'test1', int_411874)
    
    # Assigning a BinOp to a Name (line 301):
    
    # Assigning a BinOp to a Name (line 301):
    # Getting the type of 'alpha' (line 301)
    alpha_411875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'alpha')
    # Getting the type of 'beta' (line 301)
    beta_411876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'beta')
    # Applying the binary operator 'div' (line 301)
    result_div_411877 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 16), 'div', alpha_411875, beta_411876)
    
    # Assigning a type to the variable 'test2' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'test2', result_div_411877)
    
    # Assigning a BinOp to a Name (line 302):
    
    # Assigning a BinOp to a Name (line 302):
    str_411878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 15), 'str', '%6g %12.5e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 302)
    tuple_411879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 302)
    # Adding element type (line 302)
    # Getting the type of 'itn' (line 302)
    itn_411880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 31), 'itn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 31), tuple_411879, itn_411880)
    # Adding element type (line 302)
    
    # Obtaining the type of the subscript
    int_411881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 38), 'int')
    # Getting the type of 'x' (line 302)
    x_411882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 36), 'x')
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___411883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 36), x_411882, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_411884 = invoke(stypy.reporting.localization.Localization(__file__, 302, 36), getitem___411883, int_411881)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 31), tuple_411879, subscript_call_result_411884)
    
    # Applying the binary operator '%' (line 302)
    result_mod_411885 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 15), '%', str_411878, tuple_411879)
    
    # Assigning a type to the variable 'str1' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'str1', result_mod_411885)
    
    # Assigning a BinOp to a Name (line 303):
    
    # Assigning a BinOp to a Name (line 303):
    str_411886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 15), 'str', ' %10.3e %10.3e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 303)
    tuple_411887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 303)
    # Adding element type (line 303)
    # Getting the type of 'normr' (line 303)
    normr_411888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 35), 'normr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 35), tuple_411887, normr_411888)
    # Adding element type (line 303)
    # Getting the type of 'normar' (line 303)
    normar_411889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 42), 'normar')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 35), tuple_411887, normar_411889)
    
    # Applying the binary operator '%' (line 303)
    result_mod_411890 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 15), '%', str_411886, tuple_411887)
    
    # Assigning a type to the variable 'str2' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'str2', result_mod_411890)
    
    # Assigning a BinOp to a Name (line 304):
    
    # Assigning a BinOp to a Name (line 304):
    str_411891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 15), 'str', '  %8.1e %8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 304)
    tuple_411892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 304)
    # Adding element type (line 304)
    # Getting the type of 'test1' (line 304)
    test1_411893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 34), 'test1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 34), tuple_411892, test1_411893)
    # Adding element type (line 304)
    # Getting the type of 'test2' (line 304)
    test2_411894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 41), 'test2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 34), tuple_411892, test2_411894)
    
    # Applying the binary operator '%' (line 304)
    result_mod_411895 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 15), '%', str_411891, tuple_411892)
    
    # Assigning a type to the variable 'str3' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'str3', result_mod_411895)
    
    # Call to print(...): (line 305)
    # Processing the call arguments (line 305)
    
    # Call to join(...): (line 305)
    # Processing the call arguments (line 305)
    
    # Obtaining an instance of the builtin type 'list' (line 305)
    list_411899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 305)
    # Adding element type (line 305)
    # Getting the type of 'str1' (line 305)
    str1_411900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 23), 'str1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 22), list_411899, str1_411900)
    # Adding element type (line 305)
    # Getting the type of 'str2' (line 305)
    str2_411901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 29), 'str2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 22), list_411899, str2_411901)
    # Adding element type (line 305)
    # Getting the type of 'str3' (line 305)
    str3_411902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 35), 'str3', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 22), list_411899, str3_411902)
    
    # Processing the call keyword arguments (line 305)
    kwargs_411903 = {}
    str_411897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 14), 'str', '')
    # Obtaining the member 'join' of a type (line 305)
    join_411898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 14), str_411897, 'join')
    # Calling join(args, kwargs) (line 305)
    join_call_result_411904 = invoke(stypy.reporting.localization.Localization(__file__, 305, 14), join_411898, *[list_411899], **kwargs_411903)
    
    # Processing the call keyword arguments (line 305)
    kwargs_411905 = {}
    # Getting the type of 'print' (line 305)
    print_411896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'print', False)
    # Calling print(args, kwargs) (line 305)
    print_call_result_411906 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), print_411896, *[join_call_result_411904], **kwargs_411905)
    
    # SSA join for if statement (line 297)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'itn' (line 308)
    itn_411907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 10), 'itn')
    # Getting the type of 'maxiter' (line 308)
    maxiter_411908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'maxiter')
    # Applying the binary operator '<' (line 308)
    result_lt_411909 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 10), '<', itn_411907, maxiter_411908)
    
    # Testing the type of an if condition (line 308)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 4), result_lt_411909)
    # SSA begins for while statement (line 308)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 309):
    
    # Assigning a BinOp to a Name (line 309):
    # Getting the type of 'itn' (line 309)
    itn_411910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 14), 'itn')
    int_411911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 20), 'int')
    # Applying the binary operator '+' (line 309)
    result_add_411912 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 14), '+', itn_411910, int_411911)
    
    # Assigning a type to the variable 'itn' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'itn', result_add_411912)
    
    # Assigning a BinOp to a Name (line 316):
    
    # Assigning a BinOp to a Name (line 316):
    
    # Call to matvec(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'v' (line 316)
    v_411915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 21), 'v', False)
    # Processing the call keyword arguments (line 316)
    kwargs_411916 = {}
    # Getting the type of 'A' (line 316)
    A_411913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'A', False)
    # Obtaining the member 'matvec' of a type (line 316)
    matvec_411914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), A_411913, 'matvec')
    # Calling matvec(args, kwargs) (line 316)
    matvec_call_result_411917 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), matvec_411914, *[v_411915], **kwargs_411916)
    
    # Getting the type of 'alpha' (line 316)
    alpha_411918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 26), 'alpha')
    # Getting the type of 'u' (line 316)
    u_411919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 34), 'u')
    # Applying the binary operator '*' (line 316)
    result_mul_411920 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 26), '*', alpha_411918, u_411919)
    
    # Applying the binary operator '-' (line 316)
    result_sub_411921 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 12), '-', matvec_call_result_411917, result_mul_411920)
    
    # Assigning a type to the variable 'u' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'u', result_sub_411921)
    
    # Assigning a Call to a Name (line 317):
    
    # Assigning a Call to a Name (line 317):
    
    # Call to norm(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'u' (line 317)
    u_411923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'u', False)
    # Processing the call keyword arguments (line 317)
    kwargs_411924 = {}
    # Getting the type of 'norm' (line 317)
    norm_411922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'norm', False)
    # Calling norm(args, kwargs) (line 317)
    norm_call_result_411925 = invoke(stypy.reporting.localization.Localization(__file__, 317, 15), norm_411922, *[u_411923], **kwargs_411924)
    
    # Assigning a type to the variable 'beta' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'beta', norm_call_result_411925)
    
    
    # Getting the type of 'beta' (line 319)
    beta_411926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 11), 'beta')
    int_411927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 18), 'int')
    # Applying the binary operator '>' (line 319)
    result_gt_411928 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 11), '>', beta_411926, int_411927)
    
    # Testing the type of an if condition (line 319)
    if_condition_411929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 8), result_gt_411928)
    # Assigning a type to the variable 'if_condition_411929' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'if_condition_411929', if_condition_411929)
    # SSA begins for if statement (line 319)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 320):
    
    # Assigning a BinOp to a Name (line 320):
    int_411930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 17), 'int')
    # Getting the type of 'beta' (line 320)
    beta_411931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 21), 'beta')
    # Applying the binary operator 'div' (line 320)
    result_div_411932 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 17), 'div', int_411930, beta_411931)
    
    # Getting the type of 'u' (line 320)
    u_411933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 29), 'u')
    # Applying the binary operator '*' (line 320)
    result_mul_411934 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 16), '*', result_div_411932, u_411933)
    
    # Assigning a type to the variable 'u' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'u', result_mul_411934)
    
    # Assigning a BinOp to a Name (line 321):
    
    # Assigning a BinOp to a Name (line 321):
    
    # Call to rmatvec(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'u' (line 321)
    u_411937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 26), 'u', False)
    # Processing the call keyword arguments (line 321)
    kwargs_411938 = {}
    # Getting the type of 'A' (line 321)
    A_411935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'A', False)
    # Obtaining the member 'rmatvec' of a type (line 321)
    rmatvec_411936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 16), A_411935, 'rmatvec')
    # Calling rmatvec(args, kwargs) (line 321)
    rmatvec_call_result_411939 = invoke(stypy.reporting.localization.Localization(__file__, 321, 16), rmatvec_411936, *[u_411937], **kwargs_411938)
    
    # Getting the type of 'beta' (line 321)
    beta_411940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 31), 'beta')
    # Getting the type of 'v' (line 321)
    v_411941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 38), 'v')
    # Applying the binary operator '*' (line 321)
    result_mul_411942 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 31), '*', beta_411940, v_411941)
    
    # Applying the binary operator '-' (line 321)
    result_sub_411943 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 16), '-', rmatvec_call_result_411939, result_mul_411942)
    
    # Assigning a type to the variable 'v' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'v', result_sub_411943)
    
    # Assigning a Call to a Name (line 322):
    
    # Assigning a Call to a Name (line 322):
    
    # Call to norm(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'v' (line 322)
    v_411945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'v', False)
    # Processing the call keyword arguments (line 322)
    kwargs_411946 = {}
    # Getting the type of 'norm' (line 322)
    norm_411944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'norm', False)
    # Calling norm(args, kwargs) (line 322)
    norm_call_result_411947 = invoke(stypy.reporting.localization.Localization(__file__, 322, 20), norm_411944, *[v_411945], **kwargs_411946)
    
    # Assigning a type to the variable 'alpha' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'alpha', norm_call_result_411947)
    
    
    # Getting the type of 'alpha' (line 323)
    alpha_411948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'alpha')
    int_411949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 23), 'int')
    # Applying the binary operator '>' (line 323)
    result_gt_411950 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 15), '>', alpha_411948, int_411949)
    
    # Testing the type of an if condition (line 323)
    if_condition_411951 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 12), result_gt_411950)
    # Assigning a type to the variable 'if_condition_411951' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'if_condition_411951', if_condition_411951)
    # SSA begins for if statement (line 323)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 324):
    
    # Assigning a BinOp to a Name (line 324):
    int_411952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 21), 'int')
    # Getting the type of 'alpha' (line 324)
    alpha_411953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 25), 'alpha')
    # Applying the binary operator 'div' (line 324)
    result_div_411954 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 21), 'div', int_411952, alpha_411953)
    
    # Getting the type of 'v' (line 324)
    v_411955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'v')
    # Applying the binary operator '*' (line 324)
    result_mul_411956 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 20), '*', result_div_411954, v_411955)
    
    # Assigning a type to the variable 'v' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'v', result_mul_411956)
    # SSA join for if statement (line 323)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 319)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 330):
    
    # Assigning a Subscript to a Name (line 330):
    
    # Obtaining the type of the subscript
    int_411957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 8), 'int')
    
    # Call to _sym_ortho(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'alphabar' (line 330)
    alphabar_411959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 42), 'alphabar', False)
    # Getting the type of 'damp' (line 330)
    damp_411960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 52), 'damp', False)
    # Processing the call keyword arguments (line 330)
    kwargs_411961 = {}
    # Getting the type of '_sym_ortho' (line 330)
    _sym_ortho_411958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 31), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 330)
    _sym_ortho_call_result_411962 = invoke(stypy.reporting.localization.Localization(__file__, 330, 31), _sym_ortho_411958, *[alphabar_411959, damp_411960], **kwargs_411961)
    
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___411963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), _sym_ortho_call_result_411962, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_411964 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), getitem___411963, int_411957)
    
    # Assigning a type to the variable 'tuple_var_assignment_411605' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_var_assignment_411605', subscript_call_result_411964)
    
    # Assigning a Subscript to a Name (line 330):
    
    # Obtaining the type of the subscript
    int_411965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 8), 'int')
    
    # Call to _sym_ortho(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'alphabar' (line 330)
    alphabar_411967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 42), 'alphabar', False)
    # Getting the type of 'damp' (line 330)
    damp_411968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 52), 'damp', False)
    # Processing the call keyword arguments (line 330)
    kwargs_411969 = {}
    # Getting the type of '_sym_ortho' (line 330)
    _sym_ortho_411966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 31), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 330)
    _sym_ortho_call_result_411970 = invoke(stypy.reporting.localization.Localization(__file__, 330, 31), _sym_ortho_411966, *[alphabar_411967, damp_411968], **kwargs_411969)
    
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___411971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), _sym_ortho_call_result_411970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_411972 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), getitem___411971, int_411965)
    
    # Assigning a type to the variable 'tuple_var_assignment_411606' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_var_assignment_411606', subscript_call_result_411972)
    
    # Assigning a Subscript to a Name (line 330):
    
    # Obtaining the type of the subscript
    int_411973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 8), 'int')
    
    # Call to _sym_ortho(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'alphabar' (line 330)
    alphabar_411975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 42), 'alphabar', False)
    # Getting the type of 'damp' (line 330)
    damp_411976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 52), 'damp', False)
    # Processing the call keyword arguments (line 330)
    kwargs_411977 = {}
    # Getting the type of '_sym_ortho' (line 330)
    _sym_ortho_411974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 31), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 330)
    _sym_ortho_call_result_411978 = invoke(stypy.reporting.localization.Localization(__file__, 330, 31), _sym_ortho_411974, *[alphabar_411975, damp_411976], **kwargs_411977)
    
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___411979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), _sym_ortho_call_result_411978, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_411980 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), getitem___411979, int_411973)
    
    # Assigning a type to the variable 'tuple_var_assignment_411607' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_var_assignment_411607', subscript_call_result_411980)
    
    # Assigning a Name to a Name (line 330):
    # Getting the type of 'tuple_var_assignment_411605' (line 330)
    tuple_var_assignment_411605_411981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_var_assignment_411605')
    # Assigning a type to the variable 'chat' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'chat', tuple_var_assignment_411605_411981)
    
    # Assigning a Name to a Name (line 330):
    # Getting the type of 'tuple_var_assignment_411606' (line 330)
    tuple_var_assignment_411606_411982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_var_assignment_411606')
    # Assigning a type to the variable 'shat' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 14), 'shat', tuple_var_assignment_411606_411982)
    
    # Assigning a Name to a Name (line 330):
    # Getting the type of 'tuple_var_assignment_411607' (line 330)
    tuple_var_assignment_411607_411983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_var_assignment_411607')
    # Assigning a type to the variable 'alphahat' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 20), 'alphahat', tuple_var_assignment_411607_411983)
    
    # Assigning a Name to a Name (line 334):
    
    # Assigning a Name to a Name (line 334):
    # Getting the type of 'rho' (line 334)
    rho_411984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 17), 'rho')
    # Assigning a type to the variable 'rhoold' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'rhoold', rho_411984)
    
    # Assigning a Call to a Tuple (line 335):
    
    # Assigning a Subscript to a Name (line 335):
    
    # Obtaining the type of the subscript
    int_411985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 8), 'int')
    
    # Call to _sym_ortho(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'alphahat' (line 335)
    alphahat_411987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 31), 'alphahat', False)
    # Getting the type of 'beta' (line 335)
    beta_411988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 41), 'beta', False)
    # Processing the call keyword arguments (line 335)
    kwargs_411989 = {}
    # Getting the type of '_sym_ortho' (line 335)
    _sym_ortho_411986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 335)
    _sym_ortho_call_result_411990 = invoke(stypy.reporting.localization.Localization(__file__, 335, 20), _sym_ortho_411986, *[alphahat_411987, beta_411988], **kwargs_411989)
    
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___411991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), _sym_ortho_call_result_411990, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_411992 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), getitem___411991, int_411985)
    
    # Assigning a type to the variable 'tuple_var_assignment_411608' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'tuple_var_assignment_411608', subscript_call_result_411992)
    
    # Assigning a Subscript to a Name (line 335):
    
    # Obtaining the type of the subscript
    int_411993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 8), 'int')
    
    # Call to _sym_ortho(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'alphahat' (line 335)
    alphahat_411995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 31), 'alphahat', False)
    # Getting the type of 'beta' (line 335)
    beta_411996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 41), 'beta', False)
    # Processing the call keyword arguments (line 335)
    kwargs_411997 = {}
    # Getting the type of '_sym_ortho' (line 335)
    _sym_ortho_411994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 335)
    _sym_ortho_call_result_411998 = invoke(stypy.reporting.localization.Localization(__file__, 335, 20), _sym_ortho_411994, *[alphahat_411995, beta_411996], **kwargs_411997)
    
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___411999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), _sym_ortho_call_result_411998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_412000 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), getitem___411999, int_411993)
    
    # Assigning a type to the variable 'tuple_var_assignment_411609' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'tuple_var_assignment_411609', subscript_call_result_412000)
    
    # Assigning a Subscript to a Name (line 335):
    
    # Obtaining the type of the subscript
    int_412001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 8), 'int')
    
    # Call to _sym_ortho(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'alphahat' (line 335)
    alphahat_412003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 31), 'alphahat', False)
    # Getting the type of 'beta' (line 335)
    beta_412004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 41), 'beta', False)
    # Processing the call keyword arguments (line 335)
    kwargs_412005 = {}
    # Getting the type of '_sym_ortho' (line 335)
    _sym_ortho_412002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 335)
    _sym_ortho_call_result_412006 = invoke(stypy.reporting.localization.Localization(__file__, 335, 20), _sym_ortho_412002, *[alphahat_412003, beta_412004], **kwargs_412005)
    
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___412007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), _sym_ortho_call_result_412006, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_412008 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), getitem___412007, int_412001)
    
    # Assigning a type to the variable 'tuple_var_assignment_411610' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'tuple_var_assignment_411610', subscript_call_result_412008)
    
    # Assigning a Name to a Name (line 335):
    # Getting the type of 'tuple_var_assignment_411608' (line 335)
    tuple_var_assignment_411608_412009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'tuple_var_assignment_411608')
    # Assigning a type to the variable 'c' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'c', tuple_var_assignment_411608_412009)
    
    # Assigning a Name to a Name (line 335):
    # Getting the type of 'tuple_var_assignment_411609' (line 335)
    tuple_var_assignment_411609_412010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'tuple_var_assignment_411609')
    # Assigning a type to the variable 's' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 's', tuple_var_assignment_411609_412010)
    
    # Assigning a Name to a Name (line 335):
    # Getting the type of 'tuple_var_assignment_411610' (line 335)
    tuple_var_assignment_411610_412011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'tuple_var_assignment_411610')
    # Assigning a type to the variable 'rho' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 14), 'rho', tuple_var_assignment_411610_412011)
    
    # Assigning a BinOp to a Name (line 336):
    
    # Assigning a BinOp to a Name (line 336):
    # Getting the type of 's' (line 336)
    s_412012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 's')
    # Getting the type of 'alpha' (line 336)
    alpha_412013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 21), 'alpha')
    # Applying the binary operator '*' (line 336)
    result_mul_412014 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 19), '*', s_412012, alpha_412013)
    
    # Assigning a type to the variable 'thetanew' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'thetanew', result_mul_412014)
    
    # Assigning a BinOp to a Name (line 337):
    
    # Assigning a BinOp to a Name (line 337):
    # Getting the type of 'c' (line 337)
    c_412015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 19), 'c')
    # Getting the type of 'alpha' (line 337)
    alpha_412016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 21), 'alpha')
    # Applying the binary operator '*' (line 337)
    result_mul_412017 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 19), '*', c_412015, alpha_412016)
    
    # Assigning a type to the variable 'alphabar' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'alphabar', result_mul_412017)
    
    # Assigning a Name to a Name (line 341):
    
    # Assigning a Name to a Name (line 341):
    # Getting the type of 'rhobar' (line 341)
    rhobar_412018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 20), 'rhobar')
    # Assigning a type to the variable 'rhobarold' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'rhobarold', rhobar_412018)
    
    # Assigning a Name to a Name (line 342):
    
    # Assigning a Name to a Name (line 342):
    # Getting the type of 'zeta' (line 342)
    zeta_412019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 18), 'zeta')
    # Assigning a type to the variable 'zetaold' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'zetaold', zeta_412019)
    
    # Assigning a BinOp to a Name (line 343):
    
    # Assigning a BinOp to a Name (line 343):
    # Getting the type of 'sbar' (line 343)
    sbar_412020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'sbar')
    # Getting the type of 'rho' (line 343)
    rho_412021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 26), 'rho')
    # Applying the binary operator '*' (line 343)
    result_mul_412022 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 19), '*', sbar_412020, rho_412021)
    
    # Assigning a type to the variable 'thetabar' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'thetabar', result_mul_412022)
    
    # Assigning a BinOp to a Name (line 344):
    
    # Assigning a BinOp to a Name (line 344):
    # Getting the type of 'cbar' (line 344)
    cbar_412023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 18), 'cbar')
    # Getting the type of 'rho' (line 344)
    rho_412024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 25), 'rho')
    # Applying the binary operator '*' (line 344)
    result_mul_412025 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 18), '*', cbar_412023, rho_412024)
    
    # Assigning a type to the variable 'rhotemp' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'rhotemp', result_mul_412025)
    
    # Assigning a Call to a Tuple (line 345):
    
    # Assigning a Subscript to a Name (line 345):
    
    # Obtaining the type of the subscript
    int_412026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 8), 'int')
    
    # Call to _sym_ortho(...): (line 345)
    # Processing the call arguments (line 345)
    # Getting the type of 'cbar' (line 345)
    cbar_412028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 40), 'cbar', False)
    # Getting the type of 'rho' (line 345)
    rho_412029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 47), 'rho', False)
    # Applying the binary operator '*' (line 345)
    result_mul_412030 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 40), '*', cbar_412028, rho_412029)
    
    # Getting the type of 'thetanew' (line 345)
    thetanew_412031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 52), 'thetanew', False)
    # Processing the call keyword arguments (line 345)
    kwargs_412032 = {}
    # Getting the type of '_sym_ortho' (line 345)
    _sym_ortho_412027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 29), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 345)
    _sym_ortho_call_result_412033 = invoke(stypy.reporting.localization.Localization(__file__, 345, 29), _sym_ortho_412027, *[result_mul_412030, thetanew_412031], **kwargs_412032)
    
    # Obtaining the member '__getitem__' of a type (line 345)
    getitem___412034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), _sym_ortho_call_result_412033, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 345)
    subscript_call_result_412035 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), getitem___412034, int_412026)
    
    # Assigning a type to the variable 'tuple_var_assignment_411611' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_411611', subscript_call_result_412035)
    
    # Assigning a Subscript to a Name (line 345):
    
    # Obtaining the type of the subscript
    int_412036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 8), 'int')
    
    # Call to _sym_ortho(...): (line 345)
    # Processing the call arguments (line 345)
    # Getting the type of 'cbar' (line 345)
    cbar_412038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 40), 'cbar', False)
    # Getting the type of 'rho' (line 345)
    rho_412039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 47), 'rho', False)
    # Applying the binary operator '*' (line 345)
    result_mul_412040 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 40), '*', cbar_412038, rho_412039)
    
    # Getting the type of 'thetanew' (line 345)
    thetanew_412041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 52), 'thetanew', False)
    # Processing the call keyword arguments (line 345)
    kwargs_412042 = {}
    # Getting the type of '_sym_ortho' (line 345)
    _sym_ortho_412037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 29), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 345)
    _sym_ortho_call_result_412043 = invoke(stypy.reporting.localization.Localization(__file__, 345, 29), _sym_ortho_412037, *[result_mul_412040, thetanew_412041], **kwargs_412042)
    
    # Obtaining the member '__getitem__' of a type (line 345)
    getitem___412044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), _sym_ortho_call_result_412043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 345)
    subscript_call_result_412045 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), getitem___412044, int_412036)
    
    # Assigning a type to the variable 'tuple_var_assignment_411612' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_411612', subscript_call_result_412045)
    
    # Assigning a Subscript to a Name (line 345):
    
    # Obtaining the type of the subscript
    int_412046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 8), 'int')
    
    # Call to _sym_ortho(...): (line 345)
    # Processing the call arguments (line 345)
    # Getting the type of 'cbar' (line 345)
    cbar_412048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 40), 'cbar', False)
    # Getting the type of 'rho' (line 345)
    rho_412049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 47), 'rho', False)
    # Applying the binary operator '*' (line 345)
    result_mul_412050 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 40), '*', cbar_412048, rho_412049)
    
    # Getting the type of 'thetanew' (line 345)
    thetanew_412051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 52), 'thetanew', False)
    # Processing the call keyword arguments (line 345)
    kwargs_412052 = {}
    # Getting the type of '_sym_ortho' (line 345)
    _sym_ortho_412047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 29), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 345)
    _sym_ortho_call_result_412053 = invoke(stypy.reporting.localization.Localization(__file__, 345, 29), _sym_ortho_412047, *[result_mul_412050, thetanew_412051], **kwargs_412052)
    
    # Obtaining the member '__getitem__' of a type (line 345)
    getitem___412054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), _sym_ortho_call_result_412053, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 345)
    subscript_call_result_412055 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), getitem___412054, int_412046)
    
    # Assigning a type to the variable 'tuple_var_assignment_411613' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_411613', subscript_call_result_412055)
    
    # Assigning a Name to a Name (line 345):
    # Getting the type of 'tuple_var_assignment_411611' (line 345)
    tuple_var_assignment_411611_412056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_411611')
    # Assigning a type to the variable 'cbar' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'cbar', tuple_var_assignment_411611_412056)
    
    # Assigning a Name to a Name (line 345):
    # Getting the type of 'tuple_var_assignment_411612' (line 345)
    tuple_var_assignment_411612_412057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_411612')
    # Assigning a type to the variable 'sbar' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 14), 'sbar', tuple_var_assignment_411612_412057)
    
    # Assigning a Name to a Name (line 345):
    # Getting the type of 'tuple_var_assignment_411613' (line 345)
    tuple_var_assignment_411613_412058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'tuple_var_assignment_411613')
    # Assigning a type to the variable 'rhobar' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 20), 'rhobar', tuple_var_assignment_411613_412058)
    
    # Assigning a BinOp to a Name (line 346):
    
    # Assigning a BinOp to a Name (line 346):
    # Getting the type of 'cbar' (line 346)
    cbar_412059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'cbar')
    # Getting the type of 'zetabar' (line 346)
    zetabar_412060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 22), 'zetabar')
    # Applying the binary operator '*' (line 346)
    result_mul_412061 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 15), '*', cbar_412059, zetabar_412060)
    
    # Assigning a type to the variable 'zeta' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'zeta', result_mul_412061)
    
    # Assigning a BinOp to a Name (line 347):
    
    # Assigning a BinOp to a Name (line 347):
    
    # Getting the type of 'sbar' (line 347)
    sbar_412062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 20), 'sbar')
    # Applying the 'usub' unary operator (line 347)
    result___neg___412063 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 18), 'usub', sbar_412062)
    
    # Getting the type of 'zetabar' (line 347)
    zetabar_412064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 27), 'zetabar')
    # Applying the binary operator '*' (line 347)
    result_mul_412065 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 18), '*', result___neg___412063, zetabar_412064)
    
    # Assigning a type to the variable 'zetabar' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'zetabar', result_mul_412065)
    
    # Assigning a BinOp to a Name (line 351):
    
    # Assigning a BinOp to a Name (line 351):
    # Getting the type of 'h' (line 351)
    h_412066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 15), 'h')
    # Getting the type of 'thetabar' (line 351)
    thetabar_412067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 20), 'thetabar')
    # Getting the type of 'rho' (line 351)
    rho_412068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 31), 'rho')
    # Applying the binary operator '*' (line 351)
    result_mul_412069 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 20), '*', thetabar_412067, rho_412068)
    
    # Getting the type of 'rhoold' (line 351)
    rhoold_412070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 38), 'rhoold')
    # Getting the type of 'rhobarold' (line 351)
    rhobarold_412071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 47), 'rhobarold')
    # Applying the binary operator '*' (line 351)
    result_mul_412072 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 38), '*', rhoold_412070, rhobarold_412071)
    
    # Applying the binary operator 'div' (line 351)
    result_div_412073 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 35), 'div', result_mul_412069, result_mul_412072)
    
    # Getting the type of 'hbar' (line 351)
    hbar_412074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 61), 'hbar')
    # Applying the binary operator '*' (line 351)
    result_mul_412075 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 19), '*', result_div_412073, hbar_412074)
    
    # Applying the binary operator '-' (line 351)
    result_sub_412076 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 15), '-', h_412066, result_mul_412075)
    
    # Assigning a type to the variable 'hbar' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'hbar', result_sub_412076)
    
    # Assigning a BinOp to a Name (line 352):
    
    # Assigning a BinOp to a Name (line 352):
    # Getting the type of 'x' (line 352)
    x_412077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'x')
    # Getting the type of 'zeta' (line 352)
    zeta_412078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 17), 'zeta')
    # Getting the type of 'rho' (line 352)
    rho_412079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 25), 'rho')
    # Getting the type of 'rhobar' (line 352)
    rhobar_412080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 31), 'rhobar')
    # Applying the binary operator '*' (line 352)
    result_mul_412081 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 25), '*', rho_412079, rhobar_412080)
    
    # Applying the binary operator 'div' (line 352)
    result_div_412082 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 17), 'div', zeta_412078, result_mul_412081)
    
    # Getting the type of 'hbar' (line 352)
    hbar_412083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 42), 'hbar')
    # Applying the binary operator '*' (line 352)
    result_mul_412084 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 16), '*', result_div_412082, hbar_412083)
    
    # Applying the binary operator '+' (line 352)
    result_add_412085 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 12), '+', x_412077, result_mul_412084)
    
    # Assigning a type to the variable 'x' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'x', result_add_412085)
    
    # Assigning a BinOp to a Name (line 353):
    
    # Assigning a BinOp to a Name (line 353):
    # Getting the type of 'v' (line 353)
    v_412086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'v')
    # Getting the type of 'thetanew' (line 353)
    thetanew_412087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 17), 'thetanew')
    # Getting the type of 'rho' (line 353)
    rho_412088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 28), 'rho')
    # Applying the binary operator 'div' (line 353)
    result_div_412089 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 17), 'div', thetanew_412087, rho_412088)
    
    # Getting the type of 'h' (line 353)
    h_412090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 35), 'h')
    # Applying the binary operator '*' (line 353)
    result_mul_412091 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 16), '*', result_div_412089, h_412090)
    
    # Applying the binary operator '-' (line 353)
    result_sub_412092 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 12), '-', v_412086, result_mul_412091)
    
    # Assigning a type to the variable 'h' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'h', result_sub_412092)
    
    # Assigning a BinOp to a Name (line 358):
    
    # Assigning a BinOp to a Name (line 358):
    # Getting the type of 'chat' (line 358)
    chat_412093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 20), 'chat')
    # Getting the type of 'betadd' (line 358)
    betadd_412094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'betadd')
    # Applying the binary operator '*' (line 358)
    result_mul_412095 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 20), '*', chat_412093, betadd_412094)
    
    # Assigning a type to the variable 'betaacute' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'betaacute', result_mul_412095)
    
    # Assigning a BinOp to a Name (line 359):
    
    # Assigning a BinOp to a Name (line 359):
    
    # Getting the type of 'shat' (line 359)
    shat_412096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 21), 'shat')
    # Applying the 'usub' unary operator (line 359)
    result___neg___412097 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 20), 'usub', shat_412096)
    
    # Getting the type of 'betadd' (line 359)
    betadd_412098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 28), 'betadd')
    # Applying the binary operator '*' (line 359)
    result_mul_412099 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 20), '*', result___neg___412097, betadd_412098)
    
    # Assigning a type to the variable 'betacheck' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'betacheck', result_mul_412099)
    
    # Assigning a BinOp to a Name (line 362):
    
    # Assigning a BinOp to a Name (line 362):
    # Getting the type of 'c' (line 362)
    c_412100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 18), 'c')
    # Getting the type of 'betaacute' (line 362)
    betaacute_412101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 22), 'betaacute')
    # Applying the binary operator '*' (line 362)
    result_mul_412102 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 18), '*', c_412100, betaacute_412101)
    
    # Assigning a type to the variable 'betahat' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'betahat', result_mul_412102)
    
    # Assigning a BinOp to a Name (line 363):
    
    # Assigning a BinOp to a Name (line 363):
    
    # Getting the type of 's' (line 363)
    s_412103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 18), 's')
    # Applying the 'usub' unary operator (line 363)
    result___neg___412104 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 17), 'usub', s_412103)
    
    # Getting the type of 'betaacute' (line 363)
    betaacute_412105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 22), 'betaacute')
    # Applying the binary operator '*' (line 363)
    result_mul_412106 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 17), '*', result___neg___412104, betaacute_412105)
    
    # Assigning a type to the variable 'betadd' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'betadd', result_mul_412106)
    
    # Assigning a Name to a Name (line 368):
    
    # Assigning a Name to a Name (line 368):
    # Getting the type of 'thetatilde' (line 368)
    thetatilde_412107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 24), 'thetatilde')
    # Assigning a type to the variable 'thetatildeold' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'thetatildeold', thetatilde_412107)
    
    # Assigning a Call to a Tuple (line 369):
    
    # Assigning a Subscript to a Name (line 369):
    
    # Obtaining the type of the subscript
    int_412108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 8), 'int')
    
    # Call to _sym_ortho(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'rhodold' (line 369)
    rhodold_412110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 55), 'rhodold', False)
    # Getting the type of 'thetabar' (line 369)
    thetabar_412111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 64), 'thetabar', False)
    # Processing the call keyword arguments (line 369)
    kwargs_412112 = {}
    # Getting the type of '_sym_ortho' (line 369)
    _sym_ortho_412109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 44), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 369)
    _sym_ortho_call_result_412113 = invoke(stypy.reporting.localization.Localization(__file__, 369, 44), _sym_ortho_412109, *[rhodold_412110, thetabar_412111], **kwargs_412112)
    
    # Obtaining the member '__getitem__' of a type (line 369)
    getitem___412114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), _sym_ortho_call_result_412113, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 369)
    subscript_call_result_412115 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), getitem___412114, int_412108)
    
    # Assigning a type to the variable 'tuple_var_assignment_411614' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'tuple_var_assignment_411614', subscript_call_result_412115)
    
    # Assigning a Subscript to a Name (line 369):
    
    # Obtaining the type of the subscript
    int_412116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 8), 'int')
    
    # Call to _sym_ortho(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'rhodold' (line 369)
    rhodold_412118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 55), 'rhodold', False)
    # Getting the type of 'thetabar' (line 369)
    thetabar_412119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 64), 'thetabar', False)
    # Processing the call keyword arguments (line 369)
    kwargs_412120 = {}
    # Getting the type of '_sym_ortho' (line 369)
    _sym_ortho_412117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 44), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 369)
    _sym_ortho_call_result_412121 = invoke(stypy.reporting.localization.Localization(__file__, 369, 44), _sym_ortho_412117, *[rhodold_412118, thetabar_412119], **kwargs_412120)
    
    # Obtaining the member '__getitem__' of a type (line 369)
    getitem___412122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), _sym_ortho_call_result_412121, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 369)
    subscript_call_result_412123 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), getitem___412122, int_412116)
    
    # Assigning a type to the variable 'tuple_var_assignment_411615' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'tuple_var_assignment_411615', subscript_call_result_412123)
    
    # Assigning a Subscript to a Name (line 369):
    
    # Obtaining the type of the subscript
    int_412124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 8), 'int')
    
    # Call to _sym_ortho(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'rhodold' (line 369)
    rhodold_412126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 55), 'rhodold', False)
    # Getting the type of 'thetabar' (line 369)
    thetabar_412127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 64), 'thetabar', False)
    # Processing the call keyword arguments (line 369)
    kwargs_412128 = {}
    # Getting the type of '_sym_ortho' (line 369)
    _sym_ortho_412125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 44), '_sym_ortho', False)
    # Calling _sym_ortho(args, kwargs) (line 369)
    _sym_ortho_call_result_412129 = invoke(stypy.reporting.localization.Localization(__file__, 369, 44), _sym_ortho_412125, *[rhodold_412126, thetabar_412127], **kwargs_412128)
    
    # Obtaining the member '__getitem__' of a type (line 369)
    getitem___412130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), _sym_ortho_call_result_412129, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 369)
    subscript_call_result_412131 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), getitem___412130, int_412124)
    
    # Assigning a type to the variable 'tuple_var_assignment_411616' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'tuple_var_assignment_411616', subscript_call_result_412131)
    
    # Assigning a Name to a Name (line 369):
    # Getting the type of 'tuple_var_assignment_411614' (line 369)
    tuple_var_assignment_411614_412132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'tuple_var_assignment_411614')
    # Assigning a type to the variable 'ctildeold' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'ctildeold', tuple_var_assignment_411614_412132)
    
    # Assigning a Name to a Name (line 369):
    # Getting the type of 'tuple_var_assignment_411615' (line 369)
    tuple_var_assignment_411615_412133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'tuple_var_assignment_411615')
    # Assigning a type to the variable 'stildeold' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 'stildeold', tuple_var_assignment_411615_412133)
    
    # Assigning a Name to a Name (line 369):
    # Getting the type of 'tuple_var_assignment_411616' (line 369)
    tuple_var_assignment_411616_412134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'tuple_var_assignment_411616')
    # Assigning a type to the variable 'rhotildeold' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 30), 'rhotildeold', tuple_var_assignment_411616_412134)
    
    # Assigning a BinOp to a Name (line 370):
    
    # Assigning a BinOp to a Name (line 370):
    # Getting the type of 'stildeold' (line 370)
    stildeold_412135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 21), 'stildeold')
    # Getting the type of 'rhobar' (line 370)
    rhobar_412136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 33), 'rhobar')
    # Applying the binary operator '*' (line 370)
    result_mul_412137 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 21), '*', stildeold_412135, rhobar_412136)
    
    # Assigning a type to the variable 'thetatilde' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'thetatilde', result_mul_412137)
    
    # Assigning a BinOp to a Name (line 371):
    
    # Assigning a BinOp to a Name (line 371):
    # Getting the type of 'ctildeold' (line 371)
    ctildeold_412138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 18), 'ctildeold')
    # Getting the type of 'rhobar' (line 371)
    rhobar_412139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 30), 'rhobar')
    # Applying the binary operator '*' (line 371)
    result_mul_412140 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 18), '*', ctildeold_412138, rhobar_412139)
    
    # Assigning a type to the variable 'rhodold' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'rhodold', result_mul_412140)
    
    # Assigning a BinOp to a Name (line 372):
    
    # Assigning a BinOp to a Name (line 372):
    
    # Getting the type of 'stildeold' (line 372)
    stildeold_412141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 18), 'stildeold')
    # Applying the 'usub' unary operator (line 372)
    result___neg___412142 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 16), 'usub', stildeold_412141)
    
    # Getting the type of 'betad' (line 372)
    betad_412143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 30), 'betad')
    # Applying the binary operator '*' (line 372)
    result_mul_412144 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 16), '*', result___neg___412142, betad_412143)
    
    # Getting the type of 'ctildeold' (line 372)
    ctildeold_412145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 38), 'ctildeold')
    # Getting the type of 'betahat' (line 372)
    betahat_412146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 50), 'betahat')
    # Applying the binary operator '*' (line 372)
    result_mul_412147 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 38), '*', ctildeold_412145, betahat_412146)
    
    # Applying the binary operator '+' (line 372)
    result_add_412148 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 16), '+', result_mul_412144, result_mul_412147)
    
    # Assigning a type to the variable 'betad' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'betad', result_add_412148)
    
    # Assigning a BinOp to a Name (line 377):
    
    # Assigning a BinOp to a Name (line 377):
    # Getting the type of 'zetaold' (line 377)
    zetaold_412149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 23), 'zetaold')
    # Getting the type of 'thetatildeold' (line 377)
    thetatildeold_412150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 33), 'thetatildeold')
    # Getting the type of 'tautildeold' (line 377)
    tautildeold_412151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 49), 'tautildeold')
    # Applying the binary operator '*' (line 377)
    result_mul_412152 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 33), '*', thetatildeold_412150, tautildeold_412151)
    
    # Applying the binary operator '-' (line 377)
    result_sub_412153 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 23), '-', zetaold_412149, result_mul_412152)
    
    # Getting the type of 'rhotildeold' (line 377)
    rhotildeold_412154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 64), 'rhotildeold')
    # Applying the binary operator 'div' (line 377)
    result_div_412155 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 22), 'div', result_sub_412153, rhotildeold_412154)
    
    # Assigning a type to the variable 'tautildeold' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'tautildeold', result_div_412155)
    
    # Assigning a BinOp to a Name (line 378):
    
    # Assigning a BinOp to a Name (line 378):
    # Getting the type of 'zeta' (line 378)
    zeta_412156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 16), 'zeta')
    # Getting the type of 'thetatilde' (line 378)
    thetatilde_412157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 23), 'thetatilde')
    # Getting the type of 'tautildeold' (line 378)
    tautildeold_412158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 36), 'tautildeold')
    # Applying the binary operator '*' (line 378)
    result_mul_412159 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 23), '*', thetatilde_412157, tautildeold_412158)
    
    # Applying the binary operator '-' (line 378)
    result_sub_412160 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 16), '-', zeta_412156, result_mul_412159)
    
    # Getting the type of 'rhodold' (line 378)
    rhodold_412161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 51), 'rhodold')
    # Applying the binary operator 'div' (line 378)
    result_div_412162 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 15), 'div', result_sub_412160, rhodold_412161)
    
    # Assigning a type to the variable 'taud' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'taud', result_div_412162)
    
    # Assigning a BinOp to a Name (line 379):
    
    # Assigning a BinOp to a Name (line 379):
    # Getting the type of 'd' (line 379)
    d_412163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'd')
    # Getting the type of 'betacheck' (line 379)
    betacheck_412164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 16), 'betacheck')
    # Getting the type of 'betacheck' (line 379)
    betacheck_412165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 28), 'betacheck')
    # Applying the binary operator '*' (line 379)
    result_mul_412166 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 16), '*', betacheck_412164, betacheck_412165)
    
    # Applying the binary operator '+' (line 379)
    result_add_412167 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 12), '+', d_412163, result_mul_412166)
    
    # Assigning a type to the variable 'd' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'd', result_add_412167)
    
    # Assigning a Call to a Name (line 380):
    
    # Assigning a Call to a Name (line 380):
    
    # Call to sqrt(...): (line 380)
    # Processing the call arguments (line 380)
    # Getting the type of 'd' (line 380)
    d_412169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), 'd', False)
    # Getting the type of 'betad' (line 380)
    betad_412170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 26), 'betad', False)
    # Getting the type of 'taud' (line 380)
    taud_412171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 34), 'taud', False)
    # Applying the binary operator '-' (line 380)
    result_sub_412172 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 26), '-', betad_412170, taud_412171)
    
    int_412173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 41), 'int')
    # Applying the binary operator '**' (line 380)
    result_pow_412174 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 25), '**', result_sub_412172, int_412173)
    
    # Applying the binary operator '+' (line 380)
    result_add_412175 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 21), '+', d_412169, result_pow_412174)
    
    # Getting the type of 'betadd' (line 380)
    betadd_412176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 45), 'betadd', False)
    # Getting the type of 'betadd' (line 380)
    betadd_412177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 54), 'betadd', False)
    # Applying the binary operator '*' (line 380)
    result_mul_412178 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 45), '*', betadd_412176, betadd_412177)
    
    # Applying the binary operator '+' (line 380)
    result_add_412179 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 43), '+', result_add_412175, result_mul_412178)
    
    # Processing the call keyword arguments (line 380)
    kwargs_412180 = {}
    # Getting the type of 'sqrt' (line 380)
    sqrt_412168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 380)
    sqrt_call_result_412181 = invoke(stypy.reporting.localization.Localization(__file__, 380, 16), sqrt_412168, *[result_add_412179], **kwargs_412180)
    
    # Assigning a type to the variable 'normr' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'normr', sqrt_call_result_412181)
    
    # Assigning a BinOp to a Name (line 383):
    
    # Assigning a BinOp to a Name (line 383):
    # Getting the type of 'normA2' (line 383)
    normA2_412182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 17), 'normA2')
    # Getting the type of 'beta' (line 383)
    beta_412183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 26), 'beta')
    # Getting the type of 'beta' (line 383)
    beta_412184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 33), 'beta')
    # Applying the binary operator '*' (line 383)
    result_mul_412185 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 26), '*', beta_412183, beta_412184)
    
    # Applying the binary operator '+' (line 383)
    result_add_412186 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 17), '+', normA2_412182, result_mul_412185)
    
    # Assigning a type to the variable 'normA2' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'normA2', result_add_412186)
    
    # Assigning a Call to a Name (line 384):
    
    # Assigning a Call to a Name (line 384):
    
    # Call to sqrt(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'normA2' (line 384)
    normA2_412188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 21), 'normA2', False)
    # Processing the call keyword arguments (line 384)
    kwargs_412189 = {}
    # Getting the type of 'sqrt' (line 384)
    sqrt_412187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 384)
    sqrt_call_result_412190 = invoke(stypy.reporting.localization.Localization(__file__, 384, 16), sqrt_412187, *[normA2_412188], **kwargs_412189)
    
    # Assigning a type to the variable 'normA' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'normA', sqrt_call_result_412190)
    
    # Assigning a BinOp to a Name (line 385):
    
    # Assigning a BinOp to a Name (line 385):
    # Getting the type of 'normA2' (line 385)
    normA2_412191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 17), 'normA2')
    # Getting the type of 'alpha' (line 385)
    alpha_412192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 26), 'alpha')
    # Getting the type of 'alpha' (line 385)
    alpha_412193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 34), 'alpha')
    # Applying the binary operator '*' (line 385)
    result_mul_412194 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 26), '*', alpha_412192, alpha_412193)
    
    # Applying the binary operator '+' (line 385)
    result_add_412195 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 17), '+', normA2_412191, result_mul_412194)
    
    # Assigning a type to the variable 'normA2' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'normA2', result_add_412195)
    
    # Assigning a Call to a Name (line 388):
    
    # Assigning a Call to a Name (line 388):
    
    # Call to max(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'maxrbar' (line 388)
    maxrbar_412197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 22), 'maxrbar', False)
    # Getting the type of 'rhobarold' (line 388)
    rhobarold_412198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 31), 'rhobarold', False)
    # Processing the call keyword arguments (line 388)
    kwargs_412199 = {}
    # Getting the type of 'max' (line 388)
    max_412196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 18), 'max', False)
    # Calling max(args, kwargs) (line 388)
    max_call_result_412200 = invoke(stypy.reporting.localization.Localization(__file__, 388, 18), max_412196, *[maxrbar_412197, rhobarold_412198], **kwargs_412199)
    
    # Assigning a type to the variable 'maxrbar' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'maxrbar', max_call_result_412200)
    
    
    # Getting the type of 'itn' (line 389)
    itn_412201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), 'itn')
    int_412202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 17), 'int')
    # Applying the binary operator '>' (line 389)
    result_gt_412203 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 11), '>', itn_412201, int_412202)
    
    # Testing the type of an if condition (line 389)
    if_condition_412204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 8), result_gt_412203)
    # Assigning a type to the variable 'if_condition_412204' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'if_condition_412204', if_condition_412204)
    # SSA begins for if statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 390):
    
    # Assigning a Call to a Name (line 390):
    
    # Call to min(...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'minrbar' (line 390)
    minrbar_412206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 26), 'minrbar', False)
    # Getting the type of 'rhobarold' (line 390)
    rhobarold_412207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 35), 'rhobarold', False)
    # Processing the call keyword arguments (line 390)
    kwargs_412208 = {}
    # Getting the type of 'min' (line 390)
    min_412205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 22), 'min', False)
    # Calling min(args, kwargs) (line 390)
    min_call_result_412209 = invoke(stypy.reporting.localization.Localization(__file__, 390, 22), min_412205, *[minrbar_412206, rhobarold_412207], **kwargs_412208)
    
    # Assigning a type to the variable 'minrbar' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'minrbar', min_call_result_412209)
    # SSA join for if statement (line 389)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 391):
    
    # Assigning a BinOp to a Name (line 391):
    
    # Call to max(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'maxrbar' (line 391)
    maxrbar_412211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'maxrbar', False)
    # Getting the type of 'rhotemp' (line 391)
    rhotemp_412212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 29), 'rhotemp', False)
    # Processing the call keyword arguments (line 391)
    kwargs_412213 = {}
    # Getting the type of 'max' (line 391)
    max_412210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'max', False)
    # Calling max(args, kwargs) (line 391)
    max_call_result_412214 = invoke(stypy.reporting.localization.Localization(__file__, 391, 16), max_412210, *[maxrbar_412211, rhotemp_412212], **kwargs_412213)
    
    
    # Call to min(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'minrbar' (line 391)
    minrbar_412216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 44), 'minrbar', False)
    # Getting the type of 'rhotemp' (line 391)
    rhotemp_412217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 53), 'rhotemp', False)
    # Processing the call keyword arguments (line 391)
    kwargs_412218 = {}
    # Getting the type of 'min' (line 391)
    min_412215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 40), 'min', False)
    # Calling min(args, kwargs) (line 391)
    min_call_result_412219 = invoke(stypy.reporting.localization.Localization(__file__, 391, 40), min_412215, *[minrbar_412216, rhotemp_412217], **kwargs_412218)
    
    # Applying the binary operator 'div' (line 391)
    result_div_412220 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 16), 'div', max_call_result_412214, min_call_result_412219)
    
    # Assigning a type to the variable 'condA' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'condA', result_div_412220)
    
    # Assigning a Call to a Name (line 396):
    
    # Assigning a Call to a Name (line 396):
    
    # Call to abs(...): (line 396)
    # Processing the call arguments (line 396)
    # Getting the type of 'zetabar' (line 396)
    zetabar_412222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 21), 'zetabar', False)
    # Processing the call keyword arguments (line 396)
    kwargs_412223 = {}
    # Getting the type of 'abs' (line 396)
    abs_412221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), 'abs', False)
    # Calling abs(args, kwargs) (line 396)
    abs_call_result_412224 = invoke(stypy.reporting.localization.Localization(__file__, 396, 17), abs_412221, *[zetabar_412222], **kwargs_412223)
    
    # Assigning a type to the variable 'normar' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'normar', abs_call_result_412224)
    
    # Assigning a Call to a Name (line 397):
    
    # Assigning a Call to a Name (line 397):
    
    # Call to norm(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of 'x' (line 397)
    x_412226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 21), 'x', False)
    # Processing the call keyword arguments (line 397)
    kwargs_412227 = {}
    # Getting the type of 'norm' (line 397)
    norm_412225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'norm', False)
    # Calling norm(args, kwargs) (line 397)
    norm_call_result_412228 = invoke(stypy.reporting.localization.Localization(__file__, 397, 16), norm_412225, *[x_412226], **kwargs_412227)
    
    # Assigning a type to the variable 'normx' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'normx', norm_call_result_412228)
    
    # Assigning a BinOp to a Name (line 402):
    
    # Assigning a BinOp to a Name (line 402):
    # Getting the type of 'normr' (line 402)
    normr_412229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 16), 'normr')
    # Getting the type of 'normb' (line 402)
    normb_412230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 24), 'normb')
    # Applying the binary operator 'div' (line 402)
    result_div_412231 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 16), 'div', normr_412229, normb_412230)
    
    # Assigning a type to the variable 'test1' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'test1', result_div_412231)
    
    
    # Getting the type of 'normA' (line 403)
    normA_412232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'normA')
    # Getting the type of 'normr' (line 403)
    normr_412233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 20), 'normr')
    # Applying the binary operator '*' (line 403)
    result_mul_412234 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 12), '*', normA_412232, normr_412233)
    
    int_412235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 30), 'int')
    # Applying the binary operator '!=' (line 403)
    result_ne_412236 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 11), '!=', result_mul_412234, int_412235)
    
    # Testing the type of an if condition (line 403)
    if_condition_412237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 8), result_ne_412236)
    # Assigning a type to the variable 'if_condition_412237' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'if_condition_412237', if_condition_412237)
    # SSA begins for if statement (line 403)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 404):
    
    # Assigning a BinOp to a Name (line 404):
    # Getting the type of 'normar' (line 404)
    normar_412238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 20), 'normar')
    # Getting the type of 'normA' (line 404)
    normA_412239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 30), 'normA')
    # Getting the type of 'normr' (line 404)
    normr_412240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 38), 'normr')
    # Applying the binary operator '*' (line 404)
    result_mul_412241 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 30), '*', normA_412239, normr_412240)
    
    # Applying the binary operator 'div' (line 404)
    result_div_412242 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 20), 'div', normar_412238, result_mul_412241)
    
    # Assigning a type to the variable 'test2' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'test2', result_div_412242)
    # SSA branch for the else part of an if statement (line 403)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 406):
    
    # Assigning a Name to a Name (line 406):
    # Getting the type of 'infty' (line 406)
    infty_412243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'infty')
    # Assigning a type to the variable 'test2' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'test2', infty_412243)
    # SSA join for if statement (line 403)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 407):
    
    # Assigning a BinOp to a Name (line 407):
    int_412244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 16), 'int')
    # Getting the type of 'condA' (line 407)
    condA_412245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 20), 'condA')
    # Applying the binary operator 'div' (line 407)
    result_div_412246 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 16), 'div', int_412244, condA_412245)
    
    # Assigning a type to the variable 'test3' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'test3', result_div_412246)
    
    # Assigning a BinOp to a Name (line 408):
    
    # Assigning a BinOp to a Name (line 408):
    # Getting the type of 'test1' (line 408)
    test1_412247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 13), 'test1')
    int_412248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 22), 'int')
    # Getting the type of 'normA' (line 408)
    normA_412249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 26), 'normA')
    # Getting the type of 'normx' (line 408)
    normx_412250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 34), 'normx')
    # Applying the binary operator '*' (line 408)
    result_mul_412251 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 26), '*', normA_412249, normx_412250)
    
    # Getting the type of 'normb' (line 408)
    normb_412252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 42), 'normb')
    # Applying the binary operator 'div' (line 408)
    result_div_412253 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 40), 'div', result_mul_412251, normb_412252)
    
    # Applying the binary operator '+' (line 408)
    result_add_412254 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 22), '+', int_412248, result_div_412253)
    
    # Applying the binary operator 'div' (line 408)
    result_div_412255 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 13), 'div', test1_412247, result_add_412254)
    
    # Assigning a type to the variable 't1' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 't1', result_div_412255)
    
    # Assigning a BinOp to a Name (line 409):
    
    # Assigning a BinOp to a Name (line 409):
    # Getting the type of 'btol' (line 409)
    btol_412256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 15), 'btol')
    # Getting the type of 'atol' (line 409)
    atol_412257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 22), 'atol')
    # Getting the type of 'normA' (line 409)
    normA_412258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 29), 'normA')
    # Applying the binary operator '*' (line 409)
    result_mul_412259 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 22), '*', atol_412257, normA_412258)
    
    # Getting the type of 'normx' (line 409)
    normx_412260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 37), 'normx')
    # Applying the binary operator '*' (line 409)
    result_mul_412261 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 35), '*', result_mul_412259, normx_412260)
    
    # Getting the type of 'normb' (line 409)
    normb_412262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 45), 'normb')
    # Applying the binary operator 'div' (line 409)
    result_div_412263 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 43), 'div', result_mul_412261, normb_412262)
    
    # Applying the binary operator '+' (line 409)
    result_add_412264 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 15), '+', btol_412256, result_div_412263)
    
    # Assigning a type to the variable 'rtol' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'rtol', result_add_412264)
    
    
    # Getting the type of 'itn' (line 417)
    itn_412265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 11), 'itn')
    # Getting the type of 'maxiter' (line 417)
    maxiter_412266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'maxiter')
    # Applying the binary operator '>=' (line 417)
    result_ge_412267 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 11), '>=', itn_412265, maxiter_412266)
    
    # Testing the type of an if condition (line 417)
    if_condition_412268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 417, 8), result_ge_412267)
    # Assigning a type to the variable 'if_condition_412268' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'if_condition_412268', if_condition_412268)
    # SSA begins for if statement (line 417)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 418):
    
    # Assigning a Num to a Name (line 418):
    int_412269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 20), 'int')
    # Assigning a type to the variable 'istop' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'istop', int_412269)
    # SSA join for if statement (line 417)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    int_412270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 11), 'int')
    # Getting the type of 'test3' (line 419)
    test3_412271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 15), 'test3')
    # Applying the binary operator '+' (line 419)
    result_add_412272 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 11), '+', int_412270, test3_412271)
    
    int_412273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 24), 'int')
    # Applying the binary operator '<=' (line 419)
    result_le_412274 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 11), '<=', result_add_412272, int_412273)
    
    # Testing the type of an if condition (line 419)
    if_condition_412275 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 8), result_le_412274)
    # Assigning a type to the variable 'if_condition_412275' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'if_condition_412275', if_condition_412275)
    # SSA begins for if statement (line 419)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 420):
    
    # Assigning a Num to a Name (line 420):
    int_412276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 20), 'int')
    # Assigning a type to the variable 'istop' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'istop', int_412276)
    # SSA join for if statement (line 419)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    int_412277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 11), 'int')
    # Getting the type of 'test2' (line 421)
    test2_412278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), 'test2')
    # Applying the binary operator '+' (line 421)
    result_add_412279 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 11), '+', int_412277, test2_412278)
    
    int_412280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 24), 'int')
    # Applying the binary operator '<=' (line 421)
    result_le_412281 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 11), '<=', result_add_412279, int_412280)
    
    # Testing the type of an if condition (line 421)
    if_condition_412282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 421, 8), result_le_412281)
    # Assigning a type to the variable 'if_condition_412282' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'if_condition_412282', if_condition_412282)
    # SSA begins for if statement (line 421)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 422):
    
    # Assigning a Num to a Name (line 422):
    int_412283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 20), 'int')
    # Assigning a type to the variable 'istop' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'istop', int_412283)
    # SSA join for if statement (line 421)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    int_412284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 11), 'int')
    # Getting the type of 't1' (line 423)
    t1_412285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 't1')
    # Applying the binary operator '+' (line 423)
    result_add_412286 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 11), '+', int_412284, t1_412285)
    
    int_412287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 21), 'int')
    # Applying the binary operator '<=' (line 423)
    result_le_412288 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 11), '<=', result_add_412286, int_412287)
    
    # Testing the type of an if condition (line 423)
    if_condition_412289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 8), result_le_412288)
    # Assigning a type to the variable 'if_condition_412289' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'if_condition_412289', if_condition_412289)
    # SSA begins for if statement (line 423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 424):
    
    # Assigning a Num to a Name (line 424):
    int_412290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 20), 'int')
    # Assigning a type to the variable 'istop' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'istop', int_412290)
    # SSA join for if statement (line 423)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'test3' (line 428)
    test3_412291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 11), 'test3')
    # Getting the type of 'ctol' (line 428)
    ctol_412292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 20), 'ctol')
    # Applying the binary operator '<=' (line 428)
    result_le_412293 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 11), '<=', test3_412291, ctol_412292)
    
    # Testing the type of an if condition (line 428)
    if_condition_412294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 8), result_le_412293)
    # Assigning a type to the variable 'if_condition_412294' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'if_condition_412294', if_condition_412294)
    # SSA begins for if statement (line 428)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 429):
    
    # Assigning a Num to a Name (line 429):
    int_412295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 20), 'int')
    # Assigning a type to the variable 'istop' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'istop', int_412295)
    # SSA join for if statement (line 428)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'test2' (line 430)
    test2_412296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 11), 'test2')
    # Getting the type of 'atol' (line 430)
    atol_412297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 20), 'atol')
    # Applying the binary operator '<=' (line 430)
    result_le_412298 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 11), '<=', test2_412296, atol_412297)
    
    # Testing the type of an if condition (line 430)
    if_condition_412299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 430, 8), result_le_412298)
    # Assigning a type to the variable 'if_condition_412299' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'if_condition_412299', if_condition_412299)
    # SSA begins for if statement (line 430)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 431):
    
    # Assigning a Num to a Name (line 431):
    int_412300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 20), 'int')
    # Assigning a type to the variable 'istop' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'istop', int_412300)
    # SSA join for if statement (line 430)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'test1' (line 432)
    test1_412301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 11), 'test1')
    # Getting the type of 'rtol' (line 432)
    rtol_412302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 20), 'rtol')
    # Applying the binary operator '<=' (line 432)
    result_le_412303 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 11), '<=', test1_412301, rtol_412302)
    
    # Testing the type of an if condition (line 432)
    if_condition_412304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 8), result_le_412303)
    # Assigning a type to the variable 'if_condition_412304' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'if_condition_412304', if_condition_412304)
    # SSA begins for if statement (line 432)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 433):
    
    # Assigning a Num to a Name (line 433):
    int_412305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 20), 'int')
    # Assigning a type to the variable 'istop' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'istop', int_412305)
    # SSA join for if statement (line 432)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'show' (line 437)
    show_412306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'show')
    # Testing the type of an if condition (line 437)
    if_condition_412307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 8), show_412306)
    # Assigning a type to the variable 'if_condition_412307' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'if_condition_412307', if_condition_412307)
    # SSA begins for if statement (line 437)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 438)
    n_412308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'n')
    int_412309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 21), 'int')
    # Applying the binary operator '<=' (line 438)
    result_le_412310 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 16), '<=', n_412308, int_412309)
    
    
    # Getting the type of 'itn' (line 438)
    itn_412311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 29), 'itn')
    int_412312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 36), 'int')
    # Applying the binary operator '<=' (line 438)
    result_le_412313 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 29), '<=', itn_412311, int_412312)
    
    # Applying the binary operator 'or' (line 438)
    result_or_keyword_412314 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), 'or', result_le_412310, result_le_412313)
    
    # Getting the type of 'itn' (line 438)
    itn_412315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 44), 'itn')
    # Getting the type of 'maxiter' (line 438)
    maxiter_412316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 51), 'maxiter')
    int_412317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 61), 'int')
    # Applying the binary operator '-' (line 438)
    result_sub_412318 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 51), '-', maxiter_412316, int_412317)
    
    # Applying the binary operator '>=' (line 438)
    result_ge_412319 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 44), '>=', itn_412315, result_sub_412318)
    
    # Applying the binary operator 'or' (line 438)
    result_or_keyword_412320 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), 'or', result_or_keyword_412314, result_ge_412319)
    
    # Getting the type of 'itn' (line 439)
    itn_412321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'itn')
    int_412322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 22), 'int')
    # Applying the binary operator '%' (line 439)
    result_mod_412323 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 16), '%', itn_412321, int_412322)
    
    int_412324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 28), 'int')
    # Applying the binary operator '==' (line 439)
    result_eq_412325 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 16), '==', result_mod_412323, int_412324)
    
    # Applying the binary operator 'or' (line 438)
    result_or_keyword_412326 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), 'or', result_or_keyword_412320, result_eq_412325)
    
    # Getting the type of 'test3' (line 439)
    test3_412327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 35), 'test3')
    float_412328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 44), 'float')
    # Getting the type of 'ctol' (line 439)
    ctol_412329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 50), 'ctol')
    # Applying the binary operator '*' (line 439)
    result_mul_412330 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 44), '*', float_412328, ctol_412329)
    
    # Applying the binary operator '<=' (line 439)
    result_le_412331 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 35), '<=', test3_412327, result_mul_412330)
    
    # Applying the binary operator 'or' (line 438)
    result_or_keyword_412332 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), 'or', result_or_keyword_412326, result_le_412331)
    
    # Getting the type of 'test2' (line 440)
    test2_412333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'test2')
    float_412334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 25), 'float')
    # Getting the type of 'atol' (line 440)
    atol_412335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 31), 'atol')
    # Applying the binary operator '*' (line 440)
    result_mul_412336 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 25), '*', float_412334, atol_412335)
    
    # Applying the binary operator '<=' (line 440)
    result_le_412337 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 16), '<=', test2_412333, result_mul_412336)
    
    # Applying the binary operator 'or' (line 438)
    result_or_keyword_412338 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), 'or', result_or_keyword_412332, result_le_412337)
    
    # Getting the type of 'test1' (line 440)
    test1_412339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 41), 'test1')
    float_412340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 50), 'float')
    # Getting the type of 'rtol' (line 440)
    rtol_412341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 56), 'rtol')
    # Applying the binary operator '*' (line 440)
    result_mul_412342 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 50), '*', float_412340, rtol_412341)
    
    # Applying the binary operator '<=' (line 440)
    result_le_412343 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 41), '<=', test1_412339, result_mul_412342)
    
    # Applying the binary operator 'or' (line 438)
    result_or_keyword_412344 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), 'or', result_or_keyword_412338, result_le_412343)
    
    # Getting the type of 'istop' (line 441)
    istop_412345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'istop')
    int_412346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 25), 'int')
    # Applying the binary operator '!=' (line 441)
    result_ne_412347 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 16), '!=', istop_412345, int_412346)
    
    # Applying the binary operator 'or' (line 438)
    result_or_keyword_412348 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), 'or', result_or_keyword_412344, result_ne_412347)
    
    # Testing the type of an if condition (line 438)
    if_condition_412349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 12), result_or_keyword_412348)
    # Assigning a type to the variable 'if_condition_412349' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'if_condition_412349', if_condition_412349)
    # SSA begins for if statement (line 438)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'pcount' (line 443)
    pcount_412350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 19), 'pcount')
    # Getting the type of 'pfreq' (line 443)
    pfreq_412351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 29), 'pfreq')
    # Applying the binary operator '>=' (line 443)
    result_ge_412352 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 19), '>=', pcount_412350, pfreq_412351)
    
    # Testing the type of an if condition (line 443)
    if_condition_412353 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 16), result_ge_412352)
    # Assigning a type to the variable 'if_condition_412353' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 16), 'if_condition_412353', if_condition_412353)
    # SSA begins for if statement (line 443)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 444):
    
    # Assigning a Num to a Name (line 444):
    int_412354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 29), 'int')
    # Assigning a type to the variable 'pcount' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'pcount', int_412354)
    
    # Call to print(...): (line 445)
    # Processing the call arguments (line 445)
    str_412356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 26), 'str', ' ')
    # Processing the call keyword arguments (line 445)
    kwargs_412357 = {}
    # Getting the type of 'print' (line 445)
    print_412355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'print', False)
    # Calling print(args, kwargs) (line 445)
    print_call_result_412358 = invoke(stypy.reporting.localization.Localization(__file__, 445, 20), print_412355, *[str_412356], **kwargs_412357)
    
    
    # Call to print(...): (line 446)
    # Processing the call arguments (line 446)
    # Getting the type of 'hdg1' (line 446)
    hdg1_412360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 26), 'hdg1', False)
    # Getting the type of 'hdg2' (line 446)
    hdg2_412361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 32), 'hdg2', False)
    # Processing the call keyword arguments (line 446)
    kwargs_412362 = {}
    # Getting the type of 'print' (line 446)
    print_412359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 20), 'print', False)
    # Calling print(args, kwargs) (line 446)
    print_call_result_412363 = invoke(stypy.reporting.localization.Localization(__file__, 446, 20), print_412359, *[hdg1_412360, hdg2_412361], **kwargs_412362)
    
    # SSA join for if statement (line 443)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 447):
    
    # Assigning a BinOp to a Name (line 447):
    # Getting the type of 'pcount' (line 447)
    pcount_412364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 25), 'pcount')
    int_412365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 34), 'int')
    # Applying the binary operator '+' (line 447)
    result_add_412366 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 25), '+', pcount_412364, int_412365)
    
    # Assigning a type to the variable 'pcount' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 16), 'pcount', result_add_412366)
    
    # Assigning a BinOp to a Name (line 448):
    
    # Assigning a BinOp to a Name (line 448):
    str_412367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 23), 'str', '%6g %12.5e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 448)
    tuple_412368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 448)
    # Adding element type (line 448)
    # Getting the type of 'itn' (line 448)
    itn_412369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 39), 'itn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 39), tuple_412368, itn_412369)
    # Adding element type (line 448)
    
    # Obtaining the type of the subscript
    int_412370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 46), 'int')
    # Getting the type of 'x' (line 448)
    x_412371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 44), 'x')
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___412372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 44), x_412371, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_412373 = invoke(stypy.reporting.localization.Localization(__file__, 448, 44), getitem___412372, int_412370)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 39), tuple_412368, subscript_call_result_412373)
    
    # Applying the binary operator '%' (line 448)
    result_mod_412374 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 23), '%', str_412367, tuple_412368)
    
    # Assigning a type to the variable 'str1' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 16), 'str1', result_mod_412374)
    
    # Assigning a BinOp to a Name (line 449):
    
    # Assigning a BinOp to a Name (line 449):
    str_412375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 23), 'str', ' %10.3e %10.3e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 449)
    tuple_412376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 449)
    # Adding element type (line 449)
    # Getting the type of 'normr' (line 449)
    normr_412377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 43), 'normr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 43), tuple_412376, normr_412377)
    # Adding element type (line 449)
    # Getting the type of 'normar' (line 449)
    normar_412378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 50), 'normar')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 43), tuple_412376, normar_412378)
    
    # Applying the binary operator '%' (line 449)
    result_mod_412379 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 23), '%', str_412375, tuple_412376)
    
    # Assigning a type to the variable 'str2' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 16), 'str2', result_mod_412379)
    
    # Assigning a BinOp to a Name (line 450):
    
    # Assigning a BinOp to a Name (line 450):
    str_412380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 23), 'str', '  %8.1e %8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 450)
    tuple_412381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 450)
    # Adding element type (line 450)
    # Getting the type of 'test1' (line 450)
    test1_412382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 42), 'test1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 42), tuple_412381, test1_412382)
    # Adding element type (line 450)
    # Getting the type of 'test2' (line 450)
    test2_412383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 49), 'test2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 42), tuple_412381, test2_412383)
    
    # Applying the binary operator '%' (line 450)
    result_mod_412384 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 23), '%', str_412380, tuple_412381)
    
    # Assigning a type to the variable 'str3' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 'str3', result_mod_412384)
    
    # Assigning a BinOp to a Name (line 451):
    
    # Assigning a BinOp to a Name (line 451):
    str_412385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 23), 'str', ' %8.1e %8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 451)
    tuple_412386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 451)
    # Adding element type (line 451)
    # Getting the type of 'normA' (line 451)
    normA_412387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 41), 'normA')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 41), tuple_412386, normA_412387)
    # Adding element type (line 451)
    # Getting the type of 'condA' (line 451)
    condA_412388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 48), 'condA')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 41), tuple_412386, condA_412388)
    
    # Applying the binary operator '%' (line 451)
    result_mod_412389 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 23), '%', str_412385, tuple_412386)
    
    # Assigning a type to the variable 'str4' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 'str4', result_mod_412389)
    
    # Call to print(...): (line 452)
    # Processing the call arguments (line 452)
    
    # Call to join(...): (line 452)
    # Processing the call arguments (line 452)
    
    # Obtaining an instance of the builtin type 'list' (line 452)
    list_412393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 452)
    # Adding element type (line 452)
    # Getting the type of 'str1' (line 452)
    str1_412394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 31), 'str1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 30), list_412393, str1_412394)
    # Adding element type (line 452)
    # Getting the type of 'str2' (line 452)
    str2_412395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 37), 'str2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 30), list_412393, str2_412395)
    # Adding element type (line 452)
    # Getting the type of 'str3' (line 452)
    str3_412396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 43), 'str3', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 30), list_412393, str3_412396)
    # Adding element type (line 452)
    # Getting the type of 'str4' (line 452)
    str4_412397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 49), 'str4', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 30), list_412393, str4_412397)
    
    # Processing the call keyword arguments (line 452)
    kwargs_412398 = {}
    str_412391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 22), 'str', '')
    # Obtaining the member 'join' of a type (line 452)
    join_412392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 22), str_412391, 'join')
    # Calling join(args, kwargs) (line 452)
    join_call_result_412399 = invoke(stypy.reporting.localization.Localization(__file__, 452, 22), join_412392, *[list_412393], **kwargs_412398)
    
    # Processing the call keyword arguments (line 452)
    kwargs_412400 = {}
    # Getting the type of 'print' (line 452)
    print_412390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 16), 'print', False)
    # Calling print(args, kwargs) (line 452)
    print_call_result_412401 = invoke(stypy.reporting.localization.Localization(__file__, 452, 16), print_412390, *[join_call_result_412399], **kwargs_412400)
    
    # SSA join for if statement (line 438)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 437)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'istop' (line 454)
    istop_412402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 11), 'istop')
    int_412403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 19), 'int')
    # Applying the binary operator '>' (line 454)
    result_gt_412404 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 11), '>', istop_412402, int_412403)
    
    # Testing the type of an if condition (line 454)
    if_condition_412405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 8), result_gt_412404)
    # Assigning a type to the variable 'if_condition_412405' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'if_condition_412405', if_condition_412405)
    # SSA begins for if statement (line 454)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 454)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 308)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'show' (line 459)
    show_412406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 7), 'show')
    # Testing the type of an if condition (line 459)
    if_condition_412407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 4), show_412406)
    # Assigning a type to the variable 'if_condition_412407' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'if_condition_412407', if_condition_412407)
    # SSA begins for if statement (line 459)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 460)
    # Processing the call arguments (line 460)
    str_412409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 14), 'str', ' ')
    # Processing the call keyword arguments (line 460)
    kwargs_412410 = {}
    # Getting the type of 'print' (line 460)
    print_412408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'print', False)
    # Calling print(args, kwargs) (line 460)
    print_call_result_412411 = invoke(stypy.reporting.localization.Localization(__file__, 460, 8), print_412408, *[str_412409], **kwargs_412410)
    
    
    # Call to print(...): (line 461)
    # Processing the call arguments (line 461)
    str_412413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 14), 'str', 'LSMR finished')
    # Processing the call keyword arguments (line 461)
    kwargs_412414 = {}
    # Getting the type of 'print' (line 461)
    print_412412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'print', False)
    # Calling print(args, kwargs) (line 461)
    print_call_result_412415 = invoke(stypy.reporting.localization.Localization(__file__, 461, 8), print_412412, *[str_412413], **kwargs_412414)
    
    
    # Call to print(...): (line 462)
    # Processing the call arguments (line 462)
    
    # Obtaining the type of the subscript
    # Getting the type of 'istop' (line 462)
    istop_412417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 18), 'istop', False)
    # Getting the type of 'msg' (line 462)
    msg_412418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 14), 'msg', False)
    # Obtaining the member '__getitem__' of a type (line 462)
    getitem___412419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 14), msg_412418, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 462)
    subscript_call_result_412420 = invoke(stypy.reporting.localization.Localization(__file__, 462, 14), getitem___412419, istop_412417)
    
    # Processing the call keyword arguments (line 462)
    kwargs_412421 = {}
    # Getting the type of 'print' (line 462)
    print_412416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'print', False)
    # Calling print(args, kwargs) (line 462)
    print_call_result_412422 = invoke(stypy.reporting.localization.Localization(__file__, 462, 8), print_412416, *[subscript_call_result_412420], **kwargs_412421)
    
    
    # Call to print(...): (line 463)
    # Processing the call arguments (line 463)
    str_412424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 14), 'str', 'istop =%8g    normr =%8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 463)
    tuple_412425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 463)
    # Adding element type (line 463)
    # Getting the type of 'istop' (line 463)
    istop_412426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 46), 'istop', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 46), tuple_412425, istop_412426)
    # Adding element type (line 463)
    # Getting the type of 'normr' (line 463)
    normr_412427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 53), 'normr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 46), tuple_412425, normr_412427)
    
    # Applying the binary operator '%' (line 463)
    result_mod_412428 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 14), '%', str_412424, tuple_412425)
    
    # Processing the call keyword arguments (line 463)
    kwargs_412429 = {}
    # Getting the type of 'print' (line 463)
    print_412423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'print', False)
    # Calling print(args, kwargs) (line 463)
    print_call_result_412430 = invoke(stypy.reporting.localization.Localization(__file__, 463, 8), print_412423, *[result_mod_412428], **kwargs_412429)
    
    
    # Call to print(...): (line 464)
    # Processing the call arguments (line 464)
    str_412432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 14), 'str', '    normA =%8.1e    normAr =%8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 464)
    tuple_412433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 464)
    # Adding element type (line 464)
    # Getting the type of 'normA' (line 464)
    normA_412434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 53), 'normA', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 53), tuple_412433, normA_412434)
    # Adding element type (line 464)
    # Getting the type of 'normar' (line 464)
    normar_412435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 60), 'normar', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 53), tuple_412433, normar_412435)
    
    # Applying the binary operator '%' (line 464)
    result_mod_412436 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 14), '%', str_412432, tuple_412433)
    
    # Processing the call keyword arguments (line 464)
    kwargs_412437 = {}
    # Getting the type of 'print' (line 464)
    print_412431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'print', False)
    # Calling print(args, kwargs) (line 464)
    print_call_result_412438 = invoke(stypy.reporting.localization.Localization(__file__, 464, 8), print_412431, *[result_mod_412436], **kwargs_412437)
    
    
    # Call to print(...): (line 465)
    # Processing the call arguments (line 465)
    str_412440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 14), 'str', 'itn   =%8g    condA =%8.1e')
    
    # Obtaining an instance of the builtin type 'tuple' (line 465)
    tuple_412441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 465)
    # Adding element type (line 465)
    # Getting the type of 'itn' (line 465)
    itn_412442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 46), 'itn', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 46), tuple_412441, itn_412442)
    # Adding element type (line 465)
    # Getting the type of 'condA' (line 465)
    condA_412443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 51), 'condA', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 46), tuple_412441, condA_412443)
    
    # Applying the binary operator '%' (line 465)
    result_mod_412444 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 14), '%', str_412440, tuple_412441)
    
    # Processing the call keyword arguments (line 465)
    kwargs_412445 = {}
    # Getting the type of 'print' (line 465)
    print_412439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'print', False)
    # Calling print(args, kwargs) (line 465)
    print_call_result_412446 = invoke(stypy.reporting.localization.Localization(__file__, 465, 8), print_412439, *[result_mod_412444], **kwargs_412445)
    
    
    # Call to print(...): (line 466)
    # Processing the call arguments (line 466)
    str_412448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 14), 'str', '    normx =%8.1e')
    # Getting the type of 'normx' (line 466)
    normx_412449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 36), 'normx', False)
    # Applying the binary operator '%' (line 466)
    result_mod_412450 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 14), '%', str_412448, normx_412449)
    
    # Processing the call keyword arguments (line 466)
    kwargs_412451 = {}
    # Getting the type of 'print' (line 466)
    print_412447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'print', False)
    # Calling print(args, kwargs) (line 466)
    print_call_result_412452 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), print_412447, *[result_mod_412450], **kwargs_412451)
    
    
    # Call to print(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'str1' (line 467)
    str1_412454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 14), 'str1', False)
    # Getting the type of 'str2' (line 467)
    str2_412455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 20), 'str2', False)
    # Processing the call keyword arguments (line 467)
    kwargs_412456 = {}
    # Getting the type of 'print' (line 467)
    print_412453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'print', False)
    # Calling print(args, kwargs) (line 467)
    print_call_result_412457 = invoke(stypy.reporting.localization.Localization(__file__, 467, 8), print_412453, *[str1_412454, str2_412455], **kwargs_412456)
    
    
    # Call to print(...): (line 468)
    # Processing the call arguments (line 468)
    # Getting the type of 'str3' (line 468)
    str3_412459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 14), 'str3', False)
    # Getting the type of 'str4' (line 468)
    str4_412460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 20), 'str4', False)
    # Processing the call keyword arguments (line 468)
    kwargs_412461 = {}
    # Getting the type of 'print' (line 468)
    print_412458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'print', False)
    # Calling print(args, kwargs) (line 468)
    print_call_result_412462 = invoke(stypy.reporting.localization.Localization(__file__, 468, 8), print_412458, *[str3_412459, str4_412460], **kwargs_412461)
    
    # SSA join for if statement (line 459)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 470)
    tuple_412463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 470)
    # Adding element type (line 470)
    # Getting the type of 'x' (line 470)
    x_412464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 11), tuple_412463, x_412464)
    # Adding element type (line 470)
    # Getting the type of 'istop' (line 470)
    istop_412465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 14), 'istop')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 11), tuple_412463, istop_412465)
    # Adding element type (line 470)
    # Getting the type of 'itn' (line 470)
    itn_412466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 21), 'itn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 11), tuple_412463, itn_412466)
    # Adding element type (line 470)
    # Getting the type of 'normr' (line 470)
    normr_412467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'normr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 11), tuple_412463, normr_412467)
    # Adding element type (line 470)
    # Getting the type of 'normar' (line 470)
    normar_412468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 33), 'normar')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 11), tuple_412463, normar_412468)
    # Adding element type (line 470)
    # Getting the type of 'normA' (line 470)
    normA_412469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 41), 'normA')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 11), tuple_412463, normA_412469)
    # Adding element type (line 470)
    # Getting the type of 'condA' (line 470)
    condA_412470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 48), 'condA')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 11), tuple_412463, condA_412470)
    # Adding element type (line 470)
    # Getting the type of 'normx' (line 470)
    normx_412471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 55), 'normx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 11), tuple_412463, normx_412471)
    
    # Assigning a type to the variable 'stypy_return_type' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'stypy_return_type', tuple_412463)
    
    # ################# End of 'lsmr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lsmr' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_412472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_412472)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lsmr'
    return stypy_return_type_412472

# Assigning a type to the variable 'lsmr' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'lsmr', lsmr)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
