
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #-------------------------------------------------------------------------------
2: #
3: #  Define classes for (uni/multi)-variate kernel density estimation.
4: #
5: #  Currently, only Gaussian kernels are implemented.
6: #
7: #  Written by: Robert Kern
8: #
9: #  Date: 2004-08-09
10: #
11: #  Modified: 2005-02-10 by Robert Kern.
12: #              Contributed to Scipy
13: #            2005-10-07 by Robert Kern.
14: #              Some fixes to match the new scipy_core
15: #
16: #  Copyright 2004-2005 by Enthought, Inc.
17: #
18: #-------------------------------------------------------------------------------
19: 
20: from __future__ import division, print_function, absolute_import
21: 
22: # Standard library imports.
23: import warnings
24: 
25: # Scipy imports.
26: from scipy._lib.six import callable, string_types
27: from scipy import linalg, special
28: from scipy.special import logsumexp
29: 
30: from numpy import atleast_2d, reshape, zeros, newaxis, dot, exp, pi, sqrt, \
31:      ravel, power, atleast_1d, squeeze, sum, transpose
32: import numpy as np
33: from numpy.random import randint, multivariate_normal
34: 
35: # Local imports.
36: from . import mvn
37: 
38: 
39: __all__ = ['gaussian_kde']
40: 
41: 
42: class gaussian_kde(object):
43:     '''Representation of a kernel-density estimate using Gaussian kernels.
44: 
45:     Kernel density estimation is a way to estimate the probability density
46:     function (PDF) of a random variable in a non-parametric way.
47:     `gaussian_kde` works for both uni-variate and multi-variate data.   It
48:     includes automatic bandwidth determination.  The estimation works best for
49:     a unimodal distribution; bimodal or multi-modal distributions tend to be
50:     oversmoothed.
51: 
52:     Parameters
53:     ----------
54:     dataset : array_like
55:         Datapoints to estimate from. In case of univariate data this is a 1-D
56:         array, otherwise a 2-D array with shape (# of dims, # of data).
57:     bw_method : str, scalar or callable, optional
58:         The method used to calculate the estimator bandwidth.  This can be
59:         'scott', 'silverman', a scalar constant or a callable.  If a scalar,
60:         this will be used directly as `kde.factor`.  If a callable, it should
61:         take a `gaussian_kde` instance as only parameter and return a scalar.
62:         If None (default), 'scott' is used.  See Notes for more details.
63: 
64:     Attributes
65:     ----------
66:     dataset : ndarray
67:         The dataset with which `gaussian_kde` was initialized.
68:     d : int
69:         Number of dimensions.
70:     n : int
71:         Number of datapoints.
72:     factor : float
73:         The bandwidth factor, obtained from `kde.covariance_factor`, with which
74:         the covariance matrix is multiplied.
75:     covariance : ndarray
76:         The covariance matrix of `dataset`, scaled by the calculated bandwidth
77:         (`kde.factor`).
78:     inv_cov : ndarray
79:         The inverse of `covariance`.
80: 
81:     Methods
82:     -------
83:     evaluate
84:     __call__
85:     integrate_gaussian
86:     integrate_box_1d
87:     integrate_box
88:     integrate_kde
89:     pdf
90:     logpdf
91:     resample
92:     set_bandwidth
93:     covariance_factor
94: 
95:     Notes
96:     -----
97:     Bandwidth selection strongly influences the estimate obtained from the KDE
98:     (much more so than the actual shape of the kernel).  Bandwidth selection
99:     can be done by a "rule of thumb", by cross-validation, by "plug-in
100:     methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
101:     uses a rule of thumb, the default is Scott's Rule.
102: 
103:     Scott's Rule [1]_, implemented as `scotts_factor`, is::
104: 
105:         n**(-1./(d+4)),
106: 
107:     with ``n`` the number of data points and ``d`` the number of dimensions.
108:     Silverman's Rule [2]_, implemented as `silverman_factor`, is::
109: 
110:         (n * (d + 2) / 4.)**(-1. / (d + 4)).
111: 
112:     Good general descriptions of kernel density estimation can be found in [1]_
113:     and [2]_, the mathematics for this multi-dimensional implementation can be
114:     found in [1]_.
115: 
116:     References
117:     ----------
118:     .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
119:            Visualization", John Wiley & Sons, New York, Chicester, 1992.
120:     .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
121:            Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
122:            Chapman and Hall, London, 1986.
123:     .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
124:            Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
125:     .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
126:            conditional density estimation", Computational Statistics & Data
127:            Analysis, Vol. 36, pp. 279-298, 2001.
128: 
129:     Examples
130:     --------
131:     Generate some random two-dimensional data:
132: 
133:     >>> from scipy import stats
134:     >>> def measure(n):
135:     ...     "Measurement model, return two coupled measurements."
136:     ...     m1 = np.random.normal(size=n)
137:     ...     m2 = np.random.normal(scale=0.5, size=n)
138:     ...     return m1+m2, m1-m2
139: 
140:     >>> m1, m2 = measure(2000)
141:     >>> xmin = m1.min()
142:     >>> xmax = m1.max()
143:     >>> ymin = m2.min()
144:     >>> ymax = m2.max()
145: 
146:     Perform a kernel density estimate on the data:
147: 
148:     >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
149:     >>> positions = np.vstack([X.ravel(), Y.ravel()])
150:     >>> values = np.vstack([m1, m2])
151:     >>> kernel = stats.gaussian_kde(values)
152:     >>> Z = np.reshape(kernel(positions).T, X.shape)
153: 
154:     Plot the results:
155: 
156:     >>> import matplotlib.pyplot as plt
157:     >>> fig, ax = plt.subplots()
158:     >>> ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
159:     ...           extent=[xmin, xmax, ymin, ymax])
160:     >>> ax.plot(m1, m2, 'k.', markersize=2)
161:     >>> ax.set_xlim([xmin, xmax])
162:     >>> ax.set_ylim([ymin, ymax])
163:     >>> plt.show()
164: 
165:     '''
166:     def __init__(self, dataset, bw_method=None):
167:         self.dataset = atleast_2d(dataset)
168:         if not self.dataset.size > 1:
169:             raise ValueError("`dataset` input should have multiple elements.")
170: 
171:         self.d, self.n = self.dataset.shape
172:         self.set_bandwidth(bw_method=bw_method)
173: 
174:     def evaluate(self, points):
175:         '''Evaluate the estimated pdf on a set of points.
176: 
177:         Parameters
178:         ----------
179:         points : (# of dimensions, # of points)-array
180:             Alternatively, a (# of dimensions,) vector can be passed in and
181:             treated as a single point.
182: 
183:         Returns
184:         -------
185:         values : (# of points,)-array
186:             The values at each point.
187: 
188:         Raises
189:         ------
190:         ValueError : if the dimensionality of the input points is different than
191:                      the dimensionality of the KDE.
192: 
193:         '''
194:         points = atleast_2d(points)
195: 
196:         d, m = points.shape
197:         if d != self.d:
198:             if d == 1 and m == self.d:
199:                 # points was passed in as a row vector
200:                 points = reshape(points, (self.d, 1))
201:                 m = 1
202:             else:
203:                 msg = "points have dimension %s, dataset has dimension %s" % (d,
204:                     self.d)
205:                 raise ValueError(msg)
206: 
207:         result = zeros((m,), dtype=float)
208: 
209:         if m >= self.n:
210:             # there are more points than data, so loop over data
211:             for i in range(self.n):
212:                 diff = self.dataset[:, i, newaxis] - points
213:                 tdiff = dot(self.inv_cov, diff)
214:                 energy = sum(diff*tdiff,axis=0) / 2.0
215:                 result = result + exp(-energy)
216:         else:
217:             # loop over points
218:             for i in range(m):
219:                 diff = self.dataset - points[:, i, newaxis]
220:                 tdiff = dot(self.inv_cov, diff)
221:                 energy = sum(diff * tdiff, axis=0) / 2.0
222:                 result[i] = sum(exp(-energy), axis=0)
223: 
224:         result = result / self._norm_factor
225: 
226:         return result
227: 
228:     __call__ = evaluate
229: 
230:     def integrate_gaussian(self, mean, cov):
231:         '''
232:         Multiply estimated density by a multivariate Gaussian and integrate
233:         over the whole space.
234: 
235:         Parameters
236:         ----------
237:         mean : aray_like
238:             A 1-D array, specifying the mean of the Gaussian.
239:         cov : array_like
240:             A 2-D array, specifying the covariance matrix of the Gaussian.
241: 
242:         Returns
243:         -------
244:         result : scalar
245:             The value of the integral.
246: 
247:         Raises
248:         ------
249:         ValueError
250:             If the mean or covariance of the input Gaussian differs from
251:             the KDE's dimensionality.
252: 
253:         '''
254:         mean = atleast_1d(squeeze(mean))
255:         cov = atleast_2d(cov)
256: 
257:         if mean.shape != (self.d,):
258:             raise ValueError("mean does not have dimension %s" % self.d)
259:         if cov.shape != (self.d, self.d):
260:             raise ValueError("covariance does not have dimension %s" % self.d)
261: 
262:         # make mean a column vector
263:         mean = mean[:, newaxis]
264: 
265:         sum_cov = self.covariance + cov
266: 
267:         # This will raise LinAlgError if the new cov matrix is not s.p.d
268:         # cho_factor returns (ndarray, bool) where bool is a flag for whether
269:         # or not ndarray is upper or lower triangular
270:         sum_cov_chol = linalg.cho_factor(sum_cov)
271: 
272:         diff = self.dataset - mean
273:         tdiff = linalg.cho_solve(sum_cov_chol, diff)
274: 
275:         sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
276:         norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det
277: 
278:         energies = sum(diff * tdiff, axis=0) / 2.0
279:         result = sum(exp(-energies), axis=0) / norm_const / self.n
280: 
281:         return result
282: 
283:     def integrate_box_1d(self, low, high):
284:         '''
285:         Computes the integral of a 1D pdf between two bounds.
286: 
287:         Parameters
288:         ----------
289:         low : scalar
290:             Lower bound of integration.
291:         high : scalar
292:             Upper bound of integration.
293: 
294:         Returns
295:         -------
296:         value : scalar
297:             The result of the integral.
298: 
299:         Raises
300:         ------
301:         ValueError
302:             If the KDE is over more than one dimension.
303: 
304:         '''
305:         if self.d != 1:
306:             raise ValueError("integrate_box_1d() only handles 1D pdfs")
307: 
308:         stdev = ravel(sqrt(self.covariance))[0]
309: 
310:         normalized_low = ravel((low - self.dataset) / stdev)
311:         normalized_high = ravel((high - self.dataset) / stdev)
312: 
313:         value = np.mean(special.ndtr(normalized_high) -
314:                         special.ndtr(normalized_low))
315:         return value
316: 
317:     def integrate_box(self, low_bounds, high_bounds, maxpts=None):
318:         '''Computes the integral of a pdf over a rectangular interval.
319: 
320:         Parameters
321:         ----------
322:         low_bounds : array_like
323:             A 1-D array containing the lower bounds of integration.
324:         high_bounds : array_like
325:             A 1-D array containing the upper bounds of integration.
326:         maxpts : int, optional
327:             The maximum number of points to use for integration.
328: 
329:         Returns
330:         -------
331:         value : scalar
332:             The result of the integral.
333: 
334:         '''
335:         if maxpts is not None:
336:             extra_kwds = {'maxpts': maxpts}
337:         else:
338:             extra_kwds = {}
339: 
340:         value, inform = mvn.mvnun(low_bounds, high_bounds, self.dataset,
341:                                   self.covariance, **extra_kwds)
342:         if inform:
343:             msg = ('An integral in mvn.mvnun requires more points than %s' %
344:                    (self.d * 1000))
345:             warnings.warn(msg)
346: 
347:         return value
348: 
349:     def integrate_kde(self, other):
350:         '''
351:         Computes the integral of the product of this  kernel density estimate
352:         with another.
353: 
354:         Parameters
355:         ----------
356:         other : gaussian_kde instance
357:             The other kde.
358: 
359:         Returns
360:         -------
361:         value : scalar
362:             The result of the integral.
363: 
364:         Raises
365:         ------
366:         ValueError
367:             If the KDEs have different dimensionality.
368: 
369:         '''
370:         if other.d != self.d:
371:             raise ValueError("KDEs are not the same dimensionality")
372: 
373:         # we want to iterate over the smallest number of points
374:         if other.n < self.n:
375:             small = other
376:             large = self
377:         else:
378:             small = self
379:             large = other
380: 
381:         sum_cov = small.covariance + large.covariance
382:         sum_cov_chol = linalg.cho_factor(sum_cov)
383:         result = 0.0
384:         for i in range(small.n):
385:             mean = small.dataset[:, i, newaxis]
386:             diff = large.dataset - mean
387:             tdiff = linalg.cho_solve(sum_cov_chol, diff)
388: 
389:             energies = sum(diff * tdiff, axis=0) / 2.0
390:             result += sum(exp(-energies), axis=0)
391: 
392:         sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
393:         norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det
394: 
395:         result /= norm_const * large.n * small.n
396: 
397:         return result
398: 
399:     def resample(self, size=None):
400:         '''
401:         Randomly sample a dataset from the estimated pdf.
402: 
403:         Parameters
404:         ----------
405:         size : int, optional
406:             The number of samples to draw.  If not provided, then the size is
407:             the same as the underlying dataset.
408: 
409:         Returns
410:         -------
411:         resample : (self.d, `size`) ndarray
412:             The sampled dataset.
413: 
414:         '''
415:         if size is None:
416:             size = self.n
417: 
418:         norm = transpose(multivariate_normal(zeros((self.d,), float),
419:                          self.covariance, size=size))
420:         indices = randint(0, self.n, size=size)
421:         means = self.dataset[:, indices]
422: 
423:         return means + norm
424: 
425:     def scotts_factor(self):
426:         return power(self.n, -1./(self.d+4))
427: 
428:     def silverman_factor(self):
429:         return power(self.n*(self.d+2.0)/4.0, -1./(self.d+4))
430: 
431:     #  Default method to calculate bandwidth, can be overwritten by subclass
432:     covariance_factor = scotts_factor
433:     covariance_factor.__doc__ = '''Computes the coefficient (`kde.factor`) that
434:         multiplies the data covariance matrix to obtain the kernel covariance
435:         matrix. The default is `scotts_factor`.  A subclass can overwrite this
436:         method to provide a different method, or set it through a call to
437:         `kde.set_bandwidth`.'''
438: 
439:     def set_bandwidth(self, bw_method=None):
440:         '''Compute the estimator bandwidth with given method.
441: 
442:         The new bandwidth calculated after a call to `set_bandwidth` is used
443:         for subsequent evaluations of the estimated density.
444: 
445:         Parameters
446:         ----------
447:         bw_method : str, scalar or callable, optional
448:             The method used to calculate the estimator bandwidth.  This can be
449:             'scott', 'silverman', a scalar constant or a callable.  If a
450:             scalar, this will be used directly as `kde.factor`.  If a callable,
451:             it should take a `gaussian_kde` instance as only parameter and
452:             return a scalar.  If None (default), nothing happens; the current
453:             `kde.covariance_factor` method is kept.
454: 
455:         Notes
456:         -----
457:         .. versionadded:: 0.11
458: 
459:         Examples
460:         --------
461:         >>> import scipy.stats as stats
462:         >>> x1 = np.array([-7, -5, 1, 4, 5.])
463:         >>> kde = stats.gaussian_kde(x1)
464:         >>> xs = np.linspace(-10, 10, num=50)
465:         >>> y1 = kde(xs)
466:         >>> kde.set_bandwidth(bw_method='silverman')
467:         >>> y2 = kde(xs)
468:         >>> kde.set_bandwidth(bw_method=kde.factor / 3.)
469:         >>> y3 = kde(xs)
470: 
471:         >>> import matplotlib.pyplot as plt
472:         >>> fig, ax = plt.subplots()
473:         >>> ax.plot(x1, np.ones(x1.shape) / (4. * x1.size), 'bo',
474:         ...         label='Data points (rescaled)')
475:         >>> ax.plot(xs, y1, label='Scott (default)')
476:         >>> ax.plot(xs, y2, label='Silverman')
477:         >>> ax.plot(xs, y3, label='Const (1/3 * Silverman)')
478:         >>> ax.legend()
479:         >>> plt.show()
480: 
481:         '''
482:         if bw_method is None:
483:             pass
484:         elif bw_method == 'scott':
485:             self.covariance_factor = self.scotts_factor
486:         elif bw_method == 'silverman':
487:             self.covariance_factor = self.silverman_factor
488:         elif np.isscalar(bw_method) and not isinstance(bw_method, string_types):
489:             self._bw_method = 'use constant'
490:             self.covariance_factor = lambda: bw_method
491:         elif callable(bw_method):
492:             self._bw_method = bw_method
493:             self.covariance_factor = lambda: self._bw_method(self)
494:         else:
495:             msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
496:                   "or a callable."
497:             raise ValueError(msg)
498: 
499:         self._compute_covariance()
500: 
501:     def _compute_covariance(self):
502:         '''Computes the covariance matrix for each Gaussian kernel using
503:         covariance_factor().
504:         '''
505:         self.factor = self.covariance_factor()
506:         # Cache covariance and inverse covariance of the data
507:         if not hasattr(self, '_data_inv_cov'):
508:             self._data_covariance = atleast_2d(np.cov(self.dataset, rowvar=1,
509:                                                bias=False))
510:             self._data_inv_cov = linalg.inv(self._data_covariance)
511: 
512:         self.covariance = self._data_covariance * self.factor**2
513:         self.inv_cov = self._data_inv_cov / self.factor**2
514:         self._norm_factor = sqrt(linalg.det(2*pi*self.covariance)) * self.n
515: 
516:     def pdf(self, x):
517:         '''
518:         Evaluate the estimated pdf on a provided set of points.
519: 
520:         Notes
521:         -----
522:         This is an alias for `gaussian_kde.evaluate`.  See the ``evaluate``
523:         docstring for more details.
524: 
525:         '''
526:         return self.evaluate(x)
527: 
528:     def logpdf(self, x):
529:         '''
530:         Evaluate the log of the estimated pdf on a provided set of points.
531:         '''
532: 
533:         points = atleast_2d(x)
534: 
535:         d, m = points.shape
536:         if d != self.d:
537:             if d == 1 and m == self.d:
538:                 # points was passed in as a row vector
539:                 points = reshape(points, (self.d, 1))
540:                 m = 1
541:             else:
542:                 msg = "points have dimension %s, dataset has dimension %s" % (d,
543:                     self.d)
544:                 raise ValueError(msg)
545: 
546:         result = zeros((m,), dtype=float)
547: 
548:         if m >= self.n:
549:             # there are more points than data, so loop over data
550:             energy = zeros((self.n, m), dtype=float)
551:             for i in range(self.n):
552:                 diff = self.dataset[:, i, newaxis] - points
553:                 tdiff = dot(self.inv_cov, diff)
554:                 energy[i] = sum(diff*tdiff,axis=0) / 2.0
555:             result = logsumexp(-energy, b=1/self._norm_factor, axis=0)
556:         else:
557:             # loop over points
558:             for i in range(m):
559:                 diff = self.dataset - points[:, i, newaxis]
560:                 tdiff = dot(self.inv_cov, diff)
561:                 energy = sum(diff * tdiff, axis=0) / 2.0
562:                 result[i] = logsumexp(-energy, b=1/self._norm_factor)
563: 
564:         return result
565: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import warnings' statement (line 23)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from scipy._lib.six import callable, string_types' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565082 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy._lib.six')

if (type(import_565082) is not StypyTypeError):

    if (import_565082 != 'pyd_module'):
        __import__(import_565082)
        sys_modules_565083 = sys.modules[import_565082]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy._lib.six', sys_modules_565083.module_type_store, module_type_store, ['callable', 'string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_565083, sys_modules_565083.module_type_store, module_type_store)
    else:
        from scipy._lib.six import callable, string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy._lib.six', None, module_type_store, ['callable', 'string_types'], [callable, string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy._lib.six', import_565082)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from scipy import linalg, special' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565084 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy')

if (type(import_565084) is not StypyTypeError):

    if (import_565084 != 'pyd_module'):
        __import__(import_565084)
        sys_modules_565085 = sys.modules[import_565084]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy', sys_modules_565085.module_type_store, module_type_store, ['linalg', 'special'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_565085, sys_modules_565085.module_type_store, module_type_store)
    else:
        from scipy import linalg, special

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy', None, module_type_store, ['linalg', 'special'], [linalg, special])

else:
    # Assigning a type to the variable 'scipy' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy', import_565084)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from scipy.special import logsumexp' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565086 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.special')

if (type(import_565086) is not StypyTypeError):

    if (import_565086 != 'pyd_module'):
        __import__(import_565086)
        sys_modules_565087 = sys.modules[import_565086]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.special', sys_modules_565087.module_type_store, module_type_store, ['logsumexp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_565087, sys_modules_565087.module_type_store, module_type_store)
    else:
        from scipy.special import logsumexp

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.special', None, module_type_store, ['logsumexp'], [logsumexp])

else:
    # Assigning a type to the variable 'scipy.special' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.special', import_565086)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from numpy import atleast_2d, reshape, zeros, newaxis, dot, exp, pi, sqrt, ravel, power, atleast_1d, squeeze, sum, transpose' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565088 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy')

if (type(import_565088) is not StypyTypeError):

    if (import_565088 != 'pyd_module'):
        __import__(import_565088)
        sys_modules_565089 = sys.modules[import_565088]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy', sys_modules_565089.module_type_store, module_type_store, ['atleast_2d', 'reshape', 'zeros', 'newaxis', 'dot', 'exp', 'pi', 'sqrt', 'ravel', 'power', 'atleast_1d', 'squeeze', 'sum', 'transpose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_565089, sys_modules_565089.module_type_store, module_type_store)
    else:
        from numpy import atleast_2d, reshape, zeros, newaxis, dot, exp, pi, sqrt, ravel, power, atleast_1d, squeeze, sum, transpose

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy', None, module_type_store, ['atleast_2d', 'reshape', 'zeros', 'newaxis', 'dot', 'exp', 'pi', 'sqrt', 'ravel', 'power', 'atleast_1d', 'squeeze', 'sum', 'transpose'], [atleast_2d, reshape, zeros, newaxis, dot, exp, pi, sqrt, ravel, power, atleast_1d, squeeze, sum, transpose])

else:
    # Assigning a type to the variable 'numpy' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy', import_565088)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'import numpy' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565090 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy')

if (type(import_565090) is not StypyTypeError):

    if (import_565090 != 'pyd_module'):
        __import__(import_565090)
        sys_modules_565091 = sys.modules[import_565090]
        import_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'np', sys_modules_565091.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy', import_565090)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'from numpy.random import randint, multivariate_normal' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565092 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy.random')

if (type(import_565092) is not StypyTypeError):

    if (import_565092 != 'pyd_module'):
        __import__(import_565092)
        sys_modules_565093 = sys.modules[import_565092]
        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy.random', sys_modules_565093.module_type_store, module_type_store, ['randint', 'multivariate_normal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 33, 0), __file__, sys_modules_565093, sys_modules_565093.module_type_store, module_type_store)
    else:
        from numpy.random import randint, multivariate_normal

        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy.random', None, module_type_store, ['randint', 'multivariate_normal'], [randint, multivariate_normal])

else:
    # Assigning a type to the variable 'numpy.random' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy.random', import_565092)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from scipy.stats import mvn' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565094 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.stats')

if (type(import_565094) is not StypyTypeError):

    if (import_565094 != 'pyd_module'):
        __import__(import_565094)
        sys_modules_565095 = sys.modules[import_565094]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.stats', sys_modules_565095.module_type_store, module_type_store, ['mvn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_565095, sys_modules_565095.module_type_store, module_type_store)
    else:
        from scipy.stats import mvn

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.stats', None, module_type_store, ['mvn'], [mvn])

else:
    # Assigning a type to the variable 'scipy.stats' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.stats', import_565094)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


# Assigning a List to a Name (line 39):

# Assigning a List to a Name (line 39):
__all__ = ['gaussian_kde']
module_type_store.set_exportable_members(['gaussian_kde'])

# Obtaining an instance of the builtin type 'list' (line 39)
list_565096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 39)
# Adding element type (line 39)
str_565097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 11), 'str', 'gaussian_kde')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 10), list_565096, str_565097)

# Assigning a type to the variable '__all__' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '__all__', list_565096)
# Declaration of the 'gaussian_kde' class

class gaussian_kde(object, ):
    str_565098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, (-1)), 'str', 'Representation of a kernel-density estimate using Gaussian kernels.\n\n    Kernel density estimation is a way to estimate the probability density\n    function (PDF) of a random variable in a non-parametric way.\n    `gaussian_kde` works for both uni-variate and multi-variate data.   It\n    includes automatic bandwidth determination.  The estimation works best for\n    a unimodal distribution; bimodal or multi-modal distributions tend to be\n    oversmoothed.\n\n    Parameters\n    ----------\n    dataset : array_like\n        Datapoints to estimate from. In case of univariate data this is a 1-D\n        array, otherwise a 2-D array with shape (# of dims, # of data).\n    bw_method : str, scalar or callable, optional\n        The method used to calculate the estimator bandwidth.  This can be\n        \'scott\', \'silverman\', a scalar constant or a callable.  If a scalar,\n        this will be used directly as `kde.factor`.  If a callable, it should\n        take a `gaussian_kde` instance as only parameter and return a scalar.\n        If None (default), \'scott\' is used.  See Notes for more details.\n\n    Attributes\n    ----------\n    dataset : ndarray\n        The dataset with which `gaussian_kde` was initialized.\n    d : int\n        Number of dimensions.\n    n : int\n        Number of datapoints.\n    factor : float\n        The bandwidth factor, obtained from `kde.covariance_factor`, with which\n        the covariance matrix is multiplied.\n    covariance : ndarray\n        The covariance matrix of `dataset`, scaled by the calculated bandwidth\n        (`kde.factor`).\n    inv_cov : ndarray\n        The inverse of `covariance`.\n\n    Methods\n    -------\n    evaluate\n    __call__\n    integrate_gaussian\n    integrate_box_1d\n    integrate_box\n    integrate_kde\n    pdf\n    logpdf\n    resample\n    set_bandwidth\n    covariance_factor\n\n    Notes\n    -----\n    Bandwidth selection strongly influences the estimate obtained from the KDE\n    (much more so than the actual shape of the kernel).  Bandwidth selection\n    can be done by a "rule of thumb", by cross-validation, by "plug-in\n    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`\n    uses a rule of thumb, the default is Scott\'s Rule.\n\n    Scott\'s Rule [1]_, implemented as `scotts_factor`, is::\n\n        n**(-1./(d+4)),\n\n    with ``n`` the number of data points and ``d`` the number of dimensions.\n    Silverman\'s Rule [2]_, implemented as `silverman_factor`, is::\n\n        (n * (d + 2) / 4.)**(-1. / (d + 4)).\n\n    Good general descriptions of kernel density estimation can be found in [1]_\n    and [2]_, the mathematics for this multi-dimensional implementation can be\n    found in [1]_.\n\n    References\n    ----------\n    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and\n           Visualization", John Wiley & Sons, New York, Chicester, 1992.\n    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data\n           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,\n           Chapman and Hall, London, 1986.\n    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A\n           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.\n    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel\n           conditional density estimation", Computational Statistics & Data\n           Analysis, Vol. 36, pp. 279-298, 2001.\n\n    Examples\n    --------\n    Generate some random two-dimensional data:\n\n    >>> from scipy import stats\n    >>> def measure(n):\n    ...     "Measurement model, return two coupled measurements."\n    ...     m1 = np.random.normal(size=n)\n    ...     m2 = np.random.normal(scale=0.5, size=n)\n    ...     return m1+m2, m1-m2\n\n    >>> m1, m2 = measure(2000)\n    >>> xmin = m1.min()\n    >>> xmax = m1.max()\n    >>> ymin = m2.min()\n    >>> ymax = m2.max()\n\n    Perform a kernel density estimate on the data:\n\n    >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]\n    >>> positions = np.vstack([X.ravel(), Y.ravel()])\n    >>> values = np.vstack([m1, m2])\n    >>> kernel = stats.gaussian_kde(values)\n    >>> Z = np.reshape(kernel(positions).T, X.shape)\n\n    Plot the results:\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots()\n    >>> ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,\n    ...           extent=[xmin, xmax, ymin, ymax])\n    >>> ax.plot(m1, m2, \'k.\', markersize=2)\n    >>> ax.set_xlim([xmin, xmax])\n    >>> ax.set_ylim([ymin, ymax])\n    >>> plt.show()\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 166)
        None_565099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 42), 'None')
        defaults = [None_565099]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 166, 4, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.__init__', ['dataset', 'bw_method'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['dataset', 'bw_method'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 167):
        
        # Assigning a Call to a Attribute (line 167):
        
        # Call to atleast_2d(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'dataset' (line 167)
        dataset_565101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 34), 'dataset', False)
        # Processing the call keyword arguments (line 167)
        kwargs_565102 = {}
        # Getting the type of 'atleast_2d' (line 167)
        atleast_2d_565100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'atleast_2d', False)
        # Calling atleast_2d(args, kwargs) (line 167)
        atleast_2d_call_result_565103 = invoke(stypy.reporting.localization.Localization(__file__, 167, 23), atleast_2d_565100, *[dataset_565101], **kwargs_565102)
        
        # Getting the type of 'self' (line 167)
        self_565104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member 'dataset' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_565104, 'dataset', atleast_2d_call_result_565103)
        
        
        
        # Getting the type of 'self' (line 168)
        self_565105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'self')
        # Obtaining the member 'dataset' of a type (line 168)
        dataset_565106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 15), self_565105, 'dataset')
        # Obtaining the member 'size' of a type (line 168)
        size_565107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 15), dataset_565106, 'size')
        int_565108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 35), 'int')
        # Applying the binary operator '>' (line 168)
        result_gt_565109 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 15), '>', size_565107, int_565108)
        
        # Applying the 'not' unary operator (line 168)
        result_not__565110 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 11), 'not', result_gt_565109)
        
        # Testing the type of an if condition (line 168)
        if_condition_565111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 8), result_not__565110)
        # Assigning a type to the variable 'if_condition_565111' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'if_condition_565111', if_condition_565111)
        # SSA begins for if statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 169)
        # Processing the call arguments (line 169)
        str_565113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 29), 'str', '`dataset` input should have multiple elements.')
        # Processing the call keyword arguments (line 169)
        kwargs_565114 = {}
        # Getting the type of 'ValueError' (line 169)
        ValueError_565112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 169)
        ValueError_call_result_565115 = invoke(stypy.reporting.localization.Localization(__file__, 169, 18), ValueError_565112, *[str_565113], **kwargs_565114)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 169, 12), ValueError_call_result_565115, 'raise parameter', BaseException)
        # SSA join for if statement (line 168)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Tuple (line 171):
        
        # Assigning a Subscript to a Name (line 171):
        
        # Obtaining the type of the subscript
        int_565116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'int')
        # Getting the type of 'self' (line 171)
        self_565117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 25), 'self')
        # Obtaining the member 'dataset' of a type (line 171)
        dataset_565118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 25), self_565117, 'dataset')
        # Obtaining the member 'shape' of a type (line 171)
        shape_565119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 25), dataset_565118, 'shape')
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___565120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), shape_565119, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_565121 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), getitem___565120, int_565116)
        
        # Assigning a type to the variable 'tuple_var_assignment_565074' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_565074', subscript_call_result_565121)
        
        # Assigning a Subscript to a Name (line 171):
        
        # Obtaining the type of the subscript
        int_565122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'int')
        # Getting the type of 'self' (line 171)
        self_565123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 25), 'self')
        # Obtaining the member 'dataset' of a type (line 171)
        dataset_565124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 25), self_565123, 'dataset')
        # Obtaining the member 'shape' of a type (line 171)
        shape_565125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 25), dataset_565124, 'shape')
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___565126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), shape_565125, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_565127 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), getitem___565126, int_565122)
        
        # Assigning a type to the variable 'tuple_var_assignment_565075' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_565075', subscript_call_result_565127)
        
        # Assigning a Name to a Attribute (line 171):
        # Getting the type of 'tuple_var_assignment_565074' (line 171)
        tuple_var_assignment_565074_565128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_565074')
        # Getting the type of 'self' (line 171)
        self_565129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self')
        # Setting the type of the member 'd' of a type (line 171)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_565129, 'd', tuple_var_assignment_565074_565128)
        
        # Assigning a Name to a Attribute (line 171):
        # Getting the type of 'tuple_var_assignment_565075' (line 171)
        tuple_var_assignment_565075_565130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'tuple_var_assignment_565075')
        # Getting the type of 'self' (line 171)
        self_565131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'self')
        # Setting the type of the member 'n' of a type (line 171)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), self_565131, 'n', tuple_var_assignment_565075_565130)
        
        # Call to set_bandwidth(...): (line 172)
        # Processing the call keyword arguments (line 172)
        # Getting the type of 'bw_method' (line 172)
        bw_method_565134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'bw_method', False)
        keyword_565135 = bw_method_565134
        kwargs_565136 = {'bw_method': keyword_565135}
        # Getting the type of 'self' (line 172)
        self_565132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self', False)
        # Obtaining the member 'set_bandwidth' of a type (line 172)
        set_bandwidth_565133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_565132, 'set_bandwidth')
        # Calling set_bandwidth(args, kwargs) (line 172)
        set_bandwidth_call_result_565137 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), set_bandwidth_565133, *[], **kwargs_565136)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def evaluate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'evaluate'
        module_type_store = module_type_store.open_function_context('evaluate', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde.evaluate.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde.evaluate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde.evaluate.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde.evaluate.__dict__.__setitem__('stypy_function_name', 'gaussian_kde.evaluate')
        gaussian_kde.evaluate.__dict__.__setitem__('stypy_param_names_list', ['points'])
        gaussian_kde.evaluate.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde.evaluate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde.evaluate.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde.evaluate.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde.evaluate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde.evaluate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.evaluate', ['points'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'evaluate', localization, ['points'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'evaluate(...)' code ##################

        str_565138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, (-1)), 'str', 'Evaluate the estimated pdf on a set of points.\n\n        Parameters\n        ----------\n        points : (# of dimensions, # of points)-array\n            Alternatively, a (# of dimensions,) vector can be passed in and\n            treated as a single point.\n\n        Returns\n        -------\n        values : (# of points,)-array\n            The values at each point.\n\n        Raises\n        ------\n        ValueError : if the dimensionality of the input points is different than\n                     the dimensionality of the KDE.\n\n        ')
        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Call to atleast_2d(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'points' (line 194)
        points_565140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 28), 'points', False)
        # Processing the call keyword arguments (line 194)
        kwargs_565141 = {}
        # Getting the type of 'atleast_2d' (line 194)
        atleast_2d_565139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'atleast_2d', False)
        # Calling atleast_2d(args, kwargs) (line 194)
        atleast_2d_call_result_565142 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), atleast_2d_565139, *[points_565140], **kwargs_565141)
        
        # Assigning a type to the variable 'points' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'points', atleast_2d_call_result_565142)
        
        # Assigning a Attribute to a Tuple (line 196):
        
        # Assigning a Subscript to a Name (line 196):
        
        # Obtaining the type of the subscript
        int_565143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 8), 'int')
        # Getting the type of 'points' (line 196)
        points_565144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'points')
        # Obtaining the member 'shape' of a type (line 196)
        shape_565145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 15), points_565144, 'shape')
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___565146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), shape_565145, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_565147 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), getitem___565146, int_565143)
        
        # Assigning a type to the variable 'tuple_var_assignment_565076' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_565076', subscript_call_result_565147)
        
        # Assigning a Subscript to a Name (line 196):
        
        # Obtaining the type of the subscript
        int_565148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 8), 'int')
        # Getting the type of 'points' (line 196)
        points_565149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'points')
        # Obtaining the member 'shape' of a type (line 196)
        shape_565150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 15), points_565149, 'shape')
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___565151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), shape_565150, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_565152 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), getitem___565151, int_565148)
        
        # Assigning a type to the variable 'tuple_var_assignment_565077' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_565077', subscript_call_result_565152)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'tuple_var_assignment_565076' (line 196)
        tuple_var_assignment_565076_565153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_565076')
        # Assigning a type to the variable 'd' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'd', tuple_var_assignment_565076_565153)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'tuple_var_assignment_565077' (line 196)
        tuple_var_assignment_565077_565154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_565077')
        # Assigning a type to the variable 'm' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'm', tuple_var_assignment_565077_565154)
        
        
        # Getting the type of 'd' (line 197)
        d_565155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'd')
        # Getting the type of 'self' (line 197)
        self_565156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'self')
        # Obtaining the member 'd' of a type (line 197)
        d_565157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 16), self_565156, 'd')
        # Applying the binary operator '!=' (line 197)
        result_ne_565158 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 11), '!=', d_565155, d_565157)
        
        # Testing the type of an if condition (line 197)
        if_condition_565159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 8), result_ne_565158)
        # Assigning a type to the variable 'if_condition_565159' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'if_condition_565159', if_condition_565159)
        # SSA begins for if statement (line 197)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'd' (line 198)
        d_565160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'd')
        int_565161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 20), 'int')
        # Applying the binary operator '==' (line 198)
        result_eq_565162 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 15), '==', d_565160, int_565161)
        
        
        # Getting the type of 'm' (line 198)
        m_565163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 26), 'm')
        # Getting the type of 'self' (line 198)
        self_565164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 31), 'self')
        # Obtaining the member 'd' of a type (line 198)
        d_565165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 31), self_565164, 'd')
        # Applying the binary operator '==' (line 198)
        result_eq_565166 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 26), '==', m_565163, d_565165)
        
        # Applying the binary operator 'and' (line 198)
        result_and_keyword_565167 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 15), 'and', result_eq_565162, result_eq_565166)
        
        # Testing the type of an if condition (line 198)
        if_condition_565168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 12), result_and_keyword_565167)
        # Assigning a type to the variable 'if_condition_565168' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'if_condition_565168', if_condition_565168)
        # SSA begins for if statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to reshape(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'points' (line 200)
        points_565170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 33), 'points', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 200)
        tuple_565171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 200)
        # Adding element type (line 200)
        # Getting the type of 'self' (line 200)
        self_565172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 42), 'self', False)
        # Obtaining the member 'd' of a type (line 200)
        d_565173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 42), self_565172, 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 42), tuple_565171, d_565173)
        # Adding element type (line 200)
        int_565174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 42), tuple_565171, int_565174)
        
        # Processing the call keyword arguments (line 200)
        kwargs_565175 = {}
        # Getting the type of 'reshape' (line 200)
        reshape_565169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'reshape', False)
        # Calling reshape(args, kwargs) (line 200)
        reshape_call_result_565176 = invoke(stypy.reporting.localization.Localization(__file__, 200, 25), reshape_565169, *[points_565170, tuple_565171], **kwargs_565175)
        
        # Assigning a type to the variable 'points' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'points', reshape_call_result_565176)
        
        # Assigning a Num to a Name (line 201):
        
        # Assigning a Num to a Name (line 201):
        int_565177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 20), 'int')
        # Assigning a type to the variable 'm' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'm', int_565177)
        # SSA branch for the else part of an if statement (line 198)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 203):
        
        # Assigning a BinOp to a Name (line 203):
        str_565178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 22), 'str', 'points have dimension %s, dataset has dimension %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 203)
        tuple_565179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 78), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 203)
        # Adding element type (line 203)
        # Getting the type of 'd' (line 203)
        d_565180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 78), 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 78), tuple_565179, d_565180)
        # Adding element type (line 203)
        # Getting the type of 'self' (line 204)
        self_565181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'self')
        # Obtaining the member 'd' of a type (line 204)
        d_565182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 20), self_565181, 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 78), tuple_565179, d_565182)
        
        # Applying the binary operator '%' (line 203)
        result_mod_565183 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 22), '%', str_565178, tuple_565179)
        
        # Assigning a type to the variable 'msg' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'msg', result_mod_565183)
        
        # Call to ValueError(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'msg' (line 205)
        msg_565185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 33), 'msg', False)
        # Processing the call keyword arguments (line 205)
        kwargs_565186 = {}
        # Getting the type of 'ValueError' (line 205)
        ValueError_565184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 205)
        ValueError_call_result_565187 = invoke(stypy.reporting.localization.Localization(__file__, 205, 22), ValueError_565184, *[msg_565185], **kwargs_565186)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 205, 16), ValueError_call_result_565187, 'raise parameter', BaseException)
        # SSA join for if statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 197)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to zeros(...): (line 207)
        # Processing the call arguments (line 207)
        
        # Obtaining an instance of the builtin type 'tuple' (line 207)
        tuple_565189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 207)
        # Adding element type (line 207)
        # Getting the type of 'm' (line 207)
        m_565190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 24), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 24), tuple_565189, m_565190)
        
        # Processing the call keyword arguments (line 207)
        # Getting the type of 'float' (line 207)
        float_565191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 35), 'float', False)
        keyword_565192 = float_565191
        kwargs_565193 = {'dtype': keyword_565192}
        # Getting the type of 'zeros' (line 207)
        zeros_565188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 17), 'zeros', False)
        # Calling zeros(args, kwargs) (line 207)
        zeros_call_result_565194 = invoke(stypy.reporting.localization.Localization(__file__, 207, 17), zeros_565188, *[tuple_565189], **kwargs_565193)
        
        # Assigning a type to the variable 'result' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'result', zeros_call_result_565194)
        
        
        # Getting the type of 'm' (line 209)
        m_565195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'm')
        # Getting the type of 'self' (line 209)
        self_565196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'self')
        # Obtaining the member 'n' of a type (line 209)
        n_565197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 16), self_565196, 'n')
        # Applying the binary operator '>=' (line 209)
        result_ge_565198 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 11), '>=', m_565195, n_565197)
        
        # Testing the type of an if condition (line 209)
        if_condition_565199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 8), result_ge_565198)
        # Assigning a type to the variable 'if_condition_565199' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'if_condition_565199', if_condition_565199)
        # SSA begins for if statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to range(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'self' (line 211)
        self_565201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 27), 'self', False)
        # Obtaining the member 'n' of a type (line 211)
        n_565202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 27), self_565201, 'n')
        # Processing the call keyword arguments (line 211)
        kwargs_565203 = {}
        # Getting the type of 'range' (line 211)
        range_565200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'range', False)
        # Calling range(args, kwargs) (line 211)
        range_call_result_565204 = invoke(stypy.reporting.localization.Localization(__file__, 211, 21), range_565200, *[n_565202], **kwargs_565203)
        
        # Testing the type of a for loop iterable (line 211)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 211, 12), range_call_result_565204)
        # Getting the type of the for loop variable (line 211)
        for_loop_var_565205 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 211, 12), range_call_result_565204)
        # Assigning a type to the variable 'i' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'i', for_loop_var_565205)
        # SSA begins for a for statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 212):
        
        # Assigning a BinOp to a Name (line 212):
        
        # Obtaining the type of the subscript
        slice_565206 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 212, 23), None, None, None)
        # Getting the type of 'i' (line 212)
        i_565207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 39), 'i')
        # Getting the type of 'newaxis' (line 212)
        newaxis_565208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 42), 'newaxis')
        # Getting the type of 'self' (line 212)
        self_565209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 23), 'self')
        # Obtaining the member 'dataset' of a type (line 212)
        dataset_565210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 23), self_565209, 'dataset')
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___565211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 23), dataset_565210, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_565212 = invoke(stypy.reporting.localization.Localization(__file__, 212, 23), getitem___565211, (slice_565206, i_565207, newaxis_565208))
        
        # Getting the type of 'points' (line 212)
        points_565213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 53), 'points')
        # Applying the binary operator '-' (line 212)
        result_sub_565214 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 23), '-', subscript_call_result_565212, points_565213)
        
        # Assigning a type to the variable 'diff' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'diff', result_sub_565214)
        
        # Assigning a Call to a Name (line 213):
        
        # Assigning a Call to a Name (line 213):
        
        # Call to dot(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'self' (line 213)
        self_565216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'self', False)
        # Obtaining the member 'inv_cov' of a type (line 213)
        inv_cov_565217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 28), self_565216, 'inv_cov')
        # Getting the type of 'diff' (line 213)
        diff_565218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 42), 'diff', False)
        # Processing the call keyword arguments (line 213)
        kwargs_565219 = {}
        # Getting the type of 'dot' (line 213)
        dot_565215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'dot', False)
        # Calling dot(args, kwargs) (line 213)
        dot_call_result_565220 = invoke(stypy.reporting.localization.Localization(__file__, 213, 24), dot_565215, *[inv_cov_565217, diff_565218], **kwargs_565219)
        
        # Assigning a type to the variable 'tdiff' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'tdiff', dot_call_result_565220)
        
        # Assigning a BinOp to a Name (line 214):
        
        # Assigning a BinOp to a Name (line 214):
        
        # Call to sum(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'diff' (line 214)
        diff_565222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 29), 'diff', False)
        # Getting the type of 'tdiff' (line 214)
        tdiff_565223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 34), 'tdiff', False)
        # Applying the binary operator '*' (line 214)
        result_mul_565224 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 29), '*', diff_565222, tdiff_565223)
        
        # Processing the call keyword arguments (line 214)
        int_565225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 45), 'int')
        keyword_565226 = int_565225
        kwargs_565227 = {'axis': keyword_565226}
        # Getting the type of 'sum' (line 214)
        sum_565221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 25), 'sum', False)
        # Calling sum(args, kwargs) (line 214)
        sum_call_result_565228 = invoke(stypy.reporting.localization.Localization(__file__, 214, 25), sum_565221, *[result_mul_565224], **kwargs_565227)
        
        float_565229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 50), 'float')
        # Applying the binary operator 'div' (line 214)
        result_div_565230 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 25), 'div', sum_call_result_565228, float_565229)
        
        # Assigning a type to the variable 'energy' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'energy', result_div_565230)
        
        # Assigning a BinOp to a Name (line 215):
        
        # Assigning a BinOp to a Name (line 215):
        # Getting the type of 'result' (line 215)
        result_565231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 25), 'result')
        
        # Call to exp(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Getting the type of 'energy' (line 215)
        energy_565233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 39), 'energy', False)
        # Applying the 'usub' unary operator (line 215)
        result___neg___565234 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 38), 'usub', energy_565233)
        
        # Processing the call keyword arguments (line 215)
        kwargs_565235 = {}
        # Getting the type of 'exp' (line 215)
        exp_565232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 34), 'exp', False)
        # Calling exp(args, kwargs) (line 215)
        exp_call_result_565236 = invoke(stypy.reporting.localization.Localization(__file__, 215, 34), exp_565232, *[result___neg___565234], **kwargs_565235)
        
        # Applying the binary operator '+' (line 215)
        result_add_565237 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 25), '+', result_565231, exp_call_result_565236)
        
        # Assigning a type to the variable 'result' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'result', result_add_565237)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 209)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to range(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'm' (line 218)
        m_565239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'm', False)
        # Processing the call keyword arguments (line 218)
        kwargs_565240 = {}
        # Getting the type of 'range' (line 218)
        range_565238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 21), 'range', False)
        # Calling range(args, kwargs) (line 218)
        range_call_result_565241 = invoke(stypy.reporting.localization.Localization(__file__, 218, 21), range_565238, *[m_565239], **kwargs_565240)
        
        # Testing the type of a for loop iterable (line 218)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 218, 12), range_call_result_565241)
        # Getting the type of the for loop variable (line 218)
        for_loop_var_565242 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 218, 12), range_call_result_565241)
        # Assigning a type to the variable 'i' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'i', for_loop_var_565242)
        # SSA begins for a for statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 219):
        
        # Assigning a BinOp to a Name (line 219):
        # Getting the type of 'self' (line 219)
        self_565243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 23), 'self')
        # Obtaining the member 'dataset' of a type (line 219)
        dataset_565244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 23), self_565243, 'dataset')
        
        # Obtaining the type of the subscript
        slice_565245 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 219, 38), None, None, None)
        # Getting the type of 'i' (line 219)
        i_565246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 48), 'i')
        # Getting the type of 'newaxis' (line 219)
        newaxis_565247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 51), 'newaxis')
        # Getting the type of 'points' (line 219)
        points_565248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 38), 'points')
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___565249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 38), points_565248, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_565250 = invoke(stypy.reporting.localization.Localization(__file__, 219, 38), getitem___565249, (slice_565245, i_565246, newaxis_565247))
        
        # Applying the binary operator '-' (line 219)
        result_sub_565251 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 23), '-', dataset_565244, subscript_call_result_565250)
        
        # Assigning a type to the variable 'diff' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'diff', result_sub_565251)
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to dot(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'self' (line 220)
        self_565253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'self', False)
        # Obtaining the member 'inv_cov' of a type (line 220)
        inv_cov_565254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 28), self_565253, 'inv_cov')
        # Getting the type of 'diff' (line 220)
        diff_565255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 42), 'diff', False)
        # Processing the call keyword arguments (line 220)
        kwargs_565256 = {}
        # Getting the type of 'dot' (line 220)
        dot_565252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'dot', False)
        # Calling dot(args, kwargs) (line 220)
        dot_call_result_565257 = invoke(stypy.reporting.localization.Localization(__file__, 220, 24), dot_565252, *[inv_cov_565254, diff_565255], **kwargs_565256)
        
        # Assigning a type to the variable 'tdiff' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'tdiff', dot_call_result_565257)
        
        # Assigning a BinOp to a Name (line 221):
        
        # Assigning a BinOp to a Name (line 221):
        
        # Call to sum(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'diff' (line 221)
        diff_565259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'diff', False)
        # Getting the type of 'tdiff' (line 221)
        tdiff_565260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 36), 'tdiff', False)
        # Applying the binary operator '*' (line 221)
        result_mul_565261 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 29), '*', diff_565259, tdiff_565260)
        
        # Processing the call keyword arguments (line 221)
        int_565262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 48), 'int')
        keyword_565263 = int_565262
        kwargs_565264 = {'axis': keyword_565263}
        # Getting the type of 'sum' (line 221)
        sum_565258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 25), 'sum', False)
        # Calling sum(args, kwargs) (line 221)
        sum_call_result_565265 = invoke(stypy.reporting.localization.Localization(__file__, 221, 25), sum_565258, *[result_mul_565261], **kwargs_565264)
        
        float_565266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 53), 'float')
        # Applying the binary operator 'div' (line 221)
        result_div_565267 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 25), 'div', sum_call_result_565265, float_565266)
        
        # Assigning a type to the variable 'energy' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'energy', result_div_565267)
        
        # Assigning a Call to a Subscript (line 222):
        
        # Assigning a Call to a Subscript (line 222):
        
        # Call to sum(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Call to exp(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Getting the type of 'energy' (line 222)
        energy_565270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 37), 'energy', False)
        # Applying the 'usub' unary operator (line 222)
        result___neg___565271 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 36), 'usub', energy_565270)
        
        # Processing the call keyword arguments (line 222)
        kwargs_565272 = {}
        # Getting the type of 'exp' (line 222)
        exp_565269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 32), 'exp', False)
        # Calling exp(args, kwargs) (line 222)
        exp_call_result_565273 = invoke(stypy.reporting.localization.Localization(__file__, 222, 32), exp_565269, *[result___neg___565271], **kwargs_565272)
        
        # Processing the call keyword arguments (line 222)
        int_565274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 51), 'int')
        keyword_565275 = int_565274
        kwargs_565276 = {'axis': keyword_565275}
        # Getting the type of 'sum' (line 222)
        sum_565268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'sum', False)
        # Calling sum(args, kwargs) (line 222)
        sum_call_result_565277 = invoke(stypy.reporting.localization.Localization(__file__, 222, 28), sum_565268, *[exp_call_result_565273], **kwargs_565276)
        
        # Getting the type of 'result' (line 222)
        result_565278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'result')
        # Getting the type of 'i' (line 222)
        i_565279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'i')
        # Storing an element on a container (line 222)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 16), result_565278, (i_565279, sum_call_result_565277))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 209)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 224):
        
        # Assigning a BinOp to a Name (line 224):
        # Getting the type of 'result' (line 224)
        result_565280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 17), 'result')
        # Getting the type of 'self' (line 224)
        self_565281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 26), 'self')
        # Obtaining the member '_norm_factor' of a type (line 224)
        _norm_factor_565282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 26), self_565281, '_norm_factor')
        # Applying the binary operator 'div' (line 224)
        result_div_565283 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 17), 'div', result_565280, _norm_factor_565282)
        
        # Assigning a type to the variable 'result' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'result', result_div_565283)
        # Getting the type of 'result' (line 226)
        result_565284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'stypy_return_type', result_565284)
        
        # ################# End of 'evaluate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'evaluate' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_565285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565285)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'evaluate'
        return stypy_return_type_565285

    
    # Assigning a Name to a Name (line 228):

    @norecursion
    def integrate_gaussian(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'integrate_gaussian'
        module_type_store = module_type_store.open_function_context('integrate_gaussian', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde.integrate_gaussian.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde.integrate_gaussian.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde.integrate_gaussian.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde.integrate_gaussian.__dict__.__setitem__('stypy_function_name', 'gaussian_kde.integrate_gaussian')
        gaussian_kde.integrate_gaussian.__dict__.__setitem__('stypy_param_names_list', ['mean', 'cov'])
        gaussian_kde.integrate_gaussian.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde.integrate_gaussian.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde.integrate_gaussian.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde.integrate_gaussian.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde.integrate_gaussian.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde.integrate_gaussian.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.integrate_gaussian', ['mean', 'cov'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'integrate_gaussian', localization, ['mean', 'cov'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'integrate_gaussian(...)' code ##################

        str_565286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, (-1)), 'str', "\n        Multiply estimated density by a multivariate Gaussian and integrate\n        over the whole space.\n\n        Parameters\n        ----------\n        mean : aray_like\n            A 1-D array, specifying the mean of the Gaussian.\n        cov : array_like\n            A 2-D array, specifying the covariance matrix of the Gaussian.\n\n        Returns\n        -------\n        result : scalar\n            The value of the integral.\n\n        Raises\n        ------\n        ValueError\n            If the mean or covariance of the input Gaussian differs from\n            the KDE's dimensionality.\n\n        ")
        
        # Assigning a Call to a Name (line 254):
        
        # Assigning a Call to a Name (line 254):
        
        # Call to atleast_1d(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Call to squeeze(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'mean' (line 254)
        mean_565289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 34), 'mean', False)
        # Processing the call keyword arguments (line 254)
        kwargs_565290 = {}
        # Getting the type of 'squeeze' (line 254)
        squeeze_565288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'squeeze', False)
        # Calling squeeze(args, kwargs) (line 254)
        squeeze_call_result_565291 = invoke(stypy.reporting.localization.Localization(__file__, 254, 26), squeeze_565288, *[mean_565289], **kwargs_565290)
        
        # Processing the call keyword arguments (line 254)
        kwargs_565292 = {}
        # Getting the type of 'atleast_1d' (line 254)
        atleast_1d_565287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'atleast_1d', False)
        # Calling atleast_1d(args, kwargs) (line 254)
        atleast_1d_call_result_565293 = invoke(stypy.reporting.localization.Localization(__file__, 254, 15), atleast_1d_565287, *[squeeze_call_result_565291], **kwargs_565292)
        
        # Assigning a type to the variable 'mean' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'mean', atleast_1d_call_result_565293)
        
        # Assigning a Call to a Name (line 255):
        
        # Assigning a Call to a Name (line 255):
        
        # Call to atleast_2d(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'cov' (line 255)
        cov_565295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 25), 'cov', False)
        # Processing the call keyword arguments (line 255)
        kwargs_565296 = {}
        # Getting the type of 'atleast_2d' (line 255)
        atleast_2d_565294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 14), 'atleast_2d', False)
        # Calling atleast_2d(args, kwargs) (line 255)
        atleast_2d_call_result_565297 = invoke(stypy.reporting.localization.Localization(__file__, 255, 14), atleast_2d_565294, *[cov_565295], **kwargs_565296)
        
        # Assigning a type to the variable 'cov' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'cov', atleast_2d_call_result_565297)
        
        
        # Getting the type of 'mean' (line 257)
        mean_565298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 'mean')
        # Obtaining the member 'shape' of a type (line 257)
        shape_565299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 11), mean_565298, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 257)
        tuple_565300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 257)
        # Adding element type (line 257)
        # Getting the type of 'self' (line 257)
        self_565301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 26), 'self')
        # Obtaining the member 'd' of a type (line 257)
        d_565302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 26), self_565301, 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 26), tuple_565300, d_565302)
        
        # Applying the binary operator '!=' (line 257)
        result_ne_565303 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 11), '!=', shape_565299, tuple_565300)
        
        # Testing the type of an if condition (line 257)
        if_condition_565304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 8), result_ne_565303)
        # Assigning a type to the variable 'if_condition_565304' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'if_condition_565304', if_condition_565304)
        # SSA begins for if statement (line 257)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 258)
        # Processing the call arguments (line 258)
        str_565306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 29), 'str', 'mean does not have dimension %s')
        # Getting the type of 'self' (line 258)
        self_565307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 65), 'self', False)
        # Obtaining the member 'd' of a type (line 258)
        d_565308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 65), self_565307, 'd')
        # Applying the binary operator '%' (line 258)
        result_mod_565309 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 29), '%', str_565306, d_565308)
        
        # Processing the call keyword arguments (line 258)
        kwargs_565310 = {}
        # Getting the type of 'ValueError' (line 258)
        ValueError_565305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 258)
        ValueError_call_result_565311 = invoke(stypy.reporting.localization.Localization(__file__, 258, 18), ValueError_565305, *[result_mod_565309], **kwargs_565310)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 258, 12), ValueError_call_result_565311, 'raise parameter', BaseException)
        # SSA join for if statement (line 257)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'cov' (line 259)
        cov_565312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'cov')
        # Obtaining the member 'shape' of a type (line 259)
        shape_565313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 11), cov_565312, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 259)
        tuple_565314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 259)
        # Adding element type (line 259)
        # Getting the type of 'self' (line 259)
        self_565315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 25), 'self')
        # Obtaining the member 'd' of a type (line 259)
        d_565316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 25), self_565315, 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 25), tuple_565314, d_565316)
        # Adding element type (line 259)
        # Getting the type of 'self' (line 259)
        self_565317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 33), 'self')
        # Obtaining the member 'd' of a type (line 259)
        d_565318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 33), self_565317, 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 25), tuple_565314, d_565318)
        
        # Applying the binary operator '!=' (line 259)
        result_ne_565319 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 11), '!=', shape_565313, tuple_565314)
        
        # Testing the type of an if condition (line 259)
        if_condition_565320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 8), result_ne_565319)
        # Assigning a type to the variable 'if_condition_565320' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'if_condition_565320', if_condition_565320)
        # SSA begins for if statement (line 259)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 260)
        # Processing the call arguments (line 260)
        str_565322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 29), 'str', 'covariance does not have dimension %s')
        # Getting the type of 'self' (line 260)
        self_565323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 71), 'self', False)
        # Obtaining the member 'd' of a type (line 260)
        d_565324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 71), self_565323, 'd')
        # Applying the binary operator '%' (line 260)
        result_mod_565325 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 29), '%', str_565322, d_565324)
        
        # Processing the call keyword arguments (line 260)
        kwargs_565326 = {}
        # Getting the type of 'ValueError' (line 260)
        ValueError_565321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 260)
        ValueError_call_result_565327 = invoke(stypy.reporting.localization.Localization(__file__, 260, 18), ValueError_565321, *[result_mod_565325], **kwargs_565326)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 260, 12), ValueError_call_result_565327, 'raise parameter', BaseException)
        # SSA join for if statement (line 259)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        slice_565328 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 263, 15), None, None, None)
        # Getting the type of 'newaxis' (line 263)
        newaxis_565329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 23), 'newaxis')
        # Getting the type of 'mean' (line 263)
        mean_565330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'mean')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___565331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 15), mean_565330, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_565332 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), getitem___565331, (slice_565328, newaxis_565329))
        
        # Assigning a type to the variable 'mean' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'mean', subscript_call_result_565332)
        
        # Assigning a BinOp to a Name (line 265):
        
        # Assigning a BinOp to a Name (line 265):
        # Getting the type of 'self' (line 265)
        self_565333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 18), 'self')
        # Obtaining the member 'covariance' of a type (line 265)
        covariance_565334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 18), self_565333, 'covariance')
        # Getting the type of 'cov' (line 265)
        cov_565335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 36), 'cov')
        # Applying the binary operator '+' (line 265)
        result_add_565336 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 18), '+', covariance_565334, cov_565335)
        
        # Assigning a type to the variable 'sum_cov' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'sum_cov', result_add_565336)
        
        # Assigning a Call to a Name (line 270):
        
        # Assigning a Call to a Name (line 270):
        
        # Call to cho_factor(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'sum_cov' (line 270)
        sum_cov_565339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 41), 'sum_cov', False)
        # Processing the call keyword arguments (line 270)
        kwargs_565340 = {}
        # Getting the type of 'linalg' (line 270)
        linalg_565337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 23), 'linalg', False)
        # Obtaining the member 'cho_factor' of a type (line 270)
        cho_factor_565338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 23), linalg_565337, 'cho_factor')
        # Calling cho_factor(args, kwargs) (line 270)
        cho_factor_call_result_565341 = invoke(stypy.reporting.localization.Localization(__file__, 270, 23), cho_factor_565338, *[sum_cov_565339], **kwargs_565340)
        
        # Assigning a type to the variable 'sum_cov_chol' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'sum_cov_chol', cho_factor_call_result_565341)
        
        # Assigning a BinOp to a Name (line 272):
        
        # Assigning a BinOp to a Name (line 272):
        # Getting the type of 'self' (line 272)
        self_565342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'self')
        # Obtaining the member 'dataset' of a type (line 272)
        dataset_565343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 15), self_565342, 'dataset')
        # Getting the type of 'mean' (line 272)
        mean_565344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 30), 'mean')
        # Applying the binary operator '-' (line 272)
        result_sub_565345 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 15), '-', dataset_565343, mean_565344)
        
        # Assigning a type to the variable 'diff' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'diff', result_sub_565345)
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to cho_solve(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'sum_cov_chol' (line 273)
        sum_cov_chol_565348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 33), 'sum_cov_chol', False)
        # Getting the type of 'diff' (line 273)
        diff_565349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 47), 'diff', False)
        # Processing the call keyword arguments (line 273)
        kwargs_565350 = {}
        # Getting the type of 'linalg' (line 273)
        linalg_565346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'linalg', False)
        # Obtaining the member 'cho_solve' of a type (line 273)
        cho_solve_565347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), linalg_565346, 'cho_solve')
        # Calling cho_solve(args, kwargs) (line 273)
        cho_solve_call_result_565351 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), cho_solve_565347, *[sum_cov_chol_565348, diff_565349], **kwargs_565350)
        
        # Assigning a type to the variable 'tdiff' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'tdiff', cho_solve_call_result_565351)
        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to prod(...): (line 275)
        # Processing the call arguments (line 275)
        
        # Call to diagonal(...): (line 275)
        # Processing the call arguments (line 275)
        
        # Obtaining the type of the subscript
        int_565356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 52), 'int')
        # Getting the type of 'sum_cov_chol' (line 275)
        sum_cov_chol_565357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 39), 'sum_cov_chol', False)
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___565358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 39), sum_cov_chol_565357, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 275)
        subscript_call_result_565359 = invoke(stypy.reporting.localization.Localization(__file__, 275, 39), getitem___565358, int_565356)
        
        # Processing the call keyword arguments (line 275)
        kwargs_565360 = {}
        # Getting the type of 'np' (line 275)
        np_565354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'np', False)
        # Obtaining the member 'diagonal' of a type (line 275)
        diagonal_565355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 27), np_565354, 'diagonal')
        # Calling diagonal(args, kwargs) (line 275)
        diagonal_call_result_565361 = invoke(stypy.reporting.localization.Localization(__file__, 275, 27), diagonal_565355, *[subscript_call_result_565359], **kwargs_565360)
        
        # Processing the call keyword arguments (line 275)
        kwargs_565362 = {}
        # Getting the type of 'np' (line 275)
        np_565352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 'np', False)
        # Obtaining the member 'prod' of a type (line 275)
        prod_565353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 19), np_565352, 'prod')
        # Calling prod(args, kwargs) (line 275)
        prod_call_result_565363 = invoke(stypy.reporting.localization.Localization(__file__, 275, 19), prod_565353, *[diagonal_call_result_565361], **kwargs_565362)
        
        # Assigning a type to the variable 'sqrt_det' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'sqrt_det', prod_call_result_565363)
        
        # Assigning a BinOp to a Name (line 276):
        
        # Assigning a BinOp to a Name (line 276):
        
        # Call to power(...): (line 276)
        # Processing the call arguments (line 276)
        int_565365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 27), 'int')
        # Getting the type of 'pi' (line 276)
        pi_565366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 31), 'pi', False)
        # Applying the binary operator '*' (line 276)
        result_mul_565367 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 27), '*', int_565365, pi_565366)
        
        
        # Obtaining the type of the subscript
        int_565368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 49), 'int')
        # Getting the type of 'sum_cov' (line 276)
        sum_cov_565369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 35), 'sum_cov', False)
        # Obtaining the member 'shape' of a type (line 276)
        shape_565370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 35), sum_cov_565369, 'shape')
        # Obtaining the member '__getitem__' of a type (line 276)
        getitem___565371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 35), shape_565370, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 276)
        subscript_call_result_565372 = invoke(stypy.reporting.localization.Localization(__file__, 276, 35), getitem___565371, int_565368)
        
        float_565373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 54), 'float')
        # Applying the binary operator 'div' (line 276)
        result_div_565374 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 35), 'div', subscript_call_result_565372, float_565373)
        
        # Processing the call keyword arguments (line 276)
        kwargs_565375 = {}
        # Getting the type of 'power' (line 276)
        power_565364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 21), 'power', False)
        # Calling power(args, kwargs) (line 276)
        power_call_result_565376 = invoke(stypy.reporting.localization.Localization(__file__, 276, 21), power_565364, *[result_mul_565367, result_div_565374], **kwargs_565375)
        
        # Getting the type of 'sqrt_det' (line 276)
        sqrt_det_565377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 61), 'sqrt_det')
        # Applying the binary operator '*' (line 276)
        result_mul_565378 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 21), '*', power_call_result_565376, sqrt_det_565377)
        
        # Assigning a type to the variable 'norm_const' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'norm_const', result_mul_565378)
        
        # Assigning a BinOp to a Name (line 278):
        
        # Assigning a BinOp to a Name (line 278):
        
        # Call to sum(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'diff' (line 278)
        diff_565380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 23), 'diff', False)
        # Getting the type of 'tdiff' (line 278)
        tdiff_565381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 30), 'tdiff', False)
        # Applying the binary operator '*' (line 278)
        result_mul_565382 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 23), '*', diff_565380, tdiff_565381)
        
        # Processing the call keyword arguments (line 278)
        int_565383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 42), 'int')
        keyword_565384 = int_565383
        kwargs_565385 = {'axis': keyword_565384}
        # Getting the type of 'sum' (line 278)
        sum_565379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'sum', False)
        # Calling sum(args, kwargs) (line 278)
        sum_call_result_565386 = invoke(stypy.reporting.localization.Localization(__file__, 278, 19), sum_565379, *[result_mul_565382], **kwargs_565385)
        
        float_565387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 47), 'float')
        # Applying the binary operator 'div' (line 278)
        result_div_565388 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 19), 'div', sum_call_result_565386, float_565387)
        
        # Assigning a type to the variable 'energies' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'energies', result_div_565388)
        
        # Assigning a BinOp to a Name (line 279):
        
        # Assigning a BinOp to a Name (line 279):
        
        # Call to sum(...): (line 279)
        # Processing the call arguments (line 279)
        
        # Call to exp(...): (line 279)
        # Processing the call arguments (line 279)
        
        # Getting the type of 'energies' (line 279)
        energies_565391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 26), 'energies', False)
        # Applying the 'usub' unary operator (line 279)
        result___neg___565392 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 25), 'usub', energies_565391)
        
        # Processing the call keyword arguments (line 279)
        kwargs_565393 = {}
        # Getting the type of 'exp' (line 279)
        exp_565390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 21), 'exp', False)
        # Calling exp(args, kwargs) (line 279)
        exp_call_result_565394 = invoke(stypy.reporting.localization.Localization(__file__, 279, 21), exp_565390, *[result___neg___565392], **kwargs_565393)
        
        # Processing the call keyword arguments (line 279)
        int_565395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 42), 'int')
        keyword_565396 = int_565395
        kwargs_565397 = {'axis': keyword_565396}
        # Getting the type of 'sum' (line 279)
        sum_565389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 17), 'sum', False)
        # Calling sum(args, kwargs) (line 279)
        sum_call_result_565398 = invoke(stypy.reporting.localization.Localization(__file__, 279, 17), sum_565389, *[exp_call_result_565394], **kwargs_565397)
        
        # Getting the type of 'norm_const' (line 279)
        norm_const_565399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 47), 'norm_const')
        # Applying the binary operator 'div' (line 279)
        result_div_565400 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 17), 'div', sum_call_result_565398, norm_const_565399)
        
        # Getting the type of 'self' (line 279)
        self_565401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 60), 'self')
        # Obtaining the member 'n' of a type (line 279)
        n_565402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 60), self_565401, 'n')
        # Applying the binary operator 'div' (line 279)
        result_div_565403 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 58), 'div', result_div_565400, n_565402)
        
        # Assigning a type to the variable 'result' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'result', result_div_565403)
        # Getting the type of 'result' (line 281)
        result_565404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'stypy_return_type', result_565404)
        
        # ################# End of 'integrate_gaussian(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'integrate_gaussian' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_565405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565405)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'integrate_gaussian'
        return stypy_return_type_565405


    @norecursion
    def integrate_box_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'integrate_box_1d'
        module_type_store = module_type_store.open_function_context('integrate_box_1d', 283, 4, False)
        # Assigning a type to the variable 'self' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde.integrate_box_1d.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde.integrate_box_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde.integrate_box_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde.integrate_box_1d.__dict__.__setitem__('stypy_function_name', 'gaussian_kde.integrate_box_1d')
        gaussian_kde.integrate_box_1d.__dict__.__setitem__('stypy_param_names_list', ['low', 'high'])
        gaussian_kde.integrate_box_1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde.integrate_box_1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde.integrate_box_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde.integrate_box_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde.integrate_box_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde.integrate_box_1d.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.integrate_box_1d', ['low', 'high'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'integrate_box_1d', localization, ['low', 'high'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'integrate_box_1d(...)' code ##################

        str_565406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, (-1)), 'str', '\n        Computes the integral of a 1D pdf between two bounds.\n\n        Parameters\n        ----------\n        low : scalar\n            Lower bound of integration.\n        high : scalar\n            Upper bound of integration.\n\n        Returns\n        -------\n        value : scalar\n            The result of the integral.\n\n        Raises\n        ------\n        ValueError\n            If the KDE is over more than one dimension.\n\n        ')
        
        
        # Getting the type of 'self' (line 305)
        self_565407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 11), 'self')
        # Obtaining the member 'd' of a type (line 305)
        d_565408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 11), self_565407, 'd')
        int_565409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'int')
        # Applying the binary operator '!=' (line 305)
        result_ne_565410 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 11), '!=', d_565408, int_565409)
        
        # Testing the type of an if condition (line 305)
        if_condition_565411 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 8), result_ne_565410)
        # Assigning a type to the variable 'if_condition_565411' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'if_condition_565411', if_condition_565411)
        # SSA begins for if statement (line 305)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 306)
        # Processing the call arguments (line 306)
        str_565413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 29), 'str', 'integrate_box_1d() only handles 1D pdfs')
        # Processing the call keyword arguments (line 306)
        kwargs_565414 = {}
        # Getting the type of 'ValueError' (line 306)
        ValueError_565412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 306)
        ValueError_call_result_565415 = invoke(stypy.reporting.localization.Localization(__file__, 306, 18), ValueError_565412, *[str_565413], **kwargs_565414)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 306, 12), ValueError_call_result_565415, 'raise parameter', BaseException)
        # SSA join for if statement (line 305)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 308):
        
        # Assigning a Subscript to a Name (line 308):
        
        # Obtaining the type of the subscript
        int_565416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 45), 'int')
        
        # Call to ravel(...): (line 308)
        # Processing the call arguments (line 308)
        
        # Call to sqrt(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'self' (line 308)
        self_565419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 27), 'self', False)
        # Obtaining the member 'covariance' of a type (line 308)
        covariance_565420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 27), self_565419, 'covariance')
        # Processing the call keyword arguments (line 308)
        kwargs_565421 = {}
        # Getting the type of 'sqrt' (line 308)
        sqrt_565418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 22), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 308)
        sqrt_call_result_565422 = invoke(stypy.reporting.localization.Localization(__file__, 308, 22), sqrt_565418, *[covariance_565420], **kwargs_565421)
        
        # Processing the call keyword arguments (line 308)
        kwargs_565423 = {}
        # Getting the type of 'ravel' (line 308)
        ravel_565417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'ravel', False)
        # Calling ravel(args, kwargs) (line 308)
        ravel_call_result_565424 = invoke(stypy.reporting.localization.Localization(__file__, 308, 16), ravel_565417, *[sqrt_call_result_565422], **kwargs_565423)
        
        # Obtaining the member '__getitem__' of a type (line 308)
        getitem___565425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 16), ravel_call_result_565424, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 308)
        subscript_call_result_565426 = invoke(stypy.reporting.localization.Localization(__file__, 308, 16), getitem___565425, int_565416)
        
        # Assigning a type to the variable 'stdev' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'stdev', subscript_call_result_565426)
        
        # Assigning a Call to a Name (line 310):
        
        # Assigning a Call to a Name (line 310):
        
        # Call to ravel(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'low' (line 310)
        low_565428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 32), 'low', False)
        # Getting the type of 'self' (line 310)
        self_565429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 38), 'self', False)
        # Obtaining the member 'dataset' of a type (line 310)
        dataset_565430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 38), self_565429, 'dataset')
        # Applying the binary operator '-' (line 310)
        result_sub_565431 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 32), '-', low_565428, dataset_565430)
        
        # Getting the type of 'stdev' (line 310)
        stdev_565432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 54), 'stdev', False)
        # Applying the binary operator 'div' (line 310)
        result_div_565433 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 31), 'div', result_sub_565431, stdev_565432)
        
        # Processing the call keyword arguments (line 310)
        kwargs_565434 = {}
        # Getting the type of 'ravel' (line 310)
        ravel_565427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 25), 'ravel', False)
        # Calling ravel(args, kwargs) (line 310)
        ravel_call_result_565435 = invoke(stypy.reporting.localization.Localization(__file__, 310, 25), ravel_565427, *[result_div_565433], **kwargs_565434)
        
        # Assigning a type to the variable 'normalized_low' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'normalized_low', ravel_call_result_565435)
        
        # Assigning a Call to a Name (line 311):
        
        # Assigning a Call to a Name (line 311):
        
        # Call to ravel(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'high' (line 311)
        high_565437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 33), 'high', False)
        # Getting the type of 'self' (line 311)
        self_565438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 40), 'self', False)
        # Obtaining the member 'dataset' of a type (line 311)
        dataset_565439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 40), self_565438, 'dataset')
        # Applying the binary operator '-' (line 311)
        result_sub_565440 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 33), '-', high_565437, dataset_565439)
        
        # Getting the type of 'stdev' (line 311)
        stdev_565441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 56), 'stdev', False)
        # Applying the binary operator 'div' (line 311)
        result_div_565442 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 32), 'div', result_sub_565440, stdev_565441)
        
        # Processing the call keyword arguments (line 311)
        kwargs_565443 = {}
        # Getting the type of 'ravel' (line 311)
        ravel_565436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 26), 'ravel', False)
        # Calling ravel(args, kwargs) (line 311)
        ravel_call_result_565444 = invoke(stypy.reporting.localization.Localization(__file__, 311, 26), ravel_565436, *[result_div_565442], **kwargs_565443)
        
        # Assigning a type to the variable 'normalized_high' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'normalized_high', ravel_call_result_565444)
        
        # Assigning a Call to a Name (line 313):
        
        # Assigning a Call to a Name (line 313):
        
        # Call to mean(...): (line 313)
        # Processing the call arguments (line 313)
        
        # Call to ndtr(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'normalized_high' (line 313)
        normalized_high_565449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 37), 'normalized_high', False)
        # Processing the call keyword arguments (line 313)
        kwargs_565450 = {}
        # Getting the type of 'special' (line 313)
        special_565447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 24), 'special', False)
        # Obtaining the member 'ndtr' of a type (line 313)
        ndtr_565448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 24), special_565447, 'ndtr')
        # Calling ndtr(args, kwargs) (line 313)
        ndtr_call_result_565451 = invoke(stypy.reporting.localization.Localization(__file__, 313, 24), ndtr_565448, *[normalized_high_565449], **kwargs_565450)
        
        
        # Call to ndtr(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'normalized_low' (line 314)
        normalized_low_565454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 37), 'normalized_low', False)
        # Processing the call keyword arguments (line 314)
        kwargs_565455 = {}
        # Getting the type of 'special' (line 314)
        special_565452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 24), 'special', False)
        # Obtaining the member 'ndtr' of a type (line 314)
        ndtr_565453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 24), special_565452, 'ndtr')
        # Calling ndtr(args, kwargs) (line 314)
        ndtr_call_result_565456 = invoke(stypy.reporting.localization.Localization(__file__, 314, 24), ndtr_565453, *[normalized_low_565454], **kwargs_565455)
        
        # Applying the binary operator '-' (line 313)
        result_sub_565457 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 24), '-', ndtr_call_result_565451, ndtr_call_result_565456)
        
        # Processing the call keyword arguments (line 313)
        kwargs_565458 = {}
        # Getting the type of 'np' (line 313)
        np_565445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'np', False)
        # Obtaining the member 'mean' of a type (line 313)
        mean_565446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 16), np_565445, 'mean')
        # Calling mean(args, kwargs) (line 313)
        mean_call_result_565459 = invoke(stypy.reporting.localization.Localization(__file__, 313, 16), mean_565446, *[result_sub_565457], **kwargs_565458)
        
        # Assigning a type to the variable 'value' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'value', mean_call_result_565459)
        # Getting the type of 'value' (line 315)
        value_565460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 15), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'stypy_return_type', value_565460)
        
        # ################# End of 'integrate_box_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'integrate_box_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 283)
        stypy_return_type_565461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565461)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'integrate_box_1d'
        return stypy_return_type_565461


    @norecursion
    def integrate_box(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 317)
        None_565462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 60), 'None')
        defaults = [None_565462]
        # Create a new context for function 'integrate_box'
        module_type_store = module_type_store.open_function_context('integrate_box', 317, 4, False)
        # Assigning a type to the variable 'self' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde.integrate_box.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde.integrate_box.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde.integrate_box.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde.integrate_box.__dict__.__setitem__('stypy_function_name', 'gaussian_kde.integrate_box')
        gaussian_kde.integrate_box.__dict__.__setitem__('stypy_param_names_list', ['low_bounds', 'high_bounds', 'maxpts'])
        gaussian_kde.integrate_box.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde.integrate_box.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde.integrate_box.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde.integrate_box.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde.integrate_box.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde.integrate_box.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.integrate_box', ['low_bounds', 'high_bounds', 'maxpts'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'integrate_box', localization, ['low_bounds', 'high_bounds', 'maxpts'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'integrate_box(...)' code ##################

        str_565463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, (-1)), 'str', 'Computes the integral of a pdf over a rectangular interval.\n\n        Parameters\n        ----------\n        low_bounds : array_like\n            A 1-D array containing the lower bounds of integration.\n        high_bounds : array_like\n            A 1-D array containing the upper bounds of integration.\n        maxpts : int, optional\n            The maximum number of points to use for integration.\n\n        Returns\n        -------\n        value : scalar\n            The result of the integral.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 335)
        # Getting the type of 'maxpts' (line 335)
        maxpts_565464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'maxpts')
        # Getting the type of 'None' (line 335)
        None_565465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 25), 'None')
        
        (may_be_565466, more_types_in_union_565467) = may_not_be_none(maxpts_565464, None_565465)

        if may_be_565466:

            if more_types_in_union_565467:
                # Runtime conditional SSA (line 335)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Dict to a Name (line 336):
            
            # Assigning a Dict to a Name (line 336):
            
            # Obtaining an instance of the builtin type 'dict' (line 336)
            dict_565468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 25), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 336)
            # Adding element type (key, value) (line 336)
            str_565469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 26), 'str', 'maxpts')
            # Getting the type of 'maxpts' (line 336)
            maxpts_565470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 36), 'maxpts')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 25), dict_565468, (str_565469, maxpts_565470))
            
            # Assigning a type to the variable 'extra_kwds' (line 336)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'extra_kwds', dict_565468)

            if more_types_in_union_565467:
                # Runtime conditional SSA for else branch (line 335)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_565466) or more_types_in_union_565467):
            
            # Assigning a Dict to a Name (line 338):
            
            # Assigning a Dict to a Name (line 338):
            
            # Obtaining an instance of the builtin type 'dict' (line 338)
            dict_565471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 25), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 338)
            
            # Assigning a type to the variable 'extra_kwds' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'extra_kwds', dict_565471)

            if (may_be_565466 and more_types_in_union_565467):
                # SSA join for if statement (line 335)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 340):
        
        # Assigning a Subscript to a Name (line 340):
        
        # Obtaining the type of the subscript
        int_565472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 8), 'int')
        
        # Call to mvnun(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'low_bounds' (line 340)
        low_bounds_565475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 34), 'low_bounds', False)
        # Getting the type of 'high_bounds' (line 340)
        high_bounds_565476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 46), 'high_bounds', False)
        # Getting the type of 'self' (line 340)
        self_565477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 59), 'self', False)
        # Obtaining the member 'dataset' of a type (line 340)
        dataset_565478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 59), self_565477, 'dataset')
        # Getting the type of 'self' (line 341)
        self_565479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 34), 'self', False)
        # Obtaining the member 'covariance' of a type (line 341)
        covariance_565480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 34), self_565479, 'covariance')
        # Processing the call keyword arguments (line 340)
        # Getting the type of 'extra_kwds' (line 341)
        extra_kwds_565481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 53), 'extra_kwds', False)
        kwargs_565482 = {'extra_kwds_565481': extra_kwds_565481}
        # Getting the type of 'mvn' (line 340)
        mvn_565473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 24), 'mvn', False)
        # Obtaining the member 'mvnun' of a type (line 340)
        mvnun_565474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 24), mvn_565473, 'mvnun')
        # Calling mvnun(args, kwargs) (line 340)
        mvnun_call_result_565483 = invoke(stypy.reporting.localization.Localization(__file__, 340, 24), mvnun_565474, *[low_bounds_565475, high_bounds_565476, dataset_565478, covariance_565480], **kwargs_565482)
        
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___565484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), mvnun_call_result_565483, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_565485 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), getitem___565484, int_565472)
        
        # Assigning a type to the variable 'tuple_var_assignment_565078' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_565078', subscript_call_result_565485)
        
        # Assigning a Subscript to a Name (line 340):
        
        # Obtaining the type of the subscript
        int_565486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 8), 'int')
        
        # Call to mvnun(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'low_bounds' (line 340)
        low_bounds_565489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 34), 'low_bounds', False)
        # Getting the type of 'high_bounds' (line 340)
        high_bounds_565490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 46), 'high_bounds', False)
        # Getting the type of 'self' (line 340)
        self_565491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 59), 'self', False)
        # Obtaining the member 'dataset' of a type (line 340)
        dataset_565492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 59), self_565491, 'dataset')
        # Getting the type of 'self' (line 341)
        self_565493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 34), 'self', False)
        # Obtaining the member 'covariance' of a type (line 341)
        covariance_565494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 34), self_565493, 'covariance')
        # Processing the call keyword arguments (line 340)
        # Getting the type of 'extra_kwds' (line 341)
        extra_kwds_565495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 53), 'extra_kwds', False)
        kwargs_565496 = {'extra_kwds_565495': extra_kwds_565495}
        # Getting the type of 'mvn' (line 340)
        mvn_565487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 24), 'mvn', False)
        # Obtaining the member 'mvnun' of a type (line 340)
        mvnun_565488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 24), mvn_565487, 'mvnun')
        # Calling mvnun(args, kwargs) (line 340)
        mvnun_call_result_565497 = invoke(stypy.reporting.localization.Localization(__file__, 340, 24), mvnun_565488, *[low_bounds_565489, high_bounds_565490, dataset_565492, covariance_565494], **kwargs_565496)
        
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___565498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), mvnun_call_result_565497, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_565499 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), getitem___565498, int_565486)
        
        # Assigning a type to the variable 'tuple_var_assignment_565079' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_565079', subscript_call_result_565499)
        
        # Assigning a Name to a Name (line 340):
        # Getting the type of 'tuple_var_assignment_565078' (line 340)
        tuple_var_assignment_565078_565500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_565078')
        # Assigning a type to the variable 'value' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'value', tuple_var_assignment_565078_565500)
        
        # Assigning a Name to a Name (line 340):
        # Getting the type of 'tuple_var_assignment_565079' (line 340)
        tuple_var_assignment_565079_565501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'tuple_var_assignment_565079')
        # Assigning a type to the variable 'inform' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 15), 'inform', tuple_var_assignment_565079_565501)
        
        # Getting the type of 'inform' (line 342)
        inform_565502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 11), 'inform')
        # Testing the type of an if condition (line 342)
        if_condition_565503 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 8), inform_565502)
        # Assigning a type to the variable 'if_condition_565503' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'if_condition_565503', if_condition_565503)
        # SSA begins for if statement (line 342)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 343):
        
        # Assigning a BinOp to a Name (line 343):
        str_565504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 19), 'str', 'An integral in mvn.mvnun requires more points than %s')
        # Getting the type of 'self' (line 344)
        self_565505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 20), 'self')
        # Obtaining the member 'd' of a type (line 344)
        d_565506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 20), self_565505, 'd')
        int_565507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 29), 'int')
        # Applying the binary operator '*' (line 344)
        result_mul_565508 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 20), '*', d_565506, int_565507)
        
        # Applying the binary operator '%' (line 343)
        result_mod_565509 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 19), '%', str_565504, result_mul_565508)
        
        # Assigning a type to the variable 'msg' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'msg', result_mod_565509)
        
        # Call to warn(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'msg' (line 345)
        msg_565512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 26), 'msg', False)
        # Processing the call keyword arguments (line 345)
        kwargs_565513 = {}
        # Getting the type of 'warnings' (line 345)
        warnings_565510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 345)
        warn_565511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), warnings_565510, 'warn')
        # Calling warn(args, kwargs) (line 345)
        warn_call_result_565514 = invoke(stypy.reporting.localization.Localization(__file__, 345, 12), warn_565511, *[msg_565512], **kwargs_565513)
        
        # SSA join for if statement (line 342)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'value' (line 347)
        value_565515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 15), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'stypy_return_type', value_565515)
        
        # ################# End of 'integrate_box(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'integrate_box' in the type store
        # Getting the type of 'stypy_return_type' (line 317)
        stypy_return_type_565516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565516)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'integrate_box'
        return stypy_return_type_565516


    @norecursion
    def integrate_kde(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'integrate_kde'
        module_type_store = module_type_store.open_function_context('integrate_kde', 349, 4, False)
        # Assigning a type to the variable 'self' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde.integrate_kde.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde.integrate_kde.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde.integrate_kde.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde.integrate_kde.__dict__.__setitem__('stypy_function_name', 'gaussian_kde.integrate_kde')
        gaussian_kde.integrate_kde.__dict__.__setitem__('stypy_param_names_list', ['other'])
        gaussian_kde.integrate_kde.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde.integrate_kde.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde.integrate_kde.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde.integrate_kde.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde.integrate_kde.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde.integrate_kde.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.integrate_kde', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'integrate_kde', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'integrate_kde(...)' code ##################

        str_565517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, (-1)), 'str', '\n        Computes the integral of the product of this  kernel density estimate\n        with another.\n\n        Parameters\n        ----------\n        other : gaussian_kde instance\n            The other kde.\n\n        Returns\n        -------\n        value : scalar\n            The result of the integral.\n\n        Raises\n        ------\n        ValueError\n            If the KDEs have different dimensionality.\n\n        ')
        
        
        # Getting the type of 'other' (line 370)
        other_565518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 11), 'other')
        # Obtaining the member 'd' of a type (line 370)
        d_565519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 11), other_565518, 'd')
        # Getting the type of 'self' (line 370)
        self_565520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 22), 'self')
        # Obtaining the member 'd' of a type (line 370)
        d_565521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 22), self_565520, 'd')
        # Applying the binary operator '!=' (line 370)
        result_ne_565522 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 11), '!=', d_565519, d_565521)
        
        # Testing the type of an if condition (line 370)
        if_condition_565523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 8), result_ne_565522)
        # Assigning a type to the variable 'if_condition_565523' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'if_condition_565523', if_condition_565523)
        # SSA begins for if statement (line 370)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 371)
        # Processing the call arguments (line 371)
        str_565525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 29), 'str', 'KDEs are not the same dimensionality')
        # Processing the call keyword arguments (line 371)
        kwargs_565526 = {}
        # Getting the type of 'ValueError' (line 371)
        ValueError_565524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 371)
        ValueError_call_result_565527 = invoke(stypy.reporting.localization.Localization(__file__, 371, 18), ValueError_565524, *[str_565525], **kwargs_565526)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 371, 12), ValueError_call_result_565527, 'raise parameter', BaseException)
        # SSA join for if statement (line 370)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'other' (line 374)
        other_565528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 11), 'other')
        # Obtaining the member 'n' of a type (line 374)
        n_565529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 11), other_565528, 'n')
        # Getting the type of 'self' (line 374)
        self_565530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 21), 'self')
        # Obtaining the member 'n' of a type (line 374)
        n_565531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 21), self_565530, 'n')
        # Applying the binary operator '<' (line 374)
        result_lt_565532 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 11), '<', n_565529, n_565531)
        
        # Testing the type of an if condition (line 374)
        if_condition_565533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 8), result_lt_565532)
        # Assigning a type to the variable 'if_condition_565533' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'if_condition_565533', if_condition_565533)
        # SSA begins for if statement (line 374)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 375):
        
        # Assigning a Name to a Name (line 375):
        # Getting the type of 'other' (line 375)
        other_565534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 20), 'other')
        # Assigning a type to the variable 'small' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'small', other_565534)
        
        # Assigning a Name to a Name (line 376):
        
        # Assigning a Name to a Name (line 376):
        # Getting the type of 'self' (line 376)
        self_565535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 20), 'self')
        # Assigning a type to the variable 'large' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'large', self_565535)
        # SSA branch for the else part of an if statement (line 374)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 378):
        
        # Assigning a Name to a Name (line 378):
        # Getting the type of 'self' (line 378)
        self_565536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 20), 'self')
        # Assigning a type to the variable 'small' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'small', self_565536)
        
        # Assigning a Name to a Name (line 379):
        
        # Assigning a Name to a Name (line 379):
        # Getting the type of 'other' (line 379)
        other_565537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 20), 'other')
        # Assigning a type to the variable 'large' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'large', other_565537)
        # SSA join for if statement (line 374)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 381):
        
        # Assigning a BinOp to a Name (line 381):
        # Getting the type of 'small' (line 381)
        small_565538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 18), 'small')
        # Obtaining the member 'covariance' of a type (line 381)
        covariance_565539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 18), small_565538, 'covariance')
        # Getting the type of 'large' (line 381)
        large_565540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 37), 'large')
        # Obtaining the member 'covariance' of a type (line 381)
        covariance_565541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 37), large_565540, 'covariance')
        # Applying the binary operator '+' (line 381)
        result_add_565542 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 18), '+', covariance_565539, covariance_565541)
        
        # Assigning a type to the variable 'sum_cov' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'sum_cov', result_add_565542)
        
        # Assigning a Call to a Name (line 382):
        
        # Assigning a Call to a Name (line 382):
        
        # Call to cho_factor(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'sum_cov' (line 382)
        sum_cov_565545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 41), 'sum_cov', False)
        # Processing the call keyword arguments (line 382)
        kwargs_565546 = {}
        # Getting the type of 'linalg' (line 382)
        linalg_565543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 23), 'linalg', False)
        # Obtaining the member 'cho_factor' of a type (line 382)
        cho_factor_565544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 23), linalg_565543, 'cho_factor')
        # Calling cho_factor(args, kwargs) (line 382)
        cho_factor_call_result_565547 = invoke(stypy.reporting.localization.Localization(__file__, 382, 23), cho_factor_565544, *[sum_cov_565545], **kwargs_565546)
        
        # Assigning a type to the variable 'sum_cov_chol' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'sum_cov_chol', cho_factor_call_result_565547)
        
        # Assigning a Num to a Name (line 383):
        
        # Assigning a Num to a Name (line 383):
        float_565548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 17), 'float')
        # Assigning a type to the variable 'result' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'result', float_565548)
        
        
        # Call to range(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'small' (line 384)
        small_565550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 23), 'small', False)
        # Obtaining the member 'n' of a type (line 384)
        n_565551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 23), small_565550, 'n')
        # Processing the call keyword arguments (line 384)
        kwargs_565552 = {}
        # Getting the type of 'range' (line 384)
        range_565549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 17), 'range', False)
        # Calling range(args, kwargs) (line 384)
        range_call_result_565553 = invoke(stypy.reporting.localization.Localization(__file__, 384, 17), range_565549, *[n_565551], **kwargs_565552)
        
        # Testing the type of a for loop iterable (line 384)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 384, 8), range_call_result_565553)
        # Getting the type of the for loop variable (line 384)
        for_loop_var_565554 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 384, 8), range_call_result_565553)
        # Assigning a type to the variable 'i' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'i', for_loop_var_565554)
        # SSA begins for a for statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 385):
        
        # Assigning a Subscript to a Name (line 385):
        
        # Obtaining the type of the subscript
        slice_565555 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 385, 19), None, None, None)
        # Getting the type of 'i' (line 385)
        i_565556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 36), 'i')
        # Getting the type of 'newaxis' (line 385)
        newaxis_565557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 39), 'newaxis')
        # Getting the type of 'small' (line 385)
        small_565558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 19), 'small')
        # Obtaining the member 'dataset' of a type (line 385)
        dataset_565559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 19), small_565558, 'dataset')
        # Obtaining the member '__getitem__' of a type (line 385)
        getitem___565560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 19), dataset_565559, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 385)
        subscript_call_result_565561 = invoke(stypy.reporting.localization.Localization(__file__, 385, 19), getitem___565560, (slice_565555, i_565556, newaxis_565557))
        
        # Assigning a type to the variable 'mean' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'mean', subscript_call_result_565561)
        
        # Assigning a BinOp to a Name (line 386):
        
        # Assigning a BinOp to a Name (line 386):
        # Getting the type of 'large' (line 386)
        large_565562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 19), 'large')
        # Obtaining the member 'dataset' of a type (line 386)
        dataset_565563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 19), large_565562, 'dataset')
        # Getting the type of 'mean' (line 386)
        mean_565564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 35), 'mean')
        # Applying the binary operator '-' (line 386)
        result_sub_565565 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 19), '-', dataset_565563, mean_565564)
        
        # Assigning a type to the variable 'diff' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'diff', result_sub_565565)
        
        # Assigning a Call to a Name (line 387):
        
        # Assigning a Call to a Name (line 387):
        
        # Call to cho_solve(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'sum_cov_chol' (line 387)
        sum_cov_chol_565568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 37), 'sum_cov_chol', False)
        # Getting the type of 'diff' (line 387)
        diff_565569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 51), 'diff', False)
        # Processing the call keyword arguments (line 387)
        kwargs_565570 = {}
        # Getting the type of 'linalg' (line 387)
        linalg_565566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 20), 'linalg', False)
        # Obtaining the member 'cho_solve' of a type (line 387)
        cho_solve_565567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 20), linalg_565566, 'cho_solve')
        # Calling cho_solve(args, kwargs) (line 387)
        cho_solve_call_result_565571 = invoke(stypy.reporting.localization.Localization(__file__, 387, 20), cho_solve_565567, *[sum_cov_chol_565568, diff_565569], **kwargs_565570)
        
        # Assigning a type to the variable 'tdiff' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'tdiff', cho_solve_call_result_565571)
        
        # Assigning a BinOp to a Name (line 389):
        
        # Assigning a BinOp to a Name (line 389):
        
        # Call to sum(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'diff' (line 389)
        diff_565573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 27), 'diff', False)
        # Getting the type of 'tdiff' (line 389)
        tdiff_565574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 34), 'tdiff', False)
        # Applying the binary operator '*' (line 389)
        result_mul_565575 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 27), '*', diff_565573, tdiff_565574)
        
        # Processing the call keyword arguments (line 389)
        int_565576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 46), 'int')
        keyword_565577 = int_565576
        kwargs_565578 = {'axis': keyword_565577}
        # Getting the type of 'sum' (line 389)
        sum_565572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 23), 'sum', False)
        # Calling sum(args, kwargs) (line 389)
        sum_call_result_565579 = invoke(stypy.reporting.localization.Localization(__file__, 389, 23), sum_565572, *[result_mul_565575], **kwargs_565578)
        
        float_565580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 51), 'float')
        # Applying the binary operator 'div' (line 389)
        result_div_565581 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 23), 'div', sum_call_result_565579, float_565580)
        
        # Assigning a type to the variable 'energies' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'energies', result_div_565581)
        
        # Getting the type of 'result' (line 390)
        result_565582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'result')
        
        # Call to sum(...): (line 390)
        # Processing the call arguments (line 390)
        
        # Call to exp(...): (line 390)
        # Processing the call arguments (line 390)
        
        # Getting the type of 'energies' (line 390)
        energies_565585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 31), 'energies', False)
        # Applying the 'usub' unary operator (line 390)
        result___neg___565586 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 30), 'usub', energies_565585)
        
        # Processing the call keyword arguments (line 390)
        kwargs_565587 = {}
        # Getting the type of 'exp' (line 390)
        exp_565584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 26), 'exp', False)
        # Calling exp(args, kwargs) (line 390)
        exp_call_result_565588 = invoke(stypy.reporting.localization.Localization(__file__, 390, 26), exp_565584, *[result___neg___565586], **kwargs_565587)
        
        # Processing the call keyword arguments (line 390)
        int_565589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 47), 'int')
        keyword_565590 = int_565589
        kwargs_565591 = {'axis': keyword_565590}
        # Getting the type of 'sum' (line 390)
        sum_565583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 22), 'sum', False)
        # Calling sum(args, kwargs) (line 390)
        sum_call_result_565592 = invoke(stypy.reporting.localization.Localization(__file__, 390, 22), sum_565583, *[exp_call_result_565588], **kwargs_565591)
        
        # Applying the binary operator '+=' (line 390)
        result_iadd_565593 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 12), '+=', result_565582, sum_call_result_565592)
        # Assigning a type to the variable 'result' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'result', result_iadd_565593)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 392):
        
        # Assigning a Call to a Name (line 392):
        
        # Call to prod(...): (line 392)
        # Processing the call arguments (line 392)
        
        # Call to diagonal(...): (line 392)
        # Processing the call arguments (line 392)
        
        # Obtaining the type of the subscript
        int_565598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 52), 'int')
        # Getting the type of 'sum_cov_chol' (line 392)
        sum_cov_chol_565599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 39), 'sum_cov_chol', False)
        # Obtaining the member '__getitem__' of a type (line 392)
        getitem___565600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 39), sum_cov_chol_565599, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 392)
        subscript_call_result_565601 = invoke(stypy.reporting.localization.Localization(__file__, 392, 39), getitem___565600, int_565598)
        
        # Processing the call keyword arguments (line 392)
        kwargs_565602 = {}
        # Getting the type of 'np' (line 392)
        np_565596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 27), 'np', False)
        # Obtaining the member 'diagonal' of a type (line 392)
        diagonal_565597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 27), np_565596, 'diagonal')
        # Calling diagonal(args, kwargs) (line 392)
        diagonal_call_result_565603 = invoke(stypy.reporting.localization.Localization(__file__, 392, 27), diagonal_565597, *[subscript_call_result_565601], **kwargs_565602)
        
        # Processing the call keyword arguments (line 392)
        kwargs_565604 = {}
        # Getting the type of 'np' (line 392)
        np_565594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'np', False)
        # Obtaining the member 'prod' of a type (line 392)
        prod_565595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 19), np_565594, 'prod')
        # Calling prod(args, kwargs) (line 392)
        prod_call_result_565605 = invoke(stypy.reporting.localization.Localization(__file__, 392, 19), prod_565595, *[diagonal_call_result_565603], **kwargs_565604)
        
        # Assigning a type to the variable 'sqrt_det' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'sqrt_det', prod_call_result_565605)
        
        # Assigning a BinOp to a Name (line 393):
        
        # Assigning a BinOp to a Name (line 393):
        
        # Call to power(...): (line 393)
        # Processing the call arguments (line 393)
        int_565607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 27), 'int')
        # Getting the type of 'pi' (line 393)
        pi_565608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 31), 'pi', False)
        # Applying the binary operator '*' (line 393)
        result_mul_565609 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 27), '*', int_565607, pi_565608)
        
        
        # Obtaining the type of the subscript
        int_565610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 49), 'int')
        # Getting the type of 'sum_cov' (line 393)
        sum_cov_565611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 35), 'sum_cov', False)
        # Obtaining the member 'shape' of a type (line 393)
        shape_565612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 35), sum_cov_565611, 'shape')
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___565613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 35), shape_565612, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_565614 = invoke(stypy.reporting.localization.Localization(__file__, 393, 35), getitem___565613, int_565610)
        
        float_565615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 54), 'float')
        # Applying the binary operator 'div' (line 393)
        result_div_565616 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 35), 'div', subscript_call_result_565614, float_565615)
        
        # Processing the call keyword arguments (line 393)
        kwargs_565617 = {}
        # Getting the type of 'power' (line 393)
        power_565606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 21), 'power', False)
        # Calling power(args, kwargs) (line 393)
        power_call_result_565618 = invoke(stypy.reporting.localization.Localization(__file__, 393, 21), power_565606, *[result_mul_565609, result_div_565616], **kwargs_565617)
        
        # Getting the type of 'sqrt_det' (line 393)
        sqrt_det_565619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 61), 'sqrt_det')
        # Applying the binary operator '*' (line 393)
        result_mul_565620 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 21), '*', power_call_result_565618, sqrt_det_565619)
        
        # Assigning a type to the variable 'norm_const' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'norm_const', result_mul_565620)
        
        # Getting the type of 'result' (line 395)
        result_565621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'result')
        # Getting the type of 'norm_const' (line 395)
        norm_const_565622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 18), 'norm_const')
        # Getting the type of 'large' (line 395)
        large_565623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 31), 'large')
        # Obtaining the member 'n' of a type (line 395)
        n_565624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 31), large_565623, 'n')
        # Applying the binary operator '*' (line 395)
        result_mul_565625 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 18), '*', norm_const_565622, n_565624)
        
        # Getting the type of 'small' (line 395)
        small_565626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 41), 'small')
        # Obtaining the member 'n' of a type (line 395)
        n_565627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 41), small_565626, 'n')
        # Applying the binary operator '*' (line 395)
        result_mul_565628 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 39), '*', result_mul_565625, n_565627)
        
        # Applying the binary operator 'div=' (line 395)
        result_div_565629 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 8), 'div=', result_565621, result_mul_565628)
        # Assigning a type to the variable 'result' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'result', result_div_565629)
        
        # Getting the type of 'result' (line 397)
        result_565630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'stypy_return_type', result_565630)
        
        # ################# End of 'integrate_kde(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'integrate_kde' in the type store
        # Getting the type of 'stypy_return_type' (line 349)
        stypy_return_type_565631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565631)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'integrate_kde'
        return stypy_return_type_565631


    @norecursion
    def resample(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 399)
        None_565632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 28), 'None')
        defaults = [None_565632]
        # Create a new context for function 'resample'
        module_type_store = module_type_store.open_function_context('resample', 399, 4, False)
        # Assigning a type to the variable 'self' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde.resample.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde.resample.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde.resample.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde.resample.__dict__.__setitem__('stypy_function_name', 'gaussian_kde.resample')
        gaussian_kde.resample.__dict__.__setitem__('stypy_param_names_list', ['size'])
        gaussian_kde.resample.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde.resample.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde.resample.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde.resample.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde.resample.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde.resample.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.resample', ['size'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'resample', localization, ['size'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'resample(...)' code ##################

        str_565633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, (-1)), 'str', '\n        Randomly sample a dataset from the estimated pdf.\n\n        Parameters\n        ----------\n        size : int, optional\n            The number of samples to draw.  If not provided, then the size is\n            the same as the underlying dataset.\n\n        Returns\n        -------\n        resample : (self.d, `size`) ndarray\n            The sampled dataset.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 415)
        # Getting the type of 'size' (line 415)
        size_565634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 11), 'size')
        # Getting the type of 'None' (line 415)
        None_565635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 19), 'None')
        
        (may_be_565636, more_types_in_union_565637) = may_be_none(size_565634, None_565635)

        if may_be_565636:

            if more_types_in_union_565637:
                # Runtime conditional SSA (line 415)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 416):
            
            # Assigning a Attribute to a Name (line 416):
            # Getting the type of 'self' (line 416)
            self_565638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 19), 'self')
            # Obtaining the member 'n' of a type (line 416)
            n_565639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 19), self_565638, 'n')
            # Assigning a type to the variable 'size' (line 416)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'size', n_565639)

            if more_types_in_union_565637:
                # SSA join for if statement (line 415)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 418):
        
        # Assigning a Call to a Name (line 418):
        
        # Call to transpose(...): (line 418)
        # Processing the call arguments (line 418)
        
        # Call to multivariate_normal(...): (line 418)
        # Processing the call arguments (line 418)
        
        # Call to zeros(...): (line 418)
        # Processing the call arguments (line 418)
        
        # Obtaining an instance of the builtin type 'tuple' (line 418)
        tuple_565643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 418)
        # Adding element type (line 418)
        # Getting the type of 'self' (line 418)
        self_565644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 52), 'self', False)
        # Obtaining the member 'd' of a type (line 418)
        d_565645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 52), self_565644, 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 52), tuple_565643, d_565645)
        
        # Getting the type of 'float' (line 418)
        float_565646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 62), 'float', False)
        # Processing the call keyword arguments (line 418)
        kwargs_565647 = {}
        # Getting the type of 'zeros' (line 418)
        zeros_565642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 45), 'zeros', False)
        # Calling zeros(args, kwargs) (line 418)
        zeros_call_result_565648 = invoke(stypy.reporting.localization.Localization(__file__, 418, 45), zeros_565642, *[tuple_565643, float_565646], **kwargs_565647)
        
        # Getting the type of 'self' (line 419)
        self_565649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 25), 'self', False)
        # Obtaining the member 'covariance' of a type (line 419)
        covariance_565650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 25), self_565649, 'covariance')
        # Processing the call keyword arguments (line 418)
        # Getting the type of 'size' (line 419)
        size_565651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 47), 'size', False)
        keyword_565652 = size_565651
        kwargs_565653 = {'size': keyword_565652}
        # Getting the type of 'multivariate_normal' (line 418)
        multivariate_normal_565641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 25), 'multivariate_normal', False)
        # Calling multivariate_normal(args, kwargs) (line 418)
        multivariate_normal_call_result_565654 = invoke(stypy.reporting.localization.Localization(__file__, 418, 25), multivariate_normal_565641, *[zeros_call_result_565648, covariance_565650], **kwargs_565653)
        
        # Processing the call keyword arguments (line 418)
        kwargs_565655 = {}
        # Getting the type of 'transpose' (line 418)
        transpose_565640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'transpose', False)
        # Calling transpose(args, kwargs) (line 418)
        transpose_call_result_565656 = invoke(stypy.reporting.localization.Localization(__file__, 418, 15), transpose_565640, *[multivariate_normal_call_result_565654], **kwargs_565655)
        
        # Assigning a type to the variable 'norm' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'norm', transpose_call_result_565656)
        
        # Assigning a Call to a Name (line 420):
        
        # Assigning a Call to a Name (line 420):
        
        # Call to randint(...): (line 420)
        # Processing the call arguments (line 420)
        int_565658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 26), 'int')
        # Getting the type of 'self' (line 420)
        self_565659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 29), 'self', False)
        # Obtaining the member 'n' of a type (line 420)
        n_565660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 29), self_565659, 'n')
        # Processing the call keyword arguments (line 420)
        # Getting the type of 'size' (line 420)
        size_565661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 42), 'size', False)
        keyword_565662 = size_565661
        kwargs_565663 = {'size': keyword_565662}
        # Getting the type of 'randint' (line 420)
        randint_565657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 18), 'randint', False)
        # Calling randint(args, kwargs) (line 420)
        randint_call_result_565664 = invoke(stypy.reporting.localization.Localization(__file__, 420, 18), randint_565657, *[int_565658, n_565660], **kwargs_565663)
        
        # Assigning a type to the variable 'indices' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'indices', randint_call_result_565664)
        
        # Assigning a Subscript to a Name (line 421):
        
        # Assigning a Subscript to a Name (line 421):
        
        # Obtaining the type of the subscript
        slice_565665 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 421, 16), None, None, None)
        # Getting the type of 'indices' (line 421)
        indices_565666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 32), 'indices')
        # Getting the type of 'self' (line 421)
        self_565667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 16), 'self')
        # Obtaining the member 'dataset' of a type (line 421)
        dataset_565668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 16), self_565667, 'dataset')
        # Obtaining the member '__getitem__' of a type (line 421)
        getitem___565669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 16), dataset_565668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 421)
        subscript_call_result_565670 = invoke(stypy.reporting.localization.Localization(__file__, 421, 16), getitem___565669, (slice_565665, indices_565666))
        
        # Assigning a type to the variable 'means' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'means', subscript_call_result_565670)
        # Getting the type of 'means' (line 423)
        means_565671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'means')
        # Getting the type of 'norm' (line 423)
        norm_565672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 23), 'norm')
        # Applying the binary operator '+' (line 423)
        result_add_565673 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 15), '+', means_565671, norm_565672)
        
        # Assigning a type to the variable 'stypy_return_type' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'stypy_return_type', result_add_565673)
        
        # ################# End of 'resample(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'resample' in the type store
        # Getting the type of 'stypy_return_type' (line 399)
        stypy_return_type_565674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565674)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'resample'
        return stypy_return_type_565674


    @norecursion
    def scotts_factor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'scotts_factor'
        module_type_store = module_type_store.open_function_context('scotts_factor', 425, 4, False)
        # Assigning a type to the variable 'self' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde.scotts_factor.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde.scotts_factor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde.scotts_factor.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde.scotts_factor.__dict__.__setitem__('stypy_function_name', 'gaussian_kde.scotts_factor')
        gaussian_kde.scotts_factor.__dict__.__setitem__('stypy_param_names_list', [])
        gaussian_kde.scotts_factor.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde.scotts_factor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde.scotts_factor.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde.scotts_factor.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde.scotts_factor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde.scotts_factor.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.scotts_factor', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'scotts_factor', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'scotts_factor(...)' code ##################

        
        # Call to power(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'self' (line 426)
        self_565676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 21), 'self', False)
        # Obtaining the member 'n' of a type (line 426)
        n_565677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 21), self_565676, 'n')
        float_565678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 29), 'float')
        # Getting the type of 'self' (line 426)
        self_565679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 34), 'self', False)
        # Obtaining the member 'd' of a type (line 426)
        d_565680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 34), self_565679, 'd')
        int_565681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 41), 'int')
        # Applying the binary operator '+' (line 426)
        result_add_565682 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 34), '+', d_565680, int_565681)
        
        # Applying the binary operator 'div' (line 426)
        result_div_565683 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 29), 'div', float_565678, result_add_565682)
        
        # Processing the call keyword arguments (line 426)
        kwargs_565684 = {}
        # Getting the type of 'power' (line 426)
        power_565675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 15), 'power', False)
        # Calling power(args, kwargs) (line 426)
        power_call_result_565685 = invoke(stypy.reporting.localization.Localization(__file__, 426, 15), power_565675, *[n_565677, result_div_565683], **kwargs_565684)
        
        # Assigning a type to the variable 'stypy_return_type' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'stypy_return_type', power_call_result_565685)
        
        # ################# End of 'scotts_factor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'scotts_factor' in the type store
        # Getting the type of 'stypy_return_type' (line 425)
        stypy_return_type_565686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565686)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'scotts_factor'
        return stypy_return_type_565686


    @norecursion
    def silverman_factor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'silverman_factor'
        module_type_store = module_type_store.open_function_context('silverman_factor', 428, 4, False)
        # Assigning a type to the variable 'self' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde.silverman_factor.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde.silverman_factor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde.silverman_factor.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde.silverman_factor.__dict__.__setitem__('stypy_function_name', 'gaussian_kde.silverman_factor')
        gaussian_kde.silverman_factor.__dict__.__setitem__('stypy_param_names_list', [])
        gaussian_kde.silverman_factor.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde.silverman_factor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde.silverman_factor.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde.silverman_factor.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde.silverman_factor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde.silverman_factor.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.silverman_factor', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'silverman_factor', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'silverman_factor(...)' code ##################

        
        # Call to power(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'self' (line 429)
        self_565688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 21), 'self', False)
        # Obtaining the member 'n' of a type (line 429)
        n_565689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 21), self_565688, 'n')
        # Getting the type of 'self' (line 429)
        self_565690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 29), 'self', False)
        # Obtaining the member 'd' of a type (line 429)
        d_565691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 29), self_565690, 'd')
        float_565692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 36), 'float')
        # Applying the binary operator '+' (line 429)
        result_add_565693 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 29), '+', d_565691, float_565692)
        
        # Applying the binary operator '*' (line 429)
        result_mul_565694 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 21), '*', n_565689, result_add_565693)
        
        float_565695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 41), 'float')
        # Applying the binary operator 'div' (line 429)
        result_div_565696 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 40), 'div', result_mul_565694, float_565695)
        
        float_565697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 46), 'float')
        # Getting the type of 'self' (line 429)
        self_565698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 51), 'self', False)
        # Obtaining the member 'd' of a type (line 429)
        d_565699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 51), self_565698, 'd')
        int_565700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 58), 'int')
        # Applying the binary operator '+' (line 429)
        result_add_565701 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 51), '+', d_565699, int_565700)
        
        # Applying the binary operator 'div' (line 429)
        result_div_565702 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 46), 'div', float_565697, result_add_565701)
        
        # Processing the call keyword arguments (line 429)
        kwargs_565703 = {}
        # Getting the type of 'power' (line 429)
        power_565687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 15), 'power', False)
        # Calling power(args, kwargs) (line 429)
        power_call_result_565704 = invoke(stypy.reporting.localization.Localization(__file__, 429, 15), power_565687, *[result_div_565696, result_div_565702], **kwargs_565703)
        
        # Assigning a type to the variable 'stypy_return_type' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'stypy_return_type', power_call_result_565704)
        
        # ################# End of 'silverman_factor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'silverman_factor' in the type store
        # Getting the type of 'stypy_return_type' (line 428)
        stypy_return_type_565705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565705)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'silverman_factor'
        return stypy_return_type_565705

    
    # Assigning a Name to a Name (line 432):
    
    # Assigning a Str to a Attribute (line 433):

    @norecursion
    def set_bandwidth(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 439)
        None_565706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 38), 'None')
        defaults = [None_565706]
        # Create a new context for function 'set_bandwidth'
        module_type_store = module_type_store.open_function_context('set_bandwidth', 439, 4, False)
        # Assigning a type to the variable 'self' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde.set_bandwidth.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde.set_bandwidth.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde.set_bandwidth.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde.set_bandwidth.__dict__.__setitem__('stypy_function_name', 'gaussian_kde.set_bandwidth')
        gaussian_kde.set_bandwidth.__dict__.__setitem__('stypy_param_names_list', ['bw_method'])
        gaussian_kde.set_bandwidth.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde.set_bandwidth.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde.set_bandwidth.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde.set_bandwidth.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde.set_bandwidth.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde.set_bandwidth.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.set_bandwidth', ['bw_method'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_bandwidth', localization, ['bw_method'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_bandwidth(...)' code ##################

        str_565707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, (-1)), 'str', "Compute the estimator bandwidth with given method.\n\n        The new bandwidth calculated after a call to `set_bandwidth` is used\n        for subsequent evaluations of the estimated density.\n\n        Parameters\n        ----------\n        bw_method : str, scalar or callable, optional\n            The method used to calculate the estimator bandwidth.  This can be\n            'scott', 'silverman', a scalar constant or a callable.  If a\n            scalar, this will be used directly as `kde.factor`.  If a callable,\n            it should take a `gaussian_kde` instance as only parameter and\n            return a scalar.  If None (default), nothing happens; the current\n            `kde.covariance_factor` method is kept.\n\n        Notes\n        -----\n        .. versionadded:: 0.11\n\n        Examples\n        --------\n        >>> import scipy.stats as stats\n        >>> x1 = np.array([-7, -5, 1, 4, 5.])\n        >>> kde = stats.gaussian_kde(x1)\n        >>> xs = np.linspace(-10, 10, num=50)\n        >>> y1 = kde(xs)\n        >>> kde.set_bandwidth(bw_method='silverman')\n        >>> y2 = kde(xs)\n        >>> kde.set_bandwidth(bw_method=kde.factor / 3.)\n        >>> y3 = kde(xs)\n\n        >>> import matplotlib.pyplot as plt\n        >>> fig, ax = plt.subplots()\n        >>> ax.plot(x1, np.ones(x1.shape) / (4. * x1.size), 'bo',\n        ...         label='Data points (rescaled)')\n        >>> ax.plot(xs, y1, label='Scott (default)')\n        >>> ax.plot(xs, y2, label='Silverman')\n        >>> ax.plot(xs, y3, label='Const (1/3 * Silverman)')\n        >>> ax.legend()\n        >>> plt.show()\n\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 482)
        # Getting the type of 'bw_method' (line 482)
        bw_method_565708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 11), 'bw_method')
        # Getting the type of 'None' (line 482)
        None_565709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 24), 'None')
        
        (may_be_565710, more_types_in_union_565711) = may_be_none(bw_method_565708, None_565709)

        if may_be_565710:

            if more_types_in_union_565711:
                # Runtime conditional SSA (line 482)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            pass

            if more_types_in_union_565711:
                # Runtime conditional SSA for else branch (line 482)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_565710) or more_types_in_union_565711):
            
            
            # Getting the type of 'bw_method' (line 484)
            bw_method_565712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 13), 'bw_method')
            str_565713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 26), 'str', 'scott')
            # Applying the binary operator '==' (line 484)
            result_eq_565714 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 13), '==', bw_method_565712, str_565713)
            
            # Testing the type of an if condition (line 484)
            if_condition_565715 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 484, 13), result_eq_565714)
            # Assigning a type to the variable 'if_condition_565715' (line 484)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 13), 'if_condition_565715', if_condition_565715)
            # SSA begins for if statement (line 484)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 485):
            
            # Assigning a Attribute to a Attribute (line 485):
            # Getting the type of 'self' (line 485)
            self_565716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 37), 'self')
            # Obtaining the member 'scotts_factor' of a type (line 485)
            scotts_factor_565717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 37), self_565716, 'scotts_factor')
            # Getting the type of 'self' (line 485)
            self_565718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'self')
            # Setting the type of the member 'covariance_factor' of a type (line 485)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 12), self_565718, 'covariance_factor', scotts_factor_565717)
            # SSA branch for the else part of an if statement (line 484)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'bw_method' (line 486)
            bw_method_565719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 13), 'bw_method')
            str_565720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 26), 'str', 'silverman')
            # Applying the binary operator '==' (line 486)
            result_eq_565721 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 13), '==', bw_method_565719, str_565720)
            
            # Testing the type of an if condition (line 486)
            if_condition_565722 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 486, 13), result_eq_565721)
            # Assigning a type to the variable 'if_condition_565722' (line 486)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 13), 'if_condition_565722', if_condition_565722)
            # SSA begins for if statement (line 486)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 487):
            
            # Assigning a Attribute to a Attribute (line 487):
            # Getting the type of 'self' (line 487)
            self_565723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 37), 'self')
            # Obtaining the member 'silverman_factor' of a type (line 487)
            silverman_factor_565724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 37), self_565723, 'silverman_factor')
            # Getting the type of 'self' (line 487)
            self_565725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'self')
            # Setting the type of the member 'covariance_factor' of a type (line 487)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 12), self_565725, 'covariance_factor', silverman_factor_565724)
            # SSA branch for the else part of an if statement (line 486)
            module_type_store.open_ssa_branch('else')
            
            
            # Evaluating a boolean operation
            
            # Call to isscalar(...): (line 488)
            # Processing the call arguments (line 488)
            # Getting the type of 'bw_method' (line 488)
            bw_method_565728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 25), 'bw_method', False)
            # Processing the call keyword arguments (line 488)
            kwargs_565729 = {}
            # Getting the type of 'np' (line 488)
            np_565726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 13), 'np', False)
            # Obtaining the member 'isscalar' of a type (line 488)
            isscalar_565727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 13), np_565726, 'isscalar')
            # Calling isscalar(args, kwargs) (line 488)
            isscalar_call_result_565730 = invoke(stypy.reporting.localization.Localization(__file__, 488, 13), isscalar_565727, *[bw_method_565728], **kwargs_565729)
            
            
            
            # Call to isinstance(...): (line 488)
            # Processing the call arguments (line 488)
            # Getting the type of 'bw_method' (line 488)
            bw_method_565732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 55), 'bw_method', False)
            # Getting the type of 'string_types' (line 488)
            string_types_565733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 66), 'string_types', False)
            # Processing the call keyword arguments (line 488)
            kwargs_565734 = {}
            # Getting the type of 'isinstance' (line 488)
            isinstance_565731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 44), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 488)
            isinstance_call_result_565735 = invoke(stypy.reporting.localization.Localization(__file__, 488, 44), isinstance_565731, *[bw_method_565732, string_types_565733], **kwargs_565734)
            
            # Applying the 'not' unary operator (line 488)
            result_not__565736 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 40), 'not', isinstance_call_result_565735)
            
            # Applying the binary operator 'and' (line 488)
            result_and_keyword_565737 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 13), 'and', isscalar_call_result_565730, result_not__565736)
            
            # Testing the type of an if condition (line 488)
            if_condition_565738 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 488, 13), result_and_keyword_565737)
            # Assigning a type to the variable 'if_condition_565738' (line 488)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 13), 'if_condition_565738', if_condition_565738)
            # SSA begins for if statement (line 488)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Attribute (line 489):
            
            # Assigning a Str to a Attribute (line 489):
            str_565739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 30), 'str', 'use constant')
            # Getting the type of 'self' (line 489)
            self_565740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'self')
            # Setting the type of the member '_bw_method' of a type (line 489)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 12), self_565740, '_bw_method', str_565739)
            
            # Assigning a Lambda to a Attribute (line 490):
            
            # Assigning a Lambda to a Attribute (line 490):

            @norecursion
            def _stypy_temp_lambda_487(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_487'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_487', 490, 37, True)
                # Passed parameters checking function
                _stypy_temp_lambda_487.stypy_localization = localization
                _stypy_temp_lambda_487.stypy_type_of_self = None
                _stypy_temp_lambda_487.stypy_type_store = module_type_store
                _stypy_temp_lambda_487.stypy_function_name = '_stypy_temp_lambda_487'
                _stypy_temp_lambda_487.stypy_param_names_list = []
                _stypy_temp_lambda_487.stypy_varargs_param_name = None
                _stypy_temp_lambda_487.stypy_kwargs_param_name = None
                _stypy_temp_lambda_487.stypy_call_defaults = defaults
                _stypy_temp_lambda_487.stypy_call_varargs = varargs
                _stypy_temp_lambda_487.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_487', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_487', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                # Getting the type of 'bw_method' (line 490)
                bw_method_565741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 45), 'bw_method')
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 490)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 37), 'stypy_return_type', bw_method_565741)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_487' in the type store
                # Getting the type of 'stypy_return_type' (line 490)
                stypy_return_type_565742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 37), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_565742)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_487'
                return stypy_return_type_565742

            # Assigning a type to the variable '_stypy_temp_lambda_487' (line 490)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 37), '_stypy_temp_lambda_487', _stypy_temp_lambda_487)
            # Getting the type of '_stypy_temp_lambda_487' (line 490)
            _stypy_temp_lambda_487_565743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 37), '_stypy_temp_lambda_487')
            # Getting the type of 'self' (line 490)
            self_565744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'self')
            # Setting the type of the member 'covariance_factor' of a type (line 490)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), self_565744, 'covariance_factor', _stypy_temp_lambda_487_565743)
            # SSA branch for the else part of an if statement (line 488)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to callable(...): (line 491)
            # Processing the call arguments (line 491)
            # Getting the type of 'bw_method' (line 491)
            bw_method_565746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 22), 'bw_method', False)
            # Processing the call keyword arguments (line 491)
            kwargs_565747 = {}
            # Getting the type of 'callable' (line 491)
            callable_565745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 13), 'callable', False)
            # Calling callable(args, kwargs) (line 491)
            callable_call_result_565748 = invoke(stypy.reporting.localization.Localization(__file__, 491, 13), callable_565745, *[bw_method_565746], **kwargs_565747)
            
            # Testing the type of an if condition (line 491)
            if_condition_565749 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 491, 13), callable_call_result_565748)
            # Assigning a type to the variable 'if_condition_565749' (line 491)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 13), 'if_condition_565749', if_condition_565749)
            # SSA begins for if statement (line 491)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 492):
            
            # Assigning a Name to a Attribute (line 492):
            # Getting the type of 'bw_method' (line 492)
            bw_method_565750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 30), 'bw_method')
            # Getting the type of 'self' (line 492)
            self_565751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'self')
            # Setting the type of the member '_bw_method' of a type (line 492)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 12), self_565751, '_bw_method', bw_method_565750)
            
            # Assigning a Lambda to a Attribute (line 493):
            
            # Assigning a Lambda to a Attribute (line 493):

            @norecursion
            def _stypy_temp_lambda_488(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_488'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_488', 493, 37, True)
                # Passed parameters checking function
                _stypy_temp_lambda_488.stypy_localization = localization
                _stypy_temp_lambda_488.stypy_type_of_self = None
                _stypy_temp_lambda_488.stypy_type_store = module_type_store
                _stypy_temp_lambda_488.stypy_function_name = '_stypy_temp_lambda_488'
                _stypy_temp_lambda_488.stypy_param_names_list = []
                _stypy_temp_lambda_488.stypy_varargs_param_name = None
                _stypy_temp_lambda_488.stypy_kwargs_param_name = None
                _stypy_temp_lambda_488.stypy_call_defaults = defaults
                _stypy_temp_lambda_488.stypy_call_varargs = varargs
                _stypy_temp_lambda_488.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_488', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_488', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to _bw_method(...): (line 493)
                # Processing the call arguments (line 493)
                # Getting the type of 'self' (line 493)
                self_565754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 61), 'self', False)
                # Processing the call keyword arguments (line 493)
                kwargs_565755 = {}
                # Getting the type of 'self' (line 493)
                self_565752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 45), 'self', False)
                # Obtaining the member '_bw_method' of a type (line 493)
                _bw_method_565753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 45), self_565752, '_bw_method')
                # Calling _bw_method(args, kwargs) (line 493)
                _bw_method_call_result_565756 = invoke(stypy.reporting.localization.Localization(__file__, 493, 45), _bw_method_565753, *[self_565754], **kwargs_565755)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 493)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 37), 'stypy_return_type', _bw_method_call_result_565756)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_488' in the type store
                # Getting the type of 'stypy_return_type' (line 493)
                stypy_return_type_565757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 37), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_565757)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_488'
                return stypy_return_type_565757

            # Assigning a type to the variable '_stypy_temp_lambda_488' (line 493)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 37), '_stypy_temp_lambda_488', _stypy_temp_lambda_488)
            # Getting the type of '_stypy_temp_lambda_488' (line 493)
            _stypy_temp_lambda_488_565758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 37), '_stypy_temp_lambda_488')
            # Getting the type of 'self' (line 493)
            self_565759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'self')
            # Setting the type of the member 'covariance_factor' of a type (line 493)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 12), self_565759, 'covariance_factor', _stypy_temp_lambda_488_565758)
            # SSA branch for the else part of an if statement (line 491)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 495):
            
            # Assigning a Str to a Name (line 495):
            str_565760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 18), 'str', "`bw_method` should be 'scott', 'silverman', a scalar or a callable.")
            # Assigning a type to the variable 'msg' (line 495)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'msg', str_565760)
            
            # Call to ValueError(...): (line 497)
            # Processing the call arguments (line 497)
            # Getting the type of 'msg' (line 497)
            msg_565762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 29), 'msg', False)
            # Processing the call keyword arguments (line 497)
            kwargs_565763 = {}
            # Getting the type of 'ValueError' (line 497)
            ValueError_565761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 497)
            ValueError_call_result_565764 = invoke(stypy.reporting.localization.Localization(__file__, 497, 18), ValueError_565761, *[msg_565762], **kwargs_565763)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 497, 12), ValueError_call_result_565764, 'raise parameter', BaseException)
            # SSA join for if statement (line 491)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 488)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 486)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 484)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_565710 and more_types_in_union_565711):
                # SSA join for if statement (line 482)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _compute_covariance(...): (line 499)
        # Processing the call keyword arguments (line 499)
        kwargs_565767 = {}
        # Getting the type of 'self' (line 499)
        self_565765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'self', False)
        # Obtaining the member '_compute_covariance' of a type (line 499)
        _compute_covariance_565766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), self_565765, '_compute_covariance')
        # Calling _compute_covariance(args, kwargs) (line 499)
        _compute_covariance_call_result_565768 = invoke(stypy.reporting.localization.Localization(__file__, 499, 8), _compute_covariance_565766, *[], **kwargs_565767)
        
        
        # ################# End of 'set_bandwidth(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_bandwidth' in the type store
        # Getting the type of 'stypy_return_type' (line 439)
        stypy_return_type_565769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565769)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_bandwidth'
        return stypy_return_type_565769


    @norecursion
    def _compute_covariance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compute_covariance'
        module_type_store = module_type_store.open_function_context('_compute_covariance', 501, 4, False)
        # Assigning a type to the variable 'self' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde._compute_covariance.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde._compute_covariance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde._compute_covariance.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde._compute_covariance.__dict__.__setitem__('stypy_function_name', 'gaussian_kde._compute_covariance')
        gaussian_kde._compute_covariance.__dict__.__setitem__('stypy_param_names_list', [])
        gaussian_kde._compute_covariance.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde._compute_covariance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde._compute_covariance.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde._compute_covariance.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde._compute_covariance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde._compute_covariance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde._compute_covariance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_compute_covariance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_compute_covariance(...)' code ##################

        str_565770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, (-1)), 'str', 'Computes the covariance matrix for each Gaussian kernel using\n        covariance_factor().\n        ')
        
        # Assigning a Call to a Attribute (line 505):
        
        # Assigning a Call to a Attribute (line 505):
        
        # Call to covariance_factor(...): (line 505)
        # Processing the call keyword arguments (line 505)
        kwargs_565773 = {}
        # Getting the type of 'self' (line 505)
        self_565771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 22), 'self', False)
        # Obtaining the member 'covariance_factor' of a type (line 505)
        covariance_factor_565772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 22), self_565771, 'covariance_factor')
        # Calling covariance_factor(args, kwargs) (line 505)
        covariance_factor_call_result_565774 = invoke(stypy.reporting.localization.Localization(__file__, 505, 22), covariance_factor_565772, *[], **kwargs_565773)
        
        # Getting the type of 'self' (line 505)
        self_565775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'self')
        # Setting the type of the member 'factor' of a type (line 505)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 8), self_565775, 'factor', covariance_factor_call_result_565774)
        
        # Type idiom detected: calculating its left and rigth part (line 507)
        str_565776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 29), 'str', '_data_inv_cov')
        # Getting the type of 'self' (line 507)
        self_565777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 23), 'self')
        
        (may_be_565778, more_types_in_union_565779) = may_not_provide_member(str_565776, self_565777)

        if may_be_565778:

            if more_types_in_union_565779:
                # Runtime conditional SSA (line 507)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'self' (line 507)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'self', remove_member_provider_from_union(self_565777, '_data_inv_cov'))
            
            # Assigning a Call to a Attribute (line 508):
            
            # Assigning a Call to a Attribute (line 508):
            
            # Call to atleast_2d(...): (line 508)
            # Processing the call arguments (line 508)
            
            # Call to cov(...): (line 508)
            # Processing the call arguments (line 508)
            # Getting the type of 'self' (line 508)
            self_565783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 54), 'self', False)
            # Obtaining the member 'dataset' of a type (line 508)
            dataset_565784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 54), self_565783, 'dataset')
            # Processing the call keyword arguments (line 508)
            int_565785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 75), 'int')
            keyword_565786 = int_565785
            # Getting the type of 'False' (line 509)
            False_565787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 52), 'False', False)
            keyword_565788 = False_565787
            kwargs_565789 = {'rowvar': keyword_565786, 'bias': keyword_565788}
            # Getting the type of 'np' (line 508)
            np_565781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 47), 'np', False)
            # Obtaining the member 'cov' of a type (line 508)
            cov_565782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 47), np_565781, 'cov')
            # Calling cov(args, kwargs) (line 508)
            cov_call_result_565790 = invoke(stypy.reporting.localization.Localization(__file__, 508, 47), cov_565782, *[dataset_565784], **kwargs_565789)
            
            # Processing the call keyword arguments (line 508)
            kwargs_565791 = {}
            # Getting the type of 'atleast_2d' (line 508)
            atleast_2d_565780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 36), 'atleast_2d', False)
            # Calling atleast_2d(args, kwargs) (line 508)
            atleast_2d_call_result_565792 = invoke(stypy.reporting.localization.Localization(__file__, 508, 36), atleast_2d_565780, *[cov_call_result_565790], **kwargs_565791)
            
            # Getting the type of 'self' (line 508)
            self_565793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'self')
            # Setting the type of the member '_data_covariance' of a type (line 508)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 12), self_565793, '_data_covariance', atleast_2d_call_result_565792)
            
            # Assigning a Call to a Attribute (line 510):
            
            # Assigning a Call to a Attribute (line 510):
            
            # Call to inv(...): (line 510)
            # Processing the call arguments (line 510)
            # Getting the type of 'self' (line 510)
            self_565796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 44), 'self', False)
            # Obtaining the member '_data_covariance' of a type (line 510)
            _data_covariance_565797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 44), self_565796, '_data_covariance')
            # Processing the call keyword arguments (line 510)
            kwargs_565798 = {}
            # Getting the type of 'linalg' (line 510)
            linalg_565794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 33), 'linalg', False)
            # Obtaining the member 'inv' of a type (line 510)
            inv_565795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 33), linalg_565794, 'inv')
            # Calling inv(args, kwargs) (line 510)
            inv_call_result_565799 = invoke(stypy.reporting.localization.Localization(__file__, 510, 33), inv_565795, *[_data_covariance_565797], **kwargs_565798)
            
            # Getting the type of 'self' (line 510)
            self_565800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'self')
            # Setting the type of the member '_data_inv_cov' of a type (line 510)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 12), self_565800, '_data_inv_cov', inv_call_result_565799)

            if more_types_in_union_565779:
                # SSA join for if statement (line 507)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Attribute (line 512):
        
        # Assigning a BinOp to a Attribute (line 512):
        # Getting the type of 'self' (line 512)
        self_565801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 26), 'self')
        # Obtaining the member '_data_covariance' of a type (line 512)
        _data_covariance_565802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 26), self_565801, '_data_covariance')
        # Getting the type of 'self' (line 512)
        self_565803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 50), 'self')
        # Obtaining the member 'factor' of a type (line 512)
        factor_565804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 50), self_565803, 'factor')
        int_565805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 63), 'int')
        # Applying the binary operator '**' (line 512)
        result_pow_565806 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 50), '**', factor_565804, int_565805)
        
        # Applying the binary operator '*' (line 512)
        result_mul_565807 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 26), '*', _data_covariance_565802, result_pow_565806)
        
        # Getting the type of 'self' (line 512)
        self_565808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'self')
        # Setting the type of the member 'covariance' of a type (line 512)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), self_565808, 'covariance', result_mul_565807)
        
        # Assigning a BinOp to a Attribute (line 513):
        
        # Assigning a BinOp to a Attribute (line 513):
        # Getting the type of 'self' (line 513)
        self_565809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 23), 'self')
        # Obtaining the member '_data_inv_cov' of a type (line 513)
        _data_inv_cov_565810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 23), self_565809, '_data_inv_cov')
        # Getting the type of 'self' (line 513)
        self_565811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 44), 'self')
        # Obtaining the member 'factor' of a type (line 513)
        factor_565812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 44), self_565811, 'factor')
        int_565813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 57), 'int')
        # Applying the binary operator '**' (line 513)
        result_pow_565814 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 44), '**', factor_565812, int_565813)
        
        # Applying the binary operator 'div' (line 513)
        result_div_565815 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 23), 'div', _data_inv_cov_565810, result_pow_565814)
        
        # Getting the type of 'self' (line 513)
        self_565816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'self')
        # Setting the type of the member 'inv_cov' of a type (line 513)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 8), self_565816, 'inv_cov', result_div_565815)
        
        # Assigning a BinOp to a Attribute (line 514):
        
        # Assigning a BinOp to a Attribute (line 514):
        
        # Call to sqrt(...): (line 514)
        # Processing the call arguments (line 514)
        
        # Call to det(...): (line 514)
        # Processing the call arguments (line 514)
        int_565820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 44), 'int')
        # Getting the type of 'pi' (line 514)
        pi_565821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 46), 'pi', False)
        # Applying the binary operator '*' (line 514)
        result_mul_565822 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 44), '*', int_565820, pi_565821)
        
        # Getting the type of 'self' (line 514)
        self_565823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 49), 'self', False)
        # Obtaining the member 'covariance' of a type (line 514)
        covariance_565824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 49), self_565823, 'covariance')
        # Applying the binary operator '*' (line 514)
        result_mul_565825 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 48), '*', result_mul_565822, covariance_565824)
        
        # Processing the call keyword arguments (line 514)
        kwargs_565826 = {}
        # Getting the type of 'linalg' (line 514)
        linalg_565818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 33), 'linalg', False)
        # Obtaining the member 'det' of a type (line 514)
        det_565819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 33), linalg_565818, 'det')
        # Calling det(args, kwargs) (line 514)
        det_call_result_565827 = invoke(stypy.reporting.localization.Localization(__file__, 514, 33), det_565819, *[result_mul_565825], **kwargs_565826)
        
        # Processing the call keyword arguments (line 514)
        kwargs_565828 = {}
        # Getting the type of 'sqrt' (line 514)
        sqrt_565817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 28), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 514)
        sqrt_call_result_565829 = invoke(stypy.reporting.localization.Localization(__file__, 514, 28), sqrt_565817, *[det_call_result_565827], **kwargs_565828)
        
        # Getting the type of 'self' (line 514)
        self_565830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 69), 'self')
        # Obtaining the member 'n' of a type (line 514)
        n_565831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 69), self_565830, 'n')
        # Applying the binary operator '*' (line 514)
        result_mul_565832 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 28), '*', sqrt_call_result_565829, n_565831)
        
        # Getting the type of 'self' (line 514)
        self_565833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'self')
        # Setting the type of the member '_norm_factor' of a type (line 514)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 8), self_565833, '_norm_factor', result_mul_565832)
        
        # ################# End of '_compute_covariance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compute_covariance' in the type store
        # Getting the type of 'stypy_return_type' (line 501)
        stypy_return_type_565834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565834)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compute_covariance'
        return stypy_return_type_565834


    @norecursion
    def pdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pdf'
        module_type_store = module_type_store.open_function_context('pdf', 516, 4, False)
        # Assigning a type to the variable 'self' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde.pdf.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde.pdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde.pdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde.pdf.__dict__.__setitem__('stypy_function_name', 'gaussian_kde.pdf')
        gaussian_kde.pdf.__dict__.__setitem__('stypy_param_names_list', ['x'])
        gaussian_kde.pdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde.pdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde.pdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde.pdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde.pdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde.pdf.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.pdf', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pdf', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pdf(...)' code ##################

        str_565835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, (-1)), 'str', '\n        Evaluate the estimated pdf on a provided set of points.\n\n        Notes\n        -----\n        This is an alias for `gaussian_kde.evaluate`.  See the ``evaluate``\n        docstring for more details.\n\n        ')
        
        # Call to evaluate(...): (line 526)
        # Processing the call arguments (line 526)
        # Getting the type of 'x' (line 526)
        x_565838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 29), 'x', False)
        # Processing the call keyword arguments (line 526)
        kwargs_565839 = {}
        # Getting the type of 'self' (line 526)
        self_565836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 15), 'self', False)
        # Obtaining the member 'evaluate' of a type (line 526)
        evaluate_565837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 15), self_565836, 'evaluate')
        # Calling evaluate(args, kwargs) (line 526)
        evaluate_call_result_565840 = invoke(stypy.reporting.localization.Localization(__file__, 526, 15), evaluate_565837, *[x_565838], **kwargs_565839)
        
        # Assigning a type to the variable 'stypy_return_type' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'stypy_return_type', evaluate_call_result_565840)
        
        # ################# End of 'pdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pdf' in the type store
        # Getting the type of 'stypy_return_type' (line 516)
        stypy_return_type_565841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565841)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pdf'
        return stypy_return_type_565841


    @norecursion
    def logpdf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'logpdf'
        module_type_store = module_type_store.open_function_context('logpdf', 528, 4, False)
        # Assigning a type to the variable 'self' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        gaussian_kde.logpdf.__dict__.__setitem__('stypy_localization', localization)
        gaussian_kde.logpdf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        gaussian_kde.logpdf.__dict__.__setitem__('stypy_type_store', module_type_store)
        gaussian_kde.logpdf.__dict__.__setitem__('stypy_function_name', 'gaussian_kde.logpdf')
        gaussian_kde.logpdf.__dict__.__setitem__('stypy_param_names_list', ['x'])
        gaussian_kde.logpdf.__dict__.__setitem__('stypy_varargs_param_name', None)
        gaussian_kde.logpdf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        gaussian_kde.logpdf.__dict__.__setitem__('stypy_call_defaults', defaults)
        gaussian_kde.logpdf.__dict__.__setitem__('stypy_call_varargs', varargs)
        gaussian_kde.logpdf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        gaussian_kde.logpdf.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'gaussian_kde.logpdf', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'logpdf', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'logpdf(...)' code ##################

        str_565842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, (-1)), 'str', '\n        Evaluate the log of the estimated pdf on a provided set of points.\n        ')
        
        # Assigning a Call to a Name (line 533):
        
        # Assigning a Call to a Name (line 533):
        
        # Call to atleast_2d(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'x' (line 533)
        x_565844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 28), 'x', False)
        # Processing the call keyword arguments (line 533)
        kwargs_565845 = {}
        # Getting the type of 'atleast_2d' (line 533)
        atleast_2d_565843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 17), 'atleast_2d', False)
        # Calling atleast_2d(args, kwargs) (line 533)
        atleast_2d_call_result_565846 = invoke(stypy.reporting.localization.Localization(__file__, 533, 17), atleast_2d_565843, *[x_565844], **kwargs_565845)
        
        # Assigning a type to the variable 'points' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'points', atleast_2d_call_result_565846)
        
        # Assigning a Attribute to a Tuple (line 535):
        
        # Assigning a Subscript to a Name (line 535):
        
        # Obtaining the type of the subscript
        int_565847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 8), 'int')
        # Getting the type of 'points' (line 535)
        points_565848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 15), 'points')
        # Obtaining the member 'shape' of a type (line 535)
        shape_565849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 15), points_565848, 'shape')
        # Obtaining the member '__getitem__' of a type (line 535)
        getitem___565850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), shape_565849, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 535)
        subscript_call_result_565851 = invoke(stypy.reporting.localization.Localization(__file__, 535, 8), getitem___565850, int_565847)
        
        # Assigning a type to the variable 'tuple_var_assignment_565080' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'tuple_var_assignment_565080', subscript_call_result_565851)
        
        # Assigning a Subscript to a Name (line 535):
        
        # Obtaining the type of the subscript
        int_565852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 8), 'int')
        # Getting the type of 'points' (line 535)
        points_565853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 15), 'points')
        # Obtaining the member 'shape' of a type (line 535)
        shape_565854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 15), points_565853, 'shape')
        # Obtaining the member '__getitem__' of a type (line 535)
        getitem___565855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), shape_565854, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 535)
        subscript_call_result_565856 = invoke(stypy.reporting.localization.Localization(__file__, 535, 8), getitem___565855, int_565852)
        
        # Assigning a type to the variable 'tuple_var_assignment_565081' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'tuple_var_assignment_565081', subscript_call_result_565856)
        
        # Assigning a Name to a Name (line 535):
        # Getting the type of 'tuple_var_assignment_565080' (line 535)
        tuple_var_assignment_565080_565857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'tuple_var_assignment_565080')
        # Assigning a type to the variable 'd' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'd', tuple_var_assignment_565080_565857)
        
        # Assigning a Name to a Name (line 535):
        # Getting the type of 'tuple_var_assignment_565081' (line 535)
        tuple_var_assignment_565081_565858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'tuple_var_assignment_565081')
        # Assigning a type to the variable 'm' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 11), 'm', tuple_var_assignment_565081_565858)
        
        
        # Getting the type of 'd' (line 536)
        d_565859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 11), 'd')
        # Getting the type of 'self' (line 536)
        self_565860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'self')
        # Obtaining the member 'd' of a type (line 536)
        d_565861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 16), self_565860, 'd')
        # Applying the binary operator '!=' (line 536)
        result_ne_565862 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 11), '!=', d_565859, d_565861)
        
        # Testing the type of an if condition (line 536)
        if_condition_565863 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 8), result_ne_565862)
        # Assigning a type to the variable 'if_condition_565863' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'if_condition_565863', if_condition_565863)
        # SSA begins for if statement (line 536)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'd' (line 537)
        d_565864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 15), 'd')
        int_565865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 20), 'int')
        # Applying the binary operator '==' (line 537)
        result_eq_565866 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 15), '==', d_565864, int_565865)
        
        
        # Getting the type of 'm' (line 537)
        m_565867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 26), 'm')
        # Getting the type of 'self' (line 537)
        self_565868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 31), 'self')
        # Obtaining the member 'd' of a type (line 537)
        d_565869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 31), self_565868, 'd')
        # Applying the binary operator '==' (line 537)
        result_eq_565870 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 26), '==', m_565867, d_565869)
        
        # Applying the binary operator 'and' (line 537)
        result_and_keyword_565871 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 15), 'and', result_eq_565866, result_eq_565870)
        
        # Testing the type of an if condition (line 537)
        if_condition_565872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 537, 12), result_and_keyword_565871)
        # Assigning a type to the variable 'if_condition_565872' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'if_condition_565872', if_condition_565872)
        # SSA begins for if statement (line 537)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 539):
        
        # Assigning a Call to a Name (line 539):
        
        # Call to reshape(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'points' (line 539)
        points_565874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 33), 'points', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 539)
        tuple_565875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 539)
        # Adding element type (line 539)
        # Getting the type of 'self' (line 539)
        self_565876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 42), 'self', False)
        # Obtaining the member 'd' of a type (line 539)
        d_565877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 42), self_565876, 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 42), tuple_565875, d_565877)
        # Adding element type (line 539)
        int_565878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 42), tuple_565875, int_565878)
        
        # Processing the call keyword arguments (line 539)
        kwargs_565879 = {}
        # Getting the type of 'reshape' (line 539)
        reshape_565873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 25), 'reshape', False)
        # Calling reshape(args, kwargs) (line 539)
        reshape_call_result_565880 = invoke(stypy.reporting.localization.Localization(__file__, 539, 25), reshape_565873, *[points_565874, tuple_565875], **kwargs_565879)
        
        # Assigning a type to the variable 'points' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 16), 'points', reshape_call_result_565880)
        
        # Assigning a Num to a Name (line 540):
        
        # Assigning a Num to a Name (line 540):
        int_565881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 20), 'int')
        # Assigning a type to the variable 'm' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 16), 'm', int_565881)
        # SSA branch for the else part of an if statement (line 537)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 542):
        
        # Assigning a BinOp to a Name (line 542):
        str_565882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 22), 'str', 'points have dimension %s, dataset has dimension %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 542)
        tuple_565883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 78), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 542)
        # Adding element type (line 542)
        # Getting the type of 'd' (line 542)
        d_565884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 78), 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 78), tuple_565883, d_565884)
        # Adding element type (line 542)
        # Getting the type of 'self' (line 543)
        self_565885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 20), 'self')
        # Obtaining the member 'd' of a type (line 543)
        d_565886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 20), self_565885, 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 78), tuple_565883, d_565886)
        
        # Applying the binary operator '%' (line 542)
        result_mod_565887 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 22), '%', str_565882, tuple_565883)
        
        # Assigning a type to the variable 'msg' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'msg', result_mod_565887)
        
        # Call to ValueError(...): (line 544)
        # Processing the call arguments (line 544)
        # Getting the type of 'msg' (line 544)
        msg_565889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 33), 'msg', False)
        # Processing the call keyword arguments (line 544)
        kwargs_565890 = {}
        # Getting the type of 'ValueError' (line 544)
        ValueError_565888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 544)
        ValueError_call_result_565891 = invoke(stypy.reporting.localization.Localization(__file__, 544, 22), ValueError_565888, *[msg_565889], **kwargs_565890)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 544, 16), ValueError_call_result_565891, 'raise parameter', BaseException)
        # SSA join for if statement (line 537)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 536)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 546):
        
        # Assigning a Call to a Name (line 546):
        
        # Call to zeros(...): (line 546)
        # Processing the call arguments (line 546)
        
        # Obtaining an instance of the builtin type 'tuple' (line 546)
        tuple_565893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 546)
        # Adding element type (line 546)
        # Getting the type of 'm' (line 546)
        m_565894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 24), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 24), tuple_565893, m_565894)
        
        # Processing the call keyword arguments (line 546)
        # Getting the type of 'float' (line 546)
        float_565895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 35), 'float', False)
        keyword_565896 = float_565895
        kwargs_565897 = {'dtype': keyword_565896}
        # Getting the type of 'zeros' (line 546)
        zeros_565892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 17), 'zeros', False)
        # Calling zeros(args, kwargs) (line 546)
        zeros_call_result_565898 = invoke(stypy.reporting.localization.Localization(__file__, 546, 17), zeros_565892, *[tuple_565893], **kwargs_565897)
        
        # Assigning a type to the variable 'result' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'result', zeros_call_result_565898)
        
        
        # Getting the type of 'm' (line 548)
        m_565899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 11), 'm')
        # Getting the type of 'self' (line 548)
        self_565900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 16), 'self')
        # Obtaining the member 'n' of a type (line 548)
        n_565901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 16), self_565900, 'n')
        # Applying the binary operator '>=' (line 548)
        result_ge_565902 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 11), '>=', m_565899, n_565901)
        
        # Testing the type of an if condition (line 548)
        if_condition_565903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 8), result_ge_565902)
        # Assigning a type to the variable 'if_condition_565903' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'if_condition_565903', if_condition_565903)
        # SSA begins for if statement (line 548)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 550):
        
        # Assigning a Call to a Name (line 550):
        
        # Call to zeros(...): (line 550)
        # Processing the call arguments (line 550)
        
        # Obtaining an instance of the builtin type 'tuple' (line 550)
        tuple_565905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 550)
        # Adding element type (line 550)
        # Getting the type of 'self' (line 550)
        self_565906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 28), 'self', False)
        # Obtaining the member 'n' of a type (line 550)
        n_565907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 28), self_565906, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 28), tuple_565905, n_565907)
        # Adding element type (line 550)
        # Getting the type of 'm' (line 550)
        m_565908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 36), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 28), tuple_565905, m_565908)
        
        # Processing the call keyword arguments (line 550)
        # Getting the type of 'float' (line 550)
        float_565909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 46), 'float', False)
        keyword_565910 = float_565909
        kwargs_565911 = {'dtype': keyword_565910}
        # Getting the type of 'zeros' (line 550)
        zeros_565904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 21), 'zeros', False)
        # Calling zeros(args, kwargs) (line 550)
        zeros_call_result_565912 = invoke(stypy.reporting.localization.Localization(__file__, 550, 21), zeros_565904, *[tuple_565905], **kwargs_565911)
        
        # Assigning a type to the variable 'energy' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'energy', zeros_call_result_565912)
        
        
        # Call to range(...): (line 551)
        # Processing the call arguments (line 551)
        # Getting the type of 'self' (line 551)
        self_565914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 27), 'self', False)
        # Obtaining the member 'n' of a type (line 551)
        n_565915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 27), self_565914, 'n')
        # Processing the call keyword arguments (line 551)
        kwargs_565916 = {}
        # Getting the type of 'range' (line 551)
        range_565913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 21), 'range', False)
        # Calling range(args, kwargs) (line 551)
        range_call_result_565917 = invoke(stypy.reporting.localization.Localization(__file__, 551, 21), range_565913, *[n_565915], **kwargs_565916)
        
        # Testing the type of a for loop iterable (line 551)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 551, 12), range_call_result_565917)
        # Getting the type of the for loop variable (line 551)
        for_loop_var_565918 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 551, 12), range_call_result_565917)
        # Assigning a type to the variable 'i' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'i', for_loop_var_565918)
        # SSA begins for a for statement (line 551)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 552):
        
        # Assigning a BinOp to a Name (line 552):
        
        # Obtaining the type of the subscript
        slice_565919 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 552, 23), None, None, None)
        # Getting the type of 'i' (line 552)
        i_565920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 39), 'i')
        # Getting the type of 'newaxis' (line 552)
        newaxis_565921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 42), 'newaxis')
        # Getting the type of 'self' (line 552)
        self_565922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 23), 'self')
        # Obtaining the member 'dataset' of a type (line 552)
        dataset_565923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 23), self_565922, 'dataset')
        # Obtaining the member '__getitem__' of a type (line 552)
        getitem___565924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 23), dataset_565923, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 552)
        subscript_call_result_565925 = invoke(stypy.reporting.localization.Localization(__file__, 552, 23), getitem___565924, (slice_565919, i_565920, newaxis_565921))
        
        # Getting the type of 'points' (line 552)
        points_565926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 53), 'points')
        # Applying the binary operator '-' (line 552)
        result_sub_565927 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 23), '-', subscript_call_result_565925, points_565926)
        
        # Assigning a type to the variable 'diff' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'diff', result_sub_565927)
        
        # Assigning a Call to a Name (line 553):
        
        # Assigning a Call to a Name (line 553):
        
        # Call to dot(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'self' (line 553)
        self_565929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 28), 'self', False)
        # Obtaining the member 'inv_cov' of a type (line 553)
        inv_cov_565930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 28), self_565929, 'inv_cov')
        # Getting the type of 'diff' (line 553)
        diff_565931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 42), 'diff', False)
        # Processing the call keyword arguments (line 553)
        kwargs_565932 = {}
        # Getting the type of 'dot' (line 553)
        dot_565928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 24), 'dot', False)
        # Calling dot(args, kwargs) (line 553)
        dot_call_result_565933 = invoke(stypy.reporting.localization.Localization(__file__, 553, 24), dot_565928, *[inv_cov_565930, diff_565931], **kwargs_565932)
        
        # Assigning a type to the variable 'tdiff' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 16), 'tdiff', dot_call_result_565933)
        
        # Assigning a BinOp to a Subscript (line 554):
        
        # Assigning a BinOp to a Subscript (line 554):
        
        # Call to sum(...): (line 554)
        # Processing the call arguments (line 554)
        # Getting the type of 'diff' (line 554)
        diff_565935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 32), 'diff', False)
        # Getting the type of 'tdiff' (line 554)
        tdiff_565936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 37), 'tdiff', False)
        # Applying the binary operator '*' (line 554)
        result_mul_565937 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 32), '*', diff_565935, tdiff_565936)
        
        # Processing the call keyword arguments (line 554)
        int_565938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 48), 'int')
        keyword_565939 = int_565938
        kwargs_565940 = {'axis': keyword_565939}
        # Getting the type of 'sum' (line 554)
        sum_565934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 28), 'sum', False)
        # Calling sum(args, kwargs) (line 554)
        sum_call_result_565941 = invoke(stypy.reporting.localization.Localization(__file__, 554, 28), sum_565934, *[result_mul_565937], **kwargs_565940)
        
        float_565942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 53), 'float')
        # Applying the binary operator 'div' (line 554)
        result_div_565943 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 28), 'div', sum_call_result_565941, float_565942)
        
        # Getting the type of 'energy' (line 554)
        energy_565944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'energy')
        # Getting the type of 'i' (line 554)
        i_565945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 23), 'i')
        # Storing an element on a container (line 554)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 16), energy_565944, (i_565945, result_div_565943))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 555):
        
        # Assigning a Call to a Name (line 555):
        
        # Call to logsumexp(...): (line 555)
        # Processing the call arguments (line 555)
        
        # Getting the type of 'energy' (line 555)
        energy_565947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 32), 'energy', False)
        # Applying the 'usub' unary operator (line 555)
        result___neg___565948 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 31), 'usub', energy_565947)
        
        # Processing the call keyword arguments (line 555)
        int_565949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 42), 'int')
        # Getting the type of 'self' (line 555)
        self_565950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 44), 'self', False)
        # Obtaining the member '_norm_factor' of a type (line 555)
        _norm_factor_565951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 44), self_565950, '_norm_factor')
        # Applying the binary operator 'div' (line 555)
        result_div_565952 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 42), 'div', int_565949, _norm_factor_565951)
        
        keyword_565953 = result_div_565952
        int_565954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 68), 'int')
        keyword_565955 = int_565954
        kwargs_565956 = {'b': keyword_565953, 'axis': keyword_565955}
        # Getting the type of 'logsumexp' (line 555)
        logsumexp_565946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 21), 'logsumexp', False)
        # Calling logsumexp(args, kwargs) (line 555)
        logsumexp_call_result_565957 = invoke(stypy.reporting.localization.Localization(__file__, 555, 21), logsumexp_565946, *[result___neg___565948], **kwargs_565956)
        
        # Assigning a type to the variable 'result' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'result', logsumexp_call_result_565957)
        # SSA branch for the else part of an if statement (line 548)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to range(...): (line 558)
        # Processing the call arguments (line 558)
        # Getting the type of 'm' (line 558)
        m_565959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 27), 'm', False)
        # Processing the call keyword arguments (line 558)
        kwargs_565960 = {}
        # Getting the type of 'range' (line 558)
        range_565958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 21), 'range', False)
        # Calling range(args, kwargs) (line 558)
        range_call_result_565961 = invoke(stypy.reporting.localization.Localization(__file__, 558, 21), range_565958, *[m_565959], **kwargs_565960)
        
        # Testing the type of a for loop iterable (line 558)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 558, 12), range_call_result_565961)
        # Getting the type of the for loop variable (line 558)
        for_loop_var_565962 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 558, 12), range_call_result_565961)
        # Assigning a type to the variable 'i' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'i', for_loop_var_565962)
        # SSA begins for a for statement (line 558)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 559):
        
        # Assigning a BinOp to a Name (line 559):
        # Getting the type of 'self' (line 559)
        self_565963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 23), 'self')
        # Obtaining the member 'dataset' of a type (line 559)
        dataset_565964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 23), self_565963, 'dataset')
        
        # Obtaining the type of the subscript
        slice_565965 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 559, 38), None, None, None)
        # Getting the type of 'i' (line 559)
        i_565966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 48), 'i')
        # Getting the type of 'newaxis' (line 559)
        newaxis_565967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 51), 'newaxis')
        # Getting the type of 'points' (line 559)
        points_565968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 38), 'points')
        # Obtaining the member '__getitem__' of a type (line 559)
        getitem___565969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 38), points_565968, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 559)
        subscript_call_result_565970 = invoke(stypy.reporting.localization.Localization(__file__, 559, 38), getitem___565969, (slice_565965, i_565966, newaxis_565967))
        
        # Applying the binary operator '-' (line 559)
        result_sub_565971 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 23), '-', dataset_565964, subscript_call_result_565970)
        
        # Assigning a type to the variable 'diff' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 16), 'diff', result_sub_565971)
        
        # Assigning a Call to a Name (line 560):
        
        # Assigning a Call to a Name (line 560):
        
        # Call to dot(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'self' (line 560)
        self_565973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 28), 'self', False)
        # Obtaining the member 'inv_cov' of a type (line 560)
        inv_cov_565974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 28), self_565973, 'inv_cov')
        # Getting the type of 'diff' (line 560)
        diff_565975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 42), 'diff', False)
        # Processing the call keyword arguments (line 560)
        kwargs_565976 = {}
        # Getting the type of 'dot' (line 560)
        dot_565972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 24), 'dot', False)
        # Calling dot(args, kwargs) (line 560)
        dot_call_result_565977 = invoke(stypy.reporting.localization.Localization(__file__, 560, 24), dot_565972, *[inv_cov_565974, diff_565975], **kwargs_565976)
        
        # Assigning a type to the variable 'tdiff' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'tdiff', dot_call_result_565977)
        
        # Assigning a BinOp to a Name (line 561):
        
        # Assigning a BinOp to a Name (line 561):
        
        # Call to sum(...): (line 561)
        # Processing the call arguments (line 561)
        # Getting the type of 'diff' (line 561)
        diff_565979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 29), 'diff', False)
        # Getting the type of 'tdiff' (line 561)
        tdiff_565980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 36), 'tdiff', False)
        # Applying the binary operator '*' (line 561)
        result_mul_565981 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 29), '*', diff_565979, tdiff_565980)
        
        # Processing the call keyword arguments (line 561)
        int_565982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 48), 'int')
        keyword_565983 = int_565982
        kwargs_565984 = {'axis': keyword_565983}
        # Getting the type of 'sum' (line 561)
        sum_565978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 25), 'sum', False)
        # Calling sum(args, kwargs) (line 561)
        sum_call_result_565985 = invoke(stypy.reporting.localization.Localization(__file__, 561, 25), sum_565978, *[result_mul_565981], **kwargs_565984)
        
        float_565986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 53), 'float')
        # Applying the binary operator 'div' (line 561)
        result_div_565987 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 25), 'div', sum_call_result_565985, float_565986)
        
        # Assigning a type to the variable 'energy' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'energy', result_div_565987)
        
        # Assigning a Call to a Subscript (line 562):
        
        # Assigning a Call to a Subscript (line 562):
        
        # Call to logsumexp(...): (line 562)
        # Processing the call arguments (line 562)
        
        # Getting the type of 'energy' (line 562)
        energy_565989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 39), 'energy', False)
        # Applying the 'usub' unary operator (line 562)
        result___neg___565990 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 38), 'usub', energy_565989)
        
        # Processing the call keyword arguments (line 562)
        int_565991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 49), 'int')
        # Getting the type of 'self' (line 562)
        self_565992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 51), 'self', False)
        # Obtaining the member '_norm_factor' of a type (line 562)
        _norm_factor_565993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 51), self_565992, '_norm_factor')
        # Applying the binary operator 'div' (line 562)
        result_div_565994 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 49), 'div', int_565991, _norm_factor_565993)
        
        keyword_565995 = result_div_565994
        kwargs_565996 = {'b': keyword_565995}
        # Getting the type of 'logsumexp' (line 562)
        logsumexp_565988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 28), 'logsumexp', False)
        # Calling logsumexp(args, kwargs) (line 562)
        logsumexp_call_result_565997 = invoke(stypy.reporting.localization.Localization(__file__, 562, 28), logsumexp_565988, *[result___neg___565990], **kwargs_565996)
        
        # Getting the type of 'result' (line 562)
        result_565998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 16), 'result')
        # Getting the type of 'i' (line 562)
        i_565999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 23), 'i')
        # Storing an element on a container (line 562)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 16), result_565998, (i_565999, logsumexp_call_result_565997))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 548)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 564)
        result_566000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'stypy_return_type', result_566000)
        
        # ################# End of 'logpdf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'logpdf' in the type store
        # Getting the type of 'stypy_return_type' (line 528)
        stypy_return_type_566001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_566001)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'logpdf'
        return stypy_return_type_566001


# Assigning a type to the variable 'gaussian_kde' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'gaussian_kde', gaussian_kde)

# Assigning a Name to a Name (line 228):
# Getting the type of 'gaussian_kde'
gaussian_kde_566002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'gaussian_kde')
# Obtaining the member 'evaluate' of a type
evaluate_566003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), gaussian_kde_566002, 'evaluate')
# Getting the type of 'gaussian_kde'
gaussian_kde_566004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'gaussian_kde')
# Setting the type of the member '__call__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), gaussian_kde_566004, '__call__', evaluate_566003)

# Assigning a Name to a Name (line 432):
# Getting the type of 'gaussian_kde'
gaussian_kde_566005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'gaussian_kde')
# Obtaining the member 'scotts_factor' of a type
scotts_factor_566006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), gaussian_kde_566005, 'scotts_factor')
# Getting the type of 'gaussian_kde'
gaussian_kde_566007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'gaussian_kde')
# Setting the type of the member 'covariance_factor' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), gaussian_kde_566007, 'covariance_factor', scotts_factor_566006)

# Assigning a Str to a Attribute (line 433):
str_566008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, (-1)), 'str', 'Computes the coefficient (`kde.factor`) that\n        multiplies the data covariance matrix to obtain the kernel covariance\n        matrix. The default is `scotts_factor`.  A subclass can overwrite this\n        method to provide a different method, or set it through a call to\n        `kde.set_bandwidth`.')
# Getting the type of 'gaussian_kde'
gaussian_kde_566009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'gaussian_kde')
# Obtaining the member 'covariance_factor' of a type
covariance_factor_566010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), gaussian_kde_566009, 'covariance_factor')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), covariance_factor_566010, '__doc__', str_566008)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
