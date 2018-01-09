
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Additional statistics functions with support for masked arrays.
3: 
4: '''
5: 
6: # Original author (2007): Pierre GF Gerard-Marchant
7: 
8: 
9: from __future__ import division, print_function, absolute_import
10: 
11: 
12: __all__ = ['compare_medians_ms',
13:            'hdquantiles', 'hdmedian', 'hdquantiles_sd',
14:            'idealfourths',
15:            'median_cihs','mjci','mquantiles_cimj',
16:            'rsh',
17:            'trimmed_mean_ci',]
18: 
19: 
20: import numpy as np
21: from numpy import float_, int_, ndarray
22: 
23: import numpy.ma as ma
24: from numpy.ma import MaskedArray
25: 
26: from . import mstats_basic as mstats
27: 
28: from scipy.stats.distributions import norm, beta, t, binom
29: 
30: 
31: def hdquantiles(data, prob=list([.25,.5,.75]), axis=None, var=False,):
32:     '''
33:     Computes quantile estimates with the Harrell-Davis method.
34: 
35:     The quantile estimates are calculated as a weighted linear combination
36:     of order statistics.
37: 
38:     Parameters
39:     ----------
40:     data : array_like
41:         Data array.
42:     prob : sequence, optional
43:         Sequence of quantiles to compute.
44:     axis : int or None, optional
45:         Axis along which to compute the quantiles. If None, use a flattened
46:         array.
47:     var : bool, optional
48:         Whether to return the variance of the estimate.
49: 
50:     Returns
51:     -------
52:     hdquantiles : MaskedArray
53:         A (p,) array of quantiles (if `var` is False), or a (2,p) array of
54:         quantiles and variances (if `var` is True), where ``p`` is the
55:         number of quantiles.
56: 
57:     See Also
58:     --------
59:     hdquantiles_sd
60: 
61:     '''
62:     def _hd_1D(data,prob,var):
63:         "Computes the HD quantiles for a 1D array. Returns nan for invalid data."
64:         xsorted = np.squeeze(np.sort(data.compressed().view(ndarray)))
65:         # Don't use length here, in case we have a numpy scalar
66:         n = xsorted.size
67: 
68:         hd = np.empty((2,len(prob)), float_)
69:         if n < 2:
70:             hd.flat = np.nan
71:             if var:
72:                 return hd
73:             return hd[0]
74: 
75:         v = np.arange(n+1) / float(n)
76:         betacdf = beta.cdf
77:         for (i,p) in enumerate(prob):
78:             _w = betacdf(v, (n+1)*p, (n+1)*(1-p))
79:             w = _w[1:] - _w[:-1]
80:             hd_mean = np.dot(w, xsorted)
81:             hd[0,i] = hd_mean
82:             #
83:             hd[1,i] = np.dot(w, (xsorted-hd_mean)**2)
84:             #
85:         hd[0, prob == 0] = xsorted[0]
86:         hd[0, prob == 1] = xsorted[-1]
87:         if var:
88:             hd[1, prob == 0] = hd[1, prob == 1] = np.nan
89:             return hd
90:         return hd[0]
91:     # Initialization & checks
92:     data = ma.array(data, copy=False, dtype=float_)
93:     p = np.array(prob, copy=False, ndmin=1)
94:     # Computes quantiles along axis (or globally)
95:     if (axis is None) or (data.ndim == 1):
96:         result = _hd_1D(data, p, var)
97:     else:
98:         if data.ndim > 2:
99:             raise ValueError("Array 'data' must be at most two dimensional, "
100:                              "but got data.ndim = %d" % data.ndim)
101:         result = ma.apply_along_axis(_hd_1D, axis, data, p, var)
102: 
103:     return ma.fix_invalid(result, copy=False)
104: 
105: 
106: def hdmedian(data, axis=-1, var=False):
107:     '''
108:     Returns the Harrell-Davis estimate of the median along the given axis.
109: 
110:     Parameters
111:     ----------
112:     data : ndarray
113:         Data array.
114:     axis : int, optional
115:         Axis along which to compute the quantiles. If None, use a flattened
116:         array.
117:     var : bool, optional
118:         Whether to return the variance of the estimate.
119: 
120:     Returns
121:     -------
122:     hdmedian : MaskedArray
123:         The median values.  If ``var=True``, the variance is returned inside
124:         the masked array.  E.g. for a 1-D array the shape change from (1,) to
125:         (2,).
126: 
127:     '''
128:     result = hdquantiles(data,[0.5], axis=axis, var=var)
129:     return result.squeeze()
130: 
131: 
132: def hdquantiles_sd(data, prob=list([.25,.5,.75]), axis=None):
133:     '''
134:     The standard error of the Harrell-Davis quantile estimates by jackknife.
135: 
136:     Parameters
137:     ----------
138:     data : array_like
139:         Data array.
140:     prob : sequence, optional
141:         Sequence of quantiles to compute.
142:     axis : int, optional
143:         Axis along which to compute the quantiles. If None, use a flattened
144:         array.
145: 
146:     Returns
147:     -------
148:     hdquantiles_sd : MaskedArray
149:         Standard error of the Harrell-Davis quantile estimates.
150: 
151:     See Also
152:     --------
153:     hdquantiles
154: 
155:     '''
156:     def _hdsd_1D(data, prob):
157:         "Computes the std error for 1D arrays."
158:         xsorted = np.sort(data.compressed())
159:         n = len(xsorted)
160: 
161:         hdsd = np.empty(len(prob), float_)
162:         if n < 2:
163:             hdsd.flat = np.nan
164: 
165:         vv = np.arange(n) / float(n-1)
166:         betacdf = beta.cdf
167: 
168:         for (i,p) in enumerate(prob):
169:             _w = betacdf(vv, (n+1)*p, (n+1)*(1-p))
170:             w = _w[1:] - _w[:-1]
171:             mx_ = np.fromiter([np.dot(w,xsorted[np.r_[list(range(0,k)),
172:                                                       list(range(k+1,n))].astype(int_)])
173:                                   for k in range(n)], dtype=float_)
174:             mx_var = np.array(mx_.var(), copy=False, ndmin=1) * n / float(n-1)
175:             hdsd[i] = float(n-1) * np.sqrt(np.diag(mx_var).diagonal() / float(n))
176:         return hdsd
177: 
178:     # Initialization & checks
179:     data = ma.array(data, copy=False, dtype=float_)
180:     p = np.array(prob, copy=False, ndmin=1)
181:     # Computes quantiles along axis (or globally)
182:     if (axis is None):
183:         result = _hdsd_1D(data, p)
184:     else:
185:         if data.ndim > 2:
186:             raise ValueError("Array 'data' must be at most two dimensional, "
187:                              "but got data.ndim = %d" % data.ndim)
188:         result = ma.apply_along_axis(_hdsd_1D, axis, data, p)
189: 
190:     return ma.fix_invalid(result, copy=False).ravel()
191: 
192: 
193: def trimmed_mean_ci(data, limits=(0.2,0.2), inclusive=(True,True),
194:                     alpha=0.05, axis=None):
195:     '''
196:     Selected confidence interval of the trimmed mean along the given axis.
197: 
198:     Parameters
199:     ----------
200:     data : array_like
201:         Input data.
202:     limits : {None, tuple}, optional
203:         None or a two item tuple.
204:         Tuple of the percentages to cut on each side of the array, with respect
205:         to the number of unmasked data, as floats between 0. and 1. If ``n``
206:         is the number of unmasked data before trimming, then
207:         (``n * limits[0]``)th smallest data and (``n * limits[1]``)th
208:         largest data are masked.  The total number of unmasked data after
209:         trimming is ``n * (1. - sum(limits))``.
210:         The value of one limit can be set to None to indicate an open interval.
211: 
212:         Defaults to (0.2, 0.2).
213:     inclusive : (2,) tuple of boolean, optional
214:         If relative==False, tuple indicating whether values exactly equal to
215:         the absolute limits are allowed.
216:         If relative==True, tuple indicating whether the number of data being
217:         masked on each side should be rounded (True) or truncated (False).
218: 
219:         Defaults to (True, True).
220:     alpha : float, optional
221:         Confidence level of the intervals.
222: 
223:         Defaults to 0.05.
224:     axis : int, optional
225:         Axis along which to cut. If None, uses a flattened version of `data`.
226: 
227:         Defaults to None.
228: 
229:     Returns
230:     -------
231:     trimmed_mean_ci : (2,) ndarray
232:         The lower and upper confidence intervals of the trimmed data.
233: 
234:     '''
235:     data = ma.array(data, copy=False)
236:     trimmed = mstats.trimr(data, limits=limits, inclusive=inclusive, axis=axis)
237:     tmean = trimmed.mean(axis)
238:     tstde = mstats.trimmed_stde(data,limits=limits,inclusive=inclusive,axis=axis)
239:     df = trimmed.count(axis) - 1
240:     tppf = t.ppf(1-alpha/2.,df)
241:     return np.array((tmean - tppf*tstde, tmean+tppf*tstde))
242: 
243: 
244: def mjci(data, prob=[0.25,0.5,0.75], axis=None):
245:     '''
246:     Returns the Maritz-Jarrett estimators of the standard error of selected
247:     experimental quantiles of the data.
248: 
249:     Parameters
250:     ----------
251:     data : ndarray
252:         Data array.
253:     prob : sequence, optional
254:         Sequence of quantiles to compute.
255:     axis : int or None, optional
256:         Axis along which to compute the quantiles. If None, use a flattened
257:         array.
258: 
259:     '''
260:     def _mjci_1D(data, p):
261:         data = np.sort(data.compressed())
262:         n = data.size
263:         prob = (np.array(p) * n + 0.5).astype(int_)
264:         betacdf = beta.cdf
265: 
266:         mj = np.empty(len(prob), float_)
267:         x = np.arange(1,n+1, dtype=float_) / n
268:         y = x - 1./n
269:         for (i,m) in enumerate(prob):
270:             W = betacdf(x,m-1,n-m) - betacdf(y,m-1,n-m)
271:             C1 = np.dot(W,data)
272:             C2 = np.dot(W,data**2)
273:             mj[i] = np.sqrt(C2 - C1**2)
274:         return mj
275: 
276:     data = ma.array(data, copy=False)
277:     if data.ndim > 2:
278:         raise ValueError("Array 'data' must be at most two dimensional, "
279:                          "but got data.ndim = %d" % data.ndim)
280: 
281:     p = np.array(prob, copy=False, ndmin=1)
282:     # Computes quantiles along axis (or globally)
283:     if (axis is None):
284:         return _mjci_1D(data, p)
285:     else:
286:         return ma.apply_along_axis(_mjci_1D, axis, data, p)
287: 
288: 
289: def mquantiles_cimj(data, prob=[0.25,0.50,0.75], alpha=0.05, axis=None):
290:     '''
291:     Computes the alpha confidence interval for the selected quantiles of the
292:     data, with Maritz-Jarrett estimators.
293: 
294:     Parameters
295:     ----------
296:     data : ndarray
297:         Data array.
298:     prob : sequence, optional
299:         Sequence of quantiles to compute.
300:     alpha : float, optional
301:         Confidence level of the intervals.
302:     axis : int or None, optional
303:         Axis along which to compute the quantiles.
304:         If None, use a flattened array.
305: 
306:     Returns
307:     -------
308:     ci_lower : ndarray
309:         The lower boundaries of the confidence interval.  Of the same length as
310:         `prob`.
311:     ci_upper : ndarray
312:         The upper boundaries of the confidence interval.  Of the same length as
313:         `prob`.
314: 
315:     '''
316:     alpha = min(alpha, 1 - alpha)
317:     z = norm.ppf(1 - alpha/2.)
318:     xq = mstats.mquantiles(data, prob, alphap=0, betap=0, axis=axis)
319:     smj = mjci(data, prob, axis=axis)
320:     return (xq - z * smj, xq + z * smj)
321: 
322: 
323: def median_cihs(data, alpha=0.05, axis=None):
324:     '''
325:     Computes the alpha-level confidence interval for the median of the data.
326: 
327:     Uses the Hettmasperger-Sheather method.
328: 
329:     Parameters
330:     ----------
331:     data : array_like
332:         Input data. Masked values are discarded. The input should be 1D only,
333:         or `axis` should be set to None.
334:     alpha : float, optional
335:         Confidence level of the intervals.
336:     axis : int or None, optional
337:         Axis along which to compute the quantiles. If None, use a flattened
338:         array.
339: 
340:     Returns
341:     -------
342:     median_cihs
343:         Alpha level confidence interval.
344: 
345:     '''
346:     def _cihs_1D(data, alpha):
347:         data = np.sort(data.compressed())
348:         n = len(data)
349:         alpha = min(alpha, 1-alpha)
350:         k = int(binom._ppf(alpha/2., n, 0.5))
351:         gk = binom.cdf(n-k,n,0.5) - binom.cdf(k-1,n,0.5)
352:         if gk < 1-alpha:
353:             k -= 1
354:             gk = binom.cdf(n-k,n,0.5) - binom.cdf(k-1,n,0.5)
355:         gkk = binom.cdf(n-k-1,n,0.5) - binom.cdf(k,n,0.5)
356:         I = (gk - 1 + alpha)/(gk - gkk)
357:         lambd = (n-k) * I / float(k + (n-2*k)*I)
358:         lims = (lambd*data[k] + (1-lambd)*data[k-1],
359:                 lambd*data[n-k-1] + (1-lambd)*data[n-k])
360:         return lims
361:     data = ma.array(data, copy=False)
362:     # Computes quantiles along axis (or globally)
363:     if (axis is None):
364:         result = _cihs_1D(data, alpha)
365:     else:
366:         if data.ndim > 2:
367:             raise ValueError("Array 'data' must be at most two dimensional, "
368:                              "but got data.ndim = %d" % data.ndim)
369:         result = ma.apply_along_axis(_cihs_1D, axis, data, alpha)
370: 
371:     return result
372: 
373: 
374: def compare_medians_ms(group_1, group_2, axis=None):
375:     '''
376:     Compares the medians from two independent groups along the given axis.
377: 
378:     The comparison is performed using the McKean-Schrader estimate of the
379:     standard error of the medians.
380: 
381:     Parameters
382:     ----------
383:     group_1 : array_like
384:         First dataset.  Has to be of size >=7.
385:     group_2 : array_like
386:         Second dataset.  Has to be of size >=7.
387:     axis : int, optional
388:         Axis along which the medians are estimated. If None, the arrays are
389:         flattened.  If `axis` is not None, then `group_1` and `group_2`
390:         should have the same shape.
391: 
392:     Returns
393:     -------
394:     compare_medians_ms : {float, ndarray}
395:         If `axis` is None, then returns a float, otherwise returns a 1-D
396:         ndarray of floats with a length equal to the length of `group_1`
397:         along `axis`.
398: 
399:     '''
400:     (med_1, med_2) = (ma.median(group_1,axis=axis), ma.median(group_2,axis=axis))
401:     (std_1, std_2) = (mstats.stde_median(group_1, axis=axis),
402:                       mstats.stde_median(group_2, axis=axis))
403:     W = np.abs(med_1 - med_2) / ma.sqrt(std_1**2 + std_2**2)
404:     return 1 - norm.cdf(W)
405: 
406: 
407: def idealfourths(data, axis=None):
408:     '''
409:     Returns an estimate of the lower and upper quartiles.
410: 
411:     Uses the ideal fourths algorithm.
412: 
413:     Parameters
414:     ----------
415:     data : array_like
416:         Input array.
417:     axis : int, optional
418:         Axis along which the quartiles are estimated. If None, the arrays are
419:         flattened.
420: 
421:     Returns
422:     -------
423:     idealfourths : {list of floats, masked array}
424:         Returns the two internal values that divide `data` into four parts
425:         using the ideal fourths algorithm either along the flattened array
426:         (if `axis` is None) or along `axis` of `data`.
427: 
428:     '''
429:     def _idf(data):
430:         x = data.compressed()
431:         n = len(x)
432:         if n < 3:
433:             return [np.nan,np.nan]
434:         (j,h) = divmod(n/4. + 5/12.,1)
435:         j = int(j)
436:         qlo = (1-h)*x[j-1] + h*x[j]
437:         k = n - j
438:         qup = (1-h)*x[k] + h*x[k-1]
439:         return [qlo, qup]
440:     data = ma.sort(data, axis=axis).view(MaskedArray)
441:     if (axis is None):
442:         return _idf(data)
443:     else:
444:         return ma.apply_along_axis(_idf, axis, data)
445: 
446: 
447: def rsh(data, points=None):
448:     '''
449:     Evaluates Rosenblatt's shifted histogram estimators for each data point.
450: 
451:     Rosenblatt's estimator is a centered finite-difference approximation to the
452:     derivative of the empirical cumulative distribution function.
453: 
454:     Parameters
455:     ----------
456:     data : sequence
457:         Input data, should be 1-D. Masked values are ignored.
458:     points : sequence or None, optional
459:         Sequence of points where to evaluate Rosenblatt shifted histogram.
460:         If None, use the data.
461: 
462:     '''
463:     data = ma.array(data, copy=False)
464:     if points is None:
465:         points = data
466:     else:
467:         points = np.array(points, copy=False, ndmin=1)
468: 
469:     if data.ndim != 1:
470:         raise AttributeError("The input array should be 1D only !")
471: 
472:     n = data.count()
473:     r = idealfourths(data, axis=None)
474:     h = 1.2 * (r[-1]-r[0]) / n**(1./5)
475:     nhi = (data[:,None] <= points[None,:] + h).sum(0)
476:     nlo = (data[:,None] < points[None,:] - h).sum(0)
477:     return (nhi-nlo) / (2.*n*h)
478: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_578473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nAdditional statistics functions with support for masked arrays.\n\n')

# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):
__all__ = ['compare_medians_ms', 'hdquantiles', 'hdmedian', 'hdquantiles_sd', 'idealfourths', 'median_cihs', 'mjci', 'mquantiles_cimj', 'rsh', 'trimmed_mean_ci']
module_type_store.set_exportable_members(['compare_medians_ms', 'hdquantiles', 'hdmedian', 'hdquantiles_sd', 'idealfourths', 'median_cihs', 'mjci', 'mquantiles_cimj', 'rsh', 'trimmed_mean_ci'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_578474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_578475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'compare_medians_ms')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_578474, str_578475)
# Adding element type (line 12)
str_578476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'hdquantiles')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_578474, str_578476)
# Adding element type (line 12)
str_578477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 26), 'str', 'hdmedian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_578474, str_578477)
# Adding element type (line 12)
str_578478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 38), 'str', 'hdquantiles_sd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_578474, str_578478)
# Adding element type (line 12)
str_578479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'idealfourths')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_578474, str_578479)
# Adding element type (line 12)
str_578480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'median_cihs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_578474, str_578480)
# Adding element type (line 12)
str_578481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'str', 'mjci')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_578474, str_578481)
# Adding element type (line 12)
str_578482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'str', 'mquantiles_cimj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_578474, str_578482)
# Adding element type (line 12)
str_578483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'rsh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_578474, str_578483)
# Adding element type (line 12)
str_578484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'trimmed_mean_ci')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_578474, str_578484)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_578474)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import numpy' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_578485 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy')

if (type(import_578485) is not StypyTypeError):

    if (import_578485 != 'pyd_module'):
        __import__(import_578485)
        sys_modules_578486 = sys.modules[import_578485]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'np', sys_modules_578486.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy', import_578485)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from numpy import float_, int_, ndarray' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_578487 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy')

if (type(import_578487) is not StypyTypeError):

    if (import_578487 != 'pyd_module'):
        __import__(import_578487)
        sys_modules_578488 = sys.modules[import_578487]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', sys_modules_578488.module_type_store, module_type_store, ['float_', 'int_', 'ndarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_578488, sys_modules_578488.module_type_store, module_type_store)
    else:
        from numpy import float_, int_, ndarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', None, module_type_store, ['float_', 'int_', 'ndarray'], [float_, int_, ndarray])

else:
    # Assigning a type to the variable 'numpy' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', import_578487)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import numpy.ma' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_578489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.ma')

if (type(import_578489) is not StypyTypeError):

    if (import_578489 != 'pyd_module'):
        __import__(import_578489)
        sys_modules_578490 = sys.modules[import_578489]
        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'ma', sys_modules_578490.module_type_store, module_type_store)
    else:
        import numpy.ma as ma

        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'ma', numpy.ma, module_type_store)

else:
    # Assigning a type to the variable 'numpy.ma' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.ma', import_578489)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.ma import MaskedArray' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_578491 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.ma')

if (type(import_578491) is not StypyTypeError):

    if (import_578491 != 'pyd_module'):
        __import__(import_578491)
        sys_modules_578492 = sys.modules[import_578491]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.ma', sys_modules_578492.module_type_store, module_type_store, ['MaskedArray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_578492, sys_modules_578492.module_type_store, module_type_store)
    else:
        from numpy.ma import MaskedArray

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.ma', None, module_type_store, ['MaskedArray'], [MaskedArray])

else:
    # Assigning a type to the variable 'numpy.ma' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.ma', import_578491)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from scipy.stats import mstats' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_578493 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.stats')

if (type(import_578493) is not StypyTypeError):

    if (import_578493 != 'pyd_module'):
        __import__(import_578493)
        sys_modules_578494 = sys.modules[import_578493]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.stats', sys_modules_578494.module_type_store, module_type_store, ['mstats_basic'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_578494, sys_modules_578494.module_type_store, module_type_store)
    else:
        from scipy.stats import mstats_basic as mstats

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.stats', None, module_type_store, ['mstats_basic'], [mstats])

else:
    # Assigning a type to the variable 'scipy.stats' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.stats', import_578493)

# Adding an alias
module_type_store.add_alias('mstats', 'mstats_basic')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from scipy.stats.distributions import norm, beta, t, binom' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_578495 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.stats.distributions')

if (type(import_578495) is not StypyTypeError):

    if (import_578495 != 'pyd_module'):
        __import__(import_578495)
        sys_modules_578496 = sys.modules[import_578495]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.stats.distributions', sys_modules_578496.module_type_store, module_type_store, ['norm', 'beta', 't', 'binom'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_578496, sys_modules_578496.module_type_store, module_type_store)
    else:
        from scipy.stats.distributions import norm, beta, t, binom

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.stats.distributions', None, module_type_store, ['norm', 'beta', 't', 'binom'], [norm, beta, t, binom])

else:
    # Assigning a type to the variable 'scipy.stats.distributions' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.stats.distributions', import_578495)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


@norecursion
def hdquantiles(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Call to list(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_578498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    float_578499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 32), list_578498, float_578499)
    # Adding element type (line 31)
    float_578500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 37), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 32), list_578498, float_578500)
    # Adding element type (line 31)
    float_578501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 32), list_578498, float_578501)
    
    # Processing the call keyword arguments (line 31)
    kwargs_578502 = {}
    # Getting the type of 'list' (line 31)
    list_578497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'list', False)
    # Calling list(args, kwargs) (line 31)
    list_call_result_578503 = invoke(stypy.reporting.localization.Localization(__file__, 31, 27), list_578497, *[list_578498], **kwargs_578502)
    
    # Getting the type of 'None' (line 31)
    None_578504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 52), 'None')
    # Getting the type of 'False' (line 31)
    False_578505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 62), 'False')
    defaults = [list_call_result_578503, None_578504, False_578505]
    # Create a new context for function 'hdquantiles'
    module_type_store = module_type_store.open_function_context('hdquantiles', 31, 0, False)
    
    # Passed parameters checking function
    hdquantiles.stypy_localization = localization
    hdquantiles.stypy_type_of_self = None
    hdquantiles.stypy_type_store = module_type_store
    hdquantiles.stypy_function_name = 'hdquantiles'
    hdquantiles.stypy_param_names_list = ['data', 'prob', 'axis', 'var']
    hdquantiles.stypy_varargs_param_name = None
    hdquantiles.stypy_kwargs_param_name = None
    hdquantiles.stypy_call_defaults = defaults
    hdquantiles.stypy_call_varargs = varargs
    hdquantiles.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hdquantiles', ['data', 'prob', 'axis', 'var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hdquantiles', localization, ['data', 'prob', 'axis', 'var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hdquantiles(...)' code ##################

    str_578506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', '\n    Computes quantile estimates with the Harrell-Davis method.\n\n    The quantile estimates are calculated as a weighted linear combination\n    of order statistics.\n\n    Parameters\n    ----------\n    data : array_like\n        Data array.\n    prob : sequence, optional\n        Sequence of quantiles to compute.\n    axis : int or None, optional\n        Axis along which to compute the quantiles. If None, use a flattened\n        array.\n    var : bool, optional\n        Whether to return the variance of the estimate.\n\n    Returns\n    -------\n    hdquantiles : MaskedArray\n        A (p,) array of quantiles (if `var` is False), or a (2,p) array of\n        quantiles and variances (if `var` is True), where ``p`` is the\n        number of quantiles.\n\n    See Also\n    --------\n    hdquantiles_sd\n\n    ')

    @norecursion
    def _hd_1D(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_hd_1D'
        module_type_store = module_type_store.open_function_context('_hd_1D', 62, 4, False)
        
        # Passed parameters checking function
        _hd_1D.stypy_localization = localization
        _hd_1D.stypy_type_of_self = None
        _hd_1D.stypy_type_store = module_type_store
        _hd_1D.stypy_function_name = '_hd_1D'
        _hd_1D.stypy_param_names_list = ['data', 'prob', 'var']
        _hd_1D.stypy_varargs_param_name = None
        _hd_1D.stypy_kwargs_param_name = None
        _hd_1D.stypy_call_defaults = defaults
        _hd_1D.stypy_call_varargs = varargs
        _hd_1D.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_hd_1D', ['data', 'prob', 'var'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_hd_1D', localization, ['data', 'prob', 'var'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_hd_1D(...)' code ##################

        str_578507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'str', 'Computes the HD quantiles for a 1D array. Returns nan for invalid data.')
        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to squeeze(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to sort(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to view(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'ndarray' (line 64)
        ndarray_578517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 60), 'ndarray', False)
        # Processing the call keyword arguments (line 64)
        kwargs_578518 = {}
        
        # Call to compressed(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_578514 = {}
        # Getting the type of 'data' (line 64)
        data_578512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'data', False)
        # Obtaining the member 'compressed' of a type (line 64)
        compressed_578513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 37), data_578512, 'compressed')
        # Calling compressed(args, kwargs) (line 64)
        compressed_call_result_578515 = invoke(stypy.reporting.localization.Localization(__file__, 64, 37), compressed_578513, *[], **kwargs_578514)
        
        # Obtaining the member 'view' of a type (line 64)
        view_578516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 37), compressed_call_result_578515, 'view')
        # Calling view(args, kwargs) (line 64)
        view_call_result_578519 = invoke(stypy.reporting.localization.Localization(__file__, 64, 37), view_578516, *[ndarray_578517], **kwargs_578518)
        
        # Processing the call keyword arguments (line 64)
        kwargs_578520 = {}
        # Getting the type of 'np' (line 64)
        np_578510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'np', False)
        # Obtaining the member 'sort' of a type (line 64)
        sort_578511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 29), np_578510, 'sort')
        # Calling sort(args, kwargs) (line 64)
        sort_call_result_578521 = invoke(stypy.reporting.localization.Localization(__file__, 64, 29), sort_578511, *[view_call_result_578519], **kwargs_578520)
        
        # Processing the call keyword arguments (line 64)
        kwargs_578522 = {}
        # Getting the type of 'np' (line 64)
        np_578508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'np', False)
        # Obtaining the member 'squeeze' of a type (line 64)
        squeeze_578509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 18), np_578508, 'squeeze')
        # Calling squeeze(args, kwargs) (line 64)
        squeeze_call_result_578523 = invoke(stypy.reporting.localization.Localization(__file__, 64, 18), squeeze_578509, *[sort_call_result_578521], **kwargs_578522)
        
        # Assigning a type to the variable 'xsorted' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'xsorted', squeeze_call_result_578523)
        
        # Assigning a Attribute to a Name (line 66):
        
        # Assigning a Attribute to a Name (line 66):
        # Getting the type of 'xsorted' (line 66)
        xsorted_578524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'xsorted')
        # Obtaining the member 'size' of a type (line 66)
        size_578525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), xsorted_578524, 'size')
        # Assigning a type to the variable 'n' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'n', size_578525)
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to empty(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_578528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        int_578529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 23), tuple_578528, int_578529)
        # Adding element type (line 68)
        
        # Call to len(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'prob' (line 68)
        prob_578531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'prob', False)
        # Processing the call keyword arguments (line 68)
        kwargs_578532 = {}
        # Getting the type of 'len' (line 68)
        len_578530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'len', False)
        # Calling len(args, kwargs) (line 68)
        len_call_result_578533 = invoke(stypy.reporting.localization.Localization(__file__, 68, 25), len_578530, *[prob_578531], **kwargs_578532)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 23), tuple_578528, len_call_result_578533)
        
        # Getting the type of 'float_' (line 68)
        float__578534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 37), 'float_', False)
        # Processing the call keyword arguments (line 68)
        kwargs_578535 = {}
        # Getting the type of 'np' (line 68)
        np_578526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 13), 'np', False)
        # Obtaining the member 'empty' of a type (line 68)
        empty_578527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 13), np_578526, 'empty')
        # Calling empty(args, kwargs) (line 68)
        empty_call_result_578536 = invoke(stypy.reporting.localization.Localization(__file__, 68, 13), empty_578527, *[tuple_578528, float__578534], **kwargs_578535)
        
        # Assigning a type to the variable 'hd' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'hd', empty_call_result_578536)
        
        
        # Getting the type of 'n' (line 69)
        n_578537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'n')
        int_578538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 15), 'int')
        # Applying the binary operator '<' (line 69)
        result_lt_578539 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 11), '<', n_578537, int_578538)
        
        # Testing the type of an if condition (line 69)
        if_condition_578540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), result_lt_578539)
        # Assigning a type to the variable 'if_condition_578540' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_578540', if_condition_578540)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 70):
        
        # Assigning a Attribute to a Attribute (line 70):
        # Getting the type of 'np' (line 70)
        np_578541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 22), 'np')
        # Obtaining the member 'nan' of a type (line 70)
        nan_578542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 22), np_578541, 'nan')
        # Getting the type of 'hd' (line 70)
        hd_578543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'hd')
        # Setting the type of the member 'flat' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), hd_578543, 'flat', nan_578542)
        
        # Getting the type of 'var' (line 71)
        var_578544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'var')
        # Testing the type of an if condition (line 71)
        if_condition_578545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 12), var_578544)
        # Assigning a type to the variable 'if_condition_578545' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'if_condition_578545', if_condition_578545)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'hd' (line 72)
        hd_578546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'hd')
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'stypy_return_type', hd_578546)
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        int_578547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 22), 'int')
        # Getting the type of 'hd' (line 73)
        hd_578548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 19), 'hd')
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___578549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 19), hd_578548, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_578550 = invoke(stypy.reporting.localization.Localization(__file__, 73, 19), getitem___578549, int_578547)
        
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', subscript_call_result_578550)
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 75):
        
        # Assigning a BinOp to a Name (line 75):
        
        # Call to arange(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'n' (line 75)
        n_578553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'n', False)
        int_578554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 24), 'int')
        # Applying the binary operator '+' (line 75)
        result_add_578555 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 22), '+', n_578553, int_578554)
        
        # Processing the call keyword arguments (line 75)
        kwargs_578556 = {}
        # Getting the type of 'np' (line 75)
        np_578551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 75)
        arange_578552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), np_578551, 'arange')
        # Calling arange(args, kwargs) (line 75)
        arange_call_result_578557 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), arange_578552, *[result_add_578555], **kwargs_578556)
        
        
        # Call to float(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'n' (line 75)
        n_578559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 35), 'n', False)
        # Processing the call keyword arguments (line 75)
        kwargs_578560 = {}
        # Getting the type of 'float' (line 75)
        float_578558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 29), 'float', False)
        # Calling float(args, kwargs) (line 75)
        float_call_result_578561 = invoke(stypy.reporting.localization.Localization(__file__, 75, 29), float_578558, *[n_578559], **kwargs_578560)
        
        # Applying the binary operator 'div' (line 75)
        result_div_578562 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 12), 'div', arange_call_result_578557, float_call_result_578561)
        
        # Assigning a type to the variable 'v' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'v', result_div_578562)
        
        # Assigning a Attribute to a Name (line 76):
        
        # Assigning a Attribute to a Name (line 76):
        # Getting the type of 'beta' (line 76)
        beta_578563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'beta')
        # Obtaining the member 'cdf' of a type (line 76)
        cdf_578564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 18), beta_578563, 'cdf')
        # Assigning a type to the variable 'betacdf' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'betacdf', cdf_578564)
        
        
        # Call to enumerate(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'prob' (line 77)
        prob_578566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 31), 'prob', False)
        # Processing the call keyword arguments (line 77)
        kwargs_578567 = {}
        # Getting the type of 'enumerate' (line 77)
        enumerate_578565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 77)
        enumerate_call_result_578568 = invoke(stypy.reporting.localization.Localization(__file__, 77, 21), enumerate_578565, *[prob_578566], **kwargs_578567)
        
        # Testing the type of a for loop iterable (line 77)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 8), enumerate_call_result_578568)
        # Getting the type of the for loop variable (line 77)
        for_loop_var_578569 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 8), enumerate_call_result_578568)
        # Assigning a type to the variable 'i' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 8), for_loop_var_578569))
        # Assigning a type to the variable 'p' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 8), for_loop_var_578569))
        # SSA begins for a for statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to betacdf(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'v' (line 78)
        v_578571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'v', False)
        # Getting the type of 'n' (line 78)
        n_578572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 29), 'n', False)
        int_578573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 31), 'int')
        # Applying the binary operator '+' (line 78)
        result_add_578574 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 29), '+', n_578572, int_578573)
        
        # Getting the type of 'p' (line 78)
        p_578575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 34), 'p', False)
        # Applying the binary operator '*' (line 78)
        result_mul_578576 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 28), '*', result_add_578574, p_578575)
        
        # Getting the type of 'n' (line 78)
        n_578577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'n', False)
        int_578578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 40), 'int')
        # Applying the binary operator '+' (line 78)
        result_add_578579 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 38), '+', n_578577, int_578578)
        
        int_578580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 44), 'int')
        # Getting the type of 'p' (line 78)
        p_578581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 46), 'p', False)
        # Applying the binary operator '-' (line 78)
        result_sub_578582 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 44), '-', int_578580, p_578581)
        
        # Applying the binary operator '*' (line 78)
        result_mul_578583 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 37), '*', result_add_578579, result_sub_578582)
        
        # Processing the call keyword arguments (line 78)
        kwargs_578584 = {}
        # Getting the type of 'betacdf' (line 78)
        betacdf_578570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'betacdf', False)
        # Calling betacdf(args, kwargs) (line 78)
        betacdf_call_result_578585 = invoke(stypy.reporting.localization.Localization(__file__, 78, 17), betacdf_578570, *[v_578571, result_mul_578576, result_mul_578583], **kwargs_578584)
        
        # Assigning a type to the variable '_w' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), '_w', betacdf_call_result_578585)
        
        # Assigning a BinOp to a Name (line 79):
        
        # Assigning a BinOp to a Name (line 79):
        
        # Obtaining the type of the subscript
        int_578586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'int')
        slice_578587 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 79, 16), int_578586, None, None)
        # Getting the type of '_w' (line 79)
        _w_578588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), '_w')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___578589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 16), _w_578588, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_578590 = invoke(stypy.reporting.localization.Localization(__file__, 79, 16), getitem___578589, slice_578587)
        
        
        # Obtaining the type of the subscript
        int_578591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 29), 'int')
        slice_578592 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 79, 25), None, int_578591, None)
        # Getting the type of '_w' (line 79)
        _w_578593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), '_w')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___578594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), _w_578593, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_578595 = invoke(stypy.reporting.localization.Localization(__file__, 79, 25), getitem___578594, slice_578592)
        
        # Applying the binary operator '-' (line 79)
        result_sub_578596 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 16), '-', subscript_call_result_578590, subscript_call_result_578595)
        
        # Assigning a type to the variable 'w' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'w', result_sub_578596)
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to dot(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'w' (line 80)
        w_578599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'w', False)
        # Getting the type of 'xsorted' (line 80)
        xsorted_578600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 32), 'xsorted', False)
        # Processing the call keyword arguments (line 80)
        kwargs_578601 = {}
        # Getting the type of 'np' (line 80)
        np_578597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'np', False)
        # Obtaining the member 'dot' of a type (line 80)
        dot_578598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 22), np_578597, 'dot')
        # Calling dot(args, kwargs) (line 80)
        dot_call_result_578602 = invoke(stypy.reporting.localization.Localization(__file__, 80, 22), dot_578598, *[w_578599, xsorted_578600], **kwargs_578601)
        
        # Assigning a type to the variable 'hd_mean' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'hd_mean', dot_call_result_578602)
        
        # Assigning a Name to a Subscript (line 81):
        
        # Assigning a Name to a Subscript (line 81):
        # Getting the type of 'hd_mean' (line 81)
        hd_mean_578603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'hd_mean')
        # Getting the type of 'hd' (line 81)
        hd_578604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'hd')
        
        # Obtaining an instance of the builtin type 'tuple' (line 81)
        tuple_578605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 81)
        # Adding element type (line 81)
        int_578606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 15), tuple_578605, int_578606)
        # Adding element type (line 81)
        # Getting the type of 'i' (line 81)
        i_578607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 15), tuple_578605, i_578607)
        
        # Storing an element on a container (line 81)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 12), hd_578604, (tuple_578605, hd_mean_578603))
        
        # Assigning a Call to a Subscript (line 83):
        
        # Assigning a Call to a Subscript (line 83):
        
        # Call to dot(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'w' (line 83)
        w_578610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'w', False)
        # Getting the type of 'xsorted' (line 83)
        xsorted_578611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 33), 'xsorted', False)
        # Getting the type of 'hd_mean' (line 83)
        hd_mean_578612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 41), 'hd_mean', False)
        # Applying the binary operator '-' (line 83)
        result_sub_578613 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 33), '-', xsorted_578611, hd_mean_578612)
        
        int_578614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 51), 'int')
        # Applying the binary operator '**' (line 83)
        result_pow_578615 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 32), '**', result_sub_578613, int_578614)
        
        # Processing the call keyword arguments (line 83)
        kwargs_578616 = {}
        # Getting the type of 'np' (line 83)
        np_578608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'np', False)
        # Obtaining the member 'dot' of a type (line 83)
        dot_578609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 22), np_578608, 'dot')
        # Calling dot(args, kwargs) (line 83)
        dot_call_result_578617 = invoke(stypy.reporting.localization.Localization(__file__, 83, 22), dot_578609, *[w_578610, result_pow_578615], **kwargs_578616)
        
        # Getting the type of 'hd' (line 83)
        hd_578618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'hd')
        
        # Obtaining an instance of the builtin type 'tuple' (line 83)
        tuple_578619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 83)
        # Adding element type (line 83)
        int_578620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 15), tuple_578619, int_578620)
        # Adding element type (line 83)
        # Getting the type of 'i' (line 83)
        i_578621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 15), tuple_578619, i_578621)
        
        # Storing an element on a container (line 83)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 12), hd_578618, (tuple_578619, dot_call_result_578617))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Subscript (line 85):
        
        # Assigning a Subscript to a Subscript (line 85):
        
        # Obtaining the type of the subscript
        int_578622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 35), 'int')
        # Getting the type of 'xsorted' (line 85)
        xsorted_578623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'xsorted')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___578624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 27), xsorted_578623, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_578625 = invoke(stypy.reporting.localization.Localization(__file__, 85, 27), getitem___578624, int_578622)
        
        # Getting the type of 'hd' (line 85)
        hd_578626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'hd')
        
        # Obtaining an instance of the builtin type 'tuple' (line 85)
        tuple_578627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 11), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 85)
        # Adding element type (line 85)
        int_578628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 11), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), tuple_578627, int_578628)
        # Adding element type (line 85)
        
        # Getting the type of 'prob' (line 85)
        prob_578629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 14), 'prob')
        int_578630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'int')
        # Applying the binary operator '==' (line 85)
        result_eq_578631 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 14), '==', prob_578629, int_578630)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), tuple_578627, result_eq_578631)
        
        # Storing an element on a container (line 85)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), hd_578626, (tuple_578627, subscript_call_result_578625))
        
        # Assigning a Subscript to a Subscript (line 86):
        
        # Assigning a Subscript to a Subscript (line 86):
        
        # Obtaining the type of the subscript
        int_578632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 35), 'int')
        # Getting the type of 'xsorted' (line 86)
        xsorted_578633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 27), 'xsorted')
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___578634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 27), xsorted_578633, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_578635 = invoke(stypy.reporting.localization.Localization(__file__, 86, 27), getitem___578634, int_578632)
        
        # Getting the type of 'hd' (line 86)
        hd_578636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'hd')
        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_578637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 11), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        int_578638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 11), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), tuple_578637, int_578638)
        # Adding element type (line 86)
        
        # Getting the type of 'prob' (line 86)
        prob_578639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'prob')
        int_578640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 22), 'int')
        # Applying the binary operator '==' (line 86)
        result_eq_578641 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 14), '==', prob_578639, int_578640)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 11), tuple_578637, result_eq_578641)
        
        # Storing an element on a container (line 86)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), hd_578636, (tuple_578637, subscript_call_result_578635))
        
        # Getting the type of 'var' (line 87)
        var_578642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'var')
        # Testing the type of an if condition (line 87)
        if_condition_578643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 8), var_578642)
        # Assigning a type to the variable 'if_condition_578643' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'if_condition_578643', if_condition_578643)
        # SSA begins for if statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Attribute to a Subscript (line 88):
        # Getting the type of 'np' (line 88)
        np_578644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 50), 'np')
        # Obtaining the member 'nan' of a type (line 88)
        nan_578645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 50), np_578644, 'nan')
        # Getting the type of 'hd' (line 88)
        hd_578646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 31), 'hd')
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_578647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        int_578648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 34), tuple_578647, int_578648)
        # Adding element type (line 88)
        
        # Getting the type of 'prob' (line 88)
        prob_578649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 37), 'prob')
        int_578650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 45), 'int')
        # Applying the binary operator '==' (line 88)
        result_eq_578651 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 37), '==', prob_578649, int_578650)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 34), tuple_578647, result_eq_578651)
        
        # Storing an element on a container (line 88)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 31), hd_578646, (tuple_578647, nan_578645))
        
        # Assigning a Subscript to a Subscript (line 88):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_578652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        int_578653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 34), tuple_578652, int_578653)
        # Adding element type (line 88)
        
        # Getting the type of 'prob' (line 88)
        prob_578654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 37), 'prob')
        int_578655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 45), 'int')
        # Applying the binary operator '==' (line 88)
        result_eq_578656 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 37), '==', prob_578654, int_578655)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 34), tuple_578652, result_eq_578656)
        
        # Getting the type of 'hd' (line 88)
        hd_578657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 31), 'hd')
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___578658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 31), hd_578657, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_578659 = invoke(stypy.reporting.localization.Localization(__file__, 88, 31), getitem___578658, tuple_578652)
        
        # Getting the type of 'hd' (line 88)
        hd_578660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'hd')
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_578661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        int_578662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), tuple_578661, int_578662)
        # Adding element type (line 88)
        
        # Getting the type of 'prob' (line 88)
        prob_578663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'prob')
        int_578664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 26), 'int')
        # Applying the binary operator '==' (line 88)
        result_eq_578665 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 18), '==', prob_578663, int_578664)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), tuple_578661, result_eq_578665)
        
        # Storing an element on a container (line 88)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 12), hd_578660, (tuple_578661, subscript_call_result_578659))
        # Getting the type of 'hd' (line 89)
        hd_578666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'hd')
        # Assigning a type to the variable 'stypy_return_type' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'stypy_return_type', hd_578666)
        # SSA join for if statement (line 87)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        int_578667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'int')
        # Getting the type of 'hd' (line 90)
        hd_578668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'hd')
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___578669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 15), hd_578668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_578670 = invoke(stypy.reporting.localization.Localization(__file__, 90, 15), getitem___578669, int_578667)
        
        # Assigning a type to the variable 'stypy_return_type' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'stypy_return_type', subscript_call_result_578670)
        
        # ################# End of '_hd_1D(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_hd_1D' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_578671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_578671)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_hd_1D'
        return stypy_return_type_578671

    # Assigning a type to the variable '_hd_1D' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), '_hd_1D', _hd_1D)
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to array(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'data' (line 92)
    data_578674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'data', False)
    # Processing the call keyword arguments (line 92)
    # Getting the type of 'False' (line 92)
    False_578675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'False', False)
    keyword_578676 = False_578675
    # Getting the type of 'float_' (line 92)
    float__578677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 44), 'float_', False)
    keyword_578678 = float__578677
    kwargs_578679 = {'dtype': keyword_578678, 'copy': keyword_578676}
    # Getting the type of 'ma' (line 92)
    ma_578672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'ma', False)
    # Obtaining the member 'array' of a type (line 92)
    array_578673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 11), ma_578672, 'array')
    # Calling array(args, kwargs) (line 92)
    array_call_result_578680 = invoke(stypy.reporting.localization.Localization(__file__, 92, 11), array_578673, *[data_578674], **kwargs_578679)
    
    # Assigning a type to the variable 'data' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'data', array_call_result_578680)
    
    # Assigning a Call to a Name (line 93):
    
    # Assigning a Call to a Name (line 93):
    
    # Call to array(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'prob' (line 93)
    prob_578683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'prob', False)
    # Processing the call keyword arguments (line 93)
    # Getting the type of 'False' (line 93)
    False_578684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'False', False)
    keyword_578685 = False_578684
    int_578686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 41), 'int')
    keyword_578687 = int_578686
    kwargs_578688 = {'copy': keyword_578685, 'ndmin': keyword_578687}
    # Getting the type of 'np' (line 93)
    np_578681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 93)
    array_578682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), np_578681, 'array')
    # Calling array(args, kwargs) (line 93)
    array_call_result_578689 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), array_578682, *[prob_578683], **kwargs_578688)
    
    # Assigning a type to the variable 'p' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'p', array_call_result_578689)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'axis' (line 95)
    axis_578690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'axis')
    # Getting the type of 'None' (line 95)
    None_578691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'None')
    # Applying the binary operator 'is' (line 95)
    result_is__578692 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 8), 'is', axis_578690, None_578691)
    
    
    # Getting the type of 'data' (line 95)
    data_578693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'data')
    # Obtaining the member 'ndim' of a type (line 95)
    ndim_578694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 26), data_578693, 'ndim')
    int_578695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 39), 'int')
    # Applying the binary operator '==' (line 95)
    result_eq_578696 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 26), '==', ndim_578694, int_578695)
    
    # Applying the binary operator 'or' (line 95)
    result_or_keyword_578697 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 7), 'or', result_is__578692, result_eq_578696)
    
    # Testing the type of an if condition (line 95)
    if_condition_578698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 4), result_or_keyword_578697)
    # Assigning a type to the variable 'if_condition_578698' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'if_condition_578698', if_condition_578698)
    # SSA begins for if statement (line 95)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to _hd_1D(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'data' (line 96)
    data_578700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'data', False)
    # Getting the type of 'p' (line 96)
    p_578701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 30), 'p', False)
    # Getting the type of 'var' (line 96)
    var_578702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'var', False)
    # Processing the call keyword arguments (line 96)
    kwargs_578703 = {}
    # Getting the type of '_hd_1D' (line 96)
    _hd_1D_578699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), '_hd_1D', False)
    # Calling _hd_1D(args, kwargs) (line 96)
    _hd_1D_call_result_578704 = invoke(stypy.reporting.localization.Localization(__file__, 96, 17), _hd_1D_578699, *[data_578700, p_578701, var_578702], **kwargs_578703)
    
    # Assigning a type to the variable 'result' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'result', _hd_1D_call_result_578704)
    # SSA branch for the else part of an if statement (line 95)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'data' (line 98)
    data_578705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'data')
    # Obtaining the member 'ndim' of a type (line 98)
    ndim_578706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), data_578705, 'ndim')
    int_578707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 23), 'int')
    # Applying the binary operator '>' (line 98)
    result_gt_578708 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 11), '>', ndim_578706, int_578707)
    
    # Testing the type of an if condition (line 98)
    if_condition_578709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), result_gt_578708)
    # Assigning a type to the variable 'if_condition_578709' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'if_condition_578709', if_condition_578709)
    # SSA begins for if statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 99)
    # Processing the call arguments (line 99)
    str_578711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'str', "Array 'data' must be at most two dimensional, but got data.ndim = %d")
    # Getting the type of 'data' (line 100)
    data_578712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 56), 'data', False)
    # Obtaining the member 'ndim' of a type (line 100)
    ndim_578713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 56), data_578712, 'ndim')
    # Applying the binary operator '%' (line 99)
    result_mod_578714 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 29), '%', str_578711, ndim_578713)
    
    # Processing the call keyword arguments (line 99)
    kwargs_578715 = {}
    # Getting the type of 'ValueError' (line 99)
    ValueError_578710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 99)
    ValueError_call_result_578716 = invoke(stypy.reporting.localization.Localization(__file__, 99, 18), ValueError_578710, *[result_mod_578714], **kwargs_578715)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 99, 12), ValueError_call_result_578716, 'raise parameter', BaseException)
    # SSA join for if statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to apply_along_axis(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of '_hd_1D' (line 101)
    _hd_1D_578719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), '_hd_1D', False)
    # Getting the type of 'axis' (line 101)
    axis_578720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 45), 'axis', False)
    # Getting the type of 'data' (line 101)
    data_578721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'data', False)
    # Getting the type of 'p' (line 101)
    p_578722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 57), 'p', False)
    # Getting the type of 'var' (line 101)
    var_578723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'var', False)
    # Processing the call keyword arguments (line 101)
    kwargs_578724 = {}
    # Getting the type of 'ma' (line 101)
    ma_578717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'ma', False)
    # Obtaining the member 'apply_along_axis' of a type (line 101)
    apply_along_axis_578718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 17), ma_578717, 'apply_along_axis')
    # Calling apply_along_axis(args, kwargs) (line 101)
    apply_along_axis_call_result_578725 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), apply_along_axis_578718, *[_hd_1D_578719, axis_578720, data_578721, p_578722, var_578723], **kwargs_578724)
    
    # Assigning a type to the variable 'result' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'result', apply_along_axis_call_result_578725)
    # SSA join for if statement (line 95)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to fix_invalid(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'result' (line 103)
    result_578728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'result', False)
    # Processing the call keyword arguments (line 103)
    # Getting the type of 'False' (line 103)
    False_578729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 'False', False)
    keyword_578730 = False_578729
    kwargs_578731 = {'copy': keyword_578730}
    # Getting the type of 'ma' (line 103)
    ma_578726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'ma', False)
    # Obtaining the member 'fix_invalid' of a type (line 103)
    fix_invalid_578727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 11), ma_578726, 'fix_invalid')
    # Calling fix_invalid(args, kwargs) (line 103)
    fix_invalid_call_result_578732 = invoke(stypy.reporting.localization.Localization(__file__, 103, 11), fix_invalid_578727, *[result_578728], **kwargs_578731)
    
    # Assigning a type to the variable 'stypy_return_type' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type', fix_invalid_call_result_578732)
    
    # ################# End of 'hdquantiles(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hdquantiles' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_578733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_578733)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hdquantiles'
    return stypy_return_type_578733

# Assigning a type to the variable 'hdquantiles' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'hdquantiles', hdquantiles)

@norecursion
def hdmedian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_578734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'int')
    # Getting the type of 'False' (line 106)
    False_578735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 32), 'False')
    defaults = [int_578734, False_578735]
    # Create a new context for function 'hdmedian'
    module_type_store = module_type_store.open_function_context('hdmedian', 106, 0, False)
    
    # Passed parameters checking function
    hdmedian.stypy_localization = localization
    hdmedian.stypy_type_of_self = None
    hdmedian.stypy_type_store = module_type_store
    hdmedian.stypy_function_name = 'hdmedian'
    hdmedian.stypy_param_names_list = ['data', 'axis', 'var']
    hdmedian.stypy_varargs_param_name = None
    hdmedian.stypy_kwargs_param_name = None
    hdmedian.stypy_call_defaults = defaults
    hdmedian.stypy_call_varargs = varargs
    hdmedian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hdmedian', ['data', 'axis', 'var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hdmedian', localization, ['data', 'axis', 'var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hdmedian(...)' code ##################

    str_578736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, (-1)), 'str', '\n    Returns the Harrell-Davis estimate of the median along the given axis.\n\n    Parameters\n    ----------\n    data : ndarray\n        Data array.\n    axis : int, optional\n        Axis along which to compute the quantiles. If None, use a flattened\n        array.\n    var : bool, optional\n        Whether to return the variance of the estimate.\n\n    Returns\n    -------\n    hdmedian : MaskedArray\n        The median values.  If ``var=True``, the variance is returned inside\n        the masked array.  E.g. for a 1-D array the shape change from (1,) to\n        (2,).\n\n    ')
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to hdquantiles(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'data' (line 128)
    data_578738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'data', False)
    
    # Obtaining an instance of the builtin type 'list' (line 128)
    list_578739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 128)
    # Adding element type (line 128)
    float_578740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 30), list_578739, float_578740)
    
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'axis' (line 128)
    axis_578741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'axis', False)
    keyword_578742 = axis_578741
    # Getting the type of 'var' (line 128)
    var_578743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 52), 'var', False)
    keyword_578744 = var_578743
    kwargs_578745 = {'var': keyword_578744, 'axis': keyword_578742}
    # Getting the type of 'hdquantiles' (line 128)
    hdquantiles_578737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'hdquantiles', False)
    # Calling hdquantiles(args, kwargs) (line 128)
    hdquantiles_call_result_578746 = invoke(stypy.reporting.localization.Localization(__file__, 128, 13), hdquantiles_578737, *[data_578738, list_578739], **kwargs_578745)
    
    # Assigning a type to the variable 'result' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'result', hdquantiles_call_result_578746)
    
    # Call to squeeze(...): (line 129)
    # Processing the call keyword arguments (line 129)
    kwargs_578749 = {}
    # Getting the type of 'result' (line 129)
    result_578747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'result', False)
    # Obtaining the member 'squeeze' of a type (line 129)
    squeeze_578748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 11), result_578747, 'squeeze')
    # Calling squeeze(args, kwargs) (line 129)
    squeeze_call_result_578750 = invoke(stypy.reporting.localization.Localization(__file__, 129, 11), squeeze_578748, *[], **kwargs_578749)
    
    # Assigning a type to the variable 'stypy_return_type' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type', squeeze_call_result_578750)
    
    # ################# End of 'hdmedian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hdmedian' in the type store
    # Getting the type of 'stypy_return_type' (line 106)
    stypy_return_type_578751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_578751)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hdmedian'
    return stypy_return_type_578751

# Assigning a type to the variable 'hdmedian' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'hdmedian', hdmedian)

@norecursion
def hdquantiles_sd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Call to list(...): (line 132)
    # Processing the call arguments (line 132)
    
    # Obtaining an instance of the builtin type 'list' (line 132)
    list_578753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 132)
    # Adding element type (line 132)
    float_578754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 36), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 35), list_578753, float_578754)
    # Adding element type (line 132)
    float_578755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 35), list_578753, float_578755)
    # Adding element type (line 132)
    float_578756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 35), list_578753, float_578756)
    
    # Processing the call keyword arguments (line 132)
    kwargs_578757 = {}
    # Getting the type of 'list' (line 132)
    list_578752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 30), 'list', False)
    # Calling list(args, kwargs) (line 132)
    list_call_result_578758 = invoke(stypy.reporting.localization.Localization(__file__, 132, 30), list_578752, *[list_578753], **kwargs_578757)
    
    # Getting the type of 'None' (line 132)
    None_578759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 55), 'None')
    defaults = [list_call_result_578758, None_578759]
    # Create a new context for function 'hdquantiles_sd'
    module_type_store = module_type_store.open_function_context('hdquantiles_sd', 132, 0, False)
    
    # Passed parameters checking function
    hdquantiles_sd.stypy_localization = localization
    hdquantiles_sd.stypy_type_of_self = None
    hdquantiles_sd.stypy_type_store = module_type_store
    hdquantiles_sd.stypy_function_name = 'hdquantiles_sd'
    hdquantiles_sd.stypy_param_names_list = ['data', 'prob', 'axis']
    hdquantiles_sd.stypy_varargs_param_name = None
    hdquantiles_sd.stypy_kwargs_param_name = None
    hdquantiles_sd.stypy_call_defaults = defaults
    hdquantiles_sd.stypy_call_varargs = varargs
    hdquantiles_sd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hdquantiles_sd', ['data', 'prob', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hdquantiles_sd', localization, ['data', 'prob', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hdquantiles_sd(...)' code ##################

    str_578760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, (-1)), 'str', '\n    The standard error of the Harrell-Davis quantile estimates by jackknife.\n\n    Parameters\n    ----------\n    data : array_like\n        Data array.\n    prob : sequence, optional\n        Sequence of quantiles to compute.\n    axis : int, optional\n        Axis along which to compute the quantiles. If None, use a flattened\n        array.\n\n    Returns\n    -------\n    hdquantiles_sd : MaskedArray\n        Standard error of the Harrell-Davis quantile estimates.\n\n    See Also\n    --------\n    hdquantiles\n\n    ')

    @norecursion
    def _hdsd_1D(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_hdsd_1D'
        module_type_store = module_type_store.open_function_context('_hdsd_1D', 156, 4, False)
        
        # Passed parameters checking function
        _hdsd_1D.stypy_localization = localization
        _hdsd_1D.stypy_type_of_self = None
        _hdsd_1D.stypy_type_store = module_type_store
        _hdsd_1D.stypy_function_name = '_hdsd_1D'
        _hdsd_1D.stypy_param_names_list = ['data', 'prob']
        _hdsd_1D.stypy_varargs_param_name = None
        _hdsd_1D.stypy_kwargs_param_name = None
        _hdsd_1D.stypy_call_defaults = defaults
        _hdsd_1D.stypy_call_varargs = varargs
        _hdsd_1D.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_hdsd_1D', ['data', 'prob'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_hdsd_1D', localization, ['data', 'prob'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_hdsd_1D(...)' code ##################

        str_578761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'str', 'Computes the std error for 1D arrays.')
        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to sort(...): (line 158)
        # Processing the call arguments (line 158)
        
        # Call to compressed(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_578766 = {}
        # Getting the type of 'data' (line 158)
        data_578764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 26), 'data', False)
        # Obtaining the member 'compressed' of a type (line 158)
        compressed_578765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 26), data_578764, 'compressed')
        # Calling compressed(args, kwargs) (line 158)
        compressed_call_result_578767 = invoke(stypy.reporting.localization.Localization(__file__, 158, 26), compressed_578765, *[], **kwargs_578766)
        
        # Processing the call keyword arguments (line 158)
        kwargs_578768 = {}
        # Getting the type of 'np' (line 158)
        np_578762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 'np', False)
        # Obtaining the member 'sort' of a type (line 158)
        sort_578763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 18), np_578762, 'sort')
        # Calling sort(args, kwargs) (line 158)
        sort_call_result_578769 = invoke(stypy.reporting.localization.Localization(__file__, 158, 18), sort_578763, *[compressed_call_result_578767], **kwargs_578768)
        
        # Assigning a type to the variable 'xsorted' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'xsorted', sort_call_result_578769)
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to len(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'xsorted' (line 159)
        xsorted_578771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'xsorted', False)
        # Processing the call keyword arguments (line 159)
        kwargs_578772 = {}
        # Getting the type of 'len' (line 159)
        len_578770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'len', False)
        # Calling len(args, kwargs) (line 159)
        len_call_result_578773 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), len_578770, *[xsorted_578771], **kwargs_578772)
        
        # Assigning a type to the variable 'n' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'n', len_call_result_578773)
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to empty(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Call to len(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'prob' (line 161)
        prob_578777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'prob', False)
        # Processing the call keyword arguments (line 161)
        kwargs_578778 = {}
        # Getting the type of 'len' (line 161)
        len_578776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'len', False)
        # Calling len(args, kwargs) (line 161)
        len_call_result_578779 = invoke(stypy.reporting.localization.Localization(__file__, 161, 24), len_578776, *[prob_578777], **kwargs_578778)
        
        # Getting the type of 'float_' (line 161)
        float__578780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'float_', False)
        # Processing the call keyword arguments (line 161)
        kwargs_578781 = {}
        # Getting the type of 'np' (line 161)
        np_578774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'np', False)
        # Obtaining the member 'empty' of a type (line 161)
        empty_578775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 15), np_578774, 'empty')
        # Calling empty(args, kwargs) (line 161)
        empty_call_result_578782 = invoke(stypy.reporting.localization.Localization(__file__, 161, 15), empty_578775, *[len_call_result_578779, float__578780], **kwargs_578781)
        
        # Assigning a type to the variable 'hdsd' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'hdsd', empty_call_result_578782)
        
        
        # Getting the type of 'n' (line 162)
        n_578783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'n')
        int_578784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 15), 'int')
        # Applying the binary operator '<' (line 162)
        result_lt_578785 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 11), '<', n_578783, int_578784)
        
        # Testing the type of an if condition (line 162)
        if_condition_578786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 8), result_lt_578785)
        # Assigning a type to the variable 'if_condition_578786' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'if_condition_578786', if_condition_578786)
        # SSA begins for if statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 163):
        
        # Assigning a Attribute to a Attribute (line 163):
        # Getting the type of 'np' (line 163)
        np_578787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), 'np')
        # Obtaining the member 'nan' of a type (line 163)
        nan_578788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 24), np_578787, 'nan')
        # Getting the type of 'hdsd' (line 163)
        hdsd_578789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'hdsd')
        # Setting the type of the member 'flat' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), hdsd_578789, 'flat', nan_578788)
        # SSA join for if statement (line 162)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 165):
        
        # Assigning a BinOp to a Name (line 165):
        
        # Call to arange(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'n' (line 165)
        n_578792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'n', False)
        # Processing the call keyword arguments (line 165)
        kwargs_578793 = {}
        # Getting the type of 'np' (line 165)
        np_578790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'np', False)
        # Obtaining the member 'arange' of a type (line 165)
        arange_578791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 13), np_578790, 'arange')
        # Calling arange(args, kwargs) (line 165)
        arange_call_result_578794 = invoke(stypy.reporting.localization.Localization(__file__, 165, 13), arange_578791, *[n_578792], **kwargs_578793)
        
        
        # Call to float(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'n' (line 165)
        n_578796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 34), 'n', False)
        int_578797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 36), 'int')
        # Applying the binary operator '-' (line 165)
        result_sub_578798 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 34), '-', n_578796, int_578797)
        
        # Processing the call keyword arguments (line 165)
        kwargs_578799 = {}
        # Getting the type of 'float' (line 165)
        float_578795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'float', False)
        # Calling float(args, kwargs) (line 165)
        float_call_result_578800 = invoke(stypy.reporting.localization.Localization(__file__, 165, 28), float_578795, *[result_sub_578798], **kwargs_578799)
        
        # Applying the binary operator 'div' (line 165)
        result_div_578801 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 13), 'div', arange_call_result_578794, float_call_result_578800)
        
        # Assigning a type to the variable 'vv' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'vv', result_div_578801)
        
        # Assigning a Attribute to a Name (line 166):
        
        # Assigning a Attribute to a Name (line 166):
        # Getting the type of 'beta' (line 166)
        beta_578802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 18), 'beta')
        # Obtaining the member 'cdf' of a type (line 166)
        cdf_578803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 18), beta_578802, 'cdf')
        # Assigning a type to the variable 'betacdf' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'betacdf', cdf_578803)
        
        
        # Call to enumerate(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'prob' (line 168)
        prob_578805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'prob', False)
        # Processing the call keyword arguments (line 168)
        kwargs_578806 = {}
        # Getting the type of 'enumerate' (line 168)
        enumerate_578804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 168)
        enumerate_call_result_578807 = invoke(stypy.reporting.localization.Localization(__file__, 168, 21), enumerate_578804, *[prob_578805], **kwargs_578806)
        
        # Testing the type of a for loop iterable (line 168)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 168, 8), enumerate_call_result_578807)
        # Getting the type of the for loop variable (line 168)
        for_loop_var_578808 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 168, 8), enumerate_call_result_578807)
        # Assigning a type to the variable 'i' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), for_loop_var_578808))
        # Assigning a type to the variable 'p' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), for_loop_var_578808))
        # SSA begins for a for statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to betacdf(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'vv' (line 169)
        vv_578810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 25), 'vv', False)
        # Getting the type of 'n' (line 169)
        n_578811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 30), 'n', False)
        int_578812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 32), 'int')
        # Applying the binary operator '+' (line 169)
        result_add_578813 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 30), '+', n_578811, int_578812)
        
        # Getting the type of 'p' (line 169)
        p_578814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'p', False)
        # Applying the binary operator '*' (line 169)
        result_mul_578815 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 29), '*', result_add_578813, p_578814)
        
        # Getting the type of 'n' (line 169)
        n_578816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 39), 'n', False)
        int_578817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 41), 'int')
        # Applying the binary operator '+' (line 169)
        result_add_578818 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 39), '+', n_578816, int_578817)
        
        int_578819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 45), 'int')
        # Getting the type of 'p' (line 169)
        p_578820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 47), 'p', False)
        # Applying the binary operator '-' (line 169)
        result_sub_578821 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 45), '-', int_578819, p_578820)
        
        # Applying the binary operator '*' (line 169)
        result_mul_578822 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 38), '*', result_add_578818, result_sub_578821)
        
        # Processing the call keyword arguments (line 169)
        kwargs_578823 = {}
        # Getting the type of 'betacdf' (line 169)
        betacdf_578809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 17), 'betacdf', False)
        # Calling betacdf(args, kwargs) (line 169)
        betacdf_call_result_578824 = invoke(stypy.reporting.localization.Localization(__file__, 169, 17), betacdf_578809, *[vv_578810, result_mul_578815, result_mul_578822], **kwargs_578823)
        
        # Assigning a type to the variable '_w' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), '_w', betacdf_call_result_578824)
        
        # Assigning a BinOp to a Name (line 170):
        
        # Assigning a BinOp to a Name (line 170):
        
        # Obtaining the type of the subscript
        int_578825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 19), 'int')
        slice_578826 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 170, 16), int_578825, None, None)
        # Getting the type of '_w' (line 170)
        _w_578827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), '_w')
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___578828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), _w_578827, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_578829 = invoke(stypy.reporting.localization.Localization(__file__, 170, 16), getitem___578828, slice_578826)
        
        
        # Obtaining the type of the subscript
        int_578830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 29), 'int')
        slice_578831 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 170, 25), None, int_578830, None)
        # Getting the type of '_w' (line 170)
        _w_578832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 25), '_w')
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___578833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 25), _w_578832, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_578834 = invoke(stypy.reporting.localization.Localization(__file__, 170, 25), getitem___578833, slice_578831)
        
        # Applying the binary operator '-' (line 170)
        result_sub_578835 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 16), '-', subscript_call_result_578829, subscript_call_result_578834)
        
        # Assigning a type to the variable 'w' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'w', result_sub_578835)
        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to fromiter(...): (line 171)
        # Processing the call arguments (line 171)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'n' (line 173)
        n_578874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 49), 'n', False)
        # Processing the call keyword arguments (line 173)
        kwargs_578875 = {}
        # Getting the type of 'range' (line 173)
        range_578873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 43), 'range', False)
        # Calling range(args, kwargs) (line 173)
        range_call_result_578876 = invoke(stypy.reporting.localization.Localization(__file__, 173, 43), range_578873, *[n_578874], **kwargs_578875)
        
        comprehension_578877 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 31), range_call_result_578876)
        # Assigning a type to the variable 'k' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'k', comprehension_578877)
        
        # Call to dot(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'w' (line 171)
        w_578840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 38), 'w', False)
        
        # Obtaining the type of the subscript
        
        # Call to astype(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'int_' (line 172)
        int__578865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 81), 'int_', False)
        # Processing the call keyword arguments (line 171)
        kwargs_578866 = {}
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 171)
        tuple_578841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 171)
        # Adding element type (line 171)
        
        # Call to list(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Call to range(...): (line 171)
        # Processing the call arguments (line 171)
        int_578844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 65), 'int')
        # Getting the type of 'k' (line 171)
        k_578845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 67), 'k', False)
        # Processing the call keyword arguments (line 171)
        kwargs_578846 = {}
        # Getting the type of 'range' (line 171)
        range_578843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 59), 'range', False)
        # Calling range(args, kwargs) (line 171)
        range_call_result_578847 = invoke(stypy.reporting.localization.Localization(__file__, 171, 59), range_578843, *[int_578844, k_578845], **kwargs_578846)
        
        # Processing the call keyword arguments (line 171)
        kwargs_578848 = {}
        # Getting the type of 'list' (line 171)
        list_578842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 54), 'list', False)
        # Calling list(args, kwargs) (line 171)
        list_call_result_578849 = invoke(stypy.reporting.localization.Localization(__file__, 171, 54), list_578842, *[range_call_result_578847], **kwargs_578848)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 54), tuple_578841, list_call_result_578849)
        # Adding element type (line 171)
        
        # Call to list(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Call to range(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'k' (line 172)
        k_578852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 65), 'k', False)
        int_578853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 67), 'int')
        # Applying the binary operator '+' (line 172)
        result_add_578854 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 65), '+', k_578852, int_578853)
        
        # Getting the type of 'n' (line 172)
        n_578855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 69), 'n', False)
        # Processing the call keyword arguments (line 172)
        kwargs_578856 = {}
        # Getting the type of 'range' (line 172)
        range_578851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 59), 'range', False)
        # Calling range(args, kwargs) (line 172)
        range_call_result_578857 = invoke(stypy.reporting.localization.Localization(__file__, 172, 59), range_578851, *[result_add_578854, n_578855], **kwargs_578856)
        
        # Processing the call keyword arguments (line 172)
        kwargs_578858 = {}
        # Getting the type of 'list' (line 172)
        list_578850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 54), 'list', False)
        # Calling list(args, kwargs) (line 172)
        list_call_result_578859 = invoke(stypy.reporting.localization.Localization(__file__, 172, 54), list_578850, *[range_call_result_578857], **kwargs_578858)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 54), tuple_578841, list_call_result_578859)
        
        # Getting the type of 'np' (line 171)
        np_578860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 48), 'np', False)
        # Obtaining the member 'r_' of a type (line 171)
        r__578861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 48), np_578860, 'r_')
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___578862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 48), r__578861, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_578863 = invoke(stypy.reporting.localization.Localization(__file__, 171, 48), getitem___578862, tuple_578841)
        
        # Obtaining the member 'astype' of a type (line 171)
        astype_578864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 48), subscript_call_result_578863, 'astype')
        # Calling astype(args, kwargs) (line 171)
        astype_call_result_578867 = invoke(stypy.reporting.localization.Localization(__file__, 171, 48), astype_578864, *[int__578865], **kwargs_578866)
        
        # Getting the type of 'xsorted' (line 171)
        xsorted_578868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 40), 'xsorted', False)
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___578869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 40), xsorted_578868, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_578870 = invoke(stypy.reporting.localization.Localization(__file__, 171, 40), getitem___578869, astype_call_result_578867)
        
        # Processing the call keyword arguments (line 171)
        kwargs_578871 = {}
        # Getting the type of 'np' (line 171)
        np_578838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'np', False)
        # Obtaining the member 'dot' of a type (line 171)
        dot_578839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 31), np_578838, 'dot')
        # Calling dot(args, kwargs) (line 171)
        dot_call_result_578872 = invoke(stypy.reporting.localization.Localization(__file__, 171, 31), dot_578839, *[w_578840, subscript_call_result_578870], **kwargs_578871)
        
        list_578878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 31), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 31), list_578878, dot_call_result_578872)
        # Processing the call keyword arguments (line 171)
        # Getting the type of 'float_' (line 173)
        float__578879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 60), 'float_', False)
        keyword_578880 = float__578879
        kwargs_578881 = {'dtype': keyword_578880}
        # Getting the type of 'np' (line 171)
        np_578836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'np', False)
        # Obtaining the member 'fromiter' of a type (line 171)
        fromiter_578837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 18), np_578836, 'fromiter')
        # Calling fromiter(args, kwargs) (line 171)
        fromiter_call_result_578882 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), fromiter_578837, *[list_578878], **kwargs_578881)
        
        # Assigning a type to the variable 'mx_' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'mx_', fromiter_call_result_578882)
        
        # Assigning a BinOp to a Name (line 174):
        
        # Assigning a BinOp to a Name (line 174):
        
        # Call to array(...): (line 174)
        # Processing the call arguments (line 174)
        
        # Call to var(...): (line 174)
        # Processing the call keyword arguments (line 174)
        kwargs_578887 = {}
        # Getting the type of 'mx_' (line 174)
        mx__578885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 30), 'mx_', False)
        # Obtaining the member 'var' of a type (line 174)
        var_578886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 30), mx__578885, 'var')
        # Calling var(args, kwargs) (line 174)
        var_call_result_578888 = invoke(stypy.reporting.localization.Localization(__file__, 174, 30), var_578886, *[], **kwargs_578887)
        
        # Processing the call keyword arguments (line 174)
        # Getting the type of 'False' (line 174)
        False_578889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 46), 'False', False)
        keyword_578890 = False_578889
        int_578891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 59), 'int')
        keyword_578892 = int_578891
        kwargs_578893 = {'copy': keyword_578890, 'ndmin': keyword_578892}
        # Getting the type of 'np' (line 174)
        np_578883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 21), 'np', False)
        # Obtaining the member 'array' of a type (line 174)
        array_578884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 21), np_578883, 'array')
        # Calling array(args, kwargs) (line 174)
        array_call_result_578894 = invoke(stypy.reporting.localization.Localization(__file__, 174, 21), array_578884, *[var_call_result_578888], **kwargs_578893)
        
        # Getting the type of 'n' (line 174)
        n_578895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 64), 'n')
        # Applying the binary operator '*' (line 174)
        result_mul_578896 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 21), '*', array_call_result_578894, n_578895)
        
        
        # Call to float(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'n' (line 174)
        n_578898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 74), 'n', False)
        int_578899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 76), 'int')
        # Applying the binary operator '-' (line 174)
        result_sub_578900 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 74), '-', n_578898, int_578899)
        
        # Processing the call keyword arguments (line 174)
        kwargs_578901 = {}
        # Getting the type of 'float' (line 174)
        float_578897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 68), 'float', False)
        # Calling float(args, kwargs) (line 174)
        float_call_result_578902 = invoke(stypy.reporting.localization.Localization(__file__, 174, 68), float_578897, *[result_sub_578900], **kwargs_578901)
        
        # Applying the binary operator 'div' (line 174)
        result_div_578903 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 66), 'div', result_mul_578896, float_call_result_578902)
        
        # Assigning a type to the variable 'mx_var' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'mx_var', result_div_578903)
        
        # Assigning a BinOp to a Subscript (line 175):
        
        # Assigning a BinOp to a Subscript (line 175):
        
        # Call to float(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'n' (line 175)
        n_578905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 28), 'n', False)
        int_578906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 30), 'int')
        # Applying the binary operator '-' (line 175)
        result_sub_578907 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 28), '-', n_578905, int_578906)
        
        # Processing the call keyword arguments (line 175)
        kwargs_578908 = {}
        # Getting the type of 'float' (line 175)
        float_578904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 22), 'float', False)
        # Calling float(args, kwargs) (line 175)
        float_call_result_578909 = invoke(stypy.reporting.localization.Localization(__file__, 175, 22), float_578904, *[result_sub_578907], **kwargs_578908)
        
        
        # Call to sqrt(...): (line 175)
        # Processing the call arguments (line 175)
        
        # Call to diagonal(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_578918 = {}
        
        # Call to diag(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'mx_var' (line 175)
        mx_var_578914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 51), 'mx_var', False)
        # Processing the call keyword arguments (line 175)
        kwargs_578915 = {}
        # Getting the type of 'np' (line 175)
        np_578912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 43), 'np', False)
        # Obtaining the member 'diag' of a type (line 175)
        diag_578913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 43), np_578912, 'diag')
        # Calling diag(args, kwargs) (line 175)
        diag_call_result_578916 = invoke(stypy.reporting.localization.Localization(__file__, 175, 43), diag_578913, *[mx_var_578914], **kwargs_578915)
        
        # Obtaining the member 'diagonal' of a type (line 175)
        diagonal_578917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 43), diag_call_result_578916, 'diagonal')
        # Calling diagonal(args, kwargs) (line 175)
        diagonal_call_result_578919 = invoke(stypy.reporting.localization.Localization(__file__, 175, 43), diagonal_578917, *[], **kwargs_578918)
        
        
        # Call to float(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'n' (line 175)
        n_578921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 78), 'n', False)
        # Processing the call keyword arguments (line 175)
        kwargs_578922 = {}
        # Getting the type of 'float' (line 175)
        float_578920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 72), 'float', False)
        # Calling float(args, kwargs) (line 175)
        float_call_result_578923 = invoke(stypy.reporting.localization.Localization(__file__, 175, 72), float_578920, *[n_578921], **kwargs_578922)
        
        # Applying the binary operator 'div' (line 175)
        result_div_578924 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 43), 'div', diagonal_call_result_578919, float_call_result_578923)
        
        # Processing the call keyword arguments (line 175)
        kwargs_578925 = {}
        # Getting the type of 'np' (line 175)
        np_578910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 35), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 175)
        sqrt_578911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 35), np_578910, 'sqrt')
        # Calling sqrt(args, kwargs) (line 175)
        sqrt_call_result_578926 = invoke(stypy.reporting.localization.Localization(__file__, 175, 35), sqrt_578911, *[result_div_578924], **kwargs_578925)
        
        # Applying the binary operator '*' (line 175)
        result_mul_578927 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 22), '*', float_call_result_578909, sqrt_call_result_578926)
        
        # Getting the type of 'hdsd' (line 175)
        hdsd_578928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'hdsd')
        # Getting the type of 'i' (line 175)
        i_578929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'i')
        # Storing an element on a container (line 175)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 12), hdsd_578928, (i_578929, result_mul_578927))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'hdsd' (line 176)
        hdsd_578930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'hdsd')
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', hdsd_578930)
        
        # ################# End of '_hdsd_1D(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_hdsd_1D' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_578931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_578931)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_hdsd_1D'
        return stypy_return_type_578931

    # Assigning a type to the variable '_hdsd_1D' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), '_hdsd_1D', _hdsd_1D)
    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 179):
    
    # Call to array(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'data' (line 179)
    data_578934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'data', False)
    # Processing the call keyword arguments (line 179)
    # Getting the type of 'False' (line 179)
    False_578935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'False', False)
    keyword_578936 = False_578935
    # Getting the type of 'float_' (line 179)
    float__578937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 44), 'float_', False)
    keyword_578938 = float__578937
    kwargs_578939 = {'dtype': keyword_578938, 'copy': keyword_578936}
    # Getting the type of 'ma' (line 179)
    ma_578932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 11), 'ma', False)
    # Obtaining the member 'array' of a type (line 179)
    array_578933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 11), ma_578932, 'array')
    # Calling array(args, kwargs) (line 179)
    array_call_result_578940 = invoke(stypy.reporting.localization.Localization(__file__, 179, 11), array_578933, *[data_578934], **kwargs_578939)
    
    # Assigning a type to the variable 'data' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'data', array_call_result_578940)
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to array(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'prob' (line 180)
    prob_578943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'prob', False)
    # Processing the call keyword arguments (line 180)
    # Getting the type of 'False' (line 180)
    False_578944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'False', False)
    keyword_578945 = False_578944
    int_578946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 41), 'int')
    keyword_578947 = int_578946
    kwargs_578948 = {'copy': keyword_578945, 'ndmin': keyword_578947}
    # Getting the type of 'np' (line 180)
    np_578941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 180)
    array_578942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), np_578941, 'array')
    # Calling array(args, kwargs) (line 180)
    array_call_result_578949 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), array_578942, *[prob_578943], **kwargs_578948)
    
    # Assigning a type to the variable 'p' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'p', array_call_result_578949)
    
    # Type idiom detected: calculating its left and rigth part (line 182)
    # Getting the type of 'axis' (line 182)
    axis_578950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'axis')
    # Getting the type of 'None' (line 182)
    None_578951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'None')
    
    (may_be_578952, more_types_in_union_578953) = may_be_none(axis_578950, None_578951)

    if may_be_578952:

        if more_types_in_union_578953:
            # Runtime conditional SSA (line 182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to _hdsd_1D(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'data' (line 183)
        data_578955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 26), 'data', False)
        # Getting the type of 'p' (line 183)
        p_578956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 'p', False)
        # Processing the call keyword arguments (line 183)
        kwargs_578957 = {}
        # Getting the type of '_hdsd_1D' (line 183)
        _hdsd_1D_578954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), '_hdsd_1D', False)
        # Calling _hdsd_1D(args, kwargs) (line 183)
        _hdsd_1D_call_result_578958 = invoke(stypy.reporting.localization.Localization(__file__, 183, 17), _hdsd_1D_578954, *[data_578955, p_578956], **kwargs_578957)
        
        # Assigning a type to the variable 'result' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'result', _hdsd_1D_call_result_578958)

        if more_types_in_union_578953:
            # Runtime conditional SSA for else branch (line 182)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_578952) or more_types_in_union_578953):
        
        
        # Getting the type of 'data' (line 185)
        data_578959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'data')
        # Obtaining the member 'ndim' of a type (line 185)
        ndim_578960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 11), data_578959, 'ndim')
        int_578961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 23), 'int')
        # Applying the binary operator '>' (line 185)
        result_gt_578962 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 11), '>', ndim_578960, int_578961)
        
        # Testing the type of an if condition (line 185)
        if_condition_578963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 8), result_gt_578962)
        # Assigning a type to the variable 'if_condition_578963' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'if_condition_578963', if_condition_578963)
        # SSA begins for if statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 186)
        # Processing the call arguments (line 186)
        str_578965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 29), 'str', "Array 'data' must be at most two dimensional, but got data.ndim = %d")
        # Getting the type of 'data' (line 187)
        data_578966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 56), 'data', False)
        # Obtaining the member 'ndim' of a type (line 187)
        ndim_578967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 56), data_578966, 'ndim')
        # Applying the binary operator '%' (line 186)
        result_mod_578968 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 29), '%', str_578965, ndim_578967)
        
        # Processing the call keyword arguments (line 186)
        kwargs_578969 = {}
        # Getting the type of 'ValueError' (line 186)
        ValueError_578964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 186)
        ValueError_call_result_578970 = invoke(stypy.reporting.localization.Localization(__file__, 186, 18), ValueError_578964, *[result_mod_578968], **kwargs_578969)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 186, 12), ValueError_call_result_578970, 'raise parameter', BaseException)
        # SSA join for if statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 188):
        
        # Assigning a Call to a Name (line 188):
        
        # Call to apply_along_axis(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of '_hdsd_1D' (line 188)
        _hdsd_1D_578973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 37), '_hdsd_1D', False)
        # Getting the type of 'axis' (line 188)
        axis_578974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 47), 'axis', False)
        # Getting the type of 'data' (line 188)
        data_578975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 53), 'data', False)
        # Getting the type of 'p' (line 188)
        p_578976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 59), 'p', False)
        # Processing the call keyword arguments (line 188)
        kwargs_578977 = {}
        # Getting the type of 'ma' (line 188)
        ma_578971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 17), 'ma', False)
        # Obtaining the member 'apply_along_axis' of a type (line 188)
        apply_along_axis_578972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 17), ma_578971, 'apply_along_axis')
        # Calling apply_along_axis(args, kwargs) (line 188)
        apply_along_axis_call_result_578978 = invoke(stypy.reporting.localization.Localization(__file__, 188, 17), apply_along_axis_578972, *[_hdsd_1D_578973, axis_578974, data_578975, p_578976], **kwargs_578977)
        
        # Assigning a type to the variable 'result' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'result', apply_along_axis_call_result_578978)

        if (may_be_578952 and more_types_in_union_578953):
            # SSA join for if statement (line 182)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to ravel(...): (line 190)
    # Processing the call keyword arguments (line 190)
    kwargs_578987 = {}
    
    # Call to fix_invalid(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'result' (line 190)
    result_578981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 26), 'result', False)
    # Processing the call keyword arguments (line 190)
    # Getting the type of 'False' (line 190)
    False_578982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 39), 'False', False)
    keyword_578983 = False_578982
    kwargs_578984 = {'copy': keyword_578983}
    # Getting the type of 'ma' (line 190)
    ma_578979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'ma', False)
    # Obtaining the member 'fix_invalid' of a type (line 190)
    fix_invalid_578980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), ma_578979, 'fix_invalid')
    # Calling fix_invalid(args, kwargs) (line 190)
    fix_invalid_call_result_578985 = invoke(stypy.reporting.localization.Localization(__file__, 190, 11), fix_invalid_578980, *[result_578981], **kwargs_578984)
    
    # Obtaining the member 'ravel' of a type (line 190)
    ravel_578986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), fix_invalid_call_result_578985, 'ravel')
    # Calling ravel(args, kwargs) (line 190)
    ravel_call_result_578988 = invoke(stypy.reporting.localization.Localization(__file__, 190, 11), ravel_578986, *[], **kwargs_578987)
    
    # Assigning a type to the variable 'stypy_return_type' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type', ravel_call_result_578988)
    
    # ################# End of 'hdquantiles_sd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hdquantiles_sd' in the type store
    # Getting the type of 'stypy_return_type' (line 132)
    stypy_return_type_578989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_578989)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hdquantiles_sd'
    return stypy_return_type_578989

# Assigning a type to the variable 'hdquantiles_sd' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'hdquantiles_sd', hdquantiles_sd)

@norecursion
def trimmed_mean_ci(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 193)
    tuple_578990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 193)
    # Adding element type (line 193)
    float_578991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 34), tuple_578990, float_578991)
    # Adding element type (line 193)
    float_578992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 34), tuple_578990, float_578992)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 193)
    tuple_578993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 55), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 193)
    # Adding element type (line 193)
    # Getting the type of 'True' (line 193)
    True_578994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 55), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 55), tuple_578993, True_578994)
    # Adding element type (line 193)
    # Getting the type of 'True' (line 193)
    True_578995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 60), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 55), tuple_578993, True_578995)
    
    float_578996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 26), 'float')
    # Getting the type of 'None' (line 194)
    None_578997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 37), 'None')
    defaults = [tuple_578990, tuple_578993, float_578996, None_578997]
    # Create a new context for function 'trimmed_mean_ci'
    module_type_store = module_type_store.open_function_context('trimmed_mean_ci', 193, 0, False)
    
    # Passed parameters checking function
    trimmed_mean_ci.stypy_localization = localization
    trimmed_mean_ci.stypy_type_of_self = None
    trimmed_mean_ci.stypy_type_store = module_type_store
    trimmed_mean_ci.stypy_function_name = 'trimmed_mean_ci'
    trimmed_mean_ci.stypy_param_names_list = ['data', 'limits', 'inclusive', 'alpha', 'axis']
    trimmed_mean_ci.stypy_varargs_param_name = None
    trimmed_mean_ci.stypy_kwargs_param_name = None
    trimmed_mean_ci.stypy_call_defaults = defaults
    trimmed_mean_ci.stypy_call_varargs = varargs
    trimmed_mean_ci.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'trimmed_mean_ci', ['data', 'limits', 'inclusive', 'alpha', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'trimmed_mean_ci', localization, ['data', 'limits', 'inclusive', 'alpha', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'trimmed_mean_ci(...)' code ##################

    str_578998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, (-1)), 'str', '\n    Selected confidence interval of the trimmed mean along the given axis.\n\n    Parameters\n    ----------\n    data : array_like\n        Input data.\n    limits : {None, tuple}, optional\n        None or a two item tuple.\n        Tuple of the percentages to cut on each side of the array, with respect\n        to the number of unmasked data, as floats between 0. and 1. If ``n``\n        is the number of unmasked data before trimming, then\n        (``n * limits[0]``)th smallest data and (``n * limits[1]``)th\n        largest data are masked.  The total number of unmasked data after\n        trimming is ``n * (1. - sum(limits))``.\n        The value of one limit can be set to None to indicate an open interval.\n\n        Defaults to (0.2, 0.2).\n    inclusive : (2,) tuple of boolean, optional\n        If relative==False, tuple indicating whether values exactly equal to\n        the absolute limits are allowed.\n        If relative==True, tuple indicating whether the number of data being\n        masked on each side should be rounded (True) or truncated (False).\n\n        Defaults to (True, True).\n    alpha : float, optional\n        Confidence level of the intervals.\n\n        Defaults to 0.05.\n    axis : int, optional\n        Axis along which to cut. If None, uses a flattened version of `data`.\n\n        Defaults to None.\n\n    Returns\n    -------\n    trimmed_mean_ci : (2,) ndarray\n        The lower and upper confidence intervals of the trimmed data.\n\n    ')
    
    # Assigning a Call to a Name (line 235):
    
    # Assigning a Call to a Name (line 235):
    
    # Call to array(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'data' (line 235)
    data_579001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'data', False)
    # Processing the call keyword arguments (line 235)
    # Getting the type of 'False' (line 235)
    False_579002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 31), 'False', False)
    keyword_579003 = False_579002
    kwargs_579004 = {'copy': keyword_579003}
    # Getting the type of 'ma' (line 235)
    ma_578999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'ma', False)
    # Obtaining the member 'array' of a type (line 235)
    array_579000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 11), ma_578999, 'array')
    # Calling array(args, kwargs) (line 235)
    array_call_result_579005 = invoke(stypy.reporting.localization.Localization(__file__, 235, 11), array_579000, *[data_579001], **kwargs_579004)
    
    # Assigning a type to the variable 'data' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'data', array_call_result_579005)
    
    # Assigning a Call to a Name (line 236):
    
    # Assigning a Call to a Name (line 236):
    
    # Call to trimr(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'data' (line 236)
    data_579008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 27), 'data', False)
    # Processing the call keyword arguments (line 236)
    # Getting the type of 'limits' (line 236)
    limits_579009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 40), 'limits', False)
    keyword_579010 = limits_579009
    # Getting the type of 'inclusive' (line 236)
    inclusive_579011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 58), 'inclusive', False)
    keyword_579012 = inclusive_579011
    # Getting the type of 'axis' (line 236)
    axis_579013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 74), 'axis', False)
    keyword_579014 = axis_579013
    kwargs_579015 = {'axis': keyword_579014, 'limits': keyword_579010, 'inclusive': keyword_579012}
    # Getting the type of 'mstats' (line 236)
    mstats_579006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 14), 'mstats', False)
    # Obtaining the member 'trimr' of a type (line 236)
    trimr_579007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 14), mstats_579006, 'trimr')
    # Calling trimr(args, kwargs) (line 236)
    trimr_call_result_579016 = invoke(stypy.reporting.localization.Localization(__file__, 236, 14), trimr_579007, *[data_579008], **kwargs_579015)
    
    # Assigning a type to the variable 'trimmed' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'trimmed', trimr_call_result_579016)
    
    # Assigning a Call to a Name (line 237):
    
    # Assigning a Call to a Name (line 237):
    
    # Call to mean(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'axis' (line 237)
    axis_579019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 25), 'axis', False)
    # Processing the call keyword arguments (line 237)
    kwargs_579020 = {}
    # Getting the type of 'trimmed' (line 237)
    trimmed_579017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'trimmed', False)
    # Obtaining the member 'mean' of a type (line 237)
    mean_579018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), trimmed_579017, 'mean')
    # Calling mean(args, kwargs) (line 237)
    mean_call_result_579021 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), mean_579018, *[axis_579019], **kwargs_579020)
    
    # Assigning a type to the variable 'tmean' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'tmean', mean_call_result_579021)
    
    # Assigning a Call to a Name (line 238):
    
    # Assigning a Call to a Name (line 238):
    
    # Call to trimmed_stde(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'data' (line 238)
    data_579024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 32), 'data', False)
    # Processing the call keyword arguments (line 238)
    # Getting the type of 'limits' (line 238)
    limits_579025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 44), 'limits', False)
    keyword_579026 = limits_579025
    # Getting the type of 'inclusive' (line 238)
    inclusive_579027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 61), 'inclusive', False)
    keyword_579028 = inclusive_579027
    # Getting the type of 'axis' (line 238)
    axis_579029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 76), 'axis', False)
    keyword_579030 = axis_579029
    kwargs_579031 = {'axis': keyword_579030, 'limits': keyword_579026, 'inclusive': keyword_579028}
    # Getting the type of 'mstats' (line 238)
    mstats_579022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'mstats', False)
    # Obtaining the member 'trimmed_stde' of a type (line 238)
    trimmed_stde_579023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), mstats_579022, 'trimmed_stde')
    # Calling trimmed_stde(args, kwargs) (line 238)
    trimmed_stde_call_result_579032 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), trimmed_stde_579023, *[data_579024], **kwargs_579031)
    
    # Assigning a type to the variable 'tstde' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'tstde', trimmed_stde_call_result_579032)
    
    # Assigning a BinOp to a Name (line 239):
    
    # Assigning a BinOp to a Name (line 239):
    
    # Call to count(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'axis' (line 239)
    axis_579035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'axis', False)
    # Processing the call keyword arguments (line 239)
    kwargs_579036 = {}
    # Getting the type of 'trimmed' (line 239)
    trimmed_579033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 9), 'trimmed', False)
    # Obtaining the member 'count' of a type (line 239)
    count_579034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 9), trimmed_579033, 'count')
    # Calling count(args, kwargs) (line 239)
    count_call_result_579037 = invoke(stypy.reporting.localization.Localization(__file__, 239, 9), count_579034, *[axis_579035], **kwargs_579036)
    
    int_579038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 31), 'int')
    # Applying the binary operator '-' (line 239)
    result_sub_579039 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 9), '-', count_call_result_579037, int_579038)
    
    # Assigning a type to the variable 'df' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'df', result_sub_579039)
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to ppf(...): (line 240)
    # Processing the call arguments (line 240)
    int_579042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 17), 'int')
    # Getting the type of 'alpha' (line 240)
    alpha_579043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'alpha', False)
    float_579044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 25), 'float')
    # Applying the binary operator 'div' (line 240)
    result_div_579045 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 19), 'div', alpha_579043, float_579044)
    
    # Applying the binary operator '-' (line 240)
    result_sub_579046 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 17), '-', int_579042, result_div_579045)
    
    # Getting the type of 'df' (line 240)
    df_579047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 28), 'df', False)
    # Processing the call keyword arguments (line 240)
    kwargs_579048 = {}
    # Getting the type of 't' (line 240)
    t_579040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 't', False)
    # Obtaining the member 'ppf' of a type (line 240)
    ppf_579041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), t_579040, 'ppf')
    # Calling ppf(args, kwargs) (line 240)
    ppf_call_result_579049 = invoke(stypy.reporting.localization.Localization(__file__, 240, 11), ppf_579041, *[result_sub_579046, df_579047], **kwargs_579048)
    
    # Assigning a type to the variable 'tppf' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tppf', ppf_call_result_579049)
    
    # Call to array(...): (line 241)
    # Processing the call arguments (line 241)
    
    # Obtaining an instance of the builtin type 'tuple' (line 241)
    tuple_579052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 241)
    # Adding element type (line 241)
    # Getting the type of 'tmean' (line 241)
    tmean_579053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'tmean', False)
    # Getting the type of 'tppf' (line 241)
    tppf_579054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 29), 'tppf', False)
    # Getting the type of 'tstde' (line 241)
    tstde_579055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 34), 'tstde', False)
    # Applying the binary operator '*' (line 241)
    result_mul_579056 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 29), '*', tppf_579054, tstde_579055)
    
    # Applying the binary operator '-' (line 241)
    result_sub_579057 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 21), '-', tmean_579053, result_mul_579056)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 21), tuple_579052, result_sub_579057)
    # Adding element type (line 241)
    # Getting the type of 'tmean' (line 241)
    tmean_579058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 41), 'tmean', False)
    # Getting the type of 'tppf' (line 241)
    tppf_579059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 47), 'tppf', False)
    # Getting the type of 'tstde' (line 241)
    tstde_579060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 52), 'tstde', False)
    # Applying the binary operator '*' (line 241)
    result_mul_579061 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 47), '*', tppf_579059, tstde_579060)
    
    # Applying the binary operator '+' (line 241)
    result_add_579062 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 41), '+', tmean_579058, result_mul_579061)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 21), tuple_579052, result_add_579062)
    
    # Processing the call keyword arguments (line 241)
    kwargs_579063 = {}
    # Getting the type of 'np' (line 241)
    np_579050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 241)
    array_579051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 11), np_579050, 'array')
    # Calling array(args, kwargs) (line 241)
    array_call_result_579064 = invoke(stypy.reporting.localization.Localization(__file__, 241, 11), array_579051, *[tuple_579052], **kwargs_579063)
    
    # Assigning a type to the variable 'stypy_return_type' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type', array_call_result_579064)
    
    # ################# End of 'trimmed_mean_ci(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'trimmed_mean_ci' in the type store
    # Getting the type of 'stypy_return_type' (line 193)
    stypy_return_type_579065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_579065)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'trimmed_mean_ci'
    return stypy_return_type_579065

# Assigning a type to the variable 'trimmed_mean_ci' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'trimmed_mean_ci', trimmed_mean_ci)

@norecursion
def mjci(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'list' (line 244)
    list_579066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 244)
    # Adding element type (line 244)
    float_579067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 20), list_579066, float_579067)
    # Adding element type (line 244)
    float_579068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 20), list_579066, float_579068)
    # Adding element type (line 244)
    float_579069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 20), list_579066, float_579069)
    
    # Getting the type of 'None' (line 244)
    None_579070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 42), 'None')
    defaults = [list_579066, None_579070]
    # Create a new context for function 'mjci'
    module_type_store = module_type_store.open_function_context('mjci', 244, 0, False)
    
    # Passed parameters checking function
    mjci.stypy_localization = localization
    mjci.stypy_type_of_self = None
    mjci.stypy_type_store = module_type_store
    mjci.stypy_function_name = 'mjci'
    mjci.stypy_param_names_list = ['data', 'prob', 'axis']
    mjci.stypy_varargs_param_name = None
    mjci.stypy_kwargs_param_name = None
    mjci.stypy_call_defaults = defaults
    mjci.stypy_call_varargs = varargs
    mjci.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mjci', ['data', 'prob', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mjci', localization, ['data', 'prob', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mjci(...)' code ##################

    str_579071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, (-1)), 'str', '\n    Returns the Maritz-Jarrett estimators of the standard error of selected\n    experimental quantiles of the data.\n\n    Parameters\n    ----------\n    data : ndarray\n        Data array.\n    prob : sequence, optional\n        Sequence of quantiles to compute.\n    axis : int or None, optional\n        Axis along which to compute the quantiles. If None, use a flattened\n        array.\n\n    ')

    @norecursion
    def _mjci_1D(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mjci_1D'
        module_type_store = module_type_store.open_function_context('_mjci_1D', 260, 4, False)
        
        # Passed parameters checking function
        _mjci_1D.stypy_localization = localization
        _mjci_1D.stypy_type_of_self = None
        _mjci_1D.stypy_type_store = module_type_store
        _mjci_1D.stypy_function_name = '_mjci_1D'
        _mjci_1D.stypy_param_names_list = ['data', 'p']
        _mjci_1D.stypy_varargs_param_name = None
        _mjci_1D.stypy_kwargs_param_name = None
        _mjci_1D.stypy_call_defaults = defaults
        _mjci_1D.stypy_call_varargs = varargs
        _mjci_1D.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_mjci_1D', ['data', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mjci_1D', localization, ['data', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mjci_1D(...)' code ##################

        
        # Assigning a Call to a Name (line 261):
        
        # Assigning a Call to a Name (line 261):
        
        # Call to sort(...): (line 261)
        # Processing the call arguments (line 261)
        
        # Call to compressed(...): (line 261)
        # Processing the call keyword arguments (line 261)
        kwargs_579076 = {}
        # Getting the type of 'data' (line 261)
        data_579074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'data', False)
        # Obtaining the member 'compressed' of a type (line 261)
        compressed_579075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 23), data_579074, 'compressed')
        # Calling compressed(args, kwargs) (line 261)
        compressed_call_result_579077 = invoke(stypy.reporting.localization.Localization(__file__, 261, 23), compressed_579075, *[], **kwargs_579076)
        
        # Processing the call keyword arguments (line 261)
        kwargs_579078 = {}
        # Getting the type of 'np' (line 261)
        np_579072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'np', False)
        # Obtaining the member 'sort' of a type (line 261)
        sort_579073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 15), np_579072, 'sort')
        # Calling sort(args, kwargs) (line 261)
        sort_call_result_579079 = invoke(stypy.reporting.localization.Localization(__file__, 261, 15), sort_579073, *[compressed_call_result_579077], **kwargs_579078)
        
        # Assigning a type to the variable 'data' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'data', sort_call_result_579079)
        
        # Assigning a Attribute to a Name (line 262):
        
        # Assigning a Attribute to a Name (line 262):
        # Getting the type of 'data' (line 262)
        data_579080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'data')
        # Obtaining the member 'size' of a type (line 262)
        size_579081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 12), data_579080, 'size')
        # Assigning a type to the variable 'n' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'n', size_579081)
        
        # Assigning a Call to a Name (line 263):
        
        # Assigning a Call to a Name (line 263):
        
        # Call to astype(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'int_' (line 263)
        int__579092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 46), 'int_', False)
        # Processing the call keyword arguments (line 263)
        kwargs_579093 = {}
        
        # Call to array(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'p' (line 263)
        p_579084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 25), 'p', False)
        # Processing the call keyword arguments (line 263)
        kwargs_579085 = {}
        # Getting the type of 'np' (line 263)
        np_579082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 263)
        array_579083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 16), np_579082, 'array')
        # Calling array(args, kwargs) (line 263)
        array_call_result_579086 = invoke(stypy.reporting.localization.Localization(__file__, 263, 16), array_579083, *[p_579084], **kwargs_579085)
        
        # Getting the type of 'n' (line 263)
        n_579087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'n', False)
        # Applying the binary operator '*' (line 263)
        result_mul_579088 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 16), '*', array_call_result_579086, n_579087)
        
        float_579089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 34), 'float')
        # Applying the binary operator '+' (line 263)
        result_add_579090 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 16), '+', result_mul_579088, float_579089)
        
        # Obtaining the member 'astype' of a type (line 263)
        astype_579091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 16), result_add_579090, 'astype')
        # Calling astype(args, kwargs) (line 263)
        astype_call_result_579094 = invoke(stypy.reporting.localization.Localization(__file__, 263, 16), astype_579091, *[int__579092], **kwargs_579093)
        
        # Assigning a type to the variable 'prob' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'prob', astype_call_result_579094)
        
        # Assigning a Attribute to a Name (line 264):
        
        # Assigning a Attribute to a Name (line 264):
        # Getting the type of 'beta' (line 264)
        beta_579095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 18), 'beta')
        # Obtaining the member 'cdf' of a type (line 264)
        cdf_579096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 18), beta_579095, 'cdf')
        # Assigning a type to the variable 'betacdf' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'betacdf', cdf_579096)
        
        # Assigning a Call to a Name (line 266):
        
        # Assigning a Call to a Name (line 266):
        
        # Call to empty(...): (line 266)
        # Processing the call arguments (line 266)
        
        # Call to len(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'prob' (line 266)
        prob_579100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 26), 'prob', False)
        # Processing the call keyword arguments (line 266)
        kwargs_579101 = {}
        # Getting the type of 'len' (line 266)
        len_579099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'len', False)
        # Calling len(args, kwargs) (line 266)
        len_call_result_579102 = invoke(stypy.reporting.localization.Localization(__file__, 266, 22), len_579099, *[prob_579100], **kwargs_579101)
        
        # Getting the type of 'float_' (line 266)
        float__579103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 33), 'float_', False)
        # Processing the call keyword arguments (line 266)
        kwargs_579104 = {}
        # Getting the type of 'np' (line 266)
        np_579097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 13), 'np', False)
        # Obtaining the member 'empty' of a type (line 266)
        empty_579098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 13), np_579097, 'empty')
        # Calling empty(args, kwargs) (line 266)
        empty_call_result_579105 = invoke(stypy.reporting.localization.Localization(__file__, 266, 13), empty_579098, *[len_call_result_579102, float__579103], **kwargs_579104)
        
        # Assigning a type to the variable 'mj' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'mj', empty_call_result_579105)
        
        # Assigning a BinOp to a Name (line 267):
        
        # Assigning a BinOp to a Name (line 267):
        
        # Call to arange(...): (line 267)
        # Processing the call arguments (line 267)
        int_579108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 22), 'int')
        # Getting the type of 'n' (line 267)
        n_579109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 24), 'n', False)
        int_579110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 26), 'int')
        # Applying the binary operator '+' (line 267)
        result_add_579111 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 24), '+', n_579109, int_579110)
        
        # Processing the call keyword arguments (line 267)
        # Getting the type of 'float_' (line 267)
        float__579112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 35), 'float_', False)
        keyword_579113 = float__579112
        kwargs_579114 = {'dtype': keyword_579113}
        # Getting the type of 'np' (line 267)
        np_579106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 267)
        arange_579107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 12), np_579106, 'arange')
        # Calling arange(args, kwargs) (line 267)
        arange_call_result_579115 = invoke(stypy.reporting.localization.Localization(__file__, 267, 12), arange_579107, *[int_579108, result_add_579111], **kwargs_579114)
        
        # Getting the type of 'n' (line 267)
        n_579116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 45), 'n')
        # Applying the binary operator 'div' (line 267)
        result_div_579117 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 12), 'div', arange_call_result_579115, n_579116)
        
        # Assigning a type to the variable 'x' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'x', result_div_579117)
        
        # Assigning a BinOp to a Name (line 268):
        
        # Assigning a BinOp to a Name (line 268):
        # Getting the type of 'x' (line 268)
        x_579118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'x')
        float_579119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 16), 'float')
        # Getting the type of 'n' (line 268)
        n_579120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'n')
        # Applying the binary operator 'div' (line 268)
        result_div_579121 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 16), 'div', float_579119, n_579120)
        
        # Applying the binary operator '-' (line 268)
        result_sub_579122 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 12), '-', x_579118, result_div_579121)
        
        # Assigning a type to the variable 'y' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'y', result_sub_579122)
        
        
        # Call to enumerate(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'prob' (line 269)
        prob_579124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 31), 'prob', False)
        # Processing the call keyword arguments (line 269)
        kwargs_579125 = {}
        # Getting the type of 'enumerate' (line 269)
        enumerate_579123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 269)
        enumerate_call_result_579126 = invoke(stypy.reporting.localization.Localization(__file__, 269, 21), enumerate_579123, *[prob_579124], **kwargs_579125)
        
        # Testing the type of a for loop iterable (line 269)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 269, 8), enumerate_call_result_579126)
        # Getting the type of the for loop variable (line 269)
        for_loop_var_579127 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 269, 8), enumerate_call_result_579126)
        # Assigning a type to the variable 'i' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 8), for_loop_var_579127))
        # Assigning a type to the variable 'm' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 8), for_loop_var_579127))
        # SSA begins for a for statement (line 269)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 270):
        
        # Assigning a BinOp to a Name (line 270):
        
        # Call to betacdf(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'x' (line 270)
        x_579129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 24), 'x', False)
        # Getting the type of 'm' (line 270)
        m_579130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 26), 'm', False)
        int_579131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 28), 'int')
        # Applying the binary operator '-' (line 270)
        result_sub_579132 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 26), '-', m_579130, int_579131)
        
        # Getting the type of 'n' (line 270)
        n_579133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 30), 'n', False)
        # Getting the type of 'm' (line 270)
        m_579134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 32), 'm', False)
        # Applying the binary operator '-' (line 270)
        result_sub_579135 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 30), '-', n_579133, m_579134)
        
        # Processing the call keyword arguments (line 270)
        kwargs_579136 = {}
        # Getting the type of 'betacdf' (line 270)
        betacdf_579128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'betacdf', False)
        # Calling betacdf(args, kwargs) (line 270)
        betacdf_call_result_579137 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), betacdf_579128, *[x_579129, result_sub_579132, result_sub_579135], **kwargs_579136)
        
        
        # Call to betacdf(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'y' (line 270)
        y_579139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 45), 'y', False)
        # Getting the type of 'm' (line 270)
        m_579140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 47), 'm', False)
        int_579141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 49), 'int')
        # Applying the binary operator '-' (line 270)
        result_sub_579142 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 47), '-', m_579140, int_579141)
        
        # Getting the type of 'n' (line 270)
        n_579143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 51), 'n', False)
        # Getting the type of 'm' (line 270)
        m_579144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 53), 'm', False)
        # Applying the binary operator '-' (line 270)
        result_sub_579145 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 51), '-', n_579143, m_579144)
        
        # Processing the call keyword arguments (line 270)
        kwargs_579146 = {}
        # Getting the type of 'betacdf' (line 270)
        betacdf_579138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 37), 'betacdf', False)
        # Calling betacdf(args, kwargs) (line 270)
        betacdf_call_result_579147 = invoke(stypy.reporting.localization.Localization(__file__, 270, 37), betacdf_579138, *[y_579139, result_sub_579142, result_sub_579145], **kwargs_579146)
        
        # Applying the binary operator '-' (line 270)
        result_sub_579148 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 16), '-', betacdf_call_result_579137, betacdf_call_result_579147)
        
        # Assigning a type to the variable 'W' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'W', result_sub_579148)
        
        # Assigning a Call to a Name (line 271):
        
        # Assigning a Call to a Name (line 271):
        
        # Call to dot(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'W' (line 271)
        W_579151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 24), 'W', False)
        # Getting the type of 'data' (line 271)
        data_579152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 26), 'data', False)
        # Processing the call keyword arguments (line 271)
        kwargs_579153 = {}
        # Getting the type of 'np' (line 271)
        np_579149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 17), 'np', False)
        # Obtaining the member 'dot' of a type (line 271)
        dot_579150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 17), np_579149, 'dot')
        # Calling dot(args, kwargs) (line 271)
        dot_call_result_579154 = invoke(stypy.reporting.localization.Localization(__file__, 271, 17), dot_579150, *[W_579151, data_579152], **kwargs_579153)
        
        # Assigning a type to the variable 'C1' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'C1', dot_call_result_579154)
        
        # Assigning a Call to a Name (line 272):
        
        # Assigning a Call to a Name (line 272):
        
        # Call to dot(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'W' (line 272)
        W_579157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 24), 'W', False)
        # Getting the type of 'data' (line 272)
        data_579158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 26), 'data', False)
        int_579159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 32), 'int')
        # Applying the binary operator '**' (line 272)
        result_pow_579160 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 26), '**', data_579158, int_579159)
        
        # Processing the call keyword arguments (line 272)
        kwargs_579161 = {}
        # Getting the type of 'np' (line 272)
        np_579155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'np', False)
        # Obtaining the member 'dot' of a type (line 272)
        dot_579156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 17), np_579155, 'dot')
        # Calling dot(args, kwargs) (line 272)
        dot_call_result_579162 = invoke(stypy.reporting.localization.Localization(__file__, 272, 17), dot_579156, *[W_579157, result_pow_579160], **kwargs_579161)
        
        # Assigning a type to the variable 'C2' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'C2', dot_call_result_579162)
        
        # Assigning a Call to a Subscript (line 273):
        
        # Assigning a Call to a Subscript (line 273):
        
        # Call to sqrt(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'C2' (line 273)
        C2_579165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 28), 'C2', False)
        # Getting the type of 'C1' (line 273)
        C1_579166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 33), 'C1', False)
        int_579167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 37), 'int')
        # Applying the binary operator '**' (line 273)
        result_pow_579168 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 33), '**', C1_579166, int_579167)
        
        # Applying the binary operator '-' (line 273)
        result_sub_579169 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 28), '-', C2_579165, result_pow_579168)
        
        # Processing the call keyword arguments (line 273)
        kwargs_579170 = {}
        # Getting the type of 'np' (line 273)
        np_579163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 273)
        sqrt_579164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 20), np_579163, 'sqrt')
        # Calling sqrt(args, kwargs) (line 273)
        sqrt_call_result_579171 = invoke(stypy.reporting.localization.Localization(__file__, 273, 20), sqrt_579164, *[result_sub_579169], **kwargs_579170)
        
        # Getting the type of 'mj' (line 273)
        mj_579172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'mj')
        # Getting the type of 'i' (line 273)
        i_579173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 15), 'i')
        # Storing an element on a container (line 273)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 12), mj_579172, (i_579173, sqrt_call_result_579171))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'mj' (line 274)
        mj_579174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'mj')
        # Assigning a type to the variable 'stypy_return_type' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'stypy_return_type', mj_579174)
        
        # ################# End of '_mjci_1D(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mjci_1D' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_579175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_579175)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mjci_1D'
        return stypy_return_type_579175

    # Assigning a type to the variable '_mjci_1D' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), '_mjci_1D', _mjci_1D)
    
    # Assigning a Call to a Name (line 276):
    
    # Assigning a Call to a Name (line 276):
    
    # Call to array(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 'data' (line 276)
    data_579178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'data', False)
    # Processing the call keyword arguments (line 276)
    # Getting the type of 'False' (line 276)
    False_579179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 31), 'False', False)
    keyword_579180 = False_579179
    kwargs_579181 = {'copy': keyword_579180}
    # Getting the type of 'ma' (line 276)
    ma_579176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 11), 'ma', False)
    # Obtaining the member 'array' of a type (line 276)
    array_579177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 11), ma_579176, 'array')
    # Calling array(args, kwargs) (line 276)
    array_call_result_579182 = invoke(stypy.reporting.localization.Localization(__file__, 276, 11), array_579177, *[data_579178], **kwargs_579181)
    
    # Assigning a type to the variable 'data' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'data', array_call_result_579182)
    
    
    # Getting the type of 'data' (line 277)
    data_579183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 7), 'data')
    # Obtaining the member 'ndim' of a type (line 277)
    ndim_579184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 7), data_579183, 'ndim')
    int_579185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 19), 'int')
    # Applying the binary operator '>' (line 277)
    result_gt_579186 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 7), '>', ndim_579184, int_579185)
    
    # Testing the type of an if condition (line 277)
    if_condition_579187 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 4), result_gt_579186)
    # Assigning a type to the variable 'if_condition_579187' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'if_condition_579187', if_condition_579187)
    # SSA begins for if statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 278)
    # Processing the call arguments (line 278)
    str_579189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 25), 'str', "Array 'data' must be at most two dimensional, but got data.ndim = %d")
    # Getting the type of 'data' (line 279)
    data_579190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 52), 'data', False)
    # Obtaining the member 'ndim' of a type (line 279)
    ndim_579191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 52), data_579190, 'ndim')
    # Applying the binary operator '%' (line 278)
    result_mod_579192 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 25), '%', str_579189, ndim_579191)
    
    # Processing the call keyword arguments (line 278)
    kwargs_579193 = {}
    # Getting the type of 'ValueError' (line 278)
    ValueError_579188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 278)
    ValueError_call_result_579194 = invoke(stypy.reporting.localization.Localization(__file__, 278, 14), ValueError_579188, *[result_mod_579192], **kwargs_579193)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 278, 8), ValueError_call_result_579194, 'raise parameter', BaseException)
    # SSA join for if statement (line 277)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 281):
    
    # Assigning a Call to a Name (line 281):
    
    # Call to array(...): (line 281)
    # Processing the call arguments (line 281)
    # Getting the type of 'prob' (line 281)
    prob_579197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'prob', False)
    # Processing the call keyword arguments (line 281)
    # Getting the type of 'False' (line 281)
    False_579198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'False', False)
    keyword_579199 = False_579198
    int_579200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 41), 'int')
    keyword_579201 = int_579200
    kwargs_579202 = {'copy': keyword_579199, 'ndmin': keyword_579201}
    # Getting the type of 'np' (line 281)
    np_579195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 281)
    array_579196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), np_579195, 'array')
    # Calling array(args, kwargs) (line 281)
    array_call_result_579203 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), array_579196, *[prob_579197], **kwargs_579202)
    
    # Assigning a type to the variable 'p' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'p', array_call_result_579203)
    
    # Type idiom detected: calculating its left and rigth part (line 283)
    # Getting the type of 'axis' (line 283)
    axis_579204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'axis')
    # Getting the type of 'None' (line 283)
    None_579205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'None')
    
    (may_be_579206, more_types_in_union_579207) = may_be_none(axis_579204, None_579205)

    if may_be_579206:

        if more_types_in_union_579207:
            # Runtime conditional SSA (line 283)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to _mjci_1D(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'data' (line 284)
        data_579209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'data', False)
        # Getting the type of 'p' (line 284)
        p_579210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 30), 'p', False)
        # Processing the call keyword arguments (line 284)
        kwargs_579211 = {}
        # Getting the type of '_mjci_1D' (line 284)
        _mjci_1D_579208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 15), '_mjci_1D', False)
        # Calling _mjci_1D(args, kwargs) (line 284)
        _mjci_1D_call_result_579212 = invoke(stypy.reporting.localization.Localization(__file__, 284, 15), _mjci_1D_579208, *[data_579209, p_579210], **kwargs_579211)
        
        # Assigning a type to the variable 'stypy_return_type' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'stypy_return_type', _mjci_1D_call_result_579212)

        if more_types_in_union_579207:
            # Runtime conditional SSA for else branch (line 283)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_579206) or more_types_in_union_579207):
        
        # Call to apply_along_axis(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of '_mjci_1D' (line 286)
        _mjci_1D_579215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 35), '_mjci_1D', False)
        # Getting the type of 'axis' (line 286)
        axis_579216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 45), 'axis', False)
        # Getting the type of 'data' (line 286)
        data_579217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 51), 'data', False)
        # Getting the type of 'p' (line 286)
        p_579218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 57), 'p', False)
        # Processing the call keyword arguments (line 286)
        kwargs_579219 = {}
        # Getting the type of 'ma' (line 286)
        ma_579213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 15), 'ma', False)
        # Obtaining the member 'apply_along_axis' of a type (line 286)
        apply_along_axis_579214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 15), ma_579213, 'apply_along_axis')
        # Calling apply_along_axis(args, kwargs) (line 286)
        apply_along_axis_call_result_579220 = invoke(stypy.reporting.localization.Localization(__file__, 286, 15), apply_along_axis_579214, *[_mjci_1D_579215, axis_579216, data_579217, p_579218], **kwargs_579219)
        
        # Assigning a type to the variable 'stypy_return_type' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'stypy_return_type', apply_along_axis_call_result_579220)

        if (may_be_579206 and more_types_in_union_579207):
            # SSA join for if statement (line 283)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'mjci(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mjci' in the type store
    # Getting the type of 'stypy_return_type' (line 244)
    stypy_return_type_579221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_579221)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mjci'
    return stypy_return_type_579221

# Assigning a type to the variable 'mjci' (line 244)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'mjci', mjci)

@norecursion
def mquantiles_cimj(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'list' (line 289)
    list_579222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 289)
    # Adding element type (line 289)
    float_579223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 32), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 31), list_579222, float_579223)
    # Adding element type (line 289)
    float_579224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 37), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 31), list_579222, float_579224)
    # Adding element type (line 289)
    float_579225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 31), list_579222, float_579225)
    
    float_579226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 55), 'float')
    # Getting the type of 'None' (line 289)
    None_579227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 66), 'None')
    defaults = [list_579222, float_579226, None_579227]
    # Create a new context for function 'mquantiles_cimj'
    module_type_store = module_type_store.open_function_context('mquantiles_cimj', 289, 0, False)
    
    # Passed parameters checking function
    mquantiles_cimj.stypy_localization = localization
    mquantiles_cimj.stypy_type_of_self = None
    mquantiles_cimj.stypy_type_store = module_type_store
    mquantiles_cimj.stypy_function_name = 'mquantiles_cimj'
    mquantiles_cimj.stypy_param_names_list = ['data', 'prob', 'alpha', 'axis']
    mquantiles_cimj.stypy_varargs_param_name = None
    mquantiles_cimj.stypy_kwargs_param_name = None
    mquantiles_cimj.stypy_call_defaults = defaults
    mquantiles_cimj.stypy_call_varargs = varargs
    mquantiles_cimj.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mquantiles_cimj', ['data', 'prob', 'alpha', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mquantiles_cimj', localization, ['data', 'prob', 'alpha', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mquantiles_cimj(...)' code ##################

    str_579228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, (-1)), 'str', '\n    Computes the alpha confidence interval for the selected quantiles of the\n    data, with Maritz-Jarrett estimators.\n\n    Parameters\n    ----------\n    data : ndarray\n        Data array.\n    prob : sequence, optional\n        Sequence of quantiles to compute.\n    alpha : float, optional\n        Confidence level of the intervals.\n    axis : int or None, optional\n        Axis along which to compute the quantiles.\n        If None, use a flattened array.\n\n    Returns\n    -------\n    ci_lower : ndarray\n        The lower boundaries of the confidence interval.  Of the same length as\n        `prob`.\n    ci_upper : ndarray\n        The upper boundaries of the confidence interval.  Of the same length as\n        `prob`.\n\n    ')
    
    # Assigning a Call to a Name (line 316):
    
    # Assigning a Call to a Name (line 316):
    
    # Call to min(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'alpha' (line 316)
    alpha_579230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 16), 'alpha', False)
    int_579231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 23), 'int')
    # Getting the type of 'alpha' (line 316)
    alpha_579232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 27), 'alpha', False)
    # Applying the binary operator '-' (line 316)
    result_sub_579233 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 23), '-', int_579231, alpha_579232)
    
    # Processing the call keyword arguments (line 316)
    kwargs_579234 = {}
    # Getting the type of 'min' (line 316)
    min_579229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'min', False)
    # Calling min(args, kwargs) (line 316)
    min_call_result_579235 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), min_579229, *[alpha_579230, result_sub_579233], **kwargs_579234)
    
    # Assigning a type to the variable 'alpha' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'alpha', min_call_result_579235)
    
    # Assigning a Call to a Name (line 317):
    
    # Assigning a Call to a Name (line 317):
    
    # Call to ppf(...): (line 317)
    # Processing the call arguments (line 317)
    int_579238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 17), 'int')
    # Getting the type of 'alpha' (line 317)
    alpha_579239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 21), 'alpha', False)
    float_579240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 27), 'float')
    # Applying the binary operator 'div' (line 317)
    result_div_579241 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 21), 'div', alpha_579239, float_579240)
    
    # Applying the binary operator '-' (line 317)
    result_sub_579242 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 17), '-', int_579238, result_div_579241)
    
    # Processing the call keyword arguments (line 317)
    kwargs_579243 = {}
    # Getting the type of 'norm' (line 317)
    norm_579236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'norm', False)
    # Obtaining the member 'ppf' of a type (line 317)
    ppf_579237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), norm_579236, 'ppf')
    # Calling ppf(args, kwargs) (line 317)
    ppf_call_result_579244 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), ppf_579237, *[result_sub_579242], **kwargs_579243)
    
    # Assigning a type to the variable 'z' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'z', ppf_call_result_579244)
    
    # Assigning a Call to a Name (line 318):
    
    # Assigning a Call to a Name (line 318):
    
    # Call to mquantiles(...): (line 318)
    # Processing the call arguments (line 318)
    # Getting the type of 'data' (line 318)
    data_579247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 27), 'data', False)
    # Getting the type of 'prob' (line 318)
    prob_579248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 33), 'prob', False)
    # Processing the call keyword arguments (line 318)
    int_579249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 46), 'int')
    keyword_579250 = int_579249
    int_579251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 55), 'int')
    keyword_579252 = int_579251
    # Getting the type of 'axis' (line 318)
    axis_579253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 63), 'axis', False)
    keyword_579254 = axis_579253
    kwargs_579255 = {'alphap': keyword_579250, 'axis': keyword_579254, 'betap': keyword_579252}
    # Getting the type of 'mstats' (line 318)
    mstats_579245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 9), 'mstats', False)
    # Obtaining the member 'mquantiles' of a type (line 318)
    mquantiles_579246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 9), mstats_579245, 'mquantiles')
    # Calling mquantiles(args, kwargs) (line 318)
    mquantiles_call_result_579256 = invoke(stypy.reporting.localization.Localization(__file__, 318, 9), mquantiles_579246, *[data_579247, prob_579248], **kwargs_579255)
    
    # Assigning a type to the variable 'xq' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'xq', mquantiles_call_result_579256)
    
    # Assigning a Call to a Name (line 319):
    
    # Assigning a Call to a Name (line 319):
    
    # Call to mjci(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'data' (line 319)
    data_579258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 15), 'data', False)
    # Getting the type of 'prob' (line 319)
    prob_579259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 21), 'prob', False)
    # Processing the call keyword arguments (line 319)
    # Getting the type of 'axis' (line 319)
    axis_579260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 32), 'axis', False)
    keyword_579261 = axis_579260
    kwargs_579262 = {'axis': keyword_579261}
    # Getting the type of 'mjci' (line 319)
    mjci_579257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 10), 'mjci', False)
    # Calling mjci(args, kwargs) (line 319)
    mjci_call_result_579263 = invoke(stypy.reporting.localization.Localization(__file__, 319, 10), mjci_579257, *[data_579258, prob_579259], **kwargs_579262)
    
    # Assigning a type to the variable 'smj' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'smj', mjci_call_result_579263)
    
    # Obtaining an instance of the builtin type 'tuple' (line 320)
    tuple_579264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 320)
    # Adding element type (line 320)
    # Getting the type of 'xq' (line 320)
    xq_579265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'xq')
    # Getting the type of 'z' (line 320)
    z_579266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 17), 'z')
    # Getting the type of 'smj' (line 320)
    smj_579267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 21), 'smj')
    # Applying the binary operator '*' (line 320)
    result_mul_579268 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 17), '*', z_579266, smj_579267)
    
    # Applying the binary operator '-' (line 320)
    result_sub_579269 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 12), '-', xq_579265, result_mul_579268)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 12), tuple_579264, result_sub_579269)
    # Adding element type (line 320)
    # Getting the type of 'xq' (line 320)
    xq_579270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 26), 'xq')
    # Getting the type of 'z' (line 320)
    z_579271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 31), 'z')
    # Getting the type of 'smj' (line 320)
    smj_579272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 35), 'smj')
    # Applying the binary operator '*' (line 320)
    result_mul_579273 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 31), '*', z_579271, smj_579272)
    
    # Applying the binary operator '+' (line 320)
    result_add_579274 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 26), '+', xq_579270, result_mul_579273)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 12), tuple_579264, result_add_579274)
    
    # Assigning a type to the variable 'stypy_return_type' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'stypy_return_type', tuple_579264)
    
    # ################# End of 'mquantiles_cimj(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mquantiles_cimj' in the type store
    # Getting the type of 'stypy_return_type' (line 289)
    stypy_return_type_579275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_579275)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mquantiles_cimj'
    return stypy_return_type_579275

# Assigning a type to the variable 'mquantiles_cimj' (line 289)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 0), 'mquantiles_cimj', mquantiles_cimj)

@norecursion
def median_cihs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_579276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 28), 'float')
    # Getting the type of 'None' (line 323)
    None_579277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 39), 'None')
    defaults = [float_579276, None_579277]
    # Create a new context for function 'median_cihs'
    module_type_store = module_type_store.open_function_context('median_cihs', 323, 0, False)
    
    # Passed parameters checking function
    median_cihs.stypy_localization = localization
    median_cihs.stypy_type_of_self = None
    median_cihs.stypy_type_store = module_type_store
    median_cihs.stypy_function_name = 'median_cihs'
    median_cihs.stypy_param_names_list = ['data', 'alpha', 'axis']
    median_cihs.stypy_varargs_param_name = None
    median_cihs.stypy_kwargs_param_name = None
    median_cihs.stypy_call_defaults = defaults
    median_cihs.stypy_call_varargs = varargs
    median_cihs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'median_cihs', ['data', 'alpha', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'median_cihs', localization, ['data', 'alpha', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'median_cihs(...)' code ##################

    str_579278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, (-1)), 'str', '\n    Computes the alpha-level confidence interval for the median of the data.\n\n    Uses the Hettmasperger-Sheather method.\n\n    Parameters\n    ----------\n    data : array_like\n        Input data. Masked values are discarded. The input should be 1D only,\n        or `axis` should be set to None.\n    alpha : float, optional\n        Confidence level of the intervals.\n    axis : int or None, optional\n        Axis along which to compute the quantiles. If None, use a flattened\n        array.\n\n    Returns\n    -------\n    median_cihs\n        Alpha level confidence interval.\n\n    ')

    @norecursion
    def _cihs_1D(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cihs_1D'
        module_type_store = module_type_store.open_function_context('_cihs_1D', 346, 4, False)
        
        # Passed parameters checking function
        _cihs_1D.stypy_localization = localization
        _cihs_1D.stypy_type_of_self = None
        _cihs_1D.stypy_type_store = module_type_store
        _cihs_1D.stypy_function_name = '_cihs_1D'
        _cihs_1D.stypy_param_names_list = ['data', 'alpha']
        _cihs_1D.stypy_varargs_param_name = None
        _cihs_1D.stypy_kwargs_param_name = None
        _cihs_1D.stypy_call_defaults = defaults
        _cihs_1D.stypy_call_varargs = varargs
        _cihs_1D.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_cihs_1D', ['data', 'alpha'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cihs_1D', localization, ['data', 'alpha'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cihs_1D(...)' code ##################

        
        # Assigning a Call to a Name (line 347):
        
        # Assigning a Call to a Name (line 347):
        
        # Call to sort(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Call to compressed(...): (line 347)
        # Processing the call keyword arguments (line 347)
        kwargs_579283 = {}
        # Getting the type of 'data' (line 347)
        data_579281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 23), 'data', False)
        # Obtaining the member 'compressed' of a type (line 347)
        compressed_579282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 23), data_579281, 'compressed')
        # Calling compressed(args, kwargs) (line 347)
        compressed_call_result_579284 = invoke(stypy.reporting.localization.Localization(__file__, 347, 23), compressed_579282, *[], **kwargs_579283)
        
        # Processing the call keyword arguments (line 347)
        kwargs_579285 = {}
        # Getting the type of 'np' (line 347)
        np_579279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 15), 'np', False)
        # Obtaining the member 'sort' of a type (line 347)
        sort_579280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 15), np_579279, 'sort')
        # Calling sort(args, kwargs) (line 347)
        sort_call_result_579286 = invoke(stypy.reporting.localization.Localization(__file__, 347, 15), sort_579280, *[compressed_call_result_579284], **kwargs_579285)
        
        # Assigning a type to the variable 'data' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'data', sort_call_result_579286)
        
        # Assigning a Call to a Name (line 348):
        
        # Assigning a Call to a Name (line 348):
        
        # Call to len(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'data' (line 348)
        data_579288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'data', False)
        # Processing the call keyword arguments (line 348)
        kwargs_579289 = {}
        # Getting the type of 'len' (line 348)
        len_579287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'len', False)
        # Calling len(args, kwargs) (line 348)
        len_call_result_579290 = invoke(stypy.reporting.localization.Localization(__file__, 348, 12), len_579287, *[data_579288], **kwargs_579289)
        
        # Assigning a type to the variable 'n' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'n', len_call_result_579290)
        
        # Assigning a Call to a Name (line 349):
        
        # Assigning a Call to a Name (line 349):
        
        # Call to min(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'alpha' (line 349)
        alpha_579292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 20), 'alpha', False)
        int_579293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 27), 'int')
        # Getting the type of 'alpha' (line 349)
        alpha_579294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 29), 'alpha', False)
        # Applying the binary operator '-' (line 349)
        result_sub_579295 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 27), '-', int_579293, alpha_579294)
        
        # Processing the call keyword arguments (line 349)
        kwargs_579296 = {}
        # Getting the type of 'min' (line 349)
        min_579291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'min', False)
        # Calling min(args, kwargs) (line 349)
        min_call_result_579297 = invoke(stypy.reporting.localization.Localization(__file__, 349, 16), min_579291, *[alpha_579292, result_sub_579295], **kwargs_579296)
        
        # Assigning a type to the variable 'alpha' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'alpha', min_call_result_579297)
        
        # Assigning a Call to a Name (line 350):
        
        # Assigning a Call to a Name (line 350):
        
        # Call to int(...): (line 350)
        # Processing the call arguments (line 350)
        
        # Call to _ppf(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'alpha' (line 350)
        alpha_579301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 27), 'alpha', False)
        float_579302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 33), 'float')
        # Applying the binary operator 'div' (line 350)
        result_div_579303 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 27), 'div', alpha_579301, float_579302)
        
        # Getting the type of 'n' (line 350)
        n_579304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 37), 'n', False)
        float_579305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 40), 'float')
        # Processing the call keyword arguments (line 350)
        kwargs_579306 = {}
        # Getting the type of 'binom' (line 350)
        binom_579299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'binom', False)
        # Obtaining the member '_ppf' of a type (line 350)
        _ppf_579300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 16), binom_579299, '_ppf')
        # Calling _ppf(args, kwargs) (line 350)
        _ppf_call_result_579307 = invoke(stypy.reporting.localization.Localization(__file__, 350, 16), _ppf_579300, *[result_div_579303, n_579304, float_579305], **kwargs_579306)
        
        # Processing the call keyword arguments (line 350)
        kwargs_579308 = {}
        # Getting the type of 'int' (line 350)
        int_579298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'int', False)
        # Calling int(args, kwargs) (line 350)
        int_call_result_579309 = invoke(stypy.reporting.localization.Localization(__file__, 350, 12), int_579298, *[_ppf_call_result_579307], **kwargs_579308)
        
        # Assigning a type to the variable 'k' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'k', int_call_result_579309)
        
        # Assigning a BinOp to a Name (line 351):
        
        # Assigning a BinOp to a Name (line 351):
        
        # Call to cdf(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'n' (line 351)
        n_579312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 23), 'n', False)
        # Getting the type of 'k' (line 351)
        k_579313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 25), 'k', False)
        # Applying the binary operator '-' (line 351)
        result_sub_579314 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 23), '-', n_579312, k_579313)
        
        # Getting the type of 'n' (line 351)
        n_579315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 27), 'n', False)
        float_579316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 29), 'float')
        # Processing the call keyword arguments (line 351)
        kwargs_579317 = {}
        # Getting the type of 'binom' (line 351)
        binom_579310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 13), 'binom', False)
        # Obtaining the member 'cdf' of a type (line 351)
        cdf_579311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 13), binom_579310, 'cdf')
        # Calling cdf(args, kwargs) (line 351)
        cdf_call_result_579318 = invoke(stypy.reporting.localization.Localization(__file__, 351, 13), cdf_579311, *[result_sub_579314, n_579315, float_579316], **kwargs_579317)
        
        
        # Call to cdf(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'k' (line 351)
        k_579321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 46), 'k', False)
        int_579322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 48), 'int')
        # Applying the binary operator '-' (line 351)
        result_sub_579323 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 46), '-', k_579321, int_579322)
        
        # Getting the type of 'n' (line 351)
        n_579324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 50), 'n', False)
        float_579325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 52), 'float')
        # Processing the call keyword arguments (line 351)
        kwargs_579326 = {}
        # Getting the type of 'binom' (line 351)
        binom_579319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 36), 'binom', False)
        # Obtaining the member 'cdf' of a type (line 351)
        cdf_579320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 36), binom_579319, 'cdf')
        # Calling cdf(args, kwargs) (line 351)
        cdf_call_result_579327 = invoke(stypy.reporting.localization.Localization(__file__, 351, 36), cdf_579320, *[result_sub_579323, n_579324, float_579325], **kwargs_579326)
        
        # Applying the binary operator '-' (line 351)
        result_sub_579328 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 13), '-', cdf_call_result_579318, cdf_call_result_579327)
        
        # Assigning a type to the variable 'gk' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'gk', result_sub_579328)
        
        
        # Getting the type of 'gk' (line 352)
        gk_579329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 11), 'gk')
        int_579330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 16), 'int')
        # Getting the type of 'alpha' (line 352)
        alpha_579331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 18), 'alpha')
        # Applying the binary operator '-' (line 352)
        result_sub_579332 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 16), '-', int_579330, alpha_579331)
        
        # Applying the binary operator '<' (line 352)
        result_lt_579333 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 11), '<', gk_579329, result_sub_579332)
        
        # Testing the type of an if condition (line 352)
        if_condition_579334 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 8), result_lt_579333)
        # Assigning a type to the variable 'if_condition_579334' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'if_condition_579334', if_condition_579334)
        # SSA begins for if statement (line 352)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'k' (line 353)
        k_579335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'k')
        int_579336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 17), 'int')
        # Applying the binary operator '-=' (line 353)
        result_isub_579337 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 12), '-=', k_579335, int_579336)
        # Assigning a type to the variable 'k' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'k', result_isub_579337)
        
        
        # Assigning a BinOp to a Name (line 354):
        
        # Assigning a BinOp to a Name (line 354):
        
        # Call to cdf(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'n' (line 354)
        n_579340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'n', False)
        # Getting the type of 'k' (line 354)
        k_579341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 29), 'k', False)
        # Applying the binary operator '-' (line 354)
        result_sub_579342 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 27), '-', n_579340, k_579341)
        
        # Getting the type of 'n' (line 354)
        n_579343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 31), 'n', False)
        float_579344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 33), 'float')
        # Processing the call keyword arguments (line 354)
        kwargs_579345 = {}
        # Getting the type of 'binom' (line 354)
        binom_579338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 17), 'binom', False)
        # Obtaining the member 'cdf' of a type (line 354)
        cdf_579339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 17), binom_579338, 'cdf')
        # Calling cdf(args, kwargs) (line 354)
        cdf_call_result_579346 = invoke(stypy.reporting.localization.Localization(__file__, 354, 17), cdf_579339, *[result_sub_579342, n_579343, float_579344], **kwargs_579345)
        
        
        # Call to cdf(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'k' (line 354)
        k_579349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 50), 'k', False)
        int_579350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 52), 'int')
        # Applying the binary operator '-' (line 354)
        result_sub_579351 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 50), '-', k_579349, int_579350)
        
        # Getting the type of 'n' (line 354)
        n_579352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 54), 'n', False)
        float_579353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 56), 'float')
        # Processing the call keyword arguments (line 354)
        kwargs_579354 = {}
        # Getting the type of 'binom' (line 354)
        binom_579347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 40), 'binom', False)
        # Obtaining the member 'cdf' of a type (line 354)
        cdf_579348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 40), binom_579347, 'cdf')
        # Calling cdf(args, kwargs) (line 354)
        cdf_call_result_579355 = invoke(stypy.reporting.localization.Localization(__file__, 354, 40), cdf_579348, *[result_sub_579351, n_579352, float_579353], **kwargs_579354)
        
        # Applying the binary operator '-' (line 354)
        result_sub_579356 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 17), '-', cdf_call_result_579346, cdf_call_result_579355)
        
        # Assigning a type to the variable 'gk' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'gk', result_sub_579356)
        # SSA join for if statement (line 352)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 355):
        
        # Assigning a BinOp to a Name (line 355):
        
        # Call to cdf(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'n' (line 355)
        n_579359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 24), 'n', False)
        # Getting the type of 'k' (line 355)
        k_579360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 26), 'k', False)
        # Applying the binary operator '-' (line 355)
        result_sub_579361 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 24), '-', n_579359, k_579360)
        
        int_579362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 28), 'int')
        # Applying the binary operator '-' (line 355)
        result_sub_579363 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 27), '-', result_sub_579361, int_579362)
        
        # Getting the type of 'n' (line 355)
        n_579364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 30), 'n', False)
        float_579365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 32), 'float')
        # Processing the call keyword arguments (line 355)
        kwargs_579366 = {}
        # Getting the type of 'binom' (line 355)
        binom_579357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 14), 'binom', False)
        # Obtaining the member 'cdf' of a type (line 355)
        cdf_579358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 14), binom_579357, 'cdf')
        # Calling cdf(args, kwargs) (line 355)
        cdf_call_result_579367 = invoke(stypy.reporting.localization.Localization(__file__, 355, 14), cdf_579358, *[result_sub_579363, n_579364, float_579365], **kwargs_579366)
        
        
        # Call to cdf(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'k' (line 355)
        k_579370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 49), 'k', False)
        # Getting the type of 'n' (line 355)
        n_579371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 51), 'n', False)
        float_579372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 53), 'float')
        # Processing the call keyword arguments (line 355)
        kwargs_579373 = {}
        # Getting the type of 'binom' (line 355)
        binom_579368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 39), 'binom', False)
        # Obtaining the member 'cdf' of a type (line 355)
        cdf_579369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 39), binom_579368, 'cdf')
        # Calling cdf(args, kwargs) (line 355)
        cdf_call_result_579374 = invoke(stypy.reporting.localization.Localization(__file__, 355, 39), cdf_579369, *[k_579370, n_579371, float_579372], **kwargs_579373)
        
        # Applying the binary operator '-' (line 355)
        result_sub_579375 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 14), '-', cdf_call_result_579367, cdf_call_result_579374)
        
        # Assigning a type to the variable 'gkk' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'gkk', result_sub_579375)
        
        # Assigning a BinOp to a Name (line 356):
        
        # Assigning a BinOp to a Name (line 356):
        # Getting the type of 'gk' (line 356)
        gk_579376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 13), 'gk')
        int_579377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 18), 'int')
        # Applying the binary operator '-' (line 356)
        result_sub_579378 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 13), '-', gk_579376, int_579377)
        
        # Getting the type of 'alpha' (line 356)
        alpha_579379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'alpha')
        # Applying the binary operator '+' (line 356)
        result_add_579380 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 20), '+', result_sub_579378, alpha_579379)
        
        # Getting the type of 'gk' (line 356)
        gk_579381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 30), 'gk')
        # Getting the type of 'gkk' (line 356)
        gkk_579382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 35), 'gkk')
        # Applying the binary operator '-' (line 356)
        result_sub_579383 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 30), '-', gk_579381, gkk_579382)
        
        # Applying the binary operator 'div' (line 356)
        result_div_579384 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 12), 'div', result_add_579380, result_sub_579383)
        
        # Assigning a type to the variable 'I' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'I', result_div_579384)
        
        # Assigning a BinOp to a Name (line 357):
        
        # Assigning a BinOp to a Name (line 357):
        # Getting the type of 'n' (line 357)
        n_579385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 17), 'n')
        # Getting the type of 'k' (line 357)
        k_579386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 19), 'k')
        # Applying the binary operator '-' (line 357)
        result_sub_579387 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 17), '-', n_579385, k_579386)
        
        # Getting the type of 'I' (line 357)
        I_579388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 24), 'I')
        # Applying the binary operator '*' (line 357)
        result_mul_579389 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 16), '*', result_sub_579387, I_579388)
        
        
        # Call to float(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'k' (line 357)
        k_579391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 34), 'k', False)
        # Getting the type of 'n' (line 357)
        n_579392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 39), 'n', False)
        int_579393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 41), 'int')
        # Getting the type of 'k' (line 357)
        k_579394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 43), 'k', False)
        # Applying the binary operator '*' (line 357)
        result_mul_579395 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 41), '*', int_579393, k_579394)
        
        # Applying the binary operator '-' (line 357)
        result_sub_579396 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 39), '-', n_579392, result_mul_579395)
        
        # Getting the type of 'I' (line 357)
        I_579397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 46), 'I', False)
        # Applying the binary operator '*' (line 357)
        result_mul_579398 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 38), '*', result_sub_579396, I_579397)
        
        # Applying the binary operator '+' (line 357)
        result_add_579399 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 34), '+', k_579391, result_mul_579398)
        
        # Processing the call keyword arguments (line 357)
        kwargs_579400 = {}
        # Getting the type of 'float' (line 357)
        float_579390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 28), 'float', False)
        # Calling float(args, kwargs) (line 357)
        float_call_result_579401 = invoke(stypy.reporting.localization.Localization(__file__, 357, 28), float_579390, *[result_add_579399], **kwargs_579400)
        
        # Applying the binary operator 'div' (line 357)
        result_div_579402 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 26), 'div', result_mul_579389, float_call_result_579401)
        
        # Assigning a type to the variable 'lambd' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'lambd', result_div_579402)
        
        # Assigning a Tuple to a Name (line 358):
        
        # Assigning a Tuple to a Name (line 358):
        
        # Obtaining an instance of the builtin type 'tuple' (line 358)
        tuple_579403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 358)
        # Adding element type (line 358)
        # Getting the type of 'lambd' (line 358)
        lambd_579404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'lambd')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 358)
        k_579405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'k')
        # Getting the type of 'data' (line 358)
        data_579406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 22), 'data')
        # Obtaining the member '__getitem__' of a type (line 358)
        getitem___579407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 22), data_579406, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 358)
        subscript_call_result_579408 = invoke(stypy.reporting.localization.Localization(__file__, 358, 22), getitem___579407, k_579405)
        
        # Applying the binary operator '*' (line 358)
        result_mul_579409 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 16), '*', lambd_579404, subscript_call_result_579408)
        
        int_579410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 33), 'int')
        # Getting the type of 'lambd' (line 358)
        lambd_579411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 35), 'lambd')
        # Applying the binary operator '-' (line 358)
        result_sub_579412 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 33), '-', int_579410, lambd_579411)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 358)
        k_579413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 47), 'k')
        int_579414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 49), 'int')
        # Applying the binary operator '-' (line 358)
        result_sub_579415 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 47), '-', k_579413, int_579414)
        
        # Getting the type of 'data' (line 358)
        data_579416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 42), 'data')
        # Obtaining the member '__getitem__' of a type (line 358)
        getitem___579417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 42), data_579416, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 358)
        subscript_call_result_579418 = invoke(stypy.reporting.localization.Localization(__file__, 358, 42), getitem___579417, result_sub_579415)
        
        # Applying the binary operator '*' (line 358)
        result_mul_579419 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 32), '*', result_sub_579412, subscript_call_result_579418)
        
        # Applying the binary operator '+' (line 358)
        result_add_579420 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 16), '+', result_mul_579409, result_mul_579419)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 16), tuple_579403, result_add_579420)
        # Adding element type (line 358)
        # Getting the type of 'lambd' (line 359)
        lambd_579421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'lambd')
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 359)
        n_579422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 27), 'n')
        # Getting the type of 'k' (line 359)
        k_579423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 29), 'k')
        # Applying the binary operator '-' (line 359)
        result_sub_579424 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 27), '-', n_579422, k_579423)
        
        int_579425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 31), 'int')
        # Applying the binary operator '-' (line 359)
        result_sub_579426 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 30), '-', result_sub_579424, int_579425)
        
        # Getting the type of 'data' (line 359)
        data_579427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 22), 'data')
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___579428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 22), data_579427, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_579429 = invoke(stypy.reporting.localization.Localization(__file__, 359, 22), getitem___579428, result_sub_579426)
        
        # Applying the binary operator '*' (line 359)
        result_mul_579430 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 16), '*', lambd_579421, subscript_call_result_579429)
        
        int_579431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 37), 'int')
        # Getting the type of 'lambd' (line 359)
        lambd_579432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 39), 'lambd')
        # Applying the binary operator '-' (line 359)
        result_sub_579433 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 37), '-', int_579431, lambd_579432)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 359)
        n_579434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 51), 'n')
        # Getting the type of 'k' (line 359)
        k_579435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 53), 'k')
        # Applying the binary operator '-' (line 359)
        result_sub_579436 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 51), '-', n_579434, k_579435)
        
        # Getting the type of 'data' (line 359)
        data_579437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 46), 'data')
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___579438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 46), data_579437, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_579439 = invoke(stypy.reporting.localization.Localization(__file__, 359, 46), getitem___579438, result_sub_579436)
        
        # Applying the binary operator '*' (line 359)
        result_mul_579440 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 36), '*', result_sub_579433, subscript_call_result_579439)
        
        # Applying the binary operator '+' (line 359)
        result_add_579441 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 16), '+', result_mul_579430, result_mul_579440)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 16), tuple_579403, result_add_579441)
        
        # Assigning a type to the variable 'lims' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'lims', tuple_579403)
        # Getting the type of 'lims' (line 360)
        lims_579442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'lims')
        # Assigning a type to the variable 'stypy_return_type' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'stypy_return_type', lims_579442)
        
        # ################# End of '_cihs_1D(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cihs_1D' in the type store
        # Getting the type of 'stypy_return_type' (line 346)
        stypy_return_type_579443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_579443)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cihs_1D'
        return stypy_return_type_579443

    # Assigning a type to the variable '_cihs_1D' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), '_cihs_1D', _cihs_1D)
    
    # Assigning a Call to a Name (line 361):
    
    # Assigning a Call to a Name (line 361):
    
    # Call to array(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'data' (line 361)
    data_579446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'data', False)
    # Processing the call keyword arguments (line 361)
    # Getting the type of 'False' (line 361)
    False_579447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 31), 'False', False)
    keyword_579448 = False_579447
    kwargs_579449 = {'copy': keyword_579448}
    # Getting the type of 'ma' (line 361)
    ma_579444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 11), 'ma', False)
    # Obtaining the member 'array' of a type (line 361)
    array_579445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 11), ma_579444, 'array')
    # Calling array(args, kwargs) (line 361)
    array_call_result_579450 = invoke(stypy.reporting.localization.Localization(__file__, 361, 11), array_579445, *[data_579446], **kwargs_579449)
    
    # Assigning a type to the variable 'data' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'data', array_call_result_579450)
    
    # Type idiom detected: calculating its left and rigth part (line 363)
    # Getting the type of 'axis' (line 363)
    axis_579451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'axis')
    # Getting the type of 'None' (line 363)
    None_579452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'None')
    
    (may_be_579453, more_types_in_union_579454) = may_be_none(axis_579451, None_579452)

    if may_be_579453:

        if more_types_in_union_579454:
            # Runtime conditional SSA (line 363)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 364):
        
        # Assigning a Call to a Name (line 364):
        
        # Call to _cihs_1D(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'data' (line 364)
        data_579456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 26), 'data', False)
        # Getting the type of 'alpha' (line 364)
        alpha_579457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 32), 'alpha', False)
        # Processing the call keyword arguments (line 364)
        kwargs_579458 = {}
        # Getting the type of '_cihs_1D' (line 364)
        _cihs_1D_579455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 17), '_cihs_1D', False)
        # Calling _cihs_1D(args, kwargs) (line 364)
        _cihs_1D_call_result_579459 = invoke(stypy.reporting.localization.Localization(__file__, 364, 17), _cihs_1D_579455, *[data_579456, alpha_579457], **kwargs_579458)
        
        # Assigning a type to the variable 'result' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'result', _cihs_1D_call_result_579459)

        if more_types_in_union_579454:
            # Runtime conditional SSA for else branch (line 363)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_579453) or more_types_in_union_579454):
        
        
        # Getting the type of 'data' (line 366)
        data_579460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 11), 'data')
        # Obtaining the member 'ndim' of a type (line 366)
        ndim_579461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 11), data_579460, 'ndim')
        int_579462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 23), 'int')
        # Applying the binary operator '>' (line 366)
        result_gt_579463 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 11), '>', ndim_579461, int_579462)
        
        # Testing the type of an if condition (line 366)
        if_condition_579464 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 8), result_gt_579463)
        # Assigning a type to the variable 'if_condition_579464' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'if_condition_579464', if_condition_579464)
        # SSA begins for if statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 367)
        # Processing the call arguments (line 367)
        str_579466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 29), 'str', "Array 'data' must be at most two dimensional, but got data.ndim = %d")
        # Getting the type of 'data' (line 368)
        data_579467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 56), 'data', False)
        # Obtaining the member 'ndim' of a type (line 368)
        ndim_579468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 56), data_579467, 'ndim')
        # Applying the binary operator '%' (line 367)
        result_mod_579469 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 29), '%', str_579466, ndim_579468)
        
        # Processing the call keyword arguments (line 367)
        kwargs_579470 = {}
        # Getting the type of 'ValueError' (line 367)
        ValueError_579465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 367)
        ValueError_call_result_579471 = invoke(stypy.reporting.localization.Localization(__file__, 367, 18), ValueError_579465, *[result_mod_579469], **kwargs_579470)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 367, 12), ValueError_call_result_579471, 'raise parameter', BaseException)
        # SSA join for if statement (line 366)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 369):
        
        # Assigning a Call to a Name (line 369):
        
        # Call to apply_along_axis(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of '_cihs_1D' (line 369)
        _cihs_1D_579474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 37), '_cihs_1D', False)
        # Getting the type of 'axis' (line 369)
        axis_579475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 47), 'axis', False)
        # Getting the type of 'data' (line 369)
        data_579476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 53), 'data', False)
        # Getting the type of 'alpha' (line 369)
        alpha_579477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 59), 'alpha', False)
        # Processing the call keyword arguments (line 369)
        kwargs_579478 = {}
        # Getting the type of 'ma' (line 369)
        ma_579472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 17), 'ma', False)
        # Obtaining the member 'apply_along_axis' of a type (line 369)
        apply_along_axis_579473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 17), ma_579472, 'apply_along_axis')
        # Calling apply_along_axis(args, kwargs) (line 369)
        apply_along_axis_call_result_579479 = invoke(stypy.reporting.localization.Localization(__file__, 369, 17), apply_along_axis_579473, *[_cihs_1D_579474, axis_579475, data_579476, alpha_579477], **kwargs_579478)
        
        # Assigning a type to the variable 'result' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'result', apply_along_axis_call_result_579479)

        if (may_be_579453 and more_types_in_union_579454):
            # SSA join for if statement (line 363)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'result' (line 371)
    result_579480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type', result_579480)
    
    # ################# End of 'median_cihs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'median_cihs' in the type store
    # Getting the type of 'stypy_return_type' (line 323)
    stypy_return_type_579481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_579481)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'median_cihs'
    return stypy_return_type_579481

# Assigning a type to the variable 'median_cihs' (line 323)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'median_cihs', median_cihs)

@norecursion
def compare_medians_ms(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 374)
    None_579482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 46), 'None')
    defaults = [None_579482]
    # Create a new context for function 'compare_medians_ms'
    module_type_store = module_type_store.open_function_context('compare_medians_ms', 374, 0, False)
    
    # Passed parameters checking function
    compare_medians_ms.stypy_localization = localization
    compare_medians_ms.stypy_type_of_self = None
    compare_medians_ms.stypy_type_store = module_type_store
    compare_medians_ms.stypy_function_name = 'compare_medians_ms'
    compare_medians_ms.stypy_param_names_list = ['group_1', 'group_2', 'axis']
    compare_medians_ms.stypy_varargs_param_name = None
    compare_medians_ms.stypy_kwargs_param_name = None
    compare_medians_ms.stypy_call_defaults = defaults
    compare_medians_ms.stypy_call_varargs = varargs
    compare_medians_ms.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compare_medians_ms', ['group_1', 'group_2', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compare_medians_ms', localization, ['group_1', 'group_2', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compare_medians_ms(...)' code ##################

    str_579483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, (-1)), 'str', '\n    Compares the medians from two independent groups along the given axis.\n\n    The comparison is performed using the McKean-Schrader estimate of the\n    standard error of the medians.\n\n    Parameters\n    ----------\n    group_1 : array_like\n        First dataset.  Has to be of size >=7.\n    group_2 : array_like\n        Second dataset.  Has to be of size >=7.\n    axis : int, optional\n        Axis along which the medians are estimated. If None, the arrays are\n        flattened.  If `axis` is not None, then `group_1` and `group_2`\n        should have the same shape.\n\n    Returns\n    -------\n    compare_medians_ms : {float, ndarray}\n        If `axis` is None, then returns a float, otherwise returns a 1-D\n        ndarray of floats with a length equal to the length of `group_1`\n        along `axis`.\n\n    ')
    
    # Assigning a Tuple to a Tuple (line 400):
    
    # Assigning a Call to a Name (line 400):
    
    # Call to median(...): (line 400)
    # Processing the call arguments (line 400)
    # Getting the type of 'group_1' (line 400)
    group_1_579486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 32), 'group_1', False)
    # Processing the call keyword arguments (line 400)
    # Getting the type of 'axis' (line 400)
    axis_579487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 45), 'axis', False)
    keyword_579488 = axis_579487
    kwargs_579489 = {'axis': keyword_579488}
    # Getting the type of 'ma' (line 400)
    ma_579484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 22), 'ma', False)
    # Obtaining the member 'median' of a type (line 400)
    median_579485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 22), ma_579484, 'median')
    # Calling median(args, kwargs) (line 400)
    median_call_result_579490 = invoke(stypy.reporting.localization.Localization(__file__, 400, 22), median_579485, *[group_1_579486], **kwargs_579489)
    
    # Assigning a type to the variable 'tuple_assignment_578467' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'tuple_assignment_578467', median_call_result_579490)
    
    # Assigning a Call to a Name (line 400):
    
    # Call to median(...): (line 400)
    # Processing the call arguments (line 400)
    # Getting the type of 'group_2' (line 400)
    group_2_579493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 62), 'group_2', False)
    # Processing the call keyword arguments (line 400)
    # Getting the type of 'axis' (line 400)
    axis_579494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 75), 'axis', False)
    keyword_579495 = axis_579494
    kwargs_579496 = {'axis': keyword_579495}
    # Getting the type of 'ma' (line 400)
    ma_579491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 52), 'ma', False)
    # Obtaining the member 'median' of a type (line 400)
    median_579492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 52), ma_579491, 'median')
    # Calling median(args, kwargs) (line 400)
    median_call_result_579497 = invoke(stypy.reporting.localization.Localization(__file__, 400, 52), median_579492, *[group_2_579493], **kwargs_579496)
    
    # Assigning a type to the variable 'tuple_assignment_578468' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'tuple_assignment_578468', median_call_result_579497)
    
    # Assigning a Name to a Name (line 400):
    # Getting the type of 'tuple_assignment_578467' (line 400)
    tuple_assignment_578467_579498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'tuple_assignment_578467')
    # Assigning a type to the variable 'med_1' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 5), 'med_1', tuple_assignment_578467_579498)
    
    # Assigning a Name to a Name (line 400):
    # Getting the type of 'tuple_assignment_578468' (line 400)
    tuple_assignment_578468_579499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'tuple_assignment_578468')
    # Assigning a type to the variable 'med_2' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'med_2', tuple_assignment_578468_579499)
    
    # Assigning a Tuple to a Tuple (line 401):
    
    # Assigning a Call to a Name (line 401):
    
    # Call to stde_median(...): (line 401)
    # Processing the call arguments (line 401)
    # Getting the type of 'group_1' (line 401)
    group_1_579502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 41), 'group_1', False)
    # Processing the call keyword arguments (line 401)
    # Getting the type of 'axis' (line 401)
    axis_579503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 55), 'axis', False)
    keyword_579504 = axis_579503
    kwargs_579505 = {'axis': keyword_579504}
    # Getting the type of 'mstats' (line 401)
    mstats_579500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 22), 'mstats', False)
    # Obtaining the member 'stde_median' of a type (line 401)
    stde_median_579501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 22), mstats_579500, 'stde_median')
    # Calling stde_median(args, kwargs) (line 401)
    stde_median_call_result_579506 = invoke(stypy.reporting.localization.Localization(__file__, 401, 22), stde_median_579501, *[group_1_579502], **kwargs_579505)
    
    # Assigning a type to the variable 'tuple_assignment_578469' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'tuple_assignment_578469', stde_median_call_result_579506)
    
    # Assigning a Call to a Name (line 401):
    
    # Call to stde_median(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'group_2' (line 402)
    group_2_579509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 41), 'group_2', False)
    # Processing the call keyword arguments (line 402)
    # Getting the type of 'axis' (line 402)
    axis_579510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 55), 'axis', False)
    keyword_579511 = axis_579510
    kwargs_579512 = {'axis': keyword_579511}
    # Getting the type of 'mstats' (line 402)
    mstats_579507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 22), 'mstats', False)
    # Obtaining the member 'stde_median' of a type (line 402)
    stde_median_579508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 22), mstats_579507, 'stde_median')
    # Calling stde_median(args, kwargs) (line 402)
    stde_median_call_result_579513 = invoke(stypy.reporting.localization.Localization(__file__, 402, 22), stde_median_579508, *[group_2_579509], **kwargs_579512)
    
    # Assigning a type to the variable 'tuple_assignment_578470' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'tuple_assignment_578470', stde_median_call_result_579513)
    
    # Assigning a Name to a Name (line 401):
    # Getting the type of 'tuple_assignment_578469' (line 401)
    tuple_assignment_578469_579514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'tuple_assignment_578469')
    # Assigning a type to the variable 'std_1' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 5), 'std_1', tuple_assignment_578469_579514)
    
    # Assigning a Name to a Name (line 401):
    # Getting the type of 'tuple_assignment_578470' (line 401)
    tuple_assignment_578470_579515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'tuple_assignment_578470')
    # Assigning a type to the variable 'std_2' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'std_2', tuple_assignment_578470_579515)
    
    # Assigning a BinOp to a Name (line 403):
    
    # Assigning a BinOp to a Name (line 403):
    
    # Call to abs(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'med_1' (line 403)
    med_1_579518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 15), 'med_1', False)
    # Getting the type of 'med_2' (line 403)
    med_2_579519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 23), 'med_2', False)
    # Applying the binary operator '-' (line 403)
    result_sub_579520 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 15), '-', med_1_579518, med_2_579519)
    
    # Processing the call keyword arguments (line 403)
    kwargs_579521 = {}
    # Getting the type of 'np' (line 403)
    np_579516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'np', False)
    # Obtaining the member 'abs' of a type (line 403)
    abs_579517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), np_579516, 'abs')
    # Calling abs(args, kwargs) (line 403)
    abs_call_result_579522 = invoke(stypy.reporting.localization.Localization(__file__, 403, 8), abs_579517, *[result_sub_579520], **kwargs_579521)
    
    
    # Call to sqrt(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'std_1' (line 403)
    std_1_579525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 40), 'std_1', False)
    int_579526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 47), 'int')
    # Applying the binary operator '**' (line 403)
    result_pow_579527 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 40), '**', std_1_579525, int_579526)
    
    # Getting the type of 'std_2' (line 403)
    std_2_579528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 51), 'std_2', False)
    int_579529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 58), 'int')
    # Applying the binary operator '**' (line 403)
    result_pow_579530 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 51), '**', std_2_579528, int_579529)
    
    # Applying the binary operator '+' (line 403)
    result_add_579531 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 40), '+', result_pow_579527, result_pow_579530)
    
    # Processing the call keyword arguments (line 403)
    kwargs_579532 = {}
    # Getting the type of 'ma' (line 403)
    ma_579523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 32), 'ma', False)
    # Obtaining the member 'sqrt' of a type (line 403)
    sqrt_579524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 32), ma_579523, 'sqrt')
    # Calling sqrt(args, kwargs) (line 403)
    sqrt_call_result_579533 = invoke(stypy.reporting.localization.Localization(__file__, 403, 32), sqrt_579524, *[result_add_579531], **kwargs_579532)
    
    # Applying the binary operator 'div' (line 403)
    result_div_579534 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 8), 'div', abs_call_result_579522, sqrt_call_result_579533)
    
    # Assigning a type to the variable 'W' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'W', result_div_579534)
    int_579535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 11), 'int')
    
    # Call to cdf(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'W' (line 404)
    W_579538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 24), 'W', False)
    # Processing the call keyword arguments (line 404)
    kwargs_579539 = {}
    # Getting the type of 'norm' (line 404)
    norm_579536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'norm', False)
    # Obtaining the member 'cdf' of a type (line 404)
    cdf_579537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 15), norm_579536, 'cdf')
    # Calling cdf(args, kwargs) (line 404)
    cdf_call_result_579540 = invoke(stypy.reporting.localization.Localization(__file__, 404, 15), cdf_579537, *[W_579538], **kwargs_579539)
    
    # Applying the binary operator '-' (line 404)
    result_sub_579541 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 11), '-', int_579535, cdf_call_result_579540)
    
    # Assigning a type to the variable 'stypy_return_type' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'stypy_return_type', result_sub_579541)
    
    # ################# End of 'compare_medians_ms(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compare_medians_ms' in the type store
    # Getting the type of 'stypy_return_type' (line 374)
    stypy_return_type_579542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_579542)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compare_medians_ms'
    return stypy_return_type_579542

# Assigning a type to the variable 'compare_medians_ms' (line 374)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'compare_medians_ms', compare_medians_ms)

@norecursion
def idealfourths(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 407)
    None_579543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 28), 'None')
    defaults = [None_579543]
    # Create a new context for function 'idealfourths'
    module_type_store = module_type_store.open_function_context('idealfourths', 407, 0, False)
    
    # Passed parameters checking function
    idealfourths.stypy_localization = localization
    idealfourths.stypy_type_of_self = None
    idealfourths.stypy_type_store = module_type_store
    idealfourths.stypy_function_name = 'idealfourths'
    idealfourths.stypy_param_names_list = ['data', 'axis']
    idealfourths.stypy_varargs_param_name = None
    idealfourths.stypy_kwargs_param_name = None
    idealfourths.stypy_call_defaults = defaults
    idealfourths.stypy_call_varargs = varargs
    idealfourths.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idealfourths', ['data', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idealfourths', localization, ['data', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idealfourths(...)' code ##################

    str_579544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, (-1)), 'str', '\n    Returns an estimate of the lower and upper quartiles.\n\n    Uses the ideal fourths algorithm.\n\n    Parameters\n    ----------\n    data : array_like\n        Input array.\n    axis : int, optional\n        Axis along which the quartiles are estimated. If None, the arrays are\n        flattened.\n\n    Returns\n    -------\n    idealfourths : {list of floats, masked array}\n        Returns the two internal values that divide `data` into four parts\n        using the ideal fourths algorithm either along the flattened array\n        (if `axis` is None) or along `axis` of `data`.\n\n    ')

    @norecursion
    def _idf(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_idf'
        module_type_store = module_type_store.open_function_context('_idf', 429, 4, False)
        
        # Passed parameters checking function
        _idf.stypy_localization = localization
        _idf.stypy_type_of_self = None
        _idf.stypy_type_store = module_type_store
        _idf.stypy_function_name = '_idf'
        _idf.stypy_param_names_list = ['data']
        _idf.stypy_varargs_param_name = None
        _idf.stypy_kwargs_param_name = None
        _idf.stypy_call_defaults = defaults
        _idf.stypy_call_varargs = varargs
        _idf.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_idf', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_idf', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_idf(...)' code ##################

        
        # Assigning a Call to a Name (line 430):
        
        # Assigning a Call to a Name (line 430):
        
        # Call to compressed(...): (line 430)
        # Processing the call keyword arguments (line 430)
        kwargs_579547 = {}
        # Getting the type of 'data' (line 430)
        data_579545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'data', False)
        # Obtaining the member 'compressed' of a type (line 430)
        compressed_579546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), data_579545, 'compressed')
        # Calling compressed(args, kwargs) (line 430)
        compressed_call_result_579548 = invoke(stypy.reporting.localization.Localization(__file__, 430, 12), compressed_579546, *[], **kwargs_579547)
        
        # Assigning a type to the variable 'x' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'x', compressed_call_result_579548)
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Call to len(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'x' (line 431)
        x_579550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 16), 'x', False)
        # Processing the call keyword arguments (line 431)
        kwargs_579551 = {}
        # Getting the type of 'len' (line 431)
        len_579549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'len', False)
        # Calling len(args, kwargs) (line 431)
        len_call_result_579552 = invoke(stypy.reporting.localization.Localization(__file__, 431, 12), len_579549, *[x_579550], **kwargs_579551)
        
        # Assigning a type to the variable 'n' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'n', len_call_result_579552)
        
        
        # Getting the type of 'n' (line 432)
        n_579553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 11), 'n')
        int_579554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 15), 'int')
        # Applying the binary operator '<' (line 432)
        result_lt_579555 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 11), '<', n_579553, int_579554)
        
        # Testing the type of an if condition (line 432)
        if_condition_579556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 8), result_lt_579555)
        # Assigning a type to the variable 'if_condition_579556' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'if_condition_579556', if_condition_579556)
        # SSA begins for if statement (line 432)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_579557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        # Getting the type of 'np' (line 433)
        np_579558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'np')
        # Obtaining the member 'nan' of a type (line 433)
        nan_579559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 20), np_579558, 'nan')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 19), list_579557, nan_579559)
        # Adding element type (line 433)
        # Getting the type of 'np' (line 433)
        np_579560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 27), 'np')
        # Obtaining the member 'nan' of a type (line 433)
        nan_579561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 27), np_579560, 'nan')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 19), list_579557, nan_579561)
        
        # Assigning a type to the variable 'stypy_return_type' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'stypy_return_type', list_579557)
        # SSA join for if statement (line 432)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 434):
        
        # Assigning a Subscript to a Name (line 434):
        
        # Obtaining the type of the subscript
        int_579562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 8), 'int')
        
        # Call to divmod(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'n' (line 434)
        n_579564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 23), 'n', False)
        float_579565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 25), 'float')
        # Applying the binary operator 'div' (line 434)
        result_div_579566 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 23), 'div', n_579564, float_579565)
        
        int_579567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 30), 'int')
        float_579568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 32), 'float')
        # Applying the binary operator 'div' (line 434)
        result_div_579569 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 30), 'div', int_579567, float_579568)
        
        # Applying the binary operator '+' (line 434)
        result_add_579570 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 23), '+', result_div_579566, result_div_579569)
        
        int_579571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 36), 'int')
        # Processing the call keyword arguments (line 434)
        kwargs_579572 = {}
        # Getting the type of 'divmod' (line 434)
        divmod_579563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'divmod', False)
        # Calling divmod(args, kwargs) (line 434)
        divmod_call_result_579573 = invoke(stypy.reporting.localization.Localization(__file__, 434, 16), divmod_579563, *[result_add_579570, int_579571], **kwargs_579572)
        
        # Obtaining the member '__getitem__' of a type (line 434)
        getitem___579574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), divmod_call_result_579573, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
        subscript_call_result_579575 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), getitem___579574, int_579562)
        
        # Assigning a type to the variable 'tuple_var_assignment_578471' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'tuple_var_assignment_578471', subscript_call_result_579575)
        
        # Assigning a Subscript to a Name (line 434):
        
        # Obtaining the type of the subscript
        int_579576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 8), 'int')
        
        # Call to divmod(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'n' (line 434)
        n_579578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 23), 'n', False)
        float_579579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 25), 'float')
        # Applying the binary operator 'div' (line 434)
        result_div_579580 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 23), 'div', n_579578, float_579579)
        
        int_579581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 30), 'int')
        float_579582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 32), 'float')
        # Applying the binary operator 'div' (line 434)
        result_div_579583 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 30), 'div', int_579581, float_579582)
        
        # Applying the binary operator '+' (line 434)
        result_add_579584 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 23), '+', result_div_579580, result_div_579583)
        
        int_579585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 36), 'int')
        # Processing the call keyword arguments (line 434)
        kwargs_579586 = {}
        # Getting the type of 'divmod' (line 434)
        divmod_579577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'divmod', False)
        # Calling divmod(args, kwargs) (line 434)
        divmod_call_result_579587 = invoke(stypy.reporting.localization.Localization(__file__, 434, 16), divmod_579577, *[result_add_579584, int_579585], **kwargs_579586)
        
        # Obtaining the member '__getitem__' of a type (line 434)
        getitem___579588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), divmod_call_result_579587, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
        subscript_call_result_579589 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), getitem___579588, int_579576)
        
        # Assigning a type to the variable 'tuple_var_assignment_578472' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'tuple_var_assignment_578472', subscript_call_result_579589)
        
        # Assigning a Name to a Name (line 434):
        # Getting the type of 'tuple_var_assignment_578471' (line 434)
        tuple_var_assignment_578471_579590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'tuple_var_assignment_578471')
        # Assigning a type to the variable 'j' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 9), 'j', tuple_var_assignment_578471_579590)
        
        # Assigning a Name to a Name (line 434):
        # Getting the type of 'tuple_var_assignment_578472' (line 434)
        tuple_var_assignment_578472_579591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'tuple_var_assignment_578472')
        # Assigning a type to the variable 'h' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 11), 'h', tuple_var_assignment_578472_579591)
        
        # Assigning a Call to a Name (line 435):
        
        # Assigning a Call to a Name (line 435):
        
        # Call to int(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'j' (line 435)
        j_579593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 16), 'j', False)
        # Processing the call keyword arguments (line 435)
        kwargs_579594 = {}
        # Getting the type of 'int' (line 435)
        int_579592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'int', False)
        # Calling int(args, kwargs) (line 435)
        int_call_result_579595 = invoke(stypy.reporting.localization.Localization(__file__, 435, 12), int_579592, *[j_579593], **kwargs_579594)
        
        # Assigning a type to the variable 'j' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'j', int_call_result_579595)
        
        # Assigning a BinOp to a Name (line 436):
        
        # Assigning a BinOp to a Name (line 436):
        int_579596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 15), 'int')
        # Getting the type of 'h' (line 436)
        h_579597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 17), 'h')
        # Applying the binary operator '-' (line 436)
        result_sub_579598 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 15), '-', int_579596, h_579597)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 436)
        j_579599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 22), 'j')
        int_579600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 24), 'int')
        # Applying the binary operator '-' (line 436)
        result_sub_579601 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 22), '-', j_579599, int_579600)
        
        # Getting the type of 'x' (line 436)
        x_579602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 20), 'x')
        # Obtaining the member '__getitem__' of a type (line 436)
        getitem___579603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 20), x_579602, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 436)
        subscript_call_result_579604 = invoke(stypy.reporting.localization.Localization(__file__, 436, 20), getitem___579603, result_sub_579601)
        
        # Applying the binary operator '*' (line 436)
        result_mul_579605 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 14), '*', result_sub_579598, subscript_call_result_579604)
        
        # Getting the type of 'h' (line 436)
        h_579606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 29), 'h')
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 436)
        j_579607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 33), 'j')
        # Getting the type of 'x' (line 436)
        x_579608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 31), 'x')
        # Obtaining the member '__getitem__' of a type (line 436)
        getitem___579609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 31), x_579608, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 436)
        subscript_call_result_579610 = invoke(stypy.reporting.localization.Localization(__file__, 436, 31), getitem___579609, j_579607)
        
        # Applying the binary operator '*' (line 436)
        result_mul_579611 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 29), '*', h_579606, subscript_call_result_579610)
        
        # Applying the binary operator '+' (line 436)
        result_add_579612 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 14), '+', result_mul_579605, result_mul_579611)
        
        # Assigning a type to the variable 'qlo' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'qlo', result_add_579612)
        
        # Assigning a BinOp to a Name (line 437):
        
        # Assigning a BinOp to a Name (line 437):
        # Getting the type of 'n' (line 437)
        n_579613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'n')
        # Getting the type of 'j' (line 437)
        j_579614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 16), 'j')
        # Applying the binary operator '-' (line 437)
        result_sub_579615 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 12), '-', n_579613, j_579614)
        
        # Assigning a type to the variable 'k' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'k', result_sub_579615)
        
        # Assigning a BinOp to a Name (line 438):
        
        # Assigning a BinOp to a Name (line 438):
        int_579616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 15), 'int')
        # Getting the type of 'h' (line 438)
        h_579617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 17), 'h')
        # Applying the binary operator '-' (line 438)
        result_sub_579618 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), '-', int_579616, h_579617)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 438)
        k_579619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), 'k')
        # Getting the type of 'x' (line 438)
        x_579620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 20), 'x')
        # Obtaining the member '__getitem__' of a type (line 438)
        getitem___579621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 20), x_579620, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 438)
        subscript_call_result_579622 = invoke(stypy.reporting.localization.Localization(__file__, 438, 20), getitem___579621, k_579619)
        
        # Applying the binary operator '*' (line 438)
        result_mul_579623 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 14), '*', result_sub_579618, subscript_call_result_579622)
        
        # Getting the type of 'h' (line 438)
        h_579624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 27), 'h')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 438)
        k_579625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 31), 'k')
        int_579626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 33), 'int')
        # Applying the binary operator '-' (line 438)
        result_sub_579627 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 31), '-', k_579625, int_579626)
        
        # Getting the type of 'x' (line 438)
        x_579628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 29), 'x')
        # Obtaining the member '__getitem__' of a type (line 438)
        getitem___579629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 29), x_579628, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 438)
        subscript_call_result_579630 = invoke(stypy.reporting.localization.Localization(__file__, 438, 29), getitem___579629, result_sub_579627)
        
        # Applying the binary operator '*' (line 438)
        result_mul_579631 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 27), '*', h_579624, subscript_call_result_579630)
        
        # Applying the binary operator '+' (line 438)
        result_add_579632 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 14), '+', result_mul_579623, result_mul_579631)
        
        # Assigning a type to the variable 'qup' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'qup', result_add_579632)
        
        # Obtaining an instance of the builtin type 'list' (line 439)
        list_579633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 439)
        # Adding element type (line 439)
        # Getting the type of 'qlo' (line 439)
        qlo_579634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'qlo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 15), list_579633, qlo_579634)
        # Adding element type (line 439)
        # Getting the type of 'qup' (line 439)
        qup_579635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), 'qup')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 15), list_579633, qup_579635)
        
        # Assigning a type to the variable 'stypy_return_type' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'stypy_return_type', list_579633)
        
        # ################# End of '_idf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_idf' in the type store
        # Getting the type of 'stypy_return_type' (line 429)
        stypy_return_type_579636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_579636)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_idf'
        return stypy_return_type_579636

    # Assigning a type to the variable '_idf' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), '_idf', _idf)
    
    # Assigning a Call to a Name (line 440):
    
    # Assigning a Call to a Name (line 440):
    
    # Call to view(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'MaskedArray' (line 440)
    MaskedArray_579645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 41), 'MaskedArray', False)
    # Processing the call keyword arguments (line 440)
    kwargs_579646 = {}
    
    # Call to sort(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'data' (line 440)
    data_579639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 19), 'data', False)
    # Processing the call keyword arguments (line 440)
    # Getting the type of 'axis' (line 440)
    axis_579640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 30), 'axis', False)
    keyword_579641 = axis_579640
    kwargs_579642 = {'axis': keyword_579641}
    # Getting the type of 'ma' (line 440)
    ma_579637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 11), 'ma', False)
    # Obtaining the member 'sort' of a type (line 440)
    sort_579638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 11), ma_579637, 'sort')
    # Calling sort(args, kwargs) (line 440)
    sort_call_result_579643 = invoke(stypy.reporting.localization.Localization(__file__, 440, 11), sort_579638, *[data_579639], **kwargs_579642)
    
    # Obtaining the member 'view' of a type (line 440)
    view_579644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 11), sort_call_result_579643, 'view')
    # Calling view(args, kwargs) (line 440)
    view_call_result_579647 = invoke(stypy.reporting.localization.Localization(__file__, 440, 11), view_579644, *[MaskedArray_579645], **kwargs_579646)
    
    # Assigning a type to the variable 'data' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'data', view_call_result_579647)
    
    # Type idiom detected: calculating its left and rigth part (line 441)
    # Getting the type of 'axis' (line 441)
    axis_579648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'axis')
    # Getting the type of 'None' (line 441)
    None_579649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'None')
    
    (may_be_579650, more_types_in_union_579651) = may_be_none(axis_579648, None_579649)

    if may_be_579650:

        if more_types_in_union_579651:
            # Runtime conditional SSA (line 441)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to _idf(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'data' (line 442)
        data_579653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 20), 'data', False)
        # Processing the call keyword arguments (line 442)
        kwargs_579654 = {}
        # Getting the type of '_idf' (line 442)
        _idf_579652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 15), '_idf', False)
        # Calling _idf(args, kwargs) (line 442)
        _idf_call_result_579655 = invoke(stypy.reporting.localization.Localization(__file__, 442, 15), _idf_579652, *[data_579653], **kwargs_579654)
        
        # Assigning a type to the variable 'stypy_return_type' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'stypy_return_type', _idf_call_result_579655)

        if more_types_in_union_579651:
            # Runtime conditional SSA for else branch (line 441)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_579650) or more_types_in_union_579651):
        
        # Call to apply_along_axis(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of '_idf' (line 444)
        _idf_579658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 35), '_idf', False)
        # Getting the type of 'axis' (line 444)
        axis_579659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 41), 'axis', False)
        # Getting the type of 'data' (line 444)
        data_579660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 47), 'data', False)
        # Processing the call keyword arguments (line 444)
        kwargs_579661 = {}
        # Getting the type of 'ma' (line 444)
        ma_579656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 15), 'ma', False)
        # Obtaining the member 'apply_along_axis' of a type (line 444)
        apply_along_axis_579657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 15), ma_579656, 'apply_along_axis')
        # Calling apply_along_axis(args, kwargs) (line 444)
        apply_along_axis_call_result_579662 = invoke(stypy.reporting.localization.Localization(__file__, 444, 15), apply_along_axis_579657, *[_idf_579658, axis_579659, data_579660], **kwargs_579661)
        
        # Assigning a type to the variable 'stypy_return_type' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'stypy_return_type', apply_along_axis_call_result_579662)

        if (may_be_579650 and more_types_in_union_579651):
            # SSA join for if statement (line 441)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'idealfourths(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idealfourths' in the type store
    # Getting the type of 'stypy_return_type' (line 407)
    stypy_return_type_579663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_579663)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idealfourths'
    return stypy_return_type_579663

# Assigning a type to the variable 'idealfourths' (line 407)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 0), 'idealfourths', idealfourths)

@norecursion
def rsh(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 447)
    None_579664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 21), 'None')
    defaults = [None_579664]
    # Create a new context for function 'rsh'
    module_type_store = module_type_store.open_function_context('rsh', 447, 0, False)
    
    # Passed parameters checking function
    rsh.stypy_localization = localization
    rsh.stypy_type_of_self = None
    rsh.stypy_type_store = module_type_store
    rsh.stypy_function_name = 'rsh'
    rsh.stypy_param_names_list = ['data', 'points']
    rsh.stypy_varargs_param_name = None
    rsh.stypy_kwargs_param_name = None
    rsh.stypy_call_defaults = defaults
    rsh.stypy_call_varargs = varargs
    rsh.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rsh', ['data', 'points'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rsh', localization, ['data', 'points'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rsh(...)' code ##################

    str_579665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, (-1)), 'str', "\n    Evaluates Rosenblatt's shifted histogram estimators for each data point.\n\n    Rosenblatt's estimator is a centered finite-difference approximation to the\n    derivative of the empirical cumulative distribution function.\n\n    Parameters\n    ----------\n    data : sequence\n        Input data, should be 1-D. Masked values are ignored.\n    points : sequence or None, optional\n        Sequence of points where to evaluate Rosenblatt shifted histogram.\n        If None, use the data.\n\n    ")
    
    # Assigning a Call to a Name (line 463):
    
    # Assigning a Call to a Name (line 463):
    
    # Call to array(...): (line 463)
    # Processing the call arguments (line 463)
    # Getting the type of 'data' (line 463)
    data_579668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 20), 'data', False)
    # Processing the call keyword arguments (line 463)
    # Getting the type of 'False' (line 463)
    False_579669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 31), 'False', False)
    keyword_579670 = False_579669
    kwargs_579671 = {'copy': keyword_579670}
    # Getting the type of 'ma' (line 463)
    ma_579666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'ma', False)
    # Obtaining the member 'array' of a type (line 463)
    array_579667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 11), ma_579666, 'array')
    # Calling array(args, kwargs) (line 463)
    array_call_result_579672 = invoke(stypy.reporting.localization.Localization(__file__, 463, 11), array_579667, *[data_579668], **kwargs_579671)
    
    # Assigning a type to the variable 'data' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'data', array_call_result_579672)
    
    # Type idiom detected: calculating its left and rigth part (line 464)
    # Getting the type of 'points' (line 464)
    points_579673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 7), 'points')
    # Getting the type of 'None' (line 464)
    None_579674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 17), 'None')
    
    (may_be_579675, more_types_in_union_579676) = may_be_none(points_579673, None_579674)

    if may_be_579675:

        if more_types_in_union_579676:
            # Runtime conditional SSA (line 464)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 465):
        
        # Assigning a Name to a Name (line 465):
        # Getting the type of 'data' (line 465)
        data_579677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 17), 'data')
        # Assigning a type to the variable 'points' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'points', data_579677)

        if more_types_in_union_579676:
            # Runtime conditional SSA for else branch (line 464)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_579675) or more_types_in_union_579676):
        
        # Assigning a Call to a Name (line 467):
        
        # Assigning a Call to a Name (line 467):
        
        # Call to array(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'points' (line 467)
        points_579680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 26), 'points', False)
        # Processing the call keyword arguments (line 467)
        # Getting the type of 'False' (line 467)
        False_579681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 39), 'False', False)
        keyword_579682 = False_579681
        int_579683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 52), 'int')
        keyword_579684 = int_579683
        kwargs_579685 = {'copy': keyword_579682, 'ndmin': keyword_579684}
        # Getting the type of 'np' (line 467)
        np_579678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 467)
        array_579679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 17), np_579678, 'array')
        # Calling array(args, kwargs) (line 467)
        array_call_result_579686 = invoke(stypy.reporting.localization.Localization(__file__, 467, 17), array_579679, *[points_579680], **kwargs_579685)
        
        # Assigning a type to the variable 'points' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'points', array_call_result_579686)

        if (may_be_579675 and more_types_in_union_579676):
            # SSA join for if statement (line 464)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'data' (line 469)
    data_579687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 7), 'data')
    # Obtaining the member 'ndim' of a type (line 469)
    ndim_579688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 7), data_579687, 'ndim')
    int_579689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 20), 'int')
    # Applying the binary operator '!=' (line 469)
    result_ne_579690 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 7), '!=', ndim_579688, int_579689)
    
    # Testing the type of an if condition (line 469)
    if_condition_579691 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 469, 4), result_ne_579690)
    # Assigning a type to the variable 'if_condition_579691' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'if_condition_579691', if_condition_579691)
    # SSA begins for if statement (line 469)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AttributeError(...): (line 470)
    # Processing the call arguments (line 470)
    str_579693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 29), 'str', 'The input array should be 1D only !')
    # Processing the call keyword arguments (line 470)
    kwargs_579694 = {}
    # Getting the type of 'AttributeError' (line 470)
    AttributeError_579692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 14), 'AttributeError', False)
    # Calling AttributeError(args, kwargs) (line 470)
    AttributeError_call_result_579695 = invoke(stypy.reporting.localization.Localization(__file__, 470, 14), AttributeError_579692, *[str_579693], **kwargs_579694)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 470, 8), AttributeError_call_result_579695, 'raise parameter', BaseException)
    # SSA join for if statement (line 469)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 472):
    
    # Assigning a Call to a Name (line 472):
    
    # Call to count(...): (line 472)
    # Processing the call keyword arguments (line 472)
    kwargs_579698 = {}
    # Getting the type of 'data' (line 472)
    data_579696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'data', False)
    # Obtaining the member 'count' of a type (line 472)
    count_579697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), data_579696, 'count')
    # Calling count(args, kwargs) (line 472)
    count_call_result_579699 = invoke(stypy.reporting.localization.Localization(__file__, 472, 8), count_579697, *[], **kwargs_579698)
    
    # Assigning a type to the variable 'n' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'n', count_call_result_579699)
    
    # Assigning a Call to a Name (line 473):
    
    # Assigning a Call to a Name (line 473):
    
    # Call to idealfourths(...): (line 473)
    # Processing the call arguments (line 473)
    # Getting the type of 'data' (line 473)
    data_579701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 21), 'data', False)
    # Processing the call keyword arguments (line 473)
    # Getting the type of 'None' (line 473)
    None_579702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 32), 'None', False)
    keyword_579703 = None_579702
    kwargs_579704 = {'axis': keyword_579703}
    # Getting the type of 'idealfourths' (line 473)
    idealfourths_579700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'idealfourths', False)
    # Calling idealfourths(args, kwargs) (line 473)
    idealfourths_call_result_579705 = invoke(stypy.reporting.localization.Localization(__file__, 473, 8), idealfourths_579700, *[data_579701], **kwargs_579704)
    
    # Assigning a type to the variable 'r' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'r', idealfourths_call_result_579705)
    
    # Assigning a BinOp to a Name (line 474):
    
    # Assigning a BinOp to a Name (line 474):
    float_579706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 8), 'float')
    
    # Obtaining the type of the subscript
    int_579707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 17), 'int')
    # Getting the type of 'r' (line 474)
    r_579708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 15), 'r')
    # Obtaining the member '__getitem__' of a type (line 474)
    getitem___579709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 15), r_579708, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 474)
    subscript_call_result_579710 = invoke(stypy.reporting.localization.Localization(__file__, 474, 15), getitem___579709, int_579707)
    
    
    # Obtaining the type of the subscript
    int_579711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 23), 'int')
    # Getting the type of 'r' (line 474)
    r_579712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 21), 'r')
    # Obtaining the member '__getitem__' of a type (line 474)
    getitem___579713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 21), r_579712, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 474)
    subscript_call_result_579714 = invoke(stypy.reporting.localization.Localization(__file__, 474, 21), getitem___579713, int_579711)
    
    # Applying the binary operator '-' (line 474)
    result_sub_579715 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 15), '-', subscript_call_result_579710, subscript_call_result_579714)
    
    # Applying the binary operator '*' (line 474)
    result_mul_579716 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 8), '*', float_579706, result_sub_579715)
    
    # Getting the type of 'n' (line 474)
    n_579717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 29), 'n')
    float_579718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 33), 'float')
    int_579719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 36), 'int')
    # Applying the binary operator 'div' (line 474)
    result_div_579720 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 33), 'div', float_579718, int_579719)
    
    # Applying the binary operator '**' (line 474)
    result_pow_579721 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 29), '**', n_579717, result_div_579720)
    
    # Applying the binary operator 'div' (line 474)
    result_div_579722 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 27), 'div', result_mul_579716, result_pow_579721)
    
    # Assigning a type to the variable 'h' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'h', result_div_579722)
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to sum(...): (line 475)
    # Processing the call arguments (line 475)
    int_579737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 51), 'int')
    # Processing the call keyword arguments (line 475)
    kwargs_579738 = {}
    
    
    # Obtaining the type of the subscript
    slice_579723 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 475, 11), None, None, None)
    # Getting the type of 'None' (line 475)
    None_579724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 18), 'None', False)
    # Getting the type of 'data' (line 475)
    data_579725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 11), 'data', False)
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___579726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 11), data_579725, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_579727 = invoke(stypy.reporting.localization.Localization(__file__, 475, 11), getitem___579726, (slice_579723, None_579724))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'None' (line 475)
    None_579728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 34), 'None', False)
    slice_579729 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 475, 27), None, None, None)
    # Getting the type of 'points' (line 475)
    points_579730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 27), 'points', False)
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___579731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 27), points_579730, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_579732 = invoke(stypy.reporting.localization.Localization(__file__, 475, 27), getitem___579731, (None_579728, slice_579729))
    
    # Getting the type of 'h' (line 475)
    h_579733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 44), 'h', False)
    # Applying the binary operator '+' (line 475)
    result_add_579734 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 27), '+', subscript_call_result_579732, h_579733)
    
    # Applying the binary operator '<=' (line 475)
    result_le_579735 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 11), '<=', subscript_call_result_579727, result_add_579734)
    
    # Obtaining the member 'sum' of a type (line 475)
    sum_579736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 11), result_le_579735, 'sum')
    # Calling sum(args, kwargs) (line 475)
    sum_call_result_579739 = invoke(stypy.reporting.localization.Localization(__file__, 475, 11), sum_579736, *[int_579737], **kwargs_579738)
    
    # Assigning a type to the variable 'nhi' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'nhi', sum_call_result_579739)
    
    # Assigning a Call to a Name (line 476):
    
    # Assigning a Call to a Name (line 476):
    
    # Call to sum(...): (line 476)
    # Processing the call arguments (line 476)
    int_579754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 50), 'int')
    # Processing the call keyword arguments (line 476)
    kwargs_579755 = {}
    
    
    # Obtaining the type of the subscript
    slice_579740 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 476, 11), None, None, None)
    # Getting the type of 'None' (line 476)
    None_579741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 18), 'None', False)
    # Getting the type of 'data' (line 476)
    data_579742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 11), 'data', False)
    # Obtaining the member '__getitem__' of a type (line 476)
    getitem___579743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 11), data_579742, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
    subscript_call_result_579744 = invoke(stypy.reporting.localization.Localization(__file__, 476, 11), getitem___579743, (slice_579740, None_579741))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'None' (line 476)
    None_579745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 33), 'None', False)
    slice_579746 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 476, 26), None, None, None)
    # Getting the type of 'points' (line 476)
    points_579747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 26), 'points', False)
    # Obtaining the member '__getitem__' of a type (line 476)
    getitem___579748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 26), points_579747, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
    subscript_call_result_579749 = invoke(stypy.reporting.localization.Localization(__file__, 476, 26), getitem___579748, (None_579745, slice_579746))
    
    # Getting the type of 'h' (line 476)
    h_579750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 43), 'h', False)
    # Applying the binary operator '-' (line 476)
    result_sub_579751 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 26), '-', subscript_call_result_579749, h_579750)
    
    # Applying the binary operator '<' (line 476)
    result_lt_579752 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 11), '<', subscript_call_result_579744, result_sub_579751)
    
    # Obtaining the member 'sum' of a type (line 476)
    sum_579753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 11), result_lt_579752, 'sum')
    # Calling sum(args, kwargs) (line 476)
    sum_call_result_579756 = invoke(stypy.reporting.localization.Localization(__file__, 476, 11), sum_579753, *[int_579754], **kwargs_579755)
    
    # Assigning a type to the variable 'nlo' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'nlo', sum_call_result_579756)
    # Getting the type of 'nhi' (line 477)
    nhi_579757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'nhi')
    # Getting the type of 'nlo' (line 477)
    nlo_579758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 16), 'nlo')
    # Applying the binary operator '-' (line 477)
    result_sub_579759 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 12), '-', nhi_579757, nlo_579758)
    
    float_579760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 24), 'float')
    # Getting the type of 'n' (line 477)
    n_579761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 27), 'n')
    # Applying the binary operator '*' (line 477)
    result_mul_579762 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 24), '*', float_579760, n_579761)
    
    # Getting the type of 'h' (line 477)
    h_579763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 29), 'h')
    # Applying the binary operator '*' (line 477)
    result_mul_579764 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 28), '*', result_mul_579762, h_579763)
    
    # Applying the binary operator 'div' (line 477)
    result_div_579765 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 11), 'div', result_sub_579759, result_mul_579764)
    
    # Assigning a type to the variable 'stypy_return_type' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'stypy_return_type', result_div_579765)
    
    # ################# End of 'rsh(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rsh' in the type store
    # Getting the type of 'stypy_return_type' (line 447)
    stypy_return_type_579766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_579766)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rsh'
    return stypy_return_type_579766

# Assigning a type to the variable 'rsh' (line 447)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 0), 'rsh', rsh)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
