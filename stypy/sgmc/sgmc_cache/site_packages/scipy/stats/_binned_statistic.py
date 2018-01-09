
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from scipy._lib.six import callable, xrange
5: from scipy._lib._numpy_compat import suppress_warnings
6: from collections import namedtuple
7: 
8: __all__ = ['binned_statistic',
9:            'binned_statistic_2d',
10:            'binned_statistic_dd']
11: 
12: 
13: BinnedStatisticResult = namedtuple('BinnedStatisticResult',
14:                                    ('statistic', 'bin_edges', 'binnumber'))
15: 
16: 
17: def binned_statistic(x, values, statistic='mean',
18:                      bins=10, range=None):
19:     '''
20:     Compute a binned statistic for one or more sets of data.
21: 
22:     This is a generalization of a histogram function.  A histogram divides
23:     the space into bins, and returns the count of the number of points in
24:     each bin.  This function allows the computation of the sum, mean, median,
25:     or other statistic of the values (or set of values) within each bin.
26: 
27:     Parameters
28:     ----------
29:     x : (N,) array_like
30:         A sequence of values to be binned.
31:     values : (N,) array_like or list of (N,) array_like
32:         The data on which the statistic will be computed.  This must be
33:         the same shape as `x`, or a set of sequences - each the same shape as
34:         `x`.  If `values` is a set of sequences, the statistic will be computed
35:         on each independently.
36:     statistic : string or callable, optional
37:         The statistic to compute (default is 'mean').
38:         The following statistics are available:
39: 
40:           * 'mean' : compute the mean of values for points within each bin.
41:             Empty bins will be represented by NaN.
42:           * 'median' : compute the median of values for points within each
43:             bin. Empty bins will be represented by NaN.
44:           * 'count' : compute the count of points within each bin.  This is
45:             identical to an unweighted histogram.  `values` array is not
46:             referenced.
47:           * 'sum' : compute the sum of values for points within each bin.
48:             This is identical to a weighted histogram.
49:           * 'min' : compute the minimum of values for points within each bin.
50:             Empty bins will be represented by NaN.
51:           * 'max' : compute the maximum of values for point within each bin.
52:             Empty bins will be represented by NaN.
53:           * function : a user-defined function which takes a 1D array of
54:             values, and outputs a single numerical statistic. This function
55:             will be called on the values in each bin.  Empty bins will be
56:             represented by function([]), or NaN if this returns an error.
57: 
58:     bins : int or sequence of scalars, optional
59:         If `bins` is an int, it defines the number of equal-width bins in the
60:         given range (10 by default).  If `bins` is a sequence, it defines the
61:         bin edges, including the rightmost edge, allowing for non-uniform bin
62:         widths.  Values in `x` that are smaller than lowest bin edge are
63:         assigned to bin number 0, values beyond the highest bin are assigned to
64:         ``bins[-1]``.  If the bin edges are specified, the number of bins will
65:         be, (nx = len(bins)-1).
66:     range : (float, float) or [(float, float)], optional
67:         The lower and upper range of the bins.  If not provided, range
68:         is simply ``(x.min(), x.max())``.  Values outside the range are
69:         ignored.
70: 
71:     Returns
72:     -------
73:     statistic : array
74:         The values of the selected statistic in each bin.
75:     bin_edges : array of dtype float
76:         Return the bin edges ``(length(statistic)+1)``.
77:     binnumber: 1-D ndarray of ints
78:         Indices of the bins (corresponding to `bin_edges`) in which each value
79:         of `x` belongs.  Same length as `values`.  A binnumber of `i` means the
80:         corresponding value is between (bin_edges[i-1], bin_edges[i]).
81: 
82:     See Also
83:     --------
84:     numpy.digitize, numpy.histogram, binned_statistic_2d, binned_statistic_dd
85: 
86:     Notes
87:     -----
88:     All but the last (righthand-most) bin is half-open.  In other words, if
89:     `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including 1,
90:     but excluding 2) and the second ``[2, 3)``.  The last bin, however, is
91:     ``[3, 4]``, which *includes* 4.
92: 
93:     .. versionadded:: 0.11.0
94: 
95:     Examples
96:     --------
97:     >>> from scipy import stats
98:     >>> import matplotlib.pyplot as plt
99: 
100:     First some basic examples:
101: 
102:     Create two evenly spaced bins in the range of the given sample, and sum the
103:     corresponding values in each of those bins:
104: 
105:     >>> values = [1.0, 1.0, 2.0, 1.5, 3.0]
106:     >>> stats.binned_statistic([1, 1, 2, 5, 7], values, 'sum', bins=2)
107:     (array([ 4. ,  4.5]), array([ 1.,  4.,  7.]), array([1, 1, 1, 2, 2]))
108: 
109:     Multiple arrays of values can also be passed.  The statistic is calculated
110:     on each set independently:
111: 
112:     >>> values = [[1.0, 1.0, 2.0, 1.5, 3.0], [2.0, 2.0, 4.0, 3.0, 6.0]]
113:     >>> stats.binned_statistic([1, 1, 2, 5, 7], values, 'sum', bins=2)
114:     (array([[ 4. ,  4.5], [ 8. ,  9. ]]), array([ 1.,  4.,  7.]),
115:         array([1, 1, 1, 2, 2]))
116: 
117:     >>> stats.binned_statistic([1, 2, 1, 2, 4], np.arange(5), statistic='mean',
118:     ...                        bins=3)
119:     (array([ 1.,  2.,  4.]), array([ 1.,  2.,  3.,  4.]),
120:         array([1, 2, 1, 2, 3]))
121: 
122:     As a second example, we now generate some random data of sailing boat speed
123:     as a function of wind speed, and then determine how fast our boat is for
124:     certain wind speeds:
125: 
126:     >>> windspeed = 8 * np.random.rand(500)
127:     >>> boatspeed = .3 * windspeed**.5 + .2 * np.random.rand(500)
128:     >>> bin_means, bin_edges, binnumber = stats.binned_statistic(windspeed,
129:     ...                 boatspeed, statistic='median', bins=[1,2,3,4,5,6,7])
130:     >>> plt.figure()
131:     >>> plt.plot(windspeed, boatspeed, 'b.', label='raw data')
132:     >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
133:     ...            label='binned statistic of data')
134:     >>> plt.legend()
135: 
136:     Now we can use ``binnumber`` to select all datapoints with a windspeed
137:     below 1:
138: 
139:     >>> low_boatspeed = boatspeed[binnumber == 0]
140: 
141:     As a final example, we will use ``bin_edges`` and ``binnumber`` to make a
142:     plot of a distribution that shows the mean and distribution around that
143:     mean per bin, on top of a regular histogram and the probability
144:     distribution function:
145: 
146:     >>> x = np.linspace(0, 5, num=500)
147:     >>> x_pdf = stats.maxwell.pdf(x)
148:     >>> samples = stats.maxwell.rvs(size=10000)
149: 
150:     >>> bin_means, bin_edges, binnumber = stats.binned_statistic(x, x_pdf,
151:     ...         statistic='mean', bins=25)
152:     >>> bin_width = (bin_edges[1] - bin_edges[0])
153:     >>> bin_centers = bin_edges[1:] - bin_width/2
154: 
155:     >>> plt.figure()
156:     >>> plt.hist(samples, bins=50, normed=True, histtype='stepfilled',
157:     ...          alpha=0.2, label='histogram of data')
158:     >>> plt.plot(x, x_pdf, 'r-', label='analytical pdf')
159:     >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,
160:     ...            label='binned statistic of data')
161:     >>> plt.plot((binnumber - 0.5) * bin_width, x_pdf, 'g.', alpha=0.5)
162:     >>> plt.legend(fontsize=10)
163:     >>> plt.show()
164: 
165:     '''
166:     try:
167:         N = len(bins)
168:     except TypeError:
169:         N = 1
170: 
171:     if N != 1:
172:         bins = [np.asarray(bins, float)]
173: 
174:     if range is not None:
175:         if len(range) == 2:
176:             range = [range]
177: 
178:     medians, edges, binnumbers = binned_statistic_dd(
179:         [x], values, statistic, bins, range)
180: 
181:     return BinnedStatisticResult(medians, edges[0], binnumbers)
182: 
183: 
184: BinnedStatistic2dResult = namedtuple('BinnedStatistic2dResult',
185:                                      ('statistic', 'x_edge', 'y_edge',
186:                                       'binnumber'))
187: 
188: 
189: def binned_statistic_2d(x, y, values, statistic='mean',
190:                         bins=10, range=None, expand_binnumbers=False):
191:     '''
192:     Compute a bidimensional binned statistic for one or more sets of data.
193: 
194:     This is a generalization of a histogram2d function.  A histogram divides
195:     the space into bins, and returns the count of the number of points in
196:     each bin.  This function allows the computation of the sum, mean, median,
197:     or other statistic of the values (or set of values) within each bin.
198: 
199:     Parameters
200:     ----------
201:     x : (N,) array_like
202:         A sequence of values to be binned along the first dimension.
203:     y : (N,) array_like
204:         A sequence of values to be binned along the second dimension.
205:     values : (N,) array_like or list of (N,) array_like
206:         The data on which the statistic will be computed.  This must be
207:         the same shape as `x`, or a list of sequences - each with the same
208:         shape as `x`.  If `values` is such a list, the statistic will be
209:         computed on each independently.
210:     statistic : string or callable, optional
211:         The statistic to compute (default is 'mean').
212:         The following statistics are available:
213: 
214:           * 'mean' : compute the mean of values for points within each bin.
215:             Empty bins will be represented by NaN.
216:           * 'median' : compute the median of values for points within each
217:             bin. Empty bins will be represented by NaN.
218:           * 'count' : compute the count of points within each bin.  This is
219:             identical to an unweighted histogram.  `values` array is not
220:             referenced.
221:           * 'sum' : compute the sum of values for points within each bin.
222:             This is identical to a weighted histogram.
223:           * 'min' : compute the minimum of values for points within each bin.
224:             Empty bins will be represented by NaN.
225:           * 'max' : compute the maximum of values for point within each bin.
226:             Empty bins will be represented by NaN.
227:           * function : a user-defined function which takes a 1D array of
228:             values, and outputs a single numerical statistic. This function
229:             will be called on the values in each bin.  Empty bins will be
230:             represented by function([]), or NaN if this returns an error.
231: 
232:     bins : int or [int, int] or array_like or [array, array], optional
233:         The bin specification:
234: 
235:           * the number of bins for the two dimensions (nx = ny = bins),
236:           * the number of bins in each dimension (nx, ny = bins),
237:           * the bin edges for the two dimensions (x_edge = y_edge = bins),
238:           * the bin edges in each dimension (x_edge, y_edge = bins).
239: 
240:         If the bin edges are specified, the number of bins will be,
241:         (nx = len(x_edge)-1, ny = len(y_edge)-1).
242: 
243:     range : (2,2) array_like, optional
244:         The leftmost and rightmost edges of the bins along each dimension
245:         (if not specified explicitly in the `bins` parameters):
246:         [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
247:         considered outliers and not tallied in the histogram.
248:     expand_binnumbers : bool, optional
249:         'False' (default): the returned `binnumber` is a shape (N,) array of
250:         linearized bin indices.
251:         'True': the returned `binnumber` is 'unraveled' into a shape (2,N)
252:         ndarray, where each row gives the bin numbers in the corresponding
253:         dimension.
254:         See the `binnumber` returned value, and the `Examples` section.
255: 
256:         .. versionadded:: 0.17.0
257: 
258:     Returns
259:     -------
260:     statistic : (nx, ny) ndarray
261:         The values of the selected statistic in each two-dimensional bin.
262:     x_edge : (nx + 1) ndarray
263:         The bin edges along the first dimension.
264:     y_edge : (ny + 1) ndarray
265:         The bin edges along the second dimension.
266:     binnumber : (N,) array of ints or (2,N) ndarray of ints
267:         This assigns to each element of `sample` an integer that represents the
268:         bin in which this observation falls.  The representation depends on the
269:         `expand_binnumbers` argument.  See `Notes` for details.
270: 
271: 
272:     See Also
273:     --------
274:     numpy.digitize, numpy.histogram2d, binned_statistic, binned_statistic_dd
275: 
276:     Notes
277:     -----
278:     Binedges:
279:     All but the last (righthand-most) bin is half-open.  In other words, if
280:     `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including 1,
281:     but excluding 2) and the second ``[2, 3)``.  The last bin, however, is
282:     ``[3, 4]``, which *includes* 4.
283: 
284:     `binnumber`:
285:     This returned argument assigns to each element of `sample` an integer that
286:     represents the bin in which it belongs.  The representation depends on the
287:     `expand_binnumbers` argument. If 'False' (default): The returned
288:     `binnumber` is a shape (N,) array of linearized indices mapping each
289:     element of `sample` to its corresponding bin (using row-major ordering).
290:     If 'True': The returned `binnumber` is a shape (2,N) ndarray where
291:     each row indicates bin placements for each dimension respectively.  In each
292:     dimension, a binnumber of `i` means the corresponding value is between
293:     (D_edge[i-1], D_edge[i]), where 'D' is either 'x' or 'y'.
294: 
295:     .. versionadded:: 0.11.0
296: 
297:     Examples
298:     --------
299:     >>> from scipy import stats
300: 
301:     Calculate the counts with explicit bin-edges:
302: 
303:     >>> x = [0.1, 0.1, 0.1, 0.6]
304:     >>> y = [2.1, 2.6, 2.1, 2.1]
305:     >>> binx = [0.0, 0.5, 1.0]
306:     >>> biny = [2.0, 2.5, 3.0]
307:     >>> ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx,biny])
308:     >>> ret.statistic
309:     array([[ 2.,  1.],
310:            [ 1.,  0.]])
311: 
312:     The bin in which each sample is placed is given by the `binnumber`
313:     returned parameter.  By default, these are the linearized bin indices:
314: 
315:     >>> ret.binnumber
316:     array([5, 6, 5, 9])
317: 
318:     The bin indices can also be expanded into separate entries for each
319:     dimension using the `expand_binnumbers` parameter:
320: 
321:     >>> ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx,biny],
322:     ...                                 expand_binnumbers=True)
323:     >>> ret.binnumber
324:     array([[1, 1, 1, 2],
325:            [1, 2, 1, 1]])
326: 
327:     Which shows that the first three elements belong in the xbin 1, and the
328:     fourth into xbin 2; and so on for y.
329: 
330:     '''
331: 
332:     # This code is based on np.histogram2d
333:     try:
334:         N = len(bins)
335:     except TypeError:
336:         N = 1
337: 
338:     if N != 1 and N != 2:
339:         xedges = yedges = np.asarray(bins, float)
340:         bins = [xedges, yedges]
341: 
342:     medians, edges, binnumbers = binned_statistic_dd(
343:         [x, y], values, statistic, bins, range,
344:         expand_binnumbers=expand_binnumbers)
345: 
346:     return BinnedStatistic2dResult(medians, edges[0], edges[1], binnumbers)
347: 
348: 
349: BinnedStatisticddResult = namedtuple('BinnedStatisticddResult',
350:                                      ('statistic', 'bin_edges',
351:                                       'binnumber'))
352: 
353: 
354: def binned_statistic_dd(sample, values, statistic='mean',
355:                         bins=10, range=None, expand_binnumbers=False):
356:     '''
357:     Compute a multidimensional binned statistic for a set of data.
358: 
359:     This is a generalization of a histogramdd function.  A histogram divides
360:     the space into bins, and returns the count of the number of points in
361:     each bin.  This function allows the computation of the sum, mean, median,
362:     or other statistic of the values within each bin.
363: 
364:     Parameters
365:     ----------
366:     sample : array_like
367:         Data to histogram passed as a sequence of D arrays of length N, or
368:         as an (N,D) array.
369:     values : (N,) array_like or list of (N,) array_like
370:         The data on which the statistic will be computed.  This must be
371:         the same shape as `x`, or a list of sequences - each with the same
372:         shape as `x`.  If `values` is such a list, the statistic will be
373:         computed on each independently.
374:     statistic : string or callable, optional
375:         The statistic to compute (default is 'mean').
376:         The following statistics are available:
377: 
378:           * 'mean' : compute the mean of values for points within each bin.
379:             Empty bins will be represented by NaN.
380:           * 'median' : compute the median of values for points within each
381:             bin. Empty bins will be represented by NaN.
382:           * 'count' : compute the count of points within each bin.  This is
383:             identical to an unweighted histogram.  `values` array is not
384:             referenced.
385:           * 'sum' : compute the sum of values for points within each bin.
386:             This is identical to a weighted histogram.
387:           * 'min' : compute the minimum of values for points within each bin.
388:             Empty bins will be represented by NaN.
389:           * 'max' : compute the maximum of values for point within each bin.
390:             Empty bins will be represented by NaN.
391:           * function : a user-defined function which takes a 1D array of
392:             values, and outputs a single numerical statistic. This function
393:             will be called on the values in each bin.  Empty bins will be
394:             represented by function([]), or NaN if this returns an error.
395: 
396:     bins : sequence or int, optional
397:         The bin specification must be in one of the following forms:
398: 
399:           * A sequence of arrays describing the bin edges along each dimension.
400:           * The number of bins for each dimension (nx, ny, ... = bins).
401:           * The number of bins for all dimensions (nx = ny = ... = bins).
402: 
403:     range : sequence, optional
404:         A sequence of lower and upper bin edges to be used if the edges are
405:         not given explicitely in `bins`. Defaults to the minimum and maximum
406:         values along each dimension.
407:     expand_binnumbers : bool, optional
408:         'False' (default): the returned `binnumber` is a shape (N,) array of
409:         linearized bin indices.
410:         'True': the returned `binnumber` is 'unraveled' into a shape (D,N)
411:         ndarray, where each row gives the bin numbers in the corresponding
412:         dimension.
413:         See the `binnumber` returned value, and the `Examples` section of
414:         `binned_statistic_2d`.
415: 
416:         .. versionadded:: 0.17.0
417: 
418:     Returns
419:     -------
420:     statistic : ndarray, shape(nx1, nx2, nx3,...)
421:         The values of the selected statistic in each two-dimensional bin.
422:     bin_edges : list of ndarrays
423:         A list of D arrays describing the (nxi + 1) bin edges for each
424:         dimension.
425:     binnumber : (N,) array of ints or (D,N) ndarray of ints
426:         This assigns to each element of `sample` an integer that represents the
427:         bin in which this observation falls.  The representation depends on the
428:         `expand_binnumbers` argument.  See `Notes` for details.
429: 
430: 
431:     See Also
432:     --------
433:     numpy.digitize, numpy.histogramdd, binned_statistic, binned_statistic_2d
434: 
435:     Notes
436:     -----
437:     Binedges:
438:     All but the last (righthand-most) bin is half-open in each dimension.  In
439:     other words, if `bins` is ``[1, 2, 3, 4]``, then the first bin is
440:     ``[1, 2)`` (including 1, but excluding 2) and the second ``[2, 3)``.  The
441:     last bin, however, is ``[3, 4]``, which *includes* 4.
442: 
443:     `binnumber`:
444:     This returned argument assigns to each element of `sample` an integer that
445:     represents the bin in which it belongs.  The representation depends on the
446:     `expand_binnumbers` argument. If 'False' (default): The returned
447:     `binnumber` is a shape (N,) array of linearized indices mapping each
448:     element of `sample` to its corresponding bin (using row-major ordering).
449:     If 'True': The returned `binnumber` is a shape (D,N) ndarray where
450:     each row indicates bin placements for each dimension respectively.  In each
451:     dimension, a binnumber of `i` means the corresponding value is between
452:     (bin_edges[D][i-1], bin_edges[D][i]), for each dimension 'D'.
453: 
454:     .. versionadded:: 0.11.0
455: 
456:     '''
457:     known_stats = ['mean', 'median', 'count', 'sum', 'std','min','max']
458:     if not callable(statistic) and statistic not in known_stats:
459:         raise ValueError('invalid statistic %r' % (statistic,))
460: 
461:     # `Ndim` is the number of dimensions (e.g. `2` for `binned_statistic_2d`)
462:     # `Dlen` is the length of elements along each dimension.
463:     # This code is based on np.histogramdd
464:     try:
465:         # `sample` is an ND-array.
466:         Dlen, Ndim = sample.shape
467:     except (AttributeError, ValueError):
468:         # `sample` is a sequence of 1D arrays.
469:         sample = np.atleast_2d(sample).T
470:         Dlen, Ndim = sample.shape
471: 
472:     # Store initial shape of `values` to preserve it in the output
473:     values = np.asarray(values)
474:     input_shape = list(values.shape)
475:     # Make sure that `values` is 2D to iterate over rows
476:     values = np.atleast_2d(values)
477:     Vdim, Vlen = values.shape
478: 
479:     # Make sure `values` match `sample`
480:     if(statistic != 'count' and Vlen != Dlen):
481:         raise AttributeError('The number of `values` elements must match the '
482:                              'length of each `sample` dimension.')
483: 
484:     nbin = np.empty(Ndim, int)    # Number of bins in each dimension
485:     edges = Ndim * [None]         # Bin edges for each dim (will be 2D array)
486:     dedges = Ndim * [None]        # Spacing between edges (will be 2D array)
487: 
488:     try:
489:         M = len(bins)
490:         if M != Ndim:
491:             raise AttributeError('The dimension of bins must be equal '
492:                                  'to the dimension of the sample x.')
493:     except TypeError:
494:         bins = Ndim * [bins]
495: 
496:     # Select range for each dimension
497:     # Used only if number of bins is given.
498:     if range is None:
499:         smin = np.atleast_1d(np.array(sample.min(axis=0), float))
500:         smax = np.atleast_1d(np.array(sample.max(axis=0), float))
501:     else:
502:         smin = np.zeros(Ndim)
503:         smax = np.zeros(Ndim)
504:         for i in xrange(Ndim):
505:             smin[i], smax[i] = range[i]
506: 
507:     # Make sure the bins have a finite width.
508:     for i in xrange(len(smin)):
509:         if smin[i] == smax[i]:
510:             smin[i] = smin[i] - .5
511:             smax[i] = smax[i] + .5
512: 
513:     # Create edge arrays
514:     for i in xrange(Ndim):
515:         if np.isscalar(bins[i]):
516:             nbin[i] = bins[i] + 2  # +2 for outlier bins
517:             edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1)
518:         else:
519:             edges[i] = np.asarray(bins[i], float)
520:             nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
521:         dedges[i] = np.diff(edges[i])
522: 
523:     nbin = np.asarray(nbin)
524: 
525:     # Compute the bin number each sample falls into, in each dimension
526:     sampBin = [
527:         np.digitize(sample[:, i], edges[i])
528:         for i in xrange(Ndim)
529:     ]
530: 
531:     # Using `digitize`, values that fall on an edge are put in the right bin.
532:     # For the rightmost bin, we want values equal to the right
533:     # edge to be counted in the last bin, and not as an outlier.
534:     for i in xrange(Ndim):
535:         # Find the rounding precision
536:         decimal = int(-np.log10(dedges[i].min())) + 6
537:         # Find which points are on the rightmost edge.
538:         on_edge = np.where(np.around(sample[:, i], decimal) ==
539:                            np.around(edges[i][-1], decimal))[0]
540:         # Shift these points one bin to the left.
541:         sampBin[i][on_edge] -= 1
542: 
543:     # Compute the sample indices in the flattened statistic matrix.
544:     binnumbers = np.ravel_multi_index(sampBin, nbin)
545: 
546:     result = np.empty([Vdim, nbin.prod()], float)
547: 
548:     if statistic == 'mean':
549:         result.fill(np.nan)
550:         flatcount = np.bincount(binnumbers, None)
551:         a = flatcount.nonzero()
552:         for vv in xrange(Vdim):
553:             flatsum = np.bincount(binnumbers, values[vv])
554:             result[vv, a] = flatsum[a] / flatcount[a]
555:     elif statistic == 'std':
556:         result.fill(0)
557:         flatcount = np.bincount(binnumbers, None)
558:         a = flatcount.nonzero()
559:         for vv in xrange(Vdim):
560:             flatsum = np.bincount(binnumbers, values[vv])
561:             flatsum2 = np.bincount(binnumbers, values[vv] ** 2)
562:             result[vv, a] = np.sqrt(flatsum2[a] / flatcount[a] -
563:                                     (flatsum[a] / flatcount[a]) ** 2)
564:     elif statistic == 'count':
565:         result.fill(0)
566:         flatcount = np.bincount(binnumbers, None)
567:         a = np.arange(len(flatcount))
568:         result[:, a] = flatcount[np.newaxis, :]
569:     elif statistic == 'sum':
570:         result.fill(0)
571:         for vv in xrange(Vdim):
572:             flatsum = np.bincount(binnumbers, values[vv])
573:             a = np.arange(len(flatsum))
574:             result[vv, a] = flatsum
575:     elif statistic == 'median':
576:         result.fill(np.nan)
577:         for i in np.unique(binnumbers):
578:             for vv in xrange(Vdim):
579:                 result[vv, i] = np.median(values[vv, binnumbers == i])
580:     elif statistic == 'min':
581:         result.fill(np.nan)
582:         for i in np.unique(binnumbers):
583:             for vv in xrange(Vdim):
584:                 result[vv, i] = np.min(values[vv, binnumbers == i])
585:     elif statistic == 'max':
586:         result.fill(np.nan)
587:         for i in np.unique(binnumbers):
588:             for vv in xrange(Vdim):
589:                 result[vv, i] = np.max(values[vv, binnumbers == i])
590:     elif callable(statistic):
591:         with np.errstate(invalid='ignore'), suppress_warnings() as sup:
592:             sup.filter(RuntimeWarning)
593:             try:
594:                 null = statistic([])
595:             except:
596:                 null = np.nan
597:         result.fill(null)
598:         for i in np.unique(binnumbers):
599:             for vv in xrange(Vdim):
600:                 result[vv, i] = statistic(values[vv, binnumbers == i])
601: 
602:     # Shape into a proper matrix
603:     result = result.reshape(np.append(Vdim, nbin))
604: 
605:     # Remove outliers (indices 0 and -1 for each bin-dimension).
606:     core = [slice(None)] + Ndim * [slice(1, -1)]
607:     result = result[core]
608: 
609:     # Unravel binnumbers into an ndarray, each row the bins for each dimension
610:     if(expand_binnumbers and Ndim > 1):
611:         binnumbers = np.asarray(np.unravel_index(binnumbers, nbin))
612: 
613:     if np.any(result.shape[1:] != nbin - 2):
614:         raise RuntimeError('Internal Shape Error')
615: 
616:     # Reshape to have output (`reulst`) match input (`values`) shape
617:     result = result.reshape(input_shape[:-1] + list(nbin-2))
618: 
619:     return BinnedStatisticddResult(result, edges, binnumbers)
620: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_589252 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_589252) is not StypyTypeError):

    if (import_589252 != 'pyd_module'):
        __import__(import_589252)
        sys_modules_589253 = sys.modules[import_589252]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_589253.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_589252)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy._lib.six import callable, xrange' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_589254 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib.six')

if (type(import_589254) is not StypyTypeError):

    if (import_589254 != 'pyd_module'):
        __import__(import_589254)
        sys_modules_589255 = sys.modules[import_589254]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib.six', sys_modules_589255.module_type_store, module_type_store, ['callable', 'xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_589255, sys_modules_589255.module_type_store, module_type_store)
    else:
        from scipy._lib.six import callable, xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib.six', None, module_type_store, ['callable', 'xrange'], [callable, xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib.six', import_589254)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_589256 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat')

if (type(import_589256) is not StypyTypeError):

    if (import_589256 != 'pyd_module'):
        __import__(import_589256)
        sys_modules_589257 = sys.modules[import_589256]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat', sys_modules_589257.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_589257, sys_modules_589257.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat', import_589256)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from collections import namedtuple' statement (line 6)
try:
    from collections import namedtuple

except:
    namedtuple = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'collections', None, module_type_store, ['namedtuple'], [namedtuple])


# Assigning a List to a Name (line 8):

# Assigning a List to a Name (line 8):
__all__ = ['binned_statistic', 'binned_statistic_2d', 'binned_statistic_dd']
module_type_store.set_exportable_members(['binned_statistic', 'binned_statistic_2d', 'binned_statistic_dd'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_589258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_589259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'binned_statistic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_589258, str_589259)
# Adding element type (line 8)
str_589260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'binned_statistic_2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_589258, str_589260)
# Adding element type (line 8)
str_589261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'binned_statistic_dd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_589258, str_589261)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_589258)

# Assigning a Call to a Name (line 13):

# Assigning a Call to a Name (line 13):

# Call to namedtuple(...): (line 13)
# Processing the call arguments (line 13)
str_589263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 35), 'str', 'BinnedStatisticResult')

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_589264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
str_589265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 36), 'str', 'statistic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 36), tuple_589264, str_589265)
# Adding element type (line 14)
str_589266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 49), 'str', 'bin_edges')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 36), tuple_589264, str_589266)
# Adding element type (line 14)
str_589267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 62), 'str', 'binnumber')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 36), tuple_589264, str_589267)

# Processing the call keyword arguments (line 13)
kwargs_589268 = {}
# Getting the type of 'namedtuple' (line 13)
namedtuple_589262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 24), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 13)
namedtuple_call_result_589269 = invoke(stypy.reporting.localization.Localization(__file__, 13, 24), namedtuple_589262, *[str_589263, tuple_589264], **kwargs_589268)

# Assigning a type to the variable 'BinnedStatisticResult' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'BinnedStatisticResult', namedtuple_call_result_589269)

@norecursion
def binned_statistic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_589270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 42), 'str', 'mean')
    int_589271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'int')
    # Getting the type of 'None' (line 18)
    None_589272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 36), 'None')
    defaults = [str_589270, int_589271, None_589272]
    # Create a new context for function 'binned_statistic'
    module_type_store = module_type_store.open_function_context('binned_statistic', 17, 0, False)
    
    # Passed parameters checking function
    binned_statistic.stypy_localization = localization
    binned_statistic.stypy_type_of_self = None
    binned_statistic.stypy_type_store = module_type_store
    binned_statistic.stypy_function_name = 'binned_statistic'
    binned_statistic.stypy_param_names_list = ['x', 'values', 'statistic', 'bins', 'range']
    binned_statistic.stypy_varargs_param_name = None
    binned_statistic.stypy_kwargs_param_name = None
    binned_statistic.stypy_call_defaults = defaults
    binned_statistic.stypy_call_varargs = varargs
    binned_statistic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binned_statistic', ['x', 'values', 'statistic', 'bins', 'range'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binned_statistic', localization, ['x', 'values', 'statistic', 'bins', 'range'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binned_statistic(...)' code ##################

    str_589273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, (-1)), 'str', "\n    Compute a binned statistic for one or more sets of data.\n\n    This is a generalization of a histogram function.  A histogram divides\n    the space into bins, and returns the count of the number of points in\n    each bin.  This function allows the computation of the sum, mean, median,\n    or other statistic of the values (or set of values) within each bin.\n\n    Parameters\n    ----------\n    x : (N,) array_like\n        A sequence of values to be binned.\n    values : (N,) array_like or list of (N,) array_like\n        The data on which the statistic will be computed.  This must be\n        the same shape as `x`, or a set of sequences - each the same shape as\n        `x`.  If `values` is a set of sequences, the statistic will be computed\n        on each independently.\n    statistic : string or callable, optional\n        The statistic to compute (default is 'mean').\n        The following statistics are available:\n\n          * 'mean' : compute the mean of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'median' : compute the median of values for points within each\n            bin. Empty bins will be represented by NaN.\n          * 'count' : compute the count of points within each bin.  This is\n            identical to an unweighted histogram.  `values` array is not\n            referenced.\n          * 'sum' : compute the sum of values for points within each bin.\n            This is identical to a weighted histogram.\n          * 'min' : compute the minimum of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'max' : compute the maximum of values for point within each bin.\n            Empty bins will be represented by NaN.\n          * function : a user-defined function which takes a 1D array of\n            values, and outputs a single numerical statistic. This function\n            will be called on the values in each bin.  Empty bins will be\n            represented by function([]), or NaN if this returns an error.\n\n    bins : int or sequence of scalars, optional\n        If `bins` is an int, it defines the number of equal-width bins in the\n        given range (10 by default).  If `bins` is a sequence, it defines the\n        bin edges, including the rightmost edge, allowing for non-uniform bin\n        widths.  Values in `x` that are smaller than lowest bin edge are\n        assigned to bin number 0, values beyond the highest bin are assigned to\n        ``bins[-1]``.  If the bin edges are specified, the number of bins will\n        be, (nx = len(bins)-1).\n    range : (float, float) or [(float, float)], optional\n        The lower and upper range of the bins.  If not provided, range\n        is simply ``(x.min(), x.max())``.  Values outside the range are\n        ignored.\n\n    Returns\n    -------\n    statistic : array\n        The values of the selected statistic in each bin.\n    bin_edges : array of dtype float\n        Return the bin edges ``(length(statistic)+1)``.\n    binnumber: 1-D ndarray of ints\n        Indices of the bins (corresponding to `bin_edges`) in which each value\n        of `x` belongs.  Same length as `values`.  A binnumber of `i` means the\n        corresponding value is between (bin_edges[i-1], bin_edges[i]).\n\n    See Also\n    --------\n    numpy.digitize, numpy.histogram, binned_statistic_2d, binned_statistic_dd\n\n    Notes\n    -----\n    All but the last (righthand-most) bin is half-open.  In other words, if\n    `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including 1,\n    but excluding 2) and the second ``[2, 3)``.  The last bin, however, is\n    ``[3, 4]``, which *includes* 4.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    >>> from scipy import stats\n    >>> import matplotlib.pyplot as plt\n\n    First some basic examples:\n\n    Create two evenly spaced bins in the range of the given sample, and sum the\n    corresponding values in each of those bins:\n\n    >>> values = [1.0, 1.0, 2.0, 1.5, 3.0]\n    >>> stats.binned_statistic([1, 1, 2, 5, 7], values, 'sum', bins=2)\n    (array([ 4. ,  4.5]), array([ 1.,  4.,  7.]), array([1, 1, 1, 2, 2]))\n\n    Multiple arrays of values can also be passed.  The statistic is calculated\n    on each set independently:\n\n    >>> values = [[1.0, 1.0, 2.0, 1.5, 3.0], [2.0, 2.0, 4.0, 3.0, 6.0]]\n    >>> stats.binned_statistic([1, 1, 2, 5, 7], values, 'sum', bins=2)\n    (array([[ 4. ,  4.5], [ 8. ,  9. ]]), array([ 1.,  4.,  7.]),\n        array([1, 1, 1, 2, 2]))\n\n    >>> stats.binned_statistic([1, 2, 1, 2, 4], np.arange(5), statistic='mean',\n    ...                        bins=3)\n    (array([ 1.,  2.,  4.]), array([ 1.,  2.,  3.,  4.]),\n        array([1, 2, 1, 2, 3]))\n\n    As a second example, we now generate some random data of sailing boat speed\n    as a function of wind speed, and then determine how fast our boat is for\n    certain wind speeds:\n\n    >>> windspeed = 8 * np.random.rand(500)\n    >>> boatspeed = .3 * windspeed**.5 + .2 * np.random.rand(500)\n    >>> bin_means, bin_edges, binnumber = stats.binned_statistic(windspeed,\n    ...                 boatspeed, statistic='median', bins=[1,2,3,4,5,6,7])\n    >>> plt.figure()\n    >>> plt.plot(windspeed, boatspeed, 'b.', label='raw data')\n    >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,\n    ...            label='binned statistic of data')\n    >>> plt.legend()\n\n    Now we can use ``binnumber`` to select all datapoints with a windspeed\n    below 1:\n\n    >>> low_boatspeed = boatspeed[binnumber == 0]\n\n    As a final example, we will use ``bin_edges`` and ``binnumber`` to make a\n    plot of a distribution that shows the mean and distribution around that\n    mean per bin, on top of a regular histogram and the probability\n    distribution function:\n\n    >>> x = np.linspace(0, 5, num=500)\n    >>> x_pdf = stats.maxwell.pdf(x)\n    >>> samples = stats.maxwell.rvs(size=10000)\n\n    >>> bin_means, bin_edges, binnumber = stats.binned_statistic(x, x_pdf,\n    ...         statistic='mean', bins=25)\n    >>> bin_width = (bin_edges[1] - bin_edges[0])\n    >>> bin_centers = bin_edges[1:] - bin_width/2\n\n    >>> plt.figure()\n    >>> plt.hist(samples, bins=50, normed=True, histtype='stepfilled',\n    ...          alpha=0.2, label='histogram of data')\n    >>> plt.plot(x, x_pdf, 'r-', label='analytical pdf')\n    >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,\n    ...            label='binned statistic of data')\n    >>> plt.plot((binnumber - 0.5) * bin_width, x_pdf, 'g.', alpha=0.5)\n    >>> plt.legend(fontsize=10)\n    >>> plt.show()\n\n    ")
    
    
    # SSA begins for try-except statement (line 166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 167):
    
    # Assigning a Call to a Name (line 167):
    
    # Call to len(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'bins' (line 167)
    bins_589275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'bins', False)
    # Processing the call keyword arguments (line 167)
    kwargs_589276 = {}
    # Getting the type of 'len' (line 167)
    len_589274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'len', False)
    # Calling len(args, kwargs) (line 167)
    len_call_result_589277 = invoke(stypy.reporting.localization.Localization(__file__, 167, 12), len_589274, *[bins_589275], **kwargs_589276)
    
    # Assigning a type to the variable 'N' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'N', len_call_result_589277)
    # SSA branch for the except part of a try statement (line 166)
    # SSA branch for the except 'TypeError' branch of a try statement (line 166)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Num to a Name (line 169):
    
    # Assigning a Num to a Name (line 169):
    int_589278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
    # Assigning a type to the variable 'N' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'N', int_589278)
    # SSA join for try-except statement (line 166)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'N' (line 171)
    N_589279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 7), 'N')
    int_589280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 12), 'int')
    # Applying the binary operator '!=' (line 171)
    result_ne_589281 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 7), '!=', N_589279, int_589280)
    
    # Testing the type of an if condition (line 171)
    if_condition_589282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 4), result_ne_589281)
    # Assigning a type to the variable 'if_condition_589282' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'if_condition_589282', if_condition_589282)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 172):
    
    # Assigning a List to a Name (line 172):
    
    # Obtaining an instance of the builtin type 'list' (line 172)
    list_589283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 172)
    # Adding element type (line 172)
    
    # Call to asarray(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'bins' (line 172)
    bins_589286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'bins', False)
    # Getting the type of 'float' (line 172)
    float_589287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 33), 'float', False)
    # Processing the call keyword arguments (line 172)
    kwargs_589288 = {}
    # Getting the type of 'np' (line 172)
    np_589284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'np', False)
    # Obtaining the member 'asarray' of a type (line 172)
    asarray_589285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), np_589284, 'asarray')
    # Calling asarray(args, kwargs) (line 172)
    asarray_call_result_589289 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), asarray_589285, *[bins_589286, float_589287], **kwargs_589288)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 15), list_589283, asarray_call_result_589289)
    
    # Assigning a type to the variable 'bins' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'bins', list_589283)
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 174)
    # Getting the type of 'range' (line 174)
    range_589290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'range')
    # Getting the type of 'None' (line 174)
    None_589291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'None')
    
    (may_be_589292, more_types_in_union_589293) = may_not_be_none(range_589290, None_589291)

    if may_be_589292:

        if more_types_in_union_589293:
            # Runtime conditional SSA (line 174)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'range' (line 175)
        range_589295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'range', False)
        # Processing the call keyword arguments (line 175)
        kwargs_589296 = {}
        # Getting the type of 'len' (line 175)
        len_589294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'len', False)
        # Calling len(args, kwargs) (line 175)
        len_call_result_589297 = invoke(stypy.reporting.localization.Localization(__file__, 175, 11), len_589294, *[range_589295], **kwargs_589296)
        
        int_589298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 25), 'int')
        # Applying the binary operator '==' (line 175)
        result_eq_589299 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 11), '==', len_call_result_589297, int_589298)
        
        # Testing the type of an if condition (line 175)
        if_condition_589300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 8), result_eq_589299)
        # Assigning a type to the variable 'if_condition_589300' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'if_condition_589300', if_condition_589300)
        # SSA begins for if statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 176):
        
        # Assigning a List to a Name (line 176):
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_589301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        # Getting the type of 'range' (line 176)
        range_589302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), 'range')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 20), list_589301, range_589302)
        
        # Assigning a type to the variable 'range' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'range', list_589301)
        # SSA join for if statement (line 175)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_589293:
            # SSA join for if statement (line 174)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 178):
    
    # Assigning a Subscript to a Name (line 178):
    
    # Obtaining the type of the subscript
    int_589303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 4), 'int')
    
    # Call to binned_statistic_dd(...): (line 178)
    # Processing the call arguments (line 178)
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_589305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    # Adding element type (line 179)
    # Getting the type of 'x' (line 179)
    x_589306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 9), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 8), list_589305, x_589306)
    
    # Getting the type of 'values' (line 179)
    values_589307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'values', False)
    # Getting the type of 'statistic' (line 179)
    statistic_589308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'statistic', False)
    # Getting the type of 'bins' (line 179)
    bins_589309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'bins', False)
    # Getting the type of 'range' (line 179)
    range_589310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 38), 'range', False)
    # Processing the call keyword arguments (line 178)
    kwargs_589311 = {}
    # Getting the type of 'binned_statistic_dd' (line 178)
    binned_statistic_dd_589304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 33), 'binned_statistic_dd', False)
    # Calling binned_statistic_dd(args, kwargs) (line 178)
    binned_statistic_dd_call_result_589312 = invoke(stypy.reporting.localization.Localization(__file__, 178, 33), binned_statistic_dd_589304, *[list_589305, values_589307, statistic_589308, bins_589309, range_589310], **kwargs_589311)
    
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___589313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 4), binned_statistic_dd_call_result_589312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_589314 = invoke(stypy.reporting.localization.Localization(__file__, 178, 4), getitem___589313, int_589303)
    
    # Assigning a type to the variable 'tuple_var_assignment_589238' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_var_assignment_589238', subscript_call_result_589314)
    
    # Assigning a Subscript to a Name (line 178):
    
    # Obtaining the type of the subscript
    int_589315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 4), 'int')
    
    # Call to binned_statistic_dd(...): (line 178)
    # Processing the call arguments (line 178)
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_589317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    # Adding element type (line 179)
    # Getting the type of 'x' (line 179)
    x_589318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 9), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 8), list_589317, x_589318)
    
    # Getting the type of 'values' (line 179)
    values_589319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'values', False)
    # Getting the type of 'statistic' (line 179)
    statistic_589320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'statistic', False)
    # Getting the type of 'bins' (line 179)
    bins_589321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'bins', False)
    # Getting the type of 'range' (line 179)
    range_589322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 38), 'range', False)
    # Processing the call keyword arguments (line 178)
    kwargs_589323 = {}
    # Getting the type of 'binned_statistic_dd' (line 178)
    binned_statistic_dd_589316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 33), 'binned_statistic_dd', False)
    # Calling binned_statistic_dd(args, kwargs) (line 178)
    binned_statistic_dd_call_result_589324 = invoke(stypy.reporting.localization.Localization(__file__, 178, 33), binned_statistic_dd_589316, *[list_589317, values_589319, statistic_589320, bins_589321, range_589322], **kwargs_589323)
    
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___589325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 4), binned_statistic_dd_call_result_589324, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_589326 = invoke(stypy.reporting.localization.Localization(__file__, 178, 4), getitem___589325, int_589315)
    
    # Assigning a type to the variable 'tuple_var_assignment_589239' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_var_assignment_589239', subscript_call_result_589326)
    
    # Assigning a Subscript to a Name (line 178):
    
    # Obtaining the type of the subscript
    int_589327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 4), 'int')
    
    # Call to binned_statistic_dd(...): (line 178)
    # Processing the call arguments (line 178)
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_589329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    # Adding element type (line 179)
    # Getting the type of 'x' (line 179)
    x_589330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 9), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 8), list_589329, x_589330)
    
    # Getting the type of 'values' (line 179)
    values_589331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'values', False)
    # Getting the type of 'statistic' (line 179)
    statistic_589332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'statistic', False)
    # Getting the type of 'bins' (line 179)
    bins_589333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'bins', False)
    # Getting the type of 'range' (line 179)
    range_589334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 38), 'range', False)
    # Processing the call keyword arguments (line 178)
    kwargs_589335 = {}
    # Getting the type of 'binned_statistic_dd' (line 178)
    binned_statistic_dd_589328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 33), 'binned_statistic_dd', False)
    # Calling binned_statistic_dd(args, kwargs) (line 178)
    binned_statistic_dd_call_result_589336 = invoke(stypy.reporting.localization.Localization(__file__, 178, 33), binned_statistic_dd_589328, *[list_589329, values_589331, statistic_589332, bins_589333, range_589334], **kwargs_589335)
    
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___589337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 4), binned_statistic_dd_call_result_589336, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_589338 = invoke(stypy.reporting.localization.Localization(__file__, 178, 4), getitem___589337, int_589327)
    
    # Assigning a type to the variable 'tuple_var_assignment_589240' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_var_assignment_589240', subscript_call_result_589338)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_var_assignment_589238' (line 178)
    tuple_var_assignment_589238_589339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_var_assignment_589238')
    # Assigning a type to the variable 'medians' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'medians', tuple_var_assignment_589238_589339)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_var_assignment_589239' (line 178)
    tuple_var_assignment_589239_589340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_var_assignment_589239')
    # Assigning a type to the variable 'edges' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'edges', tuple_var_assignment_589239_589340)
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'tuple_var_assignment_589240' (line 178)
    tuple_var_assignment_589240_589341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'tuple_var_assignment_589240')
    # Assigning a type to the variable 'binnumbers' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'binnumbers', tuple_var_assignment_589240_589341)
    
    # Call to BinnedStatisticResult(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'medians' (line 181)
    medians_589343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 33), 'medians', False)
    
    # Obtaining the type of the subscript
    int_589344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 48), 'int')
    # Getting the type of 'edges' (line 181)
    edges_589345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 42), 'edges', False)
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___589346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 42), edges_589345, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_589347 = invoke(stypy.reporting.localization.Localization(__file__, 181, 42), getitem___589346, int_589344)
    
    # Getting the type of 'binnumbers' (line 181)
    binnumbers_589348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 52), 'binnumbers', False)
    # Processing the call keyword arguments (line 181)
    kwargs_589349 = {}
    # Getting the type of 'BinnedStatisticResult' (line 181)
    BinnedStatisticResult_589342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'BinnedStatisticResult', False)
    # Calling BinnedStatisticResult(args, kwargs) (line 181)
    BinnedStatisticResult_call_result_589350 = invoke(stypy.reporting.localization.Localization(__file__, 181, 11), BinnedStatisticResult_589342, *[medians_589343, subscript_call_result_589347, binnumbers_589348], **kwargs_589349)
    
    # Assigning a type to the variable 'stypy_return_type' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type', BinnedStatisticResult_call_result_589350)
    
    # ################# End of 'binned_statistic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binned_statistic' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_589351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_589351)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binned_statistic'
    return stypy_return_type_589351

# Assigning a type to the variable 'binned_statistic' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'binned_statistic', binned_statistic)

# Assigning a Call to a Name (line 184):

# Assigning a Call to a Name (line 184):

# Call to namedtuple(...): (line 184)
# Processing the call arguments (line 184)
str_589353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 37), 'str', 'BinnedStatistic2dResult')

# Obtaining an instance of the builtin type 'tuple' (line 185)
tuple_589354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 185)
# Adding element type (line 185)
str_589355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 38), 'str', 'statistic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 38), tuple_589354, str_589355)
# Adding element type (line 185)
str_589356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 51), 'str', 'x_edge')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 38), tuple_589354, str_589356)
# Adding element type (line 185)
str_589357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 61), 'str', 'y_edge')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 38), tuple_589354, str_589357)
# Adding element type (line 185)
str_589358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 38), 'str', 'binnumber')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 38), tuple_589354, str_589358)

# Processing the call keyword arguments (line 184)
kwargs_589359 = {}
# Getting the type of 'namedtuple' (line 184)
namedtuple_589352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 184)
namedtuple_call_result_589360 = invoke(stypy.reporting.localization.Localization(__file__, 184, 26), namedtuple_589352, *[str_589353, tuple_589354], **kwargs_589359)

# Assigning a type to the variable 'BinnedStatistic2dResult' (line 184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'BinnedStatistic2dResult', namedtuple_call_result_589360)

@norecursion
def binned_statistic_2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_589361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 48), 'str', 'mean')
    int_589362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 29), 'int')
    # Getting the type of 'None' (line 190)
    None_589363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 39), 'None')
    # Getting the type of 'False' (line 190)
    False_589364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 63), 'False')
    defaults = [str_589361, int_589362, None_589363, False_589364]
    # Create a new context for function 'binned_statistic_2d'
    module_type_store = module_type_store.open_function_context('binned_statistic_2d', 189, 0, False)
    
    # Passed parameters checking function
    binned_statistic_2d.stypy_localization = localization
    binned_statistic_2d.stypy_type_of_self = None
    binned_statistic_2d.stypy_type_store = module_type_store
    binned_statistic_2d.stypy_function_name = 'binned_statistic_2d'
    binned_statistic_2d.stypy_param_names_list = ['x', 'y', 'values', 'statistic', 'bins', 'range', 'expand_binnumbers']
    binned_statistic_2d.stypy_varargs_param_name = None
    binned_statistic_2d.stypy_kwargs_param_name = None
    binned_statistic_2d.stypy_call_defaults = defaults
    binned_statistic_2d.stypy_call_varargs = varargs
    binned_statistic_2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binned_statistic_2d', ['x', 'y', 'values', 'statistic', 'bins', 'range', 'expand_binnumbers'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binned_statistic_2d', localization, ['x', 'y', 'values', 'statistic', 'bins', 'range', 'expand_binnumbers'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binned_statistic_2d(...)' code ##################

    str_589365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, (-1)), 'str', "\n    Compute a bidimensional binned statistic for one or more sets of data.\n\n    This is a generalization of a histogram2d function.  A histogram divides\n    the space into bins, and returns the count of the number of points in\n    each bin.  This function allows the computation of the sum, mean, median,\n    or other statistic of the values (or set of values) within each bin.\n\n    Parameters\n    ----------\n    x : (N,) array_like\n        A sequence of values to be binned along the first dimension.\n    y : (N,) array_like\n        A sequence of values to be binned along the second dimension.\n    values : (N,) array_like or list of (N,) array_like\n        The data on which the statistic will be computed.  This must be\n        the same shape as `x`, or a list of sequences - each with the same\n        shape as `x`.  If `values` is such a list, the statistic will be\n        computed on each independently.\n    statistic : string or callable, optional\n        The statistic to compute (default is 'mean').\n        The following statistics are available:\n\n          * 'mean' : compute the mean of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'median' : compute the median of values for points within each\n            bin. Empty bins will be represented by NaN.\n          * 'count' : compute the count of points within each bin.  This is\n            identical to an unweighted histogram.  `values` array is not\n            referenced.\n          * 'sum' : compute the sum of values for points within each bin.\n            This is identical to a weighted histogram.\n          * 'min' : compute the minimum of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'max' : compute the maximum of values for point within each bin.\n            Empty bins will be represented by NaN.\n          * function : a user-defined function which takes a 1D array of\n            values, and outputs a single numerical statistic. This function\n            will be called on the values in each bin.  Empty bins will be\n            represented by function([]), or NaN if this returns an error.\n\n    bins : int or [int, int] or array_like or [array, array], optional\n        The bin specification:\n\n          * the number of bins for the two dimensions (nx = ny = bins),\n          * the number of bins in each dimension (nx, ny = bins),\n          * the bin edges for the two dimensions (x_edge = y_edge = bins),\n          * the bin edges in each dimension (x_edge, y_edge = bins).\n\n        If the bin edges are specified, the number of bins will be,\n        (nx = len(x_edge)-1, ny = len(y_edge)-1).\n\n    range : (2,2) array_like, optional\n        The leftmost and rightmost edges of the bins along each dimension\n        (if not specified explicitly in the `bins` parameters):\n        [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be\n        considered outliers and not tallied in the histogram.\n    expand_binnumbers : bool, optional\n        'False' (default): the returned `binnumber` is a shape (N,) array of\n        linearized bin indices.\n        'True': the returned `binnumber` is 'unraveled' into a shape (2,N)\n        ndarray, where each row gives the bin numbers in the corresponding\n        dimension.\n        See the `binnumber` returned value, and the `Examples` section.\n\n        .. versionadded:: 0.17.0\n\n    Returns\n    -------\n    statistic : (nx, ny) ndarray\n        The values of the selected statistic in each two-dimensional bin.\n    x_edge : (nx + 1) ndarray\n        The bin edges along the first dimension.\n    y_edge : (ny + 1) ndarray\n        The bin edges along the second dimension.\n    binnumber : (N,) array of ints or (2,N) ndarray of ints\n        This assigns to each element of `sample` an integer that represents the\n        bin in which this observation falls.  The representation depends on the\n        `expand_binnumbers` argument.  See `Notes` for details.\n\n\n    See Also\n    --------\n    numpy.digitize, numpy.histogram2d, binned_statistic, binned_statistic_dd\n\n    Notes\n    -----\n    Binedges:\n    All but the last (righthand-most) bin is half-open.  In other words, if\n    `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including 1,\n    but excluding 2) and the second ``[2, 3)``.  The last bin, however, is\n    ``[3, 4]``, which *includes* 4.\n\n    `binnumber`:\n    This returned argument assigns to each element of `sample` an integer that\n    represents the bin in which it belongs.  The representation depends on the\n    `expand_binnumbers` argument. If 'False' (default): The returned\n    `binnumber` is a shape (N,) array of linearized indices mapping each\n    element of `sample` to its corresponding bin (using row-major ordering).\n    If 'True': The returned `binnumber` is a shape (2,N) ndarray where\n    each row indicates bin placements for each dimension respectively.  In each\n    dimension, a binnumber of `i` means the corresponding value is between\n    (D_edge[i-1], D_edge[i]), where 'D' is either 'x' or 'y'.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    >>> from scipy import stats\n\n    Calculate the counts with explicit bin-edges:\n\n    >>> x = [0.1, 0.1, 0.1, 0.6]\n    >>> y = [2.1, 2.6, 2.1, 2.1]\n    >>> binx = [0.0, 0.5, 1.0]\n    >>> biny = [2.0, 2.5, 3.0]\n    >>> ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx,biny])\n    >>> ret.statistic\n    array([[ 2.,  1.],\n           [ 1.,  0.]])\n\n    The bin in which each sample is placed is given by the `binnumber`\n    returned parameter.  By default, these are the linearized bin indices:\n\n    >>> ret.binnumber\n    array([5, 6, 5, 9])\n\n    The bin indices can also be expanded into separate entries for each\n    dimension using the `expand_binnumbers` parameter:\n\n    >>> ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx,biny],\n    ...                                 expand_binnumbers=True)\n    >>> ret.binnumber\n    array([[1, 1, 1, 2],\n           [1, 2, 1, 1]])\n\n    Which shows that the first three elements belong in the xbin 1, and the\n    fourth into xbin 2; and so on for y.\n\n    ")
    
    
    # SSA begins for try-except statement (line 333)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 334):
    
    # Assigning a Call to a Name (line 334):
    
    # Call to len(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'bins' (line 334)
    bins_589367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'bins', False)
    # Processing the call keyword arguments (line 334)
    kwargs_589368 = {}
    # Getting the type of 'len' (line 334)
    len_589366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'len', False)
    # Calling len(args, kwargs) (line 334)
    len_call_result_589369 = invoke(stypy.reporting.localization.Localization(__file__, 334, 12), len_589366, *[bins_589367], **kwargs_589368)
    
    # Assigning a type to the variable 'N' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'N', len_call_result_589369)
    # SSA branch for the except part of a try statement (line 333)
    # SSA branch for the except 'TypeError' branch of a try statement (line 333)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Num to a Name (line 336):
    
    # Assigning a Num to a Name (line 336):
    int_589370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 12), 'int')
    # Assigning a type to the variable 'N' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'N', int_589370)
    # SSA join for try-except statement (line 333)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'N' (line 338)
    N_589371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 7), 'N')
    int_589372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 12), 'int')
    # Applying the binary operator '!=' (line 338)
    result_ne_589373 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 7), '!=', N_589371, int_589372)
    
    
    # Getting the type of 'N' (line 338)
    N_589374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'N')
    int_589375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 23), 'int')
    # Applying the binary operator '!=' (line 338)
    result_ne_589376 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 18), '!=', N_589374, int_589375)
    
    # Applying the binary operator 'and' (line 338)
    result_and_keyword_589377 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 7), 'and', result_ne_589373, result_ne_589376)
    
    # Testing the type of an if condition (line 338)
    if_condition_589378 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 4), result_and_keyword_589377)
    # Assigning a type to the variable 'if_condition_589378' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'if_condition_589378', if_condition_589378)
    # SSA begins for if statement (line 338)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Call to a Name (line 339):
    
    # Call to asarray(...): (line 339)
    # Processing the call arguments (line 339)
    # Getting the type of 'bins' (line 339)
    bins_589381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 37), 'bins', False)
    # Getting the type of 'float' (line 339)
    float_589382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 43), 'float', False)
    # Processing the call keyword arguments (line 339)
    kwargs_589383 = {}
    # Getting the type of 'np' (line 339)
    np_589379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 26), 'np', False)
    # Obtaining the member 'asarray' of a type (line 339)
    asarray_589380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 26), np_589379, 'asarray')
    # Calling asarray(args, kwargs) (line 339)
    asarray_call_result_589384 = invoke(stypy.reporting.localization.Localization(__file__, 339, 26), asarray_589380, *[bins_589381, float_589382], **kwargs_589383)
    
    # Assigning a type to the variable 'yedges' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 17), 'yedges', asarray_call_result_589384)
    
    # Assigning a Name to a Name (line 339):
    # Getting the type of 'yedges' (line 339)
    yedges_589385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 17), 'yedges')
    # Assigning a type to the variable 'xedges' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'xedges', yedges_589385)
    
    # Assigning a List to a Name (line 340):
    
    # Assigning a List to a Name (line 340):
    
    # Obtaining an instance of the builtin type 'list' (line 340)
    list_589386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 340)
    # Adding element type (line 340)
    # Getting the type of 'xedges' (line 340)
    xedges_589387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'xedges')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 15), list_589386, xedges_589387)
    # Adding element type (line 340)
    # Getting the type of 'yedges' (line 340)
    yedges_589388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 24), 'yedges')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 15), list_589386, yedges_589388)
    
    # Assigning a type to the variable 'bins' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'bins', list_589386)
    # SSA join for if statement (line 338)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 342):
    
    # Assigning a Subscript to a Name (line 342):
    
    # Obtaining the type of the subscript
    int_589389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 4), 'int')
    
    # Call to binned_statistic_dd(...): (line 342)
    # Processing the call arguments (line 342)
    
    # Obtaining an instance of the builtin type 'list' (line 343)
    list_589391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 343)
    # Adding element type (line 343)
    # Getting the type of 'x' (line 343)
    x_589392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 9), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 8), list_589391, x_589392)
    # Adding element type (line 343)
    # Getting the type of 'y' (line 343)
    y_589393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 8), list_589391, y_589393)
    
    # Getting the type of 'values' (line 343)
    values_589394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 16), 'values', False)
    # Getting the type of 'statistic' (line 343)
    statistic_589395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'statistic', False)
    # Getting the type of 'bins' (line 343)
    bins_589396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 35), 'bins', False)
    # Getting the type of 'range' (line 343)
    range_589397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 41), 'range', False)
    # Processing the call keyword arguments (line 342)
    # Getting the type of 'expand_binnumbers' (line 344)
    expand_binnumbers_589398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 26), 'expand_binnumbers', False)
    keyword_589399 = expand_binnumbers_589398
    kwargs_589400 = {'expand_binnumbers': keyword_589399}
    # Getting the type of 'binned_statistic_dd' (line 342)
    binned_statistic_dd_589390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 33), 'binned_statistic_dd', False)
    # Calling binned_statistic_dd(args, kwargs) (line 342)
    binned_statistic_dd_call_result_589401 = invoke(stypy.reporting.localization.Localization(__file__, 342, 33), binned_statistic_dd_589390, *[list_589391, values_589394, statistic_589395, bins_589396, range_589397], **kwargs_589400)
    
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___589402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 4), binned_statistic_dd_call_result_589401, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 342)
    subscript_call_result_589403 = invoke(stypy.reporting.localization.Localization(__file__, 342, 4), getitem___589402, int_589389)
    
    # Assigning a type to the variable 'tuple_var_assignment_589241' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'tuple_var_assignment_589241', subscript_call_result_589403)
    
    # Assigning a Subscript to a Name (line 342):
    
    # Obtaining the type of the subscript
    int_589404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 4), 'int')
    
    # Call to binned_statistic_dd(...): (line 342)
    # Processing the call arguments (line 342)
    
    # Obtaining an instance of the builtin type 'list' (line 343)
    list_589406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 343)
    # Adding element type (line 343)
    # Getting the type of 'x' (line 343)
    x_589407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 9), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 8), list_589406, x_589407)
    # Adding element type (line 343)
    # Getting the type of 'y' (line 343)
    y_589408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 8), list_589406, y_589408)
    
    # Getting the type of 'values' (line 343)
    values_589409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 16), 'values', False)
    # Getting the type of 'statistic' (line 343)
    statistic_589410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'statistic', False)
    # Getting the type of 'bins' (line 343)
    bins_589411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 35), 'bins', False)
    # Getting the type of 'range' (line 343)
    range_589412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 41), 'range', False)
    # Processing the call keyword arguments (line 342)
    # Getting the type of 'expand_binnumbers' (line 344)
    expand_binnumbers_589413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 26), 'expand_binnumbers', False)
    keyword_589414 = expand_binnumbers_589413
    kwargs_589415 = {'expand_binnumbers': keyword_589414}
    # Getting the type of 'binned_statistic_dd' (line 342)
    binned_statistic_dd_589405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 33), 'binned_statistic_dd', False)
    # Calling binned_statistic_dd(args, kwargs) (line 342)
    binned_statistic_dd_call_result_589416 = invoke(stypy.reporting.localization.Localization(__file__, 342, 33), binned_statistic_dd_589405, *[list_589406, values_589409, statistic_589410, bins_589411, range_589412], **kwargs_589415)
    
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___589417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 4), binned_statistic_dd_call_result_589416, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 342)
    subscript_call_result_589418 = invoke(stypy.reporting.localization.Localization(__file__, 342, 4), getitem___589417, int_589404)
    
    # Assigning a type to the variable 'tuple_var_assignment_589242' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'tuple_var_assignment_589242', subscript_call_result_589418)
    
    # Assigning a Subscript to a Name (line 342):
    
    # Obtaining the type of the subscript
    int_589419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 4), 'int')
    
    # Call to binned_statistic_dd(...): (line 342)
    # Processing the call arguments (line 342)
    
    # Obtaining an instance of the builtin type 'list' (line 343)
    list_589421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 343)
    # Adding element type (line 343)
    # Getting the type of 'x' (line 343)
    x_589422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 9), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 8), list_589421, x_589422)
    # Adding element type (line 343)
    # Getting the type of 'y' (line 343)
    y_589423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 8), list_589421, y_589423)
    
    # Getting the type of 'values' (line 343)
    values_589424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 16), 'values', False)
    # Getting the type of 'statistic' (line 343)
    statistic_589425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'statistic', False)
    # Getting the type of 'bins' (line 343)
    bins_589426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 35), 'bins', False)
    # Getting the type of 'range' (line 343)
    range_589427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 41), 'range', False)
    # Processing the call keyword arguments (line 342)
    # Getting the type of 'expand_binnumbers' (line 344)
    expand_binnumbers_589428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 26), 'expand_binnumbers', False)
    keyword_589429 = expand_binnumbers_589428
    kwargs_589430 = {'expand_binnumbers': keyword_589429}
    # Getting the type of 'binned_statistic_dd' (line 342)
    binned_statistic_dd_589420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 33), 'binned_statistic_dd', False)
    # Calling binned_statistic_dd(args, kwargs) (line 342)
    binned_statistic_dd_call_result_589431 = invoke(stypy.reporting.localization.Localization(__file__, 342, 33), binned_statistic_dd_589420, *[list_589421, values_589424, statistic_589425, bins_589426, range_589427], **kwargs_589430)
    
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___589432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 4), binned_statistic_dd_call_result_589431, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 342)
    subscript_call_result_589433 = invoke(stypy.reporting.localization.Localization(__file__, 342, 4), getitem___589432, int_589419)
    
    # Assigning a type to the variable 'tuple_var_assignment_589243' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'tuple_var_assignment_589243', subscript_call_result_589433)
    
    # Assigning a Name to a Name (line 342):
    # Getting the type of 'tuple_var_assignment_589241' (line 342)
    tuple_var_assignment_589241_589434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'tuple_var_assignment_589241')
    # Assigning a type to the variable 'medians' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'medians', tuple_var_assignment_589241_589434)
    
    # Assigning a Name to a Name (line 342):
    # Getting the type of 'tuple_var_assignment_589242' (line 342)
    tuple_var_assignment_589242_589435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'tuple_var_assignment_589242')
    # Assigning a type to the variable 'edges' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 13), 'edges', tuple_var_assignment_589242_589435)
    
    # Assigning a Name to a Name (line 342):
    # Getting the type of 'tuple_var_assignment_589243' (line 342)
    tuple_var_assignment_589243_589436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'tuple_var_assignment_589243')
    # Assigning a type to the variable 'binnumbers' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'binnumbers', tuple_var_assignment_589243_589436)
    
    # Call to BinnedStatistic2dResult(...): (line 346)
    # Processing the call arguments (line 346)
    # Getting the type of 'medians' (line 346)
    medians_589438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 35), 'medians', False)
    
    # Obtaining the type of the subscript
    int_589439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 50), 'int')
    # Getting the type of 'edges' (line 346)
    edges_589440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 44), 'edges', False)
    # Obtaining the member '__getitem__' of a type (line 346)
    getitem___589441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 44), edges_589440, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 346)
    subscript_call_result_589442 = invoke(stypy.reporting.localization.Localization(__file__, 346, 44), getitem___589441, int_589439)
    
    
    # Obtaining the type of the subscript
    int_589443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 60), 'int')
    # Getting the type of 'edges' (line 346)
    edges_589444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 54), 'edges', False)
    # Obtaining the member '__getitem__' of a type (line 346)
    getitem___589445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 54), edges_589444, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 346)
    subscript_call_result_589446 = invoke(stypy.reporting.localization.Localization(__file__, 346, 54), getitem___589445, int_589443)
    
    # Getting the type of 'binnumbers' (line 346)
    binnumbers_589447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 64), 'binnumbers', False)
    # Processing the call keyword arguments (line 346)
    kwargs_589448 = {}
    # Getting the type of 'BinnedStatistic2dResult' (line 346)
    BinnedStatistic2dResult_589437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 11), 'BinnedStatistic2dResult', False)
    # Calling BinnedStatistic2dResult(args, kwargs) (line 346)
    BinnedStatistic2dResult_call_result_589449 = invoke(stypy.reporting.localization.Localization(__file__, 346, 11), BinnedStatistic2dResult_589437, *[medians_589438, subscript_call_result_589442, subscript_call_result_589446, binnumbers_589447], **kwargs_589448)
    
    # Assigning a type to the variable 'stypy_return_type' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'stypy_return_type', BinnedStatistic2dResult_call_result_589449)
    
    # ################# End of 'binned_statistic_2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binned_statistic_2d' in the type store
    # Getting the type of 'stypy_return_type' (line 189)
    stypy_return_type_589450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_589450)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binned_statistic_2d'
    return stypy_return_type_589450

# Assigning a type to the variable 'binned_statistic_2d' (line 189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'binned_statistic_2d', binned_statistic_2d)

# Assigning a Call to a Name (line 349):

# Assigning a Call to a Name (line 349):

# Call to namedtuple(...): (line 349)
# Processing the call arguments (line 349)
str_589452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 37), 'str', 'BinnedStatisticddResult')

# Obtaining an instance of the builtin type 'tuple' (line 350)
tuple_589453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 350)
# Adding element type (line 350)
str_589454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 38), 'str', 'statistic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 38), tuple_589453, str_589454)
# Adding element type (line 350)
str_589455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 51), 'str', 'bin_edges')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 38), tuple_589453, str_589455)
# Adding element type (line 350)
str_589456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 38), 'str', 'binnumber')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 38), tuple_589453, str_589456)

# Processing the call keyword arguments (line 349)
kwargs_589457 = {}
# Getting the type of 'namedtuple' (line 349)
namedtuple_589451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 26), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 349)
namedtuple_call_result_589458 = invoke(stypy.reporting.localization.Localization(__file__, 349, 26), namedtuple_589451, *[str_589452, tuple_589453], **kwargs_589457)

# Assigning a type to the variable 'BinnedStatisticddResult' (line 349)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 0), 'BinnedStatisticddResult', namedtuple_call_result_589458)

@norecursion
def binned_statistic_dd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_589459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 50), 'str', 'mean')
    int_589460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 29), 'int')
    # Getting the type of 'None' (line 355)
    None_589461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 39), 'None')
    # Getting the type of 'False' (line 355)
    False_589462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 63), 'False')
    defaults = [str_589459, int_589460, None_589461, False_589462]
    # Create a new context for function 'binned_statistic_dd'
    module_type_store = module_type_store.open_function_context('binned_statistic_dd', 354, 0, False)
    
    # Passed parameters checking function
    binned_statistic_dd.stypy_localization = localization
    binned_statistic_dd.stypy_type_of_self = None
    binned_statistic_dd.stypy_type_store = module_type_store
    binned_statistic_dd.stypy_function_name = 'binned_statistic_dd'
    binned_statistic_dd.stypy_param_names_list = ['sample', 'values', 'statistic', 'bins', 'range', 'expand_binnumbers']
    binned_statistic_dd.stypy_varargs_param_name = None
    binned_statistic_dd.stypy_kwargs_param_name = None
    binned_statistic_dd.stypy_call_defaults = defaults
    binned_statistic_dd.stypy_call_varargs = varargs
    binned_statistic_dd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'binned_statistic_dd', ['sample', 'values', 'statistic', 'bins', 'range', 'expand_binnumbers'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'binned_statistic_dd', localization, ['sample', 'values', 'statistic', 'bins', 'range', 'expand_binnumbers'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'binned_statistic_dd(...)' code ##################

    str_589463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, (-1)), 'str', "\n    Compute a multidimensional binned statistic for a set of data.\n\n    This is a generalization of a histogramdd function.  A histogram divides\n    the space into bins, and returns the count of the number of points in\n    each bin.  This function allows the computation of the sum, mean, median,\n    or other statistic of the values within each bin.\n\n    Parameters\n    ----------\n    sample : array_like\n        Data to histogram passed as a sequence of D arrays of length N, or\n        as an (N,D) array.\n    values : (N,) array_like or list of (N,) array_like\n        The data on which the statistic will be computed.  This must be\n        the same shape as `x`, or a list of sequences - each with the same\n        shape as `x`.  If `values` is such a list, the statistic will be\n        computed on each independently.\n    statistic : string or callable, optional\n        The statistic to compute (default is 'mean').\n        The following statistics are available:\n\n          * 'mean' : compute the mean of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'median' : compute the median of values for points within each\n            bin. Empty bins will be represented by NaN.\n          * 'count' : compute the count of points within each bin.  This is\n            identical to an unweighted histogram.  `values` array is not\n            referenced.\n          * 'sum' : compute the sum of values for points within each bin.\n            This is identical to a weighted histogram.\n          * 'min' : compute the minimum of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'max' : compute the maximum of values for point within each bin.\n            Empty bins will be represented by NaN.\n          * function : a user-defined function which takes a 1D array of\n            values, and outputs a single numerical statistic. This function\n            will be called on the values in each bin.  Empty bins will be\n            represented by function([]), or NaN if this returns an error.\n\n    bins : sequence or int, optional\n        The bin specification must be in one of the following forms:\n\n          * A sequence of arrays describing the bin edges along each dimension.\n          * The number of bins for each dimension (nx, ny, ... = bins).\n          * The number of bins for all dimensions (nx = ny = ... = bins).\n\n    range : sequence, optional\n        A sequence of lower and upper bin edges to be used if the edges are\n        not given explicitely in `bins`. Defaults to the minimum and maximum\n        values along each dimension.\n    expand_binnumbers : bool, optional\n        'False' (default): the returned `binnumber` is a shape (N,) array of\n        linearized bin indices.\n        'True': the returned `binnumber` is 'unraveled' into a shape (D,N)\n        ndarray, where each row gives the bin numbers in the corresponding\n        dimension.\n        See the `binnumber` returned value, and the `Examples` section of\n        `binned_statistic_2d`.\n\n        .. versionadded:: 0.17.0\n\n    Returns\n    -------\n    statistic : ndarray, shape(nx1, nx2, nx3,...)\n        The values of the selected statistic in each two-dimensional bin.\n    bin_edges : list of ndarrays\n        A list of D arrays describing the (nxi + 1) bin edges for each\n        dimension.\n    binnumber : (N,) array of ints or (D,N) ndarray of ints\n        This assigns to each element of `sample` an integer that represents the\n        bin in which this observation falls.  The representation depends on the\n        `expand_binnumbers` argument.  See `Notes` for details.\n\n\n    See Also\n    --------\n    numpy.digitize, numpy.histogramdd, binned_statistic, binned_statistic_2d\n\n    Notes\n    -----\n    Binedges:\n    All but the last (righthand-most) bin is half-open in each dimension.  In\n    other words, if `bins` is ``[1, 2, 3, 4]``, then the first bin is\n    ``[1, 2)`` (including 1, but excluding 2) and the second ``[2, 3)``.  The\n    last bin, however, is ``[3, 4]``, which *includes* 4.\n\n    `binnumber`:\n    This returned argument assigns to each element of `sample` an integer that\n    represents the bin in which it belongs.  The representation depends on the\n    `expand_binnumbers` argument. If 'False' (default): The returned\n    `binnumber` is a shape (N,) array of linearized indices mapping each\n    element of `sample` to its corresponding bin (using row-major ordering).\n    If 'True': The returned `binnumber` is a shape (D,N) ndarray where\n    each row indicates bin placements for each dimension respectively.  In each\n    dimension, a binnumber of `i` means the corresponding value is between\n    (bin_edges[D][i-1], bin_edges[D][i]), for each dimension 'D'.\n\n    .. versionadded:: 0.11.0\n\n    ")
    
    # Assigning a List to a Name (line 457):
    
    # Assigning a List to a Name (line 457):
    
    # Obtaining an instance of the builtin type 'list' (line 457)
    list_589464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 457)
    # Adding element type (line 457)
    str_589465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 19), 'str', 'mean')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 18), list_589464, str_589465)
    # Adding element type (line 457)
    str_589466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 27), 'str', 'median')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 18), list_589464, str_589466)
    # Adding element type (line 457)
    str_589467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 37), 'str', 'count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 18), list_589464, str_589467)
    # Adding element type (line 457)
    str_589468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 46), 'str', 'sum')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 18), list_589464, str_589468)
    # Adding element type (line 457)
    str_589469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 53), 'str', 'std')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 18), list_589464, str_589469)
    # Adding element type (line 457)
    str_589470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 59), 'str', 'min')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 18), list_589464, str_589470)
    # Adding element type (line 457)
    str_589471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 65), 'str', 'max')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 18), list_589464, str_589471)
    
    # Assigning a type to the variable 'known_stats' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'known_stats', list_589464)
    
    
    # Evaluating a boolean operation
    
    
    # Call to callable(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'statistic' (line 458)
    statistic_589473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 20), 'statistic', False)
    # Processing the call keyword arguments (line 458)
    kwargs_589474 = {}
    # Getting the type of 'callable' (line 458)
    callable_589472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 11), 'callable', False)
    # Calling callable(args, kwargs) (line 458)
    callable_call_result_589475 = invoke(stypy.reporting.localization.Localization(__file__, 458, 11), callable_589472, *[statistic_589473], **kwargs_589474)
    
    # Applying the 'not' unary operator (line 458)
    result_not__589476 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 7), 'not', callable_call_result_589475)
    
    
    # Getting the type of 'statistic' (line 458)
    statistic_589477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 35), 'statistic')
    # Getting the type of 'known_stats' (line 458)
    known_stats_589478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 52), 'known_stats')
    # Applying the binary operator 'notin' (line 458)
    result_contains_589479 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 35), 'notin', statistic_589477, known_stats_589478)
    
    # Applying the binary operator 'and' (line 458)
    result_and_keyword_589480 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 7), 'and', result_not__589476, result_contains_589479)
    
    # Testing the type of an if condition (line 458)
    if_condition_589481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 4), result_and_keyword_589480)
    # Assigning a type to the variable 'if_condition_589481' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'if_condition_589481', if_condition_589481)
    # SSA begins for if statement (line 458)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 459)
    # Processing the call arguments (line 459)
    str_589483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 25), 'str', 'invalid statistic %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 459)
    tuple_589484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 459)
    # Adding element type (line 459)
    # Getting the type of 'statistic' (line 459)
    statistic_589485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 51), 'statistic', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 51), tuple_589484, statistic_589485)
    
    # Applying the binary operator '%' (line 459)
    result_mod_589486 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 25), '%', str_589483, tuple_589484)
    
    # Processing the call keyword arguments (line 459)
    kwargs_589487 = {}
    # Getting the type of 'ValueError' (line 459)
    ValueError_589482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 459)
    ValueError_call_result_589488 = invoke(stypy.reporting.localization.Localization(__file__, 459, 14), ValueError_589482, *[result_mod_589486], **kwargs_589487)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 459, 8), ValueError_call_result_589488, 'raise parameter', BaseException)
    # SSA join for if statement (line 458)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 464)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Tuple (line 466):
    
    # Assigning a Subscript to a Name (line 466):
    
    # Obtaining the type of the subscript
    int_589489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 8), 'int')
    # Getting the type of 'sample' (line 466)
    sample_589490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 21), 'sample')
    # Obtaining the member 'shape' of a type (line 466)
    shape_589491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 21), sample_589490, 'shape')
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___589492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 8), shape_589491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_589493 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), getitem___589492, int_589489)
    
    # Assigning a type to the variable 'tuple_var_assignment_589244' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'tuple_var_assignment_589244', subscript_call_result_589493)
    
    # Assigning a Subscript to a Name (line 466):
    
    # Obtaining the type of the subscript
    int_589494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 8), 'int')
    # Getting the type of 'sample' (line 466)
    sample_589495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 21), 'sample')
    # Obtaining the member 'shape' of a type (line 466)
    shape_589496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 21), sample_589495, 'shape')
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___589497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 8), shape_589496, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_589498 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), getitem___589497, int_589494)
    
    # Assigning a type to the variable 'tuple_var_assignment_589245' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'tuple_var_assignment_589245', subscript_call_result_589498)
    
    # Assigning a Name to a Name (line 466):
    # Getting the type of 'tuple_var_assignment_589244' (line 466)
    tuple_var_assignment_589244_589499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'tuple_var_assignment_589244')
    # Assigning a type to the variable 'Dlen' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'Dlen', tuple_var_assignment_589244_589499)
    
    # Assigning a Name to a Name (line 466):
    # Getting the type of 'tuple_var_assignment_589245' (line 466)
    tuple_var_assignment_589245_589500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'tuple_var_assignment_589245')
    # Assigning a type to the variable 'Ndim' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 14), 'Ndim', tuple_var_assignment_589245_589500)
    # SSA branch for the except part of a try statement (line 464)
    # SSA branch for the except 'Tuple' branch of a try statement (line 464)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Attribute to a Name (line 469):
    
    # Assigning a Attribute to a Name (line 469):
    
    # Call to atleast_2d(...): (line 469)
    # Processing the call arguments (line 469)
    # Getting the type of 'sample' (line 469)
    sample_589503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 31), 'sample', False)
    # Processing the call keyword arguments (line 469)
    kwargs_589504 = {}
    # Getting the type of 'np' (line 469)
    np_589501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 17), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 469)
    atleast_2d_589502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 17), np_589501, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 469)
    atleast_2d_call_result_589505 = invoke(stypy.reporting.localization.Localization(__file__, 469, 17), atleast_2d_589502, *[sample_589503], **kwargs_589504)
    
    # Obtaining the member 'T' of a type (line 469)
    T_589506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 17), atleast_2d_call_result_589505, 'T')
    # Assigning a type to the variable 'sample' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'sample', T_589506)
    
    # Assigning a Attribute to a Tuple (line 470):
    
    # Assigning a Subscript to a Name (line 470):
    
    # Obtaining the type of the subscript
    int_589507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 8), 'int')
    # Getting the type of 'sample' (line 470)
    sample_589508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 21), 'sample')
    # Obtaining the member 'shape' of a type (line 470)
    shape_589509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 21), sample_589508, 'shape')
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___589510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 8), shape_589509, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_589511 = invoke(stypy.reporting.localization.Localization(__file__, 470, 8), getitem___589510, int_589507)
    
    # Assigning a type to the variable 'tuple_var_assignment_589246' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'tuple_var_assignment_589246', subscript_call_result_589511)
    
    # Assigning a Subscript to a Name (line 470):
    
    # Obtaining the type of the subscript
    int_589512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 8), 'int')
    # Getting the type of 'sample' (line 470)
    sample_589513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 21), 'sample')
    # Obtaining the member 'shape' of a type (line 470)
    shape_589514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 21), sample_589513, 'shape')
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___589515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 8), shape_589514, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_589516 = invoke(stypy.reporting.localization.Localization(__file__, 470, 8), getitem___589515, int_589512)
    
    # Assigning a type to the variable 'tuple_var_assignment_589247' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'tuple_var_assignment_589247', subscript_call_result_589516)
    
    # Assigning a Name to a Name (line 470):
    # Getting the type of 'tuple_var_assignment_589246' (line 470)
    tuple_var_assignment_589246_589517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'tuple_var_assignment_589246')
    # Assigning a type to the variable 'Dlen' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'Dlen', tuple_var_assignment_589246_589517)
    
    # Assigning a Name to a Name (line 470):
    # Getting the type of 'tuple_var_assignment_589247' (line 470)
    tuple_var_assignment_589247_589518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'tuple_var_assignment_589247')
    # Assigning a type to the variable 'Ndim' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 14), 'Ndim', tuple_var_assignment_589247_589518)
    # SSA join for try-except statement (line 464)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 473):
    
    # Assigning a Call to a Name (line 473):
    
    # Call to asarray(...): (line 473)
    # Processing the call arguments (line 473)
    # Getting the type of 'values' (line 473)
    values_589521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 24), 'values', False)
    # Processing the call keyword arguments (line 473)
    kwargs_589522 = {}
    # Getting the type of 'np' (line 473)
    np_589519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 473)
    asarray_589520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 13), np_589519, 'asarray')
    # Calling asarray(args, kwargs) (line 473)
    asarray_call_result_589523 = invoke(stypy.reporting.localization.Localization(__file__, 473, 13), asarray_589520, *[values_589521], **kwargs_589522)
    
    # Assigning a type to the variable 'values' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'values', asarray_call_result_589523)
    
    # Assigning a Call to a Name (line 474):
    
    # Assigning a Call to a Name (line 474):
    
    # Call to list(...): (line 474)
    # Processing the call arguments (line 474)
    # Getting the type of 'values' (line 474)
    values_589525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), 'values', False)
    # Obtaining the member 'shape' of a type (line 474)
    shape_589526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 23), values_589525, 'shape')
    # Processing the call keyword arguments (line 474)
    kwargs_589527 = {}
    # Getting the type of 'list' (line 474)
    list_589524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 18), 'list', False)
    # Calling list(args, kwargs) (line 474)
    list_call_result_589528 = invoke(stypy.reporting.localization.Localization(__file__, 474, 18), list_589524, *[shape_589526], **kwargs_589527)
    
    # Assigning a type to the variable 'input_shape' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'input_shape', list_call_result_589528)
    
    # Assigning a Call to a Name (line 476):
    
    # Assigning a Call to a Name (line 476):
    
    # Call to atleast_2d(...): (line 476)
    # Processing the call arguments (line 476)
    # Getting the type of 'values' (line 476)
    values_589531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 27), 'values', False)
    # Processing the call keyword arguments (line 476)
    kwargs_589532 = {}
    # Getting the type of 'np' (line 476)
    np_589529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 13), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 476)
    atleast_2d_589530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 13), np_589529, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 476)
    atleast_2d_call_result_589533 = invoke(stypy.reporting.localization.Localization(__file__, 476, 13), atleast_2d_589530, *[values_589531], **kwargs_589532)
    
    # Assigning a type to the variable 'values' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'values', atleast_2d_call_result_589533)
    
    # Assigning a Attribute to a Tuple (line 477):
    
    # Assigning a Subscript to a Name (line 477):
    
    # Obtaining the type of the subscript
    int_589534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 4), 'int')
    # Getting the type of 'values' (line 477)
    values_589535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 17), 'values')
    # Obtaining the member 'shape' of a type (line 477)
    shape_589536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 17), values_589535, 'shape')
    # Obtaining the member '__getitem__' of a type (line 477)
    getitem___589537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 4), shape_589536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 477)
    subscript_call_result_589538 = invoke(stypy.reporting.localization.Localization(__file__, 477, 4), getitem___589537, int_589534)
    
    # Assigning a type to the variable 'tuple_var_assignment_589248' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'tuple_var_assignment_589248', subscript_call_result_589538)
    
    # Assigning a Subscript to a Name (line 477):
    
    # Obtaining the type of the subscript
    int_589539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 4), 'int')
    # Getting the type of 'values' (line 477)
    values_589540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 17), 'values')
    # Obtaining the member 'shape' of a type (line 477)
    shape_589541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 17), values_589540, 'shape')
    # Obtaining the member '__getitem__' of a type (line 477)
    getitem___589542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 4), shape_589541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 477)
    subscript_call_result_589543 = invoke(stypy.reporting.localization.Localization(__file__, 477, 4), getitem___589542, int_589539)
    
    # Assigning a type to the variable 'tuple_var_assignment_589249' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'tuple_var_assignment_589249', subscript_call_result_589543)
    
    # Assigning a Name to a Name (line 477):
    # Getting the type of 'tuple_var_assignment_589248' (line 477)
    tuple_var_assignment_589248_589544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'tuple_var_assignment_589248')
    # Assigning a type to the variable 'Vdim' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'Vdim', tuple_var_assignment_589248_589544)
    
    # Assigning a Name to a Name (line 477):
    # Getting the type of 'tuple_var_assignment_589249' (line 477)
    tuple_var_assignment_589249_589545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'tuple_var_assignment_589249')
    # Assigning a type to the variable 'Vlen' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 10), 'Vlen', tuple_var_assignment_589249_589545)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'statistic' (line 480)
    statistic_589546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 7), 'statistic')
    str_589547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 20), 'str', 'count')
    # Applying the binary operator '!=' (line 480)
    result_ne_589548 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 7), '!=', statistic_589546, str_589547)
    
    
    # Getting the type of 'Vlen' (line 480)
    Vlen_589549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 32), 'Vlen')
    # Getting the type of 'Dlen' (line 480)
    Dlen_589550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 40), 'Dlen')
    # Applying the binary operator '!=' (line 480)
    result_ne_589551 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 32), '!=', Vlen_589549, Dlen_589550)
    
    # Applying the binary operator 'and' (line 480)
    result_and_keyword_589552 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 7), 'and', result_ne_589548, result_ne_589551)
    
    # Testing the type of an if condition (line 480)
    if_condition_589553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 480, 4), result_and_keyword_589552)
    # Assigning a type to the variable 'if_condition_589553' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'if_condition_589553', if_condition_589553)
    # SSA begins for if statement (line 480)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AttributeError(...): (line 481)
    # Processing the call arguments (line 481)
    str_589555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 29), 'str', 'The number of `values` elements must match the length of each `sample` dimension.')
    # Processing the call keyword arguments (line 481)
    kwargs_589556 = {}
    # Getting the type of 'AttributeError' (line 481)
    AttributeError_589554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 14), 'AttributeError', False)
    # Calling AttributeError(args, kwargs) (line 481)
    AttributeError_call_result_589557 = invoke(stypy.reporting.localization.Localization(__file__, 481, 14), AttributeError_589554, *[str_589555], **kwargs_589556)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 481, 8), AttributeError_call_result_589557, 'raise parameter', BaseException)
    # SSA join for if statement (line 480)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 484):
    
    # Assigning a Call to a Name (line 484):
    
    # Call to empty(...): (line 484)
    # Processing the call arguments (line 484)
    # Getting the type of 'Ndim' (line 484)
    Ndim_589560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 20), 'Ndim', False)
    # Getting the type of 'int' (line 484)
    int_589561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 26), 'int', False)
    # Processing the call keyword arguments (line 484)
    kwargs_589562 = {}
    # Getting the type of 'np' (line 484)
    np_589558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 11), 'np', False)
    # Obtaining the member 'empty' of a type (line 484)
    empty_589559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 11), np_589558, 'empty')
    # Calling empty(args, kwargs) (line 484)
    empty_call_result_589563 = invoke(stypy.reporting.localization.Localization(__file__, 484, 11), empty_589559, *[Ndim_589560, int_589561], **kwargs_589562)
    
    # Assigning a type to the variable 'nbin' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'nbin', empty_call_result_589563)
    
    # Assigning a BinOp to a Name (line 485):
    
    # Assigning a BinOp to a Name (line 485):
    # Getting the type of 'Ndim' (line 485)
    Ndim_589564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'Ndim')
    
    # Obtaining an instance of the builtin type 'list' (line 485)
    list_589565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 485)
    # Adding element type (line 485)
    # Getting the type of 'None' (line 485)
    None_589566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 20), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 19), list_589565, None_589566)
    
    # Applying the binary operator '*' (line 485)
    result_mul_589567 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 12), '*', Ndim_589564, list_589565)
    
    # Assigning a type to the variable 'edges' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'edges', result_mul_589567)
    
    # Assigning a BinOp to a Name (line 486):
    
    # Assigning a BinOp to a Name (line 486):
    # Getting the type of 'Ndim' (line 486)
    Ndim_589568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 13), 'Ndim')
    
    # Obtaining an instance of the builtin type 'list' (line 486)
    list_589569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 486)
    # Adding element type (line 486)
    # Getting the type of 'None' (line 486)
    None_589570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 21), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 20), list_589569, None_589570)
    
    # Applying the binary operator '*' (line 486)
    result_mul_589571 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 13), '*', Ndim_589568, list_589569)
    
    # Assigning a type to the variable 'dedges' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'dedges', result_mul_589571)
    
    
    # SSA begins for try-except statement (line 488)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 489):
    
    # Assigning a Call to a Name (line 489):
    
    # Call to len(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'bins' (line 489)
    bins_589573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 16), 'bins', False)
    # Processing the call keyword arguments (line 489)
    kwargs_589574 = {}
    # Getting the type of 'len' (line 489)
    len_589572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'len', False)
    # Calling len(args, kwargs) (line 489)
    len_call_result_589575 = invoke(stypy.reporting.localization.Localization(__file__, 489, 12), len_589572, *[bins_589573], **kwargs_589574)
    
    # Assigning a type to the variable 'M' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'M', len_call_result_589575)
    
    
    # Getting the type of 'M' (line 490)
    M_589576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 11), 'M')
    # Getting the type of 'Ndim' (line 490)
    Ndim_589577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 16), 'Ndim')
    # Applying the binary operator '!=' (line 490)
    result_ne_589578 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 11), '!=', M_589576, Ndim_589577)
    
    # Testing the type of an if condition (line 490)
    if_condition_589579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 8), result_ne_589578)
    # Assigning a type to the variable 'if_condition_589579' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'if_condition_589579', if_condition_589579)
    # SSA begins for if statement (line 490)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AttributeError(...): (line 491)
    # Processing the call arguments (line 491)
    str_589581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 33), 'str', 'The dimension of bins must be equal to the dimension of the sample x.')
    # Processing the call keyword arguments (line 491)
    kwargs_589582 = {}
    # Getting the type of 'AttributeError' (line 491)
    AttributeError_589580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 18), 'AttributeError', False)
    # Calling AttributeError(args, kwargs) (line 491)
    AttributeError_call_result_589583 = invoke(stypy.reporting.localization.Localization(__file__, 491, 18), AttributeError_589580, *[str_589581], **kwargs_589582)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 491, 12), AttributeError_call_result_589583, 'raise parameter', BaseException)
    # SSA join for if statement (line 490)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 488)
    # SSA branch for the except 'TypeError' branch of a try statement (line 488)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a BinOp to a Name (line 494):
    
    # Assigning a BinOp to a Name (line 494):
    # Getting the type of 'Ndim' (line 494)
    Ndim_589584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), 'Ndim')
    
    # Obtaining an instance of the builtin type 'list' (line 494)
    list_589585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 494)
    # Adding element type (line 494)
    # Getting the type of 'bins' (line 494)
    bins_589586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 23), 'bins')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 22), list_589585, bins_589586)
    
    # Applying the binary operator '*' (line 494)
    result_mul_589587 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 15), '*', Ndim_589584, list_589585)
    
    # Assigning a type to the variable 'bins' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'bins', result_mul_589587)
    # SSA join for try-except statement (line 488)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 498)
    # Getting the type of 'range' (line 498)
    range_589588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 7), 'range')
    # Getting the type of 'None' (line 498)
    None_589589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'None')
    
    (may_be_589590, more_types_in_union_589591) = may_be_none(range_589588, None_589589)

    if may_be_589590:

        if more_types_in_union_589591:
            # Runtime conditional SSA (line 498)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 499):
        
        # Assigning a Call to a Name (line 499):
        
        # Call to atleast_1d(...): (line 499)
        # Processing the call arguments (line 499)
        
        # Call to array(...): (line 499)
        # Processing the call arguments (line 499)
        
        # Call to min(...): (line 499)
        # Processing the call keyword arguments (line 499)
        int_589598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 54), 'int')
        keyword_589599 = int_589598
        kwargs_589600 = {'axis': keyword_589599}
        # Getting the type of 'sample' (line 499)
        sample_589596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 38), 'sample', False)
        # Obtaining the member 'min' of a type (line 499)
        min_589597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 38), sample_589596, 'min')
        # Calling min(args, kwargs) (line 499)
        min_call_result_589601 = invoke(stypy.reporting.localization.Localization(__file__, 499, 38), min_589597, *[], **kwargs_589600)
        
        # Getting the type of 'float' (line 499)
        float_589602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 58), 'float', False)
        # Processing the call keyword arguments (line 499)
        kwargs_589603 = {}
        # Getting the type of 'np' (line 499)
        np_589594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 29), 'np', False)
        # Obtaining the member 'array' of a type (line 499)
        array_589595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 29), np_589594, 'array')
        # Calling array(args, kwargs) (line 499)
        array_call_result_589604 = invoke(stypy.reporting.localization.Localization(__file__, 499, 29), array_589595, *[min_call_result_589601, float_589602], **kwargs_589603)
        
        # Processing the call keyword arguments (line 499)
        kwargs_589605 = {}
        # Getting the type of 'np' (line 499)
        np_589592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 15), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 499)
        atleast_1d_589593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 15), np_589592, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 499)
        atleast_1d_call_result_589606 = invoke(stypy.reporting.localization.Localization(__file__, 499, 15), atleast_1d_589593, *[array_call_result_589604], **kwargs_589605)
        
        # Assigning a type to the variable 'smin' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'smin', atleast_1d_call_result_589606)
        
        # Assigning a Call to a Name (line 500):
        
        # Assigning a Call to a Name (line 500):
        
        # Call to atleast_1d(...): (line 500)
        # Processing the call arguments (line 500)
        
        # Call to array(...): (line 500)
        # Processing the call arguments (line 500)
        
        # Call to max(...): (line 500)
        # Processing the call keyword arguments (line 500)
        int_589613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 54), 'int')
        keyword_589614 = int_589613
        kwargs_589615 = {'axis': keyword_589614}
        # Getting the type of 'sample' (line 500)
        sample_589611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 38), 'sample', False)
        # Obtaining the member 'max' of a type (line 500)
        max_589612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 38), sample_589611, 'max')
        # Calling max(args, kwargs) (line 500)
        max_call_result_589616 = invoke(stypy.reporting.localization.Localization(__file__, 500, 38), max_589612, *[], **kwargs_589615)
        
        # Getting the type of 'float' (line 500)
        float_589617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 58), 'float', False)
        # Processing the call keyword arguments (line 500)
        kwargs_589618 = {}
        # Getting the type of 'np' (line 500)
        np_589609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 29), 'np', False)
        # Obtaining the member 'array' of a type (line 500)
        array_589610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 29), np_589609, 'array')
        # Calling array(args, kwargs) (line 500)
        array_call_result_589619 = invoke(stypy.reporting.localization.Localization(__file__, 500, 29), array_589610, *[max_call_result_589616, float_589617], **kwargs_589618)
        
        # Processing the call keyword arguments (line 500)
        kwargs_589620 = {}
        # Getting the type of 'np' (line 500)
        np_589607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 500)
        atleast_1d_589608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 15), np_589607, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 500)
        atleast_1d_call_result_589621 = invoke(stypy.reporting.localization.Localization(__file__, 500, 15), atleast_1d_589608, *[array_call_result_589619], **kwargs_589620)
        
        # Assigning a type to the variable 'smax' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'smax', atleast_1d_call_result_589621)

        if more_types_in_union_589591:
            # Runtime conditional SSA for else branch (line 498)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_589590) or more_types_in_union_589591):
        
        # Assigning a Call to a Name (line 502):
        
        # Assigning a Call to a Name (line 502):
        
        # Call to zeros(...): (line 502)
        # Processing the call arguments (line 502)
        # Getting the type of 'Ndim' (line 502)
        Ndim_589624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 24), 'Ndim', False)
        # Processing the call keyword arguments (line 502)
        kwargs_589625 = {}
        # Getting the type of 'np' (line 502)
        np_589622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 15), 'np', False)
        # Obtaining the member 'zeros' of a type (line 502)
        zeros_589623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 15), np_589622, 'zeros')
        # Calling zeros(args, kwargs) (line 502)
        zeros_call_result_589626 = invoke(stypy.reporting.localization.Localization(__file__, 502, 15), zeros_589623, *[Ndim_589624], **kwargs_589625)
        
        # Assigning a type to the variable 'smin' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'smin', zeros_call_result_589626)
        
        # Assigning a Call to a Name (line 503):
        
        # Assigning a Call to a Name (line 503):
        
        # Call to zeros(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'Ndim' (line 503)
        Ndim_589629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 24), 'Ndim', False)
        # Processing the call keyword arguments (line 503)
        kwargs_589630 = {}
        # Getting the type of 'np' (line 503)
        np_589627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 15), 'np', False)
        # Obtaining the member 'zeros' of a type (line 503)
        zeros_589628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 15), np_589627, 'zeros')
        # Calling zeros(args, kwargs) (line 503)
        zeros_call_result_589631 = invoke(stypy.reporting.localization.Localization(__file__, 503, 15), zeros_589628, *[Ndim_589629], **kwargs_589630)
        
        # Assigning a type to the variable 'smax' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'smax', zeros_call_result_589631)
        
        
        # Call to xrange(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'Ndim' (line 504)
        Ndim_589633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 24), 'Ndim', False)
        # Processing the call keyword arguments (line 504)
        kwargs_589634 = {}
        # Getting the type of 'xrange' (line 504)
        xrange_589632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 504)
        xrange_call_result_589635 = invoke(stypy.reporting.localization.Localization(__file__, 504, 17), xrange_589632, *[Ndim_589633], **kwargs_589634)
        
        # Testing the type of a for loop iterable (line 504)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 504, 8), xrange_call_result_589635)
        # Getting the type of the for loop variable (line 504)
        for_loop_var_589636 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 504, 8), xrange_call_result_589635)
        # Assigning a type to the variable 'i' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'i', for_loop_var_589636)
        # SSA begins for a for statement (line 504)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Tuple (line 505):
        
        # Assigning a Subscript to a Name (line 505):
        
        # Obtaining the type of the subscript
        int_589637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 505)
        i_589638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 37), 'i')
        # Getting the type of 'range' (line 505)
        range_589639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 31), 'range')
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___589640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 31), range_589639, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_589641 = invoke(stypy.reporting.localization.Localization(__file__, 505, 31), getitem___589640, i_589638)
        
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___589642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 12), subscript_call_result_589641, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_589643 = invoke(stypy.reporting.localization.Localization(__file__, 505, 12), getitem___589642, int_589637)
        
        # Assigning a type to the variable 'tuple_var_assignment_589250' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'tuple_var_assignment_589250', subscript_call_result_589643)
        
        # Assigning a Subscript to a Name (line 505):
        
        # Obtaining the type of the subscript
        int_589644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 505)
        i_589645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 37), 'i')
        # Getting the type of 'range' (line 505)
        range_589646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 31), 'range')
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___589647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 31), range_589646, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_589648 = invoke(stypy.reporting.localization.Localization(__file__, 505, 31), getitem___589647, i_589645)
        
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___589649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 12), subscript_call_result_589648, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_589650 = invoke(stypy.reporting.localization.Localization(__file__, 505, 12), getitem___589649, int_589644)
        
        # Assigning a type to the variable 'tuple_var_assignment_589251' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'tuple_var_assignment_589251', subscript_call_result_589650)
        
        # Assigning a Name to a Subscript (line 505):
        # Getting the type of 'tuple_var_assignment_589250' (line 505)
        tuple_var_assignment_589250_589651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'tuple_var_assignment_589250')
        # Getting the type of 'smin' (line 505)
        smin_589652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'smin')
        # Getting the type of 'i' (line 505)
        i_589653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 17), 'i')
        # Storing an element on a container (line 505)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 12), smin_589652, (i_589653, tuple_var_assignment_589250_589651))
        
        # Assigning a Name to a Subscript (line 505):
        # Getting the type of 'tuple_var_assignment_589251' (line 505)
        tuple_var_assignment_589251_589654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'tuple_var_assignment_589251')
        # Getting the type of 'smax' (line 505)
        smax_589655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 21), 'smax')
        # Getting the type of 'i' (line 505)
        i_589656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 26), 'i')
        # Storing an element on a container (line 505)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 21), smax_589655, (i_589656, tuple_var_assignment_589251_589654))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_589590 and more_types_in_union_589591):
            # SSA join for if statement (line 498)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to xrange(...): (line 508)
    # Processing the call arguments (line 508)
    
    # Call to len(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of 'smin' (line 508)
    smin_589659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 24), 'smin', False)
    # Processing the call keyword arguments (line 508)
    kwargs_589660 = {}
    # Getting the type of 'len' (line 508)
    len_589658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 20), 'len', False)
    # Calling len(args, kwargs) (line 508)
    len_call_result_589661 = invoke(stypy.reporting.localization.Localization(__file__, 508, 20), len_589658, *[smin_589659], **kwargs_589660)
    
    # Processing the call keyword arguments (line 508)
    kwargs_589662 = {}
    # Getting the type of 'xrange' (line 508)
    xrange_589657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 508)
    xrange_call_result_589663 = invoke(stypy.reporting.localization.Localization(__file__, 508, 13), xrange_589657, *[len_call_result_589661], **kwargs_589662)
    
    # Testing the type of a for loop iterable (line 508)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 508, 4), xrange_call_result_589663)
    # Getting the type of the for loop variable (line 508)
    for_loop_var_589664 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 508, 4), xrange_call_result_589663)
    # Assigning a type to the variable 'i' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'i', for_loop_var_589664)
    # SSA begins for a for statement (line 508)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 509)
    i_589665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 'i')
    # Getting the type of 'smin' (line 509)
    smin_589666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 11), 'smin')
    # Obtaining the member '__getitem__' of a type (line 509)
    getitem___589667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 11), smin_589666, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 509)
    subscript_call_result_589668 = invoke(stypy.reporting.localization.Localization(__file__, 509, 11), getitem___589667, i_589665)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 509)
    i_589669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 27), 'i')
    # Getting the type of 'smax' (line 509)
    smax_589670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 22), 'smax')
    # Obtaining the member '__getitem__' of a type (line 509)
    getitem___589671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 22), smax_589670, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 509)
    subscript_call_result_589672 = invoke(stypy.reporting.localization.Localization(__file__, 509, 22), getitem___589671, i_589669)
    
    # Applying the binary operator '==' (line 509)
    result_eq_589673 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 11), '==', subscript_call_result_589668, subscript_call_result_589672)
    
    # Testing the type of an if condition (line 509)
    if_condition_589674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 509, 8), result_eq_589673)
    # Assigning a type to the variable 'if_condition_589674' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'if_condition_589674', if_condition_589674)
    # SSA begins for if statement (line 509)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 510):
    
    # Assigning a BinOp to a Subscript (line 510):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 510)
    i_589675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 27), 'i')
    # Getting the type of 'smin' (line 510)
    smin_589676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 22), 'smin')
    # Obtaining the member '__getitem__' of a type (line 510)
    getitem___589677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 22), smin_589676, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 510)
    subscript_call_result_589678 = invoke(stypy.reporting.localization.Localization(__file__, 510, 22), getitem___589677, i_589675)
    
    float_589679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 32), 'float')
    # Applying the binary operator '-' (line 510)
    result_sub_589680 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 22), '-', subscript_call_result_589678, float_589679)
    
    # Getting the type of 'smin' (line 510)
    smin_589681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'smin')
    # Getting the type of 'i' (line 510)
    i_589682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 17), 'i')
    # Storing an element on a container (line 510)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 12), smin_589681, (i_589682, result_sub_589680))
    
    # Assigning a BinOp to a Subscript (line 511):
    
    # Assigning a BinOp to a Subscript (line 511):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 511)
    i_589683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 27), 'i')
    # Getting the type of 'smax' (line 511)
    smax_589684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 22), 'smax')
    # Obtaining the member '__getitem__' of a type (line 511)
    getitem___589685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 22), smax_589684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 511)
    subscript_call_result_589686 = invoke(stypy.reporting.localization.Localization(__file__, 511, 22), getitem___589685, i_589683)
    
    float_589687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 32), 'float')
    # Applying the binary operator '+' (line 511)
    result_add_589688 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 22), '+', subscript_call_result_589686, float_589687)
    
    # Getting the type of 'smax' (line 511)
    smax_589689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'smax')
    # Getting the type of 'i' (line 511)
    i_589690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 17), 'i')
    # Storing an element on a container (line 511)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 12), smax_589689, (i_589690, result_add_589688))
    # SSA join for if statement (line 509)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to xrange(...): (line 514)
    # Processing the call arguments (line 514)
    # Getting the type of 'Ndim' (line 514)
    Ndim_589692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 20), 'Ndim', False)
    # Processing the call keyword arguments (line 514)
    kwargs_589693 = {}
    # Getting the type of 'xrange' (line 514)
    xrange_589691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 514)
    xrange_call_result_589694 = invoke(stypy.reporting.localization.Localization(__file__, 514, 13), xrange_589691, *[Ndim_589692], **kwargs_589693)
    
    # Testing the type of a for loop iterable (line 514)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 514, 4), xrange_call_result_589694)
    # Getting the type of the for loop variable (line 514)
    for_loop_var_589695 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 514, 4), xrange_call_result_589694)
    # Assigning a type to the variable 'i' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'i', for_loop_var_589695)
    # SSA begins for a for statement (line 514)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isscalar(...): (line 515)
    # Processing the call arguments (line 515)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 515)
    i_589698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 28), 'i', False)
    # Getting the type of 'bins' (line 515)
    bins_589699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 23), 'bins', False)
    # Obtaining the member '__getitem__' of a type (line 515)
    getitem___589700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 23), bins_589699, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 515)
    subscript_call_result_589701 = invoke(stypy.reporting.localization.Localization(__file__, 515, 23), getitem___589700, i_589698)
    
    # Processing the call keyword arguments (line 515)
    kwargs_589702 = {}
    # Getting the type of 'np' (line 515)
    np_589696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 11), 'np', False)
    # Obtaining the member 'isscalar' of a type (line 515)
    isscalar_589697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 11), np_589696, 'isscalar')
    # Calling isscalar(args, kwargs) (line 515)
    isscalar_call_result_589703 = invoke(stypy.reporting.localization.Localization(__file__, 515, 11), isscalar_589697, *[subscript_call_result_589701], **kwargs_589702)
    
    # Testing the type of an if condition (line 515)
    if_condition_589704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 515, 8), isscalar_call_result_589703)
    # Assigning a type to the variable 'if_condition_589704' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'if_condition_589704', if_condition_589704)
    # SSA begins for if statement (line 515)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 516):
    
    # Assigning a BinOp to a Subscript (line 516):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 516)
    i_589705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 27), 'i')
    # Getting the type of 'bins' (line 516)
    bins_589706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 22), 'bins')
    # Obtaining the member '__getitem__' of a type (line 516)
    getitem___589707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 22), bins_589706, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 516)
    subscript_call_result_589708 = invoke(stypy.reporting.localization.Localization(__file__, 516, 22), getitem___589707, i_589705)
    
    int_589709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 32), 'int')
    # Applying the binary operator '+' (line 516)
    result_add_589710 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 22), '+', subscript_call_result_589708, int_589709)
    
    # Getting the type of 'nbin' (line 516)
    nbin_589711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'nbin')
    # Getting the type of 'i' (line 516)
    i_589712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 17), 'i')
    # Storing an element on a container (line 516)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 12), nbin_589711, (i_589712, result_add_589710))
    
    # Assigning a Call to a Subscript (line 517):
    
    # Assigning a Call to a Subscript (line 517):
    
    # Call to linspace(...): (line 517)
    # Processing the call arguments (line 517)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 517)
    i_589715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 40), 'i', False)
    # Getting the type of 'smin' (line 517)
    smin_589716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 35), 'smin', False)
    # Obtaining the member '__getitem__' of a type (line 517)
    getitem___589717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 35), smin_589716, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 517)
    subscript_call_result_589718 = invoke(stypy.reporting.localization.Localization(__file__, 517, 35), getitem___589717, i_589715)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 517)
    i_589719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 49), 'i', False)
    # Getting the type of 'smax' (line 517)
    smax_589720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 44), 'smax', False)
    # Obtaining the member '__getitem__' of a type (line 517)
    getitem___589721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 44), smax_589720, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 517)
    subscript_call_result_589722 = invoke(stypy.reporting.localization.Localization(__file__, 517, 44), getitem___589721, i_589719)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 517)
    i_589723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 58), 'i', False)
    # Getting the type of 'nbin' (line 517)
    nbin_589724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 53), 'nbin', False)
    # Obtaining the member '__getitem__' of a type (line 517)
    getitem___589725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 53), nbin_589724, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 517)
    subscript_call_result_589726 = invoke(stypy.reporting.localization.Localization(__file__, 517, 53), getitem___589725, i_589723)
    
    int_589727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 63), 'int')
    # Applying the binary operator '-' (line 517)
    result_sub_589728 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 53), '-', subscript_call_result_589726, int_589727)
    
    # Processing the call keyword arguments (line 517)
    kwargs_589729 = {}
    # Getting the type of 'np' (line 517)
    np_589713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 23), 'np', False)
    # Obtaining the member 'linspace' of a type (line 517)
    linspace_589714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 23), np_589713, 'linspace')
    # Calling linspace(args, kwargs) (line 517)
    linspace_call_result_589730 = invoke(stypy.reporting.localization.Localization(__file__, 517, 23), linspace_589714, *[subscript_call_result_589718, subscript_call_result_589722, result_sub_589728], **kwargs_589729)
    
    # Getting the type of 'edges' (line 517)
    edges_589731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'edges')
    # Getting the type of 'i' (line 517)
    i_589732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 18), 'i')
    # Storing an element on a container (line 517)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 12), edges_589731, (i_589732, linspace_call_result_589730))
    # SSA branch for the else part of an if statement (line 515)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Subscript (line 519):
    
    # Assigning a Call to a Subscript (line 519):
    
    # Call to asarray(...): (line 519)
    # Processing the call arguments (line 519)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 519)
    i_589735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 39), 'i', False)
    # Getting the type of 'bins' (line 519)
    bins_589736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 34), 'bins', False)
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___589737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 34), bins_589736, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_589738 = invoke(stypy.reporting.localization.Localization(__file__, 519, 34), getitem___589737, i_589735)
    
    # Getting the type of 'float' (line 519)
    float_589739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 43), 'float', False)
    # Processing the call keyword arguments (line 519)
    kwargs_589740 = {}
    # Getting the type of 'np' (line 519)
    np_589733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 23), 'np', False)
    # Obtaining the member 'asarray' of a type (line 519)
    asarray_589734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 23), np_589733, 'asarray')
    # Calling asarray(args, kwargs) (line 519)
    asarray_call_result_589741 = invoke(stypy.reporting.localization.Localization(__file__, 519, 23), asarray_589734, *[subscript_call_result_589738, float_589739], **kwargs_589740)
    
    # Getting the type of 'edges' (line 519)
    edges_589742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'edges')
    # Getting the type of 'i' (line 519)
    i_589743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 18), 'i')
    # Storing an element on a container (line 519)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 12), edges_589742, (i_589743, asarray_call_result_589741))
    
    # Assigning a BinOp to a Subscript (line 520):
    
    # Assigning a BinOp to a Subscript (line 520):
    
    # Call to len(...): (line 520)
    # Processing the call arguments (line 520)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 520)
    i_589745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 32), 'i', False)
    # Getting the type of 'edges' (line 520)
    edges_589746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 26), 'edges', False)
    # Obtaining the member '__getitem__' of a type (line 520)
    getitem___589747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 26), edges_589746, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 520)
    subscript_call_result_589748 = invoke(stypy.reporting.localization.Localization(__file__, 520, 26), getitem___589747, i_589745)
    
    # Processing the call keyword arguments (line 520)
    kwargs_589749 = {}
    # Getting the type of 'len' (line 520)
    len_589744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 22), 'len', False)
    # Calling len(args, kwargs) (line 520)
    len_call_result_589750 = invoke(stypy.reporting.localization.Localization(__file__, 520, 22), len_589744, *[subscript_call_result_589748], **kwargs_589749)
    
    int_589751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 38), 'int')
    # Applying the binary operator '+' (line 520)
    result_add_589752 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 22), '+', len_call_result_589750, int_589751)
    
    # Getting the type of 'nbin' (line 520)
    nbin_589753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'nbin')
    # Getting the type of 'i' (line 520)
    i_589754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 17), 'i')
    # Storing an element on a container (line 520)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 12), nbin_589753, (i_589754, result_add_589752))
    # SSA join for if statement (line 515)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 521):
    
    # Assigning a Call to a Subscript (line 521):
    
    # Call to diff(...): (line 521)
    # Processing the call arguments (line 521)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 521)
    i_589757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 34), 'i', False)
    # Getting the type of 'edges' (line 521)
    edges_589758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 28), 'edges', False)
    # Obtaining the member '__getitem__' of a type (line 521)
    getitem___589759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 28), edges_589758, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 521)
    subscript_call_result_589760 = invoke(stypy.reporting.localization.Localization(__file__, 521, 28), getitem___589759, i_589757)
    
    # Processing the call keyword arguments (line 521)
    kwargs_589761 = {}
    # Getting the type of 'np' (line 521)
    np_589755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 20), 'np', False)
    # Obtaining the member 'diff' of a type (line 521)
    diff_589756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 20), np_589755, 'diff')
    # Calling diff(args, kwargs) (line 521)
    diff_call_result_589762 = invoke(stypy.reporting.localization.Localization(__file__, 521, 20), diff_589756, *[subscript_call_result_589760], **kwargs_589761)
    
    # Getting the type of 'dedges' (line 521)
    dedges_589763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'dedges')
    # Getting the type of 'i' (line 521)
    i_589764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 15), 'i')
    # Storing an element on a container (line 521)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 8), dedges_589763, (i_589764, diff_call_result_589762))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 523):
    
    # Assigning a Call to a Name (line 523):
    
    # Call to asarray(...): (line 523)
    # Processing the call arguments (line 523)
    # Getting the type of 'nbin' (line 523)
    nbin_589767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 22), 'nbin', False)
    # Processing the call keyword arguments (line 523)
    kwargs_589768 = {}
    # Getting the type of 'np' (line 523)
    np_589765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 11), 'np', False)
    # Obtaining the member 'asarray' of a type (line 523)
    asarray_589766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 11), np_589765, 'asarray')
    # Calling asarray(args, kwargs) (line 523)
    asarray_call_result_589769 = invoke(stypy.reporting.localization.Localization(__file__, 523, 11), asarray_589766, *[nbin_589767], **kwargs_589768)
    
    # Assigning a type to the variable 'nbin' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'nbin', asarray_call_result_589769)
    
    # Assigning a ListComp to a Name (line 526):
    
    # Assigning a ListComp to a Name (line 526):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 528)
    # Processing the call arguments (line 528)
    # Getting the type of 'Ndim' (line 528)
    Ndim_589784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 24), 'Ndim', False)
    # Processing the call keyword arguments (line 528)
    kwargs_589785 = {}
    # Getting the type of 'xrange' (line 528)
    xrange_589783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 17), 'xrange', False)
    # Calling xrange(args, kwargs) (line 528)
    xrange_call_result_589786 = invoke(stypy.reporting.localization.Localization(__file__, 528, 17), xrange_589783, *[Ndim_589784], **kwargs_589785)
    
    comprehension_589787 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 8), xrange_call_result_589786)
    # Assigning a type to the variable 'i' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'i', comprehension_589787)
    
    # Call to digitize(...): (line 527)
    # Processing the call arguments (line 527)
    
    # Obtaining the type of the subscript
    slice_589772 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 527, 20), None, None, None)
    # Getting the type of 'i' (line 527)
    i_589773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 30), 'i', False)
    # Getting the type of 'sample' (line 527)
    sample_589774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 20), 'sample', False)
    # Obtaining the member '__getitem__' of a type (line 527)
    getitem___589775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 20), sample_589774, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 527)
    subscript_call_result_589776 = invoke(stypy.reporting.localization.Localization(__file__, 527, 20), getitem___589775, (slice_589772, i_589773))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 527)
    i_589777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 40), 'i', False)
    # Getting the type of 'edges' (line 527)
    edges_589778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 34), 'edges', False)
    # Obtaining the member '__getitem__' of a type (line 527)
    getitem___589779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 34), edges_589778, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 527)
    subscript_call_result_589780 = invoke(stypy.reporting.localization.Localization(__file__, 527, 34), getitem___589779, i_589777)
    
    # Processing the call keyword arguments (line 527)
    kwargs_589781 = {}
    # Getting the type of 'np' (line 527)
    np_589770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'np', False)
    # Obtaining the member 'digitize' of a type (line 527)
    digitize_589771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 8), np_589770, 'digitize')
    # Calling digitize(args, kwargs) (line 527)
    digitize_call_result_589782 = invoke(stypy.reporting.localization.Localization(__file__, 527, 8), digitize_589771, *[subscript_call_result_589776, subscript_call_result_589780], **kwargs_589781)
    
    list_589788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 8), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 8), list_589788, digitize_call_result_589782)
    # Assigning a type to the variable 'sampBin' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'sampBin', list_589788)
    
    
    # Call to xrange(...): (line 534)
    # Processing the call arguments (line 534)
    # Getting the type of 'Ndim' (line 534)
    Ndim_589790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 20), 'Ndim', False)
    # Processing the call keyword arguments (line 534)
    kwargs_589791 = {}
    # Getting the type of 'xrange' (line 534)
    xrange_589789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 534)
    xrange_call_result_589792 = invoke(stypy.reporting.localization.Localization(__file__, 534, 13), xrange_589789, *[Ndim_589790], **kwargs_589791)
    
    # Testing the type of a for loop iterable (line 534)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 534, 4), xrange_call_result_589792)
    # Getting the type of the for loop variable (line 534)
    for_loop_var_589793 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 534, 4), xrange_call_result_589792)
    # Assigning a type to the variable 'i' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'i', for_loop_var_589793)
    # SSA begins for a for statement (line 534)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 536):
    
    # Assigning a BinOp to a Name (line 536):
    
    # Call to int(...): (line 536)
    # Processing the call arguments (line 536)
    
    
    # Call to log10(...): (line 536)
    # Processing the call arguments (line 536)
    
    # Call to min(...): (line 536)
    # Processing the call keyword arguments (line 536)
    kwargs_589802 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 536)
    i_589797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 39), 'i', False)
    # Getting the type of 'dedges' (line 536)
    dedges_589798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 32), 'dedges', False)
    # Obtaining the member '__getitem__' of a type (line 536)
    getitem___589799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 32), dedges_589798, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 536)
    subscript_call_result_589800 = invoke(stypy.reporting.localization.Localization(__file__, 536, 32), getitem___589799, i_589797)
    
    # Obtaining the member 'min' of a type (line 536)
    min_589801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 32), subscript_call_result_589800, 'min')
    # Calling min(args, kwargs) (line 536)
    min_call_result_589803 = invoke(stypy.reporting.localization.Localization(__file__, 536, 32), min_589801, *[], **kwargs_589802)
    
    # Processing the call keyword arguments (line 536)
    kwargs_589804 = {}
    # Getting the type of 'np' (line 536)
    np_589795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 23), 'np', False)
    # Obtaining the member 'log10' of a type (line 536)
    log10_589796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 23), np_589795, 'log10')
    # Calling log10(args, kwargs) (line 536)
    log10_call_result_589805 = invoke(stypy.reporting.localization.Localization(__file__, 536, 23), log10_589796, *[min_call_result_589803], **kwargs_589804)
    
    # Applying the 'usub' unary operator (line 536)
    result___neg___589806 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 22), 'usub', log10_call_result_589805)
    
    # Processing the call keyword arguments (line 536)
    kwargs_589807 = {}
    # Getting the type of 'int' (line 536)
    int_589794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 18), 'int', False)
    # Calling int(args, kwargs) (line 536)
    int_call_result_589808 = invoke(stypy.reporting.localization.Localization(__file__, 536, 18), int_589794, *[result___neg___589806], **kwargs_589807)
    
    int_589809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 52), 'int')
    # Applying the binary operator '+' (line 536)
    result_add_589810 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 18), '+', int_call_result_589808, int_589809)
    
    # Assigning a type to the variable 'decimal' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'decimal', result_add_589810)
    
    # Assigning a Subscript to a Name (line 538):
    
    # Assigning a Subscript to a Name (line 538):
    
    # Obtaining the type of the subscript
    int_589811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 61), 'int')
    
    # Call to where(...): (line 538)
    # Processing the call arguments (line 538)
    
    
    # Call to around(...): (line 538)
    # Processing the call arguments (line 538)
    
    # Obtaining the type of the subscript
    slice_589816 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 538, 37), None, None, None)
    # Getting the type of 'i' (line 538)
    i_589817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 47), 'i', False)
    # Getting the type of 'sample' (line 538)
    sample_589818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 37), 'sample', False)
    # Obtaining the member '__getitem__' of a type (line 538)
    getitem___589819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 37), sample_589818, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 538)
    subscript_call_result_589820 = invoke(stypy.reporting.localization.Localization(__file__, 538, 37), getitem___589819, (slice_589816, i_589817))
    
    # Getting the type of 'decimal' (line 538)
    decimal_589821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 51), 'decimal', False)
    # Processing the call keyword arguments (line 538)
    kwargs_589822 = {}
    # Getting the type of 'np' (line 538)
    np_589814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 27), 'np', False)
    # Obtaining the member 'around' of a type (line 538)
    around_589815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 27), np_589814, 'around')
    # Calling around(args, kwargs) (line 538)
    around_call_result_589823 = invoke(stypy.reporting.localization.Localization(__file__, 538, 27), around_589815, *[subscript_call_result_589820, decimal_589821], **kwargs_589822)
    
    
    # Call to around(...): (line 539)
    # Processing the call arguments (line 539)
    
    # Obtaining the type of the subscript
    int_589826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 46), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 539)
    i_589827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 43), 'i', False)
    # Getting the type of 'edges' (line 539)
    edges_589828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 37), 'edges', False)
    # Obtaining the member '__getitem__' of a type (line 539)
    getitem___589829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 37), edges_589828, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 539)
    subscript_call_result_589830 = invoke(stypy.reporting.localization.Localization(__file__, 539, 37), getitem___589829, i_589827)
    
    # Obtaining the member '__getitem__' of a type (line 539)
    getitem___589831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 37), subscript_call_result_589830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 539)
    subscript_call_result_589832 = invoke(stypy.reporting.localization.Localization(__file__, 539, 37), getitem___589831, int_589826)
    
    # Getting the type of 'decimal' (line 539)
    decimal_589833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 51), 'decimal', False)
    # Processing the call keyword arguments (line 539)
    kwargs_589834 = {}
    # Getting the type of 'np' (line 539)
    np_589824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 27), 'np', False)
    # Obtaining the member 'around' of a type (line 539)
    around_589825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 27), np_589824, 'around')
    # Calling around(args, kwargs) (line 539)
    around_call_result_589835 = invoke(stypy.reporting.localization.Localization(__file__, 539, 27), around_589825, *[subscript_call_result_589832, decimal_589833], **kwargs_589834)
    
    # Applying the binary operator '==' (line 538)
    result_eq_589836 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 27), '==', around_call_result_589823, around_call_result_589835)
    
    # Processing the call keyword arguments (line 538)
    kwargs_589837 = {}
    # Getting the type of 'np' (line 538)
    np_589812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 18), 'np', False)
    # Obtaining the member 'where' of a type (line 538)
    where_589813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 18), np_589812, 'where')
    # Calling where(args, kwargs) (line 538)
    where_call_result_589838 = invoke(stypy.reporting.localization.Localization(__file__, 538, 18), where_589813, *[result_eq_589836], **kwargs_589837)
    
    # Obtaining the member '__getitem__' of a type (line 538)
    getitem___589839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 18), where_call_result_589838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 538)
    subscript_call_result_589840 = invoke(stypy.reporting.localization.Localization(__file__, 538, 18), getitem___589839, int_589811)
    
    # Assigning a type to the variable 'on_edge' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'on_edge', subscript_call_result_589840)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 541)
    i_589841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'i')
    # Getting the type of 'sampBin' (line 541)
    sampBin_589842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'sampBin')
    # Obtaining the member '__getitem__' of a type (line 541)
    getitem___589843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 8), sampBin_589842, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 541)
    subscript_call_result_589844 = invoke(stypy.reporting.localization.Localization(__file__, 541, 8), getitem___589843, i_589841)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'on_edge' (line 541)
    on_edge_589845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 19), 'on_edge')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 541)
    i_589846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'i')
    # Getting the type of 'sampBin' (line 541)
    sampBin_589847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'sampBin')
    # Obtaining the member '__getitem__' of a type (line 541)
    getitem___589848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 8), sampBin_589847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 541)
    subscript_call_result_589849 = invoke(stypy.reporting.localization.Localization(__file__, 541, 8), getitem___589848, i_589846)
    
    # Obtaining the member '__getitem__' of a type (line 541)
    getitem___589850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 8), subscript_call_result_589849, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 541)
    subscript_call_result_589851 = invoke(stypy.reporting.localization.Localization(__file__, 541, 8), getitem___589850, on_edge_589845)
    
    int_589852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 31), 'int')
    # Applying the binary operator '-=' (line 541)
    result_isub_589853 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 8), '-=', subscript_call_result_589851, int_589852)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 541)
    i_589854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'i')
    # Getting the type of 'sampBin' (line 541)
    sampBin_589855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'sampBin')
    # Obtaining the member '__getitem__' of a type (line 541)
    getitem___589856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 8), sampBin_589855, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 541)
    subscript_call_result_589857 = invoke(stypy.reporting.localization.Localization(__file__, 541, 8), getitem___589856, i_589854)
    
    # Getting the type of 'on_edge' (line 541)
    on_edge_589858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 19), 'on_edge')
    # Storing an element on a container (line 541)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 8), subscript_call_result_589857, (on_edge_589858, result_isub_589853))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 544):
    
    # Assigning a Call to a Name (line 544):
    
    # Call to ravel_multi_index(...): (line 544)
    # Processing the call arguments (line 544)
    # Getting the type of 'sampBin' (line 544)
    sampBin_589861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 38), 'sampBin', False)
    # Getting the type of 'nbin' (line 544)
    nbin_589862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 47), 'nbin', False)
    # Processing the call keyword arguments (line 544)
    kwargs_589863 = {}
    # Getting the type of 'np' (line 544)
    np_589859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 17), 'np', False)
    # Obtaining the member 'ravel_multi_index' of a type (line 544)
    ravel_multi_index_589860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 17), np_589859, 'ravel_multi_index')
    # Calling ravel_multi_index(args, kwargs) (line 544)
    ravel_multi_index_call_result_589864 = invoke(stypy.reporting.localization.Localization(__file__, 544, 17), ravel_multi_index_589860, *[sampBin_589861, nbin_589862], **kwargs_589863)
    
    # Assigning a type to the variable 'binnumbers' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'binnumbers', ravel_multi_index_call_result_589864)
    
    # Assigning a Call to a Name (line 546):
    
    # Assigning a Call to a Name (line 546):
    
    # Call to empty(...): (line 546)
    # Processing the call arguments (line 546)
    
    # Obtaining an instance of the builtin type 'list' (line 546)
    list_589867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 546)
    # Adding element type (line 546)
    # Getting the type of 'Vdim' (line 546)
    Vdim_589868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 23), 'Vdim', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 22), list_589867, Vdim_589868)
    # Adding element type (line 546)
    
    # Call to prod(...): (line 546)
    # Processing the call keyword arguments (line 546)
    kwargs_589871 = {}
    # Getting the type of 'nbin' (line 546)
    nbin_589869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 29), 'nbin', False)
    # Obtaining the member 'prod' of a type (line 546)
    prod_589870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 29), nbin_589869, 'prod')
    # Calling prod(args, kwargs) (line 546)
    prod_call_result_589872 = invoke(stypy.reporting.localization.Localization(__file__, 546, 29), prod_589870, *[], **kwargs_589871)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 22), list_589867, prod_call_result_589872)
    
    # Getting the type of 'float' (line 546)
    float_589873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 43), 'float', False)
    # Processing the call keyword arguments (line 546)
    kwargs_589874 = {}
    # Getting the type of 'np' (line 546)
    np_589865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 13), 'np', False)
    # Obtaining the member 'empty' of a type (line 546)
    empty_589866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 13), np_589865, 'empty')
    # Calling empty(args, kwargs) (line 546)
    empty_call_result_589875 = invoke(stypy.reporting.localization.Localization(__file__, 546, 13), empty_589866, *[list_589867, float_589873], **kwargs_589874)
    
    # Assigning a type to the variable 'result' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'result', empty_call_result_589875)
    
    
    # Getting the type of 'statistic' (line 548)
    statistic_589876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 7), 'statistic')
    str_589877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 20), 'str', 'mean')
    # Applying the binary operator '==' (line 548)
    result_eq_589878 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 7), '==', statistic_589876, str_589877)
    
    # Testing the type of an if condition (line 548)
    if_condition_589879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 4), result_eq_589878)
    # Assigning a type to the variable 'if_condition_589879' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 4), 'if_condition_589879', if_condition_589879)
    # SSA begins for if statement (line 548)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fill(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'np' (line 549)
    np_589882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'np', False)
    # Obtaining the member 'nan' of a type (line 549)
    nan_589883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 20), np_589882, 'nan')
    # Processing the call keyword arguments (line 549)
    kwargs_589884 = {}
    # Getting the type of 'result' (line 549)
    result_589880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'result', False)
    # Obtaining the member 'fill' of a type (line 549)
    fill_589881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 8), result_589880, 'fill')
    # Calling fill(args, kwargs) (line 549)
    fill_call_result_589885 = invoke(stypy.reporting.localization.Localization(__file__, 549, 8), fill_589881, *[nan_589883], **kwargs_589884)
    
    
    # Assigning a Call to a Name (line 550):
    
    # Assigning a Call to a Name (line 550):
    
    # Call to bincount(...): (line 550)
    # Processing the call arguments (line 550)
    # Getting the type of 'binnumbers' (line 550)
    binnumbers_589888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 32), 'binnumbers', False)
    # Getting the type of 'None' (line 550)
    None_589889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 44), 'None', False)
    # Processing the call keyword arguments (line 550)
    kwargs_589890 = {}
    # Getting the type of 'np' (line 550)
    np_589886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 20), 'np', False)
    # Obtaining the member 'bincount' of a type (line 550)
    bincount_589887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 20), np_589886, 'bincount')
    # Calling bincount(args, kwargs) (line 550)
    bincount_call_result_589891 = invoke(stypy.reporting.localization.Localization(__file__, 550, 20), bincount_589887, *[binnumbers_589888, None_589889], **kwargs_589890)
    
    # Assigning a type to the variable 'flatcount' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'flatcount', bincount_call_result_589891)
    
    # Assigning a Call to a Name (line 551):
    
    # Assigning a Call to a Name (line 551):
    
    # Call to nonzero(...): (line 551)
    # Processing the call keyword arguments (line 551)
    kwargs_589894 = {}
    # Getting the type of 'flatcount' (line 551)
    flatcount_589892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'flatcount', False)
    # Obtaining the member 'nonzero' of a type (line 551)
    nonzero_589893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 12), flatcount_589892, 'nonzero')
    # Calling nonzero(args, kwargs) (line 551)
    nonzero_call_result_589895 = invoke(stypy.reporting.localization.Localization(__file__, 551, 12), nonzero_589893, *[], **kwargs_589894)
    
    # Assigning a type to the variable 'a' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'a', nonzero_call_result_589895)
    
    
    # Call to xrange(...): (line 552)
    # Processing the call arguments (line 552)
    # Getting the type of 'Vdim' (line 552)
    Vdim_589897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 25), 'Vdim', False)
    # Processing the call keyword arguments (line 552)
    kwargs_589898 = {}
    # Getting the type of 'xrange' (line 552)
    xrange_589896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 18), 'xrange', False)
    # Calling xrange(args, kwargs) (line 552)
    xrange_call_result_589899 = invoke(stypy.reporting.localization.Localization(__file__, 552, 18), xrange_589896, *[Vdim_589897], **kwargs_589898)
    
    # Testing the type of a for loop iterable (line 552)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 552, 8), xrange_call_result_589899)
    # Getting the type of the for loop variable (line 552)
    for_loop_var_589900 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 552, 8), xrange_call_result_589899)
    # Assigning a type to the variable 'vv' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'vv', for_loop_var_589900)
    # SSA begins for a for statement (line 552)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 553):
    
    # Assigning a Call to a Name (line 553):
    
    # Call to bincount(...): (line 553)
    # Processing the call arguments (line 553)
    # Getting the type of 'binnumbers' (line 553)
    binnumbers_589903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 34), 'binnumbers', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'vv' (line 553)
    vv_589904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 53), 'vv', False)
    # Getting the type of 'values' (line 553)
    values_589905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 46), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 553)
    getitem___589906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 46), values_589905, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 553)
    subscript_call_result_589907 = invoke(stypy.reporting.localization.Localization(__file__, 553, 46), getitem___589906, vv_589904)
    
    # Processing the call keyword arguments (line 553)
    kwargs_589908 = {}
    # Getting the type of 'np' (line 553)
    np_589901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 22), 'np', False)
    # Obtaining the member 'bincount' of a type (line 553)
    bincount_589902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 22), np_589901, 'bincount')
    # Calling bincount(args, kwargs) (line 553)
    bincount_call_result_589909 = invoke(stypy.reporting.localization.Localization(__file__, 553, 22), bincount_589902, *[binnumbers_589903, subscript_call_result_589907], **kwargs_589908)
    
    # Assigning a type to the variable 'flatsum' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'flatsum', bincount_call_result_589909)
    
    # Assigning a BinOp to a Subscript (line 554):
    
    # Assigning a BinOp to a Subscript (line 554):
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 554)
    a_589910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 36), 'a')
    # Getting the type of 'flatsum' (line 554)
    flatsum_589911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 28), 'flatsum')
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___589912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 28), flatsum_589911, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_589913 = invoke(stypy.reporting.localization.Localization(__file__, 554, 28), getitem___589912, a_589910)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 554)
    a_589914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 51), 'a')
    # Getting the type of 'flatcount' (line 554)
    flatcount_589915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 41), 'flatcount')
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___589916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 41), flatcount_589915, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_589917 = invoke(stypy.reporting.localization.Localization(__file__, 554, 41), getitem___589916, a_589914)
    
    # Applying the binary operator 'div' (line 554)
    result_div_589918 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 28), 'div', subscript_call_result_589913, subscript_call_result_589917)
    
    # Getting the type of 'result' (line 554)
    result_589919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'result')
    
    # Obtaining an instance of the builtin type 'tuple' (line 554)
    tuple_589920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 554)
    # Adding element type (line 554)
    # Getting the type of 'vv' (line 554)
    vv_589921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 19), 'vv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 19), tuple_589920, vv_589921)
    # Adding element type (line 554)
    # Getting the type of 'a' (line 554)
    a_589922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 23), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 19), tuple_589920, a_589922)
    
    # Storing an element on a container (line 554)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 12), result_589919, (tuple_589920, result_div_589918))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 548)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'statistic' (line 555)
    statistic_589923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 9), 'statistic')
    str_589924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 22), 'str', 'std')
    # Applying the binary operator '==' (line 555)
    result_eq_589925 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 9), '==', statistic_589923, str_589924)
    
    # Testing the type of an if condition (line 555)
    if_condition_589926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 9), result_eq_589925)
    # Assigning a type to the variable 'if_condition_589926' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 9), 'if_condition_589926', if_condition_589926)
    # SSA begins for if statement (line 555)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fill(...): (line 556)
    # Processing the call arguments (line 556)
    int_589929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 20), 'int')
    # Processing the call keyword arguments (line 556)
    kwargs_589930 = {}
    # Getting the type of 'result' (line 556)
    result_589927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'result', False)
    # Obtaining the member 'fill' of a type (line 556)
    fill_589928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 8), result_589927, 'fill')
    # Calling fill(args, kwargs) (line 556)
    fill_call_result_589931 = invoke(stypy.reporting.localization.Localization(__file__, 556, 8), fill_589928, *[int_589929], **kwargs_589930)
    
    
    # Assigning a Call to a Name (line 557):
    
    # Assigning a Call to a Name (line 557):
    
    # Call to bincount(...): (line 557)
    # Processing the call arguments (line 557)
    # Getting the type of 'binnumbers' (line 557)
    binnumbers_589934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 32), 'binnumbers', False)
    # Getting the type of 'None' (line 557)
    None_589935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 44), 'None', False)
    # Processing the call keyword arguments (line 557)
    kwargs_589936 = {}
    # Getting the type of 'np' (line 557)
    np_589932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 20), 'np', False)
    # Obtaining the member 'bincount' of a type (line 557)
    bincount_589933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 20), np_589932, 'bincount')
    # Calling bincount(args, kwargs) (line 557)
    bincount_call_result_589937 = invoke(stypy.reporting.localization.Localization(__file__, 557, 20), bincount_589933, *[binnumbers_589934, None_589935], **kwargs_589936)
    
    # Assigning a type to the variable 'flatcount' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'flatcount', bincount_call_result_589937)
    
    # Assigning a Call to a Name (line 558):
    
    # Assigning a Call to a Name (line 558):
    
    # Call to nonzero(...): (line 558)
    # Processing the call keyword arguments (line 558)
    kwargs_589940 = {}
    # Getting the type of 'flatcount' (line 558)
    flatcount_589938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'flatcount', False)
    # Obtaining the member 'nonzero' of a type (line 558)
    nonzero_589939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 12), flatcount_589938, 'nonzero')
    # Calling nonzero(args, kwargs) (line 558)
    nonzero_call_result_589941 = invoke(stypy.reporting.localization.Localization(__file__, 558, 12), nonzero_589939, *[], **kwargs_589940)
    
    # Assigning a type to the variable 'a' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'a', nonzero_call_result_589941)
    
    
    # Call to xrange(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'Vdim' (line 559)
    Vdim_589943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 25), 'Vdim', False)
    # Processing the call keyword arguments (line 559)
    kwargs_589944 = {}
    # Getting the type of 'xrange' (line 559)
    xrange_589942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 18), 'xrange', False)
    # Calling xrange(args, kwargs) (line 559)
    xrange_call_result_589945 = invoke(stypy.reporting.localization.Localization(__file__, 559, 18), xrange_589942, *[Vdim_589943], **kwargs_589944)
    
    # Testing the type of a for loop iterable (line 559)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 559, 8), xrange_call_result_589945)
    # Getting the type of the for loop variable (line 559)
    for_loop_var_589946 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 559, 8), xrange_call_result_589945)
    # Assigning a type to the variable 'vv' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'vv', for_loop_var_589946)
    # SSA begins for a for statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 560):
    
    # Assigning a Call to a Name (line 560):
    
    # Call to bincount(...): (line 560)
    # Processing the call arguments (line 560)
    # Getting the type of 'binnumbers' (line 560)
    binnumbers_589949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 34), 'binnumbers', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'vv' (line 560)
    vv_589950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 53), 'vv', False)
    # Getting the type of 'values' (line 560)
    values_589951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 46), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 560)
    getitem___589952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 46), values_589951, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 560)
    subscript_call_result_589953 = invoke(stypy.reporting.localization.Localization(__file__, 560, 46), getitem___589952, vv_589950)
    
    # Processing the call keyword arguments (line 560)
    kwargs_589954 = {}
    # Getting the type of 'np' (line 560)
    np_589947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 22), 'np', False)
    # Obtaining the member 'bincount' of a type (line 560)
    bincount_589948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 22), np_589947, 'bincount')
    # Calling bincount(args, kwargs) (line 560)
    bincount_call_result_589955 = invoke(stypy.reporting.localization.Localization(__file__, 560, 22), bincount_589948, *[binnumbers_589949, subscript_call_result_589953], **kwargs_589954)
    
    # Assigning a type to the variable 'flatsum' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'flatsum', bincount_call_result_589955)
    
    # Assigning a Call to a Name (line 561):
    
    # Assigning a Call to a Name (line 561):
    
    # Call to bincount(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'binnumbers' (line 561)
    binnumbers_589958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 35), 'binnumbers', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'vv' (line 561)
    vv_589959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 54), 'vv', False)
    # Getting the type of 'values' (line 561)
    values_589960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 47), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 561)
    getitem___589961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 47), values_589960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 561)
    subscript_call_result_589962 = invoke(stypy.reporting.localization.Localization(__file__, 561, 47), getitem___589961, vv_589959)
    
    int_589963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 61), 'int')
    # Applying the binary operator '**' (line 561)
    result_pow_589964 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 47), '**', subscript_call_result_589962, int_589963)
    
    # Processing the call keyword arguments (line 561)
    kwargs_589965 = {}
    # Getting the type of 'np' (line 561)
    np_589956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 23), 'np', False)
    # Obtaining the member 'bincount' of a type (line 561)
    bincount_589957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 23), np_589956, 'bincount')
    # Calling bincount(args, kwargs) (line 561)
    bincount_call_result_589966 = invoke(stypy.reporting.localization.Localization(__file__, 561, 23), bincount_589957, *[binnumbers_589958, result_pow_589964], **kwargs_589965)
    
    # Assigning a type to the variable 'flatsum2' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'flatsum2', bincount_call_result_589966)
    
    # Assigning a Call to a Subscript (line 562):
    
    # Assigning a Call to a Subscript (line 562):
    
    # Call to sqrt(...): (line 562)
    # Processing the call arguments (line 562)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 562)
    a_589969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 45), 'a', False)
    # Getting the type of 'flatsum2' (line 562)
    flatsum2_589970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 36), 'flatsum2', False)
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___589971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 36), flatsum2_589970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_589972 = invoke(stypy.reporting.localization.Localization(__file__, 562, 36), getitem___589971, a_589969)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 562)
    a_589973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 60), 'a', False)
    # Getting the type of 'flatcount' (line 562)
    flatcount_589974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 50), 'flatcount', False)
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___589975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 50), flatcount_589974, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_589976 = invoke(stypy.reporting.localization.Localization(__file__, 562, 50), getitem___589975, a_589973)
    
    # Applying the binary operator 'div' (line 562)
    result_div_589977 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 36), 'div', subscript_call_result_589972, subscript_call_result_589976)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 563)
    a_589978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 45), 'a', False)
    # Getting the type of 'flatsum' (line 563)
    flatsum_589979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 37), 'flatsum', False)
    # Obtaining the member '__getitem__' of a type (line 563)
    getitem___589980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 37), flatsum_589979, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 563)
    subscript_call_result_589981 = invoke(stypy.reporting.localization.Localization(__file__, 563, 37), getitem___589980, a_589978)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 563)
    a_589982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 60), 'a', False)
    # Getting the type of 'flatcount' (line 563)
    flatcount_589983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 50), 'flatcount', False)
    # Obtaining the member '__getitem__' of a type (line 563)
    getitem___589984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 50), flatcount_589983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 563)
    subscript_call_result_589985 = invoke(stypy.reporting.localization.Localization(__file__, 563, 50), getitem___589984, a_589982)
    
    # Applying the binary operator 'div' (line 563)
    result_div_589986 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 37), 'div', subscript_call_result_589981, subscript_call_result_589985)
    
    int_589987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 67), 'int')
    # Applying the binary operator '**' (line 563)
    result_pow_589988 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 36), '**', result_div_589986, int_589987)
    
    # Applying the binary operator '-' (line 562)
    result_sub_589989 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 36), '-', result_div_589977, result_pow_589988)
    
    # Processing the call keyword arguments (line 562)
    kwargs_589990 = {}
    # Getting the type of 'np' (line 562)
    np_589967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 28), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 562)
    sqrt_589968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 28), np_589967, 'sqrt')
    # Calling sqrt(args, kwargs) (line 562)
    sqrt_call_result_589991 = invoke(stypy.reporting.localization.Localization(__file__, 562, 28), sqrt_589968, *[result_sub_589989], **kwargs_589990)
    
    # Getting the type of 'result' (line 562)
    result_589992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'result')
    
    # Obtaining an instance of the builtin type 'tuple' (line 562)
    tuple_589993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 562)
    # Adding element type (line 562)
    # Getting the type of 'vv' (line 562)
    vv_589994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 19), 'vv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 19), tuple_589993, vv_589994)
    # Adding element type (line 562)
    # Getting the type of 'a' (line 562)
    a_589995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 23), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 19), tuple_589993, a_589995)
    
    # Storing an element on a container (line 562)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 12), result_589992, (tuple_589993, sqrt_call_result_589991))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 555)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'statistic' (line 564)
    statistic_589996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 9), 'statistic')
    str_589997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 22), 'str', 'count')
    # Applying the binary operator '==' (line 564)
    result_eq_589998 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 9), '==', statistic_589996, str_589997)
    
    # Testing the type of an if condition (line 564)
    if_condition_589999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 564, 9), result_eq_589998)
    # Assigning a type to the variable 'if_condition_589999' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 9), 'if_condition_589999', if_condition_589999)
    # SSA begins for if statement (line 564)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fill(...): (line 565)
    # Processing the call arguments (line 565)
    int_590002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 20), 'int')
    # Processing the call keyword arguments (line 565)
    kwargs_590003 = {}
    # Getting the type of 'result' (line 565)
    result_590000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'result', False)
    # Obtaining the member 'fill' of a type (line 565)
    fill_590001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 8), result_590000, 'fill')
    # Calling fill(args, kwargs) (line 565)
    fill_call_result_590004 = invoke(stypy.reporting.localization.Localization(__file__, 565, 8), fill_590001, *[int_590002], **kwargs_590003)
    
    
    # Assigning a Call to a Name (line 566):
    
    # Assigning a Call to a Name (line 566):
    
    # Call to bincount(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'binnumbers' (line 566)
    binnumbers_590007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 32), 'binnumbers', False)
    # Getting the type of 'None' (line 566)
    None_590008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 44), 'None', False)
    # Processing the call keyword arguments (line 566)
    kwargs_590009 = {}
    # Getting the type of 'np' (line 566)
    np_590005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 20), 'np', False)
    # Obtaining the member 'bincount' of a type (line 566)
    bincount_590006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 20), np_590005, 'bincount')
    # Calling bincount(args, kwargs) (line 566)
    bincount_call_result_590010 = invoke(stypy.reporting.localization.Localization(__file__, 566, 20), bincount_590006, *[binnumbers_590007, None_590008], **kwargs_590009)
    
    # Assigning a type to the variable 'flatcount' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'flatcount', bincount_call_result_590010)
    
    # Assigning a Call to a Name (line 567):
    
    # Assigning a Call to a Name (line 567):
    
    # Call to arange(...): (line 567)
    # Processing the call arguments (line 567)
    
    # Call to len(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'flatcount' (line 567)
    flatcount_590014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 26), 'flatcount', False)
    # Processing the call keyword arguments (line 567)
    kwargs_590015 = {}
    # Getting the type of 'len' (line 567)
    len_590013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 22), 'len', False)
    # Calling len(args, kwargs) (line 567)
    len_call_result_590016 = invoke(stypy.reporting.localization.Localization(__file__, 567, 22), len_590013, *[flatcount_590014], **kwargs_590015)
    
    # Processing the call keyword arguments (line 567)
    kwargs_590017 = {}
    # Getting the type of 'np' (line 567)
    np_590011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'np', False)
    # Obtaining the member 'arange' of a type (line 567)
    arange_590012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 12), np_590011, 'arange')
    # Calling arange(args, kwargs) (line 567)
    arange_call_result_590018 = invoke(stypy.reporting.localization.Localization(__file__, 567, 12), arange_590012, *[len_call_result_590016], **kwargs_590017)
    
    # Assigning a type to the variable 'a' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'a', arange_call_result_590018)
    
    # Assigning a Subscript to a Subscript (line 568):
    
    # Assigning a Subscript to a Subscript (line 568):
    
    # Obtaining the type of the subscript
    # Getting the type of 'np' (line 568)
    np_590019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 33), 'np')
    # Obtaining the member 'newaxis' of a type (line 568)
    newaxis_590020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 33), np_590019, 'newaxis')
    slice_590021 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 568, 23), None, None, None)
    # Getting the type of 'flatcount' (line 568)
    flatcount_590022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 23), 'flatcount')
    # Obtaining the member '__getitem__' of a type (line 568)
    getitem___590023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 23), flatcount_590022, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 568)
    subscript_call_result_590024 = invoke(stypy.reporting.localization.Localization(__file__, 568, 23), getitem___590023, (newaxis_590020, slice_590021))
    
    # Getting the type of 'result' (line 568)
    result_590025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'result')
    slice_590026 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 568, 8), None, None, None)
    # Getting the type of 'a' (line 568)
    a_590027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 18), 'a')
    # Storing an element on a container (line 568)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 8), result_590025, ((slice_590026, a_590027), subscript_call_result_590024))
    # SSA branch for the else part of an if statement (line 564)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'statistic' (line 569)
    statistic_590028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 9), 'statistic')
    str_590029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 22), 'str', 'sum')
    # Applying the binary operator '==' (line 569)
    result_eq_590030 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 9), '==', statistic_590028, str_590029)
    
    # Testing the type of an if condition (line 569)
    if_condition_590031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 569, 9), result_eq_590030)
    # Assigning a type to the variable 'if_condition_590031' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 9), 'if_condition_590031', if_condition_590031)
    # SSA begins for if statement (line 569)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fill(...): (line 570)
    # Processing the call arguments (line 570)
    int_590034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 20), 'int')
    # Processing the call keyword arguments (line 570)
    kwargs_590035 = {}
    # Getting the type of 'result' (line 570)
    result_590032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'result', False)
    # Obtaining the member 'fill' of a type (line 570)
    fill_590033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), result_590032, 'fill')
    # Calling fill(args, kwargs) (line 570)
    fill_call_result_590036 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), fill_590033, *[int_590034], **kwargs_590035)
    
    
    
    # Call to xrange(...): (line 571)
    # Processing the call arguments (line 571)
    # Getting the type of 'Vdim' (line 571)
    Vdim_590038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 25), 'Vdim', False)
    # Processing the call keyword arguments (line 571)
    kwargs_590039 = {}
    # Getting the type of 'xrange' (line 571)
    xrange_590037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 18), 'xrange', False)
    # Calling xrange(args, kwargs) (line 571)
    xrange_call_result_590040 = invoke(stypy.reporting.localization.Localization(__file__, 571, 18), xrange_590037, *[Vdim_590038], **kwargs_590039)
    
    # Testing the type of a for loop iterable (line 571)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 571, 8), xrange_call_result_590040)
    # Getting the type of the for loop variable (line 571)
    for_loop_var_590041 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 571, 8), xrange_call_result_590040)
    # Assigning a type to the variable 'vv' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'vv', for_loop_var_590041)
    # SSA begins for a for statement (line 571)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 572):
    
    # Assigning a Call to a Name (line 572):
    
    # Call to bincount(...): (line 572)
    # Processing the call arguments (line 572)
    # Getting the type of 'binnumbers' (line 572)
    binnumbers_590044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 34), 'binnumbers', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'vv' (line 572)
    vv_590045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 53), 'vv', False)
    # Getting the type of 'values' (line 572)
    values_590046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 46), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 572)
    getitem___590047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 46), values_590046, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 572)
    subscript_call_result_590048 = invoke(stypy.reporting.localization.Localization(__file__, 572, 46), getitem___590047, vv_590045)
    
    # Processing the call keyword arguments (line 572)
    kwargs_590049 = {}
    # Getting the type of 'np' (line 572)
    np_590042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 22), 'np', False)
    # Obtaining the member 'bincount' of a type (line 572)
    bincount_590043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 22), np_590042, 'bincount')
    # Calling bincount(args, kwargs) (line 572)
    bincount_call_result_590050 = invoke(stypy.reporting.localization.Localization(__file__, 572, 22), bincount_590043, *[binnumbers_590044, subscript_call_result_590048], **kwargs_590049)
    
    # Assigning a type to the variable 'flatsum' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'flatsum', bincount_call_result_590050)
    
    # Assigning a Call to a Name (line 573):
    
    # Assigning a Call to a Name (line 573):
    
    # Call to arange(...): (line 573)
    # Processing the call arguments (line 573)
    
    # Call to len(...): (line 573)
    # Processing the call arguments (line 573)
    # Getting the type of 'flatsum' (line 573)
    flatsum_590054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 30), 'flatsum', False)
    # Processing the call keyword arguments (line 573)
    kwargs_590055 = {}
    # Getting the type of 'len' (line 573)
    len_590053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 26), 'len', False)
    # Calling len(args, kwargs) (line 573)
    len_call_result_590056 = invoke(stypy.reporting.localization.Localization(__file__, 573, 26), len_590053, *[flatsum_590054], **kwargs_590055)
    
    # Processing the call keyword arguments (line 573)
    kwargs_590057 = {}
    # Getting the type of 'np' (line 573)
    np_590051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 16), 'np', False)
    # Obtaining the member 'arange' of a type (line 573)
    arange_590052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 16), np_590051, 'arange')
    # Calling arange(args, kwargs) (line 573)
    arange_call_result_590058 = invoke(stypy.reporting.localization.Localization(__file__, 573, 16), arange_590052, *[len_call_result_590056], **kwargs_590057)
    
    # Assigning a type to the variable 'a' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'a', arange_call_result_590058)
    
    # Assigning a Name to a Subscript (line 574):
    
    # Assigning a Name to a Subscript (line 574):
    # Getting the type of 'flatsum' (line 574)
    flatsum_590059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 28), 'flatsum')
    # Getting the type of 'result' (line 574)
    result_590060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'result')
    
    # Obtaining an instance of the builtin type 'tuple' (line 574)
    tuple_590061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 574)
    # Adding element type (line 574)
    # Getting the type of 'vv' (line 574)
    vv_590062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 19), 'vv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 19), tuple_590061, vv_590062)
    # Adding element type (line 574)
    # Getting the type of 'a' (line 574)
    a_590063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 23), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 19), tuple_590061, a_590063)
    
    # Storing an element on a container (line 574)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 12), result_590060, (tuple_590061, flatsum_590059))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 569)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'statistic' (line 575)
    statistic_590064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 9), 'statistic')
    str_590065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 22), 'str', 'median')
    # Applying the binary operator '==' (line 575)
    result_eq_590066 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 9), '==', statistic_590064, str_590065)
    
    # Testing the type of an if condition (line 575)
    if_condition_590067 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 9), result_eq_590066)
    # Assigning a type to the variable 'if_condition_590067' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 9), 'if_condition_590067', if_condition_590067)
    # SSA begins for if statement (line 575)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fill(...): (line 576)
    # Processing the call arguments (line 576)
    # Getting the type of 'np' (line 576)
    np_590070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 20), 'np', False)
    # Obtaining the member 'nan' of a type (line 576)
    nan_590071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 20), np_590070, 'nan')
    # Processing the call keyword arguments (line 576)
    kwargs_590072 = {}
    # Getting the type of 'result' (line 576)
    result_590068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'result', False)
    # Obtaining the member 'fill' of a type (line 576)
    fill_590069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 8), result_590068, 'fill')
    # Calling fill(args, kwargs) (line 576)
    fill_call_result_590073 = invoke(stypy.reporting.localization.Localization(__file__, 576, 8), fill_590069, *[nan_590071], **kwargs_590072)
    
    
    
    # Call to unique(...): (line 577)
    # Processing the call arguments (line 577)
    # Getting the type of 'binnumbers' (line 577)
    binnumbers_590076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 27), 'binnumbers', False)
    # Processing the call keyword arguments (line 577)
    kwargs_590077 = {}
    # Getting the type of 'np' (line 577)
    np_590074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 17), 'np', False)
    # Obtaining the member 'unique' of a type (line 577)
    unique_590075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 17), np_590074, 'unique')
    # Calling unique(args, kwargs) (line 577)
    unique_call_result_590078 = invoke(stypy.reporting.localization.Localization(__file__, 577, 17), unique_590075, *[binnumbers_590076], **kwargs_590077)
    
    # Testing the type of a for loop iterable (line 577)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 577, 8), unique_call_result_590078)
    # Getting the type of the for loop variable (line 577)
    for_loop_var_590079 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 577, 8), unique_call_result_590078)
    # Assigning a type to the variable 'i' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'i', for_loop_var_590079)
    # SSA begins for a for statement (line 577)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to xrange(...): (line 578)
    # Processing the call arguments (line 578)
    # Getting the type of 'Vdim' (line 578)
    Vdim_590081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 29), 'Vdim', False)
    # Processing the call keyword arguments (line 578)
    kwargs_590082 = {}
    # Getting the type of 'xrange' (line 578)
    xrange_590080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 22), 'xrange', False)
    # Calling xrange(args, kwargs) (line 578)
    xrange_call_result_590083 = invoke(stypy.reporting.localization.Localization(__file__, 578, 22), xrange_590080, *[Vdim_590081], **kwargs_590082)
    
    # Testing the type of a for loop iterable (line 578)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 578, 12), xrange_call_result_590083)
    # Getting the type of the for loop variable (line 578)
    for_loop_var_590084 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 578, 12), xrange_call_result_590083)
    # Assigning a type to the variable 'vv' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'vv', for_loop_var_590084)
    # SSA begins for a for statement (line 578)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 579):
    
    # Assigning a Call to a Subscript (line 579):
    
    # Call to median(...): (line 579)
    # Processing the call arguments (line 579)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 579)
    tuple_590087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 579)
    # Adding element type (line 579)
    # Getting the type of 'vv' (line 579)
    vv_590088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 49), 'vv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 49), tuple_590087, vv_590088)
    # Adding element type (line 579)
    
    # Getting the type of 'binnumbers' (line 579)
    binnumbers_590089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 53), 'binnumbers', False)
    # Getting the type of 'i' (line 579)
    i_590090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 67), 'i', False)
    # Applying the binary operator '==' (line 579)
    result_eq_590091 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 53), '==', binnumbers_590089, i_590090)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 49), tuple_590087, result_eq_590091)
    
    # Getting the type of 'values' (line 579)
    values_590092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 42), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___590093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 42), values_590092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_590094 = invoke(stypy.reporting.localization.Localization(__file__, 579, 42), getitem___590093, tuple_590087)
    
    # Processing the call keyword arguments (line 579)
    kwargs_590095 = {}
    # Getting the type of 'np' (line 579)
    np_590085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 32), 'np', False)
    # Obtaining the member 'median' of a type (line 579)
    median_590086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 32), np_590085, 'median')
    # Calling median(args, kwargs) (line 579)
    median_call_result_590096 = invoke(stypy.reporting.localization.Localization(__file__, 579, 32), median_590086, *[subscript_call_result_590094], **kwargs_590095)
    
    # Getting the type of 'result' (line 579)
    result_590097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'result')
    
    # Obtaining an instance of the builtin type 'tuple' (line 579)
    tuple_590098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 579)
    # Adding element type (line 579)
    # Getting the type of 'vv' (line 579)
    vv_590099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 23), 'vv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 23), tuple_590098, vv_590099)
    # Adding element type (line 579)
    # Getting the type of 'i' (line 579)
    i_590100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 27), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 23), tuple_590098, i_590100)
    
    # Storing an element on a container (line 579)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 16), result_590097, (tuple_590098, median_call_result_590096))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 575)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'statistic' (line 580)
    statistic_590101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 9), 'statistic')
    str_590102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 22), 'str', 'min')
    # Applying the binary operator '==' (line 580)
    result_eq_590103 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 9), '==', statistic_590101, str_590102)
    
    # Testing the type of an if condition (line 580)
    if_condition_590104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 580, 9), result_eq_590103)
    # Assigning a type to the variable 'if_condition_590104' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 9), 'if_condition_590104', if_condition_590104)
    # SSA begins for if statement (line 580)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fill(...): (line 581)
    # Processing the call arguments (line 581)
    # Getting the type of 'np' (line 581)
    np_590107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 20), 'np', False)
    # Obtaining the member 'nan' of a type (line 581)
    nan_590108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 20), np_590107, 'nan')
    # Processing the call keyword arguments (line 581)
    kwargs_590109 = {}
    # Getting the type of 'result' (line 581)
    result_590105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'result', False)
    # Obtaining the member 'fill' of a type (line 581)
    fill_590106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), result_590105, 'fill')
    # Calling fill(args, kwargs) (line 581)
    fill_call_result_590110 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), fill_590106, *[nan_590108], **kwargs_590109)
    
    
    
    # Call to unique(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'binnumbers' (line 582)
    binnumbers_590113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 27), 'binnumbers', False)
    # Processing the call keyword arguments (line 582)
    kwargs_590114 = {}
    # Getting the type of 'np' (line 582)
    np_590111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 17), 'np', False)
    # Obtaining the member 'unique' of a type (line 582)
    unique_590112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 17), np_590111, 'unique')
    # Calling unique(args, kwargs) (line 582)
    unique_call_result_590115 = invoke(stypy.reporting.localization.Localization(__file__, 582, 17), unique_590112, *[binnumbers_590113], **kwargs_590114)
    
    # Testing the type of a for loop iterable (line 582)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 582, 8), unique_call_result_590115)
    # Getting the type of the for loop variable (line 582)
    for_loop_var_590116 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 582, 8), unique_call_result_590115)
    # Assigning a type to the variable 'i' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'i', for_loop_var_590116)
    # SSA begins for a for statement (line 582)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to xrange(...): (line 583)
    # Processing the call arguments (line 583)
    # Getting the type of 'Vdim' (line 583)
    Vdim_590118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 29), 'Vdim', False)
    # Processing the call keyword arguments (line 583)
    kwargs_590119 = {}
    # Getting the type of 'xrange' (line 583)
    xrange_590117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 22), 'xrange', False)
    # Calling xrange(args, kwargs) (line 583)
    xrange_call_result_590120 = invoke(stypy.reporting.localization.Localization(__file__, 583, 22), xrange_590117, *[Vdim_590118], **kwargs_590119)
    
    # Testing the type of a for loop iterable (line 583)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 583, 12), xrange_call_result_590120)
    # Getting the type of the for loop variable (line 583)
    for_loop_var_590121 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 583, 12), xrange_call_result_590120)
    # Assigning a type to the variable 'vv' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'vv', for_loop_var_590121)
    # SSA begins for a for statement (line 583)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 584):
    
    # Assigning a Call to a Subscript (line 584):
    
    # Call to min(...): (line 584)
    # Processing the call arguments (line 584)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 584)
    tuple_590124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 584)
    # Adding element type (line 584)
    # Getting the type of 'vv' (line 584)
    vv_590125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 46), 'vv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 46), tuple_590124, vv_590125)
    # Adding element type (line 584)
    
    # Getting the type of 'binnumbers' (line 584)
    binnumbers_590126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 50), 'binnumbers', False)
    # Getting the type of 'i' (line 584)
    i_590127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 64), 'i', False)
    # Applying the binary operator '==' (line 584)
    result_eq_590128 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 50), '==', binnumbers_590126, i_590127)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 46), tuple_590124, result_eq_590128)
    
    # Getting the type of 'values' (line 584)
    values_590129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 39), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 584)
    getitem___590130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 39), values_590129, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 584)
    subscript_call_result_590131 = invoke(stypy.reporting.localization.Localization(__file__, 584, 39), getitem___590130, tuple_590124)
    
    # Processing the call keyword arguments (line 584)
    kwargs_590132 = {}
    # Getting the type of 'np' (line 584)
    np_590122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 32), 'np', False)
    # Obtaining the member 'min' of a type (line 584)
    min_590123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 32), np_590122, 'min')
    # Calling min(args, kwargs) (line 584)
    min_call_result_590133 = invoke(stypy.reporting.localization.Localization(__file__, 584, 32), min_590123, *[subscript_call_result_590131], **kwargs_590132)
    
    # Getting the type of 'result' (line 584)
    result_590134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'result')
    
    # Obtaining an instance of the builtin type 'tuple' (line 584)
    tuple_590135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 584)
    # Adding element type (line 584)
    # Getting the type of 'vv' (line 584)
    vv_590136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 23), 'vv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 23), tuple_590135, vv_590136)
    # Adding element type (line 584)
    # Getting the type of 'i' (line 584)
    i_590137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 27), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 23), tuple_590135, i_590137)
    
    # Storing an element on a container (line 584)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 16), result_590134, (tuple_590135, min_call_result_590133))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 580)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'statistic' (line 585)
    statistic_590138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 9), 'statistic')
    str_590139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 22), 'str', 'max')
    # Applying the binary operator '==' (line 585)
    result_eq_590140 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 9), '==', statistic_590138, str_590139)
    
    # Testing the type of an if condition (line 585)
    if_condition_590141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 9), result_eq_590140)
    # Assigning a type to the variable 'if_condition_590141' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 9), 'if_condition_590141', if_condition_590141)
    # SSA begins for if statement (line 585)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fill(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'np' (line 586)
    np_590144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'np', False)
    # Obtaining the member 'nan' of a type (line 586)
    nan_590145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 20), np_590144, 'nan')
    # Processing the call keyword arguments (line 586)
    kwargs_590146 = {}
    # Getting the type of 'result' (line 586)
    result_590142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'result', False)
    # Obtaining the member 'fill' of a type (line 586)
    fill_590143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 8), result_590142, 'fill')
    # Calling fill(args, kwargs) (line 586)
    fill_call_result_590147 = invoke(stypy.reporting.localization.Localization(__file__, 586, 8), fill_590143, *[nan_590145], **kwargs_590146)
    
    
    
    # Call to unique(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'binnumbers' (line 587)
    binnumbers_590150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 27), 'binnumbers', False)
    # Processing the call keyword arguments (line 587)
    kwargs_590151 = {}
    # Getting the type of 'np' (line 587)
    np_590148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 17), 'np', False)
    # Obtaining the member 'unique' of a type (line 587)
    unique_590149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 17), np_590148, 'unique')
    # Calling unique(args, kwargs) (line 587)
    unique_call_result_590152 = invoke(stypy.reporting.localization.Localization(__file__, 587, 17), unique_590149, *[binnumbers_590150], **kwargs_590151)
    
    # Testing the type of a for loop iterable (line 587)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 587, 8), unique_call_result_590152)
    # Getting the type of the for loop variable (line 587)
    for_loop_var_590153 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 587, 8), unique_call_result_590152)
    # Assigning a type to the variable 'i' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'i', for_loop_var_590153)
    # SSA begins for a for statement (line 587)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to xrange(...): (line 588)
    # Processing the call arguments (line 588)
    # Getting the type of 'Vdim' (line 588)
    Vdim_590155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 29), 'Vdim', False)
    # Processing the call keyword arguments (line 588)
    kwargs_590156 = {}
    # Getting the type of 'xrange' (line 588)
    xrange_590154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 22), 'xrange', False)
    # Calling xrange(args, kwargs) (line 588)
    xrange_call_result_590157 = invoke(stypy.reporting.localization.Localization(__file__, 588, 22), xrange_590154, *[Vdim_590155], **kwargs_590156)
    
    # Testing the type of a for loop iterable (line 588)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 588, 12), xrange_call_result_590157)
    # Getting the type of the for loop variable (line 588)
    for_loop_var_590158 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 588, 12), xrange_call_result_590157)
    # Assigning a type to the variable 'vv' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'vv', for_loop_var_590158)
    # SSA begins for a for statement (line 588)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 589):
    
    # Assigning a Call to a Subscript (line 589):
    
    # Call to max(...): (line 589)
    # Processing the call arguments (line 589)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 589)
    tuple_590161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 589)
    # Adding element type (line 589)
    # Getting the type of 'vv' (line 589)
    vv_590162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 46), 'vv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 46), tuple_590161, vv_590162)
    # Adding element type (line 589)
    
    # Getting the type of 'binnumbers' (line 589)
    binnumbers_590163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 50), 'binnumbers', False)
    # Getting the type of 'i' (line 589)
    i_590164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 64), 'i', False)
    # Applying the binary operator '==' (line 589)
    result_eq_590165 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 50), '==', binnumbers_590163, i_590164)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 46), tuple_590161, result_eq_590165)
    
    # Getting the type of 'values' (line 589)
    values_590166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 39), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 589)
    getitem___590167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 39), values_590166, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 589)
    subscript_call_result_590168 = invoke(stypy.reporting.localization.Localization(__file__, 589, 39), getitem___590167, tuple_590161)
    
    # Processing the call keyword arguments (line 589)
    kwargs_590169 = {}
    # Getting the type of 'np' (line 589)
    np_590159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 32), 'np', False)
    # Obtaining the member 'max' of a type (line 589)
    max_590160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 32), np_590159, 'max')
    # Calling max(args, kwargs) (line 589)
    max_call_result_590170 = invoke(stypy.reporting.localization.Localization(__file__, 589, 32), max_590160, *[subscript_call_result_590168], **kwargs_590169)
    
    # Getting the type of 'result' (line 589)
    result_590171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'result')
    
    # Obtaining an instance of the builtin type 'tuple' (line 589)
    tuple_590172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 589)
    # Adding element type (line 589)
    # Getting the type of 'vv' (line 589)
    vv_590173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 23), 'vv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 23), tuple_590172, vv_590173)
    # Adding element type (line 589)
    # Getting the type of 'i' (line 589)
    i_590174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 27), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 23), tuple_590172, i_590174)
    
    # Storing an element on a container (line 589)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 16), result_590171, (tuple_590172, max_call_result_590170))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 585)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to callable(...): (line 590)
    # Processing the call arguments (line 590)
    # Getting the type of 'statistic' (line 590)
    statistic_590176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 18), 'statistic', False)
    # Processing the call keyword arguments (line 590)
    kwargs_590177 = {}
    # Getting the type of 'callable' (line 590)
    callable_590175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 9), 'callable', False)
    # Calling callable(args, kwargs) (line 590)
    callable_call_result_590178 = invoke(stypy.reporting.localization.Localization(__file__, 590, 9), callable_590175, *[statistic_590176], **kwargs_590177)
    
    # Testing the type of an if condition (line 590)
    if_condition_590179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 590, 9), callable_call_result_590178)
    # Assigning a type to the variable 'if_condition_590179' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 9), 'if_condition_590179', if_condition_590179)
    # SSA begins for if statement (line 590)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to errstate(...): (line 591)
    # Processing the call keyword arguments (line 591)
    str_590182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 33), 'str', 'ignore')
    keyword_590183 = str_590182
    kwargs_590184 = {'invalid': keyword_590183}
    # Getting the type of 'np' (line 591)
    np_590180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 13), 'np', False)
    # Obtaining the member 'errstate' of a type (line 591)
    errstate_590181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 13), np_590180, 'errstate')
    # Calling errstate(args, kwargs) (line 591)
    errstate_call_result_590185 = invoke(stypy.reporting.localization.Localization(__file__, 591, 13), errstate_590181, *[], **kwargs_590184)
    
    with_590186 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 591, 13), errstate_call_result_590185, 'with parameter', '__enter__', '__exit__')

    if with_590186:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 591)
        enter___590187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 13), errstate_call_result_590185, '__enter__')
        with_enter_590188 = invoke(stypy.reporting.localization.Localization(__file__, 591, 13), enter___590187)
        
        # Call to suppress_warnings(...): (line 591)
        # Processing the call keyword arguments (line 591)
        kwargs_590190 = {}
        # Getting the type of 'suppress_warnings' (line 591)
        suppress_warnings_590189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 44), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 591)
        suppress_warnings_call_result_590191 = invoke(stypy.reporting.localization.Localization(__file__, 591, 44), suppress_warnings_590189, *[], **kwargs_590190)
        
        with_590192 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 591, 44), suppress_warnings_call_result_590191, 'with parameter', '__enter__', '__exit__')

        if with_590192:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 591)
            enter___590193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 44), suppress_warnings_call_result_590191, '__enter__')
            with_enter_590194 = invoke(stypy.reporting.localization.Localization(__file__, 591, 44), enter___590193)
            # Assigning a type to the variable 'sup' (line 591)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 44), 'sup', with_enter_590194)
            
            # Call to filter(...): (line 592)
            # Processing the call arguments (line 592)
            # Getting the type of 'RuntimeWarning' (line 592)
            RuntimeWarning_590197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 23), 'RuntimeWarning', False)
            # Processing the call keyword arguments (line 592)
            kwargs_590198 = {}
            # Getting the type of 'sup' (line 592)
            sup_590195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 592)
            filter_590196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 12), sup_590195, 'filter')
            # Calling filter(args, kwargs) (line 592)
            filter_call_result_590199 = invoke(stypy.reporting.localization.Localization(__file__, 592, 12), filter_590196, *[RuntimeWarning_590197], **kwargs_590198)
            
            
            
            # SSA begins for try-except statement (line 593)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 594):
            
            # Assigning a Call to a Name (line 594):
            
            # Call to statistic(...): (line 594)
            # Processing the call arguments (line 594)
            
            # Obtaining an instance of the builtin type 'list' (line 594)
            list_590201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 594)
            
            # Processing the call keyword arguments (line 594)
            kwargs_590202 = {}
            # Getting the type of 'statistic' (line 594)
            statistic_590200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 23), 'statistic', False)
            # Calling statistic(args, kwargs) (line 594)
            statistic_call_result_590203 = invoke(stypy.reporting.localization.Localization(__file__, 594, 23), statistic_590200, *[list_590201], **kwargs_590202)
            
            # Assigning a type to the variable 'null' (line 594)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 16), 'null', statistic_call_result_590203)
            # SSA branch for the except part of a try statement (line 593)
            # SSA branch for the except '<any exception>' branch of a try statement (line 593)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Attribute to a Name (line 596):
            
            # Assigning a Attribute to a Name (line 596):
            # Getting the type of 'np' (line 596)
            np_590204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 23), 'np')
            # Obtaining the member 'nan' of a type (line 596)
            nan_590205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 23), np_590204, 'nan')
            # Assigning a type to the variable 'null' (line 596)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'null', nan_590205)
            # SSA join for try-except statement (line 593)
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 591)
            exit___590206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 44), suppress_warnings_call_result_590191, '__exit__')
            with_exit_590207 = invoke(stypy.reporting.localization.Localization(__file__, 591, 44), exit___590206, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 591)
        exit___590208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 13), errstate_call_result_590185, '__exit__')
        with_exit_590209 = invoke(stypy.reporting.localization.Localization(__file__, 591, 13), exit___590208, None, None, None)

    
    # Call to fill(...): (line 597)
    # Processing the call arguments (line 597)
    # Getting the type of 'null' (line 597)
    null_590212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 20), 'null', False)
    # Processing the call keyword arguments (line 597)
    kwargs_590213 = {}
    # Getting the type of 'result' (line 597)
    result_590210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'result', False)
    # Obtaining the member 'fill' of a type (line 597)
    fill_590211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 8), result_590210, 'fill')
    # Calling fill(args, kwargs) (line 597)
    fill_call_result_590214 = invoke(stypy.reporting.localization.Localization(__file__, 597, 8), fill_590211, *[null_590212], **kwargs_590213)
    
    
    
    # Call to unique(...): (line 598)
    # Processing the call arguments (line 598)
    # Getting the type of 'binnumbers' (line 598)
    binnumbers_590217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 27), 'binnumbers', False)
    # Processing the call keyword arguments (line 598)
    kwargs_590218 = {}
    # Getting the type of 'np' (line 598)
    np_590215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 17), 'np', False)
    # Obtaining the member 'unique' of a type (line 598)
    unique_590216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 17), np_590215, 'unique')
    # Calling unique(args, kwargs) (line 598)
    unique_call_result_590219 = invoke(stypy.reporting.localization.Localization(__file__, 598, 17), unique_590216, *[binnumbers_590217], **kwargs_590218)
    
    # Testing the type of a for loop iterable (line 598)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 598, 8), unique_call_result_590219)
    # Getting the type of the for loop variable (line 598)
    for_loop_var_590220 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 598, 8), unique_call_result_590219)
    # Assigning a type to the variable 'i' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'i', for_loop_var_590220)
    # SSA begins for a for statement (line 598)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to xrange(...): (line 599)
    # Processing the call arguments (line 599)
    # Getting the type of 'Vdim' (line 599)
    Vdim_590222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 29), 'Vdim', False)
    # Processing the call keyword arguments (line 599)
    kwargs_590223 = {}
    # Getting the type of 'xrange' (line 599)
    xrange_590221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 22), 'xrange', False)
    # Calling xrange(args, kwargs) (line 599)
    xrange_call_result_590224 = invoke(stypy.reporting.localization.Localization(__file__, 599, 22), xrange_590221, *[Vdim_590222], **kwargs_590223)
    
    # Testing the type of a for loop iterable (line 599)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 599, 12), xrange_call_result_590224)
    # Getting the type of the for loop variable (line 599)
    for_loop_var_590225 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 599, 12), xrange_call_result_590224)
    # Assigning a type to the variable 'vv' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 'vv', for_loop_var_590225)
    # SSA begins for a for statement (line 599)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 600):
    
    # Assigning a Call to a Subscript (line 600):
    
    # Call to statistic(...): (line 600)
    # Processing the call arguments (line 600)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 600)
    tuple_590227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 600)
    # Adding element type (line 600)
    # Getting the type of 'vv' (line 600)
    vv_590228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 49), 'vv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 49), tuple_590227, vv_590228)
    # Adding element type (line 600)
    
    # Getting the type of 'binnumbers' (line 600)
    binnumbers_590229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 53), 'binnumbers', False)
    # Getting the type of 'i' (line 600)
    i_590230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 67), 'i', False)
    # Applying the binary operator '==' (line 600)
    result_eq_590231 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 53), '==', binnumbers_590229, i_590230)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 49), tuple_590227, result_eq_590231)
    
    # Getting the type of 'values' (line 600)
    values_590232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 42), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 600)
    getitem___590233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 42), values_590232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 600)
    subscript_call_result_590234 = invoke(stypy.reporting.localization.Localization(__file__, 600, 42), getitem___590233, tuple_590227)
    
    # Processing the call keyword arguments (line 600)
    kwargs_590235 = {}
    # Getting the type of 'statistic' (line 600)
    statistic_590226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 32), 'statistic', False)
    # Calling statistic(args, kwargs) (line 600)
    statistic_call_result_590236 = invoke(stypy.reporting.localization.Localization(__file__, 600, 32), statistic_590226, *[subscript_call_result_590234], **kwargs_590235)
    
    # Getting the type of 'result' (line 600)
    result_590237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'result')
    
    # Obtaining an instance of the builtin type 'tuple' (line 600)
    tuple_590238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 600)
    # Adding element type (line 600)
    # Getting the type of 'vv' (line 600)
    vv_590239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 23), 'vv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 23), tuple_590238, vv_590239)
    # Adding element type (line 600)
    # Getting the type of 'i' (line 600)
    i_590240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 27), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 23), tuple_590238, i_590240)
    
    # Storing an element on a container (line 600)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 16), result_590237, (tuple_590238, statistic_call_result_590236))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 590)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 585)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 580)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 575)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 569)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 564)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 555)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 548)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 603):
    
    # Assigning a Call to a Name (line 603):
    
    # Call to reshape(...): (line 603)
    # Processing the call arguments (line 603)
    
    # Call to append(...): (line 603)
    # Processing the call arguments (line 603)
    # Getting the type of 'Vdim' (line 603)
    Vdim_590245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 38), 'Vdim', False)
    # Getting the type of 'nbin' (line 603)
    nbin_590246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 44), 'nbin', False)
    # Processing the call keyword arguments (line 603)
    kwargs_590247 = {}
    # Getting the type of 'np' (line 603)
    np_590243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 28), 'np', False)
    # Obtaining the member 'append' of a type (line 603)
    append_590244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 28), np_590243, 'append')
    # Calling append(args, kwargs) (line 603)
    append_call_result_590248 = invoke(stypy.reporting.localization.Localization(__file__, 603, 28), append_590244, *[Vdim_590245, nbin_590246], **kwargs_590247)
    
    # Processing the call keyword arguments (line 603)
    kwargs_590249 = {}
    # Getting the type of 'result' (line 603)
    result_590241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 13), 'result', False)
    # Obtaining the member 'reshape' of a type (line 603)
    reshape_590242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 13), result_590241, 'reshape')
    # Calling reshape(args, kwargs) (line 603)
    reshape_call_result_590250 = invoke(stypy.reporting.localization.Localization(__file__, 603, 13), reshape_590242, *[append_call_result_590248], **kwargs_590249)
    
    # Assigning a type to the variable 'result' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'result', reshape_call_result_590250)
    
    # Assigning a BinOp to a Name (line 606):
    
    # Assigning a BinOp to a Name (line 606):
    
    # Obtaining an instance of the builtin type 'list' (line 606)
    list_590251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 606)
    # Adding element type (line 606)
    
    # Call to slice(...): (line 606)
    # Processing the call arguments (line 606)
    # Getting the type of 'None' (line 606)
    None_590253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 18), 'None', False)
    # Processing the call keyword arguments (line 606)
    kwargs_590254 = {}
    # Getting the type of 'slice' (line 606)
    slice_590252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'slice', False)
    # Calling slice(args, kwargs) (line 606)
    slice_call_result_590255 = invoke(stypy.reporting.localization.Localization(__file__, 606, 12), slice_590252, *[None_590253], **kwargs_590254)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 11), list_590251, slice_call_result_590255)
    
    # Getting the type of 'Ndim' (line 606)
    Ndim_590256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 27), 'Ndim')
    
    # Obtaining an instance of the builtin type 'list' (line 606)
    list_590257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 606)
    # Adding element type (line 606)
    
    # Call to slice(...): (line 606)
    # Processing the call arguments (line 606)
    int_590259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 41), 'int')
    int_590260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 44), 'int')
    # Processing the call keyword arguments (line 606)
    kwargs_590261 = {}
    # Getting the type of 'slice' (line 606)
    slice_590258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 35), 'slice', False)
    # Calling slice(args, kwargs) (line 606)
    slice_call_result_590262 = invoke(stypy.reporting.localization.Localization(__file__, 606, 35), slice_590258, *[int_590259, int_590260], **kwargs_590261)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 34), list_590257, slice_call_result_590262)
    
    # Applying the binary operator '*' (line 606)
    result_mul_590263 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 27), '*', Ndim_590256, list_590257)
    
    # Applying the binary operator '+' (line 606)
    result_add_590264 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 11), '+', list_590251, result_mul_590263)
    
    # Assigning a type to the variable 'core' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'core', result_add_590264)
    
    # Assigning a Subscript to a Name (line 607):
    
    # Assigning a Subscript to a Name (line 607):
    
    # Obtaining the type of the subscript
    # Getting the type of 'core' (line 607)
    core_590265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 20), 'core')
    # Getting the type of 'result' (line 607)
    result_590266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 13), 'result')
    # Obtaining the member '__getitem__' of a type (line 607)
    getitem___590267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 13), result_590266, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 607)
    subscript_call_result_590268 = invoke(stypy.reporting.localization.Localization(__file__, 607, 13), getitem___590267, core_590265)
    
    # Assigning a type to the variable 'result' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'result', subscript_call_result_590268)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'expand_binnumbers' (line 610)
    expand_binnumbers_590269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 7), 'expand_binnumbers')
    
    # Getting the type of 'Ndim' (line 610)
    Ndim_590270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 29), 'Ndim')
    int_590271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 36), 'int')
    # Applying the binary operator '>' (line 610)
    result_gt_590272 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 29), '>', Ndim_590270, int_590271)
    
    # Applying the binary operator 'and' (line 610)
    result_and_keyword_590273 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 7), 'and', expand_binnumbers_590269, result_gt_590272)
    
    # Testing the type of an if condition (line 610)
    if_condition_590274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 610, 4), result_and_keyword_590273)
    # Assigning a type to the variable 'if_condition_590274' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'if_condition_590274', if_condition_590274)
    # SSA begins for if statement (line 610)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 611):
    
    # Assigning a Call to a Name (line 611):
    
    # Call to asarray(...): (line 611)
    # Processing the call arguments (line 611)
    
    # Call to unravel_index(...): (line 611)
    # Processing the call arguments (line 611)
    # Getting the type of 'binnumbers' (line 611)
    binnumbers_590279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 49), 'binnumbers', False)
    # Getting the type of 'nbin' (line 611)
    nbin_590280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 61), 'nbin', False)
    # Processing the call keyword arguments (line 611)
    kwargs_590281 = {}
    # Getting the type of 'np' (line 611)
    np_590277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 32), 'np', False)
    # Obtaining the member 'unravel_index' of a type (line 611)
    unravel_index_590278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 32), np_590277, 'unravel_index')
    # Calling unravel_index(args, kwargs) (line 611)
    unravel_index_call_result_590282 = invoke(stypy.reporting.localization.Localization(__file__, 611, 32), unravel_index_590278, *[binnumbers_590279, nbin_590280], **kwargs_590281)
    
    # Processing the call keyword arguments (line 611)
    kwargs_590283 = {}
    # Getting the type of 'np' (line 611)
    np_590275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 21), 'np', False)
    # Obtaining the member 'asarray' of a type (line 611)
    asarray_590276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 21), np_590275, 'asarray')
    # Calling asarray(args, kwargs) (line 611)
    asarray_call_result_590284 = invoke(stypy.reporting.localization.Localization(__file__, 611, 21), asarray_590276, *[unravel_index_call_result_590282], **kwargs_590283)
    
    # Assigning a type to the variable 'binnumbers' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'binnumbers', asarray_call_result_590284)
    # SSA join for if statement (line 610)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 613)
    # Processing the call arguments (line 613)
    
    
    # Obtaining the type of the subscript
    int_590287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 27), 'int')
    slice_590288 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 613, 14), int_590287, None, None)
    # Getting the type of 'result' (line 613)
    result_590289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 14), 'result', False)
    # Obtaining the member 'shape' of a type (line 613)
    shape_590290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 14), result_590289, 'shape')
    # Obtaining the member '__getitem__' of a type (line 613)
    getitem___590291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 14), shape_590290, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 613)
    subscript_call_result_590292 = invoke(stypy.reporting.localization.Localization(__file__, 613, 14), getitem___590291, slice_590288)
    
    # Getting the type of 'nbin' (line 613)
    nbin_590293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 34), 'nbin', False)
    int_590294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 41), 'int')
    # Applying the binary operator '-' (line 613)
    result_sub_590295 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 34), '-', nbin_590293, int_590294)
    
    # Applying the binary operator '!=' (line 613)
    result_ne_590296 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 14), '!=', subscript_call_result_590292, result_sub_590295)
    
    # Processing the call keyword arguments (line 613)
    kwargs_590297 = {}
    # Getting the type of 'np' (line 613)
    np_590285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 613)
    any_590286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 7), np_590285, 'any')
    # Calling any(args, kwargs) (line 613)
    any_call_result_590298 = invoke(stypy.reporting.localization.Localization(__file__, 613, 7), any_590286, *[result_ne_590296], **kwargs_590297)
    
    # Testing the type of an if condition (line 613)
    if_condition_590299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 613, 4), any_call_result_590298)
    # Assigning a type to the variable 'if_condition_590299' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'if_condition_590299', if_condition_590299)
    # SSA begins for if statement (line 613)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 614)
    # Processing the call arguments (line 614)
    str_590301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 27), 'str', 'Internal Shape Error')
    # Processing the call keyword arguments (line 614)
    kwargs_590302 = {}
    # Getting the type of 'RuntimeError' (line 614)
    RuntimeError_590300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 614)
    RuntimeError_call_result_590303 = invoke(stypy.reporting.localization.Localization(__file__, 614, 14), RuntimeError_590300, *[str_590301], **kwargs_590302)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 614, 8), RuntimeError_call_result_590303, 'raise parameter', BaseException)
    # SSA join for if statement (line 613)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 617):
    
    # Assigning a Call to a Name (line 617):
    
    # Call to reshape(...): (line 617)
    # Processing the call arguments (line 617)
    
    # Obtaining the type of the subscript
    int_590306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 41), 'int')
    slice_590307 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 617, 28), None, int_590306, None)
    # Getting the type of 'input_shape' (line 617)
    input_shape_590308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 28), 'input_shape', False)
    # Obtaining the member '__getitem__' of a type (line 617)
    getitem___590309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 28), input_shape_590308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 617)
    subscript_call_result_590310 = invoke(stypy.reporting.localization.Localization(__file__, 617, 28), getitem___590309, slice_590307)
    
    
    # Call to list(...): (line 617)
    # Processing the call arguments (line 617)
    # Getting the type of 'nbin' (line 617)
    nbin_590312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 52), 'nbin', False)
    int_590313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 57), 'int')
    # Applying the binary operator '-' (line 617)
    result_sub_590314 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 52), '-', nbin_590312, int_590313)
    
    # Processing the call keyword arguments (line 617)
    kwargs_590315 = {}
    # Getting the type of 'list' (line 617)
    list_590311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 47), 'list', False)
    # Calling list(args, kwargs) (line 617)
    list_call_result_590316 = invoke(stypy.reporting.localization.Localization(__file__, 617, 47), list_590311, *[result_sub_590314], **kwargs_590315)
    
    # Applying the binary operator '+' (line 617)
    result_add_590317 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 28), '+', subscript_call_result_590310, list_call_result_590316)
    
    # Processing the call keyword arguments (line 617)
    kwargs_590318 = {}
    # Getting the type of 'result' (line 617)
    result_590304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 13), 'result', False)
    # Obtaining the member 'reshape' of a type (line 617)
    reshape_590305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 13), result_590304, 'reshape')
    # Calling reshape(args, kwargs) (line 617)
    reshape_call_result_590319 = invoke(stypy.reporting.localization.Localization(__file__, 617, 13), reshape_590305, *[result_add_590317], **kwargs_590318)
    
    # Assigning a type to the variable 'result' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'result', reshape_call_result_590319)
    
    # Call to BinnedStatisticddResult(...): (line 619)
    # Processing the call arguments (line 619)
    # Getting the type of 'result' (line 619)
    result_590321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 35), 'result', False)
    # Getting the type of 'edges' (line 619)
    edges_590322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 43), 'edges', False)
    # Getting the type of 'binnumbers' (line 619)
    binnumbers_590323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 50), 'binnumbers', False)
    # Processing the call keyword arguments (line 619)
    kwargs_590324 = {}
    # Getting the type of 'BinnedStatisticddResult' (line 619)
    BinnedStatisticddResult_590320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 11), 'BinnedStatisticddResult', False)
    # Calling BinnedStatisticddResult(args, kwargs) (line 619)
    BinnedStatisticddResult_call_result_590325 = invoke(stypy.reporting.localization.Localization(__file__, 619, 11), BinnedStatisticddResult_590320, *[result_590321, edges_590322, binnumbers_590323], **kwargs_590324)
    
    # Assigning a type to the variable 'stypy_return_type' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'stypy_return_type', BinnedStatisticddResult_call_result_590325)
    
    # ################# End of 'binned_statistic_dd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'binned_statistic_dd' in the type store
    # Getting the type of 'stypy_return_type' (line 354)
    stypy_return_type_590326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_590326)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'binned_statistic_dd'
    return stypy_return_type_590326

# Assigning a type to the variable 'binned_statistic_dd' (line 354)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'binned_statistic_dd', binned_statistic_dd)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
