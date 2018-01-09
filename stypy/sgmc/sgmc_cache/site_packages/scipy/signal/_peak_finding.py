
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Functions for identifying peaks in signals.
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: 
8: from scipy._lib.six import xrange
9: from scipy.signal.wavelets import cwt, ricker
10: from scipy.stats import scoreatpercentile
11: 
12: 
13: __all__ = ['argrelmin', 'argrelmax', 'argrelextrema', 'find_peaks_cwt']
14: 
15: 
16: def _boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):
17:     '''
18:     Calculate the relative extrema of `data`.
19: 
20:     Relative extrema are calculated by finding locations where
21:     ``comparator(data[n], data[n+1:n+order+1])`` is True.
22: 
23:     Parameters
24:     ----------
25:     data : ndarray
26:         Array in which to find the relative extrema.
27:     comparator : callable
28:         Function to use to compare two data points.
29:         Should take two arrays as arguments.
30:     axis : int, optional
31:         Axis over which to select from `data`.  Default is 0.
32:     order : int, optional
33:         How many points on each side to use for the comparison
34:         to consider ``comparator(n,n+x)`` to be True.
35:     mode : str, optional
36:         How the edges of the vector are treated.  'wrap' (wrap around) or
37:         'clip' (treat overflow as the same as the last (or first) element).
38:         Default 'clip'.  See numpy.take
39: 
40:     Returns
41:     -------
42:     extrema : ndarray
43:         Boolean array of the same shape as `data` that is True at an extrema,
44:         False otherwise.
45: 
46:     See also
47:     --------
48:     argrelmax, argrelmin
49: 
50:     Examples
51:     --------
52:     >>> testdata = np.array([1,2,3,2,1])
53:     >>> _boolrelextrema(testdata, np.greater, axis=0)
54:     array([False, False,  True, False, False], dtype=bool)
55: 
56:     '''
57:     if((int(order) != order) or (order < 1)):
58:         raise ValueError('Order must be an int >= 1')
59: 
60:     datalen = data.shape[axis]
61:     locs = np.arange(0, datalen)
62: 
63:     results = np.ones(data.shape, dtype=bool)
64:     main = data.take(locs, axis=axis, mode=mode)
65:     for shift in xrange(1, order + 1):
66:         plus = data.take(locs + shift, axis=axis, mode=mode)
67:         minus = data.take(locs - shift, axis=axis, mode=mode)
68:         results &= comparator(main, plus)
69:         results &= comparator(main, minus)
70:         if(~results.any()):
71:             return results
72:     return results
73: 
74: 
75: def argrelmin(data, axis=0, order=1, mode='clip'):
76:     '''
77:     Calculate the relative minima of `data`.
78: 
79:     Parameters
80:     ----------
81:     data : ndarray
82:         Array in which to find the relative minima.
83:     axis : int, optional
84:         Axis over which to select from `data`.  Default is 0.
85:     order : int, optional
86:         How many points on each side to use for the comparison
87:         to consider ``comparator(n, n+x)`` to be True.
88:     mode : str, optional
89:         How the edges of the vector are treated.
90:         Available options are 'wrap' (wrap around) or 'clip' (treat overflow
91:         as the same as the last (or first) element).
92:         Default 'clip'. See numpy.take
93: 
94:     Returns
95:     -------
96:     extrema : tuple of ndarrays
97:         Indices of the minima in arrays of integers.  ``extrema[k]`` is
98:         the array of indices of axis `k` of `data`.  Note that the
99:         return value is a tuple even when `data` is one-dimensional.
100: 
101:     See Also
102:     --------
103:     argrelextrema, argrelmax
104: 
105:     Notes
106:     -----
107:     This function uses `argrelextrema` with np.less as comparator.
108: 
109:     .. versionadded:: 0.11.0
110: 
111:     Examples
112:     --------
113:     >>> from scipy.signal import argrelmin
114:     >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
115:     >>> argrelmin(x)
116:     (array([1, 5]),)
117:     >>> y = np.array([[1, 2, 1, 2],
118:     ...               [2, 2, 0, 0],
119:     ...               [5, 3, 4, 4]])
120:     ...
121:     >>> argrelmin(y, axis=1)
122:     (array([0, 2]), array([2, 1]))
123: 
124:     '''
125:     return argrelextrema(data, np.less, axis, order, mode)
126: 
127: 
128: def argrelmax(data, axis=0, order=1, mode='clip'):
129:     '''
130:     Calculate the relative maxima of `data`.
131: 
132:     Parameters
133:     ----------
134:     data : ndarray
135:         Array in which to find the relative maxima.
136:     axis : int, optional
137:         Axis over which to select from `data`.  Default is 0.
138:     order : int, optional
139:         How many points on each side to use for the comparison
140:         to consider ``comparator(n, n+x)`` to be True.
141:     mode : str, optional
142:         How the edges of the vector are treated.
143:         Available options are 'wrap' (wrap around) or 'clip' (treat overflow
144:         as the same as the last (or first) element).
145:         Default 'clip'.  See `numpy.take`.
146: 
147:     Returns
148:     -------
149:     extrema : tuple of ndarrays
150:         Indices of the maxima in arrays of integers.  ``extrema[k]`` is
151:         the array of indices of axis `k` of `data`.  Note that the
152:         return value is a tuple even when `data` is one-dimensional.
153: 
154:     See Also
155:     --------
156:     argrelextrema, argrelmin
157: 
158:     Notes
159:     -----
160:     This function uses `argrelextrema` with np.greater as comparator.
161: 
162:     .. versionadded:: 0.11.0
163: 
164:     Examples
165:     --------
166:     >>> from scipy.signal import argrelmax
167:     >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
168:     >>> argrelmax(x)
169:     (array([3, 6]),)
170:     >>> y = np.array([[1, 2, 1, 2],
171:     ...               [2, 2, 0, 0],
172:     ...               [5, 3, 4, 4]])
173:     ...
174:     >>> argrelmax(y, axis=1)
175:     (array([0]), array([1]))
176:     '''
177:     return argrelextrema(data, np.greater, axis, order, mode)
178: 
179: 
180: def argrelextrema(data, comparator, axis=0, order=1, mode='clip'):
181:     '''
182:     Calculate the relative extrema of `data`.
183: 
184:     Parameters
185:     ----------
186:     data : ndarray
187:         Array in which to find the relative extrema.
188:     comparator : callable
189:         Function to use to compare two data points.
190:         Should take two arrays as arguments.
191:     axis : int, optional
192:         Axis over which to select from `data`.  Default is 0.
193:     order : int, optional
194:         How many points on each side to use for the comparison
195:         to consider ``comparator(n, n+x)`` to be True.
196:     mode : str, optional
197:         How the edges of the vector are treated.  'wrap' (wrap around) or
198:         'clip' (treat overflow as the same as the last (or first) element).
199:         Default is 'clip'.  See `numpy.take`.
200: 
201:     Returns
202:     -------
203:     extrema : tuple of ndarrays
204:         Indices of the maxima in arrays of integers.  ``extrema[k]`` is
205:         the array of indices of axis `k` of `data`.  Note that the
206:         return value is a tuple even when `data` is one-dimensional.
207: 
208:     See Also
209:     --------
210:     argrelmin, argrelmax
211: 
212:     Notes
213:     -----
214: 
215:     .. versionadded:: 0.11.0
216: 
217:     Examples
218:     --------
219:     >>> from scipy.signal import argrelextrema
220:     >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
221:     >>> argrelextrema(x, np.greater)
222:     (array([3, 6]),)
223:     >>> y = np.array([[1, 2, 1, 2],
224:     ...               [2, 2, 0, 0],
225:     ...               [5, 3, 4, 4]])
226:     ...
227:     >>> argrelextrema(y, np.less, axis=1)
228:     (array([0, 2]), array([2, 1]))
229: 
230:     '''
231:     results = _boolrelextrema(data, comparator,
232:                               axis, order, mode)
233:     return np.where(results)
234: 
235: 
236: def _identify_ridge_lines(matr, max_distances, gap_thresh):
237:     '''
238:     Identify ridges in the 2-D matrix.
239: 
240:     Expect that the width of the wavelet feature increases with increasing row
241:     number.
242: 
243:     Parameters
244:     ----------
245:     matr : 2-D ndarray
246:         Matrix in which to identify ridge lines.
247:     max_distances : 1-D sequence
248:         At each row, a ridge line is only connected
249:         if the relative max at row[n] is within
250:         `max_distances`[n] from the relative max at row[n+1].
251:     gap_thresh : int
252:         If a relative maximum is not found within `max_distances`,
253:         there will be a gap. A ridge line is discontinued if
254:         there are more than `gap_thresh` points without connecting
255:         a new relative maximum.
256: 
257:     Returns
258:     -------
259:     ridge_lines : tuple
260:         Tuple of 2 1-D sequences. `ridge_lines`[ii][0] are the rows of the
261:         ii-th ridge-line, `ridge_lines`[ii][1] are the columns. Empty if none
262:         found.  Each ridge-line will be sorted by row (increasing), but the
263:         order of the ridge lines is not specified.
264: 
265:     References
266:     ----------
267:     Bioinformatics (2006) 22 (17): 2059-2065.
268:     :doi:`10.1093/bioinformatics/btl355`
269:     http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
270: 
271:     Examples
272:     --------
273:     >>> data = np.random.rand(5,5)
274:     >>> ridge_lines = _identify_ridge_lines(data, 1, 1)
275: 
276:     Notes
277:     -----
278:     This function is intended to be used in conjunction with `cwt`
279:     as part of `find_peaks_cwt`.
280: 
281:     '''
282:     if(len(max_distances) < matr.shape[0]):
283:         raise ValueError('Max_distances must have at least as many rows '
284:                          'as matr')
285: 
286:     all_max_cols = _boolrelextrema(matr, np.greater, axis=1, order=1)
287:     # Highest row for which there are any relative maxima
288:     has_relmax = np.where(all_max_cols.any(axis=1))[0]
289:     if(len(has_relmax) == 0):
290:         return []
291:     start_row = has_relmax[-1]
292:     # Each ridge line is a 3-tuple:
293:     # rows, cols,Gap number
294:     ridge_lines = [[[start_row],
295:                    [col],
296:                    0] for col in np.where(all_max_cols[start_row])[0]]
297:     final_lines = []
298:     rows = np.arange(start_row - 1, -1, -1)
299:     cols = np.arange(0, matr.shape[1])
300:     for row in rows:
301:         this_max_cols = cols[all_max_cols[row]]
302: 
303:         # Increment gap number of each line,
304:         # set it to zero later if appropriate
305:         for line in ridge_lines:
306:             line[2] += 1
307: 
308:         # XXX These should always be all_max_cols[row]
309:         # But the order might be different. Might be an efficiency gain
310:         # to make sure the order is the same and avoid this iteration
311:         prev_ridge_cols = np.array([line[1][-1] for line in ridge_lines])
312:         # Look through every relative maximum found at current row
313:         # Attempt to connect them with existing ridge lines.
314:         for ind, col in enumerate(this_max_cols):
315:             # If there is a previous ridge line within
316:             # the max_distance to connect to, do so.
317:             # Otherwise start a new one.
318:             line = None
319:             if(len(prev_ridge_cols) > 0):
320:                 diffs = np.abs(col - prev_ridge_cols)
321:                 closest = np.argmin(diffs)
322:                 if diffs[closest] <= max_distances[row]:
323:                     line = ridge_lines[closest]
324:             if(line is not None):
325:                 # Found a point close enough, extend current ridge line
326:                 line[1].append(col)
327:                 line[0].append(row)
328:                 line[2] = 0
329:             else:
330:                 new_line = [[row],
331:                             [col],
332:                             0]
333:                 ridge_lines.append(new_line)
334: 
335:         # Remove the ridge lines with gap_number too high
336:         # XXX Modifying a list while iterating over it.
337:         # Should be safe, since we iterate backwards, but
338:         # still tacky.
339:         for ind in xrange(len(ridge_lines) - 1, -1, -1):
340:             line = ridge_lines[ind]
341:             if line[2] > gap_thresh:
342:                 final_lines.append(line)
343:                 del ridge_lines[ind]
344: 
345:     out_lines = []
346:     for line in (final_lines + ridge_lines):
347:         sortargs = np.array(np.argsort(line[0]))
348:         rows, cols = np.zeros_like(sortargs), np.zeros_like(sortargs)
349:         rows[sortargs] = line[0]
350:         cols[sortargs] = line[1]
351:         out_lines.append([rows, cols])
352: 
353:     return out_lines
354: 
355: 
356: def _filter_ridge_lines(cwt, ridge_lines, window_size=None, min_length=None,
357:                         min_snr=1, noise_perc=10):
358:     '''
359:     Filter ridge lines according to prescribed criteria. Intended
360:     to be used for finding relative maxima.
361: 
362:     Parameters
363:     ----------
364:     cwt : 2-D ndarray
365:         Continuous wavelet transform from which the `ridge_lines` were defined.
366:     ridge_lines : 1-D sequence
367:         Each element should contain 2 sequences, the rows and columns
368:         of the ridge line (respectively).
369:     window_size : int, optional
370:         Size of window to use to calculate noise floor.
371:         Default is ``cwt.shape[1] / 20``.
372:     min_length : int, optional
373:         Minimum length a ridge line needs to be acceptable.
374:         Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.
375:     min_snr : float, optional
376:         Minimum SNR ratio. Default 1. The signal is the value of
377:         the cwt matrix at the shortest length scale (``cwt[0, loc]``), the
378:         noise is the `noise_perc`th percentile of datapoints contained within a
379:         window of `window_size` around ``cwt[0, loc]``.
380:     noise_perc : float, optional
381:         When calculating the noise floor, percentile of data points
382:         examined below which to consider noise. Calculated using
383:         scipy.stats.scoreatpercentile.
384: 
385:     References
386:     ----------
387:     Bioinformatics (2006) 22 (17): 2059-2065. :doi:`10.1093/bioinformatics/btl355`
388:     http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
389: 
390:     '''
391:     num_points = cwt.shape[1]
392:     if min_length is None:
393:         min_length = np.ceil(cwt.shape[0] / 4)
394:     if window_size is None:
395:         window_size = np.ceil(num_points / 20)
396: 
397:     window_size = int(window_size)
398:     hf_window, odd = divmod(window_size, 2)
399: 
400:     # Filter based on SNR
401:     row_one = cwt[0, :]
402:     noises = np.zeros_like(row_one)
403:     for ind, val in enumerate(row_one):
404:         window_start = max(ind - hf_window, 0)
405:         window_end = min(ind + hf_window + odd, num_points)
406:         noises[ind] = scoreatpercentile(row_one[window_start:window_end],
407:                                         per=noise_perc)
408: 
409:     def filt_func(line):
410:         if len(line[0]) < min_length:
411:             return False
412:         snr = abs(cwt[line[0][0], line[1][0]] / noises[line[1][0]])
413:         if snr < min_snr:
414:             return False
415:         return True
416: 
417:     return list(filter(filt_func, ridge_lines))
418: 
419: 
420: def find_peaks_cwt(vector, widths, wavelet=None, max_distances=None,
421:                    gap_thresh=None, min_length=None, min_snr=1, noise_perc=10):
422:     '''
423:     Attempt to find the peaks in a 1-D array.
424: 
425:     The general approach is to smooth `vector` by convolving it with
426:     `wavelet(width)` for each width in `widths`. Relative maxima which
427:     appear at enough length scales, and with sufficiently high SNR, are
428:     accepted.
429: 
430:     Parameters
431:     ----------
432:     vector : ndarray
433:         1-D array in which to find the peaks.
434:     widths : sequence
435:         1-D array of widths to use for calculating the CWT matrix. In general,
436:         this range should cover the expected width of peaks of interest.
437:     wavelet : callable, optional
438:         Should take two parameters and return a 1-D array to convolve
439:         with `vector`. The first parameter determines the number of points 
440:         of the returned wavelet array, the second parameter is the scale 
441:         (`width`) of the wavelet. Should be normalized and symmetric.
442:         Default is the ricker wavelet.
443:     max_distances : ndarray, optional
444:         At each row, a ridge line is only connected if the relative max at
445:         row[n] is within ``max_distances[n]`` from the relative max at
446:         ``row[n+1]``.  Default value is ``widths/4``.
447:     gap_thresh : float, optional
448:         If a relative maximum is not found within `max_distances`,
449:         there will be a gap. A ridge line is discontinued if there are more
450:         than `gap_thresh` points without connecting a new relative maximum.
451:         Default is the first value of the widths array i.e. widths[0].
452:     min_length : int, optional
453:         Minimum length a ridge line needs to be acceptable.
454:         Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.
455:     min_snr : float, optional
456:         Minimum SNR ratio. Default 1. The signal is the value of
457:         the cwt matrix at the shortest length scale (``cwt[0, loc]``), the
458:         noise is the `noise_perc`th percentile of datapoints contained within a
459:         window of `window_size` around ``cwt[0, loc]``.
460:     noise_perc : float, optional
461:         When calculating the noise floor, percentile of data points
462:         examined below which to consider noise. Calculated using
463:         `stats.scoreatpercentile`.  Default is 10.
464: 
465:     Returns
466:     -------
467:     peaks_indices : ndarray
468:         Indices of the locations in the `vector` where peaks were found.
469:         The list is sorted.
470: 
471:     See Also
472:     --------
473:     cwt
474: 
475:     Notes
476:     -----
477:     This approach was designed for finding sharp peaks among noisy data,
478:     however with proper parameter selection it should function well for
479:     different peak shapes.
480: 
481:     The algorithm is as follows:
482:      1. Perform a continuous wavelet transform on `vector`, for the supplied
483:         `widths`. This is a convolution of `vector` with `wavelet(width)` for
484:         each width in `widths`. See `cwt`
485:      2. Identify "ridge lines" in the cwt matrix. These are relative maxima
486:         at each row, connected across adjacent rows. See identify_ridge_lines
487:      3. Filter the ridge_lines using filter_ridge_lines.
488: 
489:     .. versionadded:: 0.11.0
490: 
491:     References
492:     ----------
493:     .. [1] Bioinformatics (2006) 22 (17): 2059-2065.
494:         :doi:`10.1093/bioinformatics/btl355`
495:         http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
496: 
497:     Examples
498:     --------
499:     >>> from scipy import signal
500:     >>> xs = np.arange(0, np.pi, 0.05)
501:     >>> data = np.sin(xs)
502:     >>> peakind = signal.find_peaks_cwt(data, np.arange(1,10))
503:     >>> peakind, xs[peakind], data[peakind]
504:     ([32], array([ 1.6]), array([ 0.9995736]))
505: 
506:     '''
507:     widths = np.asarray(widths)
508: 
509:     if gap_thresh is None:
510:         gap_thresh = np.ceil(widths[0])
511:     if max_distances is None:
512:         max_distances = widths / 4.0
513:     if wavelet is None:
514:         wavelet = ricker
515: 
516:     cwt_dat = cwt(vector, wavelet, widths)
517:     ridge_lines = _identify_ridge_lines(cwt_dat, max_distances, gap_thresh)
518:     filtered = _filter_ridge_lines(cwt_dat, ridge_lines, min_length=min_length,
519:                                    min_snr=min_snr, noise_perc=noise_perc)
520:     max_locs = np.asarray([x[1][0] for x in filtered])
521:     max_locs.sort()
522: 
523:     return max_locs
524: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_287427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nFunctions for identifying peaks in signals.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_287428 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_287428) is not StypyTypeError):

    if (import_287428 != 'pyd_module'):
        __import__(import_287428)
        sys_modules_287429 = sys.modules[import_287428]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_287429.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_287428)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib.six import xrange' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_287430 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six')

if (type(import_287430) is not StypyTypeError):

    if (import_287430 != 'pyd_module'):
        __import__(import_287430)
        sys_modules_287431 = sys.modules[import_287430]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', sys_modules_287431.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_287431, sys_modules_287431.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', import_287430)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.signal.wavelets import cwt, ricker' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_287432 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal.wavelets')

if (type(import_287432) is not StypyTypeError):

    if (import_287432 != 'pyd_module'):
        __import__(import_287432)
        sys_modules_287433 = sys.modules[import_287432]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal.wavelets', sys_modules_287433.module_type_store, module_type_store, ['cwt', 'ricker'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_287433, sys_modules_287433.module_type_store, module_type_store)
    else:
        from scipy.signal.wavelets import cwt, ricker

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal.wavelets', None, module_type_store, ['cwt', 'ricker'], [cwt, ricker])

else:
    # Assigning a type to the variable 'scipy.signal.wavelets' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal.wavelets', import_287432)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.stats import scoreatpercentile' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_287434 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.stats')

if (type(import_287434) is not StypyTypeError):

    if (import_287434 != 'pyd_module'):
        __import__(import_287434)
        sys_modules_287435 = sys.modules[import_287434]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.stats', sys_modules_287435.module_type_store, module_type_store, ['scoreatpercentile'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_287435, sys_modules_287435.module_type_store, module_type_store)
    else:
        from scipy.stats import scoreatpercentile

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.stats', None, module_type_store, ['scoreatpercentile'], [scoreatpercentile])

else:
    # Assigning a type to the variable 'scipy.stats' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.stats', import_287434)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a List to a Name (line 13):

# Assigning a List to a Name (line 13):
__all__ = ['argrelmin', 'argrelmax', 'argrelextrema', 'find_peaks_cwt']
module_type_store.set_exportable_members(['argrelmin', 'argrelmax', 'argrelextrema', 'find_peaks_cwt'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_287436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_287437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'argrelmin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_287436, str_287437)
# Adding element type (line 13)
str_287438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 24), 'str', 'argrelmax')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_287436, str_287438)
# Adding element type (line 13)
str_287439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 37), 'str', 'argrelextrema')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_287436, str_287439)
# Adding element type (line 13)
str_287440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 54), 'str', 'find_peaks_cwt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_287436, str_287440)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_287436)

@norecursion
def _boolrelextrema(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_287441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 43), 'int')
    int_287442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 52), 'int')
    str_287443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 60), 'str', 'clip')
    defaults = [int_287441, int_287442, str_287443]
    # Create a new context for function '_boolrelextrema'
    module_type_store = module_type_store.open_function_context('_boolrelextrema', 16, 0, False)
    
    # Passed parameters checking function
    _boolrelextrema.stypy_localization = localization
    _boolrelextrema.stypy_type_of_self = None
    _boolrelextrema.stypy_type_store = module_type_store
    _boolrelextrema.stypy_function_name = '_boolrelextrema'
    _boolrelextrema.stypy_param_names_list = ['data', 'comparator', 'axis', 'order', 'mode']
    _boolrelextrema.stypy_varargs_param_name = None
    _boolrelextrema.stypy_kwargs_param_name = None
    _boolrelextrema.stypy_call_defaults = defaults
    _boolrelextrema.stypy_call_varargs = varargs
    _boolrelextrema.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_boolrelextrema', ['data', 'comparator', 'axis', 'order', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_boolrelextrema', localization, ['data', 'comparator', 'axis', 'order', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_boolrelextrema(...)' code ##################

    str_287444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', "\n    Calculate the relative extrema of `data`.\n\n    Relative extrema are calculated by finding locations where\n    ``comparator(data[n], data[n+1:n+order+1])`` is True.\n\n    Parameters\n    ----------\n    data : ndarray\n        Array in which to find the relative extrema.\n    comparator : callable\n        Function to use to compare two data points.\n        Should take two arrays as arguments.\n    axis : int, optional\n        Axis over which to select from `data`.  Default is 0.\n    order : int, optional\n        How many points on each side to use for the comparison\n        to consider ``comparator(n,n+x)`` to be True.\n    mode : str, optional\n        How the edges of the vector are treated.  'wrap' (wrap around) or\n        'clip' (treat overflow as the same as the last (or first) element).\n        Default 'clip'.  See numpy.take\n\n    Returns\n    -------\n    extrema : ndarray\n        Boolean array of the same shape as `data` that is True at an extrema,\n        False otherwise.\n\n    See also\n    --------\n    argrelmax, argrelmin\n\n    Examples\n    --------\n    >>> testdata = np.array([1,2,3,2,1])\n    >>> _boolrelextrema(testdata, np.greater, axis=0)\n    array([False, False,  True, False, False], dtype=bool)\n\n    ")
    
    
    # Evaluating a boolean operation
    
    
    # Call to int(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'order' (line 57)
    order_287446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'order', False)
    # Processing the call keyword arguments (line 57)
    kwargs_287447 = {}
    # Getting the type of 'int' (line 57)
    int_287445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'int', False)
    # Calling int(args, kwargs) (line 57)
    int_call_result_287448 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), int_287445, *[order_287446], **kwargs_287447)
    
    # Getting the type of 'order' (line 57)
    order_287449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 22), 'order')
    # Applying the binary operator '!=' (line 57)
    result_ne_287450 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 8), '!=', int_call_result_287448, order_287449)
    
    
    # Getting the type of 'order' (line 57)
    order_287451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 33), 'order')
    int_287452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 41), 'int')
    # Applying the binary operator '<' (line 57)
    result_lt_287453 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 33), '<', order_287451, int_287452)
    
    # Applying the binary operator 'or' (line 57)
    result_or_keyword_287454 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 7), 'or', result_ne_287450, result_lt_287453)
    
    # Testing the type of an if condition (line 57)
    if_condition_287455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 4), result_or_keyword_287454)
    # Assigning a type to the variable 'if_condition_287455' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'if_condition_287455', if_condition_287455)
    # SSA begins for if statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 58)
    # Processing the call arguments (line 58)
    str_287457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'str', 'Order must be an int >= 1')
    # Processing the call keyword arguments (line 58)
    kwargs_287458 = {}
    # Getting the type of 'ValueError' (line 58)
    ValueError_287456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 58)
    ValueError_call_result_287459 = invoke(stypy.reporting.localization.Localization(__file__, 58, 14), ValueError_287456, *[str_287457], **kwargs_287458)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 58, 8), ValueError_call_result_287459, 'raise parameter', BaseException)
    # SSA join for if statement (line 57)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 60):
    
    # Assigning a Subscript to a Name (line 60):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 60)
    axis_287460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'axis')
    # Getting the type of 'data' (line 60)
    data_287461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'data')
    # Obtaining the member 'shape' of a type (line 60)
    shape_287462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 14), data_287461, 'shape')
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___287463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 14), shape_287462, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_287464 = invoke(stypy.reporting.localization.Localization(__file__, 60, 14), getitem___287463, axis_287460)
    
    # Assigning a type to the variable 'datalen' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'datalen', subscript_call_result_287464)
    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Call to arange(...): (line 61)
    # Processing the call arguments (line 61)
    int_287467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 21), 'int')
    # Getting the type of 'datalen' (line 61)
    datalen_287468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'datalen', False)
    # Processing the call keyword arguments (line 61)
    kwargs_287469 = {}
    # Getting the type of 'np' (line 61)
    np_287465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 61)
    arange_287466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 11), np_287465, 'arange')
    # Calling arange(args, kwargs) (line 61)
    arange_call_result_287470 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), arange_287466, *[int_287467, datalen_287468], **kwargs_287469)
    
    # Assigning a type to the variable 'locs' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'locs', arange_call_result_287470)
    
    # Assigning a Call to a Name (line 63):
    
    # Assigning a Call to a Name (line 63):
    
    # Call to ones(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'data' (line 63)
    data_287473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'data', False)
    # Obtaining the member 'shape' of a type (line 63)
    shape_287474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 22), data_287473, 'shape')
    # Processing the call keyword arguments (line 63)
    # Getting the type of 'bool' (line 63)
    bool_287475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), 'bool', False)
    keyword_287476 = bool_287475
    kwargs_287477 = {'dtype': keyword_287476}
    # Getting the type of 'np' (line 63)
    np_287471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), 'np', False)
    # Obtaining the member 'ones' of a type (line 63)
    ones_287472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 14), np_287471, 'ones')
    # Calling ones(args, kwargs) (line 63)
    ones_call_result_287478 = invoke(stypy.reporting.localization.Localization(__file__, 63, 14), ones_287472, *[shape_287474], **kwargs_287477)
    
    # Assigning a type to the variable 'results' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'results', ones_call_result_287478)
    
    # Assigning a Call to a Name (line 64):
    
    # Assigning a Call to a Name (line 64):
    
    # Call to take(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'locs' (line 64)
    locs_287481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'locs', False)
    # Processing the call keyword arguments (line 64)
    # Getting the type of 'axis' (line 64)
    axis_287482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), 'axis', False)
    keyword_287483 = axis_287482
    # Getting the type of 'mode' (line 64)
    mode_287484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 43), 'mode', False)
    keyword_287485 = mode_287484
    kwargs_287486 = {'mode': keyword_287485, 'axis': keyword_287483}
    # Getting the type of 'data' (line 64)
    data_287479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'data', False)
    # Obtaining the member 'take' of a type (line 64)
    take_287480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), data_287479, 'take')
    # Calling take(args, kwargs) (line 64)
    take_call_result_287487 = invoke(stypy.reporting.localization.Localization(__file__, 64, 11), take_287480, *[locs_287481], **kwargs_287486)
    
    # Assigning a type to the variable 'main' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'main', take_call_result_287487)
    
    
    # Call to xrange(...): (line 65)
    # Processing the call arguments (line 65)
    int_287489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'int')
    # Getting the type of 'order' (line 65)
    order_287490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'order', False)
    int_287491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 35), 'int')
    # Applying the binary operator '+' (line 65)
    result_add_287492 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 27), '+', order_287490, int_287491)
    
    # Processing the call keyword arguments (line 65)
    kwargs_287493 = {}
    # Getting the type of 'xrange' (line 65)
    xrange_287488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 17), 'xrange', False)
    # Calling xrange(args, kwargs) (line 65)
    xrange_call_result_287494 = invoke(stypy.reporting.localization.Localization(__file__, 65, 17), xrange_287488, *[int_287489, result_add_287492], **kwargs_287493)
    
    # Testing the type of a for loop iterable (line 65)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 4), xrange_call_result_287494)
    # Getting the type of the for loop variable (line 65)
    for_loop_var_287495 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 4), xrange_call_result_287494)
    # Assigning a type to the variable 'shift' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'shift', for_loop_var_287495)
    # SSA begins for a for statement (line 65)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 66):
    
    # Assigning a Call to a Name (line 66):
    
    # Call to take(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'locs' (line 66)
    locs_287498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'locs', False)
    # Getting the type of 'shift' (line 66)
    shift_287499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'shift', False)
    # Applying the binary operator '+' (line 66)
    result_add_287500 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 25), '+', locs_287498, shift_287499)
    
    # Processing the call keyword arguments (line 66)
    # Getting the type of 'axis' (line 66)
    axis_287501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 44), 'axis', False)
    keyword_287502 = axis_287501
    # Getting the type of 'mode' (line 66)
    mode_287503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 55), 'mode', False)
    keyword_287504 = mode_287503
    kwargs_287505 = {'mode': keyword_287504, 'axis': keyword_287502}
    # Getting the type of 'data' (line 66)
    data_287496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'data', False)
    # Obtaining the member 'take' of a type (line 66)
    take_287497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 15), data_287496, 'take')
    # Calling take(args, kwargs) (line 66)
    take_call_result_287506 = invoke(stypy.reporting.localization.Localization(__file__, 66, 15), take_287497, *[result_add_287500], **kwargs_287505)
    
    # Assigning a type to the variable 'plus' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'plus', take_call_result_287506)
    
    # Assigning a Call to a Name (line 67):
    
    # Assigning a Call to a Name (line 67):
    
    # Call to take(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'locs' (line 67)
    locs_287509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'locs', False)
    # Getting the type of 'shift' (line 67)
    shift_287510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 'shift', False)
    # Applying the binary operator '-' (line 67)
    result_sub_287511 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 26), '-', locs_287509, shift_287510)
    
    # Processing the call keyword arguments (line 67)
    # Getting the type of 'axis' (line 67)
    axis_287512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 45), 'axis', False)
    keyword_287513 = axis_287512
    # Getting the type of 'mode' (line 67)
    mode_287514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 56), 'mode', False)
    keyword_287515 = mode_287514
    kwargs_287516 = {'mode': keyword_287515, 'axis': keyword_287513}
    # Getting the type of 'data' (line 67)
    data_287507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'data', False)
    # Obtaining the member 'take' of a type (line 67)
    take_287508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 16), data_287507, 'take')
    # Calling take(args, kwargs) (line 67)
    take_call_result_287517 = invoke(stypy.reporting.localization.Localization(__file__, 67, 16), take_287508, *[result_sub_287511], **kwargs_287516)
    
    # Assigning a type to the variable 'minus' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'minus', take_call_result_287517)
    
    # Getting the type of 'results' (line 68)
    results_287518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'results')
    
    # Call to comparator(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'main' (line 68)
    main_287520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'main', False)
    # Getting the type of 'plus' (line 68)
    plus_287521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'plus', False)
    # Processing the call keyword arguments (line 68)
    kwargs_287522 = {}
    # Getting the type of 'comparator' (line 68)
    comparator_287519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'comparator', False)
    # Calling comparator(args, kwargs) (line 68)
    comparator_call_result_287523 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), comparator_287519, *[main_287520, plus_287521], **kwargs_287522)
    
    # Applying the binary operator '&=' (line 68)
    result_iand_287524 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 8), '&=', results_287518, comparator_call_result_287523)
    # Assigning a type to the variable 'results' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'results', result_iand_287524)
    
    
    # Getting the type of 'results' (line 69)
    results_287525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'results')
    
    # Call to comparator(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'main' (line 69)
    main_287527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'main', False)
    # Getting the type of 'minus' (line 69)
    minus_287528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 36), 'minus', False)
    # Processing the call keyword arguments (line 69)
    kwargs_287529 = {}
    # Getting the type of 'comparator' (line 69)
    comparator_287526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'comparator', False)
    # Calling comparator(args, kwargs) (line 69)
    comparator_call_result_287530 = invoke(stypy.reporting.localization.Localization(__file__, 69, 19), comparator_287526, *[main_287527, minus_287528], **kwargs_287529)
    
    # Applying the binary operator '&=' (line 69)
    result_iand_287531 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 8), '&=', results_287525, comparator_call_result_287530)
    # Assigning a type to the variable 'results' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'results', result_iand_287531)
    
    
    
    
    # Call to any(...): (line 70)
    # Processing the call keyword arguments (line 70)
    kwargs_287534 = {}
    # Getting the type of 'results' (line 70)
    results_287532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'results', False)
    # Obtaining the member 'any' of a type (line 70)
    any_287533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), results_287532, 'any')
    # Calling any(args, kwargs) (line 70)
    any_call_result_287535 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), any_287533, *[], **kwargs_287534)
    
    # Applying the '~' unary operator (line 70)
    result_inv_287536 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 11), '~', any_call_result_287535)
    
    # Testing the type of an if condition (line 70)
    if_condition_287537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 8), result_inv_287536)
    # Assigning a type to the variable 'if_condition_287537' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'if_condition_287537', if_condition_287537)
    # SSA begins for if statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'results' (line 71)
    results_287538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'results')
    # Assigning a type to the variable 'stypy_return_type' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'stypy_return_type', results_287538)
    # SSA join for if statement (line 70)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'results' (line 72)
    results_287539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'results')
    # Assigning a type to the variable 'stypy_return_type' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type', results_287539)
    
    # ################# End of '_boolrelextrema(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_boolrelextrema' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_287540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287540)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_boolrelextrema'
    return stypy_return_type_287540

# Assigning a type to the variable '_boolrelextrema' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '_boolrelextrema', _boolrelextrema)

@norecursion
def argrelmin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_287541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
    int_287542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'int')
    str_287543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 42), 'str', 'clip')
    defaults = [int_287541, int_287542, str_287543]
    # Create a new context for function 'argrelmin'
    module_type_store = module_type_store.open_function_context('argrelmin', 75, 0, False)
    
    # Passed parameters checking function
    argrelmin.stypy_localization = localization
    argrelmin.stypy_type_of_self = None
    argrelmin.stypy_type_store = module_type_store
    argrelmin.stypy_function_name = 'argrelmin'
    argrelmin.stypy_param_names_list = ['data', 'axis', 'order', 'mode']
    argrelmin.stypy_varargs_param_name = None
    argrelmin.stypy_kwargs_param_name = None
    argrelmin.stypy_call_defaults = defaults
    argrelmin.stypy_call_varargs = varargs
    argrelmin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'argrelmin', ['data', 'axis', 'order', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'argrelmin', localization, ['data', 'axis', 'order', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'argrelmin(...)' code ##################

    str_287544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'str', "\n    Calculate the relative minima of `data`.\n\n    Parameters\n    ----------\n    data : ndarray\n        Array in which to find the relative minima.\n    axis : int, optional\n        Axis over which to select from `data`.  Default is 0.\n    order : int, optional\n        How many points on each side to use for the comparison\n        to consider ``comparator(n, n+x)`` to be True.\n    mode : str, optional\n        How the edges of the vector are treated.\n        Available options are 'wrap' (wrap around) or 'clip' (treat overflow\n        as the same as the last (or first) element).\n        Default 'clip'. See numpy.take\n\n    Returns\n    -------\n    extrema : tuple of ndarrays\n        Indices of the minima in arrays of integers.  ``extrema[k]`` is\n        the array of indices of axis `k` of `data`.  Note that the\n        return value is a tuple even when `data` is one-dimensional.\n\n    See Also\n    --------\n    argrelextrema, argrelmax\n\n    Notes\n    -----\n    This function uses `argrelextrema` with np.less as comparator.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    >>> from scipy.signal import argrelmin\n    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])\n    >>> argrelmin(x)\n    (array([1, 5]),)\n    >>> y = np.array([[1, 2, 1, 2],\n    ...               [2, 2, 0, 0],\n    ...               [5, 3, 4, 4]])\n    ...\n    >>> argrelmin(y, axis=1)\n    (array([0, 2]), array([2, 1]))\n\n    ")
    
    # Call to argrelextrema(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'data' (line 125)
    data_287546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'data', False)
    # Getting the type of 'np' (line 125)
    np_287547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 31), 'np', False)
    # Obtaining the member 'less' of a type (line 125)
    less_287548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 31), np_287547, 'less')
    # Getting the type of 'axis' (line 125)
    axis_287549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'axis', False)
    # Getting the type of 'order' (line 125)
    order_287550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 46), 'order', False)
    # Getting the type of 'mode' (line 125)
    mode_287551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 53), 'mode', False)
    # Processing the call keyword arguments (line 125)
    kwargs_287552 = {}
    # Getting the type of 'argrelextrema' (line 125)
    argrelextrema_287545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'argrelextrema', False)
    # Calling argrelextrema(args, kwargs) (line 125)
    argrelextrema_call_result_287553 = invoke(stypy.reporting.localization.Localization(__file__, 125, 11), argrelextrema_287545, *[data_287546, less_287548, axis_287549, order_287550, mode_287551], **kwargs_287552)
    
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type', argrelextrema_call_result_287553)
    
    # ################# End of 'argrelmin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'argrelmin' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_287554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287554)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'argrelmin'
    return stypy_return_type_287554

# Assigning a type to the variable 'argrelmin' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'argrelmin', argrelmin)

@norecursion
def argrelmax(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_287555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 25), 'int')
    int_287556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 34), 'int')
    str_287557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 42), 'str', 'clip')
    defaults = [int_287555, int_287556, str_287557]
    # Create a new context for function 'argrelmax'
    module_type_store = module_type_store.open_function_context('argrelmax', 128, 0, False)
    
    # Passed parameters checking function
    argrelmax.stypy_localization = localization
    argrelmax.stypy_type_of_self = None
    argrelmax.stypy_type_store = module_type_store
    argrelmax.stypy_function_name = 'argrelmax'
    argrelmax.stypy_param_names_list = ['data', 'axis', 'order', 'mode']
    argrelmax.stypy_varargs_param_name = None
    argrelmax.stypy_kwargs_param_name = None
    argrelmax.stypy_call_defaults = defaults
    argrelmax.stypy_call_varargs = varargs
    argrelmax.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'argrelmax', ['data', 'axis', 'order', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'argrelmax', localization, ['data', 'axis', 'order', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'argrelmax(...)' code ##################

    str_287558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, (-1)), 'str', "\n    Calculate the relative maxima of `data`.\n\n    Parameters\n    ----------\n    data : ndarray\n        Array in which to find the relative maxima.\n    axis : int, optional\n        Axis over which to select from `data`.  Default is 0.\n    order : int, optional\n        How many points on each side to use for the comparison\n        to consider ``comparator(n, n+x)`` to be True.\n    mode : str, optional\n        How the edges of the vector are treated.\n        Available options are 'wrap' (wrap around) or 'clip' (treat overflow\n        as the same as the last (or first) element).\n        Default 'clip'.  See `numpy.take`.\n\n    Returns\n    -------\n    extrema : tuple of ndarrays\n        Indices of the maxima in arrays of integers.  ``extrema[k]`` is\n        the array of indices of axis `k` of `data`.  Note that the\n        return value is a tuple even when `data` is one-dimensional.\n\n    See Also\n    --------\n    argrelextrema, argrelmin\n\n    Notes\n    -----\n    This function uses `argrelextrema` with np.greater as comparator.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    >>> from scipy.signal import argrelmax\n    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])\n    >>> argrelmax(x)\n    (array([3, 6]),)\n    >>> y = np.array([[1, 2, 1, 2],\n    ...               [2, 2, 0, 0],\n    ...               [5, 3, 4, 4]])\n    ...\n    >>> argrelmax(y, axis=1)\n    (array([0]), array([1]))\n    ")
    
    # Call to argrelextrema(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'data' (line 177)
    data_287560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 'data', False)
    # Getting the type of 'np' (line 177)
    np_287561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'np', False)
    # Obtaining the member 'greater' of a type (line 177)
    greater_287562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 31), np_287561, 'greater')
    # Getting the type of 'axis' (line 177)
    axis_287563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 43), 'axis', False)
    # Getting the type of 'order' (line 177)
    order_287564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 49), 'order', False)
    # Getting the type of 'mode' (line 177)
    mode_287565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 56), 'mode', False)
    # Processing the call keyword arguments (line 177)
    kwargs_287566 = {}
    # Getting the type of 'argrelextrema' (line 177)
    argrelextrema_287559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'argrelextrema', False)
    # Calling argrelextrema(args, kwargs) (line 177)
    argrelextrema_call_result_287567 = invoke(stypy.reporting.localization.Localization(__file__, 177, 11), argrelextrema_287559, *[data_287560, greater_287562, axis_287563, order_287564, mode_287565], **kwargs_287566)
    
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type', argrelextrema_call_result_287567)
    
    # ################# End of 'argrelmax(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'argrelmax' in the type store
    # Getting the type of 'stypy_return_type' (line 128)
    stypy_return_type_287568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287568)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'argrelmax'
    return stypy_return_type_287568

# Assigning a type to the variable 'argrelmax' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'argrelmax', argrelmax)

@norecursion
def argrelextrema(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_287569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 41), 'int')
    int_287570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 50), 'int')
    str_287571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 58), 'str', 'clip')
    defaults = [int_287569, int_287570, str_287571]
    # Create a new context for function 'argrelextrema'
    module_type_store = module_type_store.open_function_context('argrelextrema', 180, 0, False)
    
    # Passed parameters checking function
    argrelextrema.stypy_localization = localization
    argrelextrema.stypy_type_of_self = None
    argrelextrema.stypy_type_store = module_type_store
    argrelextrema.stypy_function_name = 'argrelextrema'
    argrelextrema.stypy_param_names_list = ['data', 'comparator', 'axis', 'order', 'mode']
    argrelextrema.stypy_varargs_param_name = None
    argrelextrema.stypy_kwargs_param_name = None
    argrelextrema.stypy_call_defaults = defaults
    argrelextrema.stypy_call_varargs = varargs
    argrelextrema.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'argrelextrema', ['data', 'comparator', 'axis', 'order', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'argrelextrema', localization, ['data', 'comparator', 'axis', 'order', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'argrelextrema(...)' code ##################

    str_287572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, (-1)), 'str', "\n    Calculate the relative extrema of `data`.\n\n    Parameters\n    ----------\n    data : ndarray\n        Array in which to find the relative extrema.\n    comparator : callable\n        Function to use to compare two data points.\n        Should take two arrays as arguments.\n    axis : int, optional\n        Axis over which to select from `data`.  Default is 0.\n    order : int, optional\n        How many points on each side to use for the comparison\n        to consider ``comparator(n, n+x)`` to be True.\n    mode : str, optional\n        How the edges of the vector are treated.  'wrap' (wrap around) or\n        'clip' (treat overflow as the same as the last (or first) element).\n        Default is 'clip'.  See `numpy.take`.\n\n    Returns\n    -------\n    extrema : tuple of ndarrays\n        Indices of the maxima in arrays of integers.  ``extrema[k]`` is\n        the array of indices of axis `k` of `data`.  Note that the\n        return value is a tuple even when `data` is one-dimensional.\n\n    See Also\n    --------\n    argrelmin, argrelmax\n\n    Notes\n    -----\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    >>> from scipy.signal import argrelextrema\n    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])\n    >>> argrelextrema(x, np.greater)\n    (array([3, 6]),)\n    >>> y = np.array([[1, 2, 1, 2],\n    ...               [2, 2, 0, 0],\n    ...               [5, 3, 4, 4]])\n    ...\n    >>> argrelextrema(y, np.less, axis=1)\n    (array([0, 2]), array([2, 1]))\n\n    ")
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to _boolrelextrema(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'data' (line 231)
    data_287574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 30), 'data', False)
    # Getting the type of 'comparator' (line 231)
    comparator_287575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 36), 'comparator', False)
    # Getting the type of 'axis' (line 232)
    axis_287576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 30), 'axis', False)
    # Getting the type of 'order' (line 232)
    order_287577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 36), 'order', False)
    # Getting the type of 'mode' (line 232)
    mode_287578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 43), 'mode', False)
    # Processing the call keyword arguments (line 231)
    kwargs_287579 = {}
    # Getting the type of '_boolrelextrema' (line 231)
    _boolrelextrema_287573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 14), '_boolrelextrema', False)
    # Calling _boolrelextrema(args, kwargs) (line 231)
    _boolrelextrema_call_result_287580 = invoke(stypy.reporting.localization.Localization(__file__, 231, 14), _boolrelextrema_287573, *[data_287574, comparator_287575, axis_287576, order_287577, mode_287578], **kwargs_287579)
    
    # Assigning a type to the variable 'results' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'results', _boolrelextrema_call_result_287580)
    
    # Call to where(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'results' (line 233)
    results_287583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'results', False)
    # Processing the call keyword arguments (line 233)
    kwargs_287584 = {}
    # Getting the type of 'np' (line 233)
    np_287581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'np', False)
    # Obtaining the member 'where' of a type (line 233)
    where_287582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 11), np_287581, 'where')
    # Calling where(args, kwargs) (line 233)
    where_call_result_287585 = invoke(stypy.reporting.localization.Localization(__file__, 233, 11), where_287582, *[results_287583], **kwargs_287584)
    
    # Assigning a type to the variable 'stypy_return_type' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type', where_call_result_287585)
    
    # ################# End of 'argrelextrema(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'argrelextrema' in the type store
    # Getting the type of 'stypy_return_type' (line 180)
    stypy_return_type_287586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287586)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'argrelextrema'
    return stypy_return_type_287586

# Assigning a type to the variable 'argrelextrema' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'argrelextrema', argrelextrema)

@norecursion
def _identify_ridge_lines(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_identify_ridge_lines'
    module_type_store = module_type_store.open_function_context('_identify_ridge_lines', 236, 0, False)
    
    # Passed parameters checking function
    _identify_ridge_lines.stypy_localization = localization
    _identify_ridge_lines.stypy_type_of_self = None
    _identify_ridge_lines.stypy_type_store = module_type_store
    _identify_ridge_lines.stypy_function_name = '_identify_ridge_lines'
    _identify_ridge_lines.stypy_param_names_list = ['matr', 'max_distances', 'gap_thresh']
    _identify_ridge_lines.stypy_varargs_param_name = None
    _identify_ridge_lines.stypy_kwargs_param_name = None
    _identify_ridge_lines.stypy_call_defaults = defaults
    _identify_ridge_lines.stypy_call_varargs = varargs
    _identify_ridge_lines.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_identify_ridge_lines', ['matr', 'max_distances', 'gap_thresh'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_identify_ridge_lines', localization, ['matr', 'max_distances', 'gap_thresh'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_identify_ridge_lines(...)' code ##################

    str_287587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, (-1)), 'str', '\n    Identify ridges in the 2-D matrix.\n\n    Expect that the width of the wavelet feature increases with increasing row\n    number.\n\n    Parameters\n    ----------\n    matr : 2-D ndarray\n        Matrix in which to identify ridge lines.\n    max_distances : 1-D sequence\n        At each row, a ridge line is only connected\n        if the relative max at row[n] is within\n        `max_distances`[n] from the relative max at row[n+1].\n    gap_thresh : int\n        If a relative maximum is not found within `max_distances`,\n        there will be a gap. A ridge line is discontinued if\n        there are more than `gap_thresh` points without connecting\n        a new relative maximum.\n\n    Returns\n    -------\n    ridge_lines : tuple\n        Tuple of 2 1-D sequences. `ridge_lines`[ii][0] are the rows of the\n        ii-th ridge-line, `ridge_lines`[ii][1] are the columns. Empty if none\n        found.  Each ridge-line will be sorted by row (increasing), but the\n        order of the ridge lines is not specified.\n\n    References\n    ----------\n    Bioinformatics (2006) 22 (17): 2059-2065.\n    :doi:`10.1093/bioinformatics/btl355`\n    http://bioinformatics.oxfordjournals.org/content/22/17/2059.long\n\n    Examples\n    --------\n    >>> data = np.random.rand(5,5)\n    >>> ridge_lines = _identify_ridge_lines(data, 1, 1)\n\n    Notes\n    -----\n    This function is intended to be used in conjunction with `cwt`\n    as part of `find_peaks_cwt`.\n\n    ')
    
    
    
    # Call to len(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'max_distances' (line 282)
    max_distances_287589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 11), 'max_distances', False)
    # Processing the call keyword arguments (line 282)
    kwargs_287590 = {}
    # Getting the type of 'len' (line 282)
    len_287588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 7), 'len', False)
    # Calling len(args, kwargs) (line 282)
    len_call_result_287591 = invoke(stypy.reporting.localization.Localization(__file__, 282, 7), len_287588, *[max_distances_287589], **kwargs_287590)
    
    
    # Obtaining the type of the subscript
    int_287592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 39), 'int')
    # Getting the type of 'matr' (line 282)
    matr_287593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 28), 'matr')
    # Obtaining the member 'shape' of a type (line 282)
    shape_287594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 28), matr_287593, 'shape')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___287595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 28), shape_287594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_287596 = invoke(stypy.reporting.localization.Localization(__file__, 282, 28), getitem___287595, int_287592)
    
    # Applying the binary operator '<' (line 282)
    result_lt_287597 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 7), '<', len_call_result_287591, subscript_call_result_287596)
    
    # Testing the type of an if condition (line 282)
    if_condition_287598 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 4), result_lt_287597)
    # Assigning a type to the variable 'if_condition_287598' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'if_condition_287598', if_condition_287598)
    # SSA begins for if statement (line 282)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 283)
    # Processing the call arguments (line 283)
    str_287600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 25), 'str', 'Max_distances must have at least as many rows as matr')
    # Processing the call keyword arguments (line 283)
    kwargs_287601 = {}
    # Getting the type of 'ValueError' (line 283)
    ValueError_287599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 283)
    ValueError_call_result_287602 = invoke(stypy.reporting.localization.Localization(__file__, 283, 14), ValueError_287599, *[str_287600], **kwargs_287601)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 283, 8), ValueError_call_result_287602, 'raise parameter', BaseException)
    # SSA join for if statement (line 282)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 286):
    
    # Assigning a Call to a Name (line 286):
    
    # Call to _boolrelextrema(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'matr' (line 286)
    matr_287604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 35), 'matr', False)
    # Getting the type of 'np' (line 286)
    np_287605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 41), 'np', False)
    # Obtaining the member 'greater' of a type (line 286)
    greater_287606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 41), np_287605, 'greater')
    # Processing the call keyword arguments (line 286)
    int_287607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 58), 'int')
    keyword_287608 = int_287607
    int_287609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 67), 'int')
    keyword_287610 = int_287609
    kwargs_287611 = {'order': keyword_287610, 'axis': keyword_287608}
    # Getting the type of '_boolrelextrema' (line 286)
    _boolrelextrema_287603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), '_boolrelextrema', False)
    # Calling _boolrelextrema(args, kwargs) (line 286)
    _boolrelextrema_call_result_287612 = invoke(stypy.reporting.localization.Localization(__file__, 286, 19), _boolrelextrema_287603, *[matr_287604, greater_287606], **kwargs_287611)
    
    # Assigning a type to the variable 'all_max_cols' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'all_max_cols', _boolrelextrema_call_result_287612)
    
    # Assigning a Subscript to a Name (line 288):
    
    # Assigning a Subscript to a Name (line 288):
    
    # Obtaining the type of the subscript
    int_287613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 52), 'int')
    
    # Call to where(...): (line 288)
    # Processing the call arguments (line 288)
    
    # Call to any(...): (line 288)
    # Processing the call keyword arguments (line 288)
    int_287618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 48), 'int')
    keyword_287619 = int_287618
    kwargs_287620 = {'axis': keyword_287619}
    # Getting the type of 'all_max_cols' (line 288)
    all_max_cols_287616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 26), 'all_max_cols', False)
    # Obtaining the member 'any' of a type (line 288)
    any_287617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 26), all_max_cols_287616, 'any')
    # Calling any(args, kwargs) (line 288)
    any_call_result_287621 = invoke(stypy.reporting.localization.Localization(__file__, 288, 26), any_287617, *[], **kwargs_287620)
    
    # Processing the call keyword arguments (line 288)
    kwargs_287622 = {}
    # Getting the type of 'np' (line 288)
    np_287614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 17), 'np', False)
    # Obtaining the member 'where' of a type (line 288)
    where_287615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 17), np_287614, 'where')
    # Calling where(args, kwargs) (line 288)
    where_call_result_287623 = invoke(stypy.reporting.localization.Localization(__file__, 288, 17), where_287615, *[any_call_result_287621], **kwargs_287622)
    
    # Obtaining the member '__getitem__' of a type (line 288)
    getitem___287624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 17), where_call_result_287623, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 288)
    subscript_call_result_287625 = invoke(stypy.reporting.localization.Localization(__file__, 288, 17), getitem___287624, int_287613)
    
    # Assigning a type to the variable 'has_relmax' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'has_relmax', subscript_call_result_287625)
    
    
    
    # Call to len(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'has_relmax' (line 289)
    has_relmax_287627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 11), 'has_relmax', False)
    # Processing the call keyword arguments (line 289)
    kwargs_287628 = {}
    # Getting the type of 'len' (line 289)
    len_287626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 7), 'len', False)
    # Calling len(args, kwargs) (line 289)
    len_call_result_287629 = invoke(stypy.reporting.localization.Localization(__file__, 289, 7), len_287626, *[has_relmax_287627], **kwargs_287628)
    
    int_287630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 26), 'int')
    # Applying the binary operator '==' (line 289)
    result_eq_287631 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 7), '==', len_call_result_287629, int_287630)
    
    # Testing the type of an if condition (line 289)
    if_condition_287632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 4), result_eq_287631)
    # Assigning a type to the variable 'if_condition_287632' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'if_condition_287632', if_condition_287632)
    # SSA begins for if statement (line 289)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 290)
    list_287633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 290)
    
    # Assigning a type to the variable 'stypy_return_type' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'stypy_return_type', list_287633)
    # SSA join for if statement (line 289)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 291):
    
    # Assigning a Subscript to a Name (line 291):
    
    # Obtaining the type of the subscript
    int_287634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 27), 'int')
    # Getting the type of 'has_relmax' (line 291)
    has_relmax_287635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'has_relmax')
    # Obtaining the member '__getitem__' of a type (line 291)
    getitem___287636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 16), has_relmax_287635, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 291)
    subscript_call_result_287637 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), getitem___287636, int_287634)
    
    # Assigning a type to the variable 'start_row' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'start_row', subscript_call_result_287637)
    
    # Assigning a ListComp to a Name (line 294):
    
    # Assigning a ListComp to a Name (line 294):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_287644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 67), 'int')
    
    # Call to where(...): (line 296)
    # Processing the call arguments (line 296)
    
    # Obtaining the type of the subscript
    # Getting the type of 'start_row' (line 296)
    start_row_287647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 55), 'start_row', False)
    # Getting the type of 'all_max_cols' (line 296)
    all_max_cols_287648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 42), 'all_max_cols', False)
    # Obtaining the member '__getitem__' of a type (line 296)
    getitem___287649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 42), all_max_cols_287648, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 296)
    subscript_call_result_287650 = invoke(stypy.reporting.localization.Localization(__file__, 296, 42), getitem___287649, start_row_287647)
    
    # Processing the call keyword arguments (line 296)
    kwargs_287651 = {}
    # Getting the type of 'np' (line 296)
    np_287645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 33), 'np', False)
    # Obtaining the member 'where' of a type (line 296)
    where_287646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 33), np_287645, 'where')
    # Calling where(args, kwargs) (line 296)
    where_call_result_287652 = invoke(stypy.reporting.localization.Localization(__file__, 296, 33), where_287646, *[subscript_call_result_287650], **kwargs_287651)
    
    # Obtaining the member '__getitem__' of a type (line 296)
    getitem___287653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 33), where_call_result_287652, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 296)
    subscript_call_result_287654 = invoke(stypy.reporting.localization.Localization(__file__, 296, 33), getitem___287653, int_287644)
    
    comprehension_287655 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 19), subscript_call_result_287654)
    # Assigning a type to the variable 'col' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'col', comprehension_287655)
    
    # Obtaining an instance of the builtin type 'list' (line 294)
    list_287638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 294)
    # Adding element type (line 294)
    
    # Obtaining an instance of the builtin type 'list' (line 294)
    list_287639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 294)
    # Adding element type (line 294)
    # Getting the type of 'start_row' (line 294)
    start_row_287640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'start_row')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 20), list_287639, start_row_287640)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 19), list_287638, list_287639)
    # Adding element type (line 294)
    
    # Obtaining an instance of the builtin type 'list' (line 295)
    list_287641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 295)
    # Adding element type (line 295)
    # Getting the type of 'col' (line 295)
    col_287642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'col')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 19), list_287641, col_287642)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 19), list_287638, list_287641)
    # Adding element type (line 294)
    int_287643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 19), list_287638, int_287643)
    
    list_287656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 19), list_287656, list_287638)
    # Assigning a type to the variable 'ridge_lines' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'ridge_lines', list_287656)
    
    # Assigning a List to a Name (line 297):
    
    # Assigning a List to a Name (line 297):
    
    # Obtaining an instance of the builtin type 'list' (line 297)
    list_287657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 297)
    
    # Assigning a type to the variable 'final_lines' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'final_lines', list_287657)
    
    # Assigning a Call to a Name (line 298):
    
    # Assigning a Call to a Name (line 298):
    
    # Call to arange(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'start_row' (line 298)
    start_row_287660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'start_row', False)
    int_287661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 33), 'int')
    # Applying the binary operator '-' (line 298)
    result_sub_287662 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 21), '-', start_row_287660, int_287661)
    
    int_287663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 36), 'int')
    int_287664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 40), 'int')
    # Processing the call keyword arguments (line 298)
    kwargs_287665 = {}
    # Getting the type of 'np' (line 298)
    np_287658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 298)
    arange_287659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 11), np_287658, 'arange')
    # Calling arange(args, kwargs) (line 298)
    arange_call_result_287666 = invoke(stypy.reporting.localization.Localization(__file__, 298, 11), arange_287659, *[result_sub_287662, int_287663, int_287664], **kwargs_287665)
    
    # Assigning a type to the variable 'rows' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'rows', arange_call_result_287666)
    
    # Assigning a Call to a Name (line 299):
    
    # Assigning a Call to a Name (line 299):
    
    # Call to arange(...): (line 299)
    # Processing the call arguments (line 299)
    int_287669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 21), 'int')
    
    # Obtaining the type of the subscript
    int_287670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 35), 'int')
    # Getting the type of 'matr' (line 299)
    matr_287671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'matr', False)
    # Obtaining the member 'shape' of a type (line 299)
    shape_287672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 24), matr_287671, 'shape')
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___287673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 24), shape_287672, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_287674 = invoke(stypy.reporting.localization.Localization(__file__, 299, 24), getitem___287673, int_287670)
    
    # Processing the call keyword arguments (line 299)
    kwargs_287675 = {}
    # Getting the type of 'np' (line 299)
    np_287667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 299)
    arange_287668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 11), np_287667, 'arange')
    # Calling arange(args, kwargs) (line 299)
    arange_call_result_287676 = invoke(stypy.reporting.localization.Localization(__file__, 299, 11), arange_287668, *[int_287669, subscript_call_result_287674], **kwargs_287675)
    
    # Assigning a type to the variable 'cols' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'cols', arange_call_result_287676)
    
    # Getting the type of 'rows' (line 300)
    rows_287677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'rows')
    # Testing the type of a for loop iterable (line 300)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 300, 4), rows_287677)
    # Getting the type of the for loop variable (line 300)
    for_loop_var_287678 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 300, 4), rows_287677)
    # Assigning a type to the variable 'row' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'row', for_loop_var_287678)
    # SSA begins for a for statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 301):
    
    # Assigning a Subscript to a Name (line 301):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 301)
    row_287679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 42), 'row')
    # Getting the type of 'all_max_cols' (line 301)
    all_max_cols_287680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 29), 'all_max_cols')
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___287681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 29), all_max_cols_287680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_287682 = invoke(stypy.reporting.localization.Localization(__file__, 301, 29), getitem___287681, row_287679)
    
    # Getting the type of 'cols' (line 301)
    cols_287683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'cols')
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___287684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 24), cols_287683, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_287685 = invoke(stypy.reporting.localization.Localization(__file__, 301, 24), getitem___287684, subscript_call_result_287682)
    
    # Assigning a type to the variable 'this_max_cols' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'this_max_cols', subscript_call_result_287685)
    
    # Getting the type of 'ridge_lines' (line 305)
    ridge_lines_287686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), 'ridge_lines')
    # Testing the type of a for loop iterable (line 305)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 305, 8), ridge_lines_287686)
    # Getting the type of the for loop variable (line 305)
    for_loop_var_287687 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 305, 8), ridge_lines_287686)
    # Assigning a type to the variable 'line' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'line', for_loop_var_287687)
    # SSA begins for a for statement (line 305)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'line' (line 306)
    line_287688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'line')
    
    # Obtaining the type of the subscript
    int_287689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 17), 'int')
    # Getting the type of 'line' (line 306)
    line_287690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'line')
    # Obtaining the member '__getitem__' of a type (line 306)
    getitem___287691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), line_287690, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 306)
    subscript_call_result_287692 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), getitem___287691, int_287689)
    
    int_287693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 23), 'int')
    # Applying the binary operator '+=' (line 306)
    result_iadd_287694 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 12), '+=', subscript_call_result_287692, int_287693)
    # Getting the type of 'line' (line 306)
    line_287695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'line')
    int_287696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 17), 'int')
    # Storing an element on a container (line 306)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 12), line_287695, (int_287696, result_iadd_287694))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 311):
    
    # Assigning a Call to a Name (line 311):
    
    # Call to array(...): (line 311)
    # Processing the call arguments (line 311)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ridge_lines' (line 311)
    ridge_lines_287706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 60), 'ridge_lines', False)
    comprehension_287707 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 36), ridge_lines_287706)
    # Assigning a type to the variable 'line' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 36), 'line', comprehension_287707)
    
    # Obtaining the type of the subscript
    int_287699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 44), 'int')
    
    # Obtaining the type of the subscript
    int_287700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 41), 'int')
    # Getting the type of 'line' (line 311)
    line_287701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 36), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___287702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 36), line_287701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_287703 = invoke(stypy.reporting.localization.Localization(__file__, 311, 36), getitem___287702, int_287700)
    
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___287704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 36), subscript_call_result_287703, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_287705 = invoke(stypy.reporting.localization.Localization(__file__, 311, 36), getitem___287704, int_287699)
    
    list_287708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 36), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 36), list_287708, subscript_call_result_287705)
    # Processing the call keyword arguments (line 311)
    kwargs_287709 = {}
    # Getting the type of 'np' (line 311)
    np_287697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 26), 'np', False)
    # Obtaining the member 'array' of a type (line 311)
    array_287698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 26), np_287697, 'array')
    # Calling array(args, kwargs) (line 311)
    array_call_result_287710 = invoke(stypy.reporting.localization.Localization(__file__, 311, 26), array_287698, *[list_287708], **kwargs_287709)
    
    # Assigning a type to the variable 'prev_ridge_cols' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'prev_ridge_cols', array_call_result_287710)
    
    
    # Call to enumerate(...): (line 314)
    # Processing the call arguments (line 314)
    # Getting the type of 'this_max_cols' (line 314)
    this_max_cols_287712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 34), 'this_max_cols', False)
    # Processing the call keyword arguments (line 314)
    kwargs_287713 = {}
    # Getting the type of 'enumerate' (line 314)
    enumerate_287711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 24), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 314)
    enumerate_call_result_287714 = invoke(stypy.reporting.localization.Localization(__file__, 314, 24), enumerate_287711, *[this_max_cols_287712], **kwargs_287713)
    
    # Testing the type of a for loop iterable (line 314)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 314, 8), enumerate_call_result_287714)
    # Getting the type of the for loop variable (line 314)
    for_loop_var_287715 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 314, 8), enumerate_call_result_287714)
    # Assigning a type to the variable 'ind' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 8), for_loop_var_287715))
    # Assigning a type to the variable 'col' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 8), for_loop_var_287715))
    # SSA begins for a for statement (line 314)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 318):
    
    # Assigning a Name to a Name (line 318):
    # Getting the type of 'None' (line 318)
    None_287716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 19), 'None')
    # Assigning a type to the variable 'line' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'line', None_287716)
    
    
    
    # Call to len(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'prev_ridge_cols' (line 319)
    prev_ridge_cols_287718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'prev_ridge_cols', False)
    # Processing the call keyword arguments (line 319)
    kwargs_287719 = {}
    # Getting the type of 'len' (line 319)
    len_287717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 15), 'len', False)
    # Calling len(args, kwargs) (line 319)
    len_call_result_287720 = invoke(stypy.reporting.localization.Localization(__file__, 319, 15), len_287717, *[prev_ridge_cols_287718], **kwargs_287719)
    
    int_287721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 38), 'int')
    # Applying the binary operator '>' (line 319)
    result_gt_287722 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 15), '>', len_call_result_287720, int_287721)
    
    # Testing the type of an if condition (line 319)
    if_condition_287723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 12), result_gt_287722)
    # Assigning a type to the variable 'if_condition_287723' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'if_condition_287723', if_condition_287723)
    # SSA begins for if statement (line 319)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 320):
    
    # Assigning a Call to a Name (line 320):
    
    # Call to abs(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'col' (line 320)
    col_287726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 31), 'col', False)
    # Getting the type of 'prev_ridge_cols' (line 320)
    prev_ridge_cols_287727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 37), 'prev_ridge_cols', False)
    # Applying the binary operator '-' (line 320)
    result_sub_287728 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 31), '-', col_287726, prev_ridge_cols_287727)
    
    # Processing the call keyword arguments (line 320)
    kwargs_287729 = {}
    # Getting the type of 'np' (line 320)
    np_287724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'np', False)
    # Obtaining the member 'abs' of a type (line 320)
    abs_287725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 24), np_287724, 'abs')
    # Calling abs(args, kwargs) (line 320)
    abs_call_result_287730 = invoke(stypy.reporting.localization.Localization(__file__, 320, 24), abs_287725, *[result_sub_287728], **kwargs_287729)
    
    # Assigning a type to the variable 'diffs' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'diffs', abs_call_result_287730)
    
    # Assigning a Call to a Name (line 321):
    
    # Assigning a Call to a Name (line 321):
    
    # Call to argmin(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'diffs' (line 321)
    diffs_287733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 36), 'diffs', False)
    # Processing the call keyword arguments (line 321)
    kwargs_287734 = {}
    # Getting the type of 'np' (line 321)
    np_287731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 26), 'np', False)
    # Obtaining the member 'argmin' of a type (line 321)
    argmin_287732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 26), np_287731, 'argmin')
    # Calling argmin(args, kwargs) (line 321)
    argmin_call_result_287735 = invoke(stypy.reporting.localization.Localization(__file__, 321, 26), argmin_287732, *[diffs_287733], **kwargs_287734)
    
    # Assigning a type to the variable 'closest' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'closest', argmin_call_result_287735)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'closest' (line 322)
    closest_287736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'closest')
    # Getting the type of 'diffs' (line 322)
    diffs_287737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 19), 'diffs')
    # Obtaining the member '__getitem__' of a type (line 322)
    getitem___287738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 19), diffs_287737, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
    subscript_call_result_287739 = invoke(stypy.reporting.localization.Localization(__file__, 322, 19), getitem___287738, closest_287736)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 322)
    row_287740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 51), 'row')
    # Getting the type of 'max_distances' (line 322)
    max_distances_287741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 37), 'max_distances')
    # Obtaining the member '__getitem__' of a type (line 322)
    getitem___287742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 37), max_distances_287741, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
    subscript_call_result_287743 = invoke(stypy.reporting.localization.Localization(__file__, 322, 37), getitem___287742, row_287740)
    
    # Applying the binary operator '<=' (line 322)
    result_le_287744 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 19), '<=', subscript_call_result_287739, subscript_call_result_287743)
    
    # Testing the type of an if condition (line 322)
    if_condition_287745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 16), result_le_287744)
    # Assigning a type to the variable 'if_condition_287745' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 16), 'if_condition_287745', if_condition_287745)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 323):
    
    # Assigning a Subscript to a Name (line 323):
    
    # Obtaining the type of the subscript
    # Getting the type of 'closest' (line 323)
    closest_287746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 39), 'closest')
    # Getting the type of 'ridge_lines' (line 323)
    ridge_lines_287747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 27), 'ridge_lines')
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___287748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 27), ridge_lines_287747, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
    subscript_call_result_287749 = invoke(stypy.reporting.localization.Localization(__file__, 323, 27), getitem___287748, closest_287746)
    
    # Assigning a type to the variable 'line' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 20), 'line', subscript_call_result_287749)
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 319)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 324)
    # Getting the type of 'line' (line 324)
    line_287750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'line')
    # Getting the type of 'None' (line 324)
    None_287751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 27), 'None')
    
    (may_be_287752, more_types_in_union_287753) = may_not_be_none(line_287750, None_287751)

    if may_be_287752:

        if more_types_in_union_287753:
            # Runtime conditional SSA (line 324)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'col' (line 326)
        col_287759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 31), 'col', False)
        # Processing the call keyword arguments (line 326)
        kwargs_287760 = {}
        
        # Obtaining the type of the subscript
        int_287754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 21), 'int')
        # Getting the type of 'line' (line 326)
        line_287755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 326)
        getitem___287756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 16), line_287755, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 326)
        subscript_call_result_287757 = invoke(stypy.reporting.localization.Localization(__file__, 326, 16), getitem___287756, int_287754)
        
        # Obtaining the member 'append' of a type (line 326)
        append_287758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 16), subscript_call_result_287757, 'append')
        # Calling append(args, kwargs) (line 326)
        append_call_result_287761 = invoke(stypy.reporting.localization.Localization(__file__, 326, 16), append_287758, *[col_287759], **kwargs_287760)
        
        
        # Call to append(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'row' (line 327)
        row_287767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 31), 'row', False)
        # Processing the call keyword arguments (line 327)
        kwargs_287768 = {}
        
        # Obtaining the type of the subscript
        int_287762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 21), 'int')
        # Getting the type of 'line' (line 327)
        line_287763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___287764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 16), line_287763, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_287765 = invoke(stypy.reporting.localization.Localization(__file__, 327, 16), getitem___287764, int_287762)
        
        # Obtaining the member 'append' of a type (line 327)
        append_287766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 16), subscript_call_result_287765, 'append')
        # Calling append(args, kwargs) (line 327)
        append_call_result_287769 = invoke(stypy.reporting.localization.Localization(__file__, 327, 16), append_287766, *[row_287767], **kwargs_287768)
        
        
        # Assigning a Num to a Subscript (line 328):
        
        # Assigning a Num to a Subscript (line 328):
        int_287770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 26), 'int')
        # Getting the type of 'line' (line 328)
        line_287771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'line')
        int_287772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 21), 'int')
        # Storing an element on a container (line 328)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 16), line_287771, (int_287772, int_287770))

        if more_types_in_union_287753:
            # Runtime conditional SSA for else branch (line 324)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_287752) or more_types_in_union_287753):
        
        # Assigning a List to a Name (line 330):
        
        # Assigning a List to a Name (line 330):
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_287773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_287774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        # Getting the type of 'row' (line 330)
        row_287775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 29), 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 28), list_287774, row_287775)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 27), list_287773, list_287774)
        # Adding element type (line 330)
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_287776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        # Getting the type of 'col' (line 331)
        col_287777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 29), 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 28), list_287776, col_287777)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 27), list_287773, list_287776)
        # Adding element type (line 330)
        int_287778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 27), list_287773, int_287778)
        
        # Assigning a type to the variable 'new_line' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'new_line', list_287773)
        
        # Call to append(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'new_line' (line 333)
        new_line_287781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 35), 'new_line', False)
        # Processing the call keyword arguments (line 333)
        kwargs_287782 = {}
        # Getting the type of 'ridge_lines' (line 333)
        ridge_lines_287779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'ridge_lines', False)
        # Obtaining the member 'append' of a type (line 333)
        append_287780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), ridge_lines_287779, 'append')
        # Calling append(args, kwargs) (line 333)
        append_call_result_287783 = invoke(stypy.reporting.localization.Localization(__file__, 333, 16), append_287780, *[new_line_287781], **kwargs_287782)
        

        if (may_be_287752 and more_types_in_union_287753):
            # SSA join for if statement (line 324)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to xrange(...): (line 339)
    # Processing the call arguments (line 339)
    
    # Call to len(...): (line 339)
    # Processing the call arguments (line 339)
    # Getting the type of 'ridge_lines' (line 339)
    ridge_lines_287786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 30), 'ridge_lines', False)
    # Processing the call keyword arguments (line 339)
    kwargs_287787 = {}
    # Getting the type of 'len' (line 339)
    len_287785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 26), 'len', False)
    # Calling len(args, kwargs) (line 339)
    len_call_result_287788 = invoke(stypy.reporting.localization.Localization(__file__, 339, 26), len_287785, *[ridge_lines_287786], **kwargs_287787)
    
    int_287789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 45), 'int')
    # Applying the binary operator '-' (line 339)
    result_sub_287790 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 26), '-', len_call_result_287788, int_287789)
    
    int_287791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 48), 'int')
    int_287792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 52), 'int')
    # Processing the call keyword arguments (line 339)
    kwargs_287793 = {}
    # Getting the type of 'xrange' (line 339)
    xrange_287784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'xrange', False)
    # Calling xrange(args, kwargs) (line 339)
    xrange_call_result_287794 = invoke(stypy.reporting.localization.Localization(__file__, 339, 19), xrange_287784, *[result_sub_287790, int_287791, int_287792], **kwargs_287793)
    
    # Testing the type of a for loop iterable (line 339)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 339, 8), xrange_call_result_287794)
    # Getting the type of the for loop variable (line 339)
    for_loop_var_287795 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 339, 8), xrange_call_result_287794)
    # Assigning a type to the variable 'ind' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'ind', for_loop_var_287795)
    # SSA begins for a for statement (line 339)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 340):
    
    # Assigning a Subscript to a Name (line 340):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 340)
    ind_287796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 31), 'ind')
    # Getting the type of 'ridge_lines' (line 340)
    ridge_lines_287797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 19), 'ridge_lines')
    # Obtaining the member '__getitem__' of a type (line 340)
    getitem___287798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 19), ridge_lines_287797, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 340)
    subscript_call_result_287799 = invoke(stypy.reporting.localization.Localization(__file__, 340, 19), getitem___287798, ind_287796)
    
    # Assigning a type to the variable 'line' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'line', subscript_call_result_287799)
    
    
    
    # Obtaining the type of the subscript
    int_287800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 20), 'int')
    # Getting the type of 'line' (line 341)
    line_287801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'line')
    # Obtaining the member '__getitem__' of a type (line 341)
    getitem___287802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), line_287801, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 341)
    subscript_call_result_287803 = invoke(stypy.reporting.localization.Localization(__file__, 341, 15), getitem___287802, int_287800)
    
    # Getting the type of 'gap_thresh' (line 341)
    gap_thresh_287804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'gap_thresh')
    # Applying the binary operator '>' (line 341)
    result_gt_287805 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 15), '>', subscript_call_result_287803, gap_thresh_287804)
    
    # Testing the type of an if condition (line 341)
    if_condition_287806 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 12), result_gt_287805)
    # Assigning a type to the variable 'if_condition_287806' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'if_condition_287806', if_condition_287806)
    # SSA begins for if statement (line 341)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'line' (line 342)
    line_287809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 35), 'line', False)
    # Processing the call keyword arguments (line 342)
    kwargs_287810 = {}
    # Getting the type of 'final_lines' (line 342)
    final_lines_287807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'final_lines', False)
    # Obtaining the member 'append' of a type (line 342)
    append_287808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 16), final_lines_287807, 'append')
    # Calling append(args, kwargs) (line 342)
    append_call_result_287811 = invoke(stypy.reporting.localization.Localization(__file__, 342, 16), append_287808, *[line_287809], **kwargs_287810)
    
    # Deleting a member
    # Getting the type of 'ridge_lines' (line 343)
    ridge_lines_287812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 20), 'ridge_lines')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 343)
    ind_287813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 32), 'ind')
    # Getting the type of 'ridge_lines' (line 343)
    ridge_lines_287814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 20), 'ridge_lines')
    # Obtaining the member '__getitem__' of a type (line 343)
    getitem___287815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 20), ridge_lines_287814, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 343)
    subscript_call_result_287816 = invoke(stypy.reporting.localization.Localization(__file__, 343, 20), getitem___287815, ind_287813)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 16), ridge_lines_287812, subscript_call_result_287816)
    # SSA join for if statement (line 341)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 345):
    
    # Assigning a List to a Name (line 345):
    
    # Obtaining an instance of the builtin type 'list' (line 345)
    list_287817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 345)
    
    # Assigning a type to the variable 'out_lines' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'out_lines', list_287817)
    
    # Getting the type of 'final_lines' (line 346)
    final_lines_287818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 17), 'final_lines')
    # Getting the type of 'ridge_lines' (line 346)
    ridge_lines_287819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 31), 'ridge_lines')
    # Applying the binary operator '+' (line 346)
    result_add_287820 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 17), '+', final_lines_287818, ridge_lines_287819)
    
    # Testing the type of a for loop iterable (line 346)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 346, 4), result_add_287820)
    # Getting the type of the for loop variable (line 346)
    for_loop_var_287821 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 346, 4), result_add_287820)
    # Assigning a type to the variable 'line' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'line', for_loop_var_287821)
    # SSA begins for a for statement (line 346)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 347):
    
    # Assigning a Call to a Name (line 347):
    
    # Call to array(...): (line 347)
    # Processing the call arguments (line 347)
    
    # Call to argsort(...): (line 347)
    # Processing the call arguments (line 347)
    
    # Obtaining the type of the subscript
    int_287826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 44), 'int')
    # Getting the type of 'line' (line 347)
    line_287827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 39), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 347)
    getitem___287828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 39), line_287827, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 347)
    subscript_call_result_287829 = invoke(stypy.reporting.localization.Localization(__file__, 347, 39), getitem___287828, int_287826)
    
    # Processing the call keyword arguments (line 347)
    kwargs_287830 = {}
    # Getting the type of 'np' (line 347)
    np_287824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 28), 'np', False)
    # Obtaining the member 'argsort' of a type (line 347)
    argsort_287825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 28), np_287824, 'argsort')
    # Calling argsort(args, kwargs) (line 347)
    argsort_call_result_287831 = invoke(stypy.reporting.localization.Localization(__file__, 347, 28), argsort_287825, *[subscript_call_result_287829], **kwargs_287830)
    
    # Processing the call keyword arguments (line 347)
    kwargs_287832 = {}
    # Getting the type of 'np' (line 347)
    np_287822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 19), 'np', False)
    # Obtaining the member 'array' of a type (line 347)
    array_287823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 19), np_287822, 'array')
    # Calling array(args, kwargs) (line 347)
    array_call_result_287833 = invoke(stypy.reporting.localization.Localization(__file__, 347, 19), array_287823, *[argsort_call_result_287831], **kwargs_287832)
    
    # Assigning a type to the variable 'sortargs' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'sortargs', array_call_result_287833)
    
    # Assigning a Tuple to a Tuple (line 348):
    
    # Assigning a Call to a Name (line 348):
    
    # Call to zeros_like(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'sortargs' (line 348)
    sortargs_287836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 35), 'sortargs', False)
    # Processing the call keyword arguments (line 348)
    kwargs_287837 = {}
    # Getting the type of 'np' (line 348)
    np_287834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 21), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 348)
    zeros_like_287835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 21), np_287834, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 348)
    zeros_like_call_result_287838 = invoke(stypy.reporting.localization.Localization(__file__, 348, 21), zeros_like_287835, *[sortargs_287836], **kwargs_287837)
    
    # Assigning a type to the variable 'tuple_assignment_287423' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'tuple_assignment_287423', zeros_like_call_result_287838)
    
    # Assigning a Call to a Name (line 348):
    
    # Call to zeros_like(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'sortargs' (line 348)
    sortargs_287841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 60), 'sortargs', False)
    # Processing the call keyword arguments (line 348)
    kwargs_287842 = {}
    # Getting the type of 'np' (line 348)
    np_287839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 46), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 348)
    zeros_like_287840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 46), np_287839, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 348)
    zeros_like_call_result_287843 = invoke(stypy.reporting.localization.Localization(__file__, 348, 46), zeros_like_287840, *[sortargs_287841], **kwargs_287842)
    
    # Assigning a type to the variable 'tuple_assignment_287424' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'tuple_assignment_287424', zeros_like_call_result_287843)
    
    # Assigning a Name to a Name (line 348):
    # Getting the type of 'tuple_assignment_287423' (line 348)
    tuple_assignment_287423_287844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'tuple_assignment_287423')
    # Assigning a type to the variable 'rows' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'rows', tuple_assignment_287423_287844)
    
    # Assigning a Name to a Name (line 348):
    # Getting the type of 'tuple_assignment_287424' (line 348)
    tuple_assignment_287424_287845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'tuple_assignment_287424')
    # Assigning a type to the variable 'cols' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 14), 'cols', tuple_assignment_287424_287845)
    
    # Assigning a Subscript to a Subscript (line 349):
    
    # Assigning a Subscript to a Subscript (line 349):
    
    # Obtaining the type of the subscript
    int_287846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 30), 'int')
    # Getting the type of 'line' (line 349)
    line_287847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 25), 'line')
    # Obtaining the member '__getitem__' of a type (line 349)
    getitem___287848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 25), line_287847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 349)
    subscript_call_result_287849 = invoke(stypy.reporting.localization.Localization(__file__, 349, 25), getitem___287848, int_287846)
    
    # Getting the type of 'rows' (line 349)
    rows_287850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'rows')
    # Getting the type of 'sortargs' (line 349)
    sortargs_287851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 13), 'sortargs')
    # Storing an element on a container (line 349)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 8), rows_287850, (sortargs_287851, subscript_call_result_287849))
    
    # Assigning a Subscript to a Subscript (line 350):
    
    # Assigning a Subscript to a Subscript (line 350):
    
    # Obtaining the type of the subscript
    int_287852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 30), 'int')
    # Getting the type of 'line' (line 350)
    line_287853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 25), 'line')
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___287854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 25), line_287853, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_287855 = invoke(stypy.reporting.localization.Localization(__file__, 350, 25), getitem___287854, int_287852)
    
    # Getting the type of 'cols' (line 350)
    cols_287856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'cols')
    # Getting the type of 'sortargs' (line 350)
    sortargs_287857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 13), 'sortargs')
    # Storing an element on a container (line 350)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 8), cols_287856, (sortargs_287857, subscript_call_result_287855))
    
    # Call to append(...): (line 351)
    # Processing the call arguments (line 351)
    
    # Obtaining an instance of the builtin type 'list' (line 351)
    list_287860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 351)
    # Adding element type (line 351)
    # Getting the type of 'rows' (line 351)
    rows_287861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 26), 'rows', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 25), list_287860, rows_287861)
    # Adding element type (line 351)
    # Getting the type of 'cols' (line 351)
    cols_287862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 32), 'cols', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 25), list_287860, cols_287862)
    
    # Processing the call keyword arguments (line 351)
    kwargs_287863 = {}
    # Getting the type of 'out_lines' (line 351)
    out_lines_287858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'out_lines', False)
    # Obtaining the member 'append' of a type (line 351)
    append_287859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), out_lines_287858, 'append')
    # Calling append(args, kwargs) (line 351)
    append_call_result_287864 = invoke(stypy.reporting.localization.Localization(__file__, 351, 8), append_287859, *[list_287860], **kwargs_287863)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'out_lines' (line 353)
    out_lines_287865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 11), 'out_lines')
    # Assigning a type to the variable 'stypy_return_type' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'stypy_return_type', out_lines_287865)
    
    # ################# End of '_identify_ridge_lines(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_identify_ridge_lines' in the type store
    # Getting the type of 'stypy_return_type' (line 236)
    stypy_return_type_287866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287866)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_identify_ridge_lines'
    return stypy_return_type_287866

# Assigning a type to the variable '_identify_ridge_lines' (line 236)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), '_identify_ridge_lines', _identify_ridge_lines)

@norecursion
def _filter_ridge_lines(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 356)
    None_287867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 54), 'None')
    # Getting the type of 'None' (line 356)
    None_287868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 71), 'None')
    int_287869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 32), 'int')
    int_287870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 46), 'int')
    defaults = [None_287867, None_287868, int_287869, int_287870]
    # Create a new context for function '_filter_ridge_lines'
    module_type_store = module_type_store.open_function_context('_filter_ridge_lines', 356, 0, False)
    
    # Passed parameters checking function
    _filter_ridge_lines.stypy_localization = localization
    _filter_ridge_lines.stypy_type_of_self = None
    _filter_ridge_lines.stypy_type_store = module_type_store
    _filter_ridge_lines.stypy_function_name = '_filter_ridge_lines'
    _filter_ridge_lines.stypy_param_names_list = ['cwt', 'ridge_lines', 'window_size', 'min_length', 'min_snr', 'noise_perc']
    _filter_ridge_lines.stypy_varargs_param_name = None
    _filter_ridge_lines.stypy_kwargs_param_name = None
    _filter_ridge_lines.stypy_call_defaults = defaults
    _filter_ridge_lines.stypy_call_varargs = varargs
    _filter_ridge_lines.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_filter_ridge_lines', ['cwt', 'ridge_lines', 'window_size', 'min_length', 'min_snr', 'noise_perc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_filter_ridge_lines', localization, ['cwt', 'ridge_lines', 'window_size', 'min_length', 'min_snr', 'noise_perc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_filter_ridge_lines(...)' code ##################

    str_287871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, (-1)), 'str', '\n    Filter ridge lines according to prescribed criteria. Intended\n    to be used for finding relative maxima.\n\n    Parameters\n    ----------\n    cwt : 2-D ndarray\n        Continuous wavelet transform from which the `ridge_lines` were defined.\n    ridge_lines : 1-D sequence\n        Each element should contain 2 sequences, the rows and columns\n        of the ridge line (respectively).\n    window_size : int, optional\n        Size of window to use to calculate noise floor.\n        Default is ``cwt.shape[1] / 20``.\n    min_length : int, optional\n        Minimum length a ridge line needs to be acceptable.\n        Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.\n    min_snr : float, optional\n        Minimum SNR ratio. Default 1. The signal is the value of\n        the cwt matrix at the shortest length scale (``cwt[0, loc]``), the\n        noise is the `noise_perc`th percentile of datapoints contained within a\n        window of `window_size` around ``cwt[0, loc]``.\n    noise_perc : float, optional\n        When calculating the noise floor, percentile of data points\n        examined below which to consider noise. Calculated using\n        scipy.stats.scoreatpercentile.\n\n    References\n    ----------\n    Bioinformatics (2006) 22 (17): 2059-2065. :doi:`10.1093/bioinformatics/btl355`\n    http://bioinformatics.oxfordjournals.org/content/22/17/2059.long\n\n    ')
    
    # Assigning a Subscript to a Name (line 391):
    
    # Assigning a Subscript to a Name (line 391):
    
    # Obtaining the type of the subscript
    int_287872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 27), 'int')
    # Getting the type of 'cwt' (line 391)
    cwt_287873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 17), 'cwt')
    # Obtaining the member 'shape' of a type (line 391)
    shape_287874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 17), cwt_287873, 'shape')
    # Obtaining the member '__getitem__' of a type (line 391)
    getitem___287875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 17), shape_287874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 391)
    subscript_call_result_287876 = invoke(stypy.reporting.localization.Localization(__file__, 391, 17), getitem___287875, int_287872)
    
    # Assigning a type to the variable 'num_points' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'num_points', subscript_call_result_287876)
    
    # Type idiom detected: calculating its left and rigth part (line 392)
    # Getting the type of 'min_length' (line 392)
    min_length_287877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 7), 'min_length')
    # Getting the type of 'None' (line 392)
    None_287878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 21), 'None')
    
    (may_be_287879, more_types_in_union_287880) = may_be_none(min_length_287877, None_287878)

    if may_be_287879:

        if more_types_in_union_287880:
            # Runtime conditional SSA (line 392)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 393):
        
        # Assigning a Call to a Name (line 393):
        
        # Call to ceil(...): (line 393)
        # Processing the call arguments (line 393)
        
        # Obtaining the type of the subscript
        int_287883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 39), 'int')
        # Getting the type of 'cwt' (line 393)
        cwt_287884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 29), 'cwt', False)
        # Obtaining the member 'shape' of a type (line 393)
        shape_287885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 29), cwt_287884, 'shape')
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___287886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 29), shape_287885, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_287887 = invoke(stypy.reporting.localization.Localization(__file__, 393, 29), getitem___287886, int_287883)
        
        int_287888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 44), 'int')
        # Applying the binary operator 'div' (line 393)
        result_div_287889 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 29), 'div', subscript_call_result_287887, int_287888)
        
        # Processing the call keyword arguments (line 393)
        kwargs_287890 = {}
        # Getting the type of 'np' (line 393)
        np_287881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 21), 'np', False)
        # Obtaining the member 'ceil' of a type (line 393)
        ceil_287882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 21), np_287881, 'ceil')
        # Calling ceil(args, kwargs) (line 393)
        ceil_call_result_287891 = invoke(stypy.reporting.localization.Localization(__file__, 393, 21), ceil_287882, *[result_div_287889], **kwargs_287890)
        
        # Assigning a type to the variable 'min_length' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'min_length', ceil_call_result_287891)

        if more_types_in_union_287880:
            # SSA join for if statement (line 392)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 394)
    # Getting the type of 'window_size' (line 394)
    window_size_287892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 7), 'window_size')
    # Getting the type of 'None' (line 394)
    None_287893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 22), 'None')
    
    (may_be_287894, more_types_in_union_287895) = may_be_none(window_size_287892, None_287893)

    if may_be_287894:

        if more_types_in_union_287895:
            # Runtime conditional SSA (line 394)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to ceil(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'num_points' (line 395)
        num_points_287898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 30), 'num_points', False)
        int_287899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 43), 'int')
        # Applying the binary operator 'div' (line 395)
        result_div_287900 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 30), 'div', num_points_287898, int_287899)
        
        # Processing the call keyword arguments (line 395)
        kwargs_287901 = {}
        # Getting the type of 'np' (line 395)
        np_287896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 22), 'np', False)
        # Obtaining the member 'ceil' of a type (line 395)
        ceil_287897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 22), np_287896, 'ceil')
        # Calling ceil(args, kwargs) (line 395)
        ceil_call_result_287902 = invoke(stypy.reporting.localization.Localization(__file__, 395, 22), ceil_287897, *[result_div_287900], **kwargs_287901)
        
        # Assigning a type to the variable 'window_size' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'window_size', ceil_call_result_287902)

        if more_types_in_union_287895:
            # SSA join for if statement (line 394)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 397):
    
    # Assigning a Call to a Name (line 397):
    
    # Call to int(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of 'window_size' (line 397)
    window_size_287904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 22), 'window_size', False)
    # Processing the call keyword arguments (line 397)
    kwargs_287905 = {}
    # Getting the type of 'int' (line 397)
    int_287903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 18), 'int', False)
    # Calling int(args, kwargs) (line 397)
    int_call_result_287906 = invoke(stypy.reporting.localization.Localization(__file__, 397, 18), int_287903, *[window_size_287904], **kwargs_287905)
    
    # Assigning a type to the variable 'window_size' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'window_size', int_call_result_287906)
    
    # Assigning a Call to a Tuple (line 398):
    
    # Assigning a Subscript to a Name (line 398):
    
    # Obtaining the type of the subscript
    int_287907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 4), 'int')
    
    # Call to divmod(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'window_size' (line 398)
    window_size_287909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 28), 'window_size', False)
    int_287910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 41), 'int')
    # Processing the call keyword arguments (line 398)
    kwargs_287911 = {}
    # Getting the type of 'divmod' (line 398)
    divmod_287908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 21), 'divmod', False)
    # Calling divmod(args, kwargs) (line 398)
    divmod_call_result_287912 = invoke(stypy.reporting.localization.Localization(__file__, 398, 21), divmod_287908, *[window_size_287909, int_287910], **kwargs_287911)
    
    # Obtaining the member '__getitem__' of a type (line 398)
    getitem___287913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 4), divmod_call_result_287912, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 398)
    subscript_call_result_287914 = invoke(stypy.reporting.localization.Localization(__file__, 398, 4), getitem___287913, int_287907)
    
    # Assigning a type to the variable 'tuple_var_assignment_287425' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'tuple_var_assignment_287425', subscript_call_result_287914)
    
    # Assigning a Subscript to a Name (line 398):
    
    # Obtaining the type of the subscript
    int_287915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 4), 'int')
    
    # Call to divmod(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'window_size' (line 398)
    window_size_287917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 28), 'window_size', False)
    int_287918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 41), 'int')
    # Processing the call keyword arguments (line 398)
    kwargs_287919 = {}
    # Getting the type of 'divmod' (line 398)
    divmod_287916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 21), 'divmod', False)
    # Calling divmod(args, kwargs) (line 398)
    divmod_call_result_287920 = invoke(stypy.reporting.localization.Localization(__file__, 398, 21), divmod_287916, *[window_size_287917, int_287918], **kwargs_287919)
    
    # Obtaining the member '__getitem__' of a type (line 398)
    getitem___287921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 4), divmod_call_result_287920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 398)
    subscript_call_result_287922 = invoke(stypy.reporting.localization.Localization(__file__, 398, 4), getitem___287921, int_287915)
    
    # Assigning a type to the variable 'tuple_var_assignment_287426' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'tuple_var_assignment_287426', subscript_call_result_287922)
    
    # Assigning a Name to a Name (line 398):
    # Getting the type of 'tuple_var_assignment_287425' (line 398)
    tuple_var_assignment_287425_287923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'tuple_var_assignment_287425')
    # Assigning a type to the variable 'hf_window' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'hf_window', tuple_var_assignment_287425_287923)
    
    # Assigning a Name to a Name (line 398):
    # Getting the type of 'tuple_var_assignment_287426' (line 398)
    tuple_var_assignment_287426_287924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'tuple_var_assignment_287426')
    # Assigning a type to the variable 'odd' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 15), 'odd', tuple_var_assignment_287426_287924)
    
    # Assigning a Subscript to a Name (line 401):
    
    # Assigning a Subscript to a Name (line 401):
    
    # Obtaining the type of the subscript
    int_287925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 18), 'int')
    slice_287926 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 401, 14), None, None, None)
    # Getting the type of 'cwt' (line 401)
    cwt_287927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 14), 'cwt')
    # Obtaining the member '__getitem__' of a type (line 401)
    getitem___287928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 14), cwt_287927, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 401)
    subscript_call_result_287929 = invoke(stypy.reporting.localization.Localization(__file__, 401, 14), getitem___287928, (int_287925, slice_287926))
    
    # Assigning a type to the variable 'row_one' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'row_one', subscript_call_result_287929)
    
    # Assigning a Call to a Name (line 402):
    
    # Assigning a Call to a Name (line 402):
    
    # Call to zeros_like(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'row_one' (line 402)
    row_one_287932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 27), 'row_one', False)
    # Processing the call keyword arguments (line 402)
    kwargs_287933 = {}
    # Getting the type of 'np' (line 402)
    np_287930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 13), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 402)
    zeros_like_287931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 13), np_287930, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 402)
    zeros_like_call_result_287934 = invoke(stypy.reporting.localization.Localization(__file__, 402, 13), zeros_like_287931, *[row_one_287932], **kwargs_287933)
    
    # Assigning a type to the variable 'noises' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'noises', zeros_like_call_result_287934)
    
    
    # Call to enumerate(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'row_one' (line 403)
    row_one_287936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 30), 'row_one', False)
    # Processing the call keyword arguments (line 403)
    kwargs_287937 = {}
    # Getting the type of 'enumerate' (line 403)
    enumerate_287935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 20), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 403)
    enumerate_call_result_287938 = invoke(stypy.reporting.localization.Localization(__file__, 403, 20), enumerate_287935, *[row_one_287936], **kwargs_287937)
    
    # Testing the type of a for loop iterable (line 403)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 403, 4), enumerate_call_result_287938)
    # Getting the type of the for loop variable (line 403)
    for_loop_var_287939 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 403, 4), enumerate_call_result_287938)
    # Assigning a type to the variable 'ind' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 4), for_loop_var_287939))
    # Assigning a type to the variable 'val' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 4), for_loop_var_287939))
    # SSA begins for a for statement (line 403)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 404):
    
    # Assigning a Call to a Name (line 404):
    
    # Call to max(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'ind' (line 404)
    ind_287941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 27), 'ind', False)
    # Getting the type of 'hf_window' (line 404)
    hf_window_287942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 33), 'hf_window', False)
    # Applying the binary operator '-' (line 404)
    result_sub_287943 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 27), '-', ind_287941, hf_window_287942)
    
    int_287944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 44), 'int')
    # Processing the call keyword arguments (line 404)
    kwargs_287945 = {}
    # Getting the type of 'max' (line 404)
    max_287940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 23), 'max', False)
    # Calling max(args, kwargs) (line 404)
    max_call_result_287946 = invoke(stypy.reporting.localization.Localization(__file__, 404, 23), max_287940, *[result_sub_287943, int_287944], **kwargs_287945)
    
    # Assigning a type to the variable 'window_start' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'window_start', max_call_result_287946)
    
    # Assigning a Call to a Name (line 405):
    
    # Assigning a Call to a Name (line 405):
    
    # Call to min(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of 'ind' (line 405)
    ind_287948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'ind', False)
    # Getting the type of 'hf_window' (line 405)
    hf_window_287949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 31), 'hf_window', False)
    # Applying the binary operator '+' (line 405)
    result_add_287950 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 25), '+', ind_287948, hf_window_287949)
    
    # Getting the type of 'odd' (line 405)
    odd_287951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 43), 'odd', False)
    # Applying the binary operator '+' (line 405)
    result_add_287952 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 41), '+', result_add_287950, odd_287951)
    
    # Getting the type of 'num_points' (line 405)
    num_points_287953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 48), 'num_points', False)
    # Processing the call keyword arguments (line 405)
    kwargs_287954 = {}
    # Getting the type of 'min' (line 405)
    min_287947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), 'min', False)
    # Calling min(args, kwargs) (line 405)
    min_call_result_287955 = invoke(stypy.reporting.localization.Localization(__file__, 405, 21), min_287947, *[result_add_287952, num_points_287953], **kwargs_287954)
    
    # Assigning a type to the variable 'window_end' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'window_end', min_call_result_287955)
    
    # Assigning a Call to a Subscript (line 406):
    
    # Assigning a Call to a Subscript (line 406):
    
    # Call to scoreatpercentile(...): (line 406)
    # Processing the call arguments (line 406)
    
    # Obtaining the type of the subscript
    # Getting the type of 'window_start' (line 406)
    window_start_287957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 48), 'window_start', False)
    # Getting the type of 'window_end' (line 406)
    window_end_287958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 61), 'window_end', False)
    slice_287959 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 406, 40), window_start_287957, window_end_287958, None)
    # Getting the type of 'row_one' (line 406)
    row_one_287960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 40), 'row_one', False)
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___287961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 40), row_one_287960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_287962 = invoke(stypy.reporting.localization.Localization(__file__, 406, 40), getitem___287961, slice_287959)
    
    # Processing the call keyword arguments (line 406)
    # Getting the type of 'noise_perc' (line 407)
    noise_perc_287963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 44), 'noise_perc', False)
    keyword_287964 = noise_perc_287963
    kwargs_287965 = {'per': keyword_287964}
    # Getting the type of 'scoreatpercentile' (line 406)
    scoreatpercentile_287956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 22), 'scoreatpercentile', False)
    # Calling scoreatpercentile(args, kwargs) (line 406)
    scoreatpercentile_call_result_287966 = invoke(stypy.reporting.localization.Localization(__file__, 406, 22), scoreatpercentile_287956, *[subscript_call_result_287962], **kwargs_287965)
    
    # Getting the type of 'noises' (line 406)
    noises_287967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'noises')
    # Getting the type of 'ind' (line 406)
    ind_287968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 15), 'ind')
    # Storing an element on a container (line 406)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 8), noises_287967, (ind_287968, scoreatpercentile_call_result_287966))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def filt_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'filt_func'
        module_type_store = module_type_store.open_function_context('filt_func', 409, 4, False)
        
        # Passed parameters checking function
        filt_func.stypy_localization = localization
        filt_func.stypy_type_of_self = None
        filt_func.stypy_type_store = module_type_store
        filt_func.stypy_function_name = 'filt_func'
        filt_func.stypy_param_names_list = ['line']
        filt_func.stypy_varargs_param_name = None
        filt_func.stypy_kwargs_param_name = None
        filt_func.stypy_call_defaults = defaults
        filt_func.stypy_call_varargs = varargs
        filt_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'filt_func', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'filt_func', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'filt_func(...)' code ##################

        
        
        
        # Call to len(...): (line 410)
        # Processing the call arguments (line 410)
        
        # Obtaining the type of the subscript
        int_287970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 20), 'int')
        # Getting the type of 'line' (line 410)
        line_287971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 15), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 410)
        getitem___287972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 15), line_287971, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 410)
        subscript_call_result_287973 = invoke(stypy.reporting.localization.Localization(__file__, 410, 15), getitem___287972, int_287970)
        
        # Processing the call keyword arguments (line 410)
        kwargs_287974 = {}
        # Getting the type of 'len' (line 410)
        len_287969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 11), 'len', False)
        # Calling len(args, kwargs) (line 410)
        len_call_result_287975 = invoke(stypy.reporting.localization.Localization(__file__, 410, 11), len_287969, *[subscript_call_result_287973], **kwargs_287974)
        
        # Getting the type of 'min_length' (line 410)
        min_length_287976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 26), 'min_length')
        # Applying the binary operator '<' (line 410)
        result_lt_287977 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 11), '<', len_call_result_287975, min_length_287976)
        
        # Testing the type of an if condition (line 410)
        if_condition_287978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 410, 8), result_lt_287977)
        # Assigning a type to the variable 'if_condition_287978' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'if_condition_287978', if_condition_287978)
        # SSA begins for if statement (line 410)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 411)
        False_287979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'stypy_return_type', False_287979)
        # SSA join for if statement (line 410)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 412):
        
        # Assigning a Call to a Name (line 412):
        
        # Call to abs(...): (line 412)
        # Processing the call arguments (line 412)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 412)
        tuple_287981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 412)
        # Adding element type (line 412)
        
        # Obtaining the type of the subscript
        int_287982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 30), 'int')
        
        # Obtaining the type of the subscript
        int_287983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 27), 'int')
        # Getting the type of 'line' (line 412)
        line_287984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 22), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___287985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 22), line_287984, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_287986 = invoke(stypy.reporting.localization.Localization(__file__, 412, 22), getitem___287985, int_287983)
        
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___287987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 22), subscript_call_result_287986, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_287988 = invoke(stypy.reporting.localization.Localization(__file__, 412, 22), getitem___287987, int_287982)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 22), tuple_287981, subscript_call_result_287988)
        # Adding element type (line 412)
        
        # Obtaining the type of the subscript
        int_287989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 42), 'int')
        
        # Obtaining the type of the subscript
        int_287990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 39), 'int')
        # Getting the type of 'line' (line 412)
        line_287991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 34), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___287992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 34), line_287991, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_287993 = invoke(stypy.reporting.localization.Localization(__file__, 412, 34), getitem___287992, int_287990)
        
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___287994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 34), subscript_call_result_287993, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_287995 = invoke(stypy.reporting.localization.Localization(__file__, 412, 34), getitem___287994, int_287989)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 22), tuple_287981, subscript_call_result_287995)
        
        # Getting the type of 'cwt' (line 412)
        cwt_287996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 18), 'cwt', False)
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___287997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 18), cwt_287996, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_287998 = invoke(stypy.reporting.localization.Localization(__file__, 412, 18), getitem___287997, tuple_287981)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_287999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 63), 'int')
        
        # Obtaining the type of the subscript
        int_288000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 60), 'int')
        # Getting the type of 'line' (line 412)
        line_288001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 55), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___288002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 55), line_288001, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_288003 = invoke(stypy.reporting.localization.Localization(__file__, 412, 55), getitem___288002, int_288000)
        
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___288004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 55), subscript_call_result_288003, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_288005 = invoke(stypy.reporting.localization.Localization(__file__, 412, 55), getitem___288004, int_287999)
        
        # Getting the type of 'noises' (line 412)
        noises_288006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 48), 'noises', False)
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___288007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 48), noises_288006, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_288008 = invoke(stypy.reporting.localization.Localization(__file__, 412, 48), getitem___288007, subscript_call_result_288005)
        
        # Applying the binary operator 'div' (line 412)
        result_div_288009 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 18), 'div', subscript_call_result_287998, subscript_call_result_288008)
        
        # Processing the call keyword arguments (line 412)
        kwargs_288010 = {}
        # Getting the type of 'abs' (line 412)
        abs_287980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 14), 'abs', False)
        # Calling abs(args, kwargs) (line 412)
        abs_call_result_288011 = invoke(stypy.reporting.localization.Localization(__file__, 412, 14), abs_287980, *[result_div_288009], **kwargs_288010)
        
        # Assigning a type to the variable 'snr' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'snr', abs_call_result_288011)
        
        
        # Getting the type of 'snr' (line 413)
        snr_288012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 11), 'snr')
        # Getting the type of 'min_snr' (line 413)
        min_snr_288013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 17), 'min_snr')
        # Applying the binary operator '<' (line 413)
        result_lt_288014 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 11), '<', snr_288012, min_snr_288013)
        
        # Testing the type of an if condition (line 413)
        if_condition_288015 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 8), result_lt_288014)
        # Assigning a type to the variable 'if_condition_288015' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'if_condition_288015', if_condition_288015)
        # SSA begins for if statement (line 413)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 414)
        False_288016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'stypy_return_type', False_288016)
        # SSA join for if statement (line 413)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'True' (line 415)
        True_288017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'stypy_return_type', True_288017)
        
        # ################# End of 'filt_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'filt_func' in the type store
        # Getting the type of 'stypy_return_type' (line 409)
        stypy_return_type_288018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_288018)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'filt_func'
        return stypy_return_type_288018

    # Assigning a type to the variable 'filt_func' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'filt_func', filt_func)
    
    # Call to list(...): (line 417)
    # Processing the call arguments (line 417)
    
    # Call to filter(...): (line 417)
    # Processing the call arguments (line 417)
    # Getting the type of 'filt_func' (line 417)
    filt_func_288021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 23), 'filt_func', False)
    # Getting the type of 'ridge_lines' (line 417)
    ridge_lines_288022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 34), 'ridge_lines', False)
    # Processing the call keyword arguments (line 417)
    kwargs_288023 = {}
    # Getting the type of 'filter' (line 417)
    filter_288020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 16), 'filter', False)
    # Calling filter(args, kwargs) (line 417)
    filter_call_result_288024 = invoke(stypy.reporting.localization.Localization(__file__, 417, 16), filter_288020, *[filt_func_288021, ridge_lines_288022], **kwargs_288023)
    
    # Processing the call keyword arguments (line 417)
    kwargs_288025 = {}
    # Getting the type of 'list' (line 417)
    list_288019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 11), 'list', False)
    # Calling list(args, kwargs) (line 417)
    list_call_result_288026 = invoke(stypy.reporting.localization.Localization(__file__, 417, 11), list_288019, *[filter_call_result_288024], **kwargs_288025)
    
    # Assigning a type to the variable 'stypy_return_type' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'stypy_return_type', list_call_result_288026)
    
    # ################# End of '_filter_ridge_lines(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_filter_ridge_lines' in the type store
    # Getting the type of 'stypy_return_type' (line 356)
    stypy_return_type_288027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288027)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_filter_ridge_lines'
    return stypy_return_type_288027

# Assigning a type to the variable '_filter_ridge_lines' (line 356)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 0), '_filter_ridge_lines', _filter_ridge_lines)

@norecursion
def find_peaks_cwt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 420)
    None_288028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 43), 'None')
    # Getting the type of 'None' (line 420)
    None_288029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 63), 'None')
    # Getting the type of 'None' (line 421)
    None_288030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 30), 'None')
    # Getting the type of 'None' (line 421)
    None_288031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 47), 'None')
    int_288032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 61), 'int')
    int_288033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 75), 'int')
    defaults = [None_288028, None_288029, None_288030, None_288031, int_288032, int_288033]
    # Create a new context for function 'find_peaks_cwt'
    module_type_store = module_type_store.open_function_context('find_peaks_cwt', 420, 0, False)
    
    # Passed parameters checking function
    find_peaks_cwt.stypy_localization = localization
    find_peaks_cwt.stypy_type_of_self = None
    find_peaks_cwt.stypy_type_store = module_type_store
    find_peaks_cwt.stypy_function_name = 'find_peaks_cwt'
    find_peaks_cwt.stypy_param_names_list = ['vector', 'widths', 'wavelet', 'max_distances', 'gap_thresh', 'min_length', 'min_snr', 'noise_perc']
    find_peaks_cwt.stypy_varargs_param_name = None
    find_peaks_cwt.stypy_kwargs_param_name = None
    find_peaks_cwt.stypy_call_defaults = defaults
    find_peaks_cwt.stypy_call_varargs = varargs
    find_peaks_cwt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_peaks_cwt', ['vector', 'widths', 'wavelet', 'max_distances', 'gap_thresh', 'min_length', 'min_snr', 'noise_perc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_peaks_cwt', localization, ['vector', 'widths', 'wavelet', 'max_distances', 'gap_thresh', 'min_length', 'min_snr', 'noise_perc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_peaks_cwt(...)' code ##################

    str_288034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, (-1)), 'str', '\n    Attempt to find the peaks in a 1-D array.\n\n    The general approach is to smooth `vector` by convolving it with\n    `wavelet(width)` for each width in `widths`. Relative maxima which\n    appear at enough length scales, and with sufficiently high SNR, are\n    accepted.\n\n    Parameters\n    ----------\n    vector : ndarray\n        1-D array in which to find the peaks.\n    widths : sequence\n        1-D array of widths to use for calculating the CWT matrix. In general,\n        this range should cover the expected width of peaks of interest.\n    wavelet : callable, optional\n        Should take two parameters and return a 1-D array to convolve\n        with `vector`. The first parameter determines the number of points \n        of the returned wavelet array, the second parameter is the scale \n        (`width`) of the wavelet. Should be normalized and symmetric.\n        Default is the ricker wavelet.\n    max_distances : ndarray, optional\n        At each row, a ridge line is only connected if the relative max at\n        row[n] is within ``max_distances[n]`` from the relative max at\n        ``row[n+1]``.  Default value is ``widths/4``.\n    gap_thresh : float, optional\n        If a relative maximum is not found within `max_distances`,\n        there will be a gap. A ridge line is discontinued if there are more\n        than `gap_thresh` points without connecting a new relative maximum.\n        Default is the first value of the widths array i.e. widths[0].\n    min_length : int, optional\n        Minimum length a ridge line needs to be acceptable.\n        Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.\n    min_snr : float, optional\n        Minimum SNR ratio. Default 1. The signal is the value of\n        the cwt matrix at the shortest length scale (``cwt[0, loc]``), the\n        noise is the `noise_perc`th percentile of datapoints contained within a\n        window of `window_size` around ``cwt[0, loc]``.\n    noise_perc : float, optional\n        When calculating the noise floor, percentile of data points\n        examined below which to consider noise. Calculated using\n        `stats.scoreatpercentile`.  Default is 10.\n\n    Returns\n    -------\n    peaks_indices : ndarray\n        Indices of the locations in the `vector` where peaks were found.\n        The list is sorted.\n\n    See Also\n    --------\n    cwt\n\n    Notes\n    -----\n    This approach was designed for finding sharp peaks among noisy data,\n    however with proper parameter selection it should function well for\n    different peak shapes.\n\n    The algorithm is as follows:\n     1. Perform a continuous wavelet transform on `vector`, for the supplied\n        `widths`. This is a convolution of `vector` with `wavelet(width)` for\n        each width in `widths`. See `cwt`\n     2. Identify "ridge lines" in the cwt matrix. These are relative maxima\n        at each row, connected across adjacent rows. See identify_ridge_lines\n     3. Filter the ridge_lines using filter_ridge_lines.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1] Bioinformatics (2006) 22 (17): 2059-2065.\n        :doi:`10.1093/bioinformatics/btl355`\n        http://bioinformatics.oxfordjournals.org/content/22/17/2059.long\n\n    Examples\n    --------\n    >>> from scipy import signal\n    >>> xs = np.arange(0, np.pi, 0.05)\n    >>> data = np.sin(xs)\n    >>> peakind = signal.find_peaks_cwt(data, np.arange(1,10))\n    >>> peakind, xs[peakind], data[peakind]\n    ([32], array([ 1.6]), array([ 0.9995736]))\n\n    ')
    
    # Assigning a Call to a Name (line 507):
    
    # Assigning a Call to a Name (line 507):
    
    # Call to asarray(...): (line 507)
    # Processing the call arguments (line 507)
    # Getting the type of 'widths' (line 507)
    widths_288037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 24), 'widths', False)
    # Processing the call keyword arguments (line 507)
    kwargs_288038 = {}
    # Getting the type of 'np' (line 507)
    np_288035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 507)
    asarray_288036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 13), np_288035, 'asarray')
    # Calling asarray(args, kwargs) (line 507)
    asarray_call_result_288039 = invoke(stypy.reporting.localization.Localization(__file__, 507, 13), asarray_288036, *[widths_288037], **kwargs_288038)
    
    # Assigning a type to the variable 'widths' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'widths', asarray_call_result_288039)
    
    # Type idiom detected: calculating its left and rigth part (line 509)
    # Getting the type of 'gap_thresh' (line 509)
    gap_thresh_288040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 7), 'gap_thresh')
    # Getting the type of 'None' (line 509)
    None_288041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 21), 'None')
    
    (may_be_288042, more_types_in_union_288043) = may_be_none(gap_thresh_288040, None_288041)

    if may_be_288042:

        if more_types_in_union_288043:
            # Runtime conditional SSA (line 509)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 510):
        
        # Assigning a Call to a Name (line 510):
        
        # Call to ceil(...): (line 510)
        # Processing the call arguments (line 510)
        
        # Obtaining the type of the subscript
        int_288046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 36), 'int')
        # Getting the type of 'widths' (line 510)
        widths_288047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 29), 'widths', False)
        # Obtaining the member '__getitem__' of a type (line 510)
        getitem___288048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 29), widths_288047, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 510)
        subscript_call_result_288049 = invoke(stypy.reporting.localization.Localization(__file__, 510, 29), getitem___288048, int_288046)
        
        # Processing the call keyword arguments (line 510)
        kwargs_288050 = {}
        # Getting the type of 'np' (line 510)
        np_288044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 21), 'np', False)
        # Obtaining the member 'ceil' of a type (line 510)
        ceil_288045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 21), np_288044, 'ceil')
        # Calling ceil(args, kwargs) (line 510)
        ceil_call_result_288051 = invoke(stypy.reporting.localization.Localization(__file__, 510, 21), ceil_288045, *[subscript_call_result_288049], **kwargs_288050)
        
        # Assigning a type to the variable 'gap_thresh' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'gap_thresh', ceil_call_result_288051)

        if more_types_in_union_288043:
            # SSA join for if statement (line 509)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 511)
    # Getting the type of 'max_distances' (line 511)
    max_distances_288052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 7), 'max_distances')
    # Getting the type of 'None' (line 511)
    None_288053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 24), 'None')
    
    (may_be_288054, more_types_in_union_288055) = may_be_none(max_distances_288052, None_288053)

    if may_be_288054:

        if more_types_in_union_288055:
            # Runtime conditional SSA (line 511)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 512):
        
        # Assigning a BinOp to a Name (line 512):
        # Getting the type of 'widths' (line 512)
        widths_288056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 24), 'widths')
        float_288057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 33), 'float')
        # Applying the binary operator 'div' (line 512)
        result_div_288058 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 24), 'div', widths_288056, float_288057)
        
        # Assigning a type to the variable 'max_distances' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'max_distances', result_div_288058)

        if more_types_in_union_288055:
            # SSA join for if statement (line 511)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 513)
    # Getting the type of 'wavelet' (line 513)
    wavelet_288059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 7), 'wavelet')
    # Getting the type of 'None' (line 513)
    None_288060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 18), 'None')
    
    (may_be_288061, more_types_in_union_288062) = may_be_none(wavelet_288059, None_288060)

    if may_be_288061:

        if more_types_in_union_288062:
            # Runtime conditional SSA (line 513)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 514):
        
        # Assigning a Name to a Name (line 514):
        # Getting the type of 'ricker' (line 514)
        ricker_288063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 18), 'ricker')
        # Assigning a type to the variable 'wavelet' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'wavelet', ricker_288063)

        if more_types_in_union_288062:
            # SSA join for if statement (line 513)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 516):
    
    # Assigning a Call to a Name (line 516):
    
    # Call to cwt(...): (line 516)
    # Processing the call arguments (line 516)
    # Getting the type of 'vector' (line 516)
    vector_288065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 18), 'vector', False)
    # Getting the type of 'wavelet' (line 516)
    wavelet_288066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 26), 'wavelet', False)
    # Getting the type of 'widths' (line 516)
    widths_288067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 35), 'widths', False)
    # Processing the call keyword arguments (line 516)
    kwargs_288068 = {}
    # Getting the type of 'cwt' (line 516)
    cwt_288064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 14), 'cwt', False)
    # Calling cwt(args, kwargs) (line 516)
    cwt_call_result_288069 = invoke(stypy.reporting.localization.Localization(__file__, 516, 14), cwt_288064, *[vector_288065, wavelet_288066, widths_288067], **kwargs_288068)
    
    # Assigning a type to the variable 'cwt_dat' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'cwt_dat', cwt_call_result_288069)
    
    # Assigning a Call to a Name (line 517):
    
    # Assigning a Call to a Name (line 517):
    
    # Call to _identify_ridge_lines(...): (line 517)
    # Processing the call arguments (line 517)
    # Getting the type of 'cwt_dat' (line 517)
    cwt_dat_288071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 40), 'cwt_dat', False)
    # Getting the type of 'max_distances' (line 517)
    max_distances_288072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 49), 'max_distances', False)
    # Getting the type of 'gap_thresh' (line 517)
    gap_thresh_288073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 64), 'gap_thresh', False)
    # Processing the call keyword arguments (line 517)
    kwargs_288074 = {}
    # Getting the type of '_identify_ridge_lines' (line 517)
    _identify_ridge_lines_288070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 18), '_identify_ridge_lines', False)
    # Calling _identify_ridge_lines(args, kwargs) (line 517)
    _identify_ridge_lines_call_result_288075 = invoke(stypy.reporting.localization.Localization(__file__, 517, 18), _identify_ridge_lines_288070, *[cwt_dat_288071, max_distances_288072, gap_thresh_288073], **kwargs_288074)
    
    # Assigning a type to the variable 'ridge_lines' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'ridge_lines', _identify_ridge_lines_call_result_288075)
    
    # Assigning a Call to a Name (line 518):
    
    # Assigning a Call to a Name (line 518):
    
    # Call to _filter_ridge_lines(...): (line 518)
    # Processing the call arguments (line 518)
    # Getting the type of 'cwt_dat' (line 518)
    cwt_dat_288077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 35), 'cwt_dat', False)
    # Getting the type of 'ridge_lines' (line 518)
    ridge_lines_288078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 44), 'ridge_lines', False)
    # Processing the call keyword arguments (line 518)
    # Getting the type of 'min_length' (line 518)
    min_length_288079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 68), 'min_length', False)
    keyword_288080 = min_length_288079
    # Getting the type of 'min_snr' (line 519)
    min_snr_288081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 43), 'min_snr', False)
    keyword_288082 = min_snr_288081
    # Getting the type of 'noise_perc' (line 519)
    noise_perc_288083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 63), 'noise_perc', False)
    keyword_288084 = noise_perc_288083
    kwargs_288085 = {'min_length': keyword_288080, 'min_snr': keyword_288082, 'noise_perc': keyword_288084}
    # Getting the type of '_filter_ridge_lines' (line 518)
    _filter_ridge_lines_288076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 15), '_filter_ridge_lines', False)
    # Calling _filter_ridge_lines(args, kwargs) (line 518)
    _filter_ridge_lines_call_result_288086 = invoke(stypy.reporting.localization.Localization(__file__, 518, 15), _filter_ridge_lines_288076, *[cwt_dat_288077, ridge_lines_288078], **kwargs_288085)
    
    # Assigning a type to the variable 'filtered' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'filtered', _filter_ridge_lines_call_result_288086)
    
    # Assigning a Call to a Name (line 520):
    
    # Assigning a Call to a Name (line 520):
    
    # Call to asarray(...): (line 520)
    # Processing the call arguments (line 520)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'filtered' (line 520)
    filtered_288096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 44), 'filtered', False)
    comprehension_288097 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 27), filtered_288096)
    # Assigning a type to the variable 'x' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 27), 'x', comprehension_288097)
    
    # Obtaining the type of the subscript
    int_288089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 32), 'int')
    
    # Obtaining the type of the subscript
    int_288090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 29), 'int')
    # Getting the type of 'x' (line 520)
    x_288091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 27), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 520)
    getitem___288092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 27), x_288091, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 520)
    subscript_call_result_288093 = invoke(stypy.reporting.localization.Localization(__file__, 520, 27), getitem___288092, int_288090)
    
    # Obtaining the member '__getitem__' of a type (line 520)
    getitem___288094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 27), subscript_call_result_288093, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 520)
    subscript_call_result_288095 = invoke(stypy.reporting.localization.Localization(__file__, 520, 27), getitem___288094, int_288089)
    
    list_288098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 27), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 27), list_288098, subscript_call_result_288095)
    # Processing the call keyword arguments (line 520)
    kwargs_288099 = {}
    # Getting the type of 'np' (line 520)
    np_288087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 15), 'np', False)
    # Obtaining the member 'asarray' of a type (line 520)
    asarray_288088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 15), np_288087, 'asarray')
    # Calling asarray(args, kwargs) (line 520)
    asarray_call_result_288100 = invoke(stypy.reporting.localization.Localization(__file__, 520, 15), asarray_288088, *[list_288098], **kwargs_288099)
    
    # Assigning a type to the variable 'max_locs' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'max_locs', asarray_call_result_288100)
    
    # Call to sort(...): (line 521)
    # Processing the call keyword arguments (line 521)
    kwargs_288103 = {}
    # Getting the type of 'max_locs' (line 521)
    max_locs_288101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'max_locs', False)
    # Obtaining the member 'sort' of a type (line 521)
    sort_288102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 4), max_locs_288101, 'sort')
    # Calling sort(args, kwargs) (line 521)
    sort_call_result_288104 = invoke(stypy.reporting.localization.Localization(__file__, 521, 4), sort_288102, *[], **kwargs_288103)
    
    # Getting the type of 'max_locs' (line 523)
    max_locs_288105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 11), 'max_locs')
    # Assigning a type to the variable 'stypy_return_type' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'stypy_return_type', max_locs_288105)
    
    # ################# End of 'find_peaks_cwt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_peaks_cwt' in the type store
    # Getting the type of 'stypy_return_type' (line 420)
    stypy_return_type_288106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288106)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_peaks_cwt'
    return stypy_return_type_288106

# Assigning a type to the variable 'find_peaks_cwt' (line 420)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 0), 'find_peaks_cwt', find_peaks_cwt)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
