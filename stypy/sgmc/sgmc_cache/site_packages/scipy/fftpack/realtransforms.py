
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Real spectrum tranforms (DCT, DST, MDCT)
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: 
7: __all__ = ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn']
8: 
9: import numpy as np
10: from scipy.fftpack import _fftpack
11: from scipy.fftpack.basic import _datacopied, _fix_shape, _asfarray
12: 
13: import atexit
14: atexit.register(_fftpack.destroy_ddct1_cache)
15: atexit.register(_fftpack.destroy_ddct2_cache)
16: atexit.register(_fftpack.destroy_dct1_cache)
17: atexit.register(_fftpack.destroy_dct2_cache)
18: 
19: atexit.register(_fftpack.destroy_ddst1_cache)
20: atexit.register(_fftpack.destroy_ddst2_cache)
21: atexit.register(_fftpack.destroy_dst1_cache)
22: atexit.register(_fftpack.destroy_dst2_cache)
23: 
24: 
25: def _init_nd_shape_and_axes(x, shape, axes):
26:     '''Handle shape and axes arguments for dctn, idctn, dstn, idstn.'''
27:     if shape is None:
28:         if axes is None:
29:             shape = x.shape
30:         else:
31:             shape = np.take(x.shape, axes)
32:     shape = tuple(shape)
33:     for dim in shape:
34:         if dim < 1:
35:             raise ValueError("Invalid number of DCT data points "
36:                              "(%s) specified." % (shape,))
37: 
38:     if axes is None:
39:         axes = list(range(-x.ndim, 0))
40:     elif np.isscalar(axes):
41:         axes = [axes, ]
42:     if len(axes) != len(shape):
43:         raise ValueError("when given, axes and shape arguments "
44:                          "have to be of the same length")
45:     if len(np.unique(axes)) != len(axes):
46:         raise ValueError("All axes must be unique.")
47: 
48:     return shape, axes
49: 
50: 
51: def dctn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
52:     '''
53:     Return multidimensional Discrete Cosine Transform along the specified axes.
54: 
55:     Parameters
56:     ----------
57:     x : array_like
58:         The input array.
59:     type : {1, 2, 3}, optional
60:         Type of the DCT (see Notes). Default type is 2.
61:     shape : tuple of ints, optional
62:         The shape of the result.  If both `shape` and `axes` (see below) are
63:         None, `shape` is ``x.shape``; if `shape` is None but `axes` is
64:         not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
65:         If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
66:         If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
67:         length ``shape[i]``.
68:     axes : tuple or None, optional
69:         Axes along which the DCT is computed; the default is over all axes.
70:     norm : {None, 'ortho'}, optional
71:         Normalization mode (see Notes). Default is None.
72:     overwrite_x : bool, optional
73:         If True, the contents of `x` can be destroyed; the default is False.
74: 
75:     Returns
76:     -------
77:     y : ndarray of real
78:         The transformed input array.
79: 
80:     See Also
81:     --------
82:     idctn : Inverse multidimensional DCT
83: 
84:     Notes
85:     -----
86:     For full details of the DCT types and normalization modes, as well as
87:     references, see `dct`.
88: 
89:     Examples
90:     --------
91:     >>> from scipy.fftpack import dctn, idctn
92:     >>> y = np.random.randn(16, 16)
93:     >>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))
94:     True
95: 
96:     '''
97:     x = np.asanyarray(x)
98:     shape, axes = _init_nd_shape_and_axes(x, shape, axes)
99:     for n, ax in zip(shape, axes):
100:         x = dct(x, type=type, n=n, axis=ax, norm=norm, overwrite_x=overwrite_x)
101:     return x
102: 
103: 
104: def idctn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
105:     '''
106:     Return multidimensional Discrete Cosine Transform along the specified axes.
107: 
108:     Parameters
109:     ----------
110:     x : array_like
111:         The input array.
112:     type : {1, 2, 3}, optional
113:         Type of the DCT (see Notes). Default type is 2.
114:     shape : tuple of ints, optional
115:         The shape of the result.  If both `shape` and `axes` (see below) are
116:         None, `shape` is ``x.shape``; if `shape` is None but `axes` is
117:         not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
118:         If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
119:         If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
120:         length ``shape[i]``.
121:     axes : tuple or None, optional
122:         Axes along which the IDCT is computed; the default is over all axes.
123:     norm : {None, 'ortho'}, optional
124:         Normalization mode (see Notes). Default is None.
125:     overwrite_x : bool, optional
126:         If True, the contents of `x` can be destroyed; the default is False.
127: 
128:     Returns
129:     -------
130:     y : ndarray of real
131:         The transformed input array.
132: 
133:     See Also
134:     --------
135:     dctn : multidimensional DCT
136: 
137:     Notes
138:     -----
139:     For full details of the IDCT types and normalization modes, as well as
140:     references, see `idct`.
141: 
142:     Examples
143:     --------
144:     >>> from scipy.fftpack import dctn, idctn
145:     >>> y = np.random.randn(16, 16)
146:     >>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))
147:     True
148:     '''
149:     x = np.asanyarray(x)
150:     shape, axes = _init_nd_shape_and_axes(x, shape, axes)
151:     for n, ax in zip(shape, axes):
152:         x = idct(x, type=type, n=n, axis=ax, norm=norm,
153:                  overwrite_x=overwrite_x)
154:     return x
155: 
156: 
157: def dstn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
158:     '''
159:     Return multidimensional Discrete Sine Transform along the specified axes.
160: 
161:     Parameters
162:     ----------
163:     x : array_like
164:         The input array.
165:     type : {1, 2, 3}, optional
166:         Type of the DCT (see Notes). Default type is 2.
167:     shape : tuple of ints, optional
168:         The shape of the result.  If both `shape` and `axes` (see below) are
169:         None, `shape` is ``x.shape``; if `shape` is None but `axes` is
170:         not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
171:         If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
172:         If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
173:         length ``shape[i]``.
174:     axes : tuple or None, optional
175:         Axes along which the DCT is computed; the default is over all axes.
176:     norm : {None, 'ortho'}, optional
177:         Normalization mode (see Notes). Default is None.
178:     overwrite_x : bool, optional
179:         If True, the contents of `x` can be destroyed; the default is False.
180: 
181:     Returns
182:     -------
183:     y : ndarray of real
184:         The transformed input array.
185: 
186:     See Also
187:     --------
188:     idstn : Inverse multidimensional DST
189: 
190:     Notes
191:     -----
192:     For full details of the DST types and normalization modes, as well as
193:     references, see `dst`.
194: 
195:     Examples
196:     --------
197:     >>> from scipy.fftpack import dstn, idstn
198:     >>> y = np.random.randn(16, 16)
199:     >>> np.allclose(y, idstn(dstn(y, norm='ortho'), norm='ortho'))
200:     True
201: 
202:     '''
203:     x = np.asanyarray(x)
204:     shape, axes = _init_nd_shape_and_axes(x, shape, axes)
205:     for n, ax in zip(shape, axes):
206:         x = dst(x, type=type, n=n, axis=ax, norm=norm, overwrite_x=overwrite_x)
207:     return x
208: 
209: 
210: def idstn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
211:     '''
212:     Return multidimensional Discrete Sine Transform along the specified axes.
213: 
214:     Parameters
215:     ----------
216:     x : array_like
217:         The input array.
218:     type : {1, 2, 3}, optional
219:         Type of the DCT (see Notes). Default type is 2.
220:     shape : tuple of ints, optional
221:         The shape of the result.  If both `shape` and `axes` (see below) are
222:         None, `shape` is ``x.shape``; if `shape` is None but `axes` is
223:         not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
224:         If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
225:         If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
226:         length ``shape[i]``.
227:     axes : tuple or None, optional
228:         Axes along which the IDCT is computed; the default is over all axes.
229:     norm : {None, 'ortho'}, optional
230:         Normalization mode (see Notes). Default is None.
231:     overwrite_x : bool, optional
232:         If True, the contents of `x` can be destroyed; the default is False.
233: 
234:     Returns
235:     -------
236:     y : ndarray of real
237:         The transformed input array.
238: 
239:     See Also
240:     --------
241:     dctn : multidimensional DST
242: 
243:     Notes
244:     -----
245:     For full details of the IDST types and normalization modes, as well as
246:     references, see `idst`.
247: 
248:     Examples
249:     --------
250:     >>> from scipy.fftpack import dstn, idstn
251:     >>> y = np.random.randn(16, 16)
252:     >>> np.allclose(y, idstn(dstn(y, norm='ortho'), norm='ortho'))
253:     True
254:     '''
255:     x = np.asanyarray(x)
256:     shape, axes = _init_nd_shape_and_axes(x, shape, axes)
257:     for n, ax in zip(shape, axes):
258:         x = idst(x, type=type, n=n, axis=ax, norm=norm,
259:                  overwrite_x=overwrite_x)
260:     return x
261: 
262: 
263: def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
264:     '''
265:     Return the Discrete Cosine Transform of arbitrary type sequence x.
266: 
267:     Parameters
268:     ----------
269:     x : array_like
270:         The input array.
271:     type : {1, 2, 3}, optional
272:         Type of the DCT (see Notes). Default type is 2.
273:     n : int, optional
274:         Length of the transform.  If ``n < x.shape[axis]``, `x` is
275:         truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
276:         default results in ``n = x.shape[axis]``.
277:     axis : int, optional
278:         Axis along which the dct is computed; the default is over the
279:         last axis (i.e., ``axis=-1``).
280:     norm : {None, 'ortho'}, optional
281:         Normalization mode (see Notes). Default is None.
282:     overwrite_x : bool, optional
283:         If True, the contents of `x` can be destroyed; the default is False.
284: 
285:     Returns
286:     -------
287:     y : ndarray of real
288:         The transformed input array.
289: 
290:     See Also
291:     --------
292:     idct : Inverse DCT
293: 
294:     Notes
295:     -----
296:     For a single dimension array ``x``, ``dct(x, norm='ortho')`` is equal to
297:     MATLAB ``dct(x)``.
298: 
299:     There are theoretically 8 types of the DCT, only the first 3 types are
300:     implemented in scipy. 'The' DCT generally refers to DCT type 2, and 'the'
301:     Inverse DCT generally refers to DCT type 3.
302: 
303:     **Type I**
304: 
305:     There are several definitions of the DCT-I; we use the following
306:     (for ``norm=None``)::
307: 
308:                                          N-2
309:       y[k] = x[0] + (-1)**k x[N-1] + 2 * sum x[n]*cos(pi*k*n/(N-1))
310:                                          n=1
311: 
312:     Only None is supported as normalization mode for DCT-I. Note also that the
313:     DCT-I is only supported for input size > 1
314: 
315:     **Type II**
316: 
317:     There are several definitions of the DCT-II; we use the following
318:     (for ``norm=None``)::
319: 
320: 
321:                 N-1
322:       y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
323:                 n=0
324: 
325:     If ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor `f`::
326: 
327:       f = sqrt(1/(4*N)) if k = 0,
328:       f = sqrt(1/(2*N)) otherwise.
329: 
330:     Which makes the corresponding matrix of coefficients orthonormal
331:     (``OO' = Id``).
332: 
333:     **Type III**
334: 
335:     There are several definitions, we use the following
336:     (for ``norm=None``)::
337: 
338:                         N-1
339:       y[k] = x[0] + 2 * sum x[n]*cos(pi*(k+0.5)*n/N), 0 <= k < N.
340:                         n=1
341: 
342:     or, for ``norm='ortho'`` and 0 <= k < N::
343: 
344:                                           N-1
345:       y[k] = x[0] / sqrt(N) + sqrt(2/N) * sum x[n]*cos(pi*(k+0.5)*n/N)
346:                                           n=1
347: 
348:     The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up
349:     to a factor `2N`. The orthonormalized DCT-III is exactly the inverse of
350:     the orthonormalized DCT-II.
351: 
352:     References
353:     ----------
354:     .. [1] 'A Fast Cosine Transform in One and Two Dimensions', by J.
355:            Makhoul, `IEEE Transactions on acoustics, speech and signal
356:            processing` vol. 28(1), pp. 27-34,
357:            http://dx.doi.org/10.1109/TASSP.1980.1163351 (1980).
358:     .. [2] Wikipedia, "Discrete cosine transform",
359:            http://en.wikipedia.org/wiki/Discrete_cosine_transform
360: 
361:     Examples
362:     --------
363:     The Type 1 DCT is equivalent to the FFT (though faster) for real,
364:     even-symmetrical inputs.  The output is also real and even-symmetrical.
365:     Half of the FFT input is used to generate half of the FFT output:
366: 
367:     >>> from scipy.fftpack import fft, dct
368:     >>> fft(np.array([4., 3., 5., 10., 5., 3.])).real
369:     array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])
370:     >>> dct(np.array([4., 3., 5., 10.]), 1)
371:     array([ 30.,  -8.,   6.,  -2.])
372: 
373:     '''
374:     if type == 1 and norm is not None:
375:         raise NotImplementedError(
376:               "Orthonormalization not yet supported for DCT-I")
377:     return _dct(x, type, n, axis, normalize=norm, overwrite_x=overwrite_x)
378: 
379: 
380: def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
381:     '''
382:     Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.
383: 
384:     Parameters
385:     ----------
386:     x : array_like
387:         The input array.
388:     type : {1, 2, 3}, optional
389:         Type of the DCT (see Notes). Default type is 2.
390:     n : int, optional
391:         Length of the transform.  If ``n < x.shape[axis]``, `x` is
392:         truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
393:         default results in ``n = x.shape[axis]``.
394:     axis : int, optional
395:         Axis along which the idct is computed; the default is over the
396:         last axis (i.e., ``axis=-1``).
397:     norm : {None, 'ortho'}, optional
398:         Normalization mode (see Notes). Default is None.
399:     overwrite_x : bool, optional
400:         If True, the contents of `x` can be destroyed; the default is False.
401: 
402:     Returns
403:     -------
404:     idct : ndarray of real
405:         The transformed input array.
406: 
407:     See Also
408:     --------
409:     dct : Forward DCT
410: 
411:     Notes
412:     -----
413:     For a single dimension array `x`, ``idct(x, norm='ortho')`` is equal to
414:     MATLAB ``idct(x)``.
415: 
416:     'The' IDCT is the IDCT of type 2, which is the same as DCT of type 3.
417: 
418:     IDCT of type 1 is the DCT of type 1, IDCT of type 2 is the DCT of type
419:     3, and IDCT of type 3 is the DCT of type 2. For the definition of these
420:     types, see `dct`.
421: 
422:     Examples
423:     --------
424:     The Type 1 DCT is equivalent to the DFT for real, even-symmetrical
425:     inputs.  The output is also real and even-symmetrical.  Half of the IFFT
426:     input is used to generate half of the IFFT output:
427: 
428:     >>> from scipy.fftpack import ifft, idct
429:     >>> ifft(np.array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])).real
430:     array([  4.,   3.,   5.,  10.,   5.,   3.])
431:     >>> idct(np.array([ 30.,  -8.,   6.,  -2.]), 1) / 6
432:     array([  4.,   3.,   5.,  10.])
433: 
434:     '''
435:     if type == 1 and norm is not None:
436:         raise NotImplementedError(
437:               "Orthonormalization not yet supported for IDCT-I")
438:     # Inverse/forward type table
439:     _TP = {1:1, 2:3, 3:2}
440:     return _dct(x, _TP[type], n, axis, normalize=norm, overwrite_x=overwrite_x)
441: 
442: 
443: def _get_dct_fun(type, dtype):
444:     try:
445:         name = {'float64':'ddct%d', 'float32':'dct%d'}[dtype.name]
446:     except KeyError:
447:         raise ValueError("dtype %s not supported" % dtype)
448:     try:
449:         f = getattr(_fftpack, name % type)
450:     except AttributeError as e:
451:         raise ValueError(str(e) + ". Type %d not understood" % type)
452:     return f
453: 
454: 
455: def _get_norm_mode(normalize):
456:     try:
457:         nm = {None:0, 'ortho':1}[normalize]
458:     except KeyError:
459:         raise ValueError("Unknown normalize mode %s" % normalize)
460:     return nm
461: 
462: 
463: def __fix_shape(x, n, axis, dct_or_dst):
464:     tmp = _asfarray(x)
465:     copy_made = _datacopied(tmp, x)
466:     if n is None:
467:         n = tmp.shape[axis]
468:     elif n != tmp.shape[axis]:
469:         tmp, copy_made2 = _fix_shape(tmp, n, axis)
470:         copy_made = copy_made or copy_made2
471:     if n < 1:
472:         raise ValueError("Invalid number of %s data points "
473:                          "(%d) specified." % (dct_or_dst, n))
474:     return tmp, n, copy_made
475: 
476: 
477: def _raw_dct(x0, type, n, axis, nm, overwrite_x):
478:     f = _get_dct_fun(type, x0.dtype)
479:     return _eval_fun(f, x0, n, axis, nm, overwrite_x)
480: 
481: 
482: def _raw_dst(x0, type, n, axis, nm, overwrite_x):
483:     f = _get_dst_fun(type, x0.dtype)
484:     return _eval_fun(f, x0, n, axis, nm, overwrite_x)
485: 
486: 
487: def _eval_fun(f, tmp, n, axis, nm, overwrite_x):
488:     if axis == -1 or axis == len(tmp.shape) - 1:
489:         return f(tmp, n, nm, overwrite_x)
490: 
491:     tmp = np.swapaxes(tmp, axis, -1)
492:     tmp = f(tmp, n, nm, overwrite_x)
493:     return np.swapaxes(tmp, axis, -1)
494: 
495: 
496: def _dct(x, type, n=None, axis=-1, overwrite_x=False, normalize=None):
497:     '''
498:     Return Discrete Cosine Transform of arbitrary type sequence x.
499: 
500:     Parameters
501:     ----------
502:     x : array_like
503:         input array.
504:     n : int, optional
505:         Length of the transform.  If ``n < x.shape[axis]``, `x` is
506:         truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
507:         default results in ``n = x.shape[axis]``.
508:     axis : int, optional
509:         Axis along which the dct is computed; the default is over the
510:         last axis (i.e., ``axis=-1``).
511:     overwrite_x : bool, optional
512:         If True, the contents of `x` can be destroyed; the default is False.
513: 
514:     Returns
515:     -------
516:     z : ndarray
517: 
518:     '''
519:     x0, n, copy_made = __fix_shape(x, n, axis, 'DCT')
520:     if type == 1 and n < 2:
521:         raise ValueError("DCT-I is not defined for size < 2")
522:     overwrite_x = overwrite_x or copy_made
523:     nm = _get_norm_mode(normalize)
524:     if np.iscomplexobj(x0):
525:         return (_raw_dct(x0.real, type, n, axis, nm, overwrite_x) + 1j *
526:                 _raw_dct(x0.imag, type, n, axis, nm, overwrite_x))
527:     else:
528:         return _raw_dct(x0, type, n, axis, nm, overwrite_x)
529: 
530: 
531: def dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
532:     '''
533:     Return the Discrete Sine Transform of arbitrary type sequence x.
534: 
535:     Parameters
536:     ----------
537:     x : array_like
538:         The input array.
539:     type : {1, 2, 3}, optional
540:         Type of the DST (see Notes). Default type is 2.
541:     n : int, optional
542:         Length of the transform.  If ``n < x.shape[axis]``, `x` is
543:         truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
544:         default results in ``n = x.shape[axis]``.
545:     axis : int, optional
546:         Axis along which the dst is computed; the default is over the
547:         last axis (i.e., ``axis=-1``).
548:     norm : {None, 'ortho'}, optional
549:         Normalization mode (see Notes). Default is None.
550:     overwrite_x : bool, optional
551:         If True, the contents of `x` can be destroyed; the default is False.
552: 
553:     Returns
554:     -------
555:     dst : ndarray of reals
556:         The transformed input array.
557: 
558:     See Also
559:     --------
560:     idst : Inverse DST
561: 
562:     Notes
563:     -----
564:     For a single dimension array ``x``.
565: 
566:     There are theoretically 8 types of the DST for different combinations of
567:     even/odd boundary conditions and boundary off sets [1]_, only the first
568:     3 types are implemented in scipy.
569: 
570:     **Type I**
571: 
572:     There are several definitions of the DST-I; we use the following
573:     for ``norm=None``.  DST-I assumes the input is odd around n=-1 and n=N. ::
574: 
575:                  N-1
576:       y[k] = 2 * sum x[n]*sin(pi*(k+1)*(n+1)/(N+1))
577:                  n=0
578: 
579:     Only None is supported as normalization mode for DCT-I. Note also that the
580:     DCT-I is only supported for input size > 1
581:     The (unnormalized) DCT-I is its own inverse, up to a factor `2(N+1)`.
582: 
583:     **Type II**
584: 
585:     There are several definitions of the DST-II; we use the following
586:     for ``norm=None``.  DST-II assumes the input is odd around n=-1/2 and
587:     n=N-1/2; the output is odd around k=-1 and even around k=N-1 ::
588: 
589:                 N-1
590:       y[k] = 2* sum x[n]*sin(pi*(k+1)*(n+0.5)/N), 0 <= k < N.
591:                 n=0
592: 
593:     if ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor `f` ::
594: 
595:         f = sqrt(1/(4*N)) if k == 0
596:         f = sqrt(1/(2*N)) otherwise.
597: 
598:     **Type III**
599: 
600:     There are several definitions of the DST-III, we use the following
601:     (for ``norm=None``).  DST-III assumes the input is odd around n=-1
602:     and even around n=N-1 ::
603: 
604:                                  N-2
605:       y[k] = x[N-1]*(-1)**k + 2* sum x[n]*sin(pi*(k+0.5)*(n+1)/N), 0 <= k < N.
606:                                  n=0
607: 
608:     The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up
609:     to a factor `2N`.  The orthonormalized DST-III is exactly the inverse of
610:     the orthonormalized DST-II.
611: 
612:     .. versionadded:: 0.11.0
613: 
614:     References
615:     ----------
616:     .. [1] Wikipedia, "Discrete sine transform",
617:            http://en.wikipedia.org/wiki/Discrete_sine_transform
618: 
619:     '''
620:     if type == 1 and norm is not None:
621:         raise NotImplementedError(
622:               "Orthonormalization not yet supported for IDCT-I")
623:     return _dst(x, type, n, axis, normalize=norm, overwrite_x=overwrite_x)
624: 
625: 
626: def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
627:     '''
628:     Return the Inverse Discrete Sine Transform of an arbitrary type sequence.
629: 
630:     Parameters
631:     ----------
632:     x : array_like
633:         The input array.
634:     type : {1, 2, 3}, optional
635:         Type of the DST (see Notes). Default type is 2.
636:     n : int, optional
637:         Length of the transform.  If ``n < x.shape[axis]``, `x` is
638:         truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
639:         default results in ``n = x.shape[axis]``.
640:     axis : int, optional
641:         Axis along which the idst is computed; the default is over the
642:         last axis (i.e., ``axis=-1``).
643:     norm : {None, 'ortho'}, optional
644:         Normalization mode (see Notes). Default is None.
645:     overwrite_x : bool, optional
646:         If True, the contents of `x` can be destroyed; the default is False.
647: 
648:     Returns
649:     -------
650:     idst : ndarray of real
651:         The transformed input array.
652: 
653:     See Also
654:     --------
655:     dst : Forward DST
656: 
657:     Notes
658:     -----
659:     'The' IDST is the IDST of type 2, which is the same as DST of type 3.
660: 
661:     IDST of type 1 is the DST of type 1, IDST of type 2 is the DST of type
662:     3, and IDST of type 3 is the DST of type 2. For the definition of these
663:     types, see `dst`.
664: 
665:     .. versionadded:: 0.11.0
666: 
667:     '''
668:     if type == 1 and norm is not None:
669:         raise NotImplementedError(
670:               "Orthonormalization not yet supported for IDCT-I")
671:     # Inverse/forward type table
672:     _TP = {1:1, 2:3, 3:2}
673:     return _dst(x, _TP[type], n, axis, normalize=norm, overwrite_x=overwrite_x)
674: 
675: 
676: def _get_dst_fun(type, dtype):
677:     try:
678:         name = {'float64':'ddst%d', 'float32':'dst%d'}[dtype.name]
679:     except KeyError:
680:         raise ValueError("dtype %s not supported" % dtype)
681:     try:
682:         f = getattr(_fftpack, name % type)
683:     except AttributeError as e:
684:         raise ValueError(str(e) + ". Type %d not understood" % type)
685:     return f
686: 
687: 
688: def _dst(x, type, n=None, axis=-1, overwrite_x=False, normalize=None):
689:     '''
690:     Return Discrete Sine Transform of arbitrary type sequence x.
691: 
692:     Parameters
693:     ----------
694:     x : array_like
695:         input array.
696:     n : int, optional
697:         Length of the transform.
698:     axis : int, optional
699:         Axis along which the dst is computed. (default=-1)
700:     overwrite_x : bool, optional
701:         If True the contents of x can be destroyed. (default=False)
702: 
703:     Returns
704:     -------
705:     z : real ndarray
706: 
707:     '''
708:     x0, n, copy_made = __fix_shape(x, n, axis, 'DST')
709:     if type == 1 and n < 2:
710:         raise ValueError("DST-I is not defined for size < 2")
711:     overwrite_x = overwrite_x or copy_made
712:     nm = _get_norm_mode(normalize)
713:     if np.iscomplexobj(x0):
714:         return (_raw_dst(x0.real, type, n, axis, nm, overwrite_x) + 1j *
715:                 _raw_dst(x0.imag, type, n, axis, nm, overwrite_x))
716:     else:
717:         return _raw_dst(x0, type, n, axis, nm, overwrite_x)
718: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_17607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nReal spectrum tranforms (DCT, DST, MDCT)\n')

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn']
module_type_store.set_exportable_members(['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_17608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_17609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'dct')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_17608, str_17609)
# Adding element type (line 7)
str_17610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 18), 'str', 'idct')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_17608, str_17610)
# Adding element type (line 7)
str_17611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'str', 'dst')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_17608, str_17611)
# Adding element type (line 7)
str_17612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 33), 'str', 'idst')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_17608, str_17612)
# Adding element type (line 7)
str_17613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 41), 'str', 'dctn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_17608, str_17613)
# Adding element type (line 7)
str_17614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 49), 'str', 'idctn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_17608, str_17614)
# Adding element type (line 7)
str_17615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 58), 'str', 'dstn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_17608, str_17615)
# Adding element type (line 7)
str_17616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 66), 'str', 'idstn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_17608, str_17616)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_17608)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_17617 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_17617) is not StypyTypeError):

    if (import_17617 != 'pyd_module'):
        __import__(import_17617)
        sys_modules_17618 = sys.modules[import_17617]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_17618.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_17617)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.fftpack import _fftpack' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_17619 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.fftpack')

if (type(import_17619) is not StypyTypeError):

    if (import_17619 != 'pyd_module'):
        __import__(import_17619)
        sys_modules_17620 = sys.modules[import_17619]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.fftpack', sys_modules_17620.module_type_store, module_type_store, ['_fftpack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_17620, sys_modules_17620.module_type_store, module_type_store)
    else:
        from scipy.fftpack import _fftpack

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.fftpack', None, module_type_store, ['_fftpack'], [_fftpack])

else:
    # Assigning a type to the variable 'scipy.fftpack' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.fftpack', import_17619)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.fftpack.basic import _datacopied, _fix_shape, _asfarray' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_17621 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.fftpack.basic')

if (type(import_17621) is not StypyTypeError):

    if (import_17621 != 'pyd_module'):
        __import__(import_17621)
        sys_modules_17622 = sys.modules[import_17621]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.fftpack.basic', sys_modules_17622.module_type_store, module_type_store, ['_datacopied', '_fix_shape', '_asfarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_17622, sys_modules_17622.module_type_store, module_type_store)
    else:
        from scipy.fftpack.basic import _datacopied, _fix_shape, _asfarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.fftpack.basic', None, module_type_store, ['_datacopied', '_fix_shape', '_asfarray'], [_datacopied, _fix_shape, _asfarray])

else:
    # Assigning a type to the variable 'scipy.fftpack.basic' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.fftpack.basic', import_17621)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import atexit' statement (line 13)
import atexit

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'atexit', atexit, module_type_store)


# Call to register(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of '_fftpack' (line 14)
_fftpack_17625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), '_fftpack', False)
# Obtaining the member 'destroy_ddct1_cache' of a type (line 14)
destroy_ddct1_cache_17626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 16), _fftpack_17625, 'destroy_ddct1_cache')
# Processing the call keyword arguments (line 14)
kwargs_17627 = {}
# Getting the type of 'atexit' (line 14)
atexit_17623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 14)
register_17624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 0), atexit_17623, 'register')
# Calling register(args, kwargs) (line 14)
register_call_result_17628 = invoke(stypy.reporting.localization.Localization(__file__, 14, 0), register_17624, *[destroy_ddct1_cache_17626], **kwargs_17627)


# Call to register(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of '_fftpack' (line 15)
_fftpack_17631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), '_fftpack', False)
# Obtaining the member 'destroy_ddct2_cache' of a type (line 15)
destroy_ddct2_cache_17632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 16), _fftpack_17631, 'destroy_ddct2_cache')
# Processing the call keyword arguments (line 15)
kwargs_17633 = {}
# Getting the type of 'atexit' (line 15)
atexit_17629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 15)
register_17630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 0), atexit_17629, 'register')
# Calling register(args, kwargs) (line 15)
register_call_result_17634 = invoke(stypy.reporting.localization.Localization(__file__, 15, 0), register_17630, *[destroy_ddct2_cache_17632], **kwargs_17633)


# Call to register(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of '_fftpack' (line 16)
_fftpack_17637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), '_fftpack', False)
# Obtaining the member 'destroy_dct1_cache' of a type (line 16)
destroy_dct1_cache_17638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 16), _fftpack_17637, 'destroy_dct1_cache')
# Processing the call keyword arguments (line 16)
kwargs_17639 = {}
# Getting the type of 'atexit' (line 16)
atexit_17635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 16)
register_17636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 0), atexit_17635, 'register')
# Calling register(args, kwargs) (line 16)
register_call_result_17640 = invoke(stypy.reporting.localization.Localization(__file__, 16, 0), register_17636, *[destroy_dct1_cache_17638], **kwargs_17639)


# Call to register(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of '_fftpack' (line 17)
_fftpack_17643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), '_fftpack', False)
# Obtaining the member 'destroy_dct2_cache' of a type (line 17)
destroy_dct2_cache_17644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 16), _fftpack_17643, 'destroy_dct2_cache')
# Processing the call keyword arguments (line 17)
kwargs_17645 = {}
# Getting the type of 'atexit' (line 17)
atexit_17641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 17)
register_17642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 0), atexit_17641, 'register')
# Calling register(args, kwargs) (line 17)
register_call_result_17646 = invoke(stypy.reporting.localization.Localization(__file__, 17, 0), register_17642, *[destroy_dct2_cache_17644], **kwargs_17645)


# Call to register(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of '_fftpack' (line 19)
_fftpack_17649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), '_fftpack', False)
# Obtaining the member 'destroy_ddst1_cache' of a type (line 19)
destroy_ddst1_cache_17650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 16), _fftpack_17649, 'destroy_ddst1_cache')
# Processing the call keyword arguments (line 19)
kwargs_17651 = {}
# Getting the type of 'atexit' (line 19)
atexit_17647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 19)
register_17648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 0), atexit_17647, 'register')
# Calling register(args, kwargs) (line 19)
register_call_result_17652 = invoke(stypy.reporting.localization.Localization(__file__, 19, 0), register_17648, *[destroy_ddst1_cache_17650], **kwargs_17651)


# Call to register(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of '_fftpack' (line 20)
_fftpack_17655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), '_fftpack', False)
# Obtaining the member 'destroy_ddst2_cache' of a type (line 20)
destroy_ddst2_cache_17656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), _fftpack_17655, 'destroy_ddst2_cache')
# Processing the call keyword arguments (line 20)
kwargs_17657 = {}
# Getting the type of 'atexit' (line 20)
atexit_17653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 20)
register_17654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 0), atexit_17653, 'register')
# Calling register(args, kwargs) (line 20)
register_call_result_17658 = invoke(stypy.reporting.localization.Localization(__file__, 20, 0), register_17654, *[destroy_ddst2_cache_17656], **kwargs_17657)


# Call to register(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of '_fftpack' (line 21)
_fftpack_17661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), '_fftpack', False)
# Obtaining the member 'destroy_dst1_cache' of a type (line 21)
destroy_dst1_cache_17662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), _fftpack_17661, 'destroy_dst1_cache')
# Processing the call keyword arguments (line 21)
kwargs_17663 = {}
# Getting the type of 'atexit' (line 21)
atexit_17659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 21)
register_17660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 0), atexit_17659, 'register')
# Calling register(args, kwargs) (line 21)
register_call_result_17664 = invoke(stypy.reporting.localization.Localization(__file__, 21, 0), register_17660, *[destroy_dst1_cache_17662], **kwargs_17663)


# Call to register(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of '_fftpack' (line 22)
_fftpack_17667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), '_fftpack', False)
# Obtaining the member 'destroy_dst2_cache' of a type (line 22)
destroy_dst2_cache_17668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), _fftpack_17667, 'destroy_dst2_cache')
# Processing the call keyword arguments (line 22)
kwargs_17669 = {}
# Getting the type of 'atexit' (line 22)
atexit_17665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 22)
register_17666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 0), atexit_17665, 'register')
# Calling register(args, kwargs) (line 22)
register_call_result_17670 = invoke(stypy.reporting.localization.Localization(__file__, 22, 0), register_17666, *[destroy_dst2_cache_17668], **kwargs_17669)


@norecursion
def _init_nd_shape_and_axes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_init_nd_shape_and_axes'
    module_type_store = module_type_store.open_function_context('_init_nd_shape_and_axes', 25, 0, False)
    
    # Passed parameters checking function
    _init_nd_shape_and_axes.stypy_localization = localization
    _init_nd_shape_and_axes.stypy_type_of_self = None
    _init_nd_shape_and_axes.stypy_type_store = module_type_store
    _init_nd_shape_and_axes.stypy_function_name = '_init_nd_shape_and_axes'
    _init_nd_shape_and_axes.stypy_param_names_list = ['x', 'shape', 'axes']
    _init_nd_shape_and_axes.stypy_varargs_param_name = None
    _init_nd_shape_and_axes.stypy_kwargs_param_name = None
    _init_nd_shape_and_axes.stypy_call_defaults = defaults
    _init_nd_shape_and_axes.stypy_call_varargs = varargs
    _init_nd_shape_and_axes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_init_nd_shape_and_axes', ['x', 'shape', 'axes'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_init_nd_shape_and_axes', localization, ['x', 'shape', 'axes'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_init_nd_shape_and_axes(...)' code ##################

    str_17671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'str', 'Handle shape and axes arguments for dctn, idctn, dstn, idstn.')
    
    # Type idiom detected: calculating its left and rigth part (line 27)
    # Getting the type of 'shape' (line 27)
    shape_17672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), 'shape')
    # Getting the type of 'None' (line 27)
    None_17673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'None')
    
    (may_be_17674, more_types_in_union_17675) = may_be_none(shape_17672, None_17673)

    if may_be_17674:

        if more_types_in_union_17675:
            # Runtime conditional SSA (line 27)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 28)
        # Getting the type of 'axes' (line 28)
        axes_17676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'axes')
        # Getting the type of 'None' (line 28)
        None_17677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'None')
        
        (may_be_17678, more_types_in_union_17679) = may_be_none(axes_17676, None_17677)

        if may_be_17678:

            if more_types_in_union_17679:
                # Runtime conditional SSA (line 28)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 29):
            
            # Assigning a Attribute to a Name (line 29):
            # Getting the type of 'x' (line 29)
            x_17680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'x')
            # Obtaining the member 'shape' of a type (line 29)
            shape_17681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 20), x_17680, 'shape')
            # Assigning a type to the variable 'shape' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'shape', shape_17681)

            if more_types_in_union_17679:
                # Runtime conditional SSA for else branch (line 28)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_17678) or more_types_in_union_17679):
            
            # Assigning a Call to a Name (line 31):
            
            # Assigning a Call to a Name (line 31):
            
            # Call to take(...): (line 31)
            # Processing the call arguments (line 31)
            # Getting the type of 'x' (line 31)
            x_17684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 28), 'x', False)
            # Obtaining the member 'shape' of a type (line 31)
            shape_17685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 28), x_17684, 'shape')
            # Getting the type of 'axes' (line 31)
            axes_17686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'axes', False)
            # Processing the call keyword arguments (line 31)
            kwargs_17687 = {}
            # Getting the type of 'np' (line 31)
            np_17682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'np', False)
            # Obtaining the member 'take' of a type (line 31)
            take_17683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), np_17682, 'take')
            # Calling take(args, kwargs) (line 31)
            take_call_result_17688 = invoke(stypy.reporting.localization.Localization(__file__, 31, 20), take_17683, *[shape_17685, axes_17686], **kwargs_17687)
            
            # Assigning a type to the variable 'shape' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'shape', take_call_result_17688)

            if (may_be_17678 and more_types_in_union_17679):
                # SSA join for if statement (line 28)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_17675:
            # SSA join for if statement (line 27)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to tuple(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'shape' (line 32)
    shape_17690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 18), 'shape', False)
    # Processing the call keyword arguments (line 32)
    kwargs_17691 = {}
    # Getting the type of 'tuple' (line 32)
    tuple_17689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'tuple', False)
    # Calling tuple(args, kwargs) (line 32)
    tuple_call_result_17692 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), tuple_17689, *[shape_17690], **kwargs_17691)
    
    # Assigning a type to the variable 'shape' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'shape', tuple_call_result_17692)
    
    # Getting the type of 'shape' (line 33)
    shape_17693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'shape')
    # Testing the type of a for loop iterable (line 33)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 33, 4), shape_17693)
    # Getting the type of the for loop variable (line 33)
    for_loop_var_17694 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 33, 4), shape_17693)
    # Assigning a type to the variable 'dim' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'dim', for_loop_var_17694)
    # SSA begins for a for statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'dim' (line 34)
    dim_17695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'dim')
    int_17696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'int')
    # Applying the binary operator '<' (line 34)
    result_lt_17697 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 11), '<', dim_17695, int_17696)
    
    # Testing the type of an if condition (line 34)
    if_condition_17698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 8), result_lt_17697)
    # Assigning a type to the variable 'if_condition_17698' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'if_condition_17698', if_condition_17698)
    # SSA begins for if statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 35)
    # Processing the call arguments (line 35)
    str_17700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 29), 'str', 'Invalid number of DCT data points (%s) specified.')
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_17701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    # Getting the type of 'shape' (line 36)
    shape_17702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 50), 'shape', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 50), tuple_17701, shape_17702)
    
    # Applying the binary operator '%' (line 35)
    result_mod_17703 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 29), '%', str_17700, tuple_17701)
    
    # Processing the call keyword arguments (line 35)
    kwargs_17704 = {}
    # Getting the type of 'ValueError' (line 35)
    ValueError_17699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 35)
    ValueError_call_result_17705 = invoke(stypy.reporting.localization.Localization(__file__, 35, 18), ValueError_17699, *[result_mod_17703], **kwargs_17704)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 35, 12), ValueError_call_result_17705, 'raise parameter', BaseException)
    # SSA join for if statement (line 34)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 38)
    # Getting the type of 'axes' (line 38)
    axes_17706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 7), 'axes')
    # Getting the type of 'None' (line 38)
    None_17707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'None')
    
    (may_be_17708, more_types_in_union_17709) = may_be_none(axes_17706, None_17707)

    if may_be_17708:

        if more_types_in_union_17709:
            # Runtime conditional SSA (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to list(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to range(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Getting the type of 'x' (line 39)
        x_17712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'x', False)
        # Obtaining the member 'ndim' of a type (line 39)
        ndim_17713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 27), x_17712, 'ndim')
        # Applying the 'usub' unary operator (line 39)
        result___neg___17714 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 26), 'usub', ndim_17713)
        
        int_17715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 35), 'int')
        # Processing the call keyword arguments (line 39)
        kwargs_17716 = {}
        # Getting the type of 'range' (line 39)
        range_17711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'range', False)
        # Calling range(args, kwargs) (line 39)
        range_call_result_17717 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), range_17711, *[result___neg___17714, int_17715], **kwargs_17716)
        
        # Processing the call keyword arguments (line 39)
        kwargs_17718 = {}
        # Getting the type of 'list' (line 39)
        list_17710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'list', False)
        # Calling list(args, kwargs) (line 39)
        list_call_result_17719 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), list_17710, *[range_call_result_17717], **kwargs_17718)
        
        # Assigning a type to the variable 'axes' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'axes', list_call_result_17719)

        if more_types_in_union_17709:
            # Runtime conditional SSA for else branch (line 38)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_17708) or more_types_in_union_17709):
        
        
        # Call to isscalar(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'axes' (line 40)
        axes_17722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), 'axes', False)
        # Processing the call keyword arguments (line 40)
        kwargs_17723 = {}
        # Getting the type of 'np' (line 40)
        np_17720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 9), 'np', False)
        # Obtaining the member 'isscalar' of a type (line 40)
        isscalar_17721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 9), np_17720, 'isscalar')
        # Calling isscalar(args, kwargs) (line 40)
        isscalar_call_result_17724 = invoke(stypy.reporting.localization.Localization(__file__, 40, 9), isscalar_17721, *[axes_17722], **kwargs_17723)
        
        # Testing the type of an if condition (line 40)
        if_condition_17725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 9), isscalar_call_result_17724)
        # Assigning a type to the variable 'if_condition_17725' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 9), 'if_condition_17725', if_condition_17725)
        # SSA begins for if statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 41):
        
        # Assigning a List to a Name (line 41):
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_17726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        # Getting the type of 'axes' (line 41)
        axes_17727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'axes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 15), list_17726, axes_17727)
        
        # Assigning a type to the variable 'axes' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'axes', list_17726)
        # SSA join for if statement (line 40)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_17708 and more_types_in_union_17709):
            # SSA join for if statement (line 38)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to len(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'axes' (line 42)
    axes_17729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'axes', False)
    # Processing the call keyword arguments (line 42)
    kwargs_17730 = {}
    # Getting the type of 'len' (line 42)
    len_17728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 7), 'len', False)
    # Calling len(args, kwargs) (line 42)
    len_call_result_17731 = invoke(stypy.reporting.localization.Localization(__file__, 42, 7), len_17728, *[axes_17729], **kwargs_17730)
    
    
    # Call to len(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'shape' (line 42)
    shape_17733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'shape', False)
    # Processing the call keyword arguments (line 42)
    kwargs_17734 = {}
    # Getting the type of 'len' (line 42)
    len_17732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'len', False)
    # Calling len(args, kwargs) (line 42)
    len_call_result_17735 = invoke(stypy.reporting.localization.Localization(__file__, 42, 20), len_17732, *[shape_17733], **kwargs_17734)
    
    # Applying the binary operator '!=' (line 42)
    result_ne_17736 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 7), '!=', len_call_result_17731, len_call_result_17735)
    
    # Testing the type of an if condition (line 42)
    if_condition_17737 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 4), result_ne_17736)
    # Assigning a type to the variable 'if_condition_17737' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_condition_17737', if_condition_17737)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 43)
    # Processing the call arguments (line 43)
    str_17739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'str', 'when given, axes and shape arguments have to be of the same length')
    # Processing the call keyword arguments (line 43)
    kwargs_17740 = {}
    # Getting the type of 'ValueError' (line 43)
    ValueError_17738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 43)
    ValueError_call_result_17741 = invoke(stypy.reporting.localization.Localization(__file__, 43, 14), ValueError_17738, *[str_17739], **kwargs_17740)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 43, 8), ValueError_call_result_17741, 'raise parameter', BaseException)
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to unique(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'axes' (line 45)
    axes_17745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 21), 'axes', False)
    # Processing the call keyword arguments (line 45)
    kwargs_17746 = {}
    # Getting the type of 'np' (line 45)
    np_17743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'np', False)
    # Obtaining the member 'unique' of a type (line 45)
    unique_17744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 11), np_17743, 'unique')
    # Calling unique(args, kwargs) (line 45)
    unique_call_result_17747 = invoke(stypy.reporting.localization.Localization(__file__, 45, 11), unique_17744, *[axes_17745], **kwargs_17746)
    
    # Processing the call keyword arguments (line 45)
    kwargs_17748 = {}
    # Getting the type of 'len' (line 45)
    len_17742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 7), 'len', False)
    # Calling len(args, kwargs) (line 45)
    len_call_result_17749 = invoke(stypy.reporting.localization.Localization(__file__, 45, 7), len_17742, *[unique_call_result_17747], **kwargs_17748)
    
    
    # Call to len(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'axes' (line 45)
    axes_17751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 35), 'axes', False)
    # Processing the call keyword arguments (line 45)
    kwargs_17752 = {}
    # Getting the type of 'len' (line 45)
    len_17750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'len', False)
    # Calling len(args, kwargs) (line 45)
    len_call_result_17753 = invoke(stypy.reporting.localization.Localization(__file__, 45, 31), len_17750, *[axes_17751], **kwargs_17752)
    
    # Applying the binary operator '!=' (line 45)
    result_ne_17754 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 7), '!=', len_call_result_17749, len_call_result_17753)
    
    # Testing the type of an if condition (line 45)
    if_condition_17755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 4), result_ne_17754)
    # Assigning a type to the variable 'if_condition_17755' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'if_condition_17755', if_condition_17755)
    # SSA begins for if statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 46)
    # Processing the call arguments (line 46)
    str_17757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 25), 'str', 'All axes must be unique.')
    # Processing the call keyword arguments (line 46)
    kwargs_17758 = {}
    # Getting the type of 'ValueError' (line 46)
    ValueError_17756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 46)
    ValueError_call_result_17759 = invoke(stypy.reporting.localization.Localization(__file__, 46, 14), ValueError_17756, *[str_17757], **kwargs_17758)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 46, 8), ValueError_call_result_17759, 'raise parameter', BaseException)
    # SSA join for if statement (line 45)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 48)
    tuple_17760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 48)
    # Adding element type (line 48)
    # Getting the type of 'shape' (line 48)
    shape_17761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), tuple_17760, shape_17761)
    # Adding element type (line 48)
    # Getting the type of 'axes' (line 48)
    axes_17762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 18), 'axes')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 11), tuple_17760, axes_17762)
    
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type', tuple_17760)
    
    # ################# End of '_init_nd_shape_and_axes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_init_nd_shape_and_axes' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_17763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17763)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_init_nd_shape_and_axes'
    return stypy_return_type_17763

# Assigning a type to the variable '_init_nd_shape_and_axes' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '_init_nd_shape_and_axes', _init_nd_shape_and_axes)

@norecursion
def dctn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_17764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 17), 'int')
    # Getting the type of 'None' (line 51)
    None_17765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'None')
    # Getting the type of 'None' (line 51)
    None_17766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 37), 'None')
    # Getting the type of 'None' (line 51)
    None_17767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 48), 'None')
    # Getting the type of 'False' (line 51)
    False_17768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 66), 'False')
    defaults = [int_17764, None_17765, None_17766, None_17767, False_17768]
    # Create a new context for function 'dctn'
    module_type_store = module_type_store.open_function_context('dctn', 51, 0, False)
    
    # Passed parameters checking function
    dctn.stypy_localization = localization
    dctn.stypy_type_of_self = None
    dctn.stypy_type_store = module_type_store
    dctn.stypy_function_name = 'dctn'
    dctn.stypy_param_names_list = ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x']
    dctn.stypy_varargs_param_name = None
    dctn.stypy_kwargs_param_name = None
    dctn.stypy_call_defaults = defaults
    dctn.stypy_call_varargs = varargs
    dctn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dctn', ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dctn', localization, ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dctn(...)' code ##################

    str_17769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, (-1)), 'str', "\n    Return multidimensional Discrete Cosine Transform along the specified axes.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    type : {1, 2, 3}, optional\n        Type of the DCT (see Notes). Default type is 2.\n    shape : tuple of ints, optional\n        The shape of the result.  If both `shape` and `axes` (see below) are\n        None, `shape` is ``x.shape``; if `shape` is None but `axes` is\n        not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.\n        If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.\n        If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to\n        length ``shape[i]``.\n    axes : tuple or None, optional\n        Axes along which the DCT is computed; the default is over all axes.\n    norm : {None, 'ortho'}, optional\n        Normalization mode (see Notes). Default is None.\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    y : ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    idctn : Inverse multidimensional DCT\n\n    Notes\n    -----\n    For full details of the DCT types and normalization modes, as well as\n    references, see `dct`.\n\n    Examples\n    --------\n    >>> from scipy.fftpack import dctn, idctn\n    >>> y = np.random.randn(16, 16)\n    >>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))\n    True\n\n    ")
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to asanyarray(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'x' (line 97)
    x_17772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 22), 'x', False)
    # Processing the call keyword arguments (line 97)
    kwargs_17773 = {}
    # Getting the type of 'np' (line 97)
    np_17770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 97)
    asanyarray_17771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), np_17770, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 97)
    asanyarray_call_result_17774 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), asanyarray_17771, *[x_17772], **kwargs_17773)
    
    # Assigning a type to the variable 'x' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'x', asanyarray_call_result_17774)
    
    # Assigning a Call to a Tuple (line 98):
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_17775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    
    # Call to _init_nd_shape_and_axes(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'x' (line 98)
    x_17777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'x', False)
    # Getting the type of 'shape' (line 98)
    shape_17778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 45), 'shape', False)
    # Getting the type of 'axes' (line 98)
    axes_17779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 52), 'axes', False)
    # Processing the call keyword arguments (line 98)
    kwargs_17780 = {}
    # Getting the type of '_init_nd_shape_and_axes' (line 98)
    _init_nd_shape_and_axes_17776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), '_init_nd_shape_and_axes', False)
    # Calling _init_nd_shape_and_axes(args, kwargs) (line 98)
    _init_nd_shape_and_axes_call_result_17781 = invoke(stypy.reporting.localization.Localization(__file__, 98, 18), _init_nd_shape_and_axes_17776, *[x_17777, shape_17778, axes_17779], **kwargs_17780)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___17782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), _init_nd_shape_and_axes_call_result_17781, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_17783 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___17782, int_17775)
    
    # Assigning a type to the variable 'tuple_var_assignment_17591' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_17591', subscript_call_result_17783)
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_17784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'int')
    
    # Call to _init_nd_shape_and_axes(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'x' (line 98)
    x_17786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'x', False)
    # Getting the type of 'shape' (line 98)
    shape_17787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 45), 'shape', False)
    # Getting the type of 'axes' (line 98)
    axes_17788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 52), 'axes', False)
    # Processing the call keyword arguments (line 98)
    kwargs_17789 = {}
    # Getting the type of '_init_nd_shape_and_axes' (line 98)
    _init_nd_shape_and_axes_17785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), '_init_nd_shape_and_axes', False)
    # Calling _init_nd_shape_and_axes(args, kwargs) (line 98)
    _init_nd_shape_and_axes_call_result_17790 = invoke(stypy.reporting.localization.Localization(__file__, 98, 18), _init_nd_shape_and_axes_17785, *[x_17786, shape_17787, axes_17788], **kwargs_17789)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___17791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), _init_nd_shape_and_axes_call_result_17790, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_17792 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), getitem___17791, int_17784)
    
    # Assigning a type to the variable 'tuple_var_assignment_17592' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_17592', subscript_call_result_17792)
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_17591' (line 98)
    tuple_var_assignment_17591_17793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_17591')
    # Assigning a type to the variable 'shape' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'shape', tuple_var_assignment_17591_17793)
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'tuple_var_assignment_17592' (line 98)
    tuple_var_assignment_17592_17794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'tuple_var_assignment_17592')
    # Assigning a type to the variable 'axes' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'axes', tuple_var_assignment_17592_17794)
    
    
    # Call to zip(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'shape' (line 99)
    shape_17796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'shape', False)
    # Getting the type of 'axes' (line 99)
    axes_17797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 28), 'axes', False)
    # Processing the call keyword arguments (line 99)
    kwargs_17798 = {}
    # Getting the type of 'zip' (line 99)
    zip_17795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'zip', False)
    # Calling zip(args, kwargs) (line 99)
    zip_call_result_17799 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), zip_17795, *[shape_17796, axes_17797], **kwargs_17798)
    
    # Testing the type of a for loop iterable (line 99)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 4), zip_call_result_17799)
    # Getting the type of the for loop variable (line 99)
    for_loop_var_17800 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 4), zip_call_result_17799)
    # Assigning a type to the variable 'n' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 4), for_loop_var_17800))
    # Assigning a type to the variable 'ax' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'ax', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 4), for_loop_var_17800))
    # SSA begins for a for statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 100):
    
    # Assigning a Call to a Name (line 100):
    
    # Call to dct(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'x' (line 100)
    x_17802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'x', False)
    # Processing the call keyword arguments (line 100)
    # Getting the type of 'type' (line 100)
    type_17803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'type', False)
    keyword_17804 = type_17803
    # Getting the type of 'n' (line 100)
    n_17805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 32), 'n', False)
    keyword_17806 = n_17805
    # Getting the type of 'ax' (line 100)
    ax_17807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 40), 'ax', False)
    keyword_17808 = ax_17807
    # Getting the type of 'norm' (line 100)
    norm_17809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 49), 'norm', False)
    keyword_17810 = norm_17809
    # Getting the type of 'overwrite_x' (line 100)
    overwrite_x_17811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 67), 'overwrite_x', False)
    keyword_17812 = overwrite_x_17811
    kwargs_17813 = {'axis': keyword_17808, 'type': keyword_17804, 'overwrite_x': keyword_17812, 'norm': keyword_17810, 'n': keyword_17806}
    # Getting the type of 'dct' (line 100)
    dct_17801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'dct', False)
    # Calling dct(args, kwargs) (line 100)
    dct_call_result_17814 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), dct_17801, *[x_17802], **kwargs_17813)
    
    # Assigning a type to the variable 'x' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'x', dct_call_result_17814)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 101)
    x_17815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type', x_17815)
    
    # ################# End of 'dctn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dctn' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_17816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17816)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dctn'
    return stypy_return_type_17816

# Assigning a type to the variable 'dctn' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'dctn', dctn)

@norecursion
def idctn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_17817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'int')
    # Getting the type of 'None' (line 104)
    None_17818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'None')
    # Getting the type of 'None' (line 104)
    None_17819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 38), 'None')
    # Getting the type of 'None' (line 104)
    None_17820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 49), 'None')
    # Getting the type of 'False' (line 104)
    False_17821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 67), 'False')
    defaults = [int_17817, None_17818, None_17819, None_17820, False_17821]
    # Create a new context for function 'idctn'
    module_type_store = module_type_store.open_function_context('idctn', 104, 0, False)
    
    # Passed parameters checking function
    idctn.stypy_localization = localization
    idctn.stypy_type_of_self = None
    idctn.stypy_type_store = module_type_store
    idctn.stypy_function_name = 'idctn'
    idctn.stypy_param_names_list = ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x']
    idctn.stypy_varargs_param_name = None
    idctn.stypy_kwargs_param_name = None
    idctn.stypy_call_defaults = defaults
    idctn.stypy_call_varargs = varargs
    idctn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idctn', ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idctn', localization, ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idctn(...)' code ##################

    str_17822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, (-1)), 'str', "\n    Return multidimensional Discrete Cosine Transform along the specified axes.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    type : {1, 2, 3}, optional\n        Type of the DCT (see Notes). Default type is 2.\n    shape : tuple of ints, optional\n        The shape of the result.  If both `shape` and `axes` (see below) are\n        None, `shape` is ``x.shape``; if `shape` is None but `axes` is\n        not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.\n        If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.\n        If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to\n        length ``shape[i]``.\n    axes : tuple or None, optional\n        Axes along which the IDCT is computed; the default is over all axes.\n    norm : {None, 'ortho'}, optional\n        Normalization mode (see Notes). Default is None.\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    y : ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    dctn : multidimensional DCT\n\n    Notes\n    -----\n    For full details of the IDCT types and normalization modes, as well as\n    references, see `idct`.\n\n    Examples\n    --------\n    >>> from scipy.fftpack import dctn, idctn\n    >>> y = np.random.randn(16, 16)\n    >>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))\n    True\n    ")
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to asanyarray(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'x' (line 149)
    x_17825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), 'x', False)
    # Processing the call keyword arguments (line 149)
    kwargs_17826 = {}
    # Getting the type of 'np' (line 149)
    np_17823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 149)
    asanyarray_17824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), np_17823, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 149)
    asanyarray_call_result_17827 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), asanyarray_17824, *[x_17825], **kwargs_17826)
    
    # Assigning a type to the variable 'x' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'x', asanyarray_call_result_17827)
    
    # Assigning a Call to a Tuple (line 150):
    
    # Assigning a Subscript to a Name (line 150):
    
    # Obtaining the type of the subscript
    int_17828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 4), 'int')
    
    # Call to _init_nd_shape_and_axes(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'x' (line 150)
    x_17830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 42), 'x', False)
    # Getting the type of 'shape' (line 150)
    shape_17831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 45), 'shape', False)
    # Getting the type of 'axes' (line 150)
    axes_17832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 52), 'axes', False)
    # Processing the call keyword arguments (line 150)
    kwargs_17833 = {}
    # Getting the type of '_init_nd_shape_and_axes' (line 150)
    _init_nd_shape_and_axes_17829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), '_init_nd_shape_and_axes', False)
    # Calling _init_nd_shape_and_axes(args, kwargs) (line 150)
    _init_nd_shape_and_axes_call_result_17834 = invoke(stypy.reporting.localization.Localization(__file__, 150, 18), _init_nd_shape_and_axes_17829, *[x_17830, shape_17831, axes_17832], **kwargs_17833)
    
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___17835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 4), _init_nd_shape_and_axes_call_result_17834, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_17836 = invoke(stypy.reporting.localization.Localization(__file__, 150, 4), getitem___17835, int_17828)
    
    # Assigning a type to the variable 'tuple_var_assignment_17593' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_var_assignment_17593', subscript_call_result_17836)
    
    # Assigning a Subscript to a Name (line 150):
    
    # Obtaining the type of the subscript
    int_17837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 4), 'int')
    
    # Call to _init_nd_shape_and_axes(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'x' (line 150)
    x_17839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 42), 'x', False)
    # Getting the type of 'shape' (line 150)
    shape_17840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 45), 'shape', False)
    # Getting the type of 'axes' (line 150)
    axes_17841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 52), 'axes', False)
    # Processing the call keyword arguments (line 150)
    kwargs_17842 = {}
    # Getting the type of '_init_nd_shape_and_axes' (line 150)
    _init_nd_shape_and_axes_17838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), '_init_nd_shape_and_axes', False)
    # Calling _init_nd_shape_and_axes(args, kwargs) (line 150)
    _init_nd_shape_and_axes_call_result_17843 = invoke(stypy.reporting.localization.Localization(__file__, 150, 18), _init_nd_shape_and_axes_17838, *[x_17839, shape_17840, axes_17841], **kwargs_17842)
    
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___17844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 4), _init_nd_shape_and_axes_call_result_17843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_17845 = invoke(stypy.reporting.localization.Localization(__file__, 150, 4), getitem___17844, int_17837)
    
    # Assigning a type to the variable 'tuple_var_assignment_17594' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_var_assignment_17594', subscript_call_result_17845)
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'tuple_var_assignment_17593' (line 150)
    tuple_var_assignment_17593_17846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_var_assignment_17593')
    # Assigning a type to the variable 'shape' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'shape', tuple_var_assignment_17593_17846)
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'tuple_var_assignment_17594' (line 150)
    tuple_var_assignment_17594_17847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_var_assignment_17594')
    # Assigning a type to the variable 'axes' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'axes', tuple_var_assignment_17594_17847)
    
    
    # Call to zip(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'shape' (line 151)
    shape_17849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'shape', False)
    # Getting the type of 'axes' (line 151)
    axes_17850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 28), 'axes', False)
    # Processing the call keyword arguments (line 151)
    kwargs_17851 = {}
    # Getting the type of 'zip' (line 151)
    zip_17848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 17), 'zip', False)
    # Calling zip(args, kwargs) (line 151)
    zip_call_result_17852 = invoke(stypy.reporting.localization.Localization(__file__, 151, 17), zip_17848, *[shape_17849, axes_17850], **kwargs_17851)
    
    # Testing the type of a for loop iterable (line 151)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 151, 4), zip_call_result_17852)
    # Getting the type of the for loop variable (line 151)
    for_loop_var_17853 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 151, 4), zip_call_result_17852)
    # Assigning a type to the variable 'n' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 4), for_loop_var_17853))
    # Assigning a type to the variable 'ax' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'ax', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 4), for_loop_var_17853))
    # SSA begins for a for statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to idct(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'x' (line 152)
    x_17855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'x', False)
    # Processing the call keyword arguments (line 152)
    # Getting the type of 'type' (line 152)
    type_17856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'type', False)
    keyword_17857 = type_17856
    # Getting the type of 'n' (line 152)
    n_17858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 33), 'n', False)
    keyword_17859 = n_17858
    # Getting the type of 'ax' (line 152)
    ax_17860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 41), 'ax', False)
    keyword_17861 = ax_17860
    # Getting the type of 'norm' (line 152)
    norm_17862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 50), 'norm', False)
    keyword_17863 = norm_17862
    # Getting the type of 'overwrite_x' (line 153)
    overwrite_x_17864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 29), 'overwrite_x', False)
    keyword_17865 = overwrite_x_17864
    kwargs_17866 = {'axis': keyword_17861, 'type': keyword_17857, 'overwrite_x': keyword_17865, 'norm': keyword_17863, 'n': keyword_17859}
    # Getting the type of 'idct' (line 152)
    idct_17854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'idct', False)
    # Calling idct(args, kwargs) (line 152)
    idct_call_result_17867 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), idct_17854, *[x_17855], **kwargs_17866)
    
    # Assigning a type to the variable 'x' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'x', idct_call_result_17867)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 154)
    x_17868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type', x_17868)
    
    # ################# End of 'idctn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idctn' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_17869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17869)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idctn'
    return stypy_return_type_17869

# Assigning a type to the variable 'idctn' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'idctn', idctn)

@norecursion
def dstn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_17870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 17), 'int')
    # Getting the type of 'None' (line 157)
    None_17871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 26), 'None')
    # Getting the type of 'None' (line 157)
    None_17872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 37), 'None')
    # Getting the type of 'None' (line 157)
    None_17873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 48), 'None')
    # Getting the type of 'False' (line 157)
    False_17874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 66), 'False')
    defaults = [int_17870, None_17871, None_17872, None_17873, False_17874]
    # Create a new context for function 'dstn'
    module_type_store = module_type_store.open_function_context('dstn', 157, 0, False)
    
    # Passed parameters checking function
    dstn.stypy_localization = localization
    dstn.stypy_type_of_self = None
    dstn.stypy_type_store = module_type_store
    dstn.stypy_function_name = 'dstn'
    dstn.stypy_param_names_list = ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x']
    dstn.stypy_varargs_param_name = None
    dstn.stypy_kwargs_param_name = None
    dstn.stypy_call_defaults = defaults
    dstn.stypy_call_varargs = varargs
    dstn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dstn', ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dstn', localization, ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dstn(...)' code ##################

    str_17875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, (-1)), 'str', "\n    Return multidimensional Discrete Sine Transform along the specified axes.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    type : {1, 2, 3}, optional\n        Type of the DCT (see Notes). Default type is 2.\n    shape : tuple of ints, optional\n        The shape of the result.  If both `shape` and `axes` (see below) are\n        None, `shape` is ``x.shape``; if `shape` is None but `axes` is\n        not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.\n        If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.\n        If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to\n        length ``shape[i]``.\n    axes : tuple or None, optional\n        Axes along which the DCT is computed; the default is over all axes.\n    norm : {None, 'ortho'}, optional\n        Normalization mode (see Notes). Default is None.\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    y : ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    idstn : Inverse multidimensional DST\n\n    Notes\n    -----\n    For full details of the DST types and normalization modes, as well as\n    references, see `dst`.\n\n    Examples\n    --------\n    >>> from scipy.fftpack import dstn, idstn\n    >>> y = np.random.randn(16, 16)\n    >>> np.allclose(y, idstn(dstn(y, norm='ortho'), norm='ortho'))\n    True\n\n    ")
    
    # Assigning a Call to a Name (line 203):
    
    # Assigning a Call to a Name (line 203):
    
    # Call to asanyarray(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'x' (line 203)
    x_17878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 22), 'x', False)
    # Processing the call keyword arguments (line 203)
    kwargs_17879 = {}
    # Getting the type of 'np' (line 203)
    np_17876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 203)
    asanyarray_17877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), np_17876, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 203)
    asanyarray_call_result_17880 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), asanyarray_17877, *[x_17878], **kwargs_17879)
    
    # Assigning a type to the variable 'x' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'x', asanyarray_call_result_17880)
    
    # Assigning a Call to a Tuple (line 204):
    
    # Assigning a Subscript to a Name (line 204):
    
    # Obtaining the type of the subscript
    int_17881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 4), 'int')
    
    # Call to _init_nd_shape_and_axes(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'x' (line 204)
    x_17883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 42), 'x', False)
    # Getting the type of 'shape' (line 204)
    shape_17884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 45), 'shape', False)
    # Getting the type of 'axes' (line 204)
    axes_17885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 52), 'axes', False)
    # Processing the call keyword arguments (line 204)
    kwargs_17886 = {}
    # Getting the type of '_init_nd_shape_and_axes' (line 204)
    _init_nd_shape_and_axes_17882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), '_init_nd_shape_and_axes', False)
    # Calling _init_nd_shape_and_axes(args, kwargs) (line 204)
    _init_nd_shape_and_axes_call_result_17887 = invoke(stypy.reporting.localization.Localization(__file__, 204, 18), _init_nd_shape_and_axes_17882, *[x_17883, shape_17884, axes_17885], **kwargs_17886)
    
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___17888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 4), _init_nd_shape_and_axes_call_result_17887, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_17889 = invoke(stypy.reporting.localization.Localization(__file__, 204, 4), getitem___17888, int_17881)
    
    # Assigning a type to the variable 'tuple_var_assignment_17595' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'tuple_var_assignment_17595', subscript_call_result_17889)
    
    # Assigning a Subscript to a Name (line 204):
    
    # Obtaining the type of the subscript
    int_17890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 4), 'int')
    
    # Call to _init_nd_shape_and_axes(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'x' (line 204)
    x_17892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 42), 'x', False)
    # Getting the type of 'shape' (line 204)
    shape_17893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 45), 'shape', False)
    # Getting the type of 'axes' (line 204)
    axes_17894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 52), 'axes', False)
    # Processing the call keyword arguments (line 204)
    kwargs_17895 = {}
    # Getting the type of '_init_nd_shape_and_axes' (line 204)
    _init_nd_shape_and_axes_17891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), '_init_nd_shape_and_axes', False)
    # Calling _init_nd_shape_and_axes(args, kwargs) (line 204)
    _init_nd_shape_and_axes_call_result_17896 = invoke(stypy.reporting.localization.Localization(__file__, 204, 18), _init_nd_shape_and_axes_17891, *[x_17892, shape_17893, axes_17894], **kwargs_17895)
    
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___17897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 4), _init_nd_shape_and_axes_call_result_17896, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_17898 = invoke(stypy.reporting.localization.Localization(__file__, 204, 4), getitem___17897, int_17890)
    
    # Assigning a type to the variable 'tuple_var_assignment_17596' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'tuple_var_assignment_17596', subscript_call_result_17898)
    
    # Assigning a Name to a Name (line 204):
    # Getting the type of 'tuple_var_assignment_17595' (line 204)
    tuple_var_assignment_17595_17899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'tuple_var_assignment_17595')
    # Assigning a type to the variable 'shape' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'shape', tuple_var_assignment_17595_17899)
    
    # Assigning a Name to a Name (line 204):
    # Getting the type of 'tuple_var_assignment_17596' (line 204)
    tuple_var_assignment_17596_17900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'tuple_var_assignment_17596')
    # Assigning a type to the variable 'axes' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'axes', tuple_var_assignment_17596_17900)
    
    
    # Call to zip(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'shape' (line 205)
    shape_17902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'shape', False)
    # Getting the type of 'axes' (line 205)
    axes_17903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 28), 'axes', False)
    # Processing the call keyword arguments (line 205)
    kwargs_17904 = {}
    # Getting the type of 'zip' (line 205)
    zip_17901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'zip', False)
    # Calling zip(args, kwargs) (line 205)
    zip_call_result_17905 = invoke(stypy.reporting.localization.Localization(__file__, 205, 17), zip_17901, *[shape_17902, axes_17903], **kwargs_17904)
    
    # Testing the type of a for loop iterable (line 205)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 205, 4), zip_call_result_17905)
    # Getting the type of the for loop variable (line 205)
    for_loop_var_17906 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 205, 4), zip_call_result_17905)
    # Assigning a type to the variable 'n' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 4), for_loop_var_17906))
    # Assigning a type to the variable 'ax' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'ax', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 4), for_loop_var_17906))
    # SSA begins for a for statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 206):
    
    # Assigning a Call to a Name (line 206):
    
    # Call to dst(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'x' (line 206)
    x_17908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'x', False)
    # Processing the call keyword arguments (line 206)
    # Getting the type of 'type' (line 206)
    type_17909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'type', False)
    keyword_17910 = type_17909
    # Getting the type of 'n' (line 206)
    n_17911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 32), 'n', False)
    keyword_17912 = n_17911
    # Getting the type of 'ax' (line 206)
    ax_17913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 40), 'ax', False)
    keyword_17914 = ax_17913
    # Getting the type of 'norm' (line 206)
    norm_17915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 49), 'norm', False)
    keyword_17916 = norm_17915
    # Getting the type of 'overwrite_x' (line 206)
    overwrite_x_17917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 67), 'overwrite_x', False)
    keyword_17918 = overwrite_x_17917
    kwargs_17919 = {'axis': keyword_17914, 'type': keyword_17910, 'overwrite_x': keyword_17918, 'norm': keyword_17916, 'n': keyword_17912}
    # Getting the type of 'dst' (line 206)
    dst_17907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'dst', False)
    # Calling dst(args, kwargs) (line 206)
    dst_call_result_17920 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), dst_17907, *[x_17908], **kwargs_17919)
    
    # Assigning a type to the variable 'x' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'x', dst_call_result_17920)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 207)
    x_17921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type', x_17921)
    
    # ################# End of 'dstn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dstn' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_17922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17922)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dstn'
    return stypy_return_type_17922

# Assigning a type to the variable 'dstn' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'dstn', dstn)

@norecursion
def idstn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_17923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 18), 'int')
    # Getting the type of 'None' (line 210)
    None_17924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 27), 'None')
    # Getting the type of 'None' (line 210)
    None_17925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 38), 'None')
    # Getting the type of 'None' (line 210)
    None_17926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 49), 'None')
    # Getting the type of 'False' (line 210)
    False_17927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 67), 'False')
    defaults = [int_17923, None_17924, None_17925, None_17926, False_17927]
    # Create a new context for function 'idstn'
    module_type_store = module_type_store.open_function_context('idstn', 210, 0, False)
    
    # Passed parameters checking function
    idstn.stypy_localization = localization
    idstn.stypy_type_of_self = None
    idstn.stypy_type_store = module_type_store
    idstn.stypy_function_name = 'idstn'
    idstn.stypy_param_names_list = ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x']
    idstn.stypy_varargs_param_name = None
    idstn.stypy_kwargs_param_name = None
    idstn.stypy_call_defaults = defaults
    idstn.stypy_call_varargs = varargs
    idstn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idstn', ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idstn', localization, ['x', 'type', 'shape', 'axes', 'norm', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idstn(...)' code ##################

    str_17928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, (-1)), 'str', "\n    Return multidimensional Discrete Sine Transform along the specified axes.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    type : {1, 2, 3}, optional\n        Type of the DCT (see Notes). Default type is 2.\n    shape : tuple of ints, optional\n        The shape of the result.  If both `shape` and `axes` (see below) are\n        None, `shape` is ``x.shape``; if `shape` is None but `axes` is\n        not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.\n        If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.\n        If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to\n        length ``shape[i]``.\n    axes : tuple or None, optional\n        Axes along which the IDCT is computed; the default is over all axes.\n    norm : {None, 'ortho'}, optional\n        Normalization mode (see Notes). Default is None.\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    y : ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    dctn : multidimensional DST\n\n    Notes\n    -----\n    For full details of the IDST types and normalization modes, as well as\n    references, see `idst`.\n\n    Examples\n    --------\n    >>> from scipy.fftpack import dstn, idstn\n    >>> y = np.random.randn(16, 16)\n    >>> np.allclose(y, idstn(dstn(y, norm='ortho'), norm='ortho'))\n    True\n    ")
    
    # Assigning a Call to a Name (line 255):
    
    # Assigning a Call to a Name (line 255):
    
    # Call to asanyarray(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'x' (line 255)
    x_17931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 22), 'x', False)
    # Processing the call keyword arguments (line 255)
    kwargs_17932 = {}
    # Getting the type of 'np' (line 255)
    np_17929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 255)
    asanyarray_17930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), np_17929, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 255)
    asanyarray_call_result_17933 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), asanyarray_17930, *[x_17931], **kwargs_17932)
    
    # Assigning a type to the variable 'x' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'x', asanyarray_call_result_17933)
    
    # Assigning a Call to a Tuple (line 256):
    
    # Assigning a Subscript to a Name (line 256):
    
    # Obtaining the type of the subscript
    int_17934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 4), 'int')
    
    # Call to _init_nd_shape_and_axes(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'x' (line 256)
    x_17936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 42), 'x', False)
    # Getting the type of 'shape' (line 256)
    shape_17937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 45), 'shape', False)
    # Getting the type of 'axes' (line 256)
    axes_17938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 52), 'axes', False)
    # Processing the call keyword arguments (line 256)
    kwargs_17939 = {}
    # Getting the type of '_init_nd_shape_and_axes' (line 256)
    _init_nd_shape_and_axes_17935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 18), '_init_nd_shape_and_axes', False)
    # Calling _init_nd_shape_and_axes(args, kwargs) (line 256)
    _init_nd_shape_and_axes_call_result_17940 = invoke(stypy.reporting.localization.Localization(__file__, 256, 18), _init_nd_shape_and_axes_17935, *[x_17936, shape_17937, axes_17938], **kwargs_17939)
    
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___17941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 4), _init_nd_shape_and_axes_call_result_17940, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_17942 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), getitem___17941, int_17934)
    
    # Assigning a type to the variable 'tuple_var_assignment_17597' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_17597', subscript_call_result_17942)
    
    # Assigning a Subscript to a Name (line 256):
    
    # Obtaining the type of the subscript
    int_17943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 4), 'int')
    
    # Call to _init_nd_shape_and_axes(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'x' (line 256)
    x_17945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 42), 'x', False)
    # Getting the type of 'shape' (line 256)
    shape_17946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 45), 'shape', False)
    # Getting the type of 'axes' (line 256)
    axes_17947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 52), 'axes', False)
    # Processing the call keyword arguments (line 256)
    kwargs_17948 = {}
    # Getting the type of '_init_nd_shape_and_axes' (line 256)
    _init_nd_shape_and_axes_17944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 18), '_init_nd_shape_and_axes', False)
    # Calling _init_nd_shape_and_axes(args, kwargs) (line 256)
    _init_nd_shape_and_axes_call_result_17949 = invoke(stypy.reporting.localization.Localization(__file__, 256, 18), _init_nd_shape_and_axes_17944, *[x_17945, shape_17946, axes_17947], **kwargs_17948)
    
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___17950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 4), _init_nd_shape_and_axes_call_result_17949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_17951 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), getitem___17950, int_17943)
    
    # Assigning a type to the variable 'tuple_var_assignment_17598' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_17598', subscript_call_result_17951)
    
    # Assigning a Name to a Name (line 256):
    # Getting the type of 'tuple_var_assignment_17597' (line 256)
    tuple_var_assignment_17597_17952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_17597')
    # Assigning a type to the variable 'shape' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'shape', tuple_var_assignment_17597_17952)
    
    # Assigning a Name to a Name (line 256):
    # Getting the type of 'tuple_var_assignment_17598' (line 256)
    tuple_var_assignment_17598_17953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_17598')
    # Assigning a type to the variable 'axes' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'axes', tuple_var_assignment_17598_17953)
    
    
    # Call to zip(...): (line 257)
    # Processing the call arguments (line 257)
    # Getting the type of 'shape' (line 257)
    shape_17955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 21), 'shape', False)
    # Getting the type of 'axes' (line 257)
    axes_17956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'axes', False)
    # Processing the call keyword arguments (line 257)
    kwargs_17957 = {}
    # Getting the type of 'zip' (line 257)
    zip_17954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 17), 'zip', False)
    # Calling zip(args, kwargs) (line 257)
    zip_call_result_17958 = invoke(stypy.reporting.localization.Localization(__file__, 257, 17), zip_17954, *[shape_17955, axes_17956], **kwargs_17957)
    
    # Testing the type of a for loop iterable (line 257)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 257, 4), zip_call_result_17958)
    # Getting the type of the for loop variable (line 257)
    for_loop_var_17959 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 257, 4), zip_call_result_17958)
    # Assigning a type to the variable 'n' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 4), for_loop_var_17959))
    # Assigning a type to the variable 'ax' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'ax', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 4), for_loop_var_17959))
    # SSA begins for a for statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to idst(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 'x' (line 258)
    x_17961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 17), 'x', False)
    # Processing the call keyword arguments (line 258)
    # Getting the type of 'type' (line 258)
    type_17962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 25), 'type', False)
    keyword_17963 = type_17962
    # Getting the type of 'n' (line 258)
    n_17964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 33), 'n', False)
    keyword_17965 = n_17964
    # Getting the type of 'ax' (line 258)
    ax_17966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 41), 'ax', False)
    keyword_17967 = ax_17966
    # Getting the type of 'norm' (line 258)
    norm_17968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 50), 'norm', False)
    keyword_17969 = norm_17968
    # Getting the type of 'overwrite_x' (line 259)
    overwrite_x_17970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 29), 'overwrite_x', False)
    keyword_17971 = overwrite_x_17970
    kwargs_17972 = {'axis': keyword_17967, 'type': keyword_17963, 'overwrite_x': keyword_17971, 'norm': keyword_17969, 'n': keyword_17965}
    # Getting the type of 'idst' (line 258)
    idst_17960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'idst', False)
    # Calling idst(args, kwargs) (line 258)
    idst_call_result_17973 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), idst_17960, *[x_17961], **kwargs_17972)
    
    # Assigning a type to the variable 'x' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'x', idst_call_result_17973)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 260)
    x_17974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type', x_17974)
    
    # ################# End of 'idstn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idstn' in the type store
    # Getting the type of 'stypy_return_type' (line 210)
    stypy_return_type_17975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17975)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idstn'
    return stypy_return_type_17975

# Assigning a type to the variable 'idstn' (line 210)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 0), 'idstn', idstn)

@norecursion
def dct(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_17976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 16), 'int')
    # Getting the type of 'None' (line 263)
    None_17977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 21), 'None')
    int_17978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 32), 'int')
    # Getting the type of 'None' (line 263)
    None_17979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 41), 'None')
    # Getting the type of 'False' (line 263)
    False_17980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 59), 'False')
    defaults = [int_17976, None_17977, int_17978, None_17979, False_17980]
    # Create a new context for function 'dct'
    module_type_store = module_type_store.open_function_context('dct', 263, 0, False)
    
    # Passed parameters checking function
    dct.stypy_localization = localization
    dct.stypy_type_of_self = None
    dct.stypy_type_store = module_type_store
    dct.stypy_function_name = 'dct'
    dct.stypy_param_names_list = ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x']
    dct.stypy_varargs_param_name = None
    dct.stypy_kwargs_param_name = None
    dct.stypy_call_defaults = defaults
    dct.stypy_call_varargs = varargs
    dct.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dct', ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dct', localization, ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dct(...)' code ##################

    str_17981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, (-1)), 'str', '\n    Return the Discrete Cosine Transform of arbitrary type sequence x.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    type : {1, 2, 3}, optional\n        Type of the DCT (see Notes). Default type is 2.\n    n : int, optional\n        Length of the transform.  If ``n < x.shape[axis]``, `x` is\n        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The\n        default results in ``n = x.shape[axis]``.\n    axis : int, optional\n        Axis along which the dct is computed; the default is over the\n        last axis (i.e., ``axis=-1``).\n    norm : {None, \'ortho\'}, optional\n        Normalization mode (see Notes). Default is None.\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    y : ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    idct : Inverse DCT\n\n    Notes\n    -----\n    For a single dimension array ``x``, ``dct(x, norm=\'ortho\')`` is equal to\n    MATLAB ``dct(x)``.\n\n    There are theoretically 8 types of the DCT, only the first 3 types are\n    implemented in scipy. \'The\' DCT generally refers to DCT type 2, and \'the\'\n    Inverse DCT generally refers to DCT type 3.\n\n    **Type I**\n\n    There are several definitions of the DCT-I; we use the following\n    (for ``norm=None``)::\n\n                                         N-2\n      y[k] = x[0] + (-1)**k x[N-1] + 2 * sum x[n]*cos(pi*k*n/(N-1))\n                                         n=1\n\n    Only None is supported as normalization mode for DCT-I. Note also that the\n    DCT-I is only supported for input size > 1\n\n    **Type II**\n\n    There are several definitions of the DCT-II; we use the following\n    (for ``norm=None``)::\n\n\n                N-1\n      y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.\n                n=0\n\n    If ``norm=\'ortho\'``, ``y[k]`` is multiplied by a scaling factor `f`::\n\n      f = sqrt(1/(4*N)) if k = 0,\n      f = sqrt(1/(2*N)) otherwise.\n\n    Which makes the corresponding matrix of coefficients orthonormal\n    (``OO\' = Id``).\n\n    **Type III**\n\n    There are several definitions, we use the following\n    (for ``norm=None``)::\n\n                        N-1\n      y[k] = x[0] + 2 * sum x[n]*cos(pi*(k+0.5)*n/N), 0 <= k < N.\n                        n=1\n\n    or, for ``norm=\'ortho\'`` and 0 <= k < N::\n\n                                          N-1\n      y[k] = x[0] / sqrt(N) + sqrt(2/N) * sum x[n]*cos(pi*(k+0.5)*n/N)\n                                          n=1\n\n    The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up\n    to a factor `2N`. The orthonormalized DCT-III is exactly the inverse of\n    the orthonormalized DCT-II.\n\n    References\n    ----------\n    .. [1] \'A Fast Cosine Transform in One and Two Dimensions\', by J.\n           Makhoul, `IEEE Transactions on acoustics, speech and signal\n           processing` vol. 28(1), pp. 27-34,\n           http://dx.doi.org/10.1109/TASSP.1980.1163351 (1980).\n    .. [2] Wikipedia, "Discrete cosine transform",\n           http://en.wikipedia.org/wiki/Discrete_cosine_transform\n\n    Examples\n    --------\n    The Type 1 DCT is equivalent to the FFT (though faster) for real,\n    even-symmetrical inputs.  The output is also real and even-symmetrical.\n    Half of the FFT input is used to generate half of the FFT output:\n\n    >>> from scipy.fftpack import fft, dct\n    >>> fft(np.array([4., 3., 5., 10., 5., 3.])).real\n    array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])\n    >>> dct(np.array([4., 3., 5., 10.]), 1)\n    array([ 30.,  -8.,   6.,  -2.])\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'type' (line 374)
    type_17982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 7), 'type')
    int_17983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 15), 'int')
    # Applying the binary operator '==' (line 374)
    result_eq_17984 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 7), '==', type_17982, int_17983)
    
    
    # Getting the type of 'norm' (line 374)
    norm_17985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 21), 'norm')
    # Getting the type of 'None' (line 374)
    None_17986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 33), 'None')
    # Applying the binary operator 'isnot' (line 374)
    result_is_not_17987 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 21), 'isnot', norm_17985, None_17986)
    
    # Applying the binary operator 'and' (line 374)
    result_and_keyword_17988 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 7), 'and', result_eq_17984, result_is_not_17987)
    
    # Testing the type of an if condition (line 374)
    if_condition_17989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 4), result_and_keyword_17988)
    # Assigning a type to the variable 'if_condition_17989' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'if_condition_17989', if_condition_17989)
    # SSA begins for if statement (line 374)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NotImplementedError(...): (line 375)
    # Processing the call arguments (line 375)
    str_17991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 14), 'str', 'Orthonormalization not yet supported for DCT-I')
    # Processing the call keyword arguments (line 375)
    kwargs_17992 = {}
    # Getting the type of 'NotImplementedError' (line 375)
    NotImplementedError_17990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 14), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 375)
    NotImplementedError_call_result_17993 = invoke(stypy.reporting.localization.Localization(__file__, 375, 14), NotImplementedError_17990, *[str_17991], **kwargs_17992)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 375, 8), NotImplementedError_call_result_17993, 'raise parameter', BaseException)
    # SSA join for if statement (line 374)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _dct(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of 'x' (line 377)
    x_17995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'x', False)
    # Getting the type of 'type' (line 377)
    type_17996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'type', False)
    # Getting the type of 'n' (line 377)
    n_17997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 25), 'n', False)
    # Getting the type of 'axis' (line 377)
    axis_17998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 28), 'axis', False)
    # Processing the call keyword arguments (line 377)
    # Getting the type of 'norm' (line 377)
    norm_17999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 44), 'norm', False)
    keyword_18000 = norm_17999
    # Getting the type of 'overwrite_x' (line 377)
    overwrite_x_18001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 62), 'overwrite_x', False)
    keyword_18002 = overwrite_x_18001
    kwargs_18003 = {'normalize': keyword_18000, 'overwrite_x': keyword_18002}
    # Getting the type of '_dct' (line 377)
    _dct_17994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 11), '_dct', False)
    # Calling _dct(args, kwargs) (line 377)
    _dct_call_result_18004 = invoke(stypy.reporting.localization.Localization(__file__, 377, 11), _dct_17994, *[x_17995, type_17996, n_17997, axis_17998], **kwargs_18003)
    
    # Assigning a type to the variable 'stypy_return_type' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'stypy_return_type', _dct_call_result_18004)
    
    # ################# End of 'dct(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dct' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_18005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18005)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dct'
    return stypy_return_type_18005

# Assigning a type to the variable 'dct' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'dct', dct)

@norecursion
def idct(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_18006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 17), 'int')
    # Getting the type of 'None' (line 380)
    None_18007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 22), 'None')
    int_18008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 33), 'int')
    # Getting the type of 'None' (line 380)
    None_18009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 42), 'None')
    # Getting the type of 'False' (line 380)
    False_18010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 60), 'False')
    defaults = [int_18006, None_18007, int_18008, None_18009, False_18010]
    # Create a new context for function 'idct'
    module_type_store = module_type_store.open_function_context('idct', 380, 0, False)
    
    # Passed parameters checking function
    idct.stypy_localization = localization
    idct.stypy_type_of_self = None
    idct.stypy_type_store = module_type_store
    idct.stypy_function_name = 'idct'
    idct.stypy_param_names_list = ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x']
    idct.stypy_varargs_param_name = None
    idct.stypy_kwargs_param_name = None
    idct.stypy_call_defaults = defaults
    idct.stypy_call_varargs = varargs
    idct.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idct', ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idct', localization, ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idct(...)' code ##################

    str_18011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, (-1)), 'str', "\n    Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    type : {1, 2, 3}, optional\n        Type of the DCT (see Notes). Default type is 2.\n    n : int, optional\n        Length of the transform.  If ``n < x.shape[axis]``, `x` is\n        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The\n        default results in ``n = x.shape[axis]``.\n    axis : int, optional\n        Axis along which the idct is computed; the default is over the\n        last axis (i.e., ``axis=-1``).\n    norm : {None, 'ortho'}, optional\n        Normalization mode (see Notes). Default is None.\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    idct : ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    dct : Forward DCT\n\n    Notes\n    -----\n    For a single dimension array `x`, ``idct(x, norm='ortho')`` is equal to\n    MATLAB ``idct(x)``.\n\n    'The' IDCT is the IDCT of type 2, which is the same as DCT of type 3.\n\n    IDCT of type 1 is the DCT of type 1, IDCT of type 2 is the DCT of type\n    3, and IDCT of type 3 is the DCT of type 2. For the definition of these\n    types, see `dct`.\n\n    Examples\n    --------\n    The Type 1 DCT is equivalent to the DFT for real, even-symmetrical\n    inputs.  The output is also real and even-symmetrical.  Half of the IFFT\n    input is used to generate half of the IFFT output:\n\n    >>> from scipy.fftpack import ifft, idct\n    >>> ifft(np.array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])).real\n    array([  4.,   3.,   5.,  10.,   5.,   3.])\n    >>> idct(np.array([ 30.,  -8.,   6.,  -2.]), 1) / 6\n    array([  4.,   3.,   5.,  10.])\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'type' (line 435)
    type_18012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 7), 'type')
    int_18013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 15), 'int')
    # Applying the binary operator '==' (line 435)
    result_eq_18014 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 7), '==', type_18012, int_18013)
    
    
    # Getting the type of 'norm' (line 435)
    norm_18015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 21), 'norm')
    # Getting the type of 'None' (line 435)
    None_18016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 33), 'None')
    # Applying the binary operator 'isnot' (line 435)
    result_is_not_18017 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 21), 'isnot', norm_18015, None_18016)
    
    # Applying the binary operator 'and' (line 435)
    result_and_keyword_18018 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 7), 'and', result_eq_18014, result_is_not_18017)
    
    # Testing the type of an if condition (line 435)
    if_condition_18019 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 4), result_and_keyword_18018)
    # Assigning a type to the variable 'if_condition_18019' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'if_condition_18019', if_condition_18019)
    # SSA begins for if statement (line 435)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NotImplementedError(...): (line 436)
    # Processing the call arguments (line 436)
    str_18021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 14), 'str', 'Orthonormalization not yet supported for IDCT-I')
    # Processing the call keyword arguments (line 436)
    kwargs_18022 = {}
    # Getting the type of 'NotImplementedError' (line 436)
    NotImplementedError_18020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 14), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 436)
    NotImplementedError_call_result_18023 = invoke(stypy.reporting.localization.Localization(__file__, 436, 14), NotImplementedError_18020, *[str_18021], **kwargs_18022)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 436, 8), NotImplementedError_call_result_18023, 'raise parameter', BaseException)
    # SSA join for if statement (line 435)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 439):
    
    # Assigning a Dict to a Name (line 439):
    
    # Obtaining an instance of the builtin type 'dict' (line 439)
    dict_18024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 439)
    # Adding element type (key, value) (line 439)
    int_18025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 11), 'int')
    int_18026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 13), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 10), dict_18024, (int_18025, int_18026))
    # Adding element type (key, value) (line 439)
    int_18027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 16), 'int')
    int_18028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 18), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 10), dict_18024, (int_18027, int_18028))
    # Adding element type (key, value) (line 439)
    int_18029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 21), 'int')
    int_18030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 23), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 10), dict_18024, (int_18029, int_18030))
    
    # Assigning a type to the variable '_TP' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), '_TP', dict_18024)
    
    # Call to _dct(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'x' (line 440)
    x_18032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'type' (line 440)
    type_18033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 23), 'type', False)
    # Getting the type of '_TP' (line 440)
    _TP_18034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 19), '_TP', False)
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___18035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 19), _TP_18034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_18036 = invoke(stypy.reporting.localization.Localization(__file__, 440, 19), getitem___18035, type_18033)
    
    # Getting the type of 'n' (line 440)
    n_18037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 30), 'n', False)
    # Getting the type of 'axis' (line 440)
    axis_18038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 33), 'axis', False)
    # Processing the call keyword arguments (line 440)
    # Getting the type of 'norm' (line 440)
    norm_18039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 49), 'norm', False)
    keyword_18040 = norm_18039
    # Getting the type of 'overwrite_x' (line 440)
    overwrite_x_18041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 67), 'overwrite_x', False)
    keyword_18042 = overwrite_x_18041
    kwargs_18043 = {'normalize': keyword_18040, 'overwrite_x': keyword_18042}
    # Getting the type of '_dct' (line 440)
    _dct_18031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 11), '_dct', False)
    # Calling _dct(args, kwargs) (line 440)
    _dct_call_result_18044 = invoke(stypy.reporting.localization.Localization(__file__, 440, 11), _dct_18031, *[x_18032, subscript_call_result_18036, n_18037, axis_18038], **kwargs_18043)
    
    # Assigning a type to the variable 'stypy_return_type' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'stypy_return_type', _dct_call_result_18044)
    
    # ################# End of 'idct(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idct' in the type store
    # Getting the type of 'stypy_return_type' (line 380)
    stypy_return_type_18045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18045)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idct'
    return stypy_return_type_18045

# Assigning a type to the variable 'idct' (line 380)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), 'idct', idct)

@norecursion
def _get_dct_fun(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_dct_fun'
    module_type_store = module_type_store.open_function_context('_get_dct_fun', 443, 0, False)
    
    # Passed parameters checking function
    _get_dct_fun.stypy_localization = localization
    _get_dct_fun.stypy_type_of_self = None
    _get_dct_fun.stypy_type_store = module_type_store
    _get_dct_fun.stypy_function_name = '_get_dct_fun'
    _get_dct_fun.stypy_param_names_list = ['type', 'dtype']
    _get_dct_fun.stypy_varargs_param_name = None
    _get_dct_fun.stypy_kwargs_param_name = None
    _get_dct_fun.stypy_call_defaults = defaults
    _get_dct_fun.stypy_call_varargs = varargs
    _get_dct_fun.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_dct_fun', ['type', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_dct_fun', localization, ['type', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_dct_fun(...)' code ##################

    
    
    # SSA begins for try-except statement (line 444)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 445):
    
    # Assigning a Subscript to a Name (line 445):
    
    # Obtaining the type of the subscript
    # Getting the type of 'dtype' (line 445)
    dtype_18046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 55), 'dtype')
    # Obtaining the member 'name' of a type (line 445)
    name_18047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 55), dtype_18046, 'name')
    
    # Obtaining an instance of the builtin type 'dict' (line 445)
    dict_18048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 445)
    # Adding element type (key, value) (line 445)
    str_18049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 16), 'str', 'float64')
    str_18050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 26), 'str', 'ddct%d')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 15), dict_18048, (str_18049, str_18050))
    # Adding element type (key, value) (line 445)
    str_18051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 36), 'str', 'float32')
    str_18052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 46), 'str', 'dct%d')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 15), dict_18048, (str_18051, str_18052))
    
    # Obtaining the member '__getitem__' of a type (line 445)
    getitem___18053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 15), dict_18048, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 445)
    subscript_call_result_18054 = invoke(stypy.reporting.localization.Localization(__file__, 445, 15), getitem___18053, name_18047)
    
    # Assigning a type to the variable 'name' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'name', subscript_call_result_18054)
    # SSA branch for the except part of a try statement (line 444)
    # SSA branch for the except 'KeyError' branch of a try statement (line 444)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 447)
    # Processing the call arguments (line 447)
    str_18056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 25), 'str', 'dtype %s not supported')
    # Getting the type of 'dtype' (line 447)
    dtype_18057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 52), 'dtype', False)
    # Applying the binary operator '%' (line 447)
    result_mod_18058 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 25), '%', str_18056, dtype_18057)
    
    # Processing the call keyword arguments (line 447)
    kwargs_18059 = {}
    # Getting the type of 'ValueError' (line 447)
    ValueError_18055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 447)
    ValueError_call_result_18060 = invoke(stypy.reporting.localization.Localization(__file__, 447, 14), ValueError_18055, *[result_mod_18058], **kwargs_18059)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 447, 8), ValueError_call_result_18060, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 444)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 448)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 449):
    
    # Assigning a Call to a Name (line 449):
    
    # Call to getattr(...): (line 449)
    # Processing the call arguments (line 449)
    # Getting the type of '_fftpack' (line 449)
    _fftpack_18062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 20), '_fftpack', False)
    # Getting the type of 'name' (line 449)
    name_18063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 30), 'name', False)
    # Getting the type of 'type' (line 449)
    type_18064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 37), 'type', False)
    # Applying the binary operator '%' (line 449)
    result_mod_18065 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 30), '%', name_18063, type_18064)
    
    # Processing the call keyword arguments (line 449)
    kwargs_18066 = {}
    # Getting the type of 'getattr' (line 449)
    getattr_18061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'getattr', False)
    # Calling getattr(args, kwargs) (line 449)
    getattr_call_result_18067 = invoke(stypy.reporting.localization.Localization(__file__, 449, 12), getattr_18061, *[_fftpack_18062, result_mod_18065], **kwargs_18066)
    
    # Assigning a type to the variable 'f' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'f', getattr_call_result_18067)
    # SSA branch for the except part of a try statement (line 448)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 448)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'AttributeError' (line 450)
    AttributeError_18068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 11), 'AttributeError')
    # Assigning a type to the variable 'e' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'e', AttributeError_18068)
    
    # Call to ValueError(...): (line 451)
    # Processing the call arguments (line 451)
    
    # Call to str(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'e' (line 451)
    e_18071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 29), 'e', False)
    # Processing the call keyword arguments (line 451)
    kwargs_18072 = {}
    # Getting the type of 'str' (line 451)
    str_18070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 25), 'str', False)
    # Calling str(args, kwargs) (line 451)
    str_call_result_18073 = invoke(stypy.reporting.localization.Localization(__file__, 451, 25), str_18070, *[e_18071], **kwargs_18072)
    
    str_18074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 34), 'str', '. Type %d not understood')
    # Getting the type of 'type' (line 451)
    type_18075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 63), 'type', False)
    # Applying the binary operator '%' (line 451)
    result_mod_18076 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 34), '%', str_18074, type_18075)
    
    # Applying the binary operator '+' (line 451)
    result_add_18077 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 25), '+', str_call_result_18073, result_mod_18076)
    
    # Processing the call keyword arguments (line 451)
    kwargs_18078 = {}
    # Getting the type of 'ValueError' (line 451)
    ValueError_18069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 451)
    ValueError_call_result_18079 = invoke(stypy.reporting.localization.Localization(__file__, 451, 14), ValueError_18069, *[result_add_18077], **kwargs_18078)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 451, 8), ValueError_call_result_18079, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 448)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'f' (line 452)
    f_18080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 11), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type', f_18080)
    
    # ################# End of '_get_dct_fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_dct_fun' in the type store
    # Getting the type of 'stypy_return_type' (line 443)
    stypy_return_type_18081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18081)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_dct_fun'
    return stypy_return_type_18081

# Assigning a type to the variable '_get_dct_fun' (line 443)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 0), '_get_dct_fun', _get_dct_fun)

@norecursion
def _get_norm_mode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_norm_mode'
    module_type_store = module_type_store.open_function_context('_get_norm_mode', 455, 0, False)
    
    # Passed parameters checking function
    _get_norm_mode.stypy_localization = localization
    _get_norm_mode.stypy_type_of_self = None
    _get_norm_mode.stypy_type_store = module_type_store
    _get_norm_mode.stypy_function_name = '_get_norm_mode'
    _get_norm_mode.stypy_param_names_list = ['normalize']
    _get_norm_mode.stypy_varargs_param_name = None
    _get_norm_mode.stypy_kwargs_param_name = None
    _get_norm_mode.stypy_call_defaults = defaults
    _get_norm_mode.stypy_call_varargs = varargs
    _get_norm_mode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_norm_mode', ['normalize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_norm_mode', localization, ['normalize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_norm_mode(...)' code ##################

    
    
    # SSA begins for try-except statement (line 456)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 457):
    
    # Assigning a Subscript to a Name (line 457):
    
    # Obtaining the type of the subscript
    # Getting the type of 'normalize' (line 457)
    normalize_18082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 33), 'normalize')
    
    # Obtaining an instance of the builtin type 'dict' (line 457)
    dict_18083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 457)
    # Adding element type (key, value) (line 457)
    # Getting the type of 'None' (line 457)
    None_18084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 14), 'None')
    int_18085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 19), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 13), dict_18083, (None_18084, int_18085))
    # Adding element type (key, value) (line 457)
    str_18086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 22), 'str', 'ortho')
    int_18087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 30), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 13), dict_18083, (str_18086, int_18087))
    
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___18088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 13), dict_18083, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_18089 = invoke(stypy.reporting.localization.Localization(__file__, 457, 13), getitem___18088, normalize_18082)
    
    # Assigning a type to the variable 'nm' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'nm', subscript_call_result_18089)
    # SSA branch for the except part of a try statement (line 456)
    # SSA branch for the except 'KeyError' branch of a try statement (line 456)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 459)
    # Processing the call arguments (line 459)
    str_18091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 25), 'str', 'Unknown normalize mode %s')
    # Getting the type of 'normalize' (line 459)
    normalize_18092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 55), 'normalize', False)
    # Applying the binary operator '%' (line 459)
    result_mod_18093 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 25), '%', str_18091, normalize_18092)
    
    # Processing the call keyword arguments (line 459)
    kwargs_18094 = {}
    # Getting the type of 'ValueError' (line 459)
    ValueError_18090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 459)
    ValueError_call_result_18095 = invoke(stypy.reporting.localization.Localization(__file__, 459, 14), ValueError_18090, *[result_mod_18093], **kwargs_18094)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 459, 8), ValueError_call_result_18095, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 456)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'nm' (line 460)
    nm_18096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 11), 'nm')
    # Assigning a type to the variable 'stypy_return_type' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'stypy_return_type', nm_18096)
    
    # ################# End of '_get_norm_mode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_norm_mode' in the type store
    # Getting the type of 'stypy_return_type' (line 455)
    stypy_return_type_18097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18097)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_norm_mode'
    return stypy_return_type_18097

# Assigning a type to the variable '_get_norm_mode' (line 455)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 0), '_get_norm_mode', _get_norm_mode)

@norecursion
def __fix_shape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__fix_shape'
    module_type_store = module_type_store.open_function_context('__fix_shape', 463, 0, False)
    
    # Passed parameters checking function
    __fix_shape.stypy_localization = localization
    __fix_shape.stypy_type_of_self = None
    __fix_shape.stypy_type_store = module_type_store
    __fix_shape.stypy_function_name = '__fix_shape'
    __fix_shape.stypy_param_names_list = ['x', 'n', 'axis', 'dct_or_dst']
    __fix_shape.stypy_varargs_param_name = None
    __fix_shape.stypy_kwargs_param_name = None
    __fix_shape.stypy_call_defaults = defaults
    __fix_shape.stypy_call_varargs = varargs
    __fix_shape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__fix_shape', ['x', 'n', 'axis', 'dct_or_dst'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__fix_shape', localization, ['x', 'n', 'axis', 'dct_or_dst'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__fix_shape(...)' code ##################

    
    # Assigning a Call to a Name (line 464):
    
    # Assigning a Call to a Name (line 464):
    
    # Call to _asfarray(...): (line 464)
    # Processing the call arguments (line 464)
    # Getting the type of 'x' (line 464)
    x_18099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 20), 'x', False)
    # Processing the call keyword arguments (line 464)
    kwargs_18100 = {}
    # Getting the type of '_asfarray' (line 464)
    _asfarray_18098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 10), '_asfarray', False)
    # Calling _asfarray(args, kwargs) (line 464)
    _asfarray_call_result_18101 = invoke(stypy.reporting.localization.Localization(__file__, 464, 10), _asfarray_18098, *[x_18099], **kwargs_18100)
    
    # Assigning a type to the variable 'tmp' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'tmp', _asfarray_call_result_18101)
    
    # Assigning a Call to a Name (line 465):
    
    # Assigning a Call to a Name (line 465):
    
    # Call to _datacopied(...): (line 465)
    # Processing the call arguments (line 465)
    # Getting the type of 'tmp' (line 465)
    tmp_18103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 28), 'tmp', False)
    # Getting the type of 'x' (line 465)
    x_18104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 33), 'x', False)
    # Processing the call keyword arguments (line 465)
    kwargs_18105 = {}
    # Getting the type of '_datacopied' (line 465)
    _datacopied_18102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 465)
    _datacopied_call_result_18106 = invoke(stypy.reporting.localization.Localization(__file__, 465, 16), _datacopied_18102, *[tmp_18103, x_18104], **kwargs_18105)
    
    # Assigning a type to the variable 'copy_made' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'copy_made', _datacopied_call_result_18106)
    
    # Type idiom detected: calculating its left and rigth part (line 466)
    # Getting the type of 'n' (line 466)
    n_18107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 7), 'n')
    # Getting the type of 'None' (line 466)
    None_18108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'None')
    
    (may_be_18109, more_types_in_union_18110) = may_be_none(n_18107, None_18108)

    if may_be_18109:

        if more_types_in_union_18110:
            # Runtime conditional SSA (line 466)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 467):
        
        # Assigning a Subscript to a Name (line 467):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 467)
        axis_18111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 22), 'axis')
        # Getting the type of 'tmp' (line 467)
        tmp_18112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'tmp')
        # Obtaining the member 'shape' of a type (line 467)
        shape_18113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 12), tmp_18112, 'shape')
        # Obtaining the member '__getitem__' of a type (line 467)
        getitem___18114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 12), shape_18113, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 467)
        subscript_call_result_18115 = invoke(stypy.reporting.localization.Localization(__file__, 467, 12), getitem___18114, axis_18111)
        
        # Assigning a type to the variable 'n' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'n', subscript_call_result_18115)

        if more_types_in_union_18110:
            # Runtime conditional SSA for else branch (line 466)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_18109) or more_types_in_union_18110):
        
        
        # Getting the type of 'n' (line 468)
        n_18116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 9), 'n')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 468)
        axis_18117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 24), 'axis')
        # Getting the type of 'tmp' (line 468)
        tmp_18118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 14), 'tmp')
        # Obtaining the member 'shape' of a type (line 468)
        shape_18119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 14), tmp_18118, 'shape')
        # Obtaining the member '__getitem__' of a type (line 468)
        getitem___18120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 14), shape_18119, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 468)
        subscript_call_result_18121 = invoke(stypy.reporting.localization.Localization(__file__, 468, 14), getitem___18120, axis_18117)
        
        # Applying the binary operator '!=' (line 468)
        result_ne_18122 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 9), '!=', n_18116, subscript_call_result_18121)
        
        # Testing the type of an if condition (line 468)
        if_condition_18123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 9), result_ne_18122)
        # Assigning a type to the variable 'if_condition_18123' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 9), 'if_condition_18123', if_condition_18123)
        # SSA begins for if statement (line 468)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 469):
        
        # Assigning a Subscript to a Name (line 469):
        
        # Obtaining the type of the subscript
        int_18124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 8), 'int')
        
        # Call to _fix_shape(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'tmp' (line 469)
        tmp_18126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 37), 'tmp', False)
        # Getting the type of 'n' (line 469)
        n_18127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 42), 'n', False)
        # Getting the type of 'axis' (line 469)
        axis_18128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 45), 'axis', False)
        # Processing the call keyword arguments (line 469)
        kwargs_18129 = {}
        # Getting the type of '_fix_shape' (line 469)
        _fix_shape_18125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 26), '_fix_shape', False)
        # Calling _fix_shape(args, kwargs) (line 469)
        _fix_shape_call_result_18130 = invoke(stypy.reporting.localization.Localization(__file__, 469, 26), _fix_shape_18125, *[tmp_18126, n_18127, axis_18128], **kwargs_18129)
        
        # Obtaining the member '__getitem__' of a type (line 469)
        getitem___18131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 8), _fix_shape_call_result_18130, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 469)
        subscript_call_result_18132 = invoke(stypy.reporting.localization.Localization(__file__, 469, 8), getitem___18131, int_18124)
        
        # Assigning a type to the variable 'tuple_var_assignment_17599' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'tuple_var_assignment_17599', subscript_call_result_18132)
        
        # Assigning a Subscript to a Name (line 469):
        
        # Obtaining the type of the subscript
        int_18133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 8), 'int')
        
        # Call to _fix_shape(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'tmp' (line 469)
        tmp_18135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 37), 'tmp', False)
        # Getting the type of 'n' (line 469)
        n_18136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 42), 'n', False)
        # Getting the type of 'axis' (line 469)
        axis_18137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 45), 'axis', False)
        # Processing the call keyword arguments (line 469)
        kwargs_18138 = {}
        # Getting the type of '_fix_shape' (line 469)
        _fix_shape_18134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 26), '_fix_shape', False)
        # Calling _fix_shape(args, kwargs) (line 469)
        _fix_shape_call_result_18139 = invoke(stypy.reporting.localization.Localization(__file__, 469, 26), _fix_shape_18134, *[tmp_18135, n_18136, axis_18137], **kwargs_18138)
        
        # Obtaining the member '__getitem__' of a type (line 469)
        getitem___18140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 8), _fix_shape_call_result_18139, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 469)
        subscript_call_result_18141 = invoke(stypy.reporting.localization.Localization(__file__, 469, 8), getitem___18140, int_18133)
        
        # Assigning a type to the variable 'tuple_var_assignment_17600' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'tuple_var_assignment_17600', subscript_call_result_18141)
        
        # Assigning a Name to a Name (line 469):
        # Getting the type of 'tuple_var_assignment_17599' (line 469)
        tuple_var_assignment_17599_18142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'tuple_var_assignment_17599')
        # Assigning a type to the variable 'tmp' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'tmp', tuple_var_assignment_17599_18142)
        
        # Assigning a Name to a Name (line 469):
        # Getting the type of 'tuple_var_assignment_17600' (line 469)
        tuple_var_assignment_17600_18143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'tuple_var_assignment_17600')
        # Assigning a type to the variable 'copy_made2' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 13), 'copy_made2', tuple_var_assignment_17600_18143)
        
        # Assigning a BoolOp to a Name (line 470):
        
        # Assigning a BoolOp to a Name (line 470):
        
        # Evaluating a boolean operation
        # Getting the type of 'copy_made' (line 470)
        copy_made_18144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 20), 'copy_made')
        # Getting the type of 'copy_made2' (line 470)
        copy_made2_18145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 33), 'copy_made2')
        # Applying the binary operator 'or' (line 470)
        result_or_keyword_18146 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 20), 'or', copy_made_18144, copy_made2_18145)
        
        # Assigning a type to the variable 'copy_made' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'copy_made', result_or_keyword_18146)
        # SSA join for if statement (line 468)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_18109 and more_types_in_union_18110):
            # SSA join for if statement (line 466)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'n' (line 471)
    n_18147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 7), 'n')
    int_18148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 11), 'int')
    # Applying the binary operator '<' (line 471)
    result_lt_18149 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 7), '<', n_18147, int_18148)
    
    # Testing the type of an if condition (line 471)
    if_condition_18150 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 471, 4), result_lt_18149)
    # Assigning a type to the variable 'if_condition_18150' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'if_condition_18150', if_condition_18150)
    # SSA begins for if statement (line 471)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 472)
    # Processing the call arguments (line 472)
    str_18152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 25), 'str', 'Invalid number of %s data points (%d) specified.')
    
    # Obtaining an instance of the builtin type 'tuple' (line 473)
    tuple_18153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 473)
    # Adding element type (line 473)
    # Getting the type of 'dct_or_dst' (line 473)
    dct_or_dst_18154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 46), 'dct_or_dst', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 46), tuple_18153, dct_or_dst_18154)
    # Adding element type (line 473)
    # Getting the type of 'n' (line 473)
    n_18155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 58), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 46), tuple_18153, n_18155)
    
    # Applying the binary operator '%' (line 472)
    result_mod_18156 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 25), '%', str_18152, tuple_18153)
    
    # Processing the call keyword arguments (line 472)
    kwargs_18157 = {}
    # Getting the type of 'ValueError' (line 472)
    ValueError_18151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 472)
    ValueError_call_result_18158 = invoke(stypy.reporting.localization.Localization(__file__, 472, 14), ValueError_18151, *[result_mod_18156], **kwargs_18157)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 472, 8), ValueError_call_result_18158, 'raise parameter', BaseException)
    # SSA join for if statement (line 471)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 474)
    tuple_18159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 474)
    # Adding element type (line 474)
    # Getting the type of 'tmp' (line 474)
    tmp_18160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 11), 'tmp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 11), tuple_18159, tmp_18160)
    # Adding element type (line 474)
    # Getting the type of 'n' (line 474)
    n_18161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 11), tuple_18159, n_18161)
    # Adding element type (line 474)
    # Getting the type of 'copy_made' (line 474)
    copy_made_18162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 19), 'copy_made')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 11), tuple_18159, copy_made_18162)
    
    # Assigning a type to the variable 'stypy_return_type' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'stypy_return_type', tuple_18159)
    
    # ################# End of '__fix_shape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__fix_shape' in the type store
    # Getting the type of 'stypy_return_type' (line 463)
    stypy_return_type_18163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18163)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__fix_shape'
    return stypy_return_type_18163

# Assigning a type to the variable '__fix_shape' (line 463)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 0), '__fix_shape', __fix_shape)

@norecursion
def _raw_dct(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_raw_dct'
    module_type_store = module_type_store.open_function_context('_raw_dct', 477, 0, False)
    
    # Passed parameters checking function
    _raw_dct.stypy_localization = localization
    _raw_dct.stypy_type_of_self = None
    _raw_dct.stypy_type_store = module_type_store
    _raw_dct.stypy_function_name = '_raw_dct'
    _raw_dct.stypy_param_names_list = ['x0', 'type', 'n', 'axis', 'nm', 'overwrite_x']
    _raw_dct.stypy_varargs_param_name = None
    _raw_dct.stypy_kwargs_param_name = None
    _raw_dct.stypy_call_defaults = defaults
    _raw_dct.stypy_call_varargs = varargs
    _raw_dct.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_raw_dct', ['x0', 'type', 'n', 'axis', 'nm', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_raw_dct', localization, ['x0', 'type', 'n', 'axis', 'nm', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_raw_dct(...)' code ##################

    
    # Assigning a Call to a Name (line 478):
    
    # Assigning a Call to a Name (line 478):
    
    # Call to _get_dct_fun(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'type' (line 478)
    type_18165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 21), 'type', False)
    # Getting the type of 'x0' (line 478)
    x0_18166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 27), 'x0', False)
    # Obtaining the member 'dtype' of a type (line 478)
    dtype_18167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 27), x0_18166, 'dtype')
    # Processing the call keyword arguments (line 478)
    kwargs_18168 = {}
    # Getting the type of '_get_dct_fun' (line 478)
    _get_dct_fun_18164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), '_get_dct_fun', False)
    # Calling _get_dct_fun(args, kwargs) (line 478)
    _get_dct_fun_call_result_18169 = invoke(stypy.reporting.localization.Localization(__file__, 478, 8), _get_dct_fun_18164, *[type_18165, dtype_18167], **kwargs_18168)
    
    # Assigning a type to the variable 'f' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'f', _get_dct_fun_call_result_18169)
    
    # Call to _eval_fun(...): (line 479)
    # Processing the call arguments (line 479)
    # Getting the type of 'f' (line 479)
    f_18171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 21), 'f', False)
    # Getting the type of 'x0' (line 479)
    x0_18172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 24), 'x0', False)
    # Getting the type of 'n' (line 479)
    n_18173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 28), 'n', False)
    # Getting the type of 'axis' (line 479)
    axis_18174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 31), 'axis', False)
    # Getting the type of 'nm' (line 479)
    nm_18175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 37), 'nm', False)
    # Getting the type of 'overwrite_x' (line 479)
    overwrite_x_18176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 41), 'overwrite_x', False)
    # Processing the call keyword arguments (line 479)
    kwargs_18177 = {}
    # Getting the type of '_eval_fun' (line 479)
    _eval_fun_18170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 11), '_eval_fun', False)
    # Calling _eval_fun(args, kwargs) (line 479)
    _eval_fun_call_result_18178 = invoke(stypy.reporting.localization.Localization(__file__, 479, 11), _eval_fun_18170, *[f_18171, x0_18172, n_18173, axis_18174, nm_18175, overwrite_x_18176], **kwargs_18177)
    
    # Assigning a type to the variable 'stypy_return_type' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'stypy_return_type', _eval_fun_call_result_18178)
    
    # ################# End of '_raw_dct(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_raw_dct' in the type store
    # Getting the type of 'stypy_return_type' (line 477)
    stypy_return_type_18179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18179)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_raw_dct'
    return stypy_return_type_18179

# Assigning a type to the variable '_raw_dct' (line 477)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 0), '_raw_dct', _raw_dct)

@norecursion
def _raw_dst(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_raw_dst'
    module_type_store = module_type_store.open_function_context('_raw_dst', 482, 0, False)
    
    # Passed parameters checking function
    _raw_dst.stypy_localization = localization
    _raw_dst.stypy_type_of_self = None
    _raw_dst.stypy_type_store = module_type_store
    _raw_dst.stypy_function_name = '_raw_dst'
    _raw_dst.stypy_param_names_list = ['x0', 'type', 'n', 'axis', 'nm', 'overwrite_x']
    _raw_dst.stypy_varargs_param_name = None
    _raw_dst.stypy_kwargs_param_name = None
    _raw_dst.stypy_call_defaults = defaults
    _raw_dst.stypy_call_varargs = varargs
    _raw_dst.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_raw_dst', ['x0', 'type', 'n', 'axis', 'nm', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_raw_dst', localization, ['x0', 'type', 'n', 'axis', 'nm', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_raw_dst(...)' code ##################

    
    # Assigning a Call to a Name (line 483):
    
    # Assigning a Call to a Name (line 483):
    
    # Call to _get_dst_fun(...): (line 483)
    # Processing the call arguments (line 483)
    # Getting the type of 'type' (line 483)
    type_18181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 21), 'type', False)
    # Getting the type of 'x0' (line 483)
    x0_18182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 27), 'x0', False)
    # Obtaining the member 'dtype' of a type (line 483)
    dtype_18183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 27), x0_18182, 'dtype')
    # Processing the call keyword arguments (line 483)
    kwargs_18184 = {}
    # Getting the type of '_get_dst_fun' (line 483)
    _get_dst_fun_18180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), '_get_dst_fun', False)
    # Calling _get_dst_fun(args, kwargs) (line 483)
    _get_dst_fun_call_result_18185 = invoke(stypy.reporting.localization.Localization(__file__, 483, 8), _get_dst_fun_18180, *[type_18181, dtype_18183], **kwargs_18184)
    
    # Assigning a type to the variable 'f' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'f', _get_dst_fun_call_result_18185)
    
    # Call to _eval_fun(...): (line 484)
    # Processing the call arguments (line 484)
    # Getting the type of 'f' (line 484)
    f_18187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 21), 'f', False)
    # Getting the type of 'x0' (line 484)
    x0_18188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 24), 'x0', False)
    # Getting the type of 'n' (line 484)
    n_18189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 28), 'n', False)
    # Getting the type of 'axis' (line 484)
    axis_18190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 31), 'axis', False)
    # Getting the type of 'nm' (line 484)
    nm_18191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 37), 'nm', False)
    # Getting the type of 'overwrite_x' (line 484)
    overwrite_x_18192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 41), 'overwrite_x', False)
    # Processing the call keyword arguments (line 484)
    kwargs_18193 = {}
    # Getting the type of '_eval_fun' (line 484)
    _eval_fun_18186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 11), '_eval_fun', False)
    # Calling _eval_fun(args, kwargs) (line 484)
    _eval_fun_call_result_18194 = invoke(stypy.reporting.localization.Localization(__file__, 484, 11), _eval_fun_18186, *[f_18187, x0_18188, n_18189, axis_18190, nm_18191, overwrite_x_18192], **kwargs_18193)
    
    # Assigning a type to the variable 'stypy_return_type' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'stypy_return_type', _eval_fun_call_result_18194)
    
    # ################# End of '_raw_dst(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_raw_dst' in the type store
    # Getting the type of 'stypy_return_type' (line 482)
    stypy_return_type_18195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18195)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_raw_dst'
    return stypy_return_type_18195

# Assigning a type to the variable '_raw_dst' (line 482)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 0), '_raw_dst', _raw_dst)

@norecursion
def _eval_fun(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_eval_fun'
    module_type_store = module_type_store.open_function_context('_eval_fun', 487, 0, False)
    
    # Passed parameters checking function
    _eval_fun.stypy_localization = localization
    _eval_fun.stypy_type_of_self = None
    _eval_fun.stypy_type_store = module_type_store
    _eval_fun.stypy_function_name = '_eval_fun'
    _eval_fun.stypy_param_names_list = ['f', 'tmp', 'n', 'axis', 'nm', 'overwrite_x']
    _eval_fun.stypy_varargs_param_name = None
    _eval_fun.stypy_kwargs_param_name = None
    _eval_fun.stypy_call_defaults = defaults
    _eval_fun.stypy_call_varargs = varargs
    _eval_fun.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_eval_fun', ['f', 'tmp', 'n', 'axis', 'nm', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_eval_fun', localization, ['f', 'tmp', 'n', 'axis', 'nm', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_eval_fun(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'axis' (line 488)
    axis_18196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 7), 'axis')
    int_18197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 15), 'int')
    # Applying the binary operator '==' (line 488)
    result_eq_18198 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 7), '==', axis_18196, int_18197)
    
    
    # Getting the type of 'axis' (line 488)
    axis_18199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 21), 'axis')
    
    # Call to len(...): (line 488)
    # Processing the call arguments (line 488)
    # Getting the type of 'tmp' (line 488)
    tmp_18201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 33), 'tmp', False)
    # Obtaining the member 'shape' of a type (line 488)
    shape_18202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 33), tmp_18201, 'shape')
    # Processing the call keyword arguments (line 488)
    kwargs_18203 = {}
    # Getting the type of 'len' (line 488)
    len_18200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 29), 'len', False)
    # Calling len(args, kwargs) (line 488)
    len_call_result_18204 = invoke(stypy.reporting.localization.Localization(__file__, 488, 29), len_18200, *[shape_18202], **kwargs_18203)
    
    int_18205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 46), 'int')
    # Applying the binary operator '-' (line 488)
    result_sub_18206 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 29), '-', len_call_result_18204, int_18205)
    
    # Applying the binary operator '==' (line 488)
    result_eq_18207 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 21), '==', axis_18199, result_sub_18206)
    
    # Applying the binary operator 'or' (line 488)
    result_or_keyword_18208 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 7), 'or', result_eq_18198, result_eq_18207)
    
    # Testing the type of an if condition (line 488)
    if_condition_18209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 488, 4), result_or_keyword_18208)
    # Assigning a type to the variable 'if_condition_18209' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'if_condition_18209', if_condition_18209)
    # SSA begins for if statement (line 488)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to f(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'tmp' (line 489)
    tmp_18211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 17), 'tmp', False)
    # Getting the type of 'n' (line 489)
    n_18212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 22), 'n', False)
    # Getting the type of 'nm' (line 489)
    nm_18213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 25), 'nm', False)
    # Getting the type of 'overwrite_x' (line 489)
    overwrite_x_18214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 29), 'overwrite_x', False)
    # Processing the call keyword arguments (line 489)
    kwargs_18215 = {}
    # Getting the type of 'f' (line 489)
    f_18210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), 'f', False)
    # Calling f(args, kwargs) (line 489)
    f_call_result_18216 = invoke(stypy.reporting.localization.Localization(__file__, 489, 15), f_18210, *[tmp_18211, n_18212, nm_18213, overwrite_x_18214], **kwargs_18215)
    
    # Assigning a type to the variable 'stypy_return_type' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'stypy_return_type', f_call_result_18216)
    # SSA join for if statement (line 488)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 491):
    
    # Assigning a Call to a Name (line 491):
    
    # Call to swapaxes(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of 'tmp' (line 491)
    tmp_18219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 22), 'tmp', False)
    # Getting the type of 'axis' (line 491)
    axis_18220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 27), 'axis', False)
    int_18221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 33), 'int')
    # Processing the call keyword arguments (line 491)
    kwargs_18222 = {}
    # Getting the type of 'np' (line 491)
    np_18217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 10), 'np', False)
    # Obtaining the member 'swapaxes' of a type (line 491)
    swapaxes_18218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 10), np_18217, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 491)
    swapaxes_call_result_18223 = invoke(stypy.reporting.localization.Localization(__file__, 491, 10), swapaxes_18218, *[tmp_18219, axis_18220, int_18221], **kwargs_18222)
    
    # Assigning a type to the variable 'tmp' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'tmp', swapaxes_call_result_18223)
    
    # Assigning a Call to a Name (line 492):
    
    # Assigning a Call to a Name (line 492):
    
    # Call to f(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'tmp' (line 492)
    tmp_18225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'tmp', False)
    # Getting the type of 'n' (line 492)
    n_18226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 17), 'n', False)
    # Getting the type of 'nm' (line 492)
    nm_18227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 20), 'nm', False)
    # Getting the type of 'overwrite_x' (line 492)
    overwrite_x_18228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 24), 'overwrite_x', False)
    # Processing the call keyword arguments (line 492)
    kwargs_18229 = {}
    # Getting the type of 'f' (line 492)
    f_18224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 10), 'f', False)
    # Calling f(args, kwargs) (line 492)
    f_call_result_18230 = invoke(stypy.reporting.localization.Localization(__file__, 492, 10), f_18224, *[tmp_18225, n_18226, nm_18227, overwrite_x_18228], **kwargs_18229)
    
    # Assigning a type to the variable 'tmp' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'tmp', f_call_result_18230)
    
    # Call to swapaxes(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of 'tmp' (line 493)
    tmp_18233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 23), 'tmp', False)
    # Getting the type of 'axis' (line 493)
    axis_18234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 28), 'axis', False)
    int_18235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 34), 'int')
    # Processing the call keyword arguments (line 493)
    kwargs_18236 = {}
    # Getting the type of 'np' (line 493)
    np_18231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 11), 'np', False)
    # Obtaining the member 'swapaxes' of a type (line 493)
    swapaxes_18232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 11), np_18231, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 493)
    swapaxes_call_result_18237 = invoke(stypy.reporting.localization.Localization(__file__, 493, 11), swapaxes_18232, *[tmp_18233, axis_18234, int_18235], **kwargs_18236)
    
    # Assigning a type to the variable 'stypy_return_type' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'stypy_return_type', swapaxes_call_result_18237)
    
    # ################# End of '_eval_fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_eval_fun' in the type store
    # Getting the type of 'stypy_return_type' (line 487)
    stypy_return_type_18238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18238)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_eval_fun'
    return stypy_return_type_18238

# Assigning a type to the variable '_eval_fun' (line 487)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 0), '_eval_fun', _eval_fun)

@norecursion
def _dct(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 496)
    None_18239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 20), 'None')
    int_18240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 31), 'int')
    # Getting the type of 'False' (line 496)
    False_18241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 47), 'False')
    # Getting the type of 'None' (line 496)
    None_18242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 64), 'None')
    defaults = [None_18239, int_18240, False_18241, None_18242]
    # Create a new context for function '_dct'
    module_type_store = module_type_store.open_function_context('_dct', 496, 0, False)
    
    # Passed parameters checking function
    _dct.stypy_localization = localization
    _dct.stypy_type_of_self = None
    _dct.stypy_type_store = module_type_store
    _dct.stypy_function_name = '_dct'
    _dct.stypy_param_names_list = ['x', 'type', 'n', 'axis', 'overwrite_x', 'normalize']
    _dct.stypy_varargs_param_name = None
    _dct.stypy_kwargs_param_name = None
    _dct.stypy_call_defaults = defaults
    _dct.stypy_call_varargs = varargs
    _dct.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_dct', ['x', 'type', 'n', 'axis', 'overwrite_x', 'normalize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_dct', localization, ['x', 'type', 'n', 'axis', 'overwrite_x', 'normalize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_dct(...)' code ##################

    str_18243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, (-1)), 'str', '\n    Return Discrete Cosine Transform of arbitrary type sequence x.\n\n    Parameters\n    ----------\n    x : array_like\n        input array.\n    n : int, optional\n        Length of the transform.  If ``n < x.shape[axis]``, `x` is\n        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The\n        default results in ``n = x.shape[axis]``.\n    axis : int, optional\n        Axis along which the dct is computed; the default is over the\n        last axis (i.e., ``axis=-1``).\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    z : ndarray\n\n    ')
    
    # Assigning a Call to a Tuple (line 519):
    
    # Assigning a Subscript to a Name (line 519):
    
    # Obtaining the type of the subscript
    int_18244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 4), 'int')
    
    # Call to __fix_shape(...): (line 519)
    # Processing the call arguments (line 519)
    # Getting the type of 'x' (line 519)
    x_18246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 35), 'x', False)
    # Getting the type of 'n' (line 519)
    n_18247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 38), 'n', False)
    # Getting the type of 'axis' (line 519)
    axis_18248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 41), 'axis', False)
    str_18249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 47), 'str', 'DCT')
    # Processing the call keyword arguments (line 519)
    kwargs_18250 = {}
    # Getting the type of '__fix_shape' (line 519)
    fix_shape_18245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 23), '__fix_shape', False)
    # Calling __fix_shape(args, kwargs) (line 519)
    fix_shape_call_result_18251 = invoke(stypy.reporting.localization.Localization(__file__, 519, 23), fix_shape_18245, *[x_18246, n_18247, axis_18248, str_18249], **kwargs_18250)
    
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___18252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 4), fix_shape_call_result_18251, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_18253 = invoke(stypy.reporting.localization.Localization(__file__, 519, 4), getitem___18252, int_18244)
    
    # Assigning a type to the variable 'tuple_var_assignment_17601' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'tuple_var_assignment_17601', subscript_call_result_18253)
    
    # Assigning a Subscript to a Name (line 519):
    
    # Obtaining the type of the subscript
    int_18254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 4), 'int')
    
    # Call to __fix_shape(...): (line 519)
    # Processing the call arguments (line 519)
    # Getting the type of 'x' (line 519)
    x_18256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 35), 'x', False)
    # Getting the type of 'n' (line 519)
    n_18257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 38), 'n', False)
    # Getting the type of 'axis' (line 519)
    axis_18258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 41), 'axis', False)
    str_18259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 47), 'str', 'DCT')
    # Processing the call keyword arguments (line 519)
    kwargs_18260 = {}
    # Getting the type of '__fix_shape' (line 519)
    fix_shape_18255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 23), '__fix_shape', False)
    # Calling __fix_shape(args, kwargs) (line 519)
    fix_shape_call_result_18261 = invoke(stypy.reporting.localization.Localization(__file__, 519, 23), fix_shape_18255, *[x_18256, n_18257, axis_18258, str_18259], **kwargs_18260)
    
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___18262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 4), fix_shape_call_result_18261, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_18263 = invoke(stypy.reporting.localization.Localization(__file__, 519, 4), getitem___18262, int_18254)
    
    # Assigning a type to the variable 'tuple_var_assignment_17602' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'tuple_var_assignment_17602', subscript_call_result_18263)
    
    # Assigning a Subscript to a Name (line 519):
    
    # Obtaining the type of the subscript
    int_18264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 4), 'int')
    
    # Call to __fix_shape(...): (line 519)
    # Processing the call arguments (line 519)
    # Getting the type of 'x' (line 519)
    x_18266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 35), 'x', False)
    # Getting the type of 'n' (line 519)
    n_18267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 38), 'n', False)
    # Getting the type of 'axis' (line 519)
    axis_18268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 41), 'axis', False)
    str_18269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 47), 'str', 'DCT')
    # Processing the call keyword arguments (line 519)
    kwargs_18270 = {}
    # Getting the type of '__fix_shape' (line 519)
    fix_shape_18265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 23), '__fix_shape', False)
    # Calling __fix_shape(args, kwargs) (line 519)
    fix_shape_call_result_18271 = invoke(stypy.reporting.localization.Localization(__file__, 519, 23), fix_shape_18265, *[x_18266, n_18267, axis_18268, str_18269], **kwargs_18270)
    
    # Obtaining the member '__getitem__' of a type (line 519)
    getitem___18272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 4), fix_shape_call_result_18271, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 519)
    subscript_call_result_18273 = invoke(stypy.reporting.localization.Localization(__file__, 519, 4), getitem___18272, int_18264)
    
    # Assigning a type to the variable 'tuple_var_assignment_17603' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'tuple_var_assignment_17603', subscript_call_result_18273)
    
    # Assigning a Name to a Name (line 519):
    # Getting the type of 'tuple_var_assignment_17601' (line 519)
    tuple_var_assignment_17601_18274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'tuple_var_assignment_17601')
    # Assigning a type to the variable 'x0' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'x0', tuple_var_assignment_17601_18274)
    
    # Assigning a Name to a Name (line 519):
    # Getting the type of 'tuple_var_assignment_17602' (line 519)
    tuple_var_assignment_17602_18275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'tuple_var_assignment_17602')
    # Assigning a type to the variable 'n' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'n', tuple_var_assignment_17602_18275)
    
    # Assigning a Name to a Name (line 519):
    # Getting the type of 'tuple_var_assignment_17603' (line 519)
    tuple_var_assignment_17603_18276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'tuple_var_assignment_17603')
    # Assigning a type to the variable 'copy_made' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 11), 'copy_made', tuple_var_assignment_17603_18276)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'type' (line 520)
    type_18277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 7), 'type')
    int_18278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 15), 'int')
    # Applying the binary operator '==' (line 520)
    result_eq_18279 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 7), '==', type_18277, int_18278)
    
    
    # Getting the type of 'n' (line 520)
    n_18280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 21), 'n')
    int_18281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 25), 'int')
    # Applying the binary operator '<' (line 520)
    result_lt_18282 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 21), '<', n_18280, int_18281)
    
    # Applying the binary operator 'and' (line 520)
    result_and_keyword_18283 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 7), 'and', result_eq_18279, result_lt_18282)
    
    # Testing the type of an if condition (line 520)
    if_condition_18284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 520, 4), result_and_keyword_18283)
    # Assigning a type to the variable 'if_condition_18284' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'if_condition_18284', if_condition_18284)
    # SSA begins for if statement (line 520)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 521)
    # Processing the call arguments (line 521)
    str_18286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 25), 'str', 'DCT-I is not defined for size < 2')
    # Processing the call keyword arguments (line 521)
    kwargs_18287 = {}
    # Getting the type of 'ValueError' (line 521)
    ValueError_18285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 521)
    ValueError_call_result_18288 = invoke(stypy.reporting.localization.Localization(__file__, 521, 14), ValueError_18285, *[str_18286], **kwargs_18287)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 521, 8), ValueError_call_result_18288, 'raise parameter', BaseException)
    # SSA join for if statement (line 520)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 522):
    
    # Assigning a BoolOp to a Name (line 522):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_x' (line 522)
    overwrite_x_18289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 18), 'overwrite_x')
    # Getting the type of 'copy_made' (line 522)
    copy_made_18290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 33), 'copy_made')
    # Applying the binary operator 'or' (line 522)
    result_or_keyword_18291 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 18), 'or', overwrite_x_18289, copy_made_18290)
    
    # Assigning a type to the variable 'overwrite_x' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'overwrite_x', result_or_keyword_18291)
    
    # Assigning a Call to a Name (line 523):
    
    # Assigning a Call to a Name (line 523):
    
    # Call to _get_norm_mode(...): (line 523)
    # Processing the call arguments (line 523)
    # Getting the type of 'normalize' (line 523)
    normalize_18293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 24), 'normalize', False)
    # Processing the call keyword arguments (line 523)
    kwargs_18294 = {}
    # Getting the type of '_get_norm_mode' (line 523)
    _get_norm_mode_18292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 9), '_get_norm_mode', False)
    # Calling _get_norm_mode(args, kwargs) (line 523)
    _get_norm_mode_call_result_18295 = invoke(stypy.reporting.localization.Localization(__file__, 523, 9), _get_norm_mode_18292, *[normalize_18293], **kwargs_18294)
    
    # Assigning a type to the variable 'nm' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'nm', _get_norm_mode_call_result_18295)
    
    
    # Call to iscomplexobj(...): (line 524)
    # Processing the call arguments (line 524)
    # Getting the type of 'x0' (line 524)
    x0_18298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 23), 'x0', False)
    # Processing the call keyword arguments (line 524)
    kwargs_18299 = {}
    # Getting the type of 'np' (line 524)
    np_18296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 7), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 524)
    iscomplexobj_18297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 7), np_18296, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 524)
    iscomplexobj_call_result_18300 = invoke(stypy.reporting.localization.Localization(__file__, 524, 7), iscomplexobj_18297, *[x0_18298], **kwargs_18299)
    
    # Testing the type of an if condition (line 524)
    if_condition_18301 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 524, 4), iscomplexobj_call_result_18300)
    # Assigning a type to the variable 'if_condition_18301' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'if_condition_18301', if_condition_18301)
    # SSA begins for if statement (line 524)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _raw_dct(...): (line 525)
    # Processing the call arguments (line 525)
    # Getting the type of 'x0' (line 525)
    x0_18303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 25), 'x0', False)
    # Obtaining the member 'real' of a type (line 525)
    real_18304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 25), x0_18303, 'real')
    # Getting the type of 'type' (line 525)
    type_18305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 34), 'type', False)
    # Getting the type of 'n' (line 525)
    n_18306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 40), 'n', False)
    # Getting the type of 'axis' (line 525)
    axis_18307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 43), 'axis', False)
    # Getting the type of 'nm' (line 525)
    nm_18308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 49), 'nm', False)
    # Getting the type of 'overwrite_x' (line 525)
    overwrite_x_18309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 53), 'overwrite_x', False)
    # Processing the call keyword arguments (line 525)
    kwargs_18310 = {}
    # Getting the type of '_raw_dct' (line 525)
    _raw_dct_18302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), '_raw_dct', False)
    # Calling _raw_dct(args, kwargs) (line 525)
    _raw_dct_call_result_18311 = invoke(stypy.reporting.localization.Localization(__file__, 525, 16), _raw_dct_18302, *[real_18304, type_18305, n_18306, axis_18307, nm_18308, overwrite_x_18309], **kwargs_18310)
    
    complex_18312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 68), 'complex')
    
    # Call to _raw_dct(...): (line 526)
    # Processing the call arguments (line 526)
    # Getting the type of 'x0' (line 526)
    x0_18314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 25), 'x0', False)
    # Obtaining the member 'imag' of a type (line 526)
    imag_18315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 25), x0_18314, 'imag')
    # Getting the type of 'type' (line 526)
    type_18316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 34), 'type', False)
    # Getting the type of 'n' (line 526)
    n_18317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 40), 'n', False)
    # Getting the type of 'axis' (line 526)
    axis_18318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 43), 'axis', False)
    # Getting the type of 'nm' (line 526)
    nm_18319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 49), 'nm', False)
    # Getting the type of 'overwrite_x' (line 526)
    overwrite_x_18320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 53), 'overwrite_x', False)
    # Processing the call keyword arguments (line 526)
    kwargs_18321 = {}
    # Getting the type of '_raw_dct' (line 526)
    _raw_dct_18313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 16), '_raw_dct', False)
    # Calling _raw_dct(args, kwargs) (line 526)
    _raw_dct_call_result_18322 = invoke(stypy.reporting.localization.Localization(__file__, 526, 16), _raw_dct_18313, *[imag_18315, type_18316, n_18317, axis_18318, nm_18319, overwrite_x_18320], **kwargs_18321)
    
    # Applying the binary operator '*' (line 525)
    result_mul_18323 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 68), '*', complex_18312, _raw_dct_call_result_18322)
    
    # Applying the binary operator '+' (line 525)
    result_add_18324 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 16), '+', _raw_dct_call_result_18311, result_mul_18323)
    
    # Assigning a type to the variable 'stypy_return_type' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'stypy_return_type', result_add_18324)
    # SSA branch for the else part of an if statement (line 524)
    module_type_store.open_ssa_branch('else')
    
    # Call to _raw_dct(...): (line 528)
    # Processing the call arguments (line 528)
    # Getting the type of 'x0' (line 528)
    x0_18326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 24), 'x0', False)
    # Getting the type of 'type' (line 528)
    type_18327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 28), 'type', False)
    # Getting the type of 'n' (line 528)
    n_18328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 34), 'n', False)
    # Getting the type of 'axis' (line 528)
    axis_18329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 37), 'axis', False)
    # Getting the type of 'nm' (line 528)
    nm_18330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 43), 'nm', False)
    # Getting the type of 'overwrite_x' (line 528)
    overwrite_x_18331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 47), 'overwrite_x', False)
    # Processing the call keyword arguments (line 528)
    kwargs_18332 = {}
    # Getting the type of '_raw_dct' (line 528)
    _raw_dct_18325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 15), '_raw_dct', False)
    # Calling _raw_dct(args, kwargs) (line 528)
    _raw_dct_call_result_18333 = invoke(stypy.reporting.localization.Localization(__file__, 528, 15), _raw_dct_18325, *[x0_18326, type_18327, n_18328, axis_18329, nm_18330, overwrite_x_18331], **kwargs_18332)
    
    # Assigning a type to the variable 'stypy_return_type' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'stypy_return_type', _raw_dct_call_result_18333)
    # SSA join for if statement (line 524)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_dct(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_dct' in the type store
    # Getting the type of 'stypy_return_type' (line 496)
    stypy_return_type_18334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18334)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_dct'
    return stypy_return_type_18334

# Assigning a type to the variable '_dct' (line 496)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 0), '_dct', _dct)

@norecursion
def dst(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_18335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 16), 'int')
    # Getting the type of 'None' (line 531)
    None_18336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 21), 'None')
    int_18337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 32), 'int')
    # Getting the type of 'None' (line 531)
    None_18338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 41), 'None')
    # Getting the type of 'False' (line 531)
    False_18339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 59), 'False')
    defaults = [int_18335, None_18336, int_18337, None_18338, False_18339]
    # Create a new context for function 'dst'
    module_type_store = module_type_store.open_function_context('dst', 531, 0, False)
    
    # Passed parameters checking function
    dst.stypy_localization = localization
    dst.stypy_type_of_self = None
    dst.stypy_type_store = module_type_store
    dst.stypy_function_name = 'dst'
    dst.stypy_param_names_list = ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x']
    dst.stypy_varargs_param_name = None
    dst.stypy_kwargs_param_name = None
    dst.stypy_call_defaults = defaults
    dst.stypy_call_varargs = varargs
    dst.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dst', ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dst', localization, ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dst(...)' code ##################

    str_18340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, (-1)), 'str', '\n    Return the Discrete Sine Transform of arbitrary type sequence x.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    type : {1, 2, 3}, optional\n        Type of the DST (see Notes). Default type is 2.\n    n : int, optional\n        Length of the transform.  If ``n < x.shape[axis]``, `x` is\n        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The\n        default results in ``n = x.shape[axis]``.\n    axis : int, optional\n        Axis along which the dst is computed; the default is over the\n        last axis (i.e., ``axis=-1``).\n    norm : {None, \'ortho\'}, optional\n        Normalization mode (see Notes). Default is None.\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    dst : ndarray of reals\n        The transformed input array.\n\n    See Also\n    --------\n    idst : Inverse DST\n\n    Notes\n    -----\n    For a single dimension array ``x``.\n\n    There are theoretically 8 types of the DST for different combinations of\n    even/odd boundary conditions and boundary off sets [1]_, only the first\n    3 types are implemented in scipy.\n\n    **Type I**\n\n    There are several definitions of the DST-I; we use the following\n    for ``norm=None``.  DST-I assumes the input is odd around n=-1 and n=N. ::\n\n                 N-1\n      y[k] = 2 * sum x[n]*sin(pi*(k+1)*(n+1)/(N+1))\n                 n=0\n\n    Only None is supported as normalization mode for DCT-I. Note also that the\n    DCT-I is only supported for input size > 1\n    The (unnormalized) DCT-I is its own inverse, up to a factor `2(N+1)`.\n\n    **Type II**\n\n    There are several definitions of the DST-II; we use the following\n    for ``norm=None``.  DST-II assumes the input is odd around n=-1/2 and\n    n=N-1/2; the output is odd around k=-1 and even around k=N-1 ::\n\n                N-1\n      y[k] = 2* sum x[n]*sin(pi*(k+1)*(n+0.5)/N), 0 <= k < N.\n                n=0\n\n    if ``norm=\'ortho\'``, ``y[k]`` is multiplied by a scaling factor `f` ::\n\n        f = sqrt(1/(4*N)) if k == 0\n        f = sqrt(1/(2*N)) otherwise.\n\n    **Type III**\n\n    There are several definitions of the DST-III, we use the following\n    (for ``norm=None``).  DST-III assumes the input is odd around n=-1\n    and even around n=N-1 ::\n\n                                 N-2\n      y[k] = x[N-1]*(-1)**k + 2* sum x[n]*sin(pi*(k+0.5)*(n+1)/N), 0 <= k < N.\n                                 n=0\n\n    The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up\n    to a factor `2N`.  The orthonormalized DST-III is exactly the inverse of\n    the orthonormalized DST-II.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1] Wikipedia, "Discrete sine transform",\n           http://en.wikipedia.org/wiki/Discrete_sine_transform\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'type' (line 620)
    type_18341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 7), 'type')
    int_18342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 15), 'int')
    # Applying the binary operator '==' (line 620)
    result_eq_18343 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 7), '==', type_18341, int_18342)
    
    
    # Getting the type of 'norm' (line 620)
    norm_18344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 21), 'norm')
    # Getting the type of 'None' (line 620)
    None_18345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 33), 'None')
    # Applying the binary operator 'isnot' (line 620)
    result_is_not_18346 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 21), 'isnot', norm_18344, None_18345)
    
    # Applying the binary operator 'and' (line 620)
    result_and_keyword_18347 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 7), 'and', result_eq_18343, result_is_not_18346)
    
    # Testing the type of an if condition (line 620)
    if_condition_18348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 4), result_and_keyword_18347)
    # Assigning a type to the variable 'if_condition_18348' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'if_condition_18348', if_condition_18348)
    # SSA begins for if statement (line 620)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NotImplementedError(...): (line 621)
    # Processing the call arguments (line 621)
    str_18350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 14), 'str', 'Orthonormalization not yet supported for IDCT-I')
    # Processing the call keyword arguments (line 621)
    kwargs_18351 = {}
    # Getting the type of 'NotImplementedError' (line 621)
    NotImplementedError_18349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 14), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 621)
    NotImplementedError_call_result_18352 = invoke(stypy.reporting.localization.Localization(__file__, 621, 14), NotImplementedError_18349, *[str_18350], **kwargs_18351)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 621, 8), NotImplementedError_call_result_18352, 'raise parameter', BaseException)
    # SSA join for if statement (line 620)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _dst(...): (line 623)
    # Processing the call arguments (line 623)
    # Getting the type of 'x' (line 623)
    x_18354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'x', False)
    # Getting the type of 'type' (line 623)
    type_18355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 19), 'type', False)
    # Getting the type of 'n' (line 623)
    n_18356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 25), 'n', False)
    # Getting the type of 'axis' (line 623)
    axis_18357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 28), 'axis', False)
    # Processing the call keyword arguments (line 623)
    # Getting the type of 'norm' (line 623)
    norm_18358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 44), 'norm', False)
    keyword_18359 = norm_18358
    # Getting the type of 'overwrite_x' (line 623)
    overwrite_x_18360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 62), 'overwrite_x', False)
    keyword_18361 = overwrite_x_18360
    kwargs_18362 = {'normalize': keyword_18359, 'overwrite_x': keyword_18361}
    # Getting the type of '_dst' (line 623)
    _dst_18353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 11), '_dst', False)
    # Calling _dst(args, kwargs) (line 623)
    _dst_call_result_18363 = invoke(stypy.reporting.localization.Localization(__file__, 623, 11), _dst_18353, *[x_18354, type_18355, n_18356, axis_18357], **kwargs_18362)
    
    # Assigning a type to the variable 'stypy_return_type' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'stypy_return_type', _dst_call_result_18363)
    
    # ################# End of 'dst(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dst' in the type store
    # Getting the type of 'stypy_return_type' (line 531)
    stypy_return_type_18364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18364)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dst'
    return stypy_return_type_18364

# Assigning a type to the variable 'dst' (line 531)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 0), 'dst', dst)

@norecursion
def idst(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_18365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 17), 'int')
    # Getting the type of 'None' (line 626)
    None_18366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 22), 'None')
    int_18367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 33), 'int')
    # Getting the type of 'None' (line 626)
    None_18368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 42), 'None')
    # Getting the type of 'False' (line 626)
    False_18369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 60), 'False')
    defaults = [int_18365, None_18366, int_18367, None_18368, False_18369]
    # Create a new context for function 'idst'
    module_type_store = module_type_store.open_function_context('idst', 626, 0, False)
    
    # Passed parameters checking function
    idst.stypy_localization = localization
    idst.stypy_type_of_self = None
    idst.stypy_type_store = module_type_store
    idst.stypy_function_name = 'idst'
    idst.stypy_param_names_list = ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x']
    idst.stypy_varargs_param_name = None
    idst.stypy_kwargs_param_name = None
    idst.stypy_call_defaults = defaults
    idst.stypy_call_varargs = varargs
    idst.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idst', ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idst', localization, ['x', 'type', 'n', 'axis', 'norm', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idst(...)' code ##################

    str_18370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, (-1)), 'str', "\n    Return the Inverse Discrete Sine Transform of an arbitrary type sequence.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    type : {1, 2, 3}, optional\n        Type of the DST (see Notes). Default type is 2.\n    n : int, optional\n        Length of the transform.  If ``n < x.shape[axis]``, `x` is\n        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The\n        default results in ``n = x.shape[axis]``.\n    axis : int, optional\n        Axis along which the idst is computed; the default is over the\n        last axis (i.e., ``axis=-1``).\n    norm : {None, 'ortho'}, optional\n        Normalization mode (see Notes). Default is None.\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    idst : ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    dst : Forward DST\n\n    Notes\n    -----\n    'The' IDST is the IDST of type 2, which is the same as DST of type 3.\n\n    IDST of type 1 is the DST of type 1, IDST of type 2 is the DST of type\n    3, and IDST of type 3 is the DST of type 2. For the definition of these\n    types, see `dst`.\n\n    .. versionadded:: 0.11.0\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'type' (line 668)
    type_18371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 7), 'type')
    int_18372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 15), 'int')
    # Applying the binary operator '==' (line 668)
    result_eq_18373 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 7), '==', type_18371, int_18372)
    
    
    # Getting the type of 'norm' (line 668)
    norm_18374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 21), 'norm')
    # Getting the type of 'None' (line 668)
    None_18375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 33), 'None')
    # Applying the binary operator 'isnot' (line 668)
    result_is_not_18376 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 21), 'isnot', norm_18374, None_18375)
    
    # Applying the binary operator 'and' (line 668)
    result_and_keyword_18377 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 7), 'and', result_eq_18373, result_is_not_18376)
    
    # Testing the type of an if condition (line 668)
    if_condition_18378 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 668, 4), result_and_keyword_18377)
    # Assigning a type to the variable 'if_condition_18378' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 4), 'if_condition_18378', if_condition_18378)
    # SSA begins for if statement (line 668)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NotImplementedError(...): (line 669)
    # Processing the call arguments (line 669)
    str_18380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 14), 'str', 'Orthonormalization not yet supported for IDCT-I')
    # Processing the call keyword arguments (line 669)
    kwargs_18381 = {}
    # Getting the type of 'NotImplementedError' (line 669)
    NotImplementedError_18379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 14), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 669)
    NotImplementedError_call_result_18382 = invoke(stypy.reporting.localization.Localization(__file__, 669, 14), NotImplementedError_18379, *[str_18380], **kwargs_18381)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 669, 8), NotImplementedError_call_result_18382, 'raise parameter', BaseException)
    # SSA join for if statement (line 668)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 672):
    
    # Assigning a Dict to a Name (line 672):
    
    # Obtaining an instance of the builtin type 'dict' (line 672)
    dict_18383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 672)
    # Adding element type (key, value) (line 672)
    int_18384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 11), 'int')
    int_18385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 13), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 10), dict_18383, (int_18384, int_18385))
    # Adding element type (key, value) (line 672)
    int_18386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 16), 'int')
    int_18387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 18), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 10), dict_18383, (int_18386, int_18387))
    # Adding element type (key, value) (line 672)
    int_18388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 21), 'int')
    int_18389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 23), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 10), dict_18383, (int_18388, int_18389))
    
    # Assigning a type to the variable '_TP' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), '_TP', dict_18383)
    
    # Call to _dst(...): (line 673)
    # Processing the call arguments (line 673)
    # Getting the type of 'x' (line 673)
    x_18391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 16), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'type' (line 673)
    type_18392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 23), 'type', False)
    # Getting the type of '_TP' (line 673)
    _TP_18393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 19), '_TP', False)
    # Obtaining the member '__getitem__' of a type (line 673)
    getitem___18394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 19), _TP_18393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 673)
    subscript_call_result_18395 = invoke(stypy.reporting.localization.Localization(__file__, 673, 19), getitem___18394, type_18392)
    
    # Getting the type of 'n' (line 673)
    n_18396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 30), 'n', False)
    # Getting the type of 'axis' (line 673)
    axis_18397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 33), 'axis', False)
    # Processing the call keyword arguments (line 673)
    # Getting the type of 'norm' (line 673)
    norm_18398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 49), 'norm', False)
    keyword_18399 = norm_18398
    # Getting the type of 'overwrite_x' (line 673)
    overwrite_x_18400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 67), 'overwrite_x', False)
    keyword_18401 = overwrite_x_18400
    kwargs_18402 = {'normalize': keyword_18399, 'overwrite_x': keyword_18401}
    # Getting the type of '_dst' (line 673)
    _dst_18390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 11), '_dst', False)
    # Calling _dst(args, kwargs) (line 673)
    _dst_call_result_18403 = invoke(stypy.reporting.localization.Localization(__file__, 673, 11), _dst_18390, *[x_18391, subscript_call_result_18395, n_18396, axis_18397], **kwargs_18402)
    
    # Assigning a type to the variable 'stypy_return_type' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'stypy_return_type', _dst_call_result_18403)
    
    # ################# End of 'idst(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idst' in the type store
    # Getting the type of 'stypy_return_type' (line 626)
    stypy_return_type_18404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18404)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idst'
    return stypy_return_type_18404

# Assigning a type to the variable 'idst' (line 626)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 0), 'idst', idst)

@norecursion
def _get_dst_fun(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_dst_fun'
    module_type_store = module_type_store.open_function_context('_get_dst_fun', 676, 0, False)
    
    # Passed parameters checking function
    _get_dst_fun.stypy_localization = localization
    _get_dst_fun.stypy_type_of_self = None
    _get_dst_fun.stypy_type_store = module_type_store
    _get_dst_fun.stypy_function_name = '_get_dst_fun'
    _get_dst_fun.stypy_param_names_list = ['type', 'dtype']
    _get_dst_fun.stypy_varargs_param_name = None
    _get_dst_fun.stypy_kwargs_param_name = None
    _get_dst_fun.stypy_call_defaults = defaults
    _get_dst_fun.stypy_call_varargs = varargs
    _get_dst_fun.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_dst_fun', ['type', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_dst_fun', localization, ['type', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_dst_fun(...)' code ##################

    
    
    # SSA begins for try-except statement (line 677)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 678):
    
    # Assigning a Subscript to a Name (line 678):
    
    # Obtaining the type of the subscript
    # Getting the type of 'dtype' (line 678)
    dtype_18405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 55), 'dtype')
    # Obtaining the member 'name' of a type (line 678)
    name_18406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 55), dtype_18405, 'name')
    
    # Obtaining an instance of the builtin type 'dict' (line 678)
    dict_18407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 678)
    # Adding element type (key, value) (line 678)
    str_18408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 16), 'str', 'float64')
    str_18409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 26), 'str', 'ddst%d')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 678, 15), dict_18407, (str_18408, str_18409))
    # Adding element type (key, value) (line 678)
    str_18410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 36), 'str', 'float32')
    str_18411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 46), 'str', 'dst%d')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 678, 15), dict_18407, (str_18410, str_18411))
    
    # Obtaining the member '__getitem__' of a type (line 678)
    getitem___18412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 15), dict_18407, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 678)
    subscript_call_result_18413 = invoke(stypy.reporting.localization.Localization(__file__, 678, 15), getitem___18412, name_18406)
    
    # Assigning a type to the variable 'name' (line 678)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 8), 'name', subscript_call_result_18413)
    # SSA branch for the except part of a try statement (line 677)
    # SSA branch for the except 'KeyError' branch of a try statement (line 677)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 680)
    # Processing the call arguments (line 680)
    str_18415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 25), 'str', 'dtype %s not supported')
    # Getting the type of 'dtype' (line 680)
    dtype_18416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 52), 'dtype', False)
    # Applying the binary operator '%' (line 680)
    result_mod_18417 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 25), '%', str_18415, dtype_18416)
    
    # Processing the call keyword arguments (line 680)
    kwargs_18418 = {}
    # Getting the type of 'ValueError' (line 680)
    ValueError_18414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 680)
    ValueError_call_result_18419 = invoke(stypy.reporting.localization.Localization(__file__, 680, 14), ValueError_18414, *[result_mod_18417], **kwargs_18418)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 680, 8), ValueError_call_result_18419, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 677)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 681)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 682):
    
    # Assigning a Call to a Name (line 682):
    
    # Call to getattr(...): (line 682)
    # Processing the call arguments (line 682)
    # Getting the type of '_fftpack' (line 682)
    _fftpack_18421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 20), '_fftpack', False)
    # Getting the type of 'name' (line 682)
    name_18422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 30), 'name', False)
    # Getting the type of 'type' (line 682)
    type_18423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 37), 'type', False)
    # Applying the binary operator '%' (line 682)
    result_mod_18424 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 30), '%', name_18422, type_18423)
    
    # Processing the call keyword arguments (line 682)
    kwargs_18425 = {}
    # Getting the type of 'getattr' (line 682)
    getattr_18420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 12), 'getattr', False)
    # Calling getattr(args, kwargs) (line 682)
    getattr_call_result_18426 = invoke(stypy.reporting.localization.Localization(__file__, 682, 12), getattr_18420, *[_fftpack_18421, result_mod_18424], **kwargs_18425)
    
    # Assigning a type to the variable 'f' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'f', getattr_call_result_18426)
    # SSA branch for the except part of a try statement (line 681)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 681)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'AttributeError' (line 683)
    AttributeError_18427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 11), 'AttributeError')
    # Assigning a type to the variable 'e' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'e', AttributeError_18427)
    
    # Call to ValueError(...): (line 684)
    # Processing the call arguments (line 684)
    
    # Call to str(...): (line 684)
    # Processing the call arguments (line 684)
    # Getting the type of 'e' (line 684)
    e_18430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 29), 'e', False)
    # Processing the call keyword arguments (line 684)
    kwargs_18431 = {}
    # Getting the type of 'str' (line 684)
    str_18429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 25), 'str', False)
    # Calling str(args, kwargs) (line 684)
    str_call_result_18432 = invoke(stypy.reporting.localization.Localization(__file__, 684, 25), str_18429, *[e_18430], **kwargs_18431)
    
    str_18433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 34), 'str', '. Type %d not understood')
    # Getting the type of 'type' (line 684)
    type_18434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 63), 'type', False)
    # Applying the binary operator '%' (line 684)
    result_mod_18435 = python_operator(stypy.reporting.localization.Localization(__file__, 684, 34), '%', str_18433, type_18434)
    
    # Applying the binary operator '+' (line 684)
    result_add_18436 = python_operator(stypy.reporting.localization.Localization(__file__, 684, 25), '+', str_call_result_18432, result_mod_18435)
    
    # Processing the call keyword arguments (line 684)
    kwargs_18437 = {}
    # Getting the type of 'ValueError' (line 684)
    ValueError_18428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 684)
    ValueError_call_result_18438 = invoke(stypy.reporting.localization.Localization(__file__, 684, 14), ValueError_18428, *[result_add_18436], **kwargs_18437)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 684, 8), ValueError_call_result_18438, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 681)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'f' (line 685)
    f_18439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 11), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'stypy_return_type', f_18439)
    
    # ################# End of '_get_dst_fun(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_dst_fun' in the type store
    # Getting the type of 'stypy_return_type' (line 676)
    stypy_return_type_18440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18440)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_dst_fun'
    return stypy_return_type_18440

# Assigning a type to the variable '_get_dst_fun' (line 676)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 0), '_get_dst_fun', _get_dst_fun)

@norecursion
def _dst(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 688)
    None_18441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 20), 'None')
    int_18442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 31), 'int')
    # Getting the type of 'False' (line 688)
    False_18443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 47), 'False')
    # Getting the type of 'None' (line 688)
    None_18444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 64), 'None')
    defaults = [None_18441, int_18442, False_18443, None_18444]
    # Create a new context for function '_dst'
    module_type_store = module_type_store.open_function_context('_dst', 688, 0, False)
    
    # Passed parameters checking function
    _dst.stypy_localization = localization
    _dst.stypy_type_of_self = None
    _dst.stypy_type_store = module_type_store
    _dst.stypy_function_name = '_dst'
    _dst.stypy_param_names_list = ['x', 'type', 'n', 'axis', 'overwrite_x', 'normalize']
    _dst.stypy_varargs_param_name = None
    _dst.stypy_kwargs_param_name = None
    _dst.stypy_call_defaults = defaults
    _dst.stypy_call_varargs = varargs
    _dst.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_dst', ['x', 'type', 'n', 'axis', 'overwrite_x', 'normalize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_dst', localization, ['x', 'type', 'n', 'axis', 'overwrite_x', 'normalize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_dst(...)' code ##################

    str_18445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, (-1)), 'str', '\n    Return Discrete Sine Transform of arbitrary type sequence x.\n\n    Parameters\n    ----------\n    x : array_like\n        input array.\n    n : int, optional\n        Length of the transform.\n    axis : int, optional\n        Axis along which the dst is computed. (default=-1)\n    overwrite_x : bool, optional\n        If True the contents of x can be destroyed. (default=False)\n\n    Returns\n    -------\n    z : real ndarray\n\n    ')
    
    # Assigning a Call to a Tuple (line 708):
    
    # Assigning a Subscript to a Name (line 708):
    
    # Obtaining the type of the subscript
    int_18446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 4), 'int')
    
    # Call to __fix_shape(...): (line 708)
    # Processing the call arguments (line 708)
    # Getting the type of 'x' (line 708)
    x_18448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 35), 'x', False)
    # Getting the type of 'n' (line 708)
    n_18449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 38), 'n', False)
    # Getting the type of 'axis' (line 708)
    axis_18450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 41), 'axis', False)
    str_18451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 47), 'str', 'DST')
    # Processing the call keyword arguments (line 708)
    kwargs_18452 = {}
    # Getting the type of '__fix_shape' (line 708)
    fix_shape_18447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 23), '__fix_shape', False)
    # Calling __fix_shape(args, kwargs) (line 708)
    fix_shape_call_result_18453 = invoke(stypy.reporting.localization.Localization(__file__, 708, 23), fix_shape_18447, *[x_18448, n_18449, axis_18450, str_18451], **kwargs_18452)
    
    # Obtaining the member '__getitem__' of a type (line 708)
    getitem___18454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 4), fix_shape_call_result_18453, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 708)
    subscript_call_result_18455 = invoke(stypy.reporting.localization.Localization(__file__, 708, 4), getitem___18454, int_18446)
    
    # Assigning a type to the variable 'tuple_var_assignment_17604' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'tuple_var_assignment_17604', subscript_call_result_18455)
    
    # Assigning a Subscript to a Name (line 708):
    
    # Obtaining the type of the subscript
    int_18456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 4), 'int')
    
    # Call to __fix_shape(...): (line 708)
    # Processing the call arguments (line 708)
    # Getting the type of 'x' (line 708)
    x_18458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 35), 'x', False)
    # Getting the type of 'n' (line 708)
    n_18459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 38), 'n', False)
    # Getting the type of 'axis' (line 708)
    axis_18460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 41), 'axis', False)
    str_18461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 47), 'str', 'DST')
    # Processing the call keyword arguments (line 708)
    kwargs_18462 = {}
    # Getting the type of '__fix_shape' (line 708)
    fix_shape_18457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 23), '__fix_shape', False)
    # Calling __fix_shape(args, kwargs) (line 708)
    fix_shape_call_result_18463 = invoke(stypy.reporting.localization.Localization(__file__, 708, 23), fix_shape_18457, *[x_18458, n_18459, axis_18460, str_18461], **kwargs_18462)
    
    # Obtaining the member '__getitem__' of a type (line 708)
    getitem___18464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 4), fix_shape_call_result_18463, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 708)
    subscript_call_result_18465 = invoke(stypy.reporting.localization.Localization(__file__, 708, 4), getitem___18464, int_18456)
    
    # Assigning a type to the variable 'tuple_var_assignment_17605' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'tuple_var_assignment_17605', subscript_call_result_18465)
    
    # Assigning a Subscript to a Name (line 708):
    
    # Obtaining the type of the subscript
    int_18466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 4), 'int')
    
    # Call to __fix_shape(...): (line 708)
    # Processing the call arguments (line 708)
    # Getting the type of 'x' (line 708)
    x_18468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 35), 'x', False)
    # Getting the type of 'n' (line 708)
    n_18469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 38), 'n', False)
    # Getting the type of 'axis' (line 708)
    axis_18470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 41), 'axis', False)
    str_18471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 47), 'str', 'DST')
    # Processing the call keyword arguments (line 708)
    kwargs_18472 = {}
    # Getting the type of '__fix_shape' (line 708)
    fix_shape_18467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 23), '__fix_shape', False)
    # Calling __fix_shape(args, kwargs) (line 708)
    fix_shape_call_result_18473 = invoke(stypy.reporting.localization.Localization(__file__, 708, 23), fix_shape_18467, *[x_18468, n_18469, axis_18470, str_18471], **kwargs_18472)
    
    # Obtaining the member '__getitem__' of a type (line 708)
    getitem___18474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 4), fix_shape_call_result_18473, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 708)
    subscript_call_result_18475 = invoke(stypy.reporting.localization.Localization(__file__, 708, 4), getitem___18474, int_18466)
    
    # Assigning a type to the variable 'tuple_var_assignment_17606' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'tuple_var_assignment_17606', subscript_call_result_18475)
    
    # Assigning a Name to a Name (line 708):
    # Getting the type of 'tuple_var_assignment_17604' (line 708)
    tuple_var_assignment_17604_18476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'tuple_var_assignment_17604')
    # Assigning a type to the variable 'x0' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'x0', tuple_var_assignment_17604_18476)
    
    # Assigning a Name to a Name (line 708):
    # Getting the type of 'tuple_var_assignment_17605' (line 708)
    tuple_var_assignment_17605_18477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'tuple_var_assignment_17605')
    # Assigning a type to the variable 'n' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'n', tuple_var_assignment_17605_18477)
    
    # Assigning a Name to a Name (line 708):
    # Getting the type of 'tuple_var_assignment_17606' (line 708)
    tuple_var_assignment_17606_18478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'tuple_var_assignment_17606')
    # Assigning a type to the variable 'copy_made' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 11), 'copy_made', tuple_var_assignment_17606_18478)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'type' (line 709)
    type_18479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 7), 'type')
    int_18480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 15), 'int')
    # Applying the binary operator '==' (line 709)
    result_eq_18481 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 7), '==', type_18479, int_18480)
    
    
    # Getting the type of 'n' (line 709)
    n_18482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 21), 'n')
    int_18483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 25), 'int')
    # Applying the binary operator '<' (line 709)
    result_lt_18484 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 21), '<', n_18482, int_18483)
    
    # Applying the binary operator 'and' (line 709)
    result_and_keyword_18485 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 7), 'and', result_eq_18481, result_lt_18484)
    
    # Testing the type of an if condition (line 709)
    if_condition_18486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 709, 4), result_and_keyword_18485)
    # Assigning a type to the variable 'if_condition_18486' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'if_condition_18486', if_condition_18486)
    # SSA begins for if statement (line 709)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 710)
    # Processing the call arguments (line 710)
    str_18488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 25), 'str', 'DST-I is not defined for size < 2')
    # Processing the call keyword arguments (line 710)
    kwargs_18489 = {}
    # Getting the type of 'ValueError' (line 710)
    ValueError_18487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 710)
    ValueError_call_result_18490 = invoke(stypy.reporting.localization.Localization(__file__, 710, 14), ValueError_18487, *[str_18488], **kwargs_18489)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 710, 8), ValueError_call_result_18490, 'raise parameter', BaseException)
    # SSA join for if statement (line 709)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 711):
    
    # Assigning a BoolOp to a Name (line 711):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_x' (line 711)
    overwrite_x_18491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 18), 'overwrite_x')
    # Getting the type of 'copy_made' (line 711)
    copy_made_18492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 33), 'copy_made')
    # Applying the binary operator 'or' (line 711)
    result_or_keyword_18493 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 18), 'or', overwrite_x_18491, copy_made_18492)
    
    # Assigning a type to the variable 'overwrite_x' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'overwrite_x', result_or_keyword_18493)
    
    # Assigning a Call to a Name (line 712):
    
    # Assigning a Call to a Name (line 712):
    
    # Call to _get_norm_mode(...): (line 712)
    # Processing the call arguments (line 712)
    # Getting the type of 'normalize' (line 712)
    normalize_18495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 24), 'normalize', False)
    # Processing the call keyword arguments (line 712)
    kwargs_18496 = {}
    # Getting the type of '_get_norm_mode' (line 712)
    _get_norm_mode_18494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 9), '_get_norm_mode', False)
    # Calling _get_norm_mode(args, kwargs) (line 712)
    _get_norm_mode_call_result_18497 = invoke(stypy.reporting.localization.Localization(__file__, 712, 9), _get_norm_mode_18494, *[normalize_18495], **kwargs_18496)
    
    # Assigning a type to the variable 'nm' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 4), 'nm', _get_norm_mode_call_result_18497)
    
    
    # Call to iscomplexobj(...): (line 713)
    # Processing the call arguments (line 713)
    # Getting the type of 'x0' (line 713)
    x0_18500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 23), 'x0', False)
    # Processing the call keyword arguments (line 713)
    kwargs_18501 = {}
    # Getting the type of 'np' (line 713)
    np_18498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 7), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 713)
    iscomplexobj_18499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 7), np_18498, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 713)
    iscomplexobj_call_result_18502 = invoke(stypy.reporting.localization.Localization(__file__, 713, 7), iscomplexobj_18499, *[x0_18500], **kwargs_18501)
    
    # Testing the type of an if condition (line 713)
    if_condition_18503 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 713, 4), iscomplexobj_call_result_18502)
    # Assigning a type to the variable 'if_condition_18503' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 4), 'if_condition_18503', if_condition_18503)
    # SSA begins for if statement (line 713)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _raw_dst(...): (line 714)
    # Processing the call arguments (line 714)
    # Getting the type of 'x0' (line 714)
    x0_18505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 25), 'x0', False)
    # Obtaining the member 'real' of a type (line 714)
    real_18506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 25), x0_18505, 'real')
    # Getting the type of 'type' (line 714)
    type_18507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 34), 'type', False)
    # Getting the type of 'n' (line 714)
    n_18508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 40), 'n', False)
    # Getting the type of 'axis' (line 714)
    axis_18509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 43), 'axis', False)
    # Getting the type of 'nm' (line 714)
    nm_18510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 49), 'nm', False)
    # Getting the type of 'overwrite_x' (line 714)
    overwrite_x_18511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 53), 'overwrite_x', False)
    # Processing the call keyword arguments (line 714)
    kwargs_18512 = {}
    # Getting the type of '_raw_dst' (line 714)
    _raw_dst_18504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 16), '_raw_dst', False)
    # Calling _raw_dst(args, kwargs) (line 714)
    _raw_dst_call_result_18513 = invoke(stypy.reporting.localization.Localization(__file__, 714, 16), _raw_dst_18504, *[real_18506, type_18507, n_18508, axis_18509, nm_18510, overwrite_x_18511], **kwargs_18512)
    
    complex_18514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 68), 'complex')
    
    # Call to _raw_dst(...): (line 715)
    # Processing the call arguments (line 715)
    # Getting the type of 'x0' (line 715)
    x0_18516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 25), 'x0', False)
    # Obtaining the member 'imag' of a type (line 715)
    imag_18517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 25), x0_18516, 'imag')
    # Getting the type of 'type' (line 715)
    type_18518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 34), 'type', False)
    # Getting the type of 'n' (line 715)
    n_18519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 40), 'n', False)
    # Getting the type of 'axis' (line 715)
    axis_18520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 43), 'axis', False)
    # Getting the type of 'nm' (line 715)
    nm_18521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 49), 'nm', False)
    # Getting the type of 'overwrite_x' (line 715)
    overwrite_x_18522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 53), 'overwrite_x', False)
    # Processing the call keyword arguments (line 715)
    kwargs_18523 = {}
    # Getting the type of '_raw_dst' (line 715)
    _raw_dst_18515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 16), '_raw_dst', False)
    # Calling _raw_dst(args, kwargs) (line 715)
    _raw_dst_call_result_18524 = invoke(stypy.reporting.localization.Localization(__file__, 715, 16), _raw_dst_18515, *[imag_18517, type_18518, n_18519, axis_18520, nm_18521, overwrite_x_18522], **kwargs_18523)
    
    # Applying the binary operator '*' (line 714)
    result_mul_18525 = python_operator(stypy.reporting.localization.Localization(__file__, 714, 68), '*', complex_18514, _raw_dst_call_result_18524)
    
    # Applying the binary operator '+' (line 714)
    result_add_18526 = python_operator(stypy.reporting.localization.Localization(__file__, 714, 16), '+', _raw_dst_call_result_18513, result_mul_18525)
    
    # Assigning a type to the variable 'stypy_return_type' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'stypy_return_type', result_add_18526)
    # SSA branch for the else part of an if statement (line 713)
    module_type_store.open_ssa_branch('else')
    
    # Call to _raw_dst(...): (line 717)
    # Processing the call arguments (line 717)
    # Getting the type of 'x0' (line 717)
    x0_18528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 24), 'x0', False)
    # Getting the type of 'type' (line 717)
    type_18529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 28), 'type', False)
    # Getting the type of 'n' (line 717)
    n_18530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 34), 'n', False)
    # Getting the type of 'axis' (line 717)
    axis_18531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 37), 'axis', False)
    # Getting the type of 'nm' (line 717)
    nm_18532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 43), 'nm', False)
    # Getting the type of 'overwrite_x' (line 717)
    overwrite_x_18533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 47), 'overwrite_x', False)
    # Processing the call keyword arguments (line 717)
    kwargs_18534 = {}
    # Getting the type of '_raw_dst' (line 717)
    _raw_dst_18527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 15), '_raw_dst', False)
    # Calling _raw_dst(args, kwargs) (line 717)
    _raw_dst_call_result_18535 = invoke(stypy.reporting.localization.Localization(__file__, 717, 15), _raw_dst_18527, *[x0_18528, type_18529, n_18530, axis_18531, nm_18532, overwrite_x_18533], **kwargs_18534)
    
    # Assigning a type to the variable 'stypy_return_type' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'stypy_return_type', _raw_dst_call_result_18535)
    # SSA join for if statement (line 713)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_dst(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_dst' in the type store
    # Getting the type of 'stypy_return_type' (line 688)
    stypy_return_type_18536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18536)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_dst'
    return stypy_return_type_18536

# Assigning a type to the variable '_dst' (line 688)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 0), '_dst', _dst)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
