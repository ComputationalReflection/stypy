
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from os.path import join, dirname
4: 
5: import numpy as np
6: from numpy.testing import assert_array_almost_equal, assert_equal
7: from pytest import raises as assert_raises
8: 
9: from scipy.fftpack.realtransforms import (
10:     dct, idct, dst, idst, dctn, idctn, dstn, idstn)
11: 
12: # Matlab reference data
13: MDATA = np.load(join(dirname(__file__), 'test.npz'))
14: X = [MDATA['x%d' % i] for i in range(8)]
15: Y = [MDATA['y%d' % i] for i in range(8)]
16: 
17: # FFTW reference data: the data are organized as follows:
18: #    * SIZES is an array containing all available sizes
19: #    * for every type (1, 2, 3, 4) and every size, the array dct_type_size
20: #    contains the output of the DCT applied to the input np.linspace(0, size-1,
21: #    size)
22: FFTWDATA_DOUBLE = np.load(join(dirname(__file__), 'fftw_double_ref.npz'))
23: FFTWDATA_SINGLE = np.load(join(dirname(__file__), 'fftw_single_ref.npz'))
24: FFTWDATA_SIZES = FFTWDATA_DOUBLE['sizes']
25: 
26: 
27: def fftw_dct_ref(type, size, dt):
28:     x = np.linspace(0, size-1, size).astype(dt)
29:     dt = np.result_type(np.float32, dt)
30:     if dt == np.double:
31:         data = FFTWDATA_DOUBLE
32:     elif dt == np.float32:
33:         data = FFTWDATA_SINGLE
34:     else:
35:         raise ValueError()
36:     y = (data['dct_%d_%d' % (type, size)]).astype(dt)
37:     return x, y, dt
38: 
39: 
40: def fftw_dst_ref(type, size, dt):
41:     x = np.linspace(0, size-1, size).astype(dt)
42:     dt = np.result_type(np.float32, dt)
43:     if dt == np.double:
44:         data = FFTWDATA_DOUBLE
45:     elif dt == np.float32:
46:         data = FFTWDATA_SINGLE
47:     else:
48:         raise ValueError()
49:     y = (data['dst_%d_%d' % (type, size)]).astype(dt)
50:     return x, y, dt
51: 
52: 
53: def dct_2d_ref(x, **kwargs):
54:     ''' used as a reference in testing dct2. '''
55:     x = np.array(x, copy=True)
56:     for row in range(x.shape[0]):
57:         x[row, :] = dct(x[row, :], **kwargs)
58:     for col in range(x.shape[1]):
59:         x[:, col] = dct(x[:, col], **kwargs)
60:     return x
61: 
62: 
63: def idct_2d_ref(x, **kwargs):
64:     ''' used as a reference in testing idct2. '''
65:     x = np.array(x, copy=True)
66:     for row in range(x.shape[0]):
67:         x[row, :] = idct(x[row, :], **kwargs)
68:     for col in range(x.shape[1]):
69:         x[:, col] = idct(x[:, col], **kwargs)
70:     return x
71: 
72: 
73: def dst_2d_ref(x, **kwargs):
74:     ''' used as a reference in testing dst2. '''
75:     x = np.array(x, copy=True)
76:     for row in range(x.shape[0]):
77:         x[row, :] = dst(x[row, :], **kwargs)
78:     for col in range(x.shape[1]):
79:         x[:, col] = dst(x[:, col], **kwargs)
80:     return x
81: 
82: 
83: def idst_2d_ref(x, **kwargs):
84:     ''' used as a reference in testing idst2. '''
85:     x = np.array(x, copy=True)
86:     for row in range(x.shape[0]):
87:         x[row, :] = idst(x[row, :], **kwargs)
88:     for col in range(x.shape[1]):
89:         x[:, col] = idst(x[:, col], **kwargs)
90:     return x
91: 
92: 
93: class TestComplex(object):
94:     def test_dct_complex64(self):
95:         y = dct(1j*np.arange(5, dtype=np.complex64))
96:         x = 1j*dct(np.arange(5))
97:         assert_array_almost_equal(x, y)
98: 
99:     def test_dct_complex(self):
100:         y = dct(np.arange(5)*1j)
101:         x = 1j*dct(np.arange(5))
102:         assert_array_almost_equal(x, y)
103: 
104:     def test_idct_complex(self):
105:         y = idct(np.arange(5)*1j)
106:         x = 1j*idct(np.arange(5))
107:         assert_array_almost_equal(x, y)
108: 
109:     def test_dst_complex64(self):
110:         y = dst(np.arange(5, dtype=np.complex64)*1j)
111:         x = 1j*dst(np.arange(5))
112:         assert_array_almost_equal(x, y)
113: 
114:     def test_dst_complex(self):
115:         y = dst(np.arange(5)*1j)
116:         x = 1j*dst(np.arange(5))
117:         assert_array_almost_equal(x, y)
118: 
119:     def test_idst_complex(self):
120:         y = idst(np.arange(5)*1j)
121:         x = 1j*idst(np.arange(5))
122:         assert_array_almost_equal(x, y)
123: 
124: 
125: class _TestDCTBase(object):
126:     def setup_method(self):
127:         self.rdt = None
128:         self.dec = 14
129:         self.type = None
130: 
131:     def test_definition(self):
132:         for i in FFTWDATA_SIZES:
133:             x, yr, dt = fftw_dct_ref(self.type, i, self.rdt)
134:             y = dct(x, type=self.type)
135:             assert_equal(y.dtype, dt)
136:             # XXX: we divide by np.max(y) because the tests fail otherwise. We
137:             # should really use something like assert_array_approx_equal. The
138:             # difference is due to fftw using a better algorithm w.r.t error
139:             # propagation compared to the ones from fftpack.
140:             assert_array_almost_equal(y / np.max(y), yr / np.max(y), decimal=self.dec,
141:                     err_msg="Size %d failed" % i)
142: 
143:     def test_axis(self):
144:         nt = 2
145:         for i in [7, 8, 9, 16, 32, 64]:
146:             x = np.random.randn(nt, i)
147:             y = dct(x, type=self.type)
148:             for j in range(nt):
149:                 assert_array_almost_equal(y[j], dct(x[j], type=self.type),
150:                         decimal=self.dec)
151: 
152:             x = x.T
153:             y = dct(x, axis=0, type=self.type)
154:             for j in range(nt):
155:                 assert_array_almost_equal(y[:,j], dct(x[:,j], type=self.type),
156:                         decimal=self.dec)
157: 
158: 
159: class _TestDCTIIBase(_TestDCTBase):
160:     def test_definition_matlab(self):
161:         # Test correspondance with matlab (orthornomal mode).
162:         for i in range(len(X)):
163:             dt = np.result_type(np.float32, self.rdt)
164:             x = np.array(X[i], dtype=dt)
165: 
166:             yr = Y[i]
167:             y = dct(x, norm="ortho", type=2)
168:             assert_equal(y.dtype, dt)
169:             assert_array_almost_equal(y, yr, decimal=self.dec)
170: 
171: 
172: class _TestDCTIIIBase(_TestDCTBase):
173:     def test_definition_ortho(self):
174:         # Test orthornomal mode.
175:         for i in range(len(X)):
176:             x = np.array(X[i], dtype=self.rdt)
177:             dt = np.result_type(np.float32, self.rdt)
178:             y = dct(x, norm='ortho', type=2)
179:             xi = dct(y, norm="ortho", type=3)
180:             assert_equal(xi.dtype, dt)
181:             assert_array_almost_equal(xi, x, decimal=self.dec)
182: 
183: 
184: class TestDCTIDouble(_TestDCTBase):
185:     def setup_method(self):
186:         self.rdt = np.double
187:         self.dec = 10
188:         self.type = 1
189: 
190: 
191: class TestDCTIFloat(_TestDCTBase):
192:     def setup_method(self):
193:         self.rdt = np.float32
194:         self.dec = 5
195:         self.type = 1
196: 
197: 
198: class TestDCTIInt(_TestDCTBase):
199:     def setup_method(self):
200:         self.rdt = int
201:         self.dec = 5
202:         self.type = 1
203: 
204: 
205: class TestDCTIIDouble(_TestDCTIIBase):
206:     def setup_method(self):
207:         self.rdt = np.double
208:         self.dec = 10
209:         self.type = 2
210: 
211: 
212: class TestDCTIIFloat(_TestDCTIIBase):
213:     def setup_method(self):
214:         self.rdt = np.float32
215:         self.dec = 5
216:         self.type = 2
217: 
218: 
219: class TestDCTIIInt(_TestDCTIIBase):
220:     def setup_method(self):
221:         self.rdt = int
222:         self.dec = 5
223:         self.type = 2
224: 
225: 
226: class TestDCTIIIDouble(_TestDCTIIIBase):
227:     def setup_method(self):
228:         self.rdt = np.double
229:         self.dec = 14
230:         self.type = 3
231: 
232: 
233: class TestDCTIIIFloat(_TestDCTIIIBase):
234:     def setup_method(self):
235:         self.rdt = np.float32
236:         self.dec = 5
237:         self.type = 3
238: 
239: 
240: class TestDCTIIIInt(_TestDCTIIIBase):
241:     def setup_method(self):
242:         self.rdt = int
243:         self.dec = 5
244:         self.type = 3
245: 
246: 
247: class _TestIDCTBase(object):
248:     def setup_method(self):
249:         self.rdt = None
250:         self.dec = 14
251:         self.type = None
252: 
253:     def test_definition(self):
254:         for i in FFTWDATA_SIZES:
255:             xr, yr, dt = fftw_dct_ref(self.type, i, self.rdt)
256:             x = idct(yr, type=self.type)
257:             if self.type == 1:
258:                 x /= 2 * (i-1)
259:             else:
260:                 x /= 2 * i
261:             assert_equal(x.dtype, dt)
262:             # XXX: we divide by np.max(y) because the tests fail otherwise. We
263:             # should really use something like assert_array_approx_equal. The
264:             # difference is due to fftw using a better algorithm w.r.t error
265:             # propagation compared to the ones from fftpack.
266:             assert_array_almost_equal(x / np.max(x), xr / np.max(x), decimal=self.dec,
267:                     err_msg="Size %d failed" % i)
268: 
269: 
270: class TestIDCTIDouble(_TestIDCTBase):
271:     def setup_method(self):
272:         self.rdt = np.double
273:         self.dec = 10
274:         self.type = 1
275: 
276: 
277: class TestIDCTIFloat(_TestIDCTBase):
278:     def setup_method(self):
279:         self.rdt = np.float32
280:         self.dec = 4
281:         self.type = 1
282: 
283: 
284: class TestIDCTIInt(_TestIDCTBase):
285:     def setup_method(self):
286:         self.rdt = int
287:         self.dec = 4
288:         self.type = 1
289: 
290: 
291: class TestIDCTIIDouble(_TestIDCTBase):
292:     def setup_method(self):
293:         self.rdt = np.double
294:         self.dec = 10
295:         self.type = 2
296: 
297: 
298: class TestIDCTIIFloat(_TestIDCTBase):
299:     def setup_method(self):
300:         self.rdt = np.float32
301:         self.dec = 5
302:         self.type = 2
303: 
304: 
305: class TestIDCTIIInt(_TestIDCTBase):
306:     def setup_method(self):
307:         self.rdt = int
308:         self.dec = 5
309:         self.type = 2
310: 
311: 
312: class TestIDCTIIIDouble(_TestIDCTBase):
313:     def setup_method(self):
314:         self.rdt = np.double
315:         self.dec = 14
316:         self.type = 3
317: 
318: 
319: class TestIDCTIIIFloat(_TestIDCTBase):
320:     def setup_method(self):
321:         self.rdt = np.float32
322:         self.dec = 5
323:         self.type = 3
324: 
325: 
326: class TestIDCTIIIInt(_TestIDCTBase):
327:     def setup_method(self):
328:         self.rdt = int
329:         self.dec = 5
330:         self.type = 3
331: 
332: 
333: class _TestDSTBase(object):
334:     def setup_method(self):
335:         self.rdt = None  # dtype
336:         self.dec = None  # number of decimals to match
337:         self.type = None  # dst type
338: 
339:     def test_definition(self):
340:         for i in FFTWDATA_SIZES:
341:             xr, yr, dt = fftw_dst_ref(self.type, i, self.rdt)
342:             y = dst(xr, type=self.type)
343:             assert_equal(y.dtype, dt)
344:             # XXX: we divide by np.max(y) because the tests fail otherwise. We
345:             # should really use something like assert_array_approx_equal. The
346:             # difference is due to fftw using a better algorithm w.r.t error
347:             # propagation compared to the ones from fftpack.
348:             assert_array_almost_equal(y / np.max(y), yr / np.max(y), decimal=self.dec,
349:                     err_msg="Size %d failed" % i)
350: 
351: 
352: class TestDSTIDouble(_TestDSTBase):
353:     def setup_method(self):
354:         self.rdt = np.double
355:         self.dec = 14
356:         self.type = 1
357: 
358: 
359: class TestDSTIFloat(_TestDSTBase):
360:     def setup_method(self):
361:         self.rdt = np.float32
362:         self.dec = 5
363:         self.type = 1
364: 
365: 
366: class TestDSTIInt(_TestDSTBase):
367:     def setup_method(self):
368:         self.rdt = int
369:         self.dec = 5
370:         self.type = 1
371: 
372: 
373: class TestDSTIIDouble(_TestDSTBase):
374:     def setup_method(self):
375:         self.rdt = np.double
376:         self.dec = 14
377:         self.type = 2
378: 
379: 
380: class TestDSTIIFloat(_TestDSTBase):
381:     def setup_method(self):
382:         self.rdt = np.float32
383:         self.dec = 6
384:         self.type = 2
385: 
386: 
387: class TestDSTIIInt(_TestDSTBase):
388:     def setup_method(self):
389:         self.rdt = int
390:         self.dec = 6
391:         self.type = 2
392: 
393: 
394: class TestDSTIIIDouble(_TestDSTBase):
395:     def setup_method(self):
396:         self.rdt = np.double
397:         self.dec = 14
398:         self.type = 3
399: 
400: 
401: class TestDSTIIIFloat(_TestDSTBase):
402:     def setup_method(self):
403:         self.rdt = np.float32
404:         self.dec = 7
405:         self.type = 3
406: 
407: 
408: class TestDSTIIIInt(_TestDSTBase):
409:     def setup_method(self):
410:         self.rdt = int
411:         self.dec = 7
412:         self.type = 3
413: 
414: 
415: class _TestIDSTBase(object):
416:     def setup_method(self):
417:         self.rdt = None
418:         self.dec = None
419:         self.type = None
420: 
421:     def test_definition(self):
422:         for i in FFTWDATA_SIZES:
423:             xr, yr, dt = fftw_dst_ref(self.type, i, self.rdt)
424:             x = idst(yr, type=self.type)
425:             if self.type == 1:
426:                 x /= 2 * (i+1)
427:             else:
428:                 x /= 2 * i
429:             assert_equal(x.dtype, dt)
430:             # XXX: we divide by np.max(x) because the tests fail otherwise. We
431:             # should really use something like assert_array_approx_equal. The
432:             # difference is due to fftw using a better algorithm w.r.t error
433:             # propagation compared to the ones from fftpack.
434:             assert_array_almost_equal(x / np.max(x), xr / np.max(x), decimal=self.dec,
435:                     err_msg="Size %d failed" % i)
436: 
437: 
438: class TestIDSTIDouble(_TestIDSTBase):
439:     def setup_method(self):
440:         self.rdt = np.double
441:         self.dec = 12
442:         self.type = 1
443: 
444: 
445: class TestIDSTIFloat(_TestIDSTBase):
446:     def setup_method(self):
447:         self.rdt = np.float32
448:         self.dec = 4
449:         self.type = 1
450: 
451: 
452: class TestIDSTIInt(_TestIDSTBase):
453:     def setup_method(self):
454:         self.rdt = int
455:         self.dec = 4
456:         self.type = 1
457: 
458: 
459: class TestIDSTIIDouble(_TestIDSTBase):
460:     def setup_method(self):
461:         self.rdt = np.double
462:         self.dec = 14
463:         self.type = 2
464: 
465: 
466: class TestIDSTIIFloat(_TestIDSTBase):
467:     def setup_method(self):
468:         self.rdt = np.float32
469:         self.dec = 6
470:         self.type = 2
471: 
472: 
473: class TestIDSTIIInt(_TestIDSTBase):
474:     def setup_method(self):
475:         self.rdt = int
476:         self.dec = 6
477:         self.type = 2
478: 
479: 
480: class TestIDSTIIIDouble(_TestIDSTBase):
481:     def setup_method(self):
482:         self.rdt = np.double
483:         self.dec = 14
484:         self.type = 3
485: 
486: 
487: class TestIDSTIIIFloat(_TestIDSTBase):
488:     def setup_method(self):
489:         self.rdt = np.float32
490:         self.dec = 6
491:         self.type = 3
492: 
493: 
494: class TestIDSTIIIInt(_TestIDSTBase):
495:     def setup_method(self):
496:         self.rdt = int
497:         self.dec = 6
498:         self.type = 3
499: 
500: 
501: class TestOverwrite(object):
502:     '''Check input overwrite behavior '''
503: 
504:     real_dtypes = [np.float32, np.float64]
505: 
506:     def _check(self, x, routine, type, fftsize, axis, norm, overwrite_x,
507:                should_overwrite, **kw):
508:         x2 = x.copy()
509:         routine(x2, type, fftsize, axis, norm, overwrite_x=overwrite_x)
510: 
511:         sig = "%s(%s%r, %r, axis=%r, overwrite_x=%r)" % (
512:             routine.__name__, x.dtype, x.shape, fftsize, axis, overwrite_x)
513:         if not should_overwrite:
514:             assert_equal(x2, x, err_msg="spurious overwrite in %s" % sig)
515: 
516:     def _check_1d(self, routine, dtype, shape, axis, overwritable_dtypes):
517:         np.random.seed(1234)
518:         if np.issubdtype(dtype, np.complexfloating):
519:             data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
520:         else:
521:             data = np.random.randn(*shape)
522:         data = data.astype(dtype)
523: 
524:         for type in [1, 2, 3]:
525:             for overwrite_x in [True, False]:
526:                 for norm in [None, 'ortho']:
527:                     if type == 1 and norm == 'ortho':
528:                         continue
529: 
530:                     should_overwrite = (overwrite_x
531:                                         and dtype in overwritable_dtypes
532:                                         and (len(shape) == 1 or
533:                                              (axis % len(shape) == len(shape)-1
534:                                               )))
535:                     self._check(data, routine, type, None, axis, norm,
536:                                 overwrite_x, should_overwrite)
537: 
538:     def test_dct(self):
539:         overwritable = self.real_dtypes
540:         for dtype in self.real_dtypes:
541:             self._check_1d(dct, dtype, (16,), -1, overwritable)
542:             self._check_1d(dct, dtype, (16, 2), 0, overwritable)
543:             self._check_1d(dct, dtype, (2, 16), 1, overwritable)
544: 
545:     def test_idct(self):
546:         overwritable = self.real_dtypes
547:         for dtype in self.real_dtypes:
548:             self._check_1d(idct, dtype, (16,), -1, overwritable)
549:             self._check_1d(idct, dtype, (16, 2), 0, overwritable)
550:             self._check_1d(idct, dtype, (2, 16), 1, overwritable)
551: 
552:     def test_dst(self):
553:         overwritable = self.real_dtypes
554:         for dtype in self.real_dtypes:
555:             self._check_1d(dst, dtype, (16,), -1, overwritable)
556:             self._check_1d(dst, dtype, (16, 2), 0, overwritable)
557:             self._check_1d(dst, dtype, (2, 16), 1, overwritable)
558: 
559:     def test_idst(self):
560:         overwritable = self.real_dtypes
561:         for dtype in self.real_dtypes:
562:             self._check_1d(idst, dtype, (16,), -1, overwritable)
563:             self._check_1d(idst, dtype, (16, 2), 0, overwritable)
564:             self._check_1d(idst, dtype, (2, 16), 1, overwritable)
565: 
566: 
567: class Test_DCTN_IDCTN(object):
568:     dec = 14
569:     types = [1, 2, 3]
570:     norms = [None, 'ortho']
571:     rstate = np.random.RandomState(1234)
572:     shape = (32, 16)
573:     data = rstate.randn(*shape)
574:     # Sets of functions to test
575:     function_sets = [dict(forward=dctn,
576:                           inverse=idctn,
577:                           forward_ref=dct_2d_ref,
578:                           inverse_ref=idct_2d_ref),
579:                      dict(forward=dstn,
580:                           inverse=idstn,
581:                           forward_ref=dst_2d_ref,
582:                           inverse_ref=idst_2d_ref), ]
583: 
584:     def test_axes_round_trip(self):
585:         norm = 'ortho'
586:         for function_set in self.function_sets:
587:             fforward = function_set['forward']
588:             finverse = function_set['inverse']
589:             for axes in [None, (1, ), (0, ), (0, 1), (-2, -1)]:
590:                 for dct_type in self.types:
591:                     if norm == 'ortho' and dct_type == 1:
592:                         continue  # 'ortho' not supported by DCT-I
593:                     tmp = fforward(self.data, type=dct_type, axes=axes,
594:                                    norm=norm)
595:                     tmp = finverse(tmp, type=dct_type, axes=axes, norm=norm)
596:                     assert_array_almost_equal(self.data, tmp, decimal=self.dec)
597: 
598:     def test_dctn_vs_2d_reference(self):
599:         for function_set in self.function_sets:
600:             fforward = function_set['forward']
601:             fforward_ref = function_set['forward_ref']
602:             for dct_type in self.types:
603:                 for norm in self.norms:
604:                     if norm == 'ortho' and dct_type == 1:
605:                         continue  # 'ortho' not supported by DCT-I
606:                     y1 = fforward(self.data, type=dct_type, axes=None,
607:                                   norm=norm)
608:                     y2 = fforward_ref(self.data, type=dct_type, norm=norm)
609:                     assert_array_almost_equal(y1, y2, decimal=11)
610: 
611:     def test_idctn_vs_2d_reference(self):
612:         for function_set in self.function_sets:
613:             finverse = function_set['inverse']
614:             finverse_ref = function_set['inverse_ref']
615:             for dct_type in self.types:
616:                 for norm in self.norms:
617:                     print(function_set, dct_type, norm)
618:                     if norm == 'ortho' and dct_type == 1:
619:                         continue  # 'ortho' not supported by DCT-I
620:                     fdata = dctn(self.data, type=dct_type, norm=norm)
621:                     y1 = finverse(fdata, type=dct_type, norm=norm)
622:                     y2 = finverse_ref(fdata, type=dct_type, norm=norm)
623:                     assert_array_almost_equal(y1, y2, decimal=11)
624: 
625:     def test_axes_and_shape(self):
626:         for function_set in self.function_sets:
627:             fforward = function_set['forward']
628:             finverse = function_set['inverse']
629: 
630:             # shape must match the number of axes
631:             assert_raises(ValueError, fforward, self.data,
632:                           shape=(self.data.shape[0], ),
633:                           axes=(0, 1))
634:             assert_raises(ValueError, fforward, self.data,
635:                           shape=(self.data.shape[0], ),
636:                           axes=None)
637:             assert_raises(ValueError, fforward, self.data,
638:                           shape=self.data.shape,
639:                           axes=(0, ))
640:             # shape must be a tuple
641:             assert_raises(TypeError, fforward, self.data,
642:                           shape=self.data.shape[0],
643:                           axes=(0, 1))
644: 
645:             # shape=None works with a subset of axes
646:             for axes in [(0, ), (1, )]:
647:                 tmp = fforward(self.data, shape=None, axes=axes, norm='ortho')
648:                 tmp = finverse(tmp, shape=None, axes=axes, norm='ortho')
649:                 assert_array_almost_equal(self.data, tmp, decimal=self.dec)
650: 
651:             # non-default shape
652:             tmp = fforward(self.data, shape=(128, 128), axes=None)
653:             assert_equal(tmp.shape, (128, 128))
654: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from os.path import join, dirname' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_26788 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path')

if (type(import_26788) is not StypyTypeError):

    if (import_26788 != 'pyd_module'):
        __import__(import_26788)
        sys_modules_26789 = sys.modules[import_26788]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', sys_modules_26789.module_type_store, module_type_store, ['join', 'dirname'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_26789, sys_modules_26789.module_type_store, module_type_store)
    else:
        from os.path import join, dirname

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', None, module_type_store, ['join', 'dirname'], [join, dirname])

else:
    # Assigning a type to the variable 'os.path' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', import_26788)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_26790 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_26790) is not StypyTypeError):

    if (import_26790 != 'pyd_module'):
        __import__(import_26790)
        sys_modules_26791 = sys.modules[import_26790]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_26791.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_26790)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_equal' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_26792 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_26792) is not StypyTypeError):

    if (import_26792 != 'pyd_module'):
        __import__(import_26792)
        sys_modules_26793 = sys.modules[import_26792]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_26793.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_26793, sys_modules_26793.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_equal'], [assert_array_almost_equal, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_26792)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from pytest import assert_raises' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_26794 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest')

if (type(import_26794) is not StypyTypeError):

    if (import_26794 != 'pyd_module'):
        __import__(import_26794)
        sys_modules_26795 = sys.modules[import_26794]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', sys_modules_26795.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_26795, sys_modules_26795.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', import_26794)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.fftpack.realtransforms import dct, idct, dst, idst, dctn, idctn, dstn, idstn' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_26796 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.fftpack.realtransforms')

if (type(import_26796) is not StypyTypeError):

    if (import_26796 != 'pyd_module'):
        __import__(import_26796)
        sys_modules_26797 = sys.modules[import_26796]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.fftpack.realtransforms', sys_modules_26797.module_type_store, module_type_store, ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_26797, sys_modules_26797.module_type_store, module_type_store)
    else:
        from scipy.fftpack.realtransforms import dct, idct, dst, idst, dctn, idctn, dstn, idstn

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.fftpack.realtransforms', None, module_type_store, ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn'], [dct, idct, dst, idst, dctn, idctn, dstn, idstn])

else:
    # Assigning a type to the variable 'scipy.fftpack.realtransforms' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.fftpack.realtransforms', import_26796)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')


# Assigning a Call to a Name (line 13):

# Assigning a Call to a Name (line 13):

# Call to load(...): (line 13)
# Processing the call arguments (line 13)

# Call to join(...): (line 13)
# Processing the call arguments (line 13)

# Call to dirname(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of '__file__' (line 13)
file___26802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), '__file__', False)
# Processing the call keyword arguments (line 13)
kwargs_26803 = {}
# Getting the type of 'dirname' (line 13)
dirname_26801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 21), 'dirname', False)
# Calling dirname(args, kwargs) (line 13)
dirname_call_result_26804 = invoke(stypy.reporting.localization.Localization(__file__, 13, 21), dirname_26801, *[file___26802], **kwargs_26803)

str_26805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 40), 'str', 'test.npz')
# Processing the call keyword arguments (line 13)
kwargs_26806 = {}
# Getting the type of 'join' (line 13)
join_26800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'join', False)
# Calling join(args, kwargs) (line 13)
join_call_result_26807 = invoke(stypy.reporting.localization.Localization(__file__, 13, 16), join_26800, *[dirname_call_result_26804, str_26805], **kwargs_26806)

# Processing the call keyword arguments (line 13)
kwargs_26808 = {}
# Getting the type of 'np' (line 13)
np_26798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'np', False)
# Obtaining the member 'load' of a type (line 13)
load_26799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), np_26798, 'load')
# Calling load(args, kwargs) (line 13)
load_call_result_26809 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), load_26799, *[join_call_result_26807], **kwargs_26808)

# Assigning a type to the variable 'MDATA' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'MDATA', load_call_result_26809)

# Assigning a ListComp to a Name (line 14):

# Assigning a ListComp to a Name (line 14):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 14)
# Processing the call arguments (line 14)
int_26817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 37), 'int')
# Processing the call keyword arguments (line 14)
kwargs_26818 = {}
# Getting the type of 'range' (line 14)
range_26816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 31), 'range', False)
# Calling range(args, kwargs) (line 14)
range_call_result_26819 = invoke(stypy.reporting.localization.Localization(__file__, 14, 31), range_26816, *[int_26817], **kwargs_26818)

comprehension_26820 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 5), range_call_result_26819)
# Assigning a type to the variable 'i' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'i', comprehension_26820)

# Obtaining the type of the subscript
str_26810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'x%d')
# Getting the type of 'i' (line 14)
i_26811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'i')
# Applying the binary operator '%' (line 14)
result_mod_26812 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 11), '%', str_26810, i_26811)

# Getting the type of 'MDATA' (line 14)
MDATA_26813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'MDATA')
# Obtaining the member '__getitem__' of a type (line 14)
getitem___26814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), MDATA_26813, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 14)
subscript_call_result_26815 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), getitem___26814, result_mod_26812)

list_26821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 5), list_26821, subscript_call_result_26815)
# Assigning a type to the variable 'X' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'X', list_26821)

# Assigning a ListComp to a Name (line 15):

# Assigning a ListComp to a Name (line 15):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 15)
# Processing the call arguments (line 15)
int_26829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 37), 'int')
# Processing the call keyword arguments (line 15)
kwargs_26830 = {}
# Getting the type of 'range' (line 15)
range_26828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 31), 'range', False)
# Calling range(args, kwargs) (line 15)
range_call_result_26831 = invoke(stypy.reporting.localization.Localization(__file__, 15, 31), range_26828, *[int_26829], **kwargs_26830)

comprehension_26832 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 5), range_call_result_26831)
# Assigning a type to the variable 'i' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'i', comprehension_26832)

# Obtaining the type of the subscript
str_26822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'y%d')
# Getting the type of 'i' (line 15)
i_26823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'i')
# Applying the binary operator '%' (line 15)
result_mod_26824 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 11), '%', str_26822, i_26823)

# Getting the type of 'MDATA' (line 15)
MDATA_26825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'MDATA')
# Obtaining the member '__getitem__' of a type (line 15)
getitem___26826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), MDATA_26825, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 15)
subscript_call_result_26827 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), getitem___26826, result_mod_26824)

list_26833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 5), list_26833, subscript_call_result_26827)
# Assigning a type to the variable 'Y' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'Y', list_26833)

# Assigning a Call to a Name (line 22):

# Assigning a Call to a Name (line 22):

# Call to load(...): (line 22)
# Processing the call arguments (line 22)

# Call to join(...): (line 22)
# Processing the call arguments (line 22)

# Call to dirname(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of '__file__' (line 22)
file___26838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), '__file__', False)
# Processing the call keyword arguments (line 22)
kwargs_26839 = {}
# Getting the type of 'dirname' (line 22)
dirname_26837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 31), 'dirname', False)
# Calling dirname(args, kwargs) (line 22)
dirname_call_result_26840 = invoke(stypy.reporting.localization.Localization(__file__, 22, 31), dirname_26837, *[file___26838], **kwargs_26839)

str_26841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 50), 'str', 'fftw_double_ref.npz')
# Processing the call keyword arguments (line 22)
kwargs_26842 = {}
# Getting the type of 'join' (line 22)
join_26836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'join', False)
# Calling join(args, kwargs) (line 22)
join_call_result_26843 = invoke(stypy.reporting.localization.Localization(__file__, 22, 26), join_26836, *[dirname_call_result_26840, str_26841], **kwargs_26842)

# Processing the call keyword arguments (line 22)
kwargs_26844 = {}
# Getting the type of 'np' (line 22)
np_26834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'np', False)
# Obtaining the member 'load' of a type (line 22)
load_26835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 18), np_26834, 'load')
# Calling load(args, kwargs) (line 22)
load_call_result_26845 = invoke(stypy.reporting.localization.Localization(__file__, 22, 18), load_26835, *[join_call_result_26843], **kwargs_26844)

# Assigning a type to the variable 'FFTWDATA_DOUBLE' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'FFTWDATA_DOUBLE', load_call_result_26845)

# Assigning a Call to a Name (line 23):

# Assigning a Call to a Name (line 23):

# Call to load(...): (line 23)
# Processing the call arguments (line 23)

# Call to join(...): (line 23)
# Processing the call arguments (line 23)

# Call to dirname(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of '__file__' (line 23)
file___26850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 39), '__file__', False)
# Processing the call keyword arguments (line 23)
kwargs_26851 = {}
# Getting the type of 'dirname' (line 23)
dirname_26849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 31), 'dirname', False)
# Calling dirname(args, kwargs) (line 23)
dirname_call_result_26852 = invoke(stypy.reporting.localization.Localization(__file__, 23, 31), dirname_26849, *[file___26850], **kwargs_26851)

str_26853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 50), 'str', 'fftw_single_ref.npz')
# Processing the call keyword arguments (line 23)
kwargs_26854 = {}
# Getting the type of 'join' (line 23)
join_26848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'join', False)
# Calling join(args, kwargs) (line 23)
join_call_result_26855 = invoke(stypy.reporting.localization.Localization(__file__, 23, 26), join_26848, *[dirname_call_result_26852, str_26853], **kwargs_26854)

# Processing the call keyword arguments (line 23)
kwargs_26856 = {}
# Getting the type of 'np' (line 23)
np_26846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 'np', False)
# Obtaining the member 'load' of a type (line 23)
load_26847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 18), np_26846, 'load')
# Calling load(args, kwargs) (line 23)
load_call_result_26857 = invoke(stypy.reporting.localization.Localization(__file__, 23, 18), load_26847, *[join_call_result_26855], **kwargs_26856)

# Assigning a type to the variable 'FFTWDATA_SINGLE' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'FFTWDATA_SINGLE', load_call_result_26857)

# Assigning a Subscript to a Name (line 24):

# Assigning a Subscript to a Name (line 24):

# Obtaining the type of the subscript
str_26858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 33), 'str', 'sizes')
# Getting the type of 'FFTWDATA_DOUBLE' (line 24)
FFTWDATA_DOUBLE_26859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'FFTWDATA_DOUBLE')
# Obtaining the member '__getitem__' of a type (line 24)
getitem___26860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 17), FFTWDATA_DOUBLE_26859, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 24)
subscript_call_result_26861 = invoke(stypy.reporting.localization.Localization(__file__, 24, 17), getitem___26860, str_26858)

# Assigning a type to the variable 'FFTWDATA_SIZES' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'FFTWDATA_SIZES', subscript_call_result_26861)

@norecursion
def fftw_dct_ref(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fftw_dct_ref'
    module_type_store = module_type_store.open_function_context('fftw_dct_ref', 27, 0, False)
    
    # Passed parameters checking function
    fftw_dct_ref.stypy_localization = localization
    fftw_dct_ref.stypy_type_of_self = None
    fftw_dct_ref.stypy_type_store = module_type_store
    fftw_dct_ref.stypy_function_name = 'fftw_dct_ref'
    fftw_dct_ref.stypy_param_names_list = ['type', 'size', 'dt']
    fftw_dct_ref.stypy_varargs_param_name = None
    fftw_dct_ref.stypy_kwargs_param_name = None
    fftw_dct_ref.stypy_call_defaults = defaults
    fftw_dct_ref.stypy_call_varargs = varargs
    fftw_dct_ref.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fftw_dct_ref', ['type', 'size', 'dt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fftw_dct_ref', localization, ['type', 'size', 'dt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fftw_dct_ref(...)' code ##################

    
    # Assigning a Call to a Name (line 28):
    
    # Assigning a Call to a Name (line 28):
    
    # Call to astype(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'dt' (line 28)
    dt_26872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 44), 'dt', False)
    # Processing the call keyword arguments (line 28)
    kwargs_26873 = {}
    
    # Call to linspace(...): (line 28)
    # Processing the call arguments (line 28)
    int_26864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 20), 'int')
    # Getting the type of 'size' (line 28)
    size_26865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'size', False)
    int_26866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'int')
    # Applying the binary operator '-' (line 28)
    result_sub_26867 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 23), '-', size_26865, int_26866)
    
    # Getting the type of 'size' (line 28)
    size_26868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'size', False)
    # Processing the call keyword arguments (line 28)
    kwargs_26869 = {}
    # Getting the type of 'np' (line 28)
    np_26862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 28)
    linspace_26863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), np_26862, 'linspace')
    # Calling linspace(args, kwargs) (line 28)
    linspace_call_result_26870 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), linspace_26863, *[int_26864, result_sub_26867, size_26868], **kwargs_26869)
    
    # Obtaining the member 'astype' of a type (line 28)
    astype_26871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), linspace_call_result_26870, 'astype')
    # Calling astype(args, kwargs) (line 28)
    astype_call_result_26874 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), astype_26871, *[dt_26872], **kwargs_26873)
    
    # Assigning a type to the variable 'x' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'x', astype_call_result_26874)
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to result_type(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'np' (line 29)
    np_26877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'np', False)
    # Obtaining the member 'float32' of a type (line 29)
    float32_26878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 24), np_26877, 'float32')
    # Getting the type of 'dt' (line 29)
    dt_26879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 36), 'dt', False)
    # Processing the call keyword arguments (line 29)
    kwargs_26880 = {}
    # Getting the type of 'np' (line 29)
    np_26875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 9), 'np', False)
    # Obtaining the member 'result_type' of a type (line 29)
    result_type_26876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 9), np_26875, 'result_type')
    # Calling result_type(args, kwargs) (line 29)
    result_type_call_result_26881 = invoke(stypy.reporting.localization.Localization(__file__, 29, 9), result_type_26876, *[float32_26878, dt_26879], **kwargs_26880)
    
    # Assigning a type to the variable 'dt' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'dt', result_type_call_result_26881)
    
    
    # Getting the type of 'dt' (line 30)
    dt_26882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'dt')
    # Getting the type of 'np' (line 30)
    np_26883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'np')
    # Obtaining the member 'double' of a type (line 30)
    double_26884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 13), np_26883, 'double')
    # Applying the binary operator '==' (line 30)
    result_eq_26885 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 7), '==', dt_26882, double_26884)
    
    # Testing the type of an if condition (line 30)
    if_condition_26886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 4), result_eq_26885)
    # Assigning a type to the variable 'if_condition_26886' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'if_condition_26886', if_condition_26886)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 31):
    
    # Assigning a Name to a Name (line 31):
    # Getting the type of 'FFTWDATA_DOUBLE' (line 31)
    FFTWDATA_DOUBLE_26887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'FFTWDATA_DOUBLE')
    # Assigning a type to the variable 'data' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'data', FFTWDATA_DOUBLE_26887)
    # SSA branch for the else part of an if statement (line 30)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dt' (line 32)
    dt_26888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 9), 'dt')
    # Getting the type of 'np' (line 32)
    np_26889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'np')
    # Obtaining the member 'float32' of a type (line 32)
    float32_26890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), np_26889, 'float32')
    # Applying the binary operator '==' (line 32)
    result_eq_26891 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 9), '==', dt_26888, float32_26890)
    
    # Testing the type of an if condition (line 32)
    if_condition_26892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 9), result_eq_26891)
    # Assigning a type to the variable 'if_condition_26892' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 9), 'if_condition_26892', if_condition_26892)
    # SSA begins for if statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 33):
    
    # Assigning a Name to a Name (line 33):
    # Getting the type of 'FFTWDATA_SINGLE' (line 33)
    FFTWDATA_SINGLE_26893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'FFTWDATA_SINGLE')
    # Assigning a type to the variable 'data' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'data', FFTWDATA_SINGLE_26893)
    # SSA branch for the else part of an if statement (line 32)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 35)
    # Processing the call keyword arguments (line 35)
    kwargs_26895 = {}
    # Getting the type of 'ValueError' (line 35)
    ValueError_26894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 35)
    ValueError_call_result_26896 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), ValueError_26894, *[], **kwargs_26895)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 35, 8), ValueError_call_result_26896, 'raise parameter', BaseException)
    # SSA join for if statement (line 32)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 36):
    
    # Assigning a Call to a Name (line 36):
    
    # Call to astype(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'dt' (line 36)
    dt_26906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 50), 'dt', False)
    # Processing the call keyword arguments (line 36)
    kwargs_26907 = {}
    
    # Obtaining the type of the subscript
    str_26897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'str', 'dct_%d_%d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_26898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    # Getting the type of 'type' (line 36)
    type_26899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 29), 'type', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 29), tuple_26898, type_26899)
    # Adding element type (line 36)
    # Getting the type of 'size' (line 36)
    size_26900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 35), 'size', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 29), tuple_26898, size_26900)
    
    # Applying the binary operator '%' (line 36)
    result_mod_26901 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 14), '%', str_26897, tuple_26898)
    
    # Getting the type of 'data' (line 36)
    data_26902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'data', False)
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___26903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 9), data_26902, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_26904 = invoke(stypy.reporting.localization.Localization(__file__, 36, 9), getitem___26903, result_mod_26901)
    
    # Obtaining the member 'astype' of a type (line 36)
    astype_26905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 9), subscript_call_result_26904, 'astype')
    # Calling astype(args, kwargs) (line 36)
    astype_call_result_26908 = invoke(stypy.reporting.localization.Localization(__file__, 36, 9), astype_26905, *[dt_26906], **kwargs_26907)
    
    # Assigning a type to the variable 'y' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'y', astype_call_result_26908)
    
    # Obtaining an instance of the builtin type 'tuple' (line 37)
    tuple_26909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 37)
    # Adding element type (line 37)
    # Getting the type of 'x' (line 37)
    x_26910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 11), tuple_26909, x_26910)
    # Adding element type (line 37)
    # Getting the type of 'y' (line 37)
    y_26911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 11), tuple_26909, y_26911)
    # Adding element type (line 37)
    # Getting the type of 'dt' (line 37)
    dt_26912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'dt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 11), tuple_26909, dt_26912)
    
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type', tuple_26909)
    
    # ################# End of 'fftw_dct_ref(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fftw_dct_ref' in the type store
    # Getting the type of 'stypy_return_type' (line 27)
    stypy_return_type_26913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26913)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fftw_dct_ref'
    return stypy_return_type_26913

# Assigning a type to the variable 'fftw_dct_ref' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'fftw_dct_ref', fftw_dct_ref)

@norecursion
def fftw_dst_ref(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fftw_dst_ref'
    module_type_store = module_type_store.open_function_context('fftw_dst_ref', 40, 0, False)
    
    # Passed parameters checking function
    fftw_dst_ref.stypy_localization = localization
    fftw_dst_ref.stypy_type_of_self = None
    fftw_dst_ref.stypy_type_store = module_type_store
    fftw_dst_ref.stypy_function_name = 'fftw_dst_ref'
    fftw_dst_ref.stypy_param_names_list = ['type', 'size', 'dt']
    fftw_dst_ref.stypy_varargs_param_name = None
    fftw_dst_ref.stypy_kwargs_param_name = None
    fftw_dst_ref.stypy_call_defaults = defaults
    fftw_dst_ref.stypy_call_varargs = varargs
    fftw_dst_ref.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fftw_dst_ref', ['type', 'size', 'dt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fftw_dst_ref', localization, ['type', 'size', 'dt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fftw_dst_ref(...)' code ##################

    
    # Assigning a Call to a Name (line 41):
    
    # Assigning a Call to a Name (line 41):
    
    # Call to astype(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'dt' (line 41)
    dt_26924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 44), 'dt', False)
    # Processing the call keyword arguments (line 41)
    kwargs_26925 = {}
    
    # Call to linspace(...): (line 41)
    # Processing the call arguments (line 41)
    int_26916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'int')
    # Getting the type of 'size' (line 41)
    size_26917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'size', False)
    int_26918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 28), 'int')
    # Applying the binary operator '-' (line 41)
    result_sub_26919 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 23), '-', size_26917, int_26918)
    
    # Getting the type of 'size' (line 41)
    size_26920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'size', False)
    # Processing the call keyword arguments (line 41)
    kwargs_26921 = {}
    # Getting the type of 'np' (line 41)
    np_26914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 41)
    linspace_26915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), np_26914, 'linspace')
    # Calling linspace(args, kwargs) (line 41)
    linspace_call_result_26922 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), linspace_26915, *[int_26916, result_sub_26919, size_26920], **kwargs_26921)
    
    # Obtaining the member 'astype' of a type (line 41)
    astype_26923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), linspace_call_result_26922, 'astype')
    # Calling astype(args, kwargs) (line 41)
    astype_call_result_26926 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), astype_26923, *[dt_26924], **kwargs_26925)
    
    # Assigning a type to the variable 'x' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'x', astype_call_result_26926)
    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to result_type(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'np' (line 42)
    np_26929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'np', False)
    # Obtaining the member 'float32' of a type (line 42)
    float32_26930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), np_26929, 'float32')
    # Getting the type of 'dt' (line 42)
    dt_26931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 36), 'dt', False)
    # Processing the call keyword arguments (line 42)
    kwargs_26932 = {}
    # Getting the type of 'np' (line 42)
    np_26927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 9), 'np', False)
    # Obtaining the member 'result_type' of a type (line 42)
    result_type_26928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 9), np_26927, 'result_type')
    # Calling result_type(args, kwargs) (line 42)
    result_type_call_result_26933 = invoke(stypy.reporting.localization.Localization(__file__, 42, 9), result_type_26928, *[float32_26930, dt_26931], **kwargs_26932)
    
    # Assigning a type to the variable 'dt' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'dt', result_type_call_result_26933)
    
    
    # Getting the type of 'dt' (line 43)
    dt_26934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 'dt')
    # Getting the type of 'np' (line 43)
    np_26935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'np')
    # Obtaining the member 'double' of a type (line 43)
    double_26936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 13), np_26935, 'double')
    # Applying the binary operator '==' (line 43)
    result_eq_26937 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), '==', dt_26934, double_26936)
    
    # Testing the type of an if condition (line 43)
    if_condition_26938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 4), result_eq_26937)
    # Assigning a type to the variable 'if_condition_26938' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'if_condition_26938', if_condition_26938)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 44):
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'FFTWDATA_DOUBLE' (line 44)
    FFTWDATA_DOUBLE_26939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'FFTWDATA_DOUBLE')
    # Assigning a type to the variable 'data' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'data', FFTWDATA_DOUBLE_26939)
    # SSA branch for the else part of an if statement (line 43)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dt' (line 45)
    dt_26940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 9), 'dt')
    # Getting the type of 'np' (line 45)
    np_26941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'np')
    # Obtaining the member 'float32' of a type (line 45)
    float32_26942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 15), np_26941, 'float32')
    # Applying the binary operator '==' (line 45)
    result_eq_26943 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 9), '==', dt_26940, float32_26942)
    
    # Testing the type of an if condition (line 45)
    if_condition_26944 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 9), result_eq_26943)
    # Assigning a type to the variable 'if_condition_26944' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 9), 'if_condition_26944', if_condition_26944)
    # SSA begins for if statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 46):
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'FFTWDATA_SINGLE' (line 46)
    FFTWDATA_SINGLE_26945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'FFTWDATA_SINGLE')
    # Assigning a type to the variable 'data' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'data', FFTWDATA_SINGLE_26945)
    # SSA branch for the else part of an if statement (line 45)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 48)
    # Processing the call keyword arguments (line 48)
    kwargs_26947 = {}
    # Getting the type of 'ValueError' (line 48)
    ValueError_26946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 48)
    ValueError_call_result_26948 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), ValueError_26946, *[], **kwargs_26947)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 48, 8), ValueError_call_result_26948, 'raise parameter', BaseException)
    # SSA join for if statement (line 45)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 49):
    
    # Assigning a Call to a Name (line 49):
    
    # Call to astype(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'dt' (line 49)
    dt_26958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 50), 'dt', False)
    # Processing the call keyword arguments (line 49)
    kwargs_26959 = {}
    
    # Obtaining the type of the subscript
    str_26949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 14), 'str', 'dst_%d_%d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 49)
    tuple_26950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 49)
    # Adding element type (line 49)
    # Getting the type of 'type' (line 49)
    type_26951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'type', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), tuple_26950, type_26951)
    # Adding element type (line 49)
    # Getting the type of 'size' (line 49)
    size_26952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 35), 'size', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), tuple_26950, size_26952)
    
    # Applying the binary operator '%' (line 49)
    result_mod_26953 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 14), '%', str_26949, tuple_26950)
    
    # Getting the type of 'data' (line 49)
    data_26954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 9), 'data', False)
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___26955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 9), data_26954, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_26956 = invoke(stypy.reporting.localization.Localization(__file__, 49, 9), getitem___26955, result_mod_26953)
    
    # Obtaining the member 'astype' of a type (line 49)
    astype_26957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 9), subscript_call_result_26956, 'astype')
    # Calling astype(args, kwargs) (line 49)
    astype_call_result_26960 = invoke(stypy.reporting.localization.Localization(__file__, 49, 9), astype_26957, *[dt_26958], **kwargs_26959)
    
    # Assigning a type to the variable 'y' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'y', astype_call_result_26960)
    
    # Obtaining an instance of the builtin type 'tuple' (line 50)
    tuple_26961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 50)
    # Adding element type (line 50)
    # Getting the type of 'x' (line 50)
    x_26962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_26961, x_26962)
    # Adding element type (line 50)
    # Getting the type of 'y' (line 50)
    y_26963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_26961, y_26963)
    # Adding element type (line 50)
    # Getting the type of 'dt' (line 50)
    dt_26964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'dt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_26961, dt_26964)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', tuple_26961)
    
    # ################# End of 'fftw_dst_ref(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fftw_dst_ref' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_26965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26965)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fftw_dst_ref'
    return stypy_return_type_26965

# Assigning a type to the variable 'fftw_dst_ref' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'fftw_dst_ref', fftw_dst_ref)

@norecursion
def dct_2d_ref(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dct_2d_ref'
    module_type_store = module_type_store.open_function_context('dct_2d_ref', 53, 0, False)
    
    # Passed parameters checking function
    dct_2d_ref.stypy_localization = localization
    dct_2d_ref.stypy_type_of_self = None
    dct_2d_ref.stypy_type_store = module_type_store
    dct_2d_ref.stypy_function_name = 'dct_2d_ref'
    dct_2d_ref.stypy_param_names_list = ['x']
    dct_2d_ref.stypy_varargs_param_name = None
    dct_2d_ref.stypy_kwargs_param_name = 'kwargs'
    dct_2d_ref.stypy_call_defaults = defaults
    dct_2d_ref.stypy_call_varargs = varargs
    dct_2d_ref.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dct_2d_ref', ['x'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dct_2d_ref', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dct_2d_ref(...)' code ##################

    str_26966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 4), 'str', ' used as a reference in testing dct2. ')
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to array(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'x' (line 55)
    x_26969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'x', False)
    # Processing the call keyword arguments (line 55)
    # Getting the type of 'True' (line 55)
    True_26970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'True', False)
    keyword_26971 = True_26970
    kwargs_26972 = {'copy': keyword_26971}
    # Getting the type of 'np' (line 55)
    np_26967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 55)
    array_26968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), np_26967, 'array')
    # Calling array(args, kwargs) (line 55)
    array_call_result_26973 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), array_26968, *[x_26969], **kwargs_26972)
    
    # Assigning a type to the variable 'x' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'x', array_call_result_26973)
    
    
    # Call to range(...): (line 56)
    # Processing the call arguments (line 56)
    
    # Obtaining the type of the subscript
    int_26975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 29), 'int')
    # Getting the type of 'x' (line 56)
    x_26976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 56)
    shape_26977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 21), x_26976, 'shape')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___26978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 21), shape_26977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_26979 = invoke(stypy.reporting.localization.Localization(__file__, 56, 21), getitem___26978, int_26975)
    
    # Processing the call keyword arguments (line 56)
    kwargs_26980 = {}
    # Getting the type of 'range' (line 56)
    range_26974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'range', False)
    # Calling range(args, kwargs) (line 56)
    range_call_result_26981 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), range_26974, *[subscript_call_result_26979], **kwargs_26980)
    
    # Testing the type of a for loop iterable (line 56)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 56, 4), range_call_result_26981)
    # Getting the type of the for loop variable (line 56)
    for_loop_var_26982 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 56, 4), range_call_result_26981)
    # Assigning a type to the variable 'row' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'row', for_loop_var_26982)
    # SSA begins for a for statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 57):
    
    # Assigning a Call to a Subscript (line 57):
    
    # Call to dct(...): (line 57)
    # Processing the call arguments (line 57)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 57)
    row_26984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'row', False)
    slice_26985 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 57, 24), None, None, None)
    # Getting the type of 'x' (line 57)
    x_26986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___26987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 24), x_26986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_26988 = invoke(stypy.reporting.localization.Localization(__file__, 57, 24), getitem___26987, (row_26984, slice_26985))
    
    # Processing the call keyword arguments (line 57)
    # Getting the type of 'kwargs' (line 57)
    kwargs_26989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), 'kwargs', False)
    kwargs_26990 = {'kwargs_26989': kwargs_26989}
    # Getting the type of 'dct' (line 57)
    dct_26983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'dct', False)
    # Calling dct(args, kwargs) (line 57)
    dct_call_result_26991 = invoke(stypy.reporting.localization.Localization(__file__, 57, 20), dct_26983, *[subscript_call_result_26988], **kwargs_26990)
    
    # Getting the type of 'x' (line 57)
    x_26992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'x')
    # Getting the type of 'row' (line 57)
    row_26993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 10), 'row')
    slice_26994 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 57, 8), None, None, None)
    # Storing an element on a container (line 57)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 8), x_26992, ((row_26993, slice_26994), dct_call_result_26991))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 58)
    # Processing the call arguments (line 58)
    
    # Obtaining the type of the subscript
    int_26996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 29), 'int')
    # Getting the type of 'x' (line 58)
    x_26997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 58)
    shape_26998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 21), x_26997, 'shape')
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___26999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 21), shape_26998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 58)
    subscript_call_result_27000 = invoke(stypy.reporting.localization.Localization(__file__, 58, 21), getitem___26999, int_26996)
    
    # Processing the call keyword arguments (line 58)
    kwargs_27001 = {}
    # Getting the type of 'range' (line 58)
    range_26995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'range', False)
    # Calling range(args, kwargs) (line 58)
    range_call_result_27002 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), range_26995, *[subscript_call_result_27000], **kwargs_27001)
    
    # Testing the type of a for loop iterable (line 58)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 4), range_call_result_27002)
    # Getting the type of the for loop variable (line 58)
    for_loop_var_27003 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 4), range_call_result_27002)
    # Assigning a type to the variable 'col' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'col', for_loop_var_27003)
    # SSA begins for a for statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 59):
    
    # Assigning a Call to a Subscript (line 59):
    
    # Call to dct(...): (line 59)
    # Processing the call arguments (line 59)
    
    # Obtaining the type of the subscript
    slice_27005 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 59, 24), None, None, None)
    # Getting the type of 'col' (line 59)
    col_27006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'col', False)
    # Getting the type of 'x' (line 59)
    x_27007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 59)
    getitem___27008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 24), x_27007, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 59)
    subscript_call_result_27009 = invoke(stypy.reporting.localization.Localization(__file__, 59, 24), getitem___27008, (slice_27005, col_27006))
    
    # Processing the call keyword arguments (line 59)
    # Getting the type of 'kwargs' (line 59)
    kwargs_27010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 37), 'kwargs', False)
    kwargs_27011 = {'kwargs_27010': kwargs_27010}
    # Getting the type of 'dct' (line 59)
    dct_27004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'dct', False)
    # Calling dct(args, kwargs) (line 59)
    dct_call_result_27012 = invoke(stypy.reporting.localization.Localization(__file__, 59, 20), dct_27004, *[subscript_call_result_27009], **kwargs_27011)
    
    # Getting the type of 'x' (line 59)
    x_27013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'x')
    slice_27014 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 59, 8), None, None, None)
    # Getting the type of 'col' (line 59)
    col_27015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'col')
    # Storing an element on a container (line 59)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 8), x_27013, ((slice_27014, col_27015), dct_call_result_27012))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 60)
    x_27016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type', x_27016)
    
    # ################# End of 'dct_2d_ref(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dct_2d_ref' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_27017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27017)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dct_2d_ref'
    return stypy_return_type_27017

# Assigning a type to the variable 'dct_2d_ref' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'dct_2d_ref', dct_2d_ref)

@norecursion
def idct_2d_ref(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idct_2d_ref'
    module_type_store = module_type_store.open_function_context('idct_2d_ref', 63, 0, False)
    
    # Passed parameters checking function
    idct_2d_ref.stypy_localization = localization
    idct_2d_ref.stypy_type_of_self = None
    idct_2d_ref.stypy_type_store = module_type_store
    idct_2d_ref.stypy_function_name = 'idct_2d_ref'
    idct_2d_ref.stypy_param_names_list = ['x']
    idct_2d_ref.stypy_varargs_param_name = None
    idct_2d_ref.stypy_kwargs_param_name = 'kwargs'
    idct_2d_ref.stypy_call_defaults = defaults
    idct_2d_ref.stypy_call_varargs = varargs
    idct_2d_ref.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idct_2d_ref', ['x'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idct_2d_ref', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idct_2d_ref(...)' code ##################

    str_27018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'str', ' used as a reference in testing idct2. ')
    
    # Assigning a Call to a Name (line 65):
    
    # Assigning a Call to a Name (line 65):
    
    # Call to array(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'x' (line 65)
    x_27021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 17), 'x', False)
    # Processing the call keyword arguments (line 65)
    # Getting the type of 'True' (line 65)
    True_27022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'True', False)
    keyword_27023 = True_27022
    kwargs_27024 = {'copy': keyword_27023}
    # Getting the type of 'np' (line 65)
    np_27019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 65)
    array_27020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), np_27019, 'array')
    # Calling array(args, kwargs) (line 65)
    array_call_result_27025 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), array_27020, *[x_27021], **kwargs_27024)
    
    # Assigning a type to the variable 'x' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'x', array_call_result_27025)
    
    
    # Call to range(...): (line 66)
    # Processing the call arguments (line 66)
    
    # Obtaining the type of the subscript
    int_27027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 29), 'int')
    # Getting the type of 'x' (line 66)
    x_27028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 66)
    shape_27029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 21), x_27028, 'shape')
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___27030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 21), shape_27029, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_27031 = invoke(stypy.reporting.localization.Localization(__file__, 66, 21), getitem___27030, int_27027)
    
    # Processing the call keyword arguments (line 66)
    kwargs_27032 = {}
    # Getting the type of 'range' (line 66)
    range_27026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'range', False)
    # Calling range(args, kwargs) (line 66)
    range_call_result_27033 = invoke(stypy.reporting.localization.Localization(__file__, 66, 15), range_27026, *[subscript_call_result_27031], **kwargs_27032)
    
    # Testing the type of a for loop iterable (line 66)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 66, 4), range_call_result_27033)
    # Getting the type of the for loop variable (line 66)
    for_loop_var_27034 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 66, 4), range_call_result_27033)
    # Assigning a type to the variable 'row' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'row', for_loop_var_27034)
    # SSA begins for a for statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 67):
    
    # Assigning a Call to a Subscript (line 67):
    
    # Call to idct(...): (line 67)
    # Processing the call arguments (line 67)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 67)
    row_27036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 27), 'row', False)
    slice_27037 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 67, 25), None, None, None)
    # Getting the type of 'x' (line 67)
    x_27038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___27039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 25), x_27038, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_27040 = invoke(stypy.reporting.localization.Localization(__file__, 67, 25), getitem___27039, (row_27036, slice_27037))
    
    # Processing the call keyword arguments (line 67)
    # Getting the type of 'kwargs' (line 67)
    kwargs_27041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 38), 'kwargs', False)
    kwargs_27042 = {'kwargs_27041': kwargs_27041}
    # Getting the type of 'idct' (line 67)
    idct_27035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'idct', False)
    # Calling idct(args, kwargs) (line 67)
    idct_call_result_27043 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), idct_27035, *[subscript_call_result_27040], **kwargs_27042)
    
    # Getting the type of 'x' (line 67)
    x_27044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'x')
    # Getting the type of 'row' (line 67)
    row_27045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 10), 'row')
    slice_27046 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 67, 8), None, None, None)
    # Storing an element on a container (line 67)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), x_27044, ((row_27045, slice_27046), idct_call_result_27043))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining the type of the subscript
    int_27048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'int')
    # Getting the type of 'x' (line 68)
    x_27049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 68)
    shape_27050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 21), x_27049, 'shape')
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___27051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 21), shape_27050, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_27052 = invoke(stypy.reporting.localization.Localization(__file__, 68, 21), getitem___27051, int_27048)
    
    # Processing the call keyword arguments (line 68)
    kwargs_27053 = {}
    # Getting the type of 'range' (line 68)
    range_27047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'range', False)
    # Calling range(args, kwargs) (line 68)
    range_call_result_27054 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), range_27047, *[subscript_call_result_27052], **kwargs_27053)
    
    # Testing the type of a for loop iterable (line 68)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 68, 4), range_call_result_27054)
    # Getting the type of the for loop variable (line 68)
    for_loop_var_27055 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 68, 4), range_call_result_27054)
    # Assigning a type to the variable 'col' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'col', for_loop_var_27055)
    # SSA begins for a for statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 69):
    
    # Assigning a Call to a Subscript (line 69):
    
    # Call to idct(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Obtaining the type of the subscript
    slice_27057 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 25), None, None, None)
    # Getting the type of 'col' (line 69)
    col_27058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'col', False)
    # Getting the type of 'x' (line 69)
    x_27059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___27060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), x_27059, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_27061 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), getitem___27060, (slice_27057, col_27058))
    
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'kwargs' (line 69)
    kwargs_27062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 38), 'kwargs', False)
    kwargs_27063 = {'kwargs_27062': kwargs_27062}
    # Getting the type of 'idct' (line 69)
    idct_27056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'idct', False)
    # Calling idct(args, kwargs) (line 69)
    idct_call_result_27064 = invoke(stypy.reporting.localization.Localization(__file__, 69, 20), idct_27056, *[subscript_call_result_27061], **kwargs_27063)
    
    # Getting the type of 'x' (line 69)
    x_27065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'x')
    slice_27066 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 8), None, None, None)
    # Getting the type of 'col' (line 69)
    col_27067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'col')
    # Storing an element on a container (line 69)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 8), x_27065, ((slice_27066, col_27067), idct_call_result_27064))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 70)
    x_27068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', x_27068)
    
    # ################# End of 'idct_2d_ref(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idct_2d_ref' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_27069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27069)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idct_2d_ref'
    return stypy_return_type_27069

# Assigning a type to the variable 'idct_2d_ref' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'idct_2d_ref', idct_2d_ref)

@norecursion
def dst_2d_ref(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dst_2d_ref'
    module_type_store = module_type_store.open_function_context('dst_2d_ref', 73, 0, False)
    
    # Passed parameters checking function
    dst_2d_ref.stypy_localization = localization
    dst_2d_ref.stypy_type_of_self = None
    dst_2d_ref.stypy_type_store = module_type_store
    dst_2d_ref.stypy_function_name = 'dst_2d_ref'
    dst_2d_ref.stypy_param_names_list = ['x']
    dst_2d_ref.stypy_varargs_param_name = None
    dst_2d_ref.stypy_kwargs_param_name = 'kwargs'
    dst_2d_ref.stypy_call_defaults = defaults
    dst_2d_ref.stypy_call_varargs = varargs
    dst_2d_ref.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dst_2d_ref', ['x'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dst_2d_ref', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dst_2d_ref(...)' code ##################

    str_27070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'str', ' used as a reference in testing dst2. ')
    
    # Assigning a Call to a Name (line 75):
    
    # Assigning a Call to a Name (line 75):
    
    # Call to array(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'x' (line 75)
    x_27073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'x', False)
    # Processing the call keyword arguments (line 75)
    # Getting the type of 'True' (line 75)
    True_27074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'True', False)
    keyword_27075 = True_27074
    kwargs_27076 = {'copy': keyword_27075}
    # Getting the type of 'np' (line 75)
    np_27071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 75)
    array_27072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), np_27071, 'array')
    # Calling array(args, kwargs) (line 75)
    array_call_result_27077 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), array_27072, *[x_27073], **kwargs_27076)
    
    # Assigning a type to the variable 'x' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'x', array_call_result_27077)
    
    
    # Call to range(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Obtaining the type of the subscript
    int_27079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 29), 'int')
    # Getting the type of 'x' (line 76)
    x_27080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 76)
    shape_27081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 21), x_27080, 'shape')
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___27082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 21), shape_27081, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_27083 = invoke(stypy.reporting.localization.Localization(__file__, 76, 21), getitem___27082, int_27079)
    
    # Processing the call keyword arguments (line 76)
    kwargs_27084 = {}
    # Getting the type of 'range' (line 76)
    range_27078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'range', False)
    # Calling range(args, kwargs) (line 76)
    range_call_result_27085 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), range_27078, *[subscript_call_result_27083], **kwargs_27084)
    
    # Testing the type of a for loop iterable (line 76)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 76, 4), range_call_result_27085)
    # Getting the type of the for loop variable (line 76)
    for_loop_var_27086 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 76, 4), range_call_result_27085)
    # Assigning a type to the variable 'row' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'row', for_loop_var_27086)
    # SSA begins for a for statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 77):
    
    # Assigning a Call to a Subscript (line 77):
    
    # Call to dst(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 77)
    row_27088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'row', False)
    slice_27089 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 24), None, None, None)
    # Getting the type of 'x' (line 77)
    x_27090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 77)
    getitem___27091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 24), x_27090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
    subscript_call_result_27092 = invoke(stypy.reporting.localization.Localization(__file__, 77, 24), getitem___27091, (row_27088, slice_27089))
    
    # Processing the call keyword arguments (line 77)
    # Getting the type of 'kwargs' (line 77)
    kwargs_27093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'kwargs', False)
    kwargs_27094 = {'kwargs_27093': kwargs_27093}
    # Getting the type of 'dst' (line 77)
    dst_27087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'dst', False)
    # Calling dst(args, kwargs) (line 77)
    dst_call_result_27095 = invoke(stypy.reporting.localization.Localization(__file__, 77, 20), dst_27087, *[subscript_call_result_27092], **kwargs_27094)
    
    # Getting the type of 'x' (line 77)
    x_27096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'x')
    # Getting the type of 'row' (line 77)
    row_27097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 10), 'row')
    slice_27098 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 8), None, None, None)
    # Storing an element on a container (line 77)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 8), x_27096, ((row_27097, slice_27098), dst_call_result_27095))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Obtaining the type of the subscript
    int_27100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 29), 'int')
    # Getting the type of 'x' (line 78)
    x_27101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 78)
    shape_27102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), x_27101, 'shape')
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___27103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), shape_27102, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_27104 = invoke(stypy.reporting.localization.Localization(__file__, 78, 21), getitem___27103, int_27100)
    
    # Processing the call keyword arguments (line 78)
    kwargs_27105 = {}
    # Getting the type of 'range' (line 78)
    range_27099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'range', False)
    # Calling range(args, kwargs) (line 78)
    range_call_result_27106 = invoke(stypy.reporting.localization.Localization(__file__, 78, 15), range_27099, *[subscript_call_result_27104], **kwargs_27105)
    
    # Testing the type of a for loop iterable (line 78)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 4), range_call_result_27106)
    # Getting the type of the for loop variable (line 78)
    for_loop_var_27107 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 4), range_call_result_27106)
    # Assigning a type to the variable 'col' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'col', for_loop_var_27107)
    # SSA begins for a for statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 79):
    
    # Assigning a Call to a Subscript (line 79):
    
    # Call to dst(...): (line 79)
    # Processing the call arguments (line 79)
    
    # Obtaining the type of the subscript
    slice_27109 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 79, 24), None, None, None)
    # Getting the type of 'col' (line 79)
    col_27110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'col', False)
    # Getting the type of 'x' (line 79)
    x_27111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___27112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 24), x_27111, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_27113 = invoke(stypy.reporting.localization.Localization(__file__, 79, 24), getitem___27112, (slice_27109, col_27110))
    
    # Processing the call keyword arguments (line 79)
    # Getting the type of 'kwargs' (line 79)
    kwargs_27114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 37), 'kwargs', False)
    kwargs_27115 = {'kwargs_27114': kwargs_27114}
    # Getting the type of 'dst' (line 79)
    dst_27108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'dst', False)
    # Calling dst(args, kwargs) (line 79)
    dst_call_result_27116 = invoke(stypy.reporting.localization.Localization(__file__, 79, 20), dst_27108, *[subscript_call_result_27113], **kwargs_27115)
    
    # Getting the type of 'x' (line 79)
    x_27117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'x')
    slice_27118 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 79, 8), None, None, None)
    # Getting the type of 'col' (line 79)
    col_27119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'col')
    # Storing an element on a container (line 79)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 8), x_27117, ((slice_27118, col_27119), dst_call_result_27116))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 80)
    x_27120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type', x_27120)
    
    # ################# End of 'dst_2d_ref(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dst_2d_ref' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_27121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27121)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dst_2d_ref'
    return stypy_return_type_27121

# Assigning a type to the variable 'dst_2d_ref' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'dst_2d_ref', dst_2d_ref)

@norecursion
def idst_2d_ref(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idst_2d_ref'
    module_type_store = module_type_store.open_function_context('idst_2d_ref', 83, 0, False)
    
    # Passed parameters checking function
    idst_2d_ref.stypy_localization = localization
    idst_2d_ref.stypy_type_of_self = None
    idst_2d_ref.stypy_type_store = module_type_store
    idst_2d_ref.stypy_function_name = 'idst_2d_ref'
    idst_2d_ref.stypy_param_names_list = ['x']
    idst_2d_ref.stypy_varargs_param_name = None
    idst_2d_ref.stypy_kwargs_param_name = 'kwargs'
    idst_2d_ref.stypy_call_defaults = defaults
    idst_2d_ref.stypy_call_varargs = varargs
    idst_2d_ref.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idst_2d_ref', ['x'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idst_2d_ref', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idst_2d_ref(...)' code ##################

    str_27122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 4), 'str', ' used as a reference in testing idst2. ')
    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to array(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'x' (line 85)
    x_27125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'x', False)
    # Processing the call keyword arguments (line 85)
    # Getting the type of 'True' (line 85)
    True_27126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'True', False)
    keyword_27127 = True_27126
    kwargs_27128 = {'copy': keyword_27127}
    # Getting the type of 'np' (line 85)
    np_27123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 85)
    array_27124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), np_27123, 'array')
    # Calling array(args, kwargs) (line 85)
    array_call_result_27129 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), array_27124, *[x_27125], **kwargs_27128)
    
    # Assigning a type to the variable 'x' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'x', array_call_result_27129)
    
    
    # Call to range(...): (line 86)
    # Processing the call arguments (line 86)
    
    # Obtaining the type of the subscript
    int_27131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'int')
    # Getting the type of 'x' (line 86)
    x_27132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 86)
    shape_27133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 21), x_27132, 'shape')
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___27134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 21), shape_27133, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_27135 = invoke(stypy.reporting.localization.Localization(__file__, 86, 21), getitem___27134, int_27131)
    
    # Processing the call keyword arguments (line 86)
    kwargs_27136 = {}
    # Getting the type of 'range' (line 86)
    range_27130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'range', False)
    # Calling range(args, kwargs) (line 86)
    range_call_result_27137 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), range_27130, *[subscript_call_result_27135], **kwargs_27136)
    
    # Testing the type of a for loop iterable (line 86)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 86, 4), range_call_result_27137)
    # Getting the type of the for loop variable (line 86)
    for_loop_var_27138 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 86, 4), range_call_result_27137)
    # Assigning a type to the variable 'row' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'row', for_loop_var_27138)
    # SSA begins for a for statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 87):
    
    # Assigning a Call to a Subscript (line 87):
    
    # Call to idst(...): (line 87)
    # Processing the call arguments (line 87)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 87)
    row_27140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 27), 'row', False)
    slice_27141 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 87, 25), None, None, None)
    # Getting the type of 'x' (line 87)
    x_27142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 87)
    getitem___27143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 25), x_27142, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 87)
    subscript_call_result_27144 = invoke(stypy.reporting.localization.Localization(__file__, 87, 25), getitem___27143, (row_27140, slice_27141))
    
    # Processing the call keyword arguments (line 87)
    # Getting the type of 'kwargs' (line 87)
    kwargs_27145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 38), 'kwargs', False)
    kwargs_27146 = {'kwargs_27145': kwargs_27145}
    # Getting the type of 'idst' (line 87)
    idst_27139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'idst', False)
    # Calling idst(args, kwargs) (line 87)
    idst_call_result_27147 = invoke(stypy.reporting.localization.Localization(__file__, 87, 20), idst_27139, *[subscript_call_result_27144], **kwargs_27146)
    
    # Getting the type of 'x' (line 87)
    x_27148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'x')
    # Getting the type of 'row' (line 87)
    row_27149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 10), 'row')
    slice_27150 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 87, 8), None, None, None)
    # Storing an element on a container (line 87)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), x_27148, ((row_27149, slice_27150), idst_call_result_27147))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 88)
    # Processing the call arguments (line 88)
    
    # Obtaining the type of the subscript
    int_27152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 29), 'int')
    # Getting the type of 'x' (line 88)
    x_27153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 88)
    shape_27154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 21), x_27153, 'shape')
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___27155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 21), shape_27154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_27156 = invoke(stypy.reporting.localization.Localization(__file__, 88, 21), getitem___27155, int_27152)
    
    # Processing the call keyword arguments (line 88)
    kwargs_27157 = {}
    # Getting the type of 'range' (line 88)
    range_27151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'range', False)
    # Calling range(args, kwargs) (line 88)
    range_call_result_27158 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), range_27151, *[subscript_call_result_27156], **kwargs_27157)
    
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 4), range_call_result_27158)
    # Getting the type of the for loop variable (line 88)
    for_loop_var_27159 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 4), range_call_result_27158)
    # Assigning a type to the variable 'col' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'col', for_loop_var_27159)
    # SSA begins for a for statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 89):
    
    # Assigning a Call to a Subscript (line 89):
    
    # Call to idst(...): (line 89)
    # Processing the call arguments (line 89)
    
    # Obtaining the type of the subscript
    slice_27161 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 25), None, None, None)
    # Getting the type of 'col' (line 89)
    col_27162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'col', False)
    # Getting the type of 'x' (line 89)
    x_27163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___27164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 25), x_27163, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_27165 = invoke(stypy.reporting.localization.Localization(__file__, 89, 25), getitem___27164, (slice_27161, col_27162))
    
    # Processing the call keyword arguments (line 89)
    # Getting the type of 'kwargs' (line 89)
    kwargs_27166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 38), 'kwargs', False)
    kwargs_27167 = {'kwargs_27166': kwargs_27166}
    # Getting the type of 'idst' (line 89)
    idst_27160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'idst', False)
    # Calling idst(args, kwargs) (line 89)
    idst_call_result_27168 = invoke(stypy.reporting.localization.Localization(__file__, 89, 20), idst_27160, *[subscript_call_result_27165], **kwargs_27167)
    
    # Getting the type of 'x' (line 89)
    x_27169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'x')
    slice_27170 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 8), None, None, None)
    # Getting the type of 'col' (line 89)
    col_27171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'col')
    # Storing an element on a container (line 89)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), x_27169, ((slice_27170, col_27171), idst_call_result_27168))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 90)
    x_27172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type', x_27172)
    
    # ################# End of 'idst_2d_ref(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idst_2d_ref' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_27173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27173)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idst_2d_ref'
    return stypy_return_type_27173

# Assigning a type to the variable 'idst_2d_ref' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'idst_2d_ref', idst_2d_ref)
# Declaration of the 'TestComplex' class

class TestComplex(object, ):

    @norecursion
    def test_dct_complex64(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dct_complex64'
        module_type_store = module_type_store.open_function_context('test_dct_complex64', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplex.test_dct_complex64.__dict__.__setitem__('stypy_localization', localization)
        TestComplex.test_dct_complex64.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplex.test_dct_complex64.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplex.test_dct_complex64.__dict__.__setitem__('stypy_function_name', 'TestComplex.test_dct_complex64')
        TestComplex.test_dct_complex64.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplex.test_dct_complex64.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplex.test_dct_complex64.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplex.test_dct_complex64.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplex.test_dct_complex64.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplex.test_dct_complex64.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplex.test_dct_complex64.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplex.test_dct_complex64', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dct_complex64', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dct_complex64(...)' code ##################

        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to dct(...): (line 95)
        # Processing the call arguments (line 95)
        complex_27175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 16), 'complex')
        
        # Call to arange(...): (line 95)
        # Processing the call arguments (line 95)
        int_27178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 29), 'int')
        # Processing the call keyword arguments (line 95)
        # Getting the type of 'np' (line 95)
        np_27179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 38), 'np', False)
        # Obtaining the member 'complex64' of a type (line 95)
        complex64_27180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 38), np_27179, 'complex64')
        keyword_27181 = complex64_27180
        kwargs_27182 = {'dtype': keyword_27181}
        # Getting the type of 'np' (line 95)
        np_27176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'np', False)
        # Obtaining the member 'arange' of a type (line 95)
        arange_27177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 19), np_27176, 'arange')
        # Calling arange(args, kwargs) (line 95)
        arange_call_result_27183 = invoke(stypy.reporting.localization.Localization(__file__, 95, 19), arange_27177, *[int_27178], **kwargs_27182)
        
        # Applying the binary operator '*' (line 95)
        result_mul_27184 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 16), '*', complex_27175, arange_call_result_27183)
        
        # Processing the call keyword arguments (line 95)
        kwargs_27185 = {}
        # Getting the type of 'dct' (line 95)
        dct_27174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'dct', False)
        # Calling dct(args, kwargs) (line 95)
        dct_call_result_27186 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), dct_27174, *[result_mul_27184], **kwargs_27185)
        
        # Assigning a type to the variable 'y' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'y', dct_call_result_27186)
        
        # Assigning a BinOp to a Name (line 96):
        
        # Assigning a BinOp to a Name (line 96):
        complex_27187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'complex')
        
        # Call to dct(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to arange(...): (line 96)
        # Processing the call arguments (line 96)
        int_27191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 29), 'int')
        # Processing the call keyword arguments (line 96)
        kwargs_27192 = {}
        # Getting the type of 'np' (line 96)
        np_27189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'np', False)
        # Obtaining the member 'arange' of a type (line 96)
        arange_27190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 19), np_27189, 'arange')
        # Calling arange(args, kwargs) (line 96)
        arange_call_result_27193 = invoke(stypy.reporting.localization.Localization(__file__, 96, 19), arange_27190, *[int_27191], **kwargs_27192)
        
        # Processing the call keyword arguments (line 96)
        kwargs_27194 = {}
        # Getting the type of 'dct' (line 96)
        dct_27188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'dct', False)
        # Calling dct(args, kwargs) (line 96)
        dct_call_result_27195 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), dct_27188, *[arange_call_result_27193], **kwargs_27194)
        
        # Applying the binary operator '*' (line 96)
        result_mul_27196 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 12), '*', complex_27187, dct_call_result_27195)
        
        # Assigning a type to the variable 'x' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'x', result_mul_27196)
        
        # Call to assert_array_almost_equal(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'x' (line 97)
        x_27198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 34), 'x', False)
        # Getting the type of 'y' (line 97)
        y_27199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 37), 'y', False)
        # Processing the call keyword arguments (line 97)
        kwargs_27200 = {}
        # Getting the type of 'assert_array_almost_equal' (line 97)
        assert_array_almost_equal_27197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 97)
        assert_array_almost_equal_call_result_27201 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assert_array_almost_equal_27197, *[x_27198, y_27199], **kwargs_27200)
        
        
        # ################# End of 'test_dct_complex64(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dct_complex64' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_27202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27202)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dct_complex64'
        return stypy_return_type_27202


    @norecursion
    def test_dct_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dct_complex'
        module_type_store = module_type_store.open_function_context('test_dct_complex', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplex.test_dct_complex.__dict__.__setitem__('stypy_localization', localization)
        TestComplex.test_dct_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplex.test_dct_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplex.test_dct_complex.__dict__.__setitem__('stypy_function_name', 'TestComplex.test_dct_complex')
        TestComplex.test_dct_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplex.test_dct_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplex.test_dct_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplex.test_dct_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplex.test_dct_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplex.test_dct_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplex.test_dct_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplex.test_dct_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dct_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dct_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to dct(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to arange(...): (line 100)
        # Processing the call arguments (line 100)
        int_27206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 26), 'int')
        # Processing the call keyword arguments (line 100)
        kwargs_27207 = {}
        # Getting the type of 'np' (line 100)
        np_27204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'np', False)
        # Obtaining the member 'arange' of a type (line 100)
        arange_27205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 16), np_27204, 'arange')
        # Calling arange(args, kwargs) (line 100)
        arange_call_result_27208 = invoke(stypy.reporting.localization.Localization(__file__, 100, 16), arange_27205, *[int_27206], **kwargs_27207)
        
        complex_27209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 29), 'complex')
        # Applying the binary operator '*' (line 100)
        result_mul_27210 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 16), '*', arange_call_result_27208, complex_27209)
        
        # Processing the call keyword arguments (line 100)
        kwargs_27211 = {}
        # Getting the type of 'dct' (line 100)
        dct_27203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'dct', False)
        # Calling dct(args, kwargs) (line 100)
        dct_call_result_27212 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), dct_27203, *[result_mul_27210], **kwargs_27211)
        
        # Assigning a type to the variable 'y' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'y', dct_call_result_27212)
        
        # Assigning a BinOp to a Name (line 101):
        
        # Assigning a BinOp to a Name (line 101):
        complex_27213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'complex')
        
        # Call to dct(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to arange(...): (line 101)
        # Processing the call arguments (line 101)
        int_27217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'int')
        # Processing the call keyword arguments (line 101)
        kwargs_27218 = {}
        # Getting the type of 'np' (line 101)
        np_27215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'np', False)
        # Obtaining the member 'arange' of a type (line 101)
        arange_27216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 19), np_27215, 'arange')
        # Calling arange(args, kwargs) (line 101)
        arange_call_result_27219 = invoke(stypy.reporting.localization.Localization(__file__, 101, 19), arange_27216, *[int_27217], **kwargs_27218)
        
        # Processing the call keyword arguments (line 101)
        kwargs_27220 = {}
        # Getting the type of 'dct' (line 101)
        dct_27214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'dct', False)
        # Calling dct(args, kwargs) (line 101)
        dct_call_result_27221 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), dct_27214, *[arange_call_result_27219], **kwargs_27220)
        
        # Applying the binary operator '*' (line 101)
        result_mul_27222 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 12), '*', complex_27213, dct_call_result_27221)
        
        # Assigning a type to the variable 'x' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'x', result_mul_27222)
        
        # Call to assert_array_almost_equal(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'x' (line 102)
        x_27224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'x', False)
        # Getting the type of 'y' (line 102)
        y_27225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 37), 'y', False)
        # Processing the call keyword arguments (line 102)
        kwargs_27226 = {}
        # Getting the type of 'assert_array_almost_equal' (line 102)
        assert_array_almost_equal_27223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 102)
        assert_array_almost_equal_call_result_27227 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assert_array_almost_equal_27223, *[x_27224, y_27225], **kwargs_27226)
        
        
        # ################# End of 'test_dct_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dct_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_27228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dct_complex'
        return stypy_return_type_27228


    @norecursion
    def test_idct_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_idct_complex'
        module_type_store = module_type_store.open_function_context('test_idct_complex', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplex.test_idct_complex.__dict__.__setitem__('stypy_localization', localization)
        TestComplex.test_idct_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplex.test_idct_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplex.test_idct_complex.__dict__.__setitem__('stypy_function_name', 'TestComplex.test_idct_complex')
        TestComplex.test_idct_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplex.test_idct_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplex.test_idct_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplex.test_idct_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplex.test_idct_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplex.test_idct_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplex.test_idct_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplex.test_idct_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_idct_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_idct_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to idct(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to arange(...): (line 105)
        # Processing the call arguments (line 105)
        int_27232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 27), 'int')
        # Processing the call keyword arguments (line 105)
        kwargs_27233 = {}
        # Getting the type of 'np' (line 105)
        np_27230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 105)
        arange_27231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 17), np_27230, 'arange')
        # Calling arange(args, kwargs) (line 105)
        arange_call_result_27234 = invoke(stypy.reporting.localization.Localization(__file__, 105, 17), arange_27231, *[int_27232], **kwargs_27233)
        
        complex_27235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 30), 'complex')
        # Applying the binary operator '*' (line 105)
        result_mul_27236 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 17), '*', arange_call_result_27234, complex_27235)
        
        # Processing the call keyword arguments (line 105)
        kwargs_27237 = {}
        # Getting the type of 'idct' (line 105)
        idct_27229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'idct', False)
        # Calling idct(args, kwargs) (line 105)
        idct_call_result_27238 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), idct_27229, *[result_mul_27236], **kwargs_27237)
        
        # Assigning a type to the variable 'y' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'y', idct_call_result_27238)
        
        # Assigning a BinOp to a Name (line 106):
        
        # Assigning a BinOp to a Name (line 106):
        complex_27239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 12), 'complex')
        
        # Call to idct(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Call to arange(...): (line 106)
        # Processing the call arguments (line 106)
        int_27243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 30), 'int')
        # Processing the call keyword arguments (line 106)
        kwargs_27244 = {}
        # Getting the type of 'np' (line 106)
        np_27241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'np', False)
        # Obtaining the member 'arange' of a type (line 106)
        arange_27242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), np_27241, 'arange')
        # Calling arange(args, kwargs) (line 106)
        arange_call_result_27245 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), arange_27242, *[int_27243], **kwargs_27244)
        
        # Processing the call keyword arguments (line 106)
        kwargs_27246 = {}
        # Getting the type of 'idct' (line 106)
        idct_27240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'idct', False)
        # Calling idct(args, kwargs) (line 106)
        idct_call_result_27247 = invoke(stypy.reporting.localization.Localization(__file__, 106, 15), idct_27240, *[arange_call_result_27245], **kwargs_27246)
        
        # Applying the binary operator '*' (line 106)
        result_mul_27248 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 12), '*', complex_27239, idct_call_result_27247)
        
        # Assigning a type to the variable 'x' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'x', result_mul_27248)
        
        # Call to assert_array_almost_equal(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'x' (line 107)
        x_27250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'x', False)
        # Getting the type of 'y' (line 107)
        y_27251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 37), 'y', False)
        # Processing the call keyword arguments (line 107)
        kwargs_27252 = {}
        # Getting the type of 'assert_array_almost_equal' (line 107)
        assert_array_almost_equal_27249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 107)
        assert_array_almost_equal_call_result_27253 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), assert_array_almost_equal_27249, *[x_27250, y_27251], **kwargs_27252)
        
        
        # ################# End of 'test_idct_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_idct_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_27254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27254)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_idct_complex'
        return stypy_return_type_27254


    @norecursion
    def test_dst_complex64(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dst_complex64'
        module_type_store = module_type_store.open_function_context('test_dst_complex64', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplex.test_dst_complex64.__dict__.__setitem__('stypy_localization', localization)
        TestComplex.test_dst_complex64.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplex.test_dst_complex64.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplex.test_dst_complex64.__dict__.__setitem__('stypy_function_name', 'TestComplex.test_dst_complex64')
        TestComplex.test_dst_complex64.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplex.test_dst_complex64.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplex.test_dst_complex64.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplex.test_dst_complex64.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplex.test_dst_complex64.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplex.test_dst_complex64.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplex.test_dst_complex64.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplex.test_dst_complex64', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dst_complex64', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dst_complex64(...)' code ##################

        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to dst(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to arange(...): (line 110)
        # Processing the call arguments (line 110)
        int_27258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 26), 'int')
        # Processing the call keyword arguments (line 110)
        # Getting the type of 'np' (line 110)
        np_27259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 35), 'np', False)
        # Obtaining the member 'complex64' of a type (line 110)
        complex64_27260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 35), np_27259, 'complex64')
        keyword_27261 = complex64_27260
        kwargs_27262 = {'dtype': keyword_27261}
        # Getting the type of 'np' (line 110)
        np_27256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'np', False)
        # Obtaining the member 'arange' of a type (line 110)
        arange_27257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), np_27256, 'arange')
        # Calling arange(args, kwargs) (line 110)
        arange_call_result_27263 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), arange_27257, *[int_27258], **kwargs_27262)
        
        complex_27264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 49), 'complex')
        # Applying the binary operator '*' (line 110)
        result_mul_27265 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 16), '*', arange_call_result_27263, complex_27264)
        
        # Processing the call keyword arguments (line 110)
        kwargs_27266 = {}
        # Getting the type of 'dst' (line 110)
        dst_27255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'dst', False)
        # Calling dst(args, kwargs) (line 110)
        dst_call_result_27267 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), dst_27255, *[result_mul_27265], **kwargs_27266)
        
        # Assigning a type to the variable 'y' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'y', dst_call_result_27267)
        
        # Assigning a BinOp to a Name (line 111):
        
        # Assigning a BinOp to a Name (line 111):
        complex_27268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'complex')
        
        # Call to dst(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to arange(...): (line 111)
        # Processing the call arguments (line 111)
        int_27272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 29), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_27273 = {}
        # Getting the type of 'np' (line 111)
        np_27270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'np', False)
        # Obtaining the member 'arange' of a type (line 111)
        arange_27271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 19), np_27270, 'arange')
        # Calling arange(args, kwargs) (line 111)
        arange_call_result_27274 = invoke(stypy.reporting.localization.Localization(__file__, 111, 19), arange_27271, *[int_27272], **kwargs_27273)
        
        # Processing the call keyword arguments (line 111)
        kwargs_27275 = {}
        # Getting the type of 'dst' (line 111)
        dst_27269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'dst', False)
        # Calling dst(args, kwargs) (line 111)
        dst_call_result_27276 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), dst_27269, *[arange_call_result_27274], **kwargs_27275)
        
        # Applying the binary operator '*' (line 111)
        result_mul_27277 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 12), '*', complex_27268, dst_call_result_27276)
        
        # Assigning a type to the variable 'x' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'x', result_mul_27277)
        
        # Call to assert_array_almost_equal(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'x' (line 112)
        x_27279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 34), 'x', False)
        # Getting the type of 'y' (line 112)
        y_27280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 37), 'y', False)
        # Processing the call keyword arguments (line 112)
        kwargs_27281 = {}
        # Getting the type of 'assert_array_almost_equal' (line 112)
        assert_array_almost_equal_27278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 112)
        assert_array_almost_equal_call_result_27282 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), assert_array_almost_equal_27278, *[x_27279, y_27280], **kwargs_27281)
        
        
        # ################# End of 'test_dst_complex64(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dst_complex64' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_27283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27283)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dst_complex64'
        return stypy_return_type_27283


    @norecursion
    def test_dst_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dst_complex'
        module_type_store = module_type_store.open_function_context('test_dst_complex', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplex.test_dst_complex.__dict__.__setitem__('stypy_localization', localization)
        TestComplex.test_dst_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplex.test_dst_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplex.test_dst_complex.__dict__.__setitem__('stypy_function_name', 'TestComplex.test_dst_complex')
        TestComplex.test_dst_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplex.test_dst_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplex.test_dst_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplex.test_dst_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplex.test_dst_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplex.test_dst_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplex.test_dst_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplex.test_dst_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dst_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dst_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to dst(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Call to arange(...): (line 115)
        # Processing the call arguments (line 115)
        int_27287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 26), 'int')
        # Processing the call keyword arguments (line 115)
        kwargs_27288 = {}
        # Getting the type of 'np' (line 115)
        np_27285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'np', False)
        # Obtaining the member 'arange' of a type (line 115)
        arange_27286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), np_27285, 'arange')
        # Calling arange(args, kwargs) (line 115)
        arange_call_result_27289 = invoke(stypy.reporting.localization.Localization(__file__, 115, 16), arange_27286, *[int_27287], **kwargs_27288)
        
        complex_27290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 29), 'complex')
        # Applying the binary operator '*' (line 115)
        result_mul_27291 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 16), '*', arange_call_result_27289, complex_27290)
        
        # Processing the call keyword arguments (line 115)
        kwargs_27292 = {}
        # Getting the type of 'dst' (line 115)
        dst_27284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'dst', False)
        # Calling dst(args, kwargs) (line 115)
        dst_call_result_27293 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), dst_27284, *[result_mul_27291], **kwargs_27292)
        
        # Assigning a type to the variable 'y' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'y', dst_call_result_27293)
        
        # Assigning a BinOp to a Name (line 116):
        
        # Assigning a BinOp to a Name (line 116):
        complex_27294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 12), 'complex')
        
        # Call to dst(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Call to arange(...): (line 116)
        # Processing the call arguments (line 116)
        int_27298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 29), 'int')
        # Processing the call keyword arguments (line 116)
        kwargs_27299 = {}
        # Getting the type of 'np' (line 116)
        np_27296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'np', False)
        # Obtaining the member 'arange' of a type (line 116)
        arange_27297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 19), np_27296, 'arange')
        # Calling arange(args, kwargs) (line 116)
        arange_call_result_27300 = invoke(stypy.reporting.localization.Localization(__file__, 116, 19), arange_27297, *[int_27298], **kwargs_27299)
        
        # Processing the call keyword arguments (line 116)
        kwargs_27301 = {}
        # Getting the type of 'dst' (line 116)
        dst_27295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'dst', False)
        # Calling dst(args, kwargs) (line 116)
        dst_call_result_27302 = invoke(stypy.reporting.localization.Localization(__file__, 116, 15), dst_27295, *[arange_call_result_27300], **kwargs_27301)
        
        # Applying the binary operator '*' (line 116)
        result_mul_27303 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 12), '*', complex_27294, dst_call_result_27302)
        
        # Assigning a type to the variable 'x' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'x', result_mul_27303)
        
        # Call to assert_array_almost_equal(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'x' (line 117)
        x_27305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 34), 'x', False)
        # Getting the type of 'y' (line 117)
        y_27306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 37), 'y', False)
        # Processing the call keyword arguments (line 117)
        kwargs_27307 = {}
        # Getting the type of 'assert_array_almost_equal' (line 117)
        assert_array_almost_equal_27304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 117)
        assert_array_almost_equal_call_result_27308 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), assert_array_almost_equal_27304, *[x_27305, y_27306], **kwargs_27307)
        
        
        # ################# End of 'test_dst_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dst_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_27309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27309)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dst_complex'
        return stypy_return_type_27309


    @norecursion
    def test_idst_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_idst_complex'
        module_type_store = module_type_store.open_function_context('test_idst_complex', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestComplex.test_idst_complex.__dict__.__setitem__('stypy_localization', localization)
        TestComplex.test_idst_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestComplex.test_idst_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestComplex.test_idst_complex.__dict__.__setitem__('stypy_function_name', 'TestComplex.test_idst_complex')
        TestComplex.test_idst_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestComplex.test_idst_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestComplex.test_idst_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestComplex.test_idst_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestComplex.test_idst_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestComplex.test_idst_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestComplex.test_idst_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplex.test_idst_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_idst_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_idst_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to idst(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to arange(...): (line 120)
        # Processing the call arguments (line 120)
        int_27313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 27), 'int')
        # Processing the call keyword arguments (line 120)
        kwargs_27314 = {}
        # Getting the type of 'np' (line 120)
        np_27311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 120)
        arange_27312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), np_27311, 'arange')
        # Calling arange(args, kwargs) (line 120)
        arange_call_result_27315 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), arange_27312, *[int_27313], **kwargs_27314)
        
        complex_27316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 30), 'complex')
        # Applying the binary operator '*' (line 120)
        result_mul_27317 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 17), '*', arange_call_result_27315, complex_27316)
        
        # Processing the call keyword arguments (line 120)
        kwargs_27318 = {}
        # Getting the type of 'idst' (line 120)
        idst_27310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'idst', False)
        # Calling idst(args, kwargs) (line 120)
        idst_call_result_27319 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), idst_27310, *[result_mul_27317], **kwargs_27318)
        
        # Assigning a type to the variable 'y' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'y', idst_call_result_27319)
        
        # Assigning a BinOp to a Name (line 121):
        
        # Assigning a BinOp to a Name (line 121):
        complex_27320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 12), 'complex')
        
        # Call to idst(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Call to arange(...): (line 121)
        # Processing the call arguments (line 121)
        int_27324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 30), 'int')
        # Processing the call keyword arguments (line 121)
        kwargs_27325 = {}
        # Getting the type of 'np' (line 121)
        np_27322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'np', False)
        # Obtaining the member 'arange' of a type (line 121)
        arange_27323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 20), np_27322, 'arange')
        # Calling arange(args, kwargs) (line 121)
        arange_call_result_27326 = invoke(stypy.reporting.localization.Localization(__file__, 121, 20), arange_27323, *[int_27324], **kwargs_27325)
        
        # Processing the call keyword arguments (line 121)
        kwargs_27327 = {}
        # Getting the type of 'idst' (line 121)
        idst_27321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'idst', False)
        # Calling idst(args, kwargs) (line 121)
        idst_call_result_27328 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), idst_27321, *[arange_call_result_27326], **kwargs_27327)
        
        # Applying the binary operator '*' (line 121)
        result_mul_27329 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 12), '*', complex_27320, idst_call_result_27328)
        
        # Assigning a type to the variable 'x' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'x', result_mul_27329)
        
        # Call to assert_array_almost_equal(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'x' (line 122)
        x_27331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 34), 'x', False)
        # Getting the type of 'y' (line 122)
        y_27332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'y', False)
        # Processing the call keyword arguments (line 122)
        kwargs_27333 = {}
        # Getting the type of 'assert_array_almost_equal' (line 122)
        assert_array_almost_equal_27330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 122)
        assert_array_almost_equal_call_result_27334 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), assert_array_almost_equal_27330, *[x_27331, y_27332], **kwargs_27333)
        
        
        # ################# End of 'test_idst_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_idst_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_27335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27335)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_idst_complex'
        return stypy_return_type_27335


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 93, 0, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestComplex.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestComplex' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'TestComplex', TestComplex)
# Declaration of the '_TestDCTBase' class

class _TestDCTBase(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _TestDCTBase.setup_method.__dict__.__setitem__('stypy_localization', localization)
        _TestDCTBase.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _TestDCTBase.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        _TestDCTBase.setup_method.__dict__.__setitem__('stypy_function_name', '_TestDCTBase.setup_method')
        _TestDCTBase.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        _TestDCTBase.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        _TestDCTBase.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _TestDCTBase.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        _TestDCTBase.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        _TestDCTBase.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _TestDCTBase.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestDCTBase.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 127):
        
        # Assigning a Name to a Attribute (line 127):
        # Getting the type of 'None' (line 127)
        None_27336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'None')
        # Getting the type of 'self' (line 127)
        self_27337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 127)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_27337, 'rdt', None_27336)
        
        # Assigning a Num to a Attribute (line 128):
        
        # Assigning a Num to a Attribute (line 128):
        int_27338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 19), 'int')
        # Getting the type of 'self' (line 128)
        self_27339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_27339, 'dec', int_27338)
        
        # Assigning a Name to a Attribute (line 129):
        
        # Assigning a Name to a Attribute (line 129):
        # Getting the type of 'None' (line 129)
        None_27340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'None')
        # Getting the type of 'self' (line 129)
        self_27341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member 'type' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_27341, 'type', None_27340)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_27342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27342)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27342


    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _TestDCTBase.test_definition.__dict__.__setitem__('stypy_localization', localization)
        _TestDCTBase.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _TestDCTBase.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        _TestDCTBase.test_definition.__dict__.__setitem__('stypy_function_name', '_TestDCTBase.test_definition')
        _TestDCTBase.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        _TestDCTBase.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        _TestDCTBase.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _TestDCTBase.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        _TestDCTBase.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        _TestDCTBase.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _TestDCTBase.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestDCTBase.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        # Getting the type of 'FFTWDATA_SIZES' (line 132)
        FFTWDATA_SIZES_27343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'FFTWDATA_SIZES')
        # Testing the type of a for loop iterable (line 132)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 132, 8), FFTWDATA_SIZES_27343)
        # Getting the type of the for loop variable (line 132)
        for_loop_var_27344 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 132, 8), FFTWDATA_SIZES_27343)
        # Assigning a type to the variable 'i' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'i', for_loop_var_27344)
        # SSA begins for a for statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 133):
        
        # Assigning a Subscript to a Name (line 133):
        
        # Obtaining the type of the subscript
        int_27345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 12), 'int')
        
        # Call to fftw_dct_ref(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'self' (line 133)
        self_27347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 37), 'self', False)
        # Obtaining the member 'type' of a type (line 133)
        type_27348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 37), self_27347, 'type')
        # Getting the type of 'i' (line 133)
        i_27349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 48), 'i', False)
        # Getting the type of 'self' (line 133)
        self_27350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 51), 'self', False)
        # Obtaining the member 'rdt' of a type (line 133)
        rdt_27351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 51), self_27350, 'rdt')
        # Processing the call keyword arguments (line 133)
        kwargs_27352 = {}
        # Getting the type of 'fftw_dct_ref' (line 133)
        fftw_dct_ref_27346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'fftw_dct_ref', False)
        # Calling fftw_dct_ref(args, kwargs) (line 133)
        fftw_dct_ref_call_result_27353 = invoke(stypy.reporting.localization.Localization(__file__, 133, 24), fftw_dct_ref_27346, *[type_27348, i_27349, rdt_27351], **kwargs_27352)
        
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___27354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), fftw_dct_ref_call_result_27353, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_27355 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), getitem___27354, int_27345)
        
        # Assigning a type to the variable 'tuple_var_assignment_26776' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'tuple_var_assignment_26776', subscript_call_result_27355)
        
        # Assigning a Subscript to a Name (line 133):
        
        # Obtaining the type of the subscript
        int_27356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 12), 'int')
        
        # Call to fftw_dct_ref(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'self' (line 133)
        self_27358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 37), 'self', False)
        # Obtaining the member 'type' of a type (line 133)
        type_27359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 37), self_27358, 'type')
        # Getting the type of 'i' (line 133)
        i_27360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 48), 'i', False)
        # Getting the type of 'self' (line 133)
        self_27361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 51), 'self', False)
        # Obtaining the member 'rdt' of a type (line 133)
        rdt_27362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 51), self_27361, 'rdt')
        # Processing the call keyword arguments (line 133)
        kwargs_27363 = {}
        # Getting the type of 'fftw_dct_ref' (line 133)
        fftw_dct_ref_27357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'fftw_dct_ref', False)
        # Calling fftw_dct_ref(args, kwargs) (line 133)
        fftw_dct_ref_call_result_27364 = invoke(stypy.reporting.localization.Localization(__file__, 133, 24), fftw_dct_ref_27357, *[type_27359, i_27360, rdt_27362], **kwargs_27363)
        
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___27365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), fftw_dct_ref_call_result_27364, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_27366 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), getitem___27365, int_27356)
        
        # Assigning a type to the variable 'tuple_var_assignment_26777' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'tuple_var_assignment_26777', subscript_call_result_27366)
        
        # Assigning a Subscript to a Name (line 133):
        
        # Obtaining the type of the subscript
        int_27367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 12), 'int')
        
        # Call to fftw_dct_ref(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'self' (line 133)
        self_27369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 37), 'self', False)
        # Obtaining the member 'type' of a type (line 133)
        type_27370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 37), self_27369, 'type')
        # Getting the type of 'i' (line 133)
        i_27371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 48), 'i', False)
        # Getting the type of 'self' (line 133)
        self_27372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 51), 'self', False)
        # Obtaining the member 'rdt' of a type (line 133)
        rdt_27373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 51), self_27372, 'rdt')
        # Processing the call keyword arguments (line 133)
        kwargs_27374 = {}
        # Getting the type of 'fftw_dct_ref' (line 133)
        fftw_dct_ref_27368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'fftw_dct_ref', False)
        # Calling fftw_dct_ref(args, kwargs) (line 133)
        fftw_dct_ref_call_result_27375 = invoke(stypy.reporting.localization.Localization(__file__, 133, 24), fftw_dct_ref_27368, *[type_27370, i_27371, rdt_27373], **kwargs_27374)
        
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___27376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), fftw_dct_ref_call_result_27375, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_27377 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), getitem___27376, int_27367)
        
        # Assigning a type to the variable 'tuple_var_assignment_26778' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'tuple_var_assignment_26778', subscript_call_result_27377)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'tuple_var_assignment_26776' (line 133)
        tuple_var_assignment_26776_27378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'tuple_var_assignment_26776')
        # Assigning a type to the variable 'x' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'x', tuple_var_assignment_26776_27378)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'tuple_var_assignment_26777' (line 133)
        tuple_var_assignment_26777_27379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'tuple_var_assignment_26777')
        # Assigning a type to the variable 'yr' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'yr', tuple_var_assignment_26777_27379)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'tuple_var_assignment_26778' (line 133)
        tuple_var_assignment_26778_27380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'tuple_var_assignment_26778')
        # Assigning a type to the variable 'dt' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'dt', tuple_var_assignment_26778_27380)
        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to dct(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'x' (line 134)
        x_27382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'x', False)
        # Processing the call keyword arguments (line 134)
        # Getting the type of 'self' (line 134)
        self_27383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 28), 'self', False)
        # Obtaining the member 'type' of a type (line 134)
        type_27384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 28), self_27383, 'type')
        keyword_27385 = type_27384
        kwargs_27386 = {'type': keyword_27385}
        # Getting the type of 'dct' (line 134)
        dct_27381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'dct', False)
        # Calling dct(args, kwargs) (line 134)
        dct_call_result_27387 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), dct_27381, *[x_27382], **kwargs_27386)
        
        # Assigning a type to the variable 'y' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'y', dct_call_result_27387)
        
        # Call to assert_equal(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'y' (line 135)
        y_27389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 25), 'y', False)
        # Obtaining the member 'dtype' of a type (line 135)
        dtype_27390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 25), y_27389, 'dtype')
        # Getting the type of 'dt' (line 135)
        dt_27391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 34), 'dt', False)
        # Processing the call keyword arguments (line 135)
        kwargs_27392 = {}
        # Getting the type of 'assert_equal' (line 135)
        assert_equal_27388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 135)
        assert_equal_call_result_27393 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), assert_equal_27388, *[dtype_27390, dt_27391], **kwargs_27392)
        
        
        # Call to assert_array_almost_equal(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'y' (line 140)
        y_27395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 38), 'y', False)
        
        # Call to max(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'y' (line 140)
        y_27398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 49), 'y', False)
        # Processing the call keyword arguments (line 140)
        kwargs_27399 = {}
        # Getting the type of 'np' (line 140)
        np_27396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 42), 'np', False)
        # Obtaining the member 'max' of a type (line 140)
        max_27397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 42), np_27396, 'max')
        # Calling max(args, kwargs) (line 140)
        max_call_result_27400 = invoke(stypy.reporting.localization.Localization(__file__, 140, 42), max_27397, *[y_27398], **kwargs_27399)
        
        # Applying the binary operator 'div' (line 140)
        result_div_27401 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 38), 'div', y_27395, max_call_result_27400)
        
        # Getting the type of 'yr' (line 140)
        yr_27402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 53), 'yr', False)
        
        # Call to max(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'y' (line 140)
        y_27405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 65), 'y', False)
        # Processing the call keyword arguments (line 140)
        kwargs_27406 = {}
        # Getting the type of 'np' (line 140)
        np_27403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 58), 'np', False)
        # Obtaining the member 'max' of a type (line 140)
        max_27404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 58), np_27403, 'max')
        # Calling max(args, kwargs) (line 140)
        max_call_result_27407 = invoke(stypy.reporting.localization.Localization(__file__, 140, 58), max_27404, *[y_27405], **kwargs_27406)
        
        # Applying the binary operator 'div' (line 140)
        result_div_27408 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 53), 'div', yr_27402, max_call_result_27407)
        
        # Processing the call keyword arguments (line 140)
        # Getting the type of 'self' (line 140)
        self_27409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 77), 'self', False)
        # Obtaining the member 'dec' of a type (line 140)
        dec_27410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 77), self_27409, 'dec')
        keyword_27411 = dec_27410
        str_27412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 28), 'str', 'Size %d failed')
        # Getting the type of 'i' (line 141)
        i_27413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 47), 'i', False)
        # Applying the binary operator '%' (line 141)
        result_mod_27414 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 28), '%', str_27412, i_27413)
        
        keyword_27415 = result_mod_27414
        kwargs_27416 = {'decimal': keyword_27411, 'err_msg': keyword_27415}
        # Getting the type of 'assert_array_almost_equal' (line 140)
        assert_array_almost_equal_27394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 140)
        assert_array_almost_equal_call_result_27417 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), assert_array_almost_equal_27394, *[result_div_27401, result_div_27408], **kwargs_27416)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_27418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27418)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_27418


    @norecursion
    def test_axis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_axis'
        module_type_store = module_type_store.open_function_context('test_axis', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _TestDCTBase.test_axis.__dict__.__setitem__('stypy_localization', localization)
        _TestDCTBase.test_axis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _TestDCTBase.test_axis.__dict__.__setitem__('stypy_type_store', module_type_store)
        _TestDCTBase.test_axis.__dict__.__setitem__('stypy_function_name', '_TestDCTBase.test_axis')
        _TestDCTBase.test_axis.__dict__.__setitem__('stypy_param_names_list', [])
        _TestDCTBase.test_axis.__dict__.__setitem__('stypy_varargs_param_name', None)
        _TestDCTBase.test_axis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _TestDCTBase.test_axis.__dict__.__setitem__('stypy_call_defaults', defaults)
        _TestDCTBase.test_axis.__dict__.__setitem__('stypy_call_varargs', varargs)
        _TestDCTBase.test_axis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _TestDCTBase.test_axis.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestDCTBase.test_axis', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_axis', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_axis(...)' code ##################

        
        # Assigning a Num to a Name (line 144):
        
        # Assigning a Num to a Name (line 144):
        int_27419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 13), 'int')
        # Assigning a type to the variable 'nt' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'nt', int_27419)
        
        
        # Obtaining an instance of the builtin type 'list' (line 145)
        list_27420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 145)
        # Adding element type (line 145)
        int_27421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 17), list_27420, int_27421)
        # Adding element type (line 145)
        int_27422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 17), list_27420, int_27422)
        # Adding element type (line 145)
        int_27423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 17), list_27420, int_27423)
        # Adding element type (line 145)
        int_27424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 17), list_27420, int_27424)
        # Adding element type (line 145)
        int_27425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 17), list_27420, int_27425)
        # Adding element type (line 145)
        int_27426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 17), list_27420, int_27426)
        
        # Testing the type of a for loop iterable (line 145)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 145, 8), list_27420)
        # Getting the type of the for loop variable (line 145)
        for_loop_var_27427 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 145, 8), list_27420)
        # Assigning a type to the variable 'i' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'i', for_loop_var_27427)
        # SSA begins for a for statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Call to randn(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'nt' (line 146)
        nt_27431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 32), 'nt', False)
        # Getting the type of 'i' (line 146)
        i_27432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 36), 'i', False)
        # Processing the call keyword arguments (line 146)
        kwargs_27433 = {}
        # Getting the type of 'np' (line 146)
        np_27428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'np', False)
        # Obtaining the member 'random' of a type (line 146)
        random_27429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), np_27428, 'random')
        # Obtaining the member 'randn' of a type (line 146)
        randn_27430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), random_27429, 'randn')
        # Calling randn(args, kwargs) (line 146)
        randn_call_result_27434 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), randn_27430, *[nt_27431, i_27432], **kwargs_27433)
        
        # Assigning a type to the variable 'x' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'x', randn_call_result_27434)
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to dct(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'x' (line 147)
        x_27436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'x', False)
        # Processing the call keyword arguments (line 147)
        # Getting the type of 'self' (line 147)
        self_27437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 28), 'self', False)
        # Obtaining the member 'type' of a type (line 147)
        type_27438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 28), self_27437, 'type')
        keyword_27439 = type_27438
        kwargs_27440 = {'type': keyword_27439}
        # Getting the type of 'dct' (line 147)
        dct_27435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'dct', False)
        # Calling dct(args, kwargs) (line 147)
        dct_call_result_27441 = invoke(stypy.reporting.localization.Localization(__file__, 147, 16), dct_27435, *[x_27436], **kwargs_27440)
        
        # Assigning a type to the variable 'y' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'y', dct_call_result_27441)
        
        
        # Call to range(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'nt' (line 148)
        nt_27443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'nt', False)
        # Processing the call keyword arguments (line 148)
        kwargs_27444 = {}
        # Getting the type of 'range' (line 148)
        range_27442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'range', False)
        # Calling range(args, kwargs) (line 148)
        range_call_result_27445 = invoke(stypy.reporting.localization.Localization(__file__, 148, 21), range_27442, *[nt_27443], **kwargs_27444)
        
        # Testing the type of a for loop iterable (line 148)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 12), range_call_result_27445)
        # Getting the type of the for loop variable (line 148)
        for_loop_var_27446 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 12), range_call_result_27445)
        # Assigning a type to the variable 'j' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'j', for_loop_var_27446)
        # SSA begins for a for statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_array_almost_equal(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 149)
        j_27448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 44), 'j', False)
        # Getting the type of 'y' (line 149)
        y_27449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 42), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___27450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 42), y_27449, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_27451 = invoke(stypy.reporting.localization.Localization(__file__, 149, 42), getitem___27450, j_27448)
        
        
        # Call to dct(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 149)
        j_27453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 54), 'j', False)
        # Getting the type of 'x' (line 149)
        x_27454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 52), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___27455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 52), x_27454, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_27456 = invoke(stypy.reporting.localization.Localization(__file__, 149, 52), getitem___27455, j_27453)
        
        # Processing the call keyword arguments (line 149)
        # Getting the type of 'self' (line 149)
        self_27457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 63), 'self', False)
        # Obtaining the member 'type' of a type (line 149)
        type_27458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 63), self_27457, 'type')
        keyword_27459 = type_27458
        kwargs_27460 = {'type': keyword_27459}
        # Getting the type of 'dct' (line 149)
        dct_27452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 48), 'dct', False)
        # Calling dct(args, kwargs) (line 149)
        dct_call_result_27461 = invoke(stypy.reporting.localization.Localization(__file__, 149, 48), dct_27452, *[subscript_call_result_27456], **kwargs_27460)
        
        # Processing the call keyword arguments (line 149)
        # Getting the type of 'self' (line 150)
        self_27462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 32), 'self', False)
        # Obtaining the member 'dec' of a type (line 150)
        dec_27463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 32), self_27462, 'dec')
        keyword_27464 = dec_27463
        kwargs_27465 = {'decimal': keyword_27464}
        # Getting the type of 'assert_array_almost_equal' (line 149)
        assert_array_almost_equal_27447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 149)
        assert_array_almost_equal_call_result_27466 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), assert_array_almost_equal_27447, *[subscript_call_result_27451, dct_call_result_27461], **kwargs_27465)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 152):
        
        # Assigning a Attribute to a Name (line 152):
        # Getting the type of 'x' (line 152)
        x_27467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'x')
        # Obtaining the member 'T' of a type (line 152)
        T_27468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), x_27467, 'T')
        # Assigning a type to the variable 'x' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'x', T_27468)
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to dct(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'x' (line 153)
        x_27470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'x', False)
        # Processing the call keyword arguments (line 153)
        int_27471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 28), 'int')
        keyword_27472 = int_27471
        # Getting the type of 'self' (line 153)
        self_27473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 36), 'self', False)
        # Obtaining the member 'type' of a type (line 153)
        type_27474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 36), self_27473, 'type')
        keyword_27475 = type_27474
        kwargs_27476 = {'type': keyword_27475, 'axis': keyword_27472}
        # Getting the type of 'dct' (line 153)
        dct_27469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'dct', False)
        # Calling dct(args, kwargs) (line 153)
        dct_call_result_27477 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), dct_27469, *[x_27470], **kwargs_27476)
        
        # Assigning a type to the variable 'y' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'y', dct_call_result_27477)
        
        
        # Call to range(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'nt' (line 154)
        nt_27479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), 'nt', False)
        # Processing the call keyword arguments (line 154)
        kwargs_27480 = {}
        # Getting the type of 'range' (line 154)
        range_27478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 'range', False)
        # Calling range(args, kwargs) (line 154)
        range_call_result_27481 = invoke(stypy.reporting.localization.Localization(__file__, 154, 21), range_27478, *[nt_27479], **kwargs_27480)
        
        # Testing the type of a for loop iterable (line 154)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 154, 12), range_call_result_27481)
        # Getting the type of the for loop variable (line 154)
        for_loop_var_27482 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 154, 12), range_call_result_27481)
        # Assigning a type to the variable 'j' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'j', for_loop_var_27482)
        # SSA begins for a for statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_array_almost_equal(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Obtaining the type of the subscript
        slice_27484 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 155, 42), None, None, None)
        # Getting the type of 'j' (line 155)
        j_27485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 46), 'j', False)
        # Getting the type of 'y' (line 155)
        y_27486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 42), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___27487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 42), y_27486, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_27488 = invoke(stypy.reporting.localization.Localization(__file__, 155, 42), getitem___27487, (slice_27484, j_27485))
        
        
        # Call to dct(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Obtaining the type of the subscript
        slice_27490 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 155, 54), None, None, None)
        # Getting the type of 'j' (line 155)
        j_27491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 58), 'j', False)
        # Getting the type of 'x' (line 155)
        x_27492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 54), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___27493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 54), x_27492, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_27494 = invoke(stypy.reporting.localization.Localization(__file__, 155, 54), getitem___27493, (slice_27490, j_27491))
        
        # Processing the call keyword arguments (line 155)
        # Getting the type of 'self' (line 155)
        self_27495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 67), 'self', False)
        # Obtaining the member 'type' of a type (line 155)
        type_27496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 67), self_27495, 'type')
        keyword_27497 = type_27496
        kwargs_27498 = {'type': keyword_27497}
        # Getting the type of 'dct' (line 155)
        dct_27489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 50), 'dct', False)
        # Calling dct(args, kwargs) (line 155)
        dct_call_result_27499 = invoke(stypy.reporting.localization.Localization(__file__, 155, 50), dct_27489, *[subscript_call_result_27494], **kwargs_27498)
        
        # Processing the call keyword arguments (line 155)
        # Getting the type of 'self' (line 156)
        self_27500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'self', False)
        # Obtaining the member 'dec' of a type (line 156)
        dec_27501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 32), self_27500, 'dec')
        keyword_27502 = dec_27501
        kwargs_27503 = {'decimal': keyword_27502}
        # Getting the type of 'assert_array_almost_equal' (line 155)
        assert_array_almost_equal_27483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 155)
        assert_array_almost_equal_call_result_27504 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), assert_array_almost_equal_27483, *[subscript_call_result_27488, dct_call_result_27499], **kwargs_27503)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_axis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_axis' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_27505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27505)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_axis'
        return stypy_return_type_27505


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 125, 0, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestDCTBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_TestDCTBase' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), '_TestDCTBase', _TestDCTBase)
# Declaration of the '_TestDCTIIBase' class
# Getting the type of '_TestDCTBase' (line 159)
_TestDCTBase_27506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 21), '_TestDCTBase')

class _TestDCTIIBase(_TestDCTBase_27506, ):

    @norecursion
    def test_definition_matlab(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition_matlab'
        module_type_store = module_type_store.open_function_context('test_definition_matlab', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _TestDCTIIBase.test_definition_matlab.__dict__.__setitem__('stypy_localization', localization)
        _TestDCTIIBase.test_definition_matlab.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _TestDCTIIBase.test_definition_matlab.__dict__.__setitem__('stypy_type_store', module_type_store)
        _TestDCTIIBase.test_definition_matlab.__dict__.__setitem__('stypy_function_name', '_TestDCTIIBase.test_definition_matlab')
        _TestDCTIIBase.test_definition_matlab.__dict__.__setitem__('stypy_param_names_list', [])
        _TestDCTIIBase.test_definition_matlab.__dict__.__setitem__('stypy_varargs_param_name', None)
        _TestDCTIIBase.test_definition_matlab.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _TestDCTIIBase.test_definition_matlab.__dict__.__setitem__('stypy_call_defaults', defaults)
        _TestDCTIIBase.test_definition_matlab.__dict__.__setitem__('stypy_call_varargs', varargs)
        _TestDCTIIBase.test_definition_matlab.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _TestDCTIIBase.test_definition_matlab.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestDCTIIBase.test_definition_matlab', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition_matlab', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition_matlab(...)' code ##################

        
        
        # Call to range(...): (line 162)
        # Processing the call arguments (line 162)
        
        # Call to len(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'X' (line 162)
        X_27509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'X', False)
        # Processing the call keyword arguments (line 162)
        kwargs_27510 = {}
        # Getting the type of 'len' (line 162)
        len_27508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 23), 'len', False)
        # Calling len(args, kwargs) (line 162)
        len_call_result_27511 = invoke(stypy.reporting.localization.Localization(__file__, 162, 23), len_27508, *[X_27509], **kwargs_27510)
        
        # Processing the call keyword arguments (line 162)
        kwargs_27512 = {}
        # Getting the type of 'range' (line 162)
        range_27507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 17), 'range', False)
        # Calling range(args, kwargs) (line 162)
        range_call_result_27513 = invoke(stypy.reporting.localization.Localization(__file__, 162, 17), range_27507, *[len_call_result_27511], **kwargs_27512)
        
        # Testing the type of a for loop iterable (line 162)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 162, 8), range_call_result_27513)
        # Getting the type of the for loop variable (line 162)
        for_loop_var_27514 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 162, 8), range_call_result_27513)
        # Assigning a type to the variable 'i' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'i', for_loop_var_27514)
        # SSA begins for a for statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 163):
        
        # Assigning a Call to a Name (line 163):
        
        # Call to result_type(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'np' (line 163)
        np_27517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 32), 'np', False)
        # Obtaining the member 'float32' of a type (line 163)
        float32_27518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 32), np_27517, 'float32')
        # Getting the type of 'self' (line 163)
        self_27519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 44), 'self', False)
        # Obtaining the member 'rdt' of a type (line 163)
        rdt_27520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 44), self_27519, 'rdt')
        # Processing the call keyword arguments (line 163)
        kwargs_27521 = {}
        # Getting the type of 'np' (line 163)
        np_27515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 17), 'np', False)
        # Obtaining the member 'result_type' of a type (line 163)
        result_type_27516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 17), np_27515, 'result_type')
        # Calling result_type(args, kwargs) (line 163)
        result_type_call_result_27522 = invoke(stypy.reporting.localization.Localization(__file__, 163, 17), result_type_27516, *[float32_27518, rdt_27520], **kwargs_27521)
        
        # Assigning a type to the variable 'dt' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'dt', result_type_call_result_27522)
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to array(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 164)
        i_27525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'i', False)
        # Getting the type of 'X' (line 164)
        X_27526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___27527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 25), X_27526, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_27528 = invoke(stypy.reporting.localization.Localization(__file__, 164, 25), getitem___27527, i_27525)
        
        # Processing the call keyword arguments (line 164)
        # Getting the type of 'dt' (line 164)
        dt_27529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 37), 'dt', False)
        keyword_27530 = dt_27529
        kwargs_27531 = {'dtype': keyword_27530}
        # Getting the type of 'np' (line 164)
        np_27523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 164)
        array_27524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 16), np_27523, 'array')
        # Calling array(args, kwargs) (line 164)
        array_call_result_27532 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), array_27524, *[subscript_call_result_27528], **kwargs_27531)
        
        # Assigning a type to the variable 'x' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'x', array_call_result_27532)
        
        # Assigning a Subscript to a Name (line 166):
        
        # Assigning a Subscript to a Name (line 166):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 166)
        i_27533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 19), 'i')
        # Getting the type of 'Y' (line 166)
        Y_27534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'Y')
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___27535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 17), Y_27534, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_27536 = invoke(stypy.reporting.localization.Localization(__file__, 166, 17), getitem___27535, i_27533)
        
        # Assigning a type to the variable 'yr' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'yr', subscript_call_result_27536)
        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to dct(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'x' (line 167)
        x_27538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'x', False)
        # Processing the call keyword arguments (line 167)
        str_27539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 28), 'str', 'ortho')
        keyword_27540 = str_27539
        int_27541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 42), 'int')
        keyword_27542 = int_27541
        kwargs_27543 = {'type': keyword_27542, 'norm': keyword_27540}
        # Getting the type of 'dct' (line 167)
        dct_27537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'dct', False)
        # Calling dct(args, kwargs) (line 167)
        dct_call_result_27544 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), dct_27537, *[x_27538], **kwargs_27543)
        
        # Assigning a type to the variable 'y' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'y', dct_call_result_27544)
        
        # Call to assert_equal(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'y' (line 168)
        y_27546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'y', False)
        # Obtaining the member 'dtype' of a type (line 168)
        dtype_27547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 25), y_27546, 'dtype')
        # Getting the type of 'dt' (line 168)
        dt_27548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 34), 'dt', False)
        # Processing the call keyword arguments (line 168)
        kwargs_27549 = {}
        # Getting the type of 'assert_equal' (line 168)
        assert_equal_27545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 168)
        assert_equal_call_result_27550 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), assert_equal_27545, *[dtype_27547, dt_27548], **kwargs_27549)
        
        
        # Call to assert_array_almost_equal(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'y' (line 169)
        y_27552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 38), 'y', False)
        # Getting the type of 'yr' (line 169)
        yr_27553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 41), 'yr', False)
        # Processing the call keyword arguments (line 169)
        # Getting the type of 'self' (line 169)
        self_27554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 53), 'self', False)
        # Obtaining the member 'dec' of a type (line 169)
        dec_27555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 53), self_27554, 'dec')
        keyword_27556 = dec_27555
        kwargs_27557 = {'decimal': keyword_27556}
        # Getting the type of 'assert_array_almost_equal' (line 169)
        assert_array_almost_equal_27551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 169)
        assert_array_almost_equal_call_result_27558 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), assert_array_almost_equal_27551, *[y_27552, yr_27553], **kwargs_27557)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition_matlab(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition_matlab' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_27559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27559)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition_matlab'
        return stypy_return_type_27559


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 159, 0, False)
        # Assigning a type to the variable 'self' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestDCTIIBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_TestDCTIIBase' (line 159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), '_TestDCTIIBase', _TestDCTIIBase)
# Declaration of the '_TestDCTIIIBase' class
# Getting the type of '_TestDCTBase' (line 172)
_TestDCTBase_27560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 22), '_TestDCTBase')

class _TestDCTIIIBase(_TestDCTBase_27560, ):

    @norecursion
    def test_definition_ortho(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition_ortho'
        module_type_store = module_type_store.open_function_context('test_definition_ortho', 173, 4, False)
        # Assigning a type to the variable 'self' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _TestDCTIIIBase.test_definition_ortho.__dict__.__setitem__('stypy_localization', localization)
        _TestDCTIIIBase.test_definition_ortho.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _TestDCTIIIBase.test_definition_ortho.__dict__.__setitem__('stypy_type_store', module_type_store)
        _TestDCTIIIBase.test_definition_ortho.__dict__.__setitem__('stypy_function_name', '_TestDCTIIIBase.test_definition_ortho')
        _TestDCTIIIBase.test_definition_ortho.__dict__.__setitem__('stypy_param_names_list', [])
        _TestDCTIIIBase.test_definition_ortho.__dict__.__setitem__('stypy_varargs_param_name', None)
        _TestDCTIIIBase.test_definition_ortho.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _TestDCTIIIBase.test_definition_ortho.__dict__.__setitem__('stypy_call_defaults', defaults)
        _TestDCTIIIBase.test_definition_ortho.__dict__.__setitem__('stypy_call_varargs', varargs)
        _TestDCTIIIBase.test_definition_ortho.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _TestDCTIIIBase.test_definition_ortho.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestDCTIIIBase.test_definition_ortho', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition_ortho', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition_ortho(...)' code ##################

        
        
        # Call to range(...): (line 175)
        # Processing the call arguments (line 175)
        
        # Call to len(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'X' (line 175)
        X_27563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 27), 'X', False)
        # Processing the call keyword arguments (line 175)
        kwargs_27564 = {}
        # Getting the type of 'len' (line 175)
        len_27562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'len', False)
        # Calling len(args, kwargs) (line 175)
        len_call_result_27565 = invoke(stypy.reporting.localization.Localization(__file__, 175, 23), len_27562, *[X_27563], **kwargs_27564)
        
        # Processing the call keyword arguments (line 175)
        kwargs_27566 = {}
        # Getting the type of 'range' (line 175)
        range_27561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'range', False)
        # Calling range(args, kwargs) (line 175)
        range_call_result_27567 = invoke(stypy.reporting.localization.Localization(__file__, 175, 17), range_27561, *[len_call_result_27565], **kwargs_27566)
        
        # Testing the type of a for loop iterable (line 175)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 175, 8), range_call_result_27567)
        # Getting the type of the for loop variable (line 175)
        for_loop_var_27568 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 175, 8), range_call_result_27567)
        # Assigning a type to the variable 'i' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'i', for_loop_var_27568)
        # SSA begins for a for statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to array(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 176)
        i_27571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 27), 'i', False)
        # Getting the type of 'X' (line 176)
        X_27572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___27573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), X_27572, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_27574 = invoke(stypy.reporting.localization.Localization(__file__, 176, 25), getitem___27573, i_27571)
        
        # Processing the call keyword arguments (line 176)
        # Getting the type of 'self' (line 176)
        self_27575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 37), 'self', False)
        # Obtaining the member 'rdt' of a type (line 176)
        rdt_27576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 37), self_27575, 'rdt')
        keyword_27577 = rdt_27576
        kwargs_27578 = {'dtype': keyword_27577}
        # Getting the type of 'np' (line 176)
        np_27569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 176)
        array_27570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 16), np_27569, 'array')
        # Calling array(args, kwargs) (line 176)
        array_call_result_27579 = invoke(stypy.reporting.localization.Localization(__file__, 176, 16), array_27570, *[subscript_call_result_27574], **kwargs_27578)
        
        # Assigning a type to the variable 'x' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'x', array_call_result_27579)
        
        # Assigning a Call to a Name (line 177):
        
        # Assigning a Call to a Name (line 177):
        
        # Call to result_type(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'np' (line 177)
        np_27582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 32), 'np', False)
        # Obtaining the member 'float32' of a type (line 177)
        float32_27583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 32), np_27582, 'float32')
        # Getting the type of 'self' (line 177)
        self_27584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 44), 'self', False)
        # Obtaining the member 'rdt' of a type (line 177)
        rdt_27585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 44), self_27584, 'rdt')
        # Processing the call keyword arguments (line 177)
        kwargs_27586 = {}
        # Getting the type of 'np' (line 177)
        np_27580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'np', False)
        # Obtaining the member 'result_type' of a type (line 177)
        result_type_27581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 17), np_27580, 'result_type')
        # Calling result_type(args, kwargs) (line 177)
        result_type_call_result_27587 = invoke(stypy.reporting.localization.Localization(__file__, 177, 17), result_type_27581, *[float32_27583, rdt_27585], **kwargs_27586)
        
        # Assigning a type to the variable 'dt' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'dt', result_type_call_result_27587)
        
        # Assigning a Call to a Name (line 178):
        
        # Assigning a Call to a Name (line 178):
        
        # Call to dct(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'x' (line 178)
        x_27589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'x', False)
        # Processing the call keyword arguments (line 178)
        str_27590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 28), 'str', 'ortho')
        keyword_27591 = str_27590
        int_27592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 42), 'int')
        keyword_27593 = int_27592
        kwargs_27594 = {'type': keyword_27593, 'norm': keyword_27591}
        # Getting the type of 'dct' (line 178)
        dct_27588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'dct', False)
        # Calling dct(args, kwargs) (line 178)
        dct_call_result_27595 = invoke(stypy.reporting.localization.Localization(__file__, 178, 16), dct_27588, *[x_27589], **kwargs_27594)
        
        # Assigning a type to the variable 'y' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'y', dct_call_result_27595)
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to dct(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'y' (line 179)
        y_27597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'y', False)
        # Processing the call keyword arguments (line 179)
        str_27598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 29), 'str', 'ortho')
        keyword_27599 = str_27598
        int_27600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 43), 'int')
        keyword_27601 = int_27600
        kwargs_27602 = {'type': keyword_27601, 'norm': keyword_27599}
        # Getting the type of 'dct' (line 179)
        dct_27596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 17), 'dct', False)
        # Calling dct(args, kwargs) (line 179)
        dct_call_result_27603 = invoke(stypy.reporting.localization.Localization(__file__, 179, 17), dct_27596, *[y_27597], **kwargs_27602)
        
        # Assigning a type to the variable 'xi' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'xi', dct_call_result_27603)
        
        # Call to assert_equal(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'xi' (line 180)
        xi_27605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 25), 'xi', False)
        # Obtaining the member 'dtype' of a type (line 180)
        dtype_27606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 25), xi_27605, 'dtype')
        # Getting the type of 'dt' (line 180)
        dt_27607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 35), 'dt', False)
        # Processing the call keyword arguments (line 180)
        kwargs_27608 = {}
        # Getting the type of 'assert_equal' (line 180)
        assert_equal_27604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 180)
        assert_equal_call_result_27609 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), assert_equal_27604, *[dtype_27606, dt_27607], **kwargs_27608)
        
        
        # Call to assert_array_almost_equal(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'xi' (line 181)
        xi_27611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 38), 'xi', False)
        # Getting the type of 'x' (line 181)
        x_27612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 42), 'x', False)
        # Processing the call keyword arguments (line 181)
        # Getting the type of 'self' (line 181)
        self_27613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 53), 'self', False)
        # Obtaining the member 'dec' of a type (line 181)
        dec_27614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 53), self_27613, 'dec')
        keyword_27615 = dec_27614
        kwargs_27616 = {'decimal': keyword_27615}
        # Getting the type of 'assert_array_almost_equal' (line 181)
        assert_array_almost_equal_27610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 181)
        assert_array_almost_equal_call_result_27617 = invoke(stypy.reporting.localization.Localization(__file__, 181, 12), assert_array_almost_equal_27610, *[xi_27611, x_27612], **kwargs_27616)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition_ortho(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition_ortho' in the type store
        # Getting the type of 'stypy_return_type' (line 173)
        stypy_return_type_27618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27618)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition_ortho'
        return stypy_return_type_27618


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 172, 0, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestDCTIIIBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_TestDCTIIIBase' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), '_TestDCTIIIBase', _TestDCTIIIBase)
# Declaration of the 'TestDCTIDouble' class
# Getting the type of '_TestDCTBase' (line 184)
_TestDCTBase_27619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), '_TestDCTBase')

class TestDCTIDouble(_TestDCTBase_27619, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDCTIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDCTIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDCTIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDCTIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDCTIDouble.setup_method')
        TestDCTIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDCTIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDCTIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDCTIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDCTIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDCTIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDCTIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 186):
        
        # Assigning a Attribute to a Attribute (line 186):
        # Getting the type of 'np' (line 186)
        np_27620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'np')
        # Obtaining the member 'double' of a type (line 186)
        double_27621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 19), np_27620, 'double')
        # Getting the type of 'self' (line 186)
        self_27622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_27622, 'rdt', double_27621)
        
        # Assigning a Num to a Attribute (line 187):
        
        # Assigning a Num to a Attribute (line 187):
        int_27623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 19), 'int')
        # Getting the type of 'self' (line 187)
        self_27624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 187)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), self_27624, 'dec', int_27623)
        
        # Assigning a Num to a Attribute (line 188):
        
        # Assigning a Num to a Attribute (line 188):
        int_27625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 20), 'int')
        # Getting the type of 'self' (line 188)
        self_27626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self')
        # Setting the type of the member 'type' of a type (line 188)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_27626, 'type', int_27625)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_27627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27627)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27627


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 184, 0, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDCTIDouble' (line 184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'TestDCTIDouble', TestDCTIDouble)
# Declaration of the 'TestDCTIFloat' class
# Getting the type of '_TestDCTBase' (line 191)
_TestDCTBase_27628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), '_TestDCTBase')

class TestDCTIFloat(_TestDCTBase_27628, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 192, 4, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDCTIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDCTIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDCTIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDCTIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDCTIFloat.setup_method')
        TestDCTIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDCTIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDCTIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDCTIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDCTIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDCTIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDCTIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 193):
        
        # Assigning a Attribute to a Attribute (line 193):
        # Getting the type of 'np' (line 193)
        np_27629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'np')
        # Obtaining the member 'float32' of a type (line 193)
        float32_27630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 19), np_27629, 'float32')
        # Getting the type of 'self' (line 193)
        self_27631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 193)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), self_27631, 'rdt', float32_27630)
        
        # Assigning a Num to a Attribute (line 194):
        
        # Assigning a Num to a Attribute (line 194):
        int_27632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 19), 'int')
        # Getting the type of 'self' (line 194)
        self_27633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 194)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), self_27633, 'dec', int_27632)
        
        # Assigning a Num to a Attribute (line 195):
        
        # Assigning a Num to a Attribute (line 195):
        int_27634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 20), 'int')
        # Getting the type of 'self' (line 195)
        self_27635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'self')
        # Setting the type of the member 'type' of a type (line 195)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), self_27635, 'type', int_27634)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_27636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27636)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27636


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 191, 0, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDCTIFloat' (line 191)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'TestDCTIFloat', TestDCTIFloat)
# Declaration of the 'TestDCTIInt' class
# Getting the type of '_TestDCTBase' (line 198)
_TestDCTBase_27637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 18), '_TestDCTBase')

class TestDCTIInt(_TestDCTBase_27637, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 199, 4, False)
        # Assigning a type to the variable 'self' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDCTIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDCTIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDCTIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDCTIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDCTIInt.setup_method')
        TestDCTIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDCTIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDCTIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDCTIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDCTIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDCTIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDCTIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 200):
        
        # Assigning a Name to a Attribute (line 200):
        # Getting the type of 'int' (line 200)
        int_27638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'int')
        # Getting the type of 'self' (line 200)
        self_27639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 200)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_27639, 'rdt', int_27638)
        
        # Assigning a Num to a Attribute (line 201):
        
        # Assigning a Num to a Attribute (line 201):
        int_27640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 19), 'int')
        # Getting the type of 'self' (line 201)
        self_27641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 201)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), self_27641, 'dec', int_27640)
        
        # Assigning a Num to a Attribute (line 202):
        
        # Assigning a Num to a Attribute (line 202):
        int_27642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 20), 'int')
        # Getting the type of 'self' (line 202)
        self_27643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self')
        # Setting the type of the member 'type' of a type (line 202)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_27643, 'type', int_27642)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 199)
        stypy_return_type_27644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27644)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27644


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 198, 0, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDCTIInt' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'TestDCTIInt', TestDCTIInt)
# Declaration of the 'TestDCTIIDouble' class
# Getting the type of '_TestDCTIIBase' (line 205)
_TestDCTIIBase_27645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 22), '_TestDCTIIBase')

class TestDCTIIDouble(_TestDCTIIBase_27645, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 206, 4, False)
        # Assigning a type to the variable 'self' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDCTIIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDCTIIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDCTIIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDCTIIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDCTIIDouble.setup_method')
        TestDCTIIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDCTIIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDCTIIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDCTIIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDCTIIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDCTIIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDCTIIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 207):
        
        # Assigning a Attribute to a Attribute (line 207):
        # Getting the type of 'np' (line 207)
        np_27646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'np')
        # Obtaining the member 'double' of a type (line 207)
        double_27647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 19), np_27646, 'double')
        # Getting the type of 'self' (line 207)
        self_27648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_27648, 'rdt', double_27647)
        
        # Assigning a Num to a Attribute (line 208):
        
        # Assigning a Num to a Attribute (line 208):
        int_27649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 19), 'int')
        # Getting the type of 'self' (line 208)
        self_27650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 208)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), self_27650, 'dec', int_27649)
        
        # Assigning a Num to a Attribute (line 209):
        
        # Assigning a Num to a Attribute (line 209):
        int_27651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 20), 'int')
        # Getting the type of 'self' (line 209)
        self_27652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'self')
        # Setting the type of the member 'type' of a type (line 209)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), self_27652, 'type', int_27651)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 206)
        stypy_return_type_27653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27653)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27653


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 205, 0, False)
        # Assigning a type to the variable 'self' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDCTIIDouble' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), 'TestDCTIIDouble', TestDCTIIDouble)
# Declaration of the 'TestDCTIIFloat' class
# Getting the type of '_TestDCTIIBase' (line 212)
_TestDCTIIBase_27654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 21), '_TestDCTIIBase')

class TestDCTIIFloat(_TestDCTIIBase_27654, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 213, 4, False)
        # Assigning a type to the variable 'self' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDCTIIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDCTIIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDCTIIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDCTIIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDCTIIFloat.setup_method')
        TestDCTIIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDCTIIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDCTIIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDCTIIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDCTIIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDCTIIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDCTIIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 214):
        
        # Assigning a Attribute to a Attribute (line 214):
        # Getting the type of 'np' (line 214)
        np_27655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'np')
        # Obtaining the member 'float32' of a type (line 214)
        float32_27656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 19), np_27655, 'float32')
        # Getting the type of 'self' (line 214)
        self_27657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 214)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), self_27657, 'rdt', float32_27656)
        
        # Assigning a Num to a Attribute (line 215):
        
        # Assigning a Num to a Attribute (line 215):
        int_27658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 19), 'int')
        # Getting the type of 'self' (line 215)
        self_27659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 215)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), self_27659, 'dec', int_27658)
        
        # Assigning a Num to a Attribute (line 216):
        
        # Assigning a Num to a Attribute (line 216):
        int_27660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 20), 'int')
        # Getting the type of 'self' (line 216)
        self_27661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'self')
        # Setting the type of the member 'type' of a type (line 216)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), self_27661, 'type', int_27660)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_27662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27662)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27662


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 212, 0, False)
        # Assigning a type to the variable 'self' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDCTIIFloat' (line 212)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'TestDCTIIFloat', TestDCTIIFloat)
# Declaration of the 'TestDCTIIInt' class
# Getting the type of '_TestDCTIIBase' (line 219)
_TestDCTIIBase_27663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 19), '_TestDCTIIBase')

class TestDCTIIInt(_TestDCTIIBase_27663, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 220, 4, False)
        # Assigning a type to the variable 'self' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDCTIIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDCTIIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDCTIIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDCTIIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDCTIIInt.setup_method')
        TestDCTIIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDCTIIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDCTIIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDCTIIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDCTIIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDCTIIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDCTIIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 221):
        
        # Assigning a Name to a Attribute (line 221):
        # Getting the type of 'int' (line 221)
        int_27664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'int')
        # Getting the type of 'self' (line 221)
        self_27665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_27665, 'rdt', int_27664)
        
        # Assigning a Num to a Attribute (line 222):
        
        # Assigning a Num to a Attribute (line 222):
        int_27666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 19), 'int')
        # Getting the type of 'self' (line 222)
        self_27667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 222)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_27667, 'dec', int_27666)
        
        # Assigning a Num to a Attribute (line 223):
        
        # Assigning a Num to a Attribute (line 223):
        int_27668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 20), 'int')
        # Getting the type of 'self' (line 223)
        self_27669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'self')
        # Setting the type of the member 'type' of a type (line 223)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), self_27669, 'type', int_27668)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 220)
        stypy_return_type_27670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27670)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27670


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 219, 0, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDCTIIInt' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'TestDCTIIInt', TestDCTIIInt)
# Declaration of the 'TestDCTIIIDouble' class
# Getting the type of '_TestDCTIIIBase' (line 226)
_TestDCTIIIBase_27671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), '_TestDCTIIIBase')

class TestDCTIIIDouble(_TestDCTIIIBase_27671, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 227, 4, False)
        # Assigning a type to the variable 'self' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDCTIIIDouble.setup_method')
        TestDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 228):
        
        # Assigning a Attribute to a Attribute (line 228):
        # Getting the type of 'np' (line 228)
        np_27672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'np')
        # Obtaining the member 'double' of a type (line 228)
        double_27673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), np_27672, 'double')
        # Getting the type of 'self' (line 228)
        self_27674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 228)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), self_27674, 'rdt', double_27673)
        
        # Assigning a Num to a Attribute (line 229):
        
        # Assigning a Num to a Attribute (line 229):
        int_27675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 19), 'int')
        # Getting the type of 'self' (line 229)
        self_27676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), self_27676, 'dec', int_27675)
        
        # Assigning a Num to a Attribute (line 230):
        
        # Assigning a Num to a Attribute (line 230):
        int_27677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 20), 'int')
        # Getting the type of 'self' (line 230)
        self_27678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self')
        # Setting the type of the member 'type' of a type (line 230)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_27678, 'type', int_27677)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 227)
        stypy_return_type_27679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27679


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 226, 0, False)
        # Assigning a type to the variable 'self' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDCTIIIDouble' (line 226)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 0), 'TestDCTIIIDouble', TestDCTIIIDouble)
# Declaration of the 'TestDCTIIIFloat' class
# Getting the type of '_TestDCTIIIBase' (line 233)
_TestDCTIIIBase_27680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 22), '_TestDCTIIIBase')

class TestDCTIIIFloat(_TestDCTIIIBase_27680, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDCTIIIFloat.setup_method')
        TestDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 235):
        
        # Assigning a Attribute to a Attribute (line 235):
        # Getting the type of 'np' (line 235)
        np_27681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'np')
        # Obtaining the member 'float32' of a type (line 235)
        float32_27682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 19), np_27681, 'float32')
        # Getting the type of 'self' (line 235)
        self_27683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_27683, 'rdt', float32_27682)
        
        # Assigning a Num to a Attribute (line 236):
        
        # Assigning a Num to a Attribute (line 236):
        int_27684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 19), 'int')
        # Getting the type of 'self' (line 236)
        self_27685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 236)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_27685, 'dec', int_27684)
        
        # Assigning a Num to a Attribute (line 237):
        
        # Assigning a Num to a Attribute (line 237):
        int_27686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 20), 'int')
        # Getting the type of 'self' (line 237)
        self_27687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'self')
        # Setting the type of the member 'type' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), self_27687, 'type', int_27686)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_27688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27688)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27688


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 233, 0, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDCTIIIFloat' (line 233)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 0), 'TestDCTIIIFloat', TestDCTIIIFloat)
# Declaration of the 'TestDCTIIIInt' class
# Getting the type of '_TestDCTIIIBase' (line 240)
_TestDCTIIIBase_27689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), '_TestDCTIIIBase')

class TestDCTIIIInt(_TestDCTIIIBase_27689, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 241, 4, False)
        # Assigning a type to the variable 'self' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDCTIIIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDCTIIIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDCTIIIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDCTIIIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDCTIIIInt.setup_method')
        TestDCTIIIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDCTIIIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDCTIIIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDCTIIIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDCTIIIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDCTIIIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDCTIIIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 242):
        
        # Assigning a Name to a Attribute (line 242):
        # Getting the type of 'int' (line 242)
        int_27690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'int')
        # Getting the type of 'self' (line 242)
        self_27691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_27691, 'rdt', int_27690)
        
        # Assigning a Num to a Attribute (line 243):
        
        # Assigning a Num to a Attribute (line 243):
        int_27692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 19), 'int')
        # Getting the type of 'self' (line 243)
        self_27693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 243)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), self_27693, 'dec', int_27692)
        
        # Assigning a Num to a Attribute (line 244):
        
        # Assigning a Num to a Attribute (line 244):
        int_27694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 20), 'int')
        # Getting the type of 'self' (line 244)
        self_27695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self')
        # Setting the type of the member 'type' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_27695, 'type', int_27694)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 241)
        stypy_return_type_27696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27696)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27696


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 240, 0, False)
        # Assigning a type to the variable 'self' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDCTIIIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDCTIIIInt' (line 240)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), 'TestDCTIIIInt', TestDCTIIIInt)
# Declaration of the '_TestIDCTBase' class

class _TestIDCTBase(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _TestIDCTBase.setup_method.__dict__.__setitem__('stypy_localization', localization)
        _TestIDCTBase.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _TestIDCTBase.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        _TestIDCTBase.setup_method.__dict__.__setitem__('stypy_function_name', '_TestIDCTBase.setup_method')
        _TestIDCTBase.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        _TestIDCTBase.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        _TestIDCTBase.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _TestIDCTBase.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        _TestIDCTBase.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        _TestIDCTBase.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _TestIDCTBase.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestIDCTBase.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 249):
        
        # Assigning a Name to a Attribute (line 249):
        # Getting the type of 'None' (line 249)
        None_27697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'None')
        # Getting the type of 'self' (line 249)
        self_27698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_27698, 'rdt', None_27697)
        
        # Assigning a Num to a Attribute (line 250):
        
        # Assigning a Num to a Attribute (line 250):
        int_27699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 19), 'int')
        # Getting the type of 'self' (line 250)
        self_27700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_27700, 'dec', int_27699)
        
        # Assigning a Name to a Attribute (line 251):
        
        # Assigning a Name to a Attribute (line 251):
        # Getting the type of 'None' (line 251)
        None_27701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'None')
        # Getting the type of 'self' (line 251)
        self_27702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self')
        # Setting the type of the member 'type' of a type (line 251)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_27702, 'type', None_27701)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_27703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27703)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27703


    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _TestIDCTBase.test_definition.__dict__.__setitem__('stypy_localization', localization)
        _TestIDCTBase.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _TestIDCTBase.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        _TestIDCTBase.test_definition.__dict__.__setitem__('stypy_function_name', '_TestIDCTBase.test_definition')
        _TestIDCTBase.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        _TestIDCTBase.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        _TestIDCTBase.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _TestIDCTBase.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        _TestIDCTBase.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        _TestIDCTBase.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _TestIDCTBase.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestIDCTBase.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        # Getting the type of 'FFTWDATA_SIZES' (line 254)
        FFTWDATA_SIZES_27704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'FFTWDATA_SIZES')
        # Testing the type of a for loop iterable (line 254)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 254, 8), FFTWDATA_SIZES_27704)
        # Getting the type of the for loop variable (line 254)
        for_loop_var_27705 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 254, 8), FFTWDATA_SIZES_27704)
        # Assigning a type to the variable 'i' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'i', for_loop_var_27705)
        # SSA begins for a for statement (line 254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 255):
        
        # Assigning a Subscript to a Name (line 255):
        
        # Obtaining the type of the subscript
        int_27706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 12), 'int')
        
        # Call to fftw_dct_ref(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'self' (line 255)
        self_27708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 38), 'self', False)
        # Obtaining the member 'type' of a type (line 255)
        type_27709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 38), self_27708, 'type')
        # Getting the type of 'i' (line 255)
        i_27710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 49), 'i', False)
        # Getting the type of 'self' (line 255)
        self_27711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 52), 'self', False)
        # Obtaining the member 'rdt' of a type (line 255)
        rdt_27712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 52), self_27711, 'rdt')
        # Processing the call keyword arguments (line 255)
        kwargs_27713 = {}
        # Getting the type of 'fftw_dct_ref' (line 255)
        fftw_dct_ref_27707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 25), 'fftw_dct_ref', False)
        # Calling fftw_dct_ref(args, kwargs) (line 255)
        fftw_dct_ref_call_result_27714 = invoke(stypy.reporting.localization.Localization(__file__, 255, 25), fftw_dct_ref_27707, *[type_27709, i_27710, rdt_27712], **kwargs_27713)
        
        # Obtaining the member '__getitem__' of a type (line 255)
        getitem___27715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), fftw_dct_ref_call_result_27714, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 255)
        subscript_call_result_27716 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), getitem___27715, int_27706)
        
        # Assigning a type to the variable 'tuple_var_assignment_26779' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'tuple_var_assignment_26779', subscript_call_result_27716)
        
        # Assigning a Subscript to a Name (line 255):
        
        # Obtaining the type of the subscript
        int_27717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 12), 'int')
        
        # Call to fftw_dct_ref(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'self' (line 255)
        self_27719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 38), 'self', False)
        # Obtaining the member 'type' of a type (line 255)
        type_27720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 38), self_27719, 'type')
        # Getting the type of 'i' (line 255)
        i_27721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 49), 'i', False)
        # Getting the type of 'self' (line 255)
        self_27722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 52), 'self', False)
        # Obtaining the member 'rdt' of a type (line 255)
        rdt_27723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 52), self_27722, 'rdt')
        # Processing the call keyword arguments (line 255)
        kwargs_27724 = {}
        # Getting the type of 'fftw_dct_ref' (line 255)
        fftw_dct_ref_27718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 25), 'fftw_dct_ref', False)
        # Calling fftw_dct_ref(args, kwargs) (line 255)
        fftw_dct_ref_call_result_27725 = invoke(stypy.reporting.localization.Localization(__file__, 255, 25), fftw_dct_ref_27718, *[type_27720, i_27721, rdt_27723], **kwargs_27724)
        
        # Obtaining the member '__getitem__' of a type (line 255)
        getitem___27726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), fftw_dct_ref_call_result_27725, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 255)
        subscript_call_result_27727 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), getitem___27726, int_27717)
        
        # Assigning a type to the variable 'tuple_var_assignment_26780' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'tuple_var_assignment_26780', subscript_call_result_27727)
        
        # Assigning a Subscript to a Name (line 255):
        
        # Obtaining the type of the subscript
        int_27728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 12), 'int')
        
        # Call to fftw_dct_ref(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'self' (line 255)
        self_27730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 38), 'self', False)
        # Obtaining the member 'type' of a type (line 255)
        type_27731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 38), self_27730, 'type')
        # Getting the type of 'i' (line 255)
        i_27732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 49), 'i', False)
        # Getting the type of 'self' (line 255)
        self_27733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 52), 'self', False)
        # Obtaining the member 'rdt' of a type (line 255)
        rdt_27734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 52), self_27733, 'rdt')
        # Processing the call keyword arguments (line 255)
        kwargs_27735 = {}
        # Getting the type of 'fftw_dct_ref' (line 255)
        fftw_dct_ref_27729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 25), 'fftw_dct_ref', False)
        # Calling fftw_dct_ref(args, kwargs) (line 255)
        fftw_dct_ref_call_result_27736 = invoke(stypy.reporting.localization.Localization(__file__, 255, 25), fftw_dct_ref_27729, *[type_27731, i_27732, rdt_27734], **kwargs_27735)
        
        # Obtaining the member '__getitem__' of a type (line 255)
        getitem___27737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), fftw_dct_ref_call_result_27736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 255)
        subscript_call_result_27738 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), getitem___27737, int_27728)
        
        # Assigning a type to the variable 'tuple_var_assignment_26781' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'tuple_var_assignment_26781', subscript_call_result_27738)
        
        # Assigning a Name to a Name (line 255):
        # Getting the type of 'tuple_var_assignment_26779' (line 255)
        tuple_var_assignment_26779_27739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'tuple_var_assignment_26779')
        # Assigning a type to the variable 'xr' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'xr', tuple_var_assignment_26779_27739)
        
        # Assigning a Name to a Name (line 255):
        # Getting the type of 'tuple_var_assignment_26780' (line 255)
        tuple_var_assignment_26780_27740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'tuple_var_assignment_26780')
        # Assigning a type to the variable 'yr' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'yr', tuple_var_assignment_26780_27740)
        
        # Assigning a Name to a Name (line 255):
        # Getting the type of 'tuple_var_assignment_26781' (line 255)
        tuple_var_assignment_26781_27741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'tuple_var_assignment_26781')
        # Assigning a type to the variable 'dt' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'dt', tuple_var_assignment_26781_27741)
        
        # Assigning a Call to a Name (line 256):
        
        # Assigning a Call to a Name (line 256):
        
        # Call to idct(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'yr' (line 256)
        yr_27743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 21), 'yr', False)
        # Processing the call keyword arguments (line 256)
        # Getting the type of 'self' (line 256)
        self_27744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 30), 'self', False)
        # Obtaining the member 'type' of a type (line 256)
        type_27745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 30), self_27744, 'type')
        keyword_27746 = type_27745
        kwargs_27747 = {'type': keyword_27746}
        # Getting the type of 'idct' (line 256)
        idct_27742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'idct', False)
        # Calling idct(args, kwargs) (line 256)
        idct_call_result_27748 = invoke(stypy.reporting.localization.Localization(__file__, 256, 16), idct_27742, *[yr_27743], **kwargs_27747)
        
        # Assigning a type to the variable 'x' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'x', idct_call_result_27748)
        
        
        # Getting the type of 'self' (line 257)
        self_27749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'self')
        # Obtaining the member 'type' of a type (line 257)
        type_27750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 15), self_27749, 'type')
        int_27751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 28), 'int')
        # Applying the binary operator '==' (line 257)
        result_eq_27752 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 15), '==', type_27750, int_27751)
        
        # Testing the type of an if condition (line 257)
        if_condition_27753 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 12), result_eq_27752)
        # Assigning a type to the variable 'if_condition_27753' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'if_condition_27753', if_condition_27753)
        # SSA begins for if statement (line 257)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'x' (line 258)
        x_27754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'x')
        int_27755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 21), 'int')
        # Getting the type of 'i' (line 258)
        i_27756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 26), 'i')
        int_27757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 28), 'int')
        # Applying the binary operator '-' (line 258)
        result_sub_27758 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 26), '-', i_27756, int_27757)
        
        # Applying the binary operator '*' (line 258)
        result_mul_27759 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 21), '*', int_27755, result_sub_27758)
        
        # Applying the binary operator 'div=' (line 258)
        result_div_27760 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 16), 'div=', x_27754, result_mul_27759)
        # Assigning a type to the variable 'x' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'x', result_div_27760)
        
        # SSA branch for the else part of an if statement (line 257)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'x' (line 260)
        x_27761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'x')
        int_27762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 21), 'int')
        # Getting the type of 'i' (line 260)
        i_27763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 25), 'i')
        # Applying the binary operator '*' (line 260)
        result_mul_27764 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 21), '*', int_27762, i_27763)
        
        # Applying the binary operator 'div=' (line 260)
        result_div_27765 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 16), 'div=', x_27761, result_mul_27764)
        # Assigning a type to the variable 'x' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'x', result_div_27765)
        
        # SSA join for if statement (line 257)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'x' (line 261)
        x_27767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 25), 'x', False)
        # Obtaining the member 'dtype' of a type (line 261)
        dtype_27768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 25), x_27767, 'dtype')
        # Getting the type of 'dt' (line 261)
        dt_27769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 34), 'dt', False)
        # Processing the call keyword arguments (line 261)
        kwargs_27770 = {}
        # Getting the type of 'assert_equal' (line 261)
        assert_equal_27766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 261)
        assert_equal_call_result_27771 = invoke(stypy.reporting.localization.Localization(__file__, 261, 12), assert_equal_27766, *[dtype_27768, dt_27769], **kwargs_27770)
        
        
        # Call to assert_array_almost_equal(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'x' (line 266)
        x_27773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 38), 'x', False)
        
        # Call to max(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'x' (line 266)
        x_27776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 49), 'x', False)
        # Processing the call keyword arguments (line 266)
        kwargs_27777 = {}
        # Getting the type of 'np' (line 266)
        np_27774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 42), 'np', False)
        # Obtaining the member 'max' of a type (line 266)
        max_27775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 42), np_27774, 'max')
        # Calling max(args, kwargs) (line 266)
        max_call_result_27778 = invoke(stypy.reporting.localization.Localization(__file__, 266, 42), max_27775, *[x_27776], **kwargs_27777)
        
        # Applying the binary operator 'div' (line 266)
        result_div_27779 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 38), 'div', x_27773, max_call_result_27778)
        
        # Getting the type of 'xr' (line 266)
        xr_27780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 53), 'xr', False)
        
        # Call to max(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'x' (line 266)
        x_27783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 65), 'x', False)
        # Processing the call keyword arguments (line 266)
        kwargs_27784 = {}
        # Getting the type of 'np' (line 266)
        np_27781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 58), 'np', False)
        # Obtaining the member 'max' of a type (line 266)
        max_27782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 58), np_27781, 'max')
        # Calling max(args, kwargs) (line 266)
        max_call_result_27785 = invoke(stypy.reporting.localization.Localization(__file__, 266, 58), max_27782, *[x_27783], **kwargs_27784)
        
        # Applying the binary operator 'div' (line 266)
        result_div_27786 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 53), 'div', xr_27780, max_call_result_27785)
        
        # Processing the call keyword arguments (line 266)
        # Getting the type of 'self' (line 266)
        self_27787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 77), 'self', False)
        # Obtaining the member 'dec' of a type (line 266)
        dec_27788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 77), self_27787, 'dec')
        keyword_27789 = dec_27788
        str_27790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 28), 'str', 'Size %d failed')
        # Getting the type of 'i' (line 267)
        i_27791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 47), 'i', False)
        # Applying the binary operator '%' (line 267)
        result_mod_27792 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 28), '%', str_27790, i_27791)
        
        keyword_27793 = result_mod_27792
        kwargs_27794 = {'decimal': keyword_27789, 'err_msg': keyword_27793}
        # Getting the type of 'assert_array_almost_equal' (line 266)
        assert_array_almost_equal_27772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 266)
        assert_array_almost_equal_call_result_27795 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), assert_array_almost_equal_27772, *[result_div_27779, result_div_27786], **kwargs_27794)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_27796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27796)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_27796


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 247, 0, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestIDCTBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_TestIDCTBase' (line 247)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 0), '_TestIDCTBase', _TestIDCTBase)
# Declaration of the 'TestIDCTIDouble' class
# Getting the type of '_TestIDCTBase' (line 270)
_TestIDCTBase_27797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 22), '_TestIDCTBase')

class TestIDCTIDouble(_TestIDCTBase_27797, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 271, 4, False)
        # Assigning a type to the variable 'self' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDCTIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDCTIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDCTIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDCTIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDCTIDouble.setup_method')
        TestIDCTIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDCTIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDCTIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDCTIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDCTIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDCTIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDCTIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 272):
        
        # Assigning a Attribute to a Attribute (line 272):
        # Getting the type of 'np' (line 272)
        np_27798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'np')
        # Obtaining the member 'double' of a type (line 272)
        double_27799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 19), np_27798, 'double')
        # Getting the type of 'self' (line 272)
        self_27800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 272)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_27800, 'rdt', double_27799)
        
        # Assigning a Num to a Attribute (line 273):
        
        # Assigning a Num to a Attribute (line 273):
        int_27801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 19), 'int')
        # Getting the type of 'self' (line 273)
        self_27802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 273)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), self_27802, 'dec', int_27801)
        
        # Assigning a Num to a Attribute (line 274):
        
        # Assigning a Num to a Attribute (line 274):
        int_27803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 20), 'int')
        # Getting the type of 'self' (line 274)
        self_27804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'self')
        # Setting the type of the member 'type' of a type (line 274)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), self_27804, 'type', int_27803)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 271)
        stypy_return_type_27805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27805)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27805


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 270, 0, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDCTIDouble' (line 270)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'TestIDCTIDouble', TestIDCTIDouble)
# Declaration of the 'TestIDCTIFloat' class
# Getting the type of '_TestIDCTBase' (line 277)
_TestIDCTBase_27806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), '_TestIDCTBase')

class TestIDCTIFloat(_TestIDCTBase_27806, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDCTIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDCTIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDCTIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDCTIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDCTIFloat.setup_method')
        TestIDCTIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDCTIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDCTIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDCTIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDCTIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDCTIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDCTIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 279):
        
        # Assigning a Attribute to a Attribute (line 279):
        # Getting the type of 'np' (line 279)
        np_27807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 19), 'np')
        # Obtaining the member 'float32' of a type (line 279)
        float32_27808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 19), np_27807, 'float32')
        # Getting the type of 'self' (line 279)
        self_27809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 279)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), self_27809, 'rdt', float32_27808)
        
        # Assigning a Num to a Attribute (line 280):
        
        # Assigning a Num to a Attribute (line 280):
        int_27810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 19), 'int')
        # Getting the type of 'self' (line 280)
        self_27811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 280)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), self_27811, 'dec', int_27810)
        
        # Assigning a Num to a Attribute (line 281):
        
        # Assigning a Num to a Attribute (line 281):
        int_27812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 20), 'int')
        # Getting the type of 'self' (line 281)
        self_27813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'self')
        # Setting the type of the member 'type' of a type (line 281)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), self_27813, 'type', int_27812)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_27814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27814)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27814


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 277, 0, False)
        # Assigning a type to the variable 'self' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDCTIFloat' (line 277)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 0), 'TestIDCTIFloat', TestIDCTIFloat)
# Declaration of the 'TestIDCTIInt' class
# Getting the type of '_TestIDCTBase' (line 284)
_TestIDCTBase_27815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 19), '_TestIDCTBase')

class TestIDCTIInt(_TestIDCTBase_27815, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDCTIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDCTIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDCTIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDCTIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDCTIInt.setup_method')
        TestIDCTIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDCTIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDCTIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDCTIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDCTIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDCTIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDCTIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 286):
        
        # Assigning a Name to a Attribute (line 286):
        # Getting the type of 'int' (line 286)
        int_27816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'int')
        # Getting the type of 'self' (line 286)
        self_27817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 286)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), self_27817, 'rdt', int_27816)
        
        # Assigning a Num to a Attribute (line 287):
        
        # Assigning a Num to a Attribute (line 287):
        int_27818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 19), 'int')
        # Getting the type of 'self' (line 287)
        self_27819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 287)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_27819, 'dec', int_27818)
        
        # Assigning a Num to a Attribute (line 288):
        
        # Assigning a Num to a Attribute (line 288):
        int_27820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 20), 'int')
        # Getting the type of 'self' (line 288)
        self_27821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'self')
        # Setting the type of the member 'type' of a type (line 288)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), self_27821, 'type', int_27820)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_27822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27822


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 284, 0, False)
        # Assigning a type to the variable 'self' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDCTIInt' (line 284)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'TestIDCTIInt', TestIDCTIInt)
# Declaration of the 'TestIDCTIIDouble' class
# Getting the type of '_TestIDCTBase' (line 291)
_TestIDCTBase_27823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 23), '_TestIDCTBase')

class TestIDCTIIDouble(_TestIDCTBase_27823, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 292, 4, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDCTIIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDCTIIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDCTIIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDCTIIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDCTIIDouble.setup_method')
        TestIDCTIIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDCTIIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDCTIIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDCTIIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDCTIIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDCTIIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDCTIIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 293):
        
        # Assigning a Attribute to a Attribute (line 293):
        # Getting the type of 'np' (line 293)
        np_27824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'np')
        # Obtaining the member 'double' of a type (line 293)
        double_27825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 19), np_27824, 'double')
        # Getting the type of 'self' (line 293)
        self_27826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 293)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), self_27826, 'rdt', double_27825)
        
        # Assigning a Num to a Attribute (line 294):
        
        # Assigning a Num to a Attribute (line 294):
        int_27827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 19), 'int')
        # Getting the type of 'self' (line 294)
        self_27828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 294)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), self_27828, 'dec', int_27827)
        
        # Assigning a Num to a Attribute (line 295):
        
        # Assigning a Num to a Attribute (line 295):
        int_27829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 20), 'int')
        # Getting the type of 'self' (line 295)
        self_27830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'self')
        # Setting the type of the member 'type' of a type (line 295)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), self_27830, 'type', int_27829)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 292)
        stypy_return_type_27831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27831)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27831


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 291, 0, False)
        # Assigning a type to the variable 'self' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDCTIIDouble' (line 291)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 0), 'TestIDCTIIDouble', TestIDCTIIDouble)
# Declaration of the 'TestIDCTIIFloat' class
# Getting the type of '_TestIDCTBase' (line 298)
_TestIDCTBase_27832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 22), '_TestIDCTBase')

class TestIDCTIIFloat(_TestIDCTBase_27832, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 299, 4, False)
        # Assigning a type to the variable 'self' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDCTIIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDCTIIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDCTIIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDCTIIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDCTIIFloat.setup_method')
        TestIDCTIIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDCTIIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDCTIIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDCTIIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDCTIIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDCTIIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDCTIIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 300):
        
        # Assigning a Attribute to a Attribute (line 300):
        # Getting the type of 'np' (line 300)
        np_27833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 19), 'np')
        # Obtaining the member 'float32' of a type (line 300)
        float32_27834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 19), np_27833, 'float32')
        # Getting the type of 'self' (line 300)
        self_27835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 300)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), self_27835, 'rdt', float32_27834)
        
        # Assigning a Num to a Attribute (line 301):
        
        # Assigning a Num to a Attribute (line 301):
        int_27836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 19), 'int')
        # Getting the type of 'self' (line 301)
        self_27837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 301)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), self_27837, 'dec', int_27836)
        
        # Assigning a Num to a Attribute (line 302):
        
        # Assigning a Num to a Attribute (line 302):
        int_27838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 20), 'int')
        # Getting the type of 'self' (line 302)
        self_27839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'self')
        # Setting the type of the member 'type' of a type (line 302)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), self_27839, 'type', int_27838)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 299)
        stypy_return_type_27840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27840


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 298, 0, False)
        # Assigning a type to the variable 'self' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDCTIIFloat' (line 298)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'TestIDCTIIFloat', TestIDCTIIFloat)
# Declaration of the 'TestIDCTIIInt' class
# Getting the type of '_TestIDCTBase' (line 305)
_TestIDCTBase_27841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), '_TestIDCTBase')

class TestIDCTIIInt(_TestIDCTBase_27841, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 306, 4, False)
        # Assigning a type to the variable 'self' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDCTIIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDCTIIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDCTIIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDCTIIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDCTIIInt.setup_method')
        TestIDCTIIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDCTIIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDCTIIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDCTIIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDCTIIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDCTIIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDCTIIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 307):
        
        # Assigning a Name to a Attribute (line 307):
        # Getting the type of 'int' (line 307)
        int_27842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 'int')
        # Getting the type of 'self' (line 307)
        self_27843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 307)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), self_27843, 'rdt', int_27842)
        
        # Assigning a Num to a Attribute (line 308):
        
        # Assigning a Num to a Attribute (line 308):
        int_27844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 19), 'int')
        # Getting the type of 'self' (line 308)
        self_27845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 308)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), self_27845, 'dec', int_27844)
        
        # Assigning a Num to a Attribute (line 309):
        
        # Assigning a Num to a Attribute (line 309):
        int_27846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 20), 'int')
        # Getting the type of 'self' (line 309)
        self_27847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self')
        # Setting the type of the member 'type' of a type (line 309)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), self_27847, 'type', int_27846)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 306)
        stypy_return_type_27848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27848)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27848


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 305, 0, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDCTIIInt' (line 305)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 0), 'TestIDCTIIInt', TestIDCTIIInt)
# Declaration of the 'TestIDCTIIIDouble' class
# Getting the type of '_TestIDCTBase' (line 312)
_TestIDCTBase_27849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 24), '_TestIDCTBase')

class TestIDCTIIIDouble(_TestIDCTBase_27849, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 313, 4, False)
        # Assigning a type to the variable 'self' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDCTIIIDouble.setup_method')
        TestIDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDCTIIIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 314):
        
        # Assigning a Attribute to a Attribute (line 314):
        # Getting the type of 'np' (line 314)
        np_27850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 19), 'np')
        # Obtaining the member 'double' of a type (line 314)
        double_27851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 19), np_27850, 'double')
        # Getting the type of 'self' (line 314)
        self_27852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 314)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), self_27852, 'rdt', double_27851)
        
        # Assigning a Num to a Attribute (line 315):
        
        # Assigning a Num to a Attribute (line 315):
        int_27853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 19), 'int')
        # Getting the type of 'self' (line 315)
        self_27854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_27854, 'dec', int_27853)
        
        # Assigning a Num to a Attribute (line 316):
        
        # Assigning a Num to a Attribute (line 316):
        int_27855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 20), 'int')
        # Getting the type of 'self' (line 316)
        self_27856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'self')
        # Setting the type of the member 'type' of a type (line 316)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), self_27856, 'type', int_27855)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 313)
        stypy_return_type_27857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27857)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27857


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 312, 0, False)
        # Assigning a type to the variable 'self' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDCTIIIDouble' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'TestIDCTIIIDouble', TestIDCTIIIDouble)
# Declaration of the 'TestIDCTIIIFloat' class
# Getting the type of '_TestIDCTBase' (line 319)
_TestIDCTBase_27858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 23), '_TestIDCTBase')

class TestIDCTIIIFloat(_TestIDCTBase_27858, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 320, 4, False)
        # Assigning a type to the variable 'self' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDCTIIIFloat.setup_method')
        TestIDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDCTIIIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 321):
        
        # Assigning a Attribute to a Attribute (line 321):
        # Getting the type of 'np' (line 321)
        np_27859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 'np')
        # Obtaining the member 'float32' of a type (line 321)
        float32_27860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 19), np_27859, 'float32')
        # Getting the type of 'self' (line 321)
        self_27861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 321)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), self_27861, 'rdt', float32_27860)
        
        # Assigning a Num to a Attribute (line 322):
        
        # Assigning a Num to a Attribute (line 322):
        int_27862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 19), 'int')
        # Getting the type of 'self' (line 322)
        self_27863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 322)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), self_27863, 'dec', int_27862)
        
        # Assigning a Num to a Attribute (line 323):
        
        # Assigning a Num to a Attribute (line 323):
        int_27864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 20), 'int')
        # Getting the type of 'self' (line 323)
        self_27865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'self')
        # Setting the type of the member 'type' of a type (line 323)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), self_27865, 'type', int_27864)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 320)
        stypy_return_type_27866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27866)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27866


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 319, 0, False)
        # Assigning a type to the variable 'self' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDCTIIIFloat' (line 319)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 0), 'TestIDCTIIIFloat', TestIDCTIIIFloat)
# Declaration of the 'TestIDCTIIIInt' class
# Getting the type of '_TestIDCTBase' (line 326)
_TestIDCTBase_27867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 21), '_TestIDCTBase')

class TestIDCTIIIInt(_TestIDCTBase_27867, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 327, 4, False)
        # Assigning a type to the variable 'self' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDCTIIIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDCTIIIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDCTIIIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDCTIIIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDCTIIIInt.setup_method')
        TestIDCTIIIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDCTIIIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDCTIIIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDCTIIIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDCTIIIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDCTIIIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDCTIIIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 328):
        
        # Assigning a Name to a Attribute (line 328):
        # Getting the type of 'int' (line 328)
        int_27868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 19), 'int')
        # Getting the type of 'self' (line 328)
        self_27869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 328)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 8), self_27869, 'rdt', int_27868)
        
        # Assigning a Num to a Attribute (line 329):
        
        # Assigning a Num to a Attribute (line 329):
        int_27870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 19), 'int')
        # Getting the type of 'self' (line 329)
        self_27871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 329)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), self_27871, 'dec', int_27870)
        
        # Assigning a Num to a Attribute (line 330):
        
        # Assigning a Num to a Attribute (line 330):
        int_27872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 20), 'int')
        # Getting the type of 'self' (line 330)
        self_27873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'self')
        # Setting the type of the member 'type' of a type (line 330)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), self_27873, 'type', int_27872)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 327)
        stypy_return_type_27874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27874)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27874


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 326, 0, False)
        # Assigning a type to the variable 'self' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDCTIIIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDCTIIIInt' (line 326)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 0), 'TestIDCTIIIInt', TestIDCTIIIInt)
# Declaration of the '_TestDSTBase' class

class _TestDSTBase(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _TestDSTBase.setup_method.__dict__.__setitem__('stypy_localization', localization)
        _TestDSTBase.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _TestDSTBase.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        _TestDSTBase.setup_method.__dict__.__setitem__('stypy_function_name', '_TestDSTBase.setup_method')
        _TestDSTBase.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        _TestDSTBase.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        _TestDSTBase.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _TestDSTBase.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        _TestDSTBase.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        _TestDSTBase.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _TestDSTBase.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestDSTBase.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 335):
        
        # Assigning a Name to a Attribute (line 335):
        # Getting the type of 'None' (line 335)
        None_27875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'None')
        # Getting the type of 'self' (line 335)
        self_27876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 335)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), self_27876, 'rdt', None_27875)
        
        # Assigning a Name to a Attribute (line 336):
        
        # Assigning a Name to a Attribute (line 336):
        # Getting the type of 'None' (line 336)
        None_27877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 'None')
        # Getting the type of 'self' (line 336)
        self_27878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 336)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), self_27878, 'dec', None_27877)
        
        # Assigning a Name to a Attribute (line 337):
        
        # Assigning a Name to a Attribute (line 337):
        # Getting the type of 'None' (line 337)
        None_27879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'None')
        # Getting the type of 'self' (line 337)
        self_27880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'self')
        # Setting the type of the member 'type' of a type (line 337)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), self_27880, 'type', None_27879)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_27881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27881)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27881


    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 339, 4, False)
        # Assigning a type to the variable 'self' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _TestDSTBase.test_definition.__dict__.__setitem__('stypy_localization', localization)
        _TestDSTBase.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _TestDSTBase.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        _TestDSTBase.test_definition.__dict__.__setitem__('stypy_function_name', '_TestDSTBase.test_definition')
        _TestDSTBase.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        _TestDSTBase.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        _TestDSTBase.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _TestDSTBase.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        _TestDSTBase.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        _TestDSTBase.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _TestDSTBase.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestDSTBase.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        # Getting the type of 'FFTWDATA_SIZES' (line 340)
        FFTWDATA_SIZES_27882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 17), 'FFTWDATA_SIZES')
        # Testing the type of a for loop iterable (line 340)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 340, 8), FFTWDATA_SIZES_27882)
        # Getting the type of the for loop variable (line 340)
        for_loop_var_27883 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 340, 8), FFTWDATA_SIZES_27882)
        # Assigning a type to the variable 'i' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'i', for_loop_var_27883)
        # SSA begins for a for statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 341):
        
        # Assigning a Subscript to a Name (line 341):
        
        # Obtaining the type of the subscript
        int_27884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 12), 'int')
        
        # Call to fftw_dst_ref(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'self' (line 341)
        self_27886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 38), 'self', False)
        # Obtaining the member 'type' of a type (line 341)
        type_27887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 38), self_27886, 'type')
        # Getting the type of 'i' (line 341)
        i_27888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 49), 'i', False)
        # Getting the type of 'self' (line 341)
        self_27889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 52), 'self', False)
        # Obtaining the member 'rdt' of a type (line 341)
        rdt_27890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 52), self_27889, 'rdt')
        # Processing the call keyword arguments (line 341)
        kwargs_27891 = {}
        # Getting the type of 'fftw_dst_ref' (line 341)
        fftw_dst_ref_27885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'fftw_dst_ref', False)
        # Calling fftw_dst_ref(args, kwargs) (line 341)
        fftw_dst_ref_call_result_27892 = invoke(stypy.reporting.localization.Localization(__file__, 341, 25), fftw_dst_ref_27885, *[type_27887, i_27888, rdt_27890], **kwargs_27891)
        
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___27893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), fftw_dst_ref_call_result_27892, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_27894 = invoke(stypy.reporting.localization.Localization(__file__, 341, 12), getitem___27893, int_27884)
        
        # Assigning a type to the variable 'tuple_var_assignment_26782' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_26782', subscript_call_result_27894)
        
        # Assigning a Subscript to a Name (line 341):
        
        # Obtaining the type of the subscript
        int_27895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 12), 'int')
        
        # Call to fftw_dst_ref(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'self' (line 341)
        self_27897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 38), 'self', False)
        # Obtaining the member 'type' of a type (line 341)
        type_27898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 38), self_27897, 'type')
        # Getting the type of 'i' (line 341)
        i_27899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 49), 'i', False)
        # Getting the type of 'self' (line 341)
        self_27900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 52), 'self', False)
        # Obtaining the member 'rdt' of a type (line 341)
        rdt_27901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 52), self_27900, 'rdt')
        # Processing the call keyword arguments (line 341)
        kwargs_27902 = {}
        # Getting the type of 'fftw_dst_ref' (line 341)
        fftw_dst_ref_27896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'fftw_dst_ref', False)
        # Calling fftw_dst_ref(args, kwargs) (line 341)
        fftw_dst_ref_call_result_27903 = invoke(stypy.reporting.localization.Localization(__file__, 341, 25), fftw_dst_ref_27896, *[type_27898, i_27899, rdt_27901], **kwargs_27902)
        
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___27904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), fftw_dst_ref_call_result_27903, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_27905 = invoke(stypy.reporting.localization.Localization(__file__, 341, 12), getitem___27904, int_27895)
        
        # Assigning a type to the variable 'tuple_var_assignment_26783' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_26783', subscript_call_result_27905)
        
        # Assigning a Subscript to a Name (line 341):
        
        # Obtaining the type of the subscript
        int_27906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 12), 'int')
        
        # Call to fftw_dst_ref(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'self' (line 341)
        self_27908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 38), 'self', False)
        # Obtaining the member 'type' of a type (line 341)
        type_27909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 38), self_27908, 'type')
        # Getting the type of 'i' (line 341)
        i_27910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 49), 'i', False)
        # Getting the type of 'self' (line 341)
        self_27911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 52), 'self', False)
        # Obtaining the member 'rdt' of a type (line 341)
        rdt_27912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 52), self_27911, 'rdt')
        # Processing the call keyword arguments (line 341)
        kwargs_27913 = {}
        # Getting the type of 'fftw_dst_ref' (line 341)
        fftw_dst_ref_27907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'fftw_dst_ref', False)
        # Calling fftw_dst_ref(args, kwargs) (line 341)
        fftw_dst_ref_call_result_27914 = invoke(stypy.reporting.localization.Localization(__file__, 341, 25), fftw_dst_ref_27907, *[type_27909, i_27910, rdt_27912], **kwargs_27913)
        
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___27915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), fftw_dst_ref_call_result_27914, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_27916 = invoke(stypy.reporting.localization.Localization(__file__, 341, 12), getitem___27915, int_27906)
        
        # Assigning a type to the variable 'tuple_var_assignment_26784' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_26784', subscript_call_result_27916)
        
        # Assigning a Name to a Name (line 341):
        # Getting the type of 'tuple_var_assignment_26782' (line 341)
        tuple_var_assignment_26782_27917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_26782')
        # Assigning a type to the variable 'xr' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'xr', tuple_var_assignment_26782_27917)
        
        # Assigning a Name to a Name (line 341):
        # Getting the type of 'tuple_var_assignment_26783' (line 341)
        tuple_var_assignment_26783_27918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_26783')
        # Assigning a type to the variable 'yr' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'yr', tuple_var_assignment_26783_27918)
        
        # Assigning a Name to a Name (line 341):
        # Getting the type of 'tuple_var_assignment_26784' (line 341)
        tuple_var_assignment_26784_27919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'tuple_var_assignment_26784')
        # Assigning a type to the variable 'dt' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 20), 'dt', tuple_var_assignment_26784_27919)
        
        # Assigning a Call to a Name (line 342):
        
        # Assigning a Call to a Name (line 342):
        
        # Call to dst(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'xr' (line 342)
        xr_27921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'xr', False)
        # Processing the call keyword arguments (line 342)
        # Getting the type of 'self' (line 342)
        self_27922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 29), 'self', False)
        # Obtaining the member 'type' of a type (line 342)
        type_27923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 29), self_27922, 'type')
        keyword_27924 = type_27923
        kwargs_27925 = {'type': keyword_27924}
        # Getting the type of 'dst' (line 342)
        dst_27920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'dst', False)
        # Calling dst(args, kwargs) (line 342)
        dst_call_result_27926 = invoke(stypy.reporting.localization.Localization(__file__, 342, 16), dst_27920, *[xr_27921], **kwargs_27925)
        
        # Assigning a type to the variable 'y' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'y', dst_call_result_27926)
        
        # Call to assert_equal(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'y' (line 343)
        y_27928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'y', False)
        # Obtaining the member 'dtype' of a type (line 343)
        dtype_27929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 25), y_27928, 'dtype')
        # Getting the type of 'dt' (line 343)
        dt_27930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 34), 'dt', False)
        # Processing the call keyword arguments (line 343)
        kwargs_27931 = {}
        # Getting the type of 'assert_equal' (line 343)
        assert_equal_27927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 343)
        assert_equal_call_result_27932 = invoke(stypy.reporting.localization.Localization(__file__, 343, 12), assert_equal_27927, *[dtype_27929, dt_27930], **kwargs_27931)
        
        
        # Call to assert_array_almost_equal(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'y' (line 348)
        y_27934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 38), 'y', False)
        
        # Call to max(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'y' (line 348)
        y_27937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 49), 'y', False)
        # Processing the call keyword arguments (line 348)
        kwargs_27938 = {}
        # Getting the type of 'np' (line 348)
        np_27935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 42), 'np', False)
        # Obtaining the member 'max' of a type (line 348)
        max_27936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 42), np_27935, 'max')
        # Calling max(args, kwargs) (line 348)
        max_call_result_27939 = invoke(stypy.reporting.localization.Localization(__file__, 348, 42), max_27936, *[y_27937], **kwargs_27938)
        
        # Applying the binary operator 'div' (line 348)
        result_div_27940 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 38), 'div', y_27934, max_call_result_27939)
        
        # Getting the type of 'yr' (line 348)
        yr_27941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 53), 'yr', False)
        
        # Call to max(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'y' (line 348)
        y_27944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 65), 'y', False)
        # Processing the call keyword arguments (line 348)
        kwargs_27945 = {}
        # Getting the type of 'np' (line 348)
        np_27942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 58), 'np', False)
        # Obtaining the member 'max' of a type (line 348)
        max_27943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 58), np_27942, 'max')
        # Calling max(args, kwargs) (line 348)
        max_call_result_27946 = invoke(stypy.reporting.localization.Localization(__file__, 348, 58), max_27943, *[y_27944], **kwargs_27945)
        
        # Applying the binary operator 'div' (line 348)
        result_div_27947 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 53), 'div', yr_27941, max_call_result_27946)
        
        # Processing the call keyword arguments (line 348)
        # Getting the type of 'self' (line 348)
        self_27948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 77), 'self', False)
        # Obtaining the member 'dec' of a type (line 348)
        dec_27949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 77), self_27948, 'dec')
        keyword_27950 = dec_27949
        str_27951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 28), 'str', 'Size %d failed')
        # Getting the type of 'i' (line 349)
        i_27952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 47), 'i', False)
        # Applying the binary operator '%' (line 349)
        result_mod_27953 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 28), '%', str_27951, i_27952)
        
        keyword_27954 = result_mod_27953
        kwargs_27955 = {'decimal': keyword_27950, 'err_msg': keyword_27954}
        # Getting the type of 'assert_array_almost_equal' (line 348)
        assert_array_almost_equal_27933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 348)
        assert_array_almost_equal_call_result_27956 = invoke(stypy.reporting.localization.Localization(__file__, 348, 12), assert_array_almost_equal_27933, *[result_div_27940, result_div_27947], **kwargs_27955)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 339)
        stypy_return_type_27957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27957)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_27957


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 333, 0, False)
        # Assigning a type to the variable 'self' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestDSTBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_TestDSTBase' (line 333)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 0), '_TestDSTBase', _TestDSTBase)
# Declaration of the 'TestDSTIDouble' class
# Getting the type of '_TestDSTBase' (line 352)
_TestDSTBase_27958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 21), '_TestDSTBase')

class TestDSTIDouble(_TestDSTBase_27958, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 353, 4, False)
        # Assigning a type to the variable 'self' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDSTIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDSTIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDSTIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDSTIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDSTIDouble.setup_method')
        TestDSTIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDSTIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDSTIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDSTIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDSTIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDSTIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDSTIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 354):
        
        # Assigning a Attribute to a Attribute (line 354):
        # Getting the type of 'np' (line 354)
        np_27959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 19), 'np')
        # Obtaining the member 'double' of a type (line 354)
        double_27960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 19), np_27959, 'double')
        # Getting the type of 'self' (line 354)
        self_27961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 354)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), self_27961, 'rdt', double_27960)
        
        # Assigning a Num to a Attribute (line 355):
        
        # Assigning a Num to a Attribute (line 355):
        int_27962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 19), 'int')
        # Getting the type of 'self' (line 355)
        self_27963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 355)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), self_27963, 'dec', int_27962)
        
        # Assigning a Num to a Attribute (line 356):
        
        # Assigning a Num to a Attribute (line 356):
        int_27964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 20), 'int')
        # Getting the type of 'self' (line 356)
        self_27965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'self')
        # Setting the type of the member 'type' of a type (line 356)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), self_27965, 'type', int_27964)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 353)
        stypy_return_type_27966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27966)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27966


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 352, 0, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDSTIDouble' (line 352)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'TestDSTIDouble', TestDSTIDouble)
# Declaration of the 'TestDSTIFloat' class
# Getting the type of '_TestDSTBase' (line 359)
_TestDSTBase_27967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 20), '_TestDSTBase')

class TestDSTIFloat(_TestDSTBase_27967, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 360, 4, False)
        # Assigning a type to the variable 'self' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDSTIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDSTIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDSTIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDSTIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDSTIFloat.setup_method')
        TestDSTIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDSTIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDSTIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDSTIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDSTIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDSTIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDSTIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 361):
        
        # Assigning a Attribute to a Attribute (line 361):
        # Getting the type of 'np' (line 361)
        np_27968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 19), 'np')
        # Obtaining the member 'float32' of a type (line 361)
        float32_27969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 19), np_27968, 'float32')
        # Getting the type of 'self' (line 361)
        self_27970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 361)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), self_27970, 'rdt', float32_27969)
        
        # Assigning a Num to a Attribute (line 362):
        
        # Assigning a Num to a Attribute (line 362):
        int_27971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 19), 'int')
        # Getting the type of 'self' (line 362)
        self_27972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 362)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), self_27972, 'dec', int_27971)
        
        # Assigning a Num to a Attribute (line 363):
        
        # Assigning a Num to a Attribute (line 363):
        int_27973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 20), 'int')
        # Getting the type of 'self' (line 363)
        self_27974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'self')
        # Setting the type of the member 'type' of a type (line 363)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), self_27974, 'type', int_27973)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 360)
        stypy_return_type_27975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27975)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27975


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 359, 0, False)
        # Assigning a type to the variable 'self' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDSTIFloat' (line 359)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 0), 'TestDSTIFloat', TestDSTIFloat)
# Declaration of the 'TestDSTIInt' class
# Getting the type of '_TestDSTBase' (line 366)
_TestDSTBase_27976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 18), '_TestDSTBase')

class TestDSTIInt(_TestDSTBase_27976, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 367, 4, False)
        # Assigning a type to the variable 'self' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDSTIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDSTIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDSTIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDSTIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDSTIInt.setup_method')
        TestDSTIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDSTIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDSTIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDSTIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDSTIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDSTIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDSTIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 368):
        
        # Assigning a Name to a Attribute (line 368):
        # Getting the type of 'int' (line 368)
        int_27977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 19), 'int')
        # Getting the type of 'self' (line 368)
        self_27978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 368)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), self_27978, 'rdt', int_27977)
        
        # Assigning a Num to a Attribute (line 369):
        
        # Assigning a Num to a Attribute (line 369):
        int_27979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 19), 'int')
        # Getting the type of 'self' (line 369)
        self_27980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 369)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), self_27980, 'dec', int_27979)
        
        # Assigning a Num to a Attribute (line 370):
        
        # Assigning a Num to a Attribute (line 370):
        int_27981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 20), 'int')
        # Getting the type of 'self' (line 370)
        self_27982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'self')
        # Setting the type of the member 'type' of a type (line 370)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), self_27982, 'type', int_27981)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 367)
        stypy_return_type_27983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27983)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27983


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 366, 0, False)
        # Assigning a type to the variable 'self' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDSTIInt' (line 366)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 0), 'TestDSTIInt', TestDSTIInt)
# Declaration of the 'TestDSTIIDouble' class
# Getting the type of '_TestDSTBase' (line 373)
_TestDSTBase_27984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), '_TestDSTBase')

class TestDSTIIDouble(_TestDSTBase_27984, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 374, 4, False)
        # Assigning a type to the variable 'self' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDSTIIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDSTIIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDSTIIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDSTIIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDSTIIDouble.setup_method')
        TestDSTIIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDSTIIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDSTIIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDSTIIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDSTIIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDSTIIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDSTIIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 375):
        
        # Assigning a Attribute to a Attribute (line 375):
        # Getting the type of 'np' (line 375)
        np_27985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 19), 'np')
        # Obtaining the member 'double' of a type (line 375)
        double_27986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 19), np_27985, 'double')
        # Getting the type of 'self' (line 375)
        self_27987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 375)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 8), self_27987, 'rdt', double_27986)
        
        # Assigning a Num to a Attribute (line 376):
        
        # Assigning a Num to a Attribute (line 376):
        int_27988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 19), 'int')
        # Getting the type of 'self' (line 376)
        self_27989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 376)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), self_27989, 'dec', int_27988)
        
        # Assigning a Num to a Attribute (line 377):
        
        # Assigning a Num to a Attribute (line 377):
        int_27990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 20), 'int')
        # Getting the type of 'self' (line 377)
        self_27991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'self')
        # Setting the type of the member 'type' of a type (line 377)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), self_27991, 'type', int_27990)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 374)
        stypy_return_type_27992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27992)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_27992


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 373, 0, False)
        # Assigning a type to the variable 'self' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDSTIIDouble' (line 373)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 0), 'TestDSTIIDouble', TestDSTIIDouble)
# Declaration of the 'TestDSTIIFloat' class
# Getting the type of '_TestDSTBase' (line 380)
_TestDSTBase_27993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), '_TestDSTBase')

class TestDSTIIFloat(_TestDSTBase_27993, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 381, 4, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDSTIIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDSTIIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDSTIIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDSTIIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDSTIIFloat.setup_method')
        TestDSTIIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDSTIIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDSTIIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDSTIIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDSTIIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDSTIIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDSTIIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 382):
        
        # Assigning a Attribute to a Attribute (line 382):
        # Getting the type of 'np' (line 382)
        np_27994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 19), 'np')
        # Obtaining the member 'float32' of a type (line 382)
        float32_27995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 19), np_27994, 'float32')
        # Getting the type of 'self' (line 382)
        self_27996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), self_27996, 'rdt', float32_27995)
        
        # Assigning a Num to a Attribute (line 383):
        
        # Assigning a Num to a Attribute (line 383):
        int_27997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 19), 'int')
        # Getting the type of 'self' (line 383)
        self_27998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_27998, 'dec', int_27997)
        
        # Assigning a Num to a Attribute (line 384):
        
        # Assigning a Num to a Attribute (line 384):
        int_27999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 20), 'int')
        # Getting the type of 'self' (line 384)
        self_28000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'self')
        # Setting the type of the member 'type' of a type (line 384)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), self_28000, 'type', int_27999)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_28001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28001)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28001


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 380, 0, False)
        # Assigning a type to the variable 'self' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDSTIIFloat' (line 380)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), 'TestDSTIIFloat', TestDSTIIFloat)
# Declaration of the 'TestDSTIIInt' class
# Getting the type of '_TestDSTBase' (line 387)
_TestDSTBase_28002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 19), '_TestDSTBase')

class TestDSTIIInt(_TestDSTBase_28002, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 388, 4, False)
        # Assigning a type to the variable 'self' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDSTIIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDSTIIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDSTIIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDSTIIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDSTIIInt.setup_method')
        TestDSTIIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDSTIIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDSTIIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDSTIIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDSTIIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDSTIIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDSTIIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 389):
        
        # Assigning a Name to a Attribute (line 389):
        # Getting the type of 'int' (line 389)
        int_28003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'int')
        # Getting the type of 'self' (line 389)
        self_28004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 389)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_28004, 'rdt', int_28003)
        
        # Assigning a Num to a Attribute (line 390):
        
        # Assigning a Num to a Attribute (line 390):
        int_28005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 19), 'int')
        # Getting the type of 'self' (line 390)
        self_28006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 390)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_28006, 'dec', int_28005)
        
        # Assigning a Num to a Attribute (line 391):
        
        # Assigning a Num to a Attribute (line 391):
        int_28007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 20), 'int')
        # Getting the type of 'self' (line 391)
        self_28008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'self')
        # Setting the type of the member 'type' of a type (line 391)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), self_28008, 'type', int_28007)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 388)
        stypy_return_type_28009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28009)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28009


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 387, 0, False)
        # Assigning a type to the variable 'self' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDSTIIInt' (line 387)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 0), 'TestDSTIIInt', TestDSTIIInt)
# Declaration of the 'TestDSTIIIDouble' class
# Getting the type of '_TestDSTBase' (line 394)
_TestDSTBase_28010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 23), '_TestDSTBase')

class TestDSTIIIDouble(_TestDSTBase_28010, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 395, 4, False)
        # Assigning a type to the variable 'self' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDSTIIIDouble.setup_method')
        TestDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 396):
        
        # Assigning a Attribute to a Attribute (line 396):
        # Getting the type of 'np' (line 396)
        np_28011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'np')
        # Obtaining the member 'double' of a type (line 396)
        double_28012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 19), np_28011, 'double')
        # Getting the type of 'self' (line 396)
        self_28013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 396)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), self_28013, 'rdt', double_28012)
        
        # Assigning a Num to a Attribute (line 397):
        
        # Assigning a Num to a Attribute (line 397):
        int_28014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 19), 'int')
        # Getting the type of 'self' (line 397)
        self_28015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 397)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_28015, 'dec', int_28014)
        
        # Assigning a Num to a Attribute (line 398):
        
        # Assigning a Num to a Attribute (line 398):
        int_28016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 20), 'int')
        # Getting the type of 'self' (line 398)
        self_28017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'self')
        # Setting the type of the member 'type' of a type (line 398)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), self_28017, 'type', int_28016)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 395)
        stypy_return_type_28018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28018)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28018


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 394, 0, False)
        # Assigning a type to the variable 'self' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDSTIIIDouble' (line 394)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 0), 'TestDSTIIIDouble', TestDSTIIIDouble)
# Declaration of the 'TestDSTIIIFloat' class
# Getting the type of '_TestDSTBase' (line 401)
_TestDSTBase_28019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 22), '_TestDSTBase')

class TestDSTIIIFloat(_TestDSTBase_28019, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 402, 4, False)
        # Assigning a type to the variable 'self' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDSTIIIFloat.setup_method')
        TestDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 403):
        
        # Assigning a Attribute to a Attribute (line 403):
        # Getting the type of 'np' (line 403)
        np_28020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'np')
        # Obtaining the member 'float32' of a type (line 403)
        float32_28021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 19), np_28020, 'float32')
        # Getting the type of 'self' (line 403)
        self_28022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 403)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), self_28022, 'rdt', float32_28021)
        
        # Assigning a Num to a Attribute (line 404):
        
        # Assigning a Num to a Attribute (line 404):
        int_28023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 19), 'int')
        # Getting the type of 'self' (line 404)
        self_28024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 404)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), self_28024, 'dec', int_28023)
        
        # Assigning a Num to a Attribute (line 405):
        
        # Assigning a Num to a Attribute (line 405):
        int_28025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 20), 'int')
        # Getting the type of 'self' (line 405)
        self_28026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'self')
        # Setting the type of the member 'type' of a type (line 405)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), self_28026, 'type', int_28025)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 402)
        stypy_return_type_28027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28027)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28027


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 401, 0, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDSTIIIFloat' (line 401)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 0), 'TestDSTIIIFloat', TestDSTIIIFloat)
# Declaration of the 'TestDSTIIIInt' class
# Getting the type of '_TestDSTBase' (line 408)
_TestDSTBase_28028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 20), '_TestDSTBase')

class TestDSTIIIInt(_TestDSTBase_28028, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 409, 4, False)
        # Assigning a type to the variable 'self' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDSTIIIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDSTIIIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDSTIIIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDSTIIIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDSTIIIInt.setup_method')
        TestDSTIIIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDSTIIIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDSTIIIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDSTIIIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDSTIIIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDSTIIIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDSTIIIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 410):
        
        # Assigning a Name to a Attribute (line 410):
        # Getting the type of 'int' (line 410)
        int_28029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 19), 'int')
        # Getting the type of 'self' (line 410)
        self_28030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 410)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 8), self_28030, 'rdt', int_28029)
        
        # Assigning a Num to a Attribute (line 411):
        
        # Assigning a Num to a Attribute (line 411):
        int_28031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 19), 'int')
        # Getting the type of 'self' (line 411)
        self_28032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 411)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), self_28032, 'dec', int_28031)
        
        # Assigning a Num to a Attribute (line 412):
        
        # Assigning a Num to a Attribute (line 412):
        int_28033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 20), 'int')
        # Getting the type of 'self' (line 412)
        self_28034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'self')
        # Setting the type of the member 'type' of a type (line 412)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 8), self_28034, 'type', int_28033)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 409)
        stypy_return_type_28035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28035)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28035


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 408, 0, False)
        # Assigning a type to the variable 'self' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDSTIIIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDSTIIIInt' (line 408)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 0), 'TestDSTIIIInt', TestDSTIIIInt)
# Declaration of the '_TestIDSTBase' class

class _TestIDSTBase(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 416, 4, False)
        # Assigning a type to the variable 'self' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _TestIDSTBase.setup_method.__dict__.__setitem__('stypy_localization', localization)
        _TestIDSTBase.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _TestIDSTBase.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        _TestIDSTBase.setup_method.__dict__.__setitem__('stypy_function_name', '_TestIDSTBase.setup_method')
        _TestIDSTBase.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        _TestIDSTBase.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        _TestIDSTBase.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _TestIDSTBase.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        _TestIDSTBase.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        _TestIDSTBase.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _TestIDSTBase.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestIDSTBase.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 417):
        
        # Assigning a Name to a Attribute (line 417):
        # Getting the type of 'None' (line 417)
        None_28036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 19), 'None')
        # Getting the type of 'self' (line 417)
        self_28037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 417)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 8), self_28037, 'rdt', None_28036)
        
        # Assigning a Name to a Attribute (line 418):
        
        # Assigning a Name to a Attribute (line 418):
        # Getting the type of 'None' (line 418)
        None_28038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 19), 'None')
        # Getting the type of 'self' (line 418)
        self_28039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 418)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), self_28039, 'dec', None_28038)
        
        # Assigning a Name to a Attribute (line 419):
        
        # Assigning a Name to a Attribute (line 419):
        # Getting the type of 'None' (line 419)
        None_28040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 20), 'None')
        # Getting the type of 'self' (line 419)
        self_28041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'self')
        # Setting the type of the member 'type' of a type (line 419)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 8), self_28041, 'type', None_28040)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 416)
        stypy_return_type_28042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28042)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28042


    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 421, 4, False)
        # Assigning a type to the variable 'self' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _TestIDSTBase.test_definition.__dict__.__setitem__('stypy_localization', localization)
        _TestIDSTBase.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _TestIDSTBase.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        _TestIDSTBase.test_definition.__dict__.__setitem__('stypy_function_name', '_TestIDSTBase.test_definition')
        _TestIDSTBase.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        _TestIDSTBase.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        _TestIDSTBase.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _TestIDSTBase.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        _TestIDSTBase.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        _TestIDSTBase.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _TestIDSTBase.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestIDSTBase.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        # Getting the type of 'FFTWDATA_SIZES' (line 422)
        FFTWDATA_SIZES_28043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 17), 'FFTWDATA_SIZES')
        # Testing the type of a for loop iterable (line 422)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 422, 8), FFTWDATA_SIZES_28043)
        # Getting the type of the for loop variable (line 422)
        for_loop_var_28044 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 422, 8), FFTWDATA_SIZES_28043)
        # Assigning a type to the variable 'i' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'i', for_loop_var_28044)
        # SSA begins for a for statement (line 422)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 423):
        
        # Assigning a Subscript to a Name (line 423):
        
        # Obtaining the type of the subscript
        int_28045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 12), 'int')
        
        # Call to fftw_dst_ref(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'self' (line 423)
        self_28047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 38), 'self', False)
        # Obtaining the member 'type' of a type (line 423)
        type_28048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 38), self_28047, 'type')
        # Getting the type of 'i' (line 423)
        i_28049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 49), 'i', False)
        # Getting the type of 'self' (line 423)
        self_28050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 52), 'self', False)
        # Obtaining the member 'rdt' of a type (line 423)
        rdt_28051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 52), self_28050, 'rdt')
        # Processing the call keyword arguments (line 423)
        kwargs_28052 = {}
        # Getting the type of 'fftw_dst_ref' (line 423)
        fftw_dst_ref_28046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 25), 'fftw_dst_ref', False)
        # Calling fftw_dst_ref(args, kwargs) (line 423)
        fftw_dst_ref_call_result_28053 = invoke(stypy.reporting.localization.Localization(__file__, 423, 25), fftw_dst_ref_28046, *[type_28048, i_28049, rdt_28051], **kwargs_28052)
        
        # Obtaining the member '__getitem__' of a type (line 423)
        getitem___28054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), fftw_dst_ref_call_result_28053, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
        subscript_call_result_28055 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), getitem___28054, int_28045)
        
        # Assigning a type to the variable 'tuple_var_assignment_26785' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'tuple_var_assignment_26785', subscript_call_result_28055)
        
        # Assigning a Subscript to a Name (line 423):
        
        # Obtaining the type of the subscript
        int_28056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 12), 'int')
        
        # Call to fftw_dst_ref(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'self' (line 423)
        self_28058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 38), 'self', False)
        # Obtaining the member 'type' of a type (line 423)
        type_28059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 38), self_28058, 'type')
        # Getting the type of 'i' (line 423)
        i_28060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 49), 'i', False)
        # Getting the type of 'self' (line 423)
        self_28061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 52), 'self', False)
        # Obtaining the member 'rdt' of a type (line 423)
        rdt_28062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 52), self_28061, 'rdt')
        # Processing the call keyword arguments (line 423)
        kwargs_28063 = {}
        # Getting the type of 'fftw_dst_ref' (line 423)
        fftw_dst_ref_28057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 25), 'fftw_dst_ref', False)
        # Calling fftw_dst_ref(args, kwargs) (line 423)
        fftw_dst_ref_call_result_28064 = invoke(stypy.reporting.localization.Localization(__file__, 423, 25), fftw_dst_ref_28057, *[type_28059, i_28060, rdt_28062], **kwargs_28063)
        
        # Obtaining the member '__getitem__' of a type (line 423)
        getitem___28065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), fftw_dst_ref_call_result_28064, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
        subscript_call_result_28066 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), getitem___28065, int_28056)
        
        # Assigning a type to the variable 'tuple_var_assignment_26786' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'tuple_var_assignment_26786', subscript_call_result_28066)
        
        # Assigning a Subscript to a Name (line 423):
        
        # Obtaining the type of the subscript
        int_28067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 12), 'int')
        
        # Call to fftw_dst_ref(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'self' (line 423)
        self_28069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 38), 'self', False)
        # Obtaining the member 'type' of a type (line 423)
        type_28070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 38), self_28069, 'type')
        # Getting the type of 'i' (line 423)
        i_28071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 49), 'i', False)
        # Getting the type of 'self' (line 423)
        self_28072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 52), 'self', False)
        # Obtaining the member 'rdt' of a type (line 423)
        rdt_28073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 52), self_28072, 'rdt')
        # Processing the call keyword arguments (line 423)
        kwargs_28074 = {}
        # Getting the type of 'fftw_dst_ref' (line 423)
        fftw_dst_ref_28068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 25), 'fftw_dst_ref', False)
        # Calling fftw_dst_ref(args, kwargs) (line 423)
        fftw_dst_ref_call_result_28075 = invoke(stypy.reporting.localization.Localization(__file__, 423, 25), fftw_dst_ref_28068, *[type_28070, i_28071, rdt_28073], **kwargs_28074)
        
        # Obtaining the member '__getitem__' of a type (line 423)
        getitem___28076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), fftw_dst_ref_call_result_28075, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
        subscript_call_result_28077 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), getitem___28076, int_28067)
        
        # Assigning a type to the variable 'tuple_var_assignment_26787' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'tuple_var_assignment_26787', subscript_call_result_28077)
        
        # Assigning a Name to a Name (line 423):
        # Getting the type of 'tuple_var_assignment_26785' (line 423)
        tuple_var_assignment_26785_28078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'tuple_var_assignment_26785')
        # Assigning a type to the variable 'xr' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'xr', tuple_var_assignment_26785_28078)
        
        # Assigning a Name to a Name (line 423):
        # Getting the type of 'tuple_var_assignment_26786' (line 423)
        tuple_var_assignment_26786_28079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'tuple_var_assignment_26786')
        # Assigning a type to the variable 'yr' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'yr', tuple_var_assignment_26786_28079)
        
        # Assigning a Name to a Name (line 423):
        # Getting the type of 'tuple_var_assignment_26787' (line 423)
        tuple_var_assignment_26787_28080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'tuple_var_assignment_26787')
        # Assigning a type to the variable 'dt' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 20), 'dt', tuple_var_assignment_26787_28080)
        
        # Assigning a Call to a Name (line 424):
        
        # Assigning a Call to a Name (line 424):
        
        # Call to idst(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'yr' (line 424)
        yr_28082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 21), 'yr', False)
        # Processing the call keyword arguments (line 424)
        # Getting the type of 'self' (line 424)
        self_28083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 30), 'self', False)
        # Obtaining the member 'type' of a type (line 424)
        type_28084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 30), self_28083, 'type')
        keyword_28085 = type_28084
        kwargs_28086 = {'type': keyword_28085}
        # Getting the type of 'idst' (line 424)
        idst_28081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 16), 'idst', False)
        # Calling idst(args, kwargs) (line 424)
        idst_call_result_28087 = invoke(stypy.reporting.localization.Localization(__file__, 424, 16), idst_28081, *[yr_28082], **kwargs_28086)
        
        # Assigning a type to the variable 'x' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'x', idst_call_result_28087)
        
        
        # Getting the type of 'self' (line 425)
        self_28088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'self')
        # Obtaining the member 'type' of a type (line 425)
        type_28089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 15), self_28088, 'type')
        int_28090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 28), 'int')
        # Applying the binary operator '==' (line 425)
        result_eq_28091 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 15), '==', type_28089, int_28090)
        
        # Testing the type of an if condition (line 425)
        if_condition_28092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 12), result_eq_28091)
        # Assigning a type to the variable 'if_condition_28092' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'if_condition_28092', if_condition_28092)
        # SSA begins for if statement (line 425)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'x' (line 426)
        x_28093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 16), 'x')
        int_28094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 21), 'int')
        # Getting the type of 'i' (line 426)
        i_28095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 26), 'i')
        int_28096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 28), 'int')
        # Applying the binary operator '+' (line 426)
        result_add_28097 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 26), '+', i_28095, int_28096)
        
        # Applying the binary operator '*' (line 426)
        result_mul_28098 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 21), '*', int_28094, result_add_28097)
        
        # Applying the binary operator 'div=' (line 426)
        result_div_28099 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 16), 'div=', x_28093, result_mul_28098)
        # Assigning a type to the variable 'x' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 16), 'x', result_div_28099)
        
        # SSA branch for the else part of an if statement (line 425)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'x' (line 428)
        x_28100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 16), 'x')
        int_28101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 21), 'int')
        # Getting the type of 'i' (line 428)
        i_28102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 25), 'i')
        # Applying the binary operator '*' (line 428)
        result_mul_28103 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 21), '*', int_28101, i_28102)
        
        # Applying the binary operator 'div=' (line 428)
        result_div_28104 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 16), 'div=', x_28100, result_mul_28103)
        # Assigning a type to the variable 'x' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 16), 'x', result_div_28104)
        
        # SSA join for if statement (line 425)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'x' (line 429)
        x_28106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 25), 'x', False)
        # Obtaining the member 'dtype' of a type (line 429)
        dtype_28107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 25), x_28106, 'dtype')
        # Getting the type of 'dt' (line 429)
        dt_28108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 34), 'dt', False)
        # Processing the call keyword arguments (line 429)
        kwargs_28109 = {}
        # Getting the type of 'assert_equal' (line 429)
        assert_equal_28105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 429)
        assert_equal_call_result_28110 = invoke(stypy.reporting.localization.Localization(__file__, 429, 12), assert_equal_28105, *[dtype_28107, dt_28108], **kwargs_28109)
        
        
        # Call to assert_array_almost_equal(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'x' (line 434)
        x_28112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 38), 'x', False)
        
        # Call to max(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'x' (line 434)
        x_28115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 49), 'x', False)
        # Processing the call keyword arguments (line 434)
        kwargs_28116 = {}
        # Getting the type of 'np' (line 434)
        np_28113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 42), 'np', False)
        # Obtaining the member 'max' of a type (line 434)
        max_28114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 42), np_28113, 'max')
        # Calling max(args, kwargs) (line 434)
        max_call_result_28117 = invoke(stypy.reporting.localization.Localization(__file__, 434, 42), max_28114, *[x_28115], **kwargs_28116)
        
        # Applying the binary operator 'div' (line 434)
        result_div_28118 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 38), 'div', x_28112, max_call_result_28117)
        
        # Getting the type of 'xr' (line 434)
        xr_28119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 53), 'xr', False)
        
        # Call to max(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'x' (line 434)
        x_28122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 65), 'x', False)
        # Processing the call keyword arguments (line 434)
        kwargs_28123 = {}
        # Getting the type of 'np' (line 434)
        np_28120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 58), 'np', False)
        # Obtaining the member 'max' of a type (line 434)
        max_28121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 58), np_28120, 'max')
        # Calling max(args, kwargs) (line 434)
        max_call_result_28124 = invoke(stypy.reporting.localization.Localization(__file__, 434, 58), max_28121, *[x_28122], **kwargs_28123)
        
        # Applying the binary operator 'div' (line 434)
        result_div_28125 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 53), 'div', xr_28119, max_call_result_28124)
        
        # Processing the call keyword arguments (line 434)
        # Getting the type of 'self' (line 434)
        self_28126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 77), 'self', False)
        # Obtaining the member 'dec' of a type (line 434)
        dec_28127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 77), self_28126, 'dec')
        keyword_28128 = dec_28127
        str_28129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 28), 'str', 'Size %d failed')
        # Getting the type of 'i' (line 435)
        i_28130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 47), 'i', False)
        # Applying the binary operator '%' (line 435)
        result_mod_28131 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 28), '%', str_28129, i_28130)
        
        keyword_28132 = result_mod_28131
        kwargs_28133 = {'decimal': keyword_28128, 'err_msg': keyword_28132}
        # Getting the type of 'assert_array_almost_equal' (line 434)
        assert_array_almost_equal_28111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 434)
        assert_array_almost_equal_call_result_28134 = invoke(stypy.reporting.localization.Localization(__file__, 434, 12), assert_array_almost_equal_28111, *[result_div_28118, result_div_28125], **kwargs_28133)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 421)
        stypy_return_type_28135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28135)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_28135


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 415, 0, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_TestIDSTBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_TestIDSTBase' (line 415)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 0), '_TestIDSTBase', _TestIDSTBase)
# Declaration of the 'TestIDSTIDouble' class
# Getting the type of '_TestIDSTBase' (line 438)
_TestIDSTBase_28136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), '_TestIDSTBase')

class TestIDSTIDouble(_TestIDSTBase_28136, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 439, 4, False)
        # Assigning a type to the variable 'self' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDSTIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDSTIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDSTIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDSTIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDSTIDouble.setup_method')
        TestIDSTIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDSTIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDSTIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDSTIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDSTIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDSTIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDSTIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 440):
        
        # Assigning a Attribute to a Attribute (line 440):
        # Getting the type of 'np' (line 440)
        np_28137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 19), 'np')
        # Obtaining the member 'double' of a type (line 440)
        double_28138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 19), np_28137, 'double')
        # Getting the type of 'self' (line 440)
        self_28139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 440)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 8), self_28139, 'rdt', double_28138)
        
        # Assigning a Num to a Attribute (line 441):
        
        # Assigning a Num to a Attribute (line 441):
        int_28140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 19), 'int')
        # Getting the type of 'self' (line 441)
        self_28141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 441)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), self_28141, 'dec', int_28140)
        
        # Assigning a Num to a Attribute (line 442):
        
        # Assigning a Num to a Attribute (line 442):
        int_28142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 20), 'int')
        # Getting the type of 'self' (line 442)
        self_28143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'self')
        # Setting the type of the member 'type' of a type (line 442)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 8), self_28143, 'type', int_28142)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 439)
        stypy_return_type_28144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28144)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28144


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 438, 0, False)
        # Assigning a type to the variable 'self' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDSTIDouble' (line 438)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 0), 'TestIDSTIDouble', TestIDSTIDouble)
# Declaration of the 'TestIDSTIFloat' class
# Getting the type of '_TestIDSTBase' (line 445)
_TestIDSTBase_28145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), '_TestIDSTBase')

class TestIDSTIFloat(_TestIDSTBase_28145, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 446, 4, False)
        # Assigning a type to the variable 'self' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDSTIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDSTIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDSTIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDSTIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDSTIFloat.setup_method')
        TestIDSTIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDSTIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDSTIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDSTIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDSTIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDSTIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDSTIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 447):
        
        # Assigning a Attribute to a Attribute (line 447):
        # Getting the type of 'np' (line 447)
        np_28146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 19), 'np')
        # Obtaining the member 'float32' of a type (line 447)
        float32_28147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 19), np_28146, 'float32')
        # Getting the type of 'self' (line 447)
        self_28148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 447)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 8), self_28148, 'rdt', float32_28147)
        
        # Assigning a Num to a Attribute (line 448):
        
        # Assigning a Num to a Attribute (line 448):
        int_28149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 19), 'int')
        # Getting the type of 'self' (line 448)
        self_28150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 448)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), self_28150, 'dec', int_28149)
        
        # Assigning a Num to a Attribute (line 449):
        
        # Assigning a Num to a Attribute (line 449):
        int_28151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 20), 'int')
        # Getting the type of 'self' (line 449)
        self_28152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'self')
        # Setting the type of the member 'type' of a type (line 449)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), self_28152, 'type', int_28151)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 446)
        stypy_return_type_28153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28153)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28153


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 445, 0, False)
        # Assigning a type to the variable 'self' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDSTIFloat' (line 445)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 0), 'TestIDSTIFloat', TestIDSTIFloat)
# Declaration of the 'TestIDSTIInt' class
# Getting the type of '_TestIDSTBase' (line 452)
_TestIDSTBase_28154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 19), '_TestIDSTBase')

class TestIDSTIInt(_TestIDSTBase_28154, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 453, 4, False)
        # Assigning a type to the variable 'self' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDSTIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDSTIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDSTIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDSTIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDSTIInt.setup_method')
        TestIDSTIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDSTIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDSTIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDSTIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDSTIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDSTIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDSTIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 454):
        
        # Assigning a Name to a Attribute (line 454):
        # Getting the type of 'int' (line 454)
        int_28155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 19), 'int')
        # Getting the type of 'self' (line 454)
        self_28156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 454)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), self_28156, 'rdt', int_28155)
        
        # Assigning a Num to a Attribute (line 455):
        
        # Assigning a Num to a Attribute (line 455):
        int_28157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 19), 'int')
        # Getting the type of 'self' (line 455)
        self_28158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 455)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), self_28158, 'dec', int_28157)
        
        # Assigning a Num to a Attribute (line 456):
        
        # Assigning a Num to a Attribute (line 456):
        int_28159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 20), 'int')
        # Getting the type of 'self' (line 456)
        self_28160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'self')
        # Setting the type of the member 'type' of a type (line 456)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), self_28160, 'type', int_28159)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 453)
        stypy_return_type_28161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28161)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28161


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 452, 0, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDSTIInt' (line 452)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 0), 'TestIDSTIInt', TestIDSTIInt)
# Declaration of the 'TestIDSTIIDouble' class
# Getting the type of '_TestIDSTBase' (line 459)
_TestIDSTBase_28162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 23), '_TestIDSTBase')

class TestIDSTIIDouble(_TestIDSTBase_28162, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 460, 4, False)
        # Assigning a type to the variable 'self' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDSTIIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDSTIIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDSTIIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDSTIIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDSTIIDouble.setup_method')
        TestIDSTIIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDSTIIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDSTIIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDSTIIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDSTIIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDSTIIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDSTIIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 461):
        
        # Assigning a Attribute to a Attribute (line 461):
        # Getting the type of 'np' (line 461)
        np_28163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 'np')
        # Obtaining the member 'double' of a type (line 461)
        double_28164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 19), np_28163, 'double')
        # Getting the type of 'self' (line 461)
        self_28165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 461)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 8), self_28165, 'rdt', double_28164)
        
        # Assigning a Num to a Attribute (line 462):
        
        # Assigning a Num to a Attribute (line 462):
        int_28166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 19), 'int')
        # Getting the type of 'self' (line 462)
        self_28167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 462)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), self_28167, 'dec', int_28166)
        
        # Assigning a Num to a Attribute (line 463):
        
        # Assigning a Num to a Attribute (line 463):
        int_28168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 20), 'int')
        # Getting the type of 'self' (line 463)
        self_28169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'self')
        # Setting the type of the member 'type' of a type (line 463)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 8), self_28169, 'type', int_28168)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 460)
        stypy_return_type_28170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28170)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28170


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 459, 0, False)
        # Assigning a type to the variable 'self' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDSTIIDouble' (line 459)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 0), 'TestIDSTIIDouble', TestIDSTIIDouble)
# Declaration of the 'TestIDSTIIFloat' class
# Getting the type of '_TestIDSTBase' (line 466)
_TestIDSTBase_28171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 22), '_TestIDSTBase')

class TestIDSTIIFloat(_TestIDSTBase_28171, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 467, 4, False)
        # Assigning a type to the variable 'self' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDSTIIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDSTIIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDSTIIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDSTIIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDSTIIFloat.setup_method')
        TestIDSTIIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDSTIIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDSTIIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDSTIIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDSTIIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDSTIIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDSTIIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 468):
        
        # Assigning a Attribute to a Attribute (line 468):
        # Getting the type of 'np' (line 468)
        np_28172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 19), 'np')
        # Obtaining the member 'float32' of a type (line 468)
        float32_28173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 19), np_28172, 'float32')
        # Getting the type of 'self' (line 468)
        self_28174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 468)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), self_28174, 'rdt', float32_28173)
        
        # Assigning a Num to a Attribute (line 469):
        
        # Assigning a Num to a Attribute (line 469):
        int_28175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 19), 'int')
        # Getting the type of 'self' (line 469)
        self_28176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 469)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 8), self_28176, 'dec', int_28175)
        
        # Assigning a Num to a Attribute (line 470):
        
        # Assigning a Num to a Attribute (line 470):
        int_28177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 20), 'int')
        # Getting the type of 'self' (line 470)
        self_28178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'self')
        # Setting the type of the member 'type' of a type (line 470)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 8), self_28178, 'type', int_28177)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 467)
        stypy_return_type_28179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28179)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28179


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 466, 0, False)
        # Assigning a type to the variable 'self' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDSTIIFloat' (line 466)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'TestIDSTIIFloat', TestIDSTIIFloat)
# Declaration of the 'TestIDSTIIInt' class
# Getting the type of '_TestIDSTBase' (line 473)
_TestIDSTBase_28180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 20), '_TestIDSTBase')

class TestIDSTIIInt(_TestIDSTBase_28180, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 474, 4, False)
        # Assigning a type to the variable 'self' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDSTIIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDSTIIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDSTIIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDSTIIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDSTIIInt.setup_method')
        TestIDSTIIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDSTIIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDSTIIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDSTIIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDSTIIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDSTIIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDSTIIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 475):
        
        # Assigning a Name to a Attribute (line 475):
        # Getting the type of 'int' (line 475)
        int_28181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'int')
        # Getting the type of 'self' (line 475)
        self_28182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 475)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), self_28182, 'rdt', int_28181)
        
        # Assigning a Num to a Attribute (line 476):
        
        # Assigning a Num to a Attribute (line 476):
        int_28183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 19), 'int')
        # Getting the type of 'self' (line 476)
        self_28184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 476)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 8), self_28184, 'dec', int_28183)
        
        # Assigning a Num to a Attribute (line 477):
        
        # Assigning a Num to a Attribute (line 477):
        int_28185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 20), 'int')
        # Getting the type of 'self' (line 477)
        self_28186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'self')
        # Setting the type of the member 'type' of a type (line 477)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), self_28186, 'type', int_28185)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 474)
        stypy_return_type_28187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28187)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28187


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 473, 0, False)
        # Assigning a type to the variable 'self' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDSTIIInt' (line 473)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 0), 'TestIDSTIIInt', TestIDSTIIInt)
# Declaration of the 'TestIDSTIIIDouble' class
# Getting the type of '_TestIDSTBase' (line 480)
_TestIDSTBase_28188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 24), '_TestIDSTBase')

class TestIDSTIIIDouble(_TestIDSTBase_28188, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 481, 4, False)
        # Assigning a type to the variable 'self' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDSTIIIDouble.setup_method')
        TestIDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDSTIIIDouble.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIIDouble.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 482):
        
        # Assigning a Attribute to a Attribute (line 482):
        # Getting the type of 'np' (line 482)
        np_28189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 19), 'np')
        # Obtaining the member 'double' of a type (line 482)
        double_28190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 19), np_28189, 'double')
        # Getting the type of 'self' (line 482)
        self_28191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 482)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 8), self_28191, 'rdt', double_28190)
        
        # Assigning a Num to a Attribute (line 483):
        
        # Assigning a Num to a Attribute (line 483):
        int_28192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 19), 'int')
        # Getting the type of 'self' (line 483)
        self_28193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 483)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), self_28193, 'dec', int_28192)
        
        # Assigning a Num to a Attribute (line 484):
        
        # Assigning a Num to a Attribute (line 484):
        int_28194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 20), 'int')
        # Getting the type of 'self' (line 484)
        self_28195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'self')
        # Setting the type of the member 'type' of a type (line 484)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), self_28195, 'type', int_28194)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 481)
        stypy_return_type_28196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28196


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 480, 0, False)
        # Assigning a type to the variable 'self' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIIDouble.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDSTIIIDouble' (line 480)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 0), 'TestIDSTIIIDouble', TestIDSTIIIDouble)
# Declaration of the 'TestIDSTIIIFloat' class
# Getting the type of '_TestIDSTBase' (line 487)
_TestIDSTBase_28197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 23), '_TestIDSTBase')

class TestIDSTIIIFloat(_TestIDSTBase_28197, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 488, 4, False)
        # Assigning a type to the variable 'self' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDSTIIIFloat.setup_method')
        TestIDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDSTIIIFloat.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIIFloat.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 489):
        
        # Assigning a Attribute to a Attribute (line 489):
        # Getting the type of 'np' (line 489)
        np_28198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 19), 'np')
        # Obtaining the member 'float32' of a type (line 489)
        float32_28199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 19), np_28198, 'float32')
        # Getting the type of 'self' (line 489)
        self_28200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 489)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 8), self_28200, 'rdt', float32_28199)
        
        # Assigning a Num to a Attribute (line 490):
        
        # Assigning a Num to a Attribute (line 490):
        int_28201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 19), 'int')
        # Getting the type of 'self' (line 490)
        self_28202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 490)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), self_28202, 'dec', int_28201)
        
        # Assigning a Num to a Attribute (line 491):
        
        # Assigning a Num to a Attribute (line 491):
        int_28203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 20), 'int')
        # Getting the type of 'self' (line 491)
        self_28204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'self')
        # Setting the type of the member 'type' of a type (line 491)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 8), self_28204, 'type', int_28203)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 488)
        stypy_return_type_28205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28205)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28205


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 487, 0, False)
        # Assigning a type to the variable 'self' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIIFloat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDSTIIIFloat' (line 487)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 0), 'TestIDSTIIIFloat', TestIDSTIIIFloat)
# Declaration of the 'TestIDSTIIIInt' class
# Getting the type of '_TestIDSTBase' (line 494)
_TestIDSTBase_28206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 21), '_TestIDSTBase')

class TestIDSTIIIInt(_TestIDSTBase_28206, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 495, 4, False)
        # Assigning a type to the variable 'self' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIDSTIIIInt.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestIDSTIIIInt.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIDSTIIIInt.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIDSTIIIInt.setup_method.__dict__.__setitem__('stypy_function_name', 'TestIDSTIIIInt.setup_method')
        TestIDSTIIIInt.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestIDSTIIIInt.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIDSTIIIInt.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIDSTIIIInt.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIDSTIIIInt.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIDSTIIIInt.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIDSTIIIInt.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIIInt.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 496):
        
        # Assigning a Name to a Attribute (line 496):
        # Getting the type of 'int' (line 496)
        int_28207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 19), 'int')
        # Getting the type of 'self' (line 496)
        self_28208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'self')
        # Setting the type of the member 'rdt' of a type (line 496)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 8), self_28208, 'rdt', int_28207)
        
        # Assigning a Num to a Attribute (line 497):
        
        # Assigning a Num to a Attribute (line 497):
        int_28209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 19), 'int')
        # Getting the type of 'self' (line 497)
        self_28210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'self')
        # Setting the type of the member 'dec' of a type (line 497)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), self_28210, 'dec', int_28209)
        
        # Assigning a Num to a Attribute (line 498):
        
        # Assigning a Num to a Attribute (line 498):
        int_28211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 20), 'int')
        # Getting the type of 'self' (line 498)
        self_28212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'self')
        # Setting the type of the member 'type' of a type (line 498)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), self_28212, 'type', int_28211)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 495)
        stypy_return_type_28213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28213)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_28213


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 494, 0, False)
        # Assigning a type to the variable 'self' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIDSTIIIInt.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIDSTIIIInt' (line 494)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 0), 'TestIDSTIIIInt', TestIDSTIIIInt)
# Declaration of the 'TestOverwrite' class

class TestOverwrite(object, ):
    str_28214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 4), 'str', 'Check input overwrite behavior ')
    
    # Assigning a List to a Name (line 504):

    @norecursion
    def _check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check'
        module_type_store = module_type_store.open_function_context('_check', 506, 4, False)
        # Assigning a type to the variable 'self' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite._check.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite._check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite._check.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite._check.__dict__.__setitem__('stypy_function_name', 'TestOverwrite._check')
        TestOverwrite._check.__dict__.__setitem__('stypy_param_names_list', ['x', 'routine', 'type', 'fftsize', 'axis', 'norm', 'overwrite_x', 'should_overwrite'])
        TestOverwrite._check.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite._check.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        TestOverwrite._check.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite._check.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite._check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite._check.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite._check', ['x', 'routine', 'type', 'fftsize', 'axis', 'norm', 'overwrite_x', 'should_overwrite'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check', localization, ['x', 'routine', 'type', 'fftsize', 'axis', 'norm', 'overwrite_x', 'should_overwrite'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check(...)' code ##################

        
        # Assigning a Call to a Name (line 508):
        
        # Assigning a Call to a Name (line 508):
        
        # Call to copy(...): (line 508)
        # Processing the call keyword arguments (line 508)
        kwargs_28217 = {}
        # Getting the type of 'x' (line 508)
        x_28215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 13), 'x', False)
        # Obtaining the member 'copy' of a type (line 508)
        copy_28216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 13), x_28215, 'copy')
        # Calling copy(args, kwargs) (line 508)
        copy_call_result_28218 = invoke(stypy.reporting.localization.Localization(__file__, 508, 13), copy_28216, *[], **kwargs_28217)
        
        # Assigning a type to the variable 'x2' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'x2', copy_call_result_28218)
        
        # Call to routine(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'x2' (line 509)
        x2_28220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 'x2', False)
        # Getting the type of 'type' (line 509)
        type_28221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 20), 'type', False)
        # Getting the type of 'fftsize' (line 509)
        fftsize_28222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 26), 'fftsize', False)
        # Getting the type of 'axis' (line 509)
        axis_28223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 35), 'axis', False)
        # Getting the type of 'norm' (line 509)
        norm_28224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 41), 'norm', False)
        # Processing the call keyword arguments (line 509)
        # Getting the type of 'overwrite_x' (line 509)
        overwrite_x_28225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 59), 'overwrite_x', False)
        keyword_28226 = overwrite_x_28225
        kwargs_28227 = {'overwrite_x': keyword_28226}
        # Getting the type of 'routine' (line 509)
        routine_28219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'routine', False)
        # Calling routine(args, kwargs) (line 509)
        routine_call_result_28228 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), routine_28219, *[x2_28220, type_28221, fftsize_28222, axis_28223, norm_28224], **kwargs_28227)
        
        
        # Assigning a BinOp to a Name (line 511):
        
        # Assigning a BinOp to a Name (line 511):
        str_28229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 14), 'str', '%s(%s%r, %r, axis=%r, overwrite_x=%r)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 512)
        tuple_28230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 512)
        # Adding element type (line 512)
        # Getting the type of 'routine' (line 512)
        routine_28231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'routine')
        # Obtaining the member '__name__' of a type (line 512)
        name___28232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 12), routine_28231, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 12), tuple_28230, name___28232)
        # Adding element type (line 512)
        # Getting the type of 'x' (line 512)
        x_28233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 30), 'x')
        # Obtaining the member 'dtype' of a type (line 512)
        dtype_28234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 30), x_28233, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 12), tuple_28230, dtype_28234)
        # Adding element type (line 512)
        # Getting the type of 'x' (line 512)
        x_28235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 39), 'x')
        # Obtaining the member 'shape' of a type (line 512)
        shape_28236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 39), x_28235, 'shape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 12), tuple_28230, shape_28236)
        # Adding element type (line 512)
        # Getting the type of 'fftsize' (line 512)
        fftsize_28237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 48), 'fftsize')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 12), tuple_28230, fftsize_28237)
        # Adding element type (line 512)
        # Getting the type of 'axis' (line 512)
        axis_28238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 57), 'axis')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 12), tuple_28230, axis_28238)
        # Adding element type (line 512)
        # Getting the type of 'overwrite_x' (line 512)
        overwrite_x_28239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 63), 'overwrite_x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 12), tuple_28230, overwrite_x_28239)
        
        # Applying the binary operator '%' (line 511)
        result_mod_28240 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 14), '%', str_28229, tuple_28230)
        
        # Assigning a type to the variable 'sig' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'sig', result_mod_28240)
        
        
        # Getting the type of 'should_overwrite' (line 513)
        should_overwrite_28241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 'should_overwrite')
        # Applying the 'not' unary operator (line 513)
        result_not__28242 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 11), 'not', should_overwrite_28241)
        
        # Testing the type of an if condition (line 513)
        if_condition_28243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 513, 8), result_not__28242)
        # Assigning a type to the variable 'if_condition_28243' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'if_condition_28243', if_condition_28243)
        # SSA begins for if statement (line 513)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_equal(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'x2' (line 514)
        x2_28245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 25), 'x2', False)
        # Getting the type of 'x' (line 514)
        x_28246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 29), 'x', False)
        # Processing the call keyword arguments (line 514)
        str_28247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 40), 'str', 'spurious overwrite in %s')
        # Getting the type of 'sig' (line 514)
        sig_28248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 69), 'sig', False)
        # Applying the binary operator '%' (line 514)
        result_mod_28249 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 40), '%', str_28247, sig_28248)
        
        keyword_28250 = result_mod_28249
        kwargs_28251 = {'err_msg': keyword_28250}
        # Getting the type of 'assert_equal' (line 514)
        assert_equal_28244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 514)
        assert_equal_call_result_28252 = invoke(stypy.reporting.localization.Localization(__file__, 514, 12), assert_equal_28244, *[x2_28245, x_28246], **kwargs_28251)
        
        # SSA join for if statement (line 513)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check' in the type store
        # Getting the type of 'stypy_return_type' (line 506)
        stypy_return_type_28253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28253)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check'
        return stypy_return_type_28253


    @norecursion
    def _check_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_1d'
        module_type_store = module_type_store.open_function_context('_check_1d', 516, 4, False)
        # Assigning a type to the variable 'self' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_function_name', 'TestOverwrite._check_1d')
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_param_names_list', ['routine', 'dtype', 'shape', 'axis', 'overwritable_dtypes'])
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite._check_1d', ['routine', 'dtype', 'shape', 'axis', 'overwritable_dtypes'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_1d', localization, ['routine', 'dtype', 'shape', 'axis', 'overwritable_dtypes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_1d(...)' code ##################

        
        # Call to seed(...): (line 517)
        # Processing the call arguments (line 517)
        int_28257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 23), 'int')
        # Processing the call keyword arguments (line 517)
        kwargs_28258 = {}
        # Getting the type of 'np' (line 517)
        np_28254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 517)
        random_28255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 8), np_28254, 'random')
        # Obtaining the member 'seed' of a type (line 517)
        seed_28256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 8), random_28255, 'seed')
        # Calling seed(args, kwargs) (line 517)
        seed_call_result_28259 = invoke(stypy.reporting.localization.Localization(__file__, 517, 8), seed_28256, *[int_28257], **kwargs_28258)
        
        
        
        # Call to issubdtype(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'dtype' (line 518)
        dtype_28262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 25), 'dtype', False)
        # Getting the type of 'np' (line 518)
        np_28263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 32), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 518)
        complexfloating_28264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 32), np_28263, 'complexfloating')
        # Processing the call keyword arguments (line 518)
        kwargs_28265 = {}
        # Getting the type of 'np' (line 518)
        np_28260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 11), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 518)
        issubdtype_28261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 11), np_28260, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 518)
        issubdtype_call_result_28266 = invoke(stypy.reporting.localization.Localization(__file__, 518, 11), issubdtype_28261, *[dtype_28262, complexfloating_28264], **kwargs_28265)
        
        # Testing the type of an if condition (line 518)
        if_condition_28267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 518, 8), issubdtype_call_result_28266)
        # Assigning a type to the variable 'if_condition_28267' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'if_condition_28267', if_condition_28267)
        # SSA begins for if statement (line 518)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 519):
        
        # Assigning a BinOp to a Name (line 519):
        
        # Call to randn(...): (line 519)
        # Getting the type of 'shape' (line 519)
        shape_28271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 36), 'shape', False)
        # Processing the call keyword arguments (line 519)
        kwargs_28272 = {}
        # Getting the type of 'np' (line 519)
        np_28268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 19), 'np', False)
        # Obtaining the member 'random' of a type (line 519)
        random_28269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 19), np_28268, 'random')
        # Obtaining the member 'randn' of a type (line 519)
        randn_28270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 19), random_28269, 'randn')
        # Calling randn(args, kwargs) (line 519)
        randn_call_result_28273 = invoke(stypy.reporting.localization.Localization(__file__, 519, 19), randn_28270, *[shape_28271], **kwargs_28272)
        
        complex_28274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 45), 'complex')
        
        # Call to randn(...): (line 519)
        # Getting the type of 'shape' (line 519)
        shape_28278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 65), 'shape', False)
        # Processing the call keyword arguments (line 519)
        kwargs_28279 = {}
        # Getting the type of 'np' (line 519)
        np_28275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 48), 'np', False)
        # Obtaining the member 'random' of a type (line 519)
        random_28276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 48), np_28275, 'random')
        # Obtaining the member 'randn' of a type (line 519)
        randn_28277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 48), random_28276, 'randn')
        # Calling randn(args, kwargs) (line 519)
        randn_call_result_28280 = invoke(stypy.reporting.localization.Localization(__file__, 519, 48), randn_28277, *[shape_28278], **kwargs_28279)
        
        # Applying the binary operator '*' (line 519)
        result_mul_28281 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 45), '*', complex_28274, randn_call_result_28280)
        
        # Applying the binary operator '+' (line 519)
        result_add_28282 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 19), '+', randn_call_result_28273, result_mul_28281)
        
        # Assigning a type to the variable 'data' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'data', result_add_28282)
        # SSA branch for the else part of an if statement (line 518)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 521):
        
        # Assigning a Call to a Name (line 521):
        
        # Call to randn(...): (line 521)
        # Getting the type of 'shape' (line 521)
        shape_28286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 36), 'shape', False)
        # Processing the call keyword arguments (line 521)
        kwargs_28287 = {}
        # Getting the type of 'np' (line 521)
        np_28283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 19), 'np', False)
        # Obtaining the member 'random' of a type (line 521)
        random_28284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 19), np_28283, 'random')
        # Obtaining the member 'randn' of a type (line 521)
        randn_28285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 19), random_28284, 'randn')
        # Calling randn(args, kwargs) (line 521)
        randn_call_result_28288 = invoke(stypy.reporting.localization.Localization(__file__, 521, 19), randn_28285, *[shape_28286], **kwargs_28287)
        
        # Assigning a type to the variable 'data' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'data', randn_call_result_28288)
        # SSA join for if statement (line 518)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 522):
        
        # Assigning a Call to a Name (line 522):
        
        # Call to astype(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'dtype' (line 522)
        dtype_28291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 27), 'dtype', False)
        # Processing the call keyword arguments (line 522)
        kwargs_28292 = {}
        # Getting the type of 'data' (line 522)
        data_28289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'data', False)
        # Obtaining the member 'astype' of a type (line 522)
        astype_28290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 15), data_28289, 'astype')
        # Calling astype(args, kwargs) (line 522)
        astype_call_result_28293 = invoke(stypy.reporting.localization.Localization(__file__, 522, 15), astype_28290, *[dtype_28291], **kwargs_28292)
        
        # Assigning a type to the variable 'data' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'data', astype_call_result_28293)
        
        
        # Obtaining an instance of the builtin type 'list' (line 524)
        list_28294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 524)
        # Adding element type (line 524)
        int_28295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 20), list_28294, int_28295)
        # Adding element type (line 524)
        int_28296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 20), list_28294, int_28296)
        # Adding element type (line 524)
        int_28297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 20), list_28294, int_28297)
        
        # Testing the type of a for loop iterable (line 524)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 524, 8), list_28294)
        # Getting the type of the for loop variable (line 524)
        for_loop_var_28298 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 524, 8), list_28294)
        # Assigning a type to the variable 'type' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'type', for_loop_var_28298)
        # SSA begins for a for statement (line 524)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 525)
        list_28299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 525)
        # Adding element type (line 525)
        # Getting the type of 'True' (line 525)
        True_28300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 32), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 31), list_28299, True_28300)
        # Adding element type (line 525)
        # Getting the type of 'False' (line 525)
        False_28301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 38), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 31), list_28299, False_28301)
        
        # Testing the type of a for loop iterable (line 525)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 525, 12), list_28299)
        # Getting the type of the for loop variable (line 525)
        for_loop_var_28302 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 525, 12), list_28299)
        # Assigning a type to the variable 'overwrite_x' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'overwrite_x', for_loop_var_28302)
        # SSA begins for a for statement (line 525)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 526)
        list_28303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 526)
        # Adding element type (line 526)
        # Getting the type of 'None' (line 526)
        None_28304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 29), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 28), list_28303, None_28304)
        # Adding element type (line 526)
        str_28305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 35), 'str', 'ortho')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 28), list_28303, str_28305)
        
        # Testing the type of a for loop iterable (line 526)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 526, 16), list_28303)
        # Getting the type of the for loop variable (line 526)
        for_loop_var_28306 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 526, 16), list_28303)
        # Assigning a type to the variable 'norm' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 16), 'norm', for_loop_var_28306)
        # SSA begins for a for statement (line 526)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'type' (line 527)
        type_28307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 23), 'type')
        int_28308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 31), 'int')
        # Applying the binary operator '==' (line 527)
        result_eq_28309 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 23), '==', type_28307, int_28308)
        
        
        # Getting the type of 'norm' (line 527)
        norm_28310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 37), 'norm')
        str_28311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 45), 'str', 'ortho')
        # Applying the binary operator '==' (line 527)
        result_eq_28312 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 37), '==', norm_28310, str_28311)
        
        # Applying the binary operator 'and' (line 527)
        result_and_keyword_28313 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 23), 'and', result_eq_28309, result_eq_28312)
        
        # Testing the type of an if condition (line 527)
        if_condition_28314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 527, 20), result_and_keyword_28313)
        # Assigning a type to the variable 'if_condition_28314' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 20), 'if_condition_28314', if_condition_28314)
        # SSA begins for if statement (line 527)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 527)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 530):
        
        # Assigning a BoolOp to a Name (line 530):
        
        # Evaluating a boolean operation
        # Getting the type of 'overwrite_x' (line 530)
        overwrite_x_28315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 40), 'overwrite_x')
        
        # Getting the type of 'dtype' (line 531)
        dtype_28316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 44), 'dtype')
        # Getting the type of 'overwritable_dtypes' (line 531)
        overwritable_dtypes_28317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 53), 'overwritable_dtypes')
        # Applying the binary operator 'in' (line 531)
        result_contains_28318 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 44), 'in', dtype_28316, overwritable_dtypes_28317)
        
        # Applying the binary operator 'and' (line 530)
        result_and_keyword_28319 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 40), 'and', overwrite_x_28315, result_contains_28318)
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 'shape' (line 532)
        shape_28321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 49), 'shape', False)
        # Processing the call keyword arguments (line 532)
        kwargs_28322 = {}
        # Getting the type of 'len' (line 532)
        len_28320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 45), 'len', False)
        # Calling len(args, kwargs) (line 532)
        len_call_result_28323 = invoke(stypy.reporting.localization.Localization(__file__, 532, 45), len_28320, *[shape_28321], **kwargs_28322)
        
        int_28324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 59), 'int')
        # Applying the binary operator '==' (line 532)
        result_eq_28325 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 45), '==', len_call_result_28323, int_28324)
        
        
        # Getting the type of 'axis' (line 533)
        axis_28326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 46), 'axis')
        
        # Call to len(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'shape' (line 533)
        shape_28328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 57), 'shape', False)
        # Processing the call keyword arguments (line 533)
        kwargs_28329 = {}
        # Getting the type of 'len' (line 533)
        len_28327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 53), 'len', False)
        # Calling len(args, kwargs) (line 533)
        len_call_result_28330 = invoke(stypy.reporting.localization.Localization(__file__, 533, 53), len_28327, *[shape_28328], **kwargs_28329)
        
        # Applying the binary operator '%' (line 533)
        result_mod_28331 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 46), '%', axis_28326, len_call_result_28330)
        
        
        # Call to len(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'shape' (line 533)
        shape_28333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 71), 'shape', False)
        # Processing the call keyword arguments (line 533)
        kwargs_28334 = {}
        # Getting the type of 'len' (line 533)
        len_28332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 67), 'len', False)
        # Calling len(args, kwargs) (line 533)
        len_call_result_28335 = invoke(stypy.reporting.localization.Localization(__file__, 533, 67), len_28332, *[shape_28333], **kwargs_28334)
        
        int_28336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 78), 'int')
        # Applying the binary operator '-' (line 533)
        result_sub_28337 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 67), '-', len_call_result_28335, int_28336)
        
        # Applying the binary operator '==' (line 533)
        result_eq_28338 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 46), '==', result_mod_28331, result_sub_28337)
        
        # Applying the binary operator 'or' (line 532)
        result_or_keyword_28339 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 45), 'or', result_eq_28325, result_eq_28338)
        
        # Applying the binary operator 'and' (line 530)
        result_and_keyword_28340 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 40), 'and', result_and_keyword_28319, result_or_keyword_28339)
        
        # Assigning a type to the variable 'should_overwrite' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 20), 'should_overwrite', result_and_keyword_28340)
        
        # Call to _check(...): (line 535)
        # Processing the call arguments (line 535)
        # Getting the type of 'data' (line 535)
        data_28343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 32), 'data', False)
        # Getting the type of 'routine' (line 535)
        routine_28344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 38), 'routine', False)
        # Getting the type of 'type' (line 535)
        type_28345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 47), 'type', False)
        # Getting the type of 'None' (line 535)
        None_28346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 53), 'None', False)
        # Getting the type of 'axis' (line 535)
        axis_28347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 59), 'axis', False)
        # Getting the type of 'norm' (line 535)
        norm_28348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 65), 'norm', False)
        # Getting the type of 'overwrite_x' (line 536)
        overwrite_x_28349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 32), 'overwrite_x', False)
        # Getting the type of 'should_overwrite' (line 536)
        should_overwrite_28350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 45), 'should_overwrite', False)
        # Processing the call keyword arguments (line 535)
        kwargs_28351 = {}
        # Getting the type of 'self' (line 535)
        self_28341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 20), 'self', False)
        # Obtaining the member '_check' of a type (line 535)
        _check_28342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 20), self_28341, '_check')
        # Calling _check(args, kwargs) (line 535)
        _check_call_result_28352 = invoke(stypy.reporting.localization.Localization(__file__, 535, 20), _check_28342, *[data_28343, routine_28344, type_28345, None_28346, axis_28347, norm_28348, overwrite_x_28349, should_overwrite_28350], **kwargs_28351)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_check_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 516)
        stypy_return_type_28353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_1d'
        return stypy_return_type_28353


    @norecursion
    def test_dct(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dct'
        module_type_store = module_type_store.open_function_context('test_dct', 538, 4, False)
        # Assigning a type to the variable 'self' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_dct.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_dct.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_dct.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_dct.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_dct')
        TestOverwrite.test_dct.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_dct.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_dct.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_dct.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_dct.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_dct.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_dct.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_dct', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dct', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dct(...)' code ##################

        
        # Assigning a Attribute to a Name (line 539):
        
        # Assigning a Attribute to a Name (line 539):
        # Getting the type of 'self' (line 539)
        self_28354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 23), 'self')
        # Obtaining the member 'real_dtypes' of a type (line 539)
        real_dtypes_28355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 23), self_28354, 'real_dtypes')
        # Assigning a type to the variable 'overwritable' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'overwritable', real_dtypes_28355)
        
        # Getting the type of 'self' (line 540)
        self_28356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 21), 'self')
        # Obtaining the member 'real_dtypes' of a type (line 540)
        real_dtypes_28357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 21), self_28356, 'real_dtypes')
        # Testing the type of a for loop iterable (line 540)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 540, 8), real_dtypes_28357)
        # Getting the type of the for loop variable (line 540)
        for_loop_var_28358 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 540, 8), real_dtypes_28357)
        # Assigning a type to the variable 'dtype' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'dtype', for_loop_var_28358)
        # SSA begins for a for statement (line 540)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 541)
        # Processing the call arguments (line 541)
        # Getting the type of 'dct' (line 541)
        dct_28361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 27), 'dct', False)
        # Getting the type of 'dtype' (line 541)
        dtype_28362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 32), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 541)
        tuple_28363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 541)
        # Adding element type (line 541)
        int_28364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 40), tuple_28363, int_28364)
        
        int_28365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 46), 'int')
        # Getting the type of 'overwritable' (line 541)
        overwritable_28366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 50), 'overwritable', False)
        # Processing the call keyword arguments (line 541)
        kwargs_28367 = {}
        # Getting the type of 'self' (line 541)
        self_28359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 541)
        _check_1d_28360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 12), self_28359, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 541)
        _check_1d_call_result_28368 = invoke(stypy.reporting.localization.Localization(__file__, 541, 12), _check_1d_28360, *[dct_28361, dtype_28362, tuple_28363, int_28365, overwritable_28366], **kwargs_28367)
        
        
        # Call to _check_1d(...): (line 542)
        # Processing the call arguments (line 542)
        # Getting the type of 'dct' (line 542)
        dct_28371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 27), 'dct', False)
        # Getting the type of 'dtype' (line 542)
        dtype_28372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 32), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 542)
        tuple_28373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 542)
        # Adding element type (line 542)
        int_28374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 40), tuple_28373, int_28374)
        # Adding element type (line 542)
        int_28375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 40), tuple_28373, int_28375)
        
        int_28376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 48), 'int')
        # Getting the type of 'overwritable' (line 542)
        overwritable_28377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 51), 'overwritable', False)
        # Processing the call keyword arguments (line 542)
        kwargs_28378 = {}
        # Getting the type of 'self' (line 542)
        self_28369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 542)
        _check_1d_28370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 12), self_28369, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 542)
        _check_1d_call_result_28379 = invoke(stypy.reporting.localization.Localization(__file__, 542, 12), _check_1d_28370, *[dct_28371, dtype_28372, tuple_28373, int_28376, overwritable_28377], **kwargs_28378)
        
        
        # Call to _check_1d(...): (line 543)
        # Processing the call arguments (line 543)
        # Getting the type of 'dct' (line 543)
        dct_28382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 27), 'dct', False)
        # Getting the type of 'dtype' (line 543)
        dtype_28383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 32), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 543)
        tuple_28384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 543)
        # Adding element type (line 543)
        int_28385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 40), tuple_28384, int_28385)
        # Adding element type (line 543)
        int_28386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 40), tuple_28384, int_28386)
        
        int_28387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 48), 'int')
        # Getting the type of 'overwritable' (line 543)
        overwritable_28388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 51), 'overwritable', False)
        # Processing the call keyword arguments (line 543)
        kwargs_28389 = {}
        # Getting the type of 'self' (line 543)
        self_28380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 543)
        _check_1d_28381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 12), self_28380, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 543)
        _check_1d_call_result_28390 = invoke(stypy.reporting.localization.Localization(__file__, 543, 12), _check_1d_28381, *[dct_28382, dtype_28383, tuple_28384, int_28387, overwritable_28388], **kwargs_28389)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dct(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dct' in the type store
        # Getting the type of 'stypy_return_type' (line 538)
        stypy_return_type_28391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28391)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dct'
        return stypy_return_type_28391


    @norecursion
    def test_idct(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_idct'
        module_type_store = module_type_store.open_function_context('test_idct', 545, 4, False)
        # Assigning a type to the variable 'self' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_idct.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_idct.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_idct.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_idct.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_idct')
        TestOverwrite.test_idct.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_idct.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_idct.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_idct.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_idct.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_idct.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_idct.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_idct', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_idct', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_idct(...)' code ##################

        
        # Assigning a Attribute to a Name (line 546):
        
        # Assigning a Attribute to a Name (line 546):
        # Getting the type of 'self' (line 546)
        self_28392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 23), 'self')
        # Obtaining the member 'real_dtypes' of a type (line 546)
        real_dtypes_28393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 23), self_28392, 'real_dtypes')
        # Assigning a type to the variable 'overwritable' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'overwritable', real_dtypes_28393)
        
        # Getting the type of 'self' (line 547)
        self_28394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 21), 'self')
        # Obtaining the member 'real_dtypes' of a type (line 547)
        real_dtypes_28395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 21), self_28394, 'real_dtypes')
        # Testing the type of a for loop iterable (line 547)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 547, 8), real_dtypes_28395)
        # Getting the type of the for loop variable (line 547)
        for_loop_var_28396 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 547, 8), real_dtypes_28395)
        # Assigning a type to the variable 'dtype' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'dtype', for_loop_var_28396)
        # SSA begins for a for statement (line 547)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'idct' (line 548)
        idct_28399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 27), 'idct', False)
        # Getting the type of 'dtype' (line 548)
        dtype_28400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 33), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 548)
        tuple_28401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 548)
        # Adding element type (line 548)
        int_28402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 41), tuple_28401, int_28402)
        
        int_28403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 47), 'int')
        # Getting the type of 'overwritable' (line 548)
        overwritable_28404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 51), 'overwritable', False)
        # Processing the call keyword arguments (line 548)
        kwargs_28405 = {}
        # Getting the type of 'self' (line 548)
        self_28397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 548)
        _check_1d_28398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 12), self_28397, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 548)
        _check_1d_call_result_28406 = invoke(stypy.reporting.localization.Localization(__file__, 548, 12), _check_1d_28398, *[idct_28399, dtype_28400, tuple_28401, int_28403, overwritable_28404], **kwargs_28405)
        
        
        # Call to _check_1d(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'idct' (line 549)
        idct_28409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 27), 'idct', False)
        # Getting the type of 'dtype' (line 549)
        dtype_28410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 33), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 549)
        tuple_28411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 549)
        # Adding element type (line 549)
        int_28412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 41), tuple_28411, int_28412)
        # Adding element type (line 549)
        int_28413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 41), tuple_28411, int_28413)
        
        int_28414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 49), 'int')
        # Getting the type of 'overwritable' (line 549)
        overwritable_28415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 52), 'overwritable', False)
        # Processing the call keyword arguments (line 549)
        kwargs_28416 = {}
        # Getting the type of 'self' (line 549)
        self_28407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 549)
        _check_1d_28408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 12), self_28407, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 549)
        _check_1d_call_result_28417 = invoke(stypy.reporting.localization.Localization(__file__, 549, 12), _check_1d_28408, *[idct_28409, dtype_28410, tuple_28411, int_28414, overwritable_28415], **kwargs_28416)
        
        
        # Call to _check_1d(...): (line 550)
        # Processing the call arguments (line 550)
        # Getting the type of 'idct' (line 550)
        idct_28420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 27), 'idct', False)
        # Getting the type of 'dtype' (line 550)
        dtype_28421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 33), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 550)
        tuple_28422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 550)
        # Adding element type (line 550)
        int_28423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 41), tuple_28422, int_28423)
        # Adding element type (line 550)
        int_28424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 41), tuple_28422, int_28424)
        
        int_28425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 49), 'int')
        # Getting the type of 'overwritable' (line 550)
        overwritable_28426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 52), 'overwritable', False)
        # Processing the call keyword arguments (line 550)
        kwargs_28427 = {}
        # Getting the type of 'self' (line 550)
        self_28418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 550)
        _check_1d_28419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 12), self_28418, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 550)
        _check_1d_call_result_28428 = invoke(stypy.reporting.localization.Localization(__file__, 550, 12), _check_1d_28419, *[idct_28420, dtype_28421, tuple_28422, int_28425, overwritable_28426], **kwargs_28427)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_idct(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_idct' in the type store
        # Getting the type of 'stypy_return_type' (line 545)
        stypy_return_type_28429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28429)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_idct'
        return stypy_return_type_28429


    @norecursion
    def test_dst(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dst'
        module_type_store = module_type_store.open_function_context('test_dst', 552, 4, False)
        # Assigning a type to the variable 'self' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_dst.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_dst.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_dst.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_dst.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_dst')
        TestOverwrite.test_dst.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_dst.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_dst.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_dst.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_dst.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_dst.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_dst.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_dst', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dst', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dst(...)' code ##################

        
        # Assigning a Attribute to a Name (line 553):
        
        # Assigning a Attribute to a Name (line 553):
        # Getting the type of 'self' (line 553)
        self_28430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 23), 'self')
        # Obtaining the member 'real_dtypes' of a type (line 553)
        real_dtypes_28431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 23), self_28430, 'real_dtypes')
        # Assigning a type to the variable 'overwritable' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'overwritable', real_dtypes_28431)
        
        # Getting the type of 'self' (line 554)
        self_28432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 21), 'self')
        # Obtaining the member 'real_dtypes' of a type (line 554)
        real_dtypes_28433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 21), self_28432, 'real_dtypes')
        # Testing the type of a for loop iterable (line 554)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 554, 8), real_dtypes_28433)
        # Getting the type of the for loop variable (line 554)
        for_loop_var_28434 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 554, 8), real_dtypes_28433)
        # Assigning a type to the variable 'dtype' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'dtype', for_loop_var_28434)
        # SSA begins for a for statement (line 554)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 555)
        # Processing the call arguments (line 555)
        # Getting the type of 'dst' (line 555)
        dst_28437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 27), 'dst', False)
        # Getting the type of 'dtype' (line 555)
        dtype_28438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 32), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 555)
        tuple_28439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 555)
        # Adding element type (line 555)
        int_28440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 555, 40), tuple_28439, int_28440)
        
        int_28441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 46), 'int')
        # Getting the type of 'overwritable' (line 555)
        overwritable_28442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 50), 'overwritable', False)
        # Processing the call keyword arguments (line 555)
        kwargs_28443 = {}
        # Getting the type of 'self' (line 555)
        self_28435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 555)
        _check_1d_28436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 12), self_28435, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 555)
        _check_1d_call_result_28444 = invoke(stypy.reporting.localization.Localization(__file__, 555, 12), _check_1d_28436, *[dst_28437, dtype_28438, tuple_28439, int_28441, overwritable_28442], **kwargs_28443)
        
        
        # Call to _check_1d(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'dst' (line 556)
        dst_28447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 27), 'dst', False)
        # Getting the type of 'dtype' (line 556)
        dtype_28448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 32), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 556)
        tuple_28449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 556)
        # Adding element type (line 556)
        int_28450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 40), tuple_28449, int_28450)
        # Adding element type (line 556)
        int_28451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 40), tuple_28449, int_28451)
        
        int_28452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 48), 'int')
        # Getting the type of 'overwritable' (line 556)
        overwritable_28453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 51), 'overwritable', False)
        # Processing the call keyword arguments (line 556)
        kwargs_28454 = {}
        # Getting the type of 'self' (line 556)
        self_28445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 556)
        _check_1d_28446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 12), self_28445, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 556)
        _check_1d_call_result_28455 = invoke(stypy.reporting.localization.Localization(__file__, 556, 12), _check_1d_28446, *[dst_28447, dtype_28448, tuple_28449, int_28452, overwritable_28453], **kwargs_28454)
        
        
        # Call to _check_1d(...): (line 557)
        # Processing the call arguments (line 557)
        # Getting the type of 'dst' (line 557)
        dst_28458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 27), 'dst', False)
        # Getting the type of 'dtype' (line 557)
        dtype_28459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 32), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 557)
        tuple_28460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 557)
        # Adding element type (line 557)
        int_28461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 40), tuple_28460, int_28461)
        # Adding element type (line 557)
        int_28462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 40), tuple_28460, int_28462)
        
        int_28463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 48), 'int')
        # Getting the type of 'overwritable' (line 557)
        overwritable_28464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 51), 'overwritable', False)
        # Processing the call keyword arguments (line 557)
        kwargs_28465 = {}
        # Getting the type of 'self' (line 557)
        self_28456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 557)
        _check_1d_28457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 12), self_28456, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 557)
        _check_1d_call_result_28466 = invoke(stypy.reporting.localization.Localization(__file__, 557, 12), _check_1d_28457, *[dst_28458, dtype_28459, tuple_28460, int_28463, overwritable_28464], **kwargs_28465)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dst(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dst' in the type store
        # Getting the type of 'stypy_return_type' (line 552)
        stypy_return_type_28467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28467)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dst'
        return stypy_return_type_28467


    @norecursion
    def test_idst(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_idst'
        module_type_store = module_type_store.open_function_context('test_idst', 559, 4, False)
        # Assigning a type to the variable 'self' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_idst.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_idst.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_idst.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_idst.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_idst')
        TestOverwrite.test_idst.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_idst.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_idst.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_idst.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_idst.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_idst.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_idst.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_idst', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_idst', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_idst(...)' code ##################

        
        # Assigning a Attribute to a Name (line 560):
        
        # Assigning a Attribute to a Name (line 560):
        # Getting the type of 'self' (line 560)
        self_28468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 23), 'self')
        # Obtaining the member 'real_dtypes' of a type (line 560)
        real_dtypes_28469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 23), self_28468, 'real_dtypes')
        # Assigning a type to the variable 'overwritable' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'overwritable', real_dtypes_28469)
        
        # Getting the type of 'self' (line 561)
        self_28470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 21), 'self')
        # Obtaining the member 'real_dtypes' of a type (line 561)
        real_dtypes_28471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 21), self_28470, 'real_dtypes')
        # Testing the type of a for loop iterable (line 561)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 561, 8), real_dtypes_28471)
        # Getting the type of the for loop variable (line 561)
        for_loop_var_28472 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 561, 8), real_dtypes_28471)
        # Assigning a type to the variable 'dtype' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'dtype', for_loop_var_28472)
        # SSA begins for a for statement (line 561)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'idst' (line 562)
        idst_28475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 27), 'idst', False)
        # Getting the type of 'dtype' (line 562)
        dtype_28476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 33), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 562)
        tuple_28477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 562)
        # Adding element type (line 562)
        int_28478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 41), tuple_28477, int_28478)
        
        int_28479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 47), 'int')
        # Getting the type of 'overwritable' (line 562)
        overwritable_28480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 51), 'overwritable', False)
        # Processing the call keyword arguments (line 562)
        kwargs_28481 = {}
        # Getting the type of 'self' (line 562)
        self_28473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 562)
        _check_1d_28474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 12), self_28473, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 562)
        _check_1d_call_result_28482 = invoke(stypy.reporting.localization.Localization(__file__, 562, 12), _check_1d_28474, *[idst_28475, dtype_28476, tuple_28477, int_28479, overwritable_28480], **kwargs_28481)
        
        
        # Call to _check_1d(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'idst' (line 563)
        idst_28485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 27), 'idst', False)
        # Getting the type of 'dtype' (line 563)
        dtype_28486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 33), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 563)
        tuple_28487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 563)
        # Adding element type (line 563)
        int_28488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 41), tuple_28487, int_28488)
        # Adding element type (line 563)
        int_28489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 41), tuple_28487, int_28489)
        
        int_28490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 49), 'int')
        # Getting the type of 'overwritable' (line 563)
        overwritable_28491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 52), 'overwritable', False)
        # Processing the call keyword arguments (line 563)
        kwargs_28492 = {}
        # Getting the type of 'self' (line 563)
        self_28483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 563)
        _check_1d_28484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 12), self_28483, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 563)
        _check_1d_call_result_28493 = invoke(stypy.reporting.localization.Localization(__file__, 563, 12), _check_1d_28484, *[idst_28485, dtype_28486, tuple_28487, int_28490, overwritable_28491], **kwargs_28492)
        
        
        # Call to _check_1d(...): (line 564)
        # Processing the call arguments (line 564)
        # Getting the type of 'idst' (line 564)
        idst_28496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 27), 'idst', False)
        # Getting the type of 'dtype' (line 564)
        dtype_28497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 33), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 564)
        tuple_28498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 564)
        # Adding element type (line 564)
        int_28499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 41), tuple_28498, int_28499)
        # Adding element type (line 564)
        int_28500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 41), tuple_28498, int_28500)
        
        int_28501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 49), 'int')
        # Getting the type of 'overwritable' (line 564)
        overwritable_28502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 52), 'overwritable', False)
        # Processing the call keyword arguments (line 564)
        kwargs_28503 = {}
        # Getting the type of 'self' (line 564)
        self_28494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 564)
        _check_1d_28495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 12), self_28494, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 564)
        _check_1d_call_result_28504 = invoke(stypy.reporting.localization.Localization(__file__, 564, 12), _check_1d_28495, *[idst_28496, dtype_28497, tuple_28498, int_28501, overwritable_28502], **kwargs_28503)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_idst(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_idst' in the type store
        # Getting the type of 'stypy_return_type' (line 559)
        stypy_return_type_28505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28505)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_idst'
        return stypy_return_type_28505


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 501, 0, False)
        # Assigning a type to the variable 'self' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestOverwrite' (line 501)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 0), 'TestOverwrite', TestOverwrite)

# Assigning a List to a Name (line 504):

# Obtaining an instance of the builtin type 'list' (line 504)
list_28506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 504)
# Adding element type (line 504)
# Getting the type of 'np' (line 504)
np_28507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 19), 'np')
# Obtaining the member 'float32' of a type (line 504)
float32_28508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 19), np_28507, 'float32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 18), list_28506, float32_28508)
# Adding element type (line 504)
# Getting the type of 'np' (line 504)
np_28509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 31), 'np')
# Obtaining the member 'float64' of a type (line 504)
float64_28510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 31), np_28509, 'float64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 18), list_28506, float64_28510)

# Getting the type of 'TestOverwrite'
TestOverwrite_28511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestOverwrite')
# Setting the type of the member 'real_dtypes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestOverwrite_28511, 'real_dtypes', list_28506)
# Declaration of the 'Test_DCTN_IDCTN' class

class Test_DCTN_IDCTN(object, ):
    
    # Assigning a Num to a Name (line 568):
    
    # Assigning a List to a Name (line 569):
    
    # Assigning a List to a Name (line 570):
    
    # Assigning a Call to a Name (line 571):
    
    # Assigning a Tuple to a Name (line 572):
    
    # Assigning a Call to a Name (line 573):
    
    # Assigning a List to a Name (line 575):

    @norecursion
    def test_axes_round_trip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_axes_round_trip'
        module_type_store = module_type_store.open_function_context('test_axes_round_trip', 584, 4, False)
        # Assigning a type to the variable 'self' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_DCTN_IDCTN.test_axes_round_trip.__dict__.__setitem__('stypy_localization', localization)
        Test_DCTN_IDCTN.test_axes_round_trip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_DCTN_IDCTN.test_axes_round_trip.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_DCTN_IDCTN.test_axes_round_trip.__dict__.__setitem__('stypy_function_name', 'Test_DCTN_IDCTN.test_axes_round_trip')
        Test_DCTN_IDCTN.test_axes_round_trip.__dict__.__setitem__('stypy_param_names_list', [])
        Test_DCTN_IDCTN.test_axes_round_trip.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_DCTN_IDCTN.test_axes_round_trip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_DCTN_IDCTN.test_axes_round_trip.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_DCTN_IDCTN.test_axes_round_trip.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_DCTN_IDCTN.test_axes_round_trip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_DCTN_IDCTN.test_axes_round_trip.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_DCTN_IDCTN.test_axes_round_trip', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_axes_round_trip', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_axes_round_trip(...)' code ##################

        
        # Assigning a Str to a Name (line 585):
        
        # Assigning a Str to a Name (line 585):
        str_28512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 15), 'str', 'ortho')
        # Assigning a type to the variable 'norm' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'norm', str_28512)
        
        # Getting the type of 'self' (line 586)
        self_28513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 28), 'self')
        # Obtaining the member 'function_sets' of a type (line 586)
        function_sets_28514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 28), self_28513, 'function_sets')
        # Testing the type of a for loop iterable (line 586)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 586, 8), function_sets_28514)
        # Getting the type of the for loop variable (line 586)
        for_loop_var_28515 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 586, 8), function_sets_28514)
        # Assigning a type to the variable 'function_set' (line 586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'function_set', for_loop_var_28515)
        # SSA begins for a for statement (line 586)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 587):
        
        # Assigning a Subscript to a Name (line 587):
        
        # Obtaining the type of the subscript
        str_28516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 36), 'str', 'forward')
        # Getting the type of 'function_set' (line 587)
        function_set_28517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 23), 'function_set')
        # Obtaining the member '__getitem__' of a type (line 587)
        getitem___28518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 23), function_set_28517, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 587)
        subscript_call_result_28519 = invoke(stypy.reporting.localization.Localization(__file__, 587, 23), getitem___28518, str_28516)
        
        # Assigning a type to the variable 'fforward' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'fforward', subscript_call_result_28519)
        
        # Assigning a Subscript to a Name (line 588):
        
        # Assigning a Subscript to a Name (line 588):
        
        # Obtaining the type of the subscript
        str_28520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 36), 'str', 'inverse')
        # Getting the type of 'function_set' (line 588)
        function_set_28521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 23), 'function_set')
        # Obtaining the member '__getitem__' of a type (line 588)
        getitem___28522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 23), function_set_28521, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 588)
        subscript_call_result_28523 = invoke(stypy.reporting.localization.Localization(__file__, 588, 23), getitem___28522, str_28520)
        
        # Assigning a type to the variable 'finverse' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'finverse', subscript_call_result_28523)
        
        
        # Obtaining an instance of the builtin type 'list' (line 589)
        list_28524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 589)
        # Adding element type (line 589)
        # Getting the type of 'None' (line 589)
        None_28525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 25), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 24), list_28524, None_28525)
        # Adding element type (line 589)
        
        # Obtaining an instance of the builtin type 'tuple' (line 589)
        tuple_28526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 589)
        # Adding element type (line 589)
        int_28527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 32), tuple_28526, int_28527)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 24), list_28524, tuple_28526)
        # Adding element type (line 589)
        
        # Obtaining an instance of the builtin type 'tuple' (line 589)
        tuple_28528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 589)
        # Adding element type (line 589)
        int_28529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 39), tuple_28528, int_28529)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 24), list_28524, tuple_28528)
        # Adding element type (line 589)
        
        # Obtaining an instance of the builtin type 'tuple' (line 589)
        tuple_28530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 589)
        # Adding element type (line 589)
        int_28531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 46), tuple_28530, int_28531)
        # Adding element type (line 589)
        int_28532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 46), tuple_28530, int_28532)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 24), list_28524, tuple_28530)
        # Adding element type (line 589)
        
        # Obtaining an instance of the builtin type 'tuple' (line 589)
        tuple_28533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 589)
        # Adding element type (line 589)
        int_28534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 54), tuple_28533, int_28534)
        # Adding element type (line 589)
        int_28535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 54), tuple_28533, int_28535)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 24), list_28524, tuple_28533)
        
        # Testing the type of a for loop iterable (line 589)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 589, 12), list_28524)
        # Getting the type of the for loop variable (line 589)
        for_loop_var_28536 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 589, 12), list_28524)
        # Assigning a type to the variable 'axes' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'axes', for_loop_var_28536)
        # SSA begins for a for statement (line 589)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 590)
        self_28537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 32), 'self')
        # Obtaining the member 'types' of a type (line 590)
        types_28538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 32), self_28537, 'types')
        # Testing the type of a for loop iterable (line 590)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 590, 16), types_28538)
        # Getting the type of the for loop variable (line 590)
        for_loop_var_28539 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 590, 16), types_28538)
        # Assigning a type to the variable 'dct_type' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 16), 'dct_type', for_loop_var_28539)
        # SSA begins for a for statement (line 590)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'norm' (line 591)
        norm_28540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 23), 'norm')
        str_28541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 31), 'str', 'ortho')
        # Applying the binary operator '==' (line 591)
        result_eq_28542 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 23), '==', norm_28540, str_28541)
        
        
        # Getting the type of 'dct_type' (line 591)
        dct_type_28543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 43), 'dct_type')
        int_28544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 55), 'int')
        # Applying the binary operator '==' (line 591)
        result_eq_28545 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 43), '==', dct_type_28543, int_28544)
        
        # Applying the binary operator 'and' (line 591)
        result_and_keyword_28546 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 23), 'and', result_eq_28542, result_eq_28545)
        
        # Testing the type of an if condition (line 591)
        if_condition_28547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 591, 20), result_and_keyword_28546)
        # Assigning a type to the variable 'if_condition_28547' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 20), 'if_condition_28547', if_condition_28547)
        # SSA begins for if statement (line 591)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 591)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 593):
        
        # Assigning a Call to a Name (line 593):
        
        # Call to fforward(...): (line 593)
        # Processing the call arguments (line 593)
        # Getting the type of 'self' (line 593)
        self_28549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 35), 'self', False)
        # Obtaining the member 'data' of a type (line 593)
        data_28550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 35), self_28549, 'data')
        # Processing the call keyword arguments (line 593)
        # Getting the type of 'dct_type' (line 593)
        dct_type_28551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 51), 'dct_type', False)
        keyword_28552 = dct_type_28551
        # Getting the type of 'axes' (line 593)
        axes_28553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 66), 'axes', False)
        keyword_28554 = axes_28553
        # Getting the type of 'norm' (line 594)
        norm_28555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 40), 'norm', False)
        keyword_28556 = norm_28555
        kwargs_28557 = {'axes': keyword_28554, 'type': keyword_28552, 'norm': keyword_28556}
        # Getting the type of 'fforward' (line 593)
        fforward_28548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 26), 'fforward', False)
        # Calling fforward(args, kwargs) (line 593)
        fforward_call_result_28558 = invoke(stypy.reporting.localization.Localization(__file__, 593, 26), fforward_28548, *[data_28550], **kwargs_28557)
        
        # Assigning a type to the variable 'tmp' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 20), 'tmp', fforward_call_result_28558)
        
        # Assigning a Call to a Name (line 595):
        
        # Assigning a Call to a Name (line 595):
        
        # Call to finverse(...): (line 595)
        # Processing the call arguments (line 595)
        # Getting the type of 'tmp' (line 595)
        tmp_28560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 35), 'tmp', False)
        # Processing the call keyword arguments (line 595)
        # Getting the type of 'dct_type' (line 595)
        dct_type_28561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 45), 'dct_type', False)
        keyword_28562 = dct_type_28561
        # Getting the type of 'axes' (line 595)
        axes_28563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 60), 'axes', False)
        keyword_28564 = axes_28563
        # Getting the type of 'norm' (line 595)
        norm_28565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 71), 'norm', False)
        keyword_28566 = norm_28565
        kwargs_28567 = {'axes': keyword_28564, 'type': keyword_28562, 'norm': keyword_28566}
        # Getting the type of 'finverse' (line 595)
        finverse_28559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 26), 'finverse', False)
        # Calling finverse(args, kwargs) (line 595)
        finverse_call_result_28568 = invoke(stypy.reporting.localization.Localization(__file__, 595, 26), finverse_28559, *[tmp_28560], **kwargs_28567)
        
        # Assigning a type to the variable 'tmp' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 20), 'tmp', finverse_call_result_28568)
        
        # Call to assert_array_almost_equal(...): (line 596)
        # Processing the call arguments (line 596)
        # Getting the type of 'self' (line 596)
        self_28570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 46), 'self', False)
        # Obtaining the member 'data' of a type (line 596)
        data_28571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 46), self_28570, 'data')
        # Getting the type of 'tmp' (line 596)
        tmp_28572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 57), 'tmp', False)
        # Processing the call keyword arguments (line 596)
        # Getting the type of 'self' (line 596)
        self_28573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 70), 'self', False)
        # Obtaining the member 'dec' of a type (line 596)
        dec_28574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 70), self_28573, 'dec')
        keyword_28575 = dec_28574
        kwargs_28576 = {'decimal': keyword_28575}
        # Getting the type of 'assert_array_almost_equal' (line 596)
        assert_array_almost_equal_28569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 20), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 596)
        assert_array_almost_equal_call_result_28577 = invoke(stypy.reporting.localization.Localization(__file__, 596, 20), assert_array_almost_equal_28569, *[data_28571, tmp_28572], **kwargs_28576)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_axes_round_trip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_axes_round_trip' in the type store
        # Getting the type of 'stypy_return_type' (line 584)
        stypy_return_type_28578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28578)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_axes_round_trip'
        return stypy_return_type_28578


    @norecursion
    def test_dctn_vs_2d_reference(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dctn_vs_2d_reference'
        module_type_store = module_type_store.open_function_context('test_dctn_vs_2d_reference', 598, 4, False)
        # Assigning a type to the variable 'self' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_DCTN_IDCTN.test_dctn_vs_2d_reference.__dict__.__setitem__('stypy_localization', localization)
        Test_DCTN_IDCTN.test_dctn_vs_2d_reference.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_DCTN_IDCTN.test_dctn_vs_2d_reference.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_DCTN_IDCTN.test_dctn_vs_2d_reference.__dict__.__setitem__('stypy_function_name', 'Test_DCTN_IDCTN.test_dctn_vs_2d_reference')
        Test_DCTN_IDCTN.test_dctn_vs_2d_reference.__dict__.__setitem__('stypy_param_names_list', [])
        Test_DCTN_IDCTN.test_dctn_vs_2d_reference.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_DCTN_IDCTN.test_dctn_vs_2d_reference.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_DCTN_IDCTN.test_dctn_vs_2d_reference.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_DCTN_IDCTN.test_dctn_vs_2d_reference.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_DCTN_IDCTN.test_dctn_vs_2d_reference.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_DCTN_IDCTN.test_dctn_vs_2d_reference.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_DCTN_IDCTN.test_dctn_vs_2d_reference', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dctn_vs_2d_reference', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dctn_vs_2d_reference(...)' code ##################

        
        # Getting the type of 'self' (line 599)
        self_28579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 28), 'self')
        # Obtaining the member 'function_sets' of a type (line 599)
        function_sets_28580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 28), self_28579, 'function_sets')
        # Testing the type of a for loop iterable (line 599)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 599, 8), function_sets_28580)
        # Getting the type of the for loop variable (line 599)
        for_loop_var_28581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 599, 8), function_sets_28580)
        # Assigning a type to the variable 'function_set' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'function_set', for_loop_var_28581)
        # SSA begins for a for statement (line 599)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 600):
        
        # Assigning a Subscript to a Name (line 600):
        
        # Obtaining the type of the subscript
        str_28582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 36), 'str', 'forward')
        # Getting the type of 'function_set' (line 600)
        function_set_28583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 23), 'function_set')
        # Obtaining the member '__getitem__' of a type (line 600)
        getitem___28584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 23), function_set_28583, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 600)
        subscript_call_result_28585 = invoke(stypy.reporting.localization.Localization(__file__, 600, 23), getitem___28584, str_28582)
        
        # Assigning a type to the variable 'fforward' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'fforward', subscript_call_result_28585)
        
        # Assigning a Subscript to a Name (line 601):
        
        # Assigning a Subscript to a Name (line 601):
        
        # Obtaining the type of the subscript
        str_28586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 40), 'str', 'forward_ref')
        # Getting the type of 'function_set' (line 601)
        function_set_28587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 27), 'function_set')
        # Obtaining the member '__getitem__' of a type (line 601)
        getitem___28588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 27), function_set_28587, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 601)
        subscript_call_result_28589 = invoke(stypy.reporting.localization.Localization(__file__, 601, 27), getitem___28588, str_28586)
        
        # Assigning a type to the variable 'fforward_ref' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'fforward_ref', subscript_call_result_28589)
        
        # Getting the type of 'self' (line 602)
        self_28590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 28), 'self')
        # Obtaining the member 'types' of a type (line 602)
        types_28591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 28), self_28590, 'types')
        # Testing the type of a for loop iterable (line 602)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 602, 12), types_28591)
        # Getting the type of the for loop variable (line 602)
        for_loop_var_28592 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 602, 12), types_28591)
        # Assigning a type to the variable 'dct_type' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'dct_type', for_loop_var_28592)
        # SSA begins for a for statement (line 602)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 603)
        self_28593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 28), 'self')
        # Obtaining the member 'norms' of a type (line 603)
        norms_28594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 28), self_28593, 'norms')
        # Testing the type of a for loop iterable (line 603)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 603, 16), norms_28594)
        # Getting the type of the for loop variable (line 603)
        for_loop_var_28595 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 603, 16), norms_28594)
        # Assigning a type to the variable 'norm' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 16), 'norm', for_loop_var_28595)
        # SSA begins for a for statement (line 603)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'norm' (line 604)
        norm_28596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 23), 'norm')
        str_28597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 31), 'str', 'ortho')
        # Applying the binary operator '==' (line 604)
        result_eq_28598 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 23), '==', norm_28596, str_28597)
        
        
        # Getting the type of 'dct_type' (line 604)
        dct_type_28599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 43), 'dct_type')
        int_28600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 55), 'int')
        # Applying the binary operator '==' (line 604)
        result_eq_28601 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 43), '==', dct_type_28599, int_28600)
        
        # Applying the binary operator 'and' (line 604)
        result_and_keyword_28602 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 23), 'and', result_eq_28598, result_eq_28601)
        
        # Testing the type of an if condition (line 604)
        if_condition_28603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 20), result_and_keyword_28602)
        # Assigning a type to the variable 'if_condition_28603' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 20), 'if_condition_28603', if_condition_28603)
        # SSA begins for if statement (line 604)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 604)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 606):
        
        # Assigning a Call to a Name (line 606):
        
        # Call to fforward(...): (line 606)
        # Processing the call arguments (line 606)
        # Getting the type of 'self' (line 606)
        self_28605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 34), 'self', False)
        # Obtaining the member 'data' of a type (line 606)
        data_28606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 34), self_28605, 'data')
        # Processing the call keyword arguments (line 606)
        # Getting the type of 'dct_type' (line 606)
        dct_type_28607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 50), 'dct_type', False)
        keyword_28608 = dct_type_28607
        # Getting the type of 'None' (line 606)
        None_28609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 65), 'None', False)
        keyword_28610 = None_28609
        # Getting the type of 'norm' (line 607)
        norm_28611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 39), 'norm', False)
        keyword_28612 = norm_28611
        kwargs_28613 = {'axes': keyword_28610, 'type': keyword_28608, 'norm': keyword_28612}
        # Getting the type of 'fforward' (line 606)
        fforward_28604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 25), 'fforward', False)
        # Calling fforward(args, kwargs) (line 606)
        fforward_call_result_28614 = invoke(stypy.reporting.localization.Localization(__file__, 606, 25), fforward_28604, *[data_28606], **kwargs_28613)
        
        # Assigning a type to the variable 'y1' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 20), 'y1', fforward_call_result_28614)
        
        # Assigning a Call to a Name (line 608):
        
        # Assigning a Call to a Name (line 608):
        
        # Call to fforward_ref(...): (line 608)
        # Processing the call arguments (line 608)
        # Getting the type of 'self' (line 608)
        self_28616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 38), 'self', False)
        # Obtaining the member 'data' of a type (line 608)
        data_28617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 38), self_28616, 'data')
        # Processing the call keyword arguments (line 608)
        # Getting the type of 'dct_type' (line 608)
        dct_type_28618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 54), 'dct_type', False)
        keyword_28619 = dct_type_28618
        # Getting the type of 'norm' (line 608)
        norm_28620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 69), 'norm', False)
        keyword_28621 = norm_28620
        kwargs_28622 = {'type': keyword_28619, 'norm': keyword_28621}
        # Getting the type of 'fforward_ref' (line 608)
        fforward_ref_28615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 25), 'fforward_ref', False)
        # Calling fforward_ref(args, kwargs) (line 608)
        fforward_ref_call_result_28623 = invoke(stypy.reporting.localization.Localization(__file__, 608, 25), fforward_ref_28615, *[data_28617], **kwargs_28622)
        
        # Assigning a type to the variable 'y2' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 20), 'y2', fforward_ref_call_result_28623)
        
        # Call to assert_array_almost_equal(...): (line 609)
        # Processing the call arguments (line 609)
        # Getting the type of 'y1' (line 609)
        y1_28625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 46), 'y1', False)
        # Getting the type of 'y2' (line 609)
        y2_28626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 50), 'y2', False)
        # Processing the call keyword arguments (line 609)
        int_28627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 62), 'int')
        keyword_28628 = int_28627
        kwargs_28629 = {'decimal': keyword_28628}
        # Getting the type of 'assert_array_almost_equal' (line 609)
        assert_array_almost_equal_28624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 20), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 609)
        assert_array_almost_equal_call_result_28630 = invoke(stypy.reporting.localization.Localization(__file__, 609, 20), assert_array_almost_equal_28624, *[y1_28625, y2_28626], **kwargs_28629)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dctn_vs_2d_reference(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dctn_vs_2d_reference' in the type store
        # Getting the type of 'stypy_return_type' (line 598)
        stypy_return_type_28631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28631)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dctn_vs_2d_reference'
        return stypy_return_type_28631


    @norecursion
    def test_idctn_vs_2d_reference(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_idctn_vs_2d_reference'
        module_type_store = module_type_store.open_function_context('test_idctn_vs_2d_reference', 611, 4, False)
        # Assigning a type to the variable 'self' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_DCTN_IDCTN.test_idctn_vs_2d_reference.__dict__.__setitem__('stypy_localization', localization)
        Test_DCTN_IDCTN.test_idctn_vs_2d_reference.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_DCTN_IDCTN.test_idctn_vs_2d_reference.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_DCTN_IDCTN.test_idctn_vs_2d_reference.__dict__.__setitem__('stypy_function_name', 'Test_DCTN_IDCTN.test_idctn_vs_2d_reference')
        Test_DCTN_IDCTN.test_idctn_vs_2d_reference.__dict__.__setitem__('stypy_param_names_list', [])
        Test_DCTN_IDCTN.test_idctn_vs_2d_reference.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_DCTN_IDCTN.test_idctn_vs_2d_reference.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_DCTN_IDCTN.test_idctn_vs_2d_reference.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_DCTN_IDCTN.test_idctn_vs_2d_reference.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_DCTN_IDCTN.test_idctn_vs_2d_reference.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_DCTN_IDCTN.test_idctn_vs_2d_reference.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_DCTN_IDCTN.test_idctn_vs_2d_reference', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_idctn_vs_2d_reference', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_idctn_vs_2d_reference(...)' code ##################

        
        # Getting the type of 'self' (line 612)
        self_28632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 28), 'self')
        # Obtaining the member 'function_sets' of a type (line 612)
        function_sets_28633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 28), self_28632, 'function_sets')
        # Testing the type of a for loop iterable (line 612)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 612, 8), function_sets_28633)
        # Getting the type of the for loop variable (line 612)
        for_loop_var_28634 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 612, 8), function_sets_28633)
        # Assigning a type to the variable 'function_set' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'function_set', for_loop_var_28634)
        # SSA begins for a for statement (line 612)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 613):
        
        # Assigning a Subscript to a Name (line 613):
        
        # Obtaining the type of the subscript
        str_28635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 36), 'str', 'inverse')
        # Getting the type of 'function_set' (line 613)
        function_set_28636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 23), 'function_set')
        # Obtaining the member '__getitem__' of a type (line 613)
        getitem___28637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 23), function_set_28636, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 613)
        subscript_call_result_28638 = invoke(stypy.reporting.localization.Localization(__file__, 613, 23), getitem___28637, str_28635)
        
        # Assigning a type to the variable 'finverse' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'finverse', subscript_call_result_28638)
        
        # Assigning a Subscript to a Name (line 614):
        
        # Assigning a Subscript to a Name (line 614):
        
        # Obtaining the type of the subscript
        str_28639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 40), 'str', 'inverse_ref')
        # Getting the type of 'function_set' (line 614)
        function_set_28640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 27), 'function_set')
        # Obtaining the member '__getitem__' of a type (line 614)
        getitem___28641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 27), function_set_28640, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 614)
        subscript_call_result_28642 = invoke(stypy.reporting.localization.Localization(__file__, 614, 27), getitem___28641, str_28639)
        
        # Assigning a type to the variable 'finverse_ref' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'finverse_ref', subscript_call_result_28642)
        
        # Getting the type of 'self' (line 615)
        self_28643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 28), 'self')
        # Obtaining the member 'types' of a type (line 615)
        types_28644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 28), self_28643, 'types')
        # Testing the type of a for loop iterable (line 615)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 615, 12), types_28644)
        # Getting the type of the for loop variable (line 615)
        for_loop_var_28645 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 615, 12), types_28644)
        # Assigning a type to the variable 'dct_type' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'dct_type', for_loop_var_28645)
        # SSA begins for a for statement (line 615)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 616)
        self_28646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 28), 'self')
        # Obtaining the member 'norms' of a type (line 616)
        norms_28647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 28), self_28646, 'norms')
        # Testing the type of a for loop iterable (line 616)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 616, 16), norms_28647)
        # Getting the type of the for loop variable (line 616)
        for_loop_var_28648 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 616, 16), norms_28647)
        # Assigning a type to the variable 'norm' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'norm', for_loop_var_28648)
        # SSA begins for a for statement (line 616)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to print(...): (line 617)
        # Processing the call arguments (line 617)
        # Getting the type of 'function_set' (line 617)
        function_set_28650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 26), 'function_set', False)
        # Getting the type of 'dct_type' (line 617)
        dct_type_28651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 40), 'dct_type', False)
        # Getting the type of 'norm' (line 617)
        norm_28652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 50), 'norm', False)
        # Processing the call keyword arguments (line 617)
        kwargs_28653 = {}
        # Getting the type of 'print' (line 617)
        print_28649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 20), 'print', False)
        # Calling print(args, kwargs) (line 617)
        print_call_result_28654 = invoke(stypy.reporting.localization.Localization(__file__, 617, 20), print_28649, *[function_set_28650, dct_type_28651, norm_28652], **kwargs_28653)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'norm' (line 618)
        norm_28655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 23), 'norm')
        str_28656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 31), 'str', 'ortho')
        # Applying the binary operator '==' (line 618)
        result_eq_28657 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 23), '==', norm_28655, str_28656)
        
        
        # Getting the type of 'dct_type' (line 618)
        dct_type_28658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 43), 'dct_type')
        int_28659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 55), 'int')
        # Applying the binary operator '==' (line 618)
        result_eq_28660 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 43), '==', dct_type_28658, int_28659)
        
        # Applying the binary operator 'and' (line 618)
        result_and_keyword_28661 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 23), 'and', result_eq_28657, result_eq_28660)
        
        # Testing the type of an if condition (line 618)
        if_condition_28662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 618, 20), result_and_keyword_28661)
        # Assigning a type to the variable 'if_condition_28662' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 20), 'if_condition_28662', if_condition_28662)
        # SSA begins for if statement (line 618)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 618)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 620):
        
        # Assigning a Call to a Name (line 620):
        
        # Call to dctn(...): (line 620)
        # Processing the call arguments (line 620)
        # Getting the type of 'self' (line 620)
        self_28664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 33), 'self', False)
        # Obtaining the member 'data' of a type (line 620)
        data_28665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 33), self_28664, 'data')
        # Processing the call keyword arguments (line 620)
        # Getting the type of 'dct_type' (line 620)
        dct_type_28666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 49), 'dct_type', False)
        keyword_28667 = dct_type_28666
        # Getting the type of 'norm' (line 620)
        norm_28668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 64), 'norm', False)
        keyword_28669 = norm_28668
        kwargs_28670 = {'type': keyword_28667, 'norm': keyword_28669}
        # Getting the type of 'dctn' (line 620)
        dctn_28663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 28), 'dctn', False)
        # Calling dctn(args, kwargs) (line 620)
        dctn_call_result_28671 = invoke(stypy.reporting.localization.Localization(__file__, 620, 28), dctn_28663, *[data_28665], **kwargs_28670)
        
        # Assigning a type to the variable 'fdata' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 20), 'fdata', dctn_call_result_28671)
        
        # Assigning a Call to a Name (line 621):
        
        # Assigning a Call to a Name (line 621):
        
        # Call to finverse(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'fdata' (line 621)
        fdata_28673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 34), 'fdata', False)
        # Processing the call keyword arguments (line 621)
        # Getting the type of 'dct_type' (line 621)
        dct_type_28674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 46), 'dct_type', False)
        keyword_28675 = dct_type_28674
        # Getting the type of 'norm' (line 621)
        norm_28676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 61), 'norm', False)
        keyword_28677 = norm_28676
        kwargs_28678 = {'type': keyword_28675, 'norm': keyword_28677}
        # Getting the type of 'finverse' (line 621)
        finverse_28672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 25), 'finverse', False)
        # Calling finverse(args, kwargs) (line 621)
        finverse_call_result_28679 = invoke(stypy.reporting.localization.Localization(__file__, 621, 25), finverse_28672, *[fdata_28673], **kwargs_28678)
        
        # Assigning a type to the variable 'y1' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 20), 'y1', finverse_call_result_28679)
        
        # Assigning a Call to a Name (line 622):
        
        # Assigning a Call to a Name (line 622):
        
        # Call to finverse_ref(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 'fdata' (line 622)
        fdata_28681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 38), 'fdata', False)
        # Processing the call keyword arguments (line 622)
        # Getting the type of 'dct_type' (line 622)
        dct_type_28682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 50), 'dct_type', False)
        keyword_28683 = dct_type_28682
        # Getting the type of 'norm' (line 622)
        norm_28684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 65), 'norm', False)
        keyword_28685 = norm_28684
        kwargs_28686 = {'type': keyword_28683, 'norm': keyword_28685}
        # Getting the type of 'finverse_ref' (line 622)
        finverse_ref_28680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 25), 'finverse_ref', False)
        # Calling finverse_ref(args, kwargs) (line 622)
        finverse_ref_call_result_28687 = invoke(stypy.reporting.localization.Localization(__file__, 622, 25), finverse_ref_28680, *[fdata_28681], **kwargs_28686)
        
        # Assigning a type to the variable 'y2' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 20), 'y2', finverse_ref_call_result_28687)
        
        # Call to assert_array_almost_equal(...): (line 623)
        # Processing the call arguments (line 623)
        # Getting the type of 'y1' (line 623)
        y1_28689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 46), 'y1', False)
        # Getting the type of 'y2' (line 623)
        y2_28690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 50), 'y2', False)
        # Processing the call keyword arguments (line 623)
        int_28691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 62), 'int')
        keyword_28692 = int_28691
        kwargs_28693 = {'decimal': keyword_28692}
        # Getting the type of 'assert_array_almost_equal' (line 623)
        assert_array_almost_equal_28688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 20), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 623)
        assert_array_almost_equal_call_result_28694 = invoke(stypy.reporting.localization.Localization(__file__, 623, 20), assert_array_almost_equal_28688, *[y1_28689, y2_28690], **kwargs_28693)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_idctn_vs_2d_reference(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_idctn_vs_2d_reference' in the type store
        # Getting the type of 'stypy_return_type' (line 611)
        stypy_return_type_28695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28695)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_idctn_vs_2d_reference'
        return stypy_return_type_28695


    @norecursion
    def test_axes_and_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_axes_and_shape'
        module_type_store = module_type_store.open_function_context('test_axes_and_shape', 625, 4, False)
        # Assigning a type to the variable 'self' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_DCTN_IDCTN.test_axes_and_shape.__dict__.__setitem__('stypy_localization', localization)
        Test_DCTN_IDCTN.test_axes_and_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_DCTN_IDCTN.test_axes_and_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_DCTN_IDCTN.test_axes_and_shape.__dict__.__setitem__('stypy_function_name', 'Test_DCTN_IDCTN.test_axes_and_shape')
        Test_DCTN_IDCTN.test_axes_and_shape.__dict__.__setitem__('stypy_param_names_list', [])
        Test_DCTN_IDCTN.test_axes_and_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_DCTN_IDCTN.test_axes_and_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_DCTN_IDCTN.test_axes_and_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_DCTN_IDCTN.test_axes_and_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_DCTN_IDCTN.test_axes_and_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_DCTN_IDCTN.test_axes_and_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_DCTN_IDCTN.test_axes_and_shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_axes_and_shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_axes_and_shape(...)' code ##################

        
        # Getting the type of 'self' (line 626)
        self_28696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 28), 'self')
        # Obtaining the member 'function_sets' of a type (line 626)
        function_sets_28697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 28), self_28696, 'function_sets')
        # Testing the type of a for loop iterable (line 626)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 626, 8), function_sets_28697)
        # Getting the type of the for loop variable (line 626)
        for_loop_var_28698 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 626, 8), function_sets_28697)
        # Assigning a type to the variable 'function_set' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'function_set', for_loop_var_28698)
        # SSA begins for a for statement (line 626)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 627):
        
        # Assigning a Subscript to a Name (line 627):
        
        # Obtaining the type of the subscript
        str_28699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 36), 'str', 'forward')
        # Getting the type of 'function_set' (line 627)
        function_set_28700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 23), 'function_set')
        # Obtaining the member '__getitem__' of a type (line 627)
        getitem___28701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 23), function_set_28700, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 627)
        subscript_call_result_28702 = invoke(stypy.reporting.localization.Localization(__file__, 627, 23), getitem___28701, str_28699)
        
        # Assigning a type to the variable 'fforward' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'fforward', subscript_call_result_28702)
        
        # Assigning a Subscript to a Name (line 628):
        
        # Assigning a Subscript to a Name (line 628):
        
        # Obtaining the type of the subscript
        str_28703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 36), 'str', 'inverse')
        # Getting the type of 'function_set' (line 628)
        function_set_28704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 23), 'function_set')
        # Obtaining the member '__getitem__' of a type (line 628)
        getitem___28705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 23), function_set_28704, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 628)
        subscript_call_result_28706 = invoke(stypy.reporting.localization.Localization(__file__, 628, 23), getitem___28705, str_28703)
        
        # Assigning a type to the variable 'finverse' (line 628)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 12), 'finverse', subscript_call_result_28706)
        
        # Call to assert_raises(...): (line 631)
        # Processing the call arguments (line 631)
        # Getting the type of 'ValueError' (line 631)
        ValueError_28708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 26), 'ValueError', False)
        # Getting the type of 'fforward' (line 631)
        fforward_28709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 38), 'fforward', False)
        # Getting the type of 'self' (line 631)
        self_28710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 48), 'self', False)
        # Obtaining the member 'data' of a type (line 631)
        data_28711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 48), self_28710, 'data')
        # Processing the call keyword arguments (line 631)
        
        # Obtaining an instance of the builtin type 'tuple' (line 632)
        tuple_28712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 632)
        # Adding element type (line 632)
        
        # Obtaining the type of the subscript
        int_28713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 49), 'int')
        # Getting the type of 'self' (line 632)
        self_28714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 33), 'self', False)
        # Obtaining the member 'data' of a type (line 632)
        data_28715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 33), self_28714, 'data')
        # Obtaining the member 'shape' of a type (line 632)
        shape_28716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 33), data_28715, 'shape')
        # Obtaining the member '__getitem__' of a type (line 632)
        getitem___28717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 33), shape_28716, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 632)
        subscript_call_result_28718 = invoke(stypy.reporting.localization.Localization(__file__, 632, 33), getitem___28717, int_28713)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 33), tuple_28712, subscript_call_result_28718)
        
        keyword_28719 = tuple_28712
        
        # Obtaining an instance of the builtin type 'tuple' (line 633)
        tuple_28720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 633)
        # Adding element type (line 633)
        int_28721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 633, 32), tuple_28720, int_28721)
        # Adding element type (line 633)
        int_28722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 633, 32), tuple_28720, int_28722)
        
        keyword_28723 = tuple_28720
        kwargs_28724 = {'shape': keyword_28719, 'axes': keyword_28723}
        # Getting the type of 'assert_raises' (line 631)
        assert_raises_28707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 631)
        assert_raises_call_result_28725 = invoke(stypy.reporting.localization.Localization(__file__, 631, 12), assert_raises_28707, *[ValueError_28708, fforward_28709, data_28711], **kwargs_28724)
        
        
        # Call to assert_raises(...): (line 634)
        # Processing the call arguments (line 634)
        # Getting the type of 'ValueError' (line 634)
        ValueError_28727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 26), 'ValueError', False)
        # Getting the type of 'fforward' (line 634)
        fforward_28728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 38), 'fforward', False)
        # Getting the type of 'self' (line 634)
        self_28729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 48), 'self', False)
        # Obtaining the member 'data' of a type (line 634)
        data_28730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 48), self_28729, 'data')
        # Processing the call keyword arguments (line 634)
        
        # Obtaining an instance of the builtin type 'tuple' (line 635)
        tuple_28731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 635)
        # Adding element type (line 635)
        
        # Obtaining the type of the subscript
        int_28732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 49), 'int')
        # Getting the type of 'self' (line 635)
        self_28733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 33), 'self', False)
        # Obtaining the member 'data' of a type (line 635)
        data_28734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 33), self_28733, 'data')
        # Obtaining the member 'shape' of a type (line 635)
        shape_28735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 33), data_28734, 'shape')
        # Obtaining the member '__getitem__' of a type (line 635)
        getitem___28736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 33), shape_28735, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 635)
        subscript_call_result_28737 = invoke(stypy.reporting.localization.Localization(__file__, 635, 33), getitem___28736, int_28732)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 635, 33), tuple_28731, subscript_call_result_28737)
        
        keyword_28738 = tuple_28731
        # Getting the type of 'None' (line 636)
        None_28739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 31), 'None', False)
        keyword_28740 = None_28739
        kwargs_28741 = {'shape': keyword_28738, 'axes': keyword_28740}
        # Getting the type of 'assert_raises' (line 634)
        assert_raises_28726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 634)
        assert_raises_call_result_28742 = invoke(stypy.reporting.localization.Localization(__file__, 634, 12), assert_raises_28726, *[ValueError_28727, fforward_28728, data_28730], **kwargs_28741)
        
        
        # Call to assert_raises(...): (line 637)
        # Processing the call arguments (line 637)
        # Getting the type of 'ValueError' (line 637)
        ValueError_28744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 26), 'ValueError', False)
        # Getting the type of 'fforward' (line 637)
        fforward_28745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 38), 'fforward', False)
        # Getting the type of 'self' (line 637)
        self_28746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 48), 'self', False)
        # Obtaining the member 'data' of a type (line 637)
        data_28747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 48), self_28746, 'data')
        # Processing the call keyword arguments (line 637)
        # Getting the type of 'self' (line 638)
        self_28748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 32), 'self', False)
        # Obtaining the member 'data' of a type (line 638)
        data_28749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 32), self_28748, 'data')
        # Obtaining the member 'shape' of a type (line 638)
        shape_28750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 32), data_28749, 'shape')
        keyword_28751 = shape_28750
        
        # Obtaining an instance of the builtin type 'tuple' (line 639)
        tuple_28752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 639)
        # Adding element type (line 639)
        int_28753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 639, 32), tuple_28752, int_28753)
        
        keyword_28754 = tuple_28752
        kwargs_28755 = {'shape': keyword_28751, 'axes': keyword_28754}
        # Getting the type of 'assert_raises' (line 637)
        assert_raises_28743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 637)
        assert_raises_call_result_28756 = invoke(stypy.reporting.localization.Localization(__file__, 637, 12), assert_raises_28743, *[ValueError_28744, fforward_28745, data_28747], **kwargs_28755)
        
        
        # Call to assert_raises(...): (line 641)
        # Processing the call arguments (line 641)
        # Getting the type of 'TypeError' (line 641)
        TypeError_28758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 26), 'TypeError', False)
        # Getting the type of 'fforward' (line 641)
        fforward_28759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 37), 'fforward', False)
        # Getting the type of 'self' (line 641)
        self_28760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 47), 'self', False)
        # Obtaining the member 'data' of a type (line 641)
        data_28761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 47), self_28760, 'data')
        # Processing the call keyword arguments (line 641)
        
        # Obtaining the type of the subscript
        int_28762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 48), 'int')
        # Getting the type of 'self' (line 642)
        self_28763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 32), 'self', False)
        # Obtaining the member 'data' of a type (line 642)
        data_28764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 32), self_28763, 'data')
        # Obtaining the member 'shape' of a type (line 642)
        shape_28765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 32), data_28764, 'shape')
        # Obtaining the member '__getitem__' of a type (line 642)
        getitem___28766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 32), shape_28765, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 642)
        subscript_call_result_28767 = invoke(stypy.reporting.localization.Localization(__file__, 642, 32), getitem___28766, int_28762)
        
        keyword_28768 = subscript_call_result_28767
        
        # Obtaining an instance of the builtin type 'tuple' (line 643)
        tuple_28769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 643)
        # Adding element type (line 643)
        int_28770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 643, 32), tuple_28769, int_28770)
        # Adding element type (line 643)
        int_28771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 643, 32), tuple_28769, int_28771)
        
        keyword_28772 = tuple_28769
        kwargs_28773 = {'shape': keyword_28768, 'axes': keyword_28772}
        # Getting the type of 'assert_raises' (line 641)
        assert_raises_28757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 641)
        assert_raises_call_result_28774 = invoke(stypy.reporting.localization.Localization(__file__, 641, 12), assert_raises_28757, *[TypeError_28758, fforward_28759, data_28761], **kwargs_28773)
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 646)
        list_28775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 646)
        # Adding element type (line 646)
        
        # Obtaining an instance of the builtin type 'tuple' (line 646)
        tuple_28776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 646)
        # Adding element type (line 646)
        int_28777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 646, 26), tuple_28776, int_28777)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 646, 24), list_28775, tuple_28776)
        # Adding element type (line 646)
        
        # Obtaining an instance of the builtin type 'tuple' (line 646)
        tuple_28778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 646)
        # Adding element type (line 646)
        int_28779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 646, 33), tuple_28778, int_28779)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 646, 24), list_28775, tuple_28778)
        
        # Testing the type of a for loop iterable (line 646)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 646, 12), list_28775)
        # Getting the type of the for loop variable (line 646)
        for_loop_var_28780 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 646, 12), list_28775)
        # Assigning a type to the variable 'axes' (line 646)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 12), 'axes', for_loop_var_28780)
        # SSA begins for a for statement (line 646)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 647):
        
        # Assigning a Call to a Name (line 647):
        
        # Call to fforward(...): (line 647)
        # Processing the call arguments (line 647)
        # Getting the type of 'self' (line 647)
        self_28782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 31), 'self', False)
        # Obtaining the member 'data' of a type (line 647)
        data_28783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 31), self_28782, 'data')
        # Processing the call keyword arguments (line 647)
        # Getting the type of 'None' (line 647)
        None_28784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 48), 'None', False)
        keyword_28785 = None_28784
        # Getting the type of 'axes' (line 647)
        axes_28786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 59), 'axes', False)
        keyword_28787 = axes_28786
        str_28788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 70), 'str', 'ortho')
        keyword_28789 = str_28788
        kwargs_28790 = {'shape': keyword_28785, 'axes': keyword_28787, 'norm': keyword_28789}
        # Getting the type of 'fforward' (line 647)
        fforward_28781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 22), 'fforward', False)
        # Calling fforward(args, kwargs) (line 647)
        fforward_call_result_28791 = invoke(stypy.reporting.localization.Localization(__file__, 647, 22), fforward_28781, *[data_28783], **kwargs_28790)
        
        # Assigning a type to the variable 'tmp' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 16), 'tmp', fforward_call_result_28791)
        
        # Assigning a Call to a Name (line 648):
        
        # Assigning a Call to a Name (line 648):
        
        # Call to finverse(...): (line 648)
        # Processing the call arguments (line 648)
        # Getting the type of 'tmp' (line 648)
        tmp_28793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 31), 'tmp', False)
        # Processing the call keyword arguments (line 648)
        # Getting the type of 'None' (line 648)
        None_28794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 42), 'None', False)
        keyword_28795 = None_28794
        # Getting the type of 'axes' (line 648)
        axes_28796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 53), 'axes', False)
        keyword_28797 = axes_28796
        str_28798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 64), 'str', 'ortho')
        keyword_28799 = str_28798
        kwargs_28800 = {'shape': keyword_28795, 'axes': keyword_28797, 'norm': keyword_28799}
        # Getting the type of 'finverse' (line 648)
        finverse_28792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 22), 'finverse', False)
        # Calling finverse(args, kwargs) (line 648)
        finverse_call_result_28801 = invoke(stypy.reporting.localization.Localization(__file__, 648, 22), finverse_28792, *[tmp_28793], **kwargs_28800)
        
        # Assigning a type to the variable 'tmp' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'tmp', finverse_call_result_28801)
        
        # Call to assert_array_almost_equal(...): (line 649)
        # Processing the call arguments (line 649)
        # Getting the type of 'self' (line 649)
        self_28803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 42), 'self', False)
        # Obtaining the member 'data' of a type (line 649)
        data_28804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 42), self_28803, 'data')
        # Getting the type of 'tmp' (line 649)
        tmp_28805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 53), 'tmp', False)
        # Processing the call keyword arguments (line 649)
        # Getting the type of 'self' (line 649)
        self_28806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 66), 'self', False)
        # Obtaining the member 'dec' of a type (line 649)
        dec_28807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 66), self_28806, 'dec')
        keyword_28808 = dec_28807
        kwargs_28809 = {'decimal': keyword_28808}
        # Getting the type of 'assert_array_almost_equal' (line 649)
        assert_array_almost_equal_28802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 649)
        assert_array_almost_equal_call_result_28810 = invoke(stypy.reporting.localization.Localization(__file__, 649, 16), assert_array_almost_equal_28802, *[data_28804, tmp_28805], **kwargs_28809)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 652):
        
        # Assigning a Call to a Name (line 652):
        
        # Call to fforward(...): (line 652)
        # Processing the call arguments (line 652)
        # Getting the type of 'self' (line 652)
        self_28812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 27), 'self', False)
        # Obtaining the member 'data' of a type (line 652)
        data_28813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 27), self_28812, 'data')
        # Processing the call keyword arguments (line 652)
        
        # Obtaining an instance of the builtin type 'tuple' (line 652)
        tuple_28814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 652)
        # Adding element type (line 652)
        int_28815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 652, 45), tuple_28814, int_28815)
        # Adding element type (line 652)
        int_28816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 652, 45), tuple_28814, int_28816)
        
        keyword_28817 = tuple_28814
        # Getting the type of 'None' (line 652)
        None_28818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 61), 'None', False)
        keyword_28819 = None_28818
        kwargs_28820 = {'shape': keyword_28817, 'axes': keyword_28819}
        # Getting the type of 'fforward' (line 652)
        fforward_28811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 18), 'fforward', False)
        # Calling fforward(args, kwargs) (line 652)
        fforward_call_result_28821 = invoke(stypy.reporting.localization.Localization(__file__, 652, 18), fforward_28811, *[data_28813], **kwargs_28820)
        
        # Assigning a type to the variable 'tmp' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'tmp', fforward_call_result_28821)
        
        # Call to assert_equal(...): (line 653)
        # Processing the call arguments (line 653)
        # Getting the type of 'tmp' (line 653)
        tmp_28823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 25), 'tmp', False)
        # Obtaining the member 'shape' of a type (line 653)
        shape_28824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 25), tmp_28823, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 653)
        tuple_28825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 653)
        # Adding element type (line 653)
        int_28826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 653, 37), tuple_28825, int_28826)
        # Adding element type (line 653)
        int_28827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 653, 37), tuple_28825, int_28827)
        
        # Processing the call keyword arguments (line 653)
        kwargs_28828 = {}
        # Getting the type of 'assert_equal' (line 653)
        assert_equal_28822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 653)
        assert_equal_call_result_28829 = invoke(stypy.reporting.localization.Localization(__file__, 653, 12), assert_equal_28822, *[shape_28824, tuple_28825], **kwargs_28828)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_axes_and_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_axes_and_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 625)
        stypy_return_type_28830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28830)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_axes_and_shape'
        return stypy_return_type_28830


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 567, 0, False)
        # Assigning a type to the variable 'self' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_DCTN_IDCTN.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_DCTN_IDCTN' (line 567)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 0), 'Test_DCTN_IDCTN', Test_DCTN_IDCTN)

# Assigning a Num to a Name (line 568):
int_28831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 10), 'int')
# Getting the type of 'Test_DCTN_IDCTN'
Test_DCTN_IDCTN_28832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test_DCTN_IDCTN')
# Setting the type of the member 'dec' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_DCTN_IDCTN_28832, 'dec', int_28831)

# Assigning a List to a Name (line 569):

# Obtaining an instance of the builtin type 'list' (line 569)
list_28833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 569)
# Adding element type (line 569)
int_28834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 12), list_28833, int_28834)
# Adding element type (line 569)
int_28835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 12), list_28833, int_28835)
# Adding element type (line 569)
int_28836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 12), list_28833, int_28836)

# Getting the type of 'Test_DCTN_IDCTN'
Test_DCTN_IDCTN_28837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test_DCTN_IDCTN')
# Setting the type of the member 'types' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_DCTN_IDCTN_28837, 'types', list_28833)

# Assigning a List to a Name (line 570):

# Obtaining an instance of the builtin type 'list' (line 570)
list_28838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 570)
# Adding element type (line 570)
# Getting the type of 'None' (line 570)
None_28839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 13), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 12), list_28838, None_28839)
# Adding element type (line 570)
str_28840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 19), 'str', 'ortho')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 12), list_28838, str_28840)

# Getting the type of 'Test_DCTN_IDCTN'
Test_DCTN_IDCTN_28841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test_DCTN_IDCTN')
# Setting the type of the member 'norms' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_DCTN_IDCTN_28841, 'norms', list_28838)

# Assigning a Call to a Name (line 571):

# Call to RandomState(...): (line 571)
# Processing the call arguments (line 571)
int_28845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 35), 'int')
# Processing the call keyword arguments (line 571)
kwargs_28846 = {}
# Getting the type of 'np' (line 571)
np_28842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 13), 'np', False)
# Obtaining the member 'random' of a type (line 571)
random_28843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 13), np_28842, 'random')
# Obtaining the member 'RandomState' of a type (line 571)
RandomState_28844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 13), random_28843, 'RandomState')
# Calling RandomState(args, kwargs) (line 571)
RandomState_call_result_28847 = invoke(stypy.reporting.localization.Localization(__file__, 571, 13), RandomState_28844, *[int_28845], **kwargs_28846)

# Getting the type of 'Test_DCTN_IDCTN'
Test_DCTN_IDCTN_28848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test_DCTN_IDCTN')
# Setting the type of the member 'rstate' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_DCTN_IDCTN_28848, 'rstate', RandomState_call_result_28847)

# Assigning a Tuple to a Name (line 572):

# Obtaining an instance of the builtin type 'tuple' (line 572)
tuple_28849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 572)
# Adding element type (line 572)
int_28850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 13), tuple_28849, int_28850)
# Adding element type (line 572)
int_28851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 13), tuple_28849, int_28851)

# Getting the type of 'Test_DCTN_IDCTN'
Test_DCTN_IDCTN_28852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test_DCTN_IDCTN')
# Setting the type of the member 'shape' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_DCTN_IDCTN_28852, 'shape', tuple_28849)

# Assigning a Call to a Name (line 573):

# Call to randn(...): (line 573)
# Getting the type of 'shape' (line 573)
shape_28856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 25), 'shape', False)
# Processing the call keyword arguments (line 573)
kwargs_28857 = {}
# Getting the type of 'Test_DCTN_IDCTN'
Test_DCTN_IDCTN_28853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test_DCTN_IDCTN', False)
# Obtaining the member 'rstate' of a type
rstate_28854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_DCTN_IDCTN_28853, 'rstate')
# Obtaining the member 'randn' of a type
randn_28855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), rstate_28854, 'randn')
# Calling randn(args, kwargs) (line 573)
randn_call_result_28858 = invoke(stypy.reporting.localization.Localization(__file__, 573, 11), randn_28855, *[shape_28856], **kwargs_28857)

# Getting the type of 'Test_DCTN_IDCTN'
Test_DCTN_IDCTN_28859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test_DCTN_IDCTN')
# Setting the type of the member 'data' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_DCTN_IDCTN_28859, 'data', randn_call_result_28858)

# Assigning a List to a Name (line 575):

# Obtaining an instance of the builtin type 'list' (line 575)
list_28860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 575)
# Adding element type (line 575)

# Call to dict(...): (line 575)
# Processing the call keyword arguments (line 575)
# Getting the type of 'dctn' (line 575)
dctn_28862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 34), 'dctn', False)
keyword_28863 = dctn_28862
# Getting the type of 'idctn' (line 576)
idctn_28864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 34), 'idctn', False)
keyword_28865 = idctn_28864
# Getting the type of 'dct_2d_ref' (line 577)
dct_2d_ref_28866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 38), 'dct_2d_ref', False)
keyword_28867 = dct_2d_ref_28866
# Getting the type of 'idct_2d_ref' (line 578)
idct_2d_ref_28868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 38), 'idct_2d_ref', False)
keyword_28869 = idct_2d_ref_28868
kwargs_28870 = {'forward': keyword_28863, 'inverse_ref': keyword_28869, 'inverse': keyword_28865, 'forward_ref': keyword_28867}
# Getting the type of 'dict' (line 575)
dict_28861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 21), 'dict', False)
# Calling dict(args, kwargs) (line 575)
dict_call_result_28871 = invoke(stypy.reporting.localization.Localization(__file__, 575, 21), dict_28861, *[], **kwargs_28870)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 20), list_28860, dict_call_result_28871)
# Adding element type (line 575)

# Call to dict(...): (line 579)
# Processing the call keyword arguments (line 579)
# Getting the type of 'dstn' (line 579)
dstn_28873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 34), 'dstn', False)
keyword_28874 = dstn_28873
# Getting the type of 'idstn' (line 580)
idstn_28875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 34), 'idstn', False)
keyword_28876 = idstn_28875
# Getting the type of 'dst_2d_ref' (line 581)
dst_2d_ref_28877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 38), 'dst_2d_ref', False)
keyword_28878 = dst_2d_ref_28877
# Getting the type of 'idst_2d_ref' (line 582)
idst_2d_ref_28879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 38), 'idst_2d_ref', False)
keyword_28880 = idst_2d_ref_28879
kwargs_28881 = {'forward': keyword_28874, 'inverse_ref': keyword_28880, 'inverse': keyword_28876, 'forward_ref': keyword_28878}
# Getting the type of 'dict' (line 579)
dict_28872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 21), 'dict', False)
# Calling dict(args, kwargs) (line 579)
dict_call_result_28882 = invoke(stypy.reporting.localization.Localization(__file__, 579, 21), dict_28872, *[], **kwargs_28881)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 20), list_28860, dict_call_result_28882)

# Getting the type of 'Test_DCTN_IDCTN'
Test_DCTN_IDCTN_28883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test_DCTN_IDCTN')
# Setting the type of the member 'function_sets' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_DCTN_IDCTN_28883, 'function_sets', list_28860)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
