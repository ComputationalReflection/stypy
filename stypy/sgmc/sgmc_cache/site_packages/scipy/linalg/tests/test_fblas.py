
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Test interfaces to fortran blas.
2: #
3: # The tests are more of interface than they are of the underlying blas.
4: # Only very small matrices checked -- N=3 or so.
5: #
6: # !! Complex calculations really aren't checked that carefully.
7: # !! Only real valued complex numbers are used in tests.
8: 
9: from __future__ import division, print_function, absolute_import
10: 
11: from numpy import float32, float64, complex64, complex128, arange, array, \
12:                   zeros, shape, transpose, newaxis, common_type, conjugate
13: 
14: from scipy.linalg import _fblas as fblas
15: 
16: from scipy._lib.six import xrange
17: 
18: from numpy.testing import assert_array_equal, \
19:     assert_allclose, assert_array_almost_equal, assert_
20: 
21: # decimal accuracy to require between Python and LAPACK/BLAS calculations
22: accuracy = 5
23: 
24: # Since numpy.dot likely uses the same blas, use this routine
25: # to check.
26: 
27: 
28: def matrixmultiply(a, b):
29:     if len(b.shape) == 1:
30:         b_is_vector = True
31:         b = b[:, newaxis]
32:     else:
33:         b_is_vector = False
34:     assert_(a.shape[1] == b.shape[0])
35:     c = zeros((a.shape[0], b.shape[1]), common_type(a, b))
36:     for i in xrange(a.shape[0]):
37:         for j in xrange(b.shape[1]):
38:             s = 0
39:             for k in xrange(a.shape[1]):
40:                 s += a[i, k] * b[k, j]
41:             c[i, j] = s
42:     if b_is_vector:
43:         c = c.reshape((a.shape[0],))
44:     return c
45: 
46: ##################################################
47: # Test blas ?axpy
48: 
49: 
50: class BaseAxpy(object):
51:     ''' Mixin class for axpy tests '''
52: 
53:     def test_default_a(self):
54:         x = arange(3., dtype=self.dtype)
55:         y = arange(3., dtype=x.dtype)
56:         real_y = x*1.+y
57:         y = self.blas_func(x, y)
58:         assert_array_equal(real_y, y)
59: 
60:     def test_simple(self):
61:         x = arange(3., dtype=self.dtype)
62:         y = arange(3., dtype=x.dtype)
63:         real_y = x*3.+y
64:         y = self.blas_func(x, y, a=3.)
65:         assert_array_equal(real_y, y)
66: 
67:     def test_x_stride(self):
68:         x = arange(6., dtype=self.dtype)
69:         y = zeros(3, x.dtype)
70:         y = arange(3., dtype=x.dtype)
71:         real_y = x[::2]*3.+y
72:         y = self.blas_func(x, y, a=3., n=3, incx=2)
73:         assert_array_equal(real_y, y)
74: 
75:     def test_y_stride(self):
76:         x = arange(3., dtype=self.dtype)
77:         y = zeros(6, x.dtype)
78:         real_y = x*3.+y[::2]
79:         y = self.blas_func(x, y, a=3., n=3, incy=2)
80:         assert_array_equal(real_y, y[::2])
81: 
82:     def test_x_and_y_stride(self):
83:         x = arange(12., dtype=self.dtype)
84:         y = zeros(6, x.dtype)
85:         real_y = x[::4]*3.+y[::2]
86:         y = self.blas_func(x, y, a=3., n=3, incx=4, incy=2)
87:         assert_array_equal(real_y, y[::2])
88: 
89:     def test_x_bad_size(self):
90:         x = arange(12., dtype=self.dtype)
91:         y = zeros(6, x.dtype)
92:         try:
93:             self.blas_func(x, y, n=4, incx=5)
94:         except:  # what kind of error should be caught?
95:             return
96:         # should catch error and never get here
97:         assert_(0)
98: 
99:     def test_y_bad_size(self):
100:         x = arange(12., dtype=self.dtype)
101:         y = zeros(6, x.dtype)
102:         try:
103:             self.blas_func(x, y, n=3, incy=5)
104:         except:  # what kind of error should be caught?
105:             return
106:         # should catch error and never get here
107:         assert_(0)
108: 
109: 
110: try:
111:     class TestSaxpy(BaseAxpy):
112:         blas_func = fblas.saxpy
113:         dtype = float32
114: except AttributeError:
115:     class TestSaxpy:
116:         pass
117: 
118: 
119: class TestDaxpy(BaseAxpy):
120:     blas_func = fblas.daxpy
121:     dtype = float64
122: 
123: 
124: try:
125:     class TestCaxpy(BaseAxpy):
126:         blas_func = fblas.caxpy
127:         dtype = complex64
128: except AttributeError:
129:     class TestCaxpy:
130:         pass
131: 
132: 
133: class TestZaxpy(BaseAxpy):
134:     blas_func = fblas.zaxpy
135:     dtype = complex128
136: 
137: 
138: ##################################################
139: # Test blas ?scal
140: 
141: class BaseScal(object):
142:     ''' Mixin class for scal testing '''
143: 
144:     def test_simple(self):
145:         x = arange(3., dtype=self.dtype)
146:         real_x = x*3.
147:         x = self.blas_func(3., x)
148:         assert_array_equal(real_x, x)
149: 
150:     def test_x_stride(self):
151:         x = arange(6., dtype=self.dtype)
152:         real_x = x.copy()
153:         real_x[::2] = x[::2]*array(3., self.dtype)
154:         x = self.blas_func(3., x, n=3, incx=2)
155:         assert_array_equal(real_x, x)
156: 
157:     def test_x_bad_size(self):
158:         x = arange(12., dtype=self.dtype)
159:         try:
160:             self.blas_func(2., x, n=4, incx=5)
161:         except:  # what kind of error should be caught?
162:             return
163:         # should catch error and never get here
164:         assert_(0)
165: 
166: 
167: try:
168:     class TestSscal(BaseScal):
169:         blas_func = fblas.sscal
170:         dtype = float32
171: except AttributeError:
172:     class TestSscal:
173:         pass
174: 
175: 
176: class TestDscal(BaseScal):
177:     blas_func = fblas.dscal
178:     dtype = float64
179: 
180: 
181: try:
182:     class TestCscal(BaseScal):
183:         blas_func = fblas.cscal
184:         dtype = complex64
185: except AttributeError:
186:     class TestCscal:
187:         pass
188: 
189: 
190: class TestZscal(BaseScal):
191:     blas_func = fblas.zscal
192:     dtype = complex128
193: 
194: 
195: ##################################################
196: # Test blas ?copy
197: 
198: class BaseCopy(object):
199:     ''' Mixin class for copy testing '''
200: 
201:     def test_simple(self):
202:         x = arange(3., dtype=self.dtype)
203:         y = zeros(shape(x), x.dtype)
204:         y = self.blas_func(x, y)
205:         assert_array_equal(x, y)
206: 
207:     def test_x_stride(self):
208:         x = arange(6., dtype=self.dtype)
209:         y = zeros(3, x.dtype)
210:         y = self.blas_func(x, y, n=3, incx=2)
211:         assert_array_equal(x[::2], y)
212: 
213:     def test_y_stride(self):
214:         x = arange(3., dtype=self.dtype)
215:         y = zeros(6, x.dtype)
216:         y = self.blas_func(x, y, n=3, incy=2)
217:         assert_array_equal(x, y[::2])
218: 
219:     def test_x_and_y_stride(self):
220:         x = arange(12., dtype=self.dtype)
221:         y = zeros(6, x.dtype)
222:         y = self.blas_func(x, y, n=3, incx=4, incy=2)
223:         assert_array_equal(x[::4], y[::2])
224: 
225:     def test_x_bad_size(self):
226:         x = arange(12., dtype=self.dtype)
227:         y = zeros(6, x.dtype)
228:         try:
229:             self.blas_func(x, y, n=4, incx=5)
230:         except:  # what kind of error should be caught?
231:             return
232:         # should catch error and never get here
233:         assert_(0)
234: 
235:     def test_y_bad_size(self):
236:         x = arange(12., dtype=self.dtype)
237:         y = zeros(6, x.dtype)
238:         try:
239:             self.blas_func(x, y, n=3, incy=5)
240:         except:  # what kind of error should be caught?
241:             return
242:         # should catch error and never get here
243:         assert_(0)
244: 
245:     # def test_y_bad_type(self):
246:     ##   Hmmm. Should this work?  What should be the output.
247:     #    x = arange(3.,dtype=self.dtype)
248:     #    y = zeros(shape(x))
249:     #    self.blas_func(x,y)
250:     #    assert_array_equal(x,y)
251: 
252: 
253: try:
254:     class TestScopy(BaseCopy):
255:         blas_func = fblas.scopy
256:         dtype = float32
257: except AttributeError:
258:     class TestScopy:
259:         pass
260: 
261: 
262: class TestDcopy(BaseCopy):
263:     blas_func = fblas.dcopy
264:     dtype = float64
265: 
266: 
267: try:
268:     class TestCcopy(BaseCopy):
269:         blas_func = fblas.ccopy
270:         dtype = complex64
271: except AttributeError:
272:     class TestCcopy:
273:         pass
274: 
275: 
276: class TestZcopy(BaseCopy):
277:     blas_func = fblas.zcopy
278:     dtype = complex128
279: 
280: 
281: ##################################################
282: # Test blas ?swap
283: 
284: class BaseSwap(object):
285:     ''' Mixin class for swap tests '''
286: 
287:     def test_simple(self):
288:         x = arange(3., dtype=self.dtype)
289:         y = zeros(shape(x), x.dtype)
290:         desired_x = y.copy()
291:         desired_y = x.copy()
292:         x, y = self.blas_func(x, y)
293:         assert_array_equal(desired_x, x)
294:         assert_array_equal(desired_y, y)
295: 
296:     def test_x_stride(self):
297:         x = arange(6., dtype=self.dtype)
298:         y = zeros(3, x.dtype)
299:         desired_x = y.copy()
300:         desired_y = x.copy()[::2]
301:         x, y = self.blas_func(x, y, n=3, incx=2)
302:         assert_array_equal(desired_x, x[::2])
303:         assert_array_equal(desired_y, y)
304: 
305:     def test_y_stride(self):
306:         x = arange(3., dtype=self.dtype)
307:         y = zeros(6, x.dtype)
308:         desired_x = y.copy()[::2]
309:         desired_y = x.copy()
310:         x, y = self.blas_func(x, y, n=3, incy=2)
311:         assert_array_equal(desired_x, x)
312:         assert_array_equal(desired_y, y[::2])
313: 
314:     def test_x_and_y_stride(self):
315:         x = arange(12., dtype=self.dtype)
316:         y = zeros(6, x.dtype)
317:         desired_x = y.copy()[::2]
318:         desired_y = x.copy()[::4]
319:         x, y = self.blas_func(x, y, n=3, incx=4, incy=2)
320:         assert_array_equal(desired_x, x[::4])
321:         assert_array_equal(desired_y, y[::2])
322: 
323:     def test_x_bad_size(self):
324:         x = arange(12., dtype=self.dtype)
325:         y = zeros(6, x.dtype)
326:         try:
327:             self.blas_func(x, y, n=4, incx=5)
328:         except:  # what kind of error should be caught?
329:             return
330:         # should catch error and never get here
331:         assert_(0)
332: 
333:     def test_y_bad_size(self):
334:         x = arange(12., dtype=self.dtype)
335:         y = zeros(6, x.dtype)
336:         try:
337:             self.blas_func(x, y, n=3, incy=5)
338:         except:  # what kind of error should be caught?
339:             return
340:         # should catch error and never get here
341:         assert_(0)
342: 
343: 
344: try:
345:     class TestSswap(BaseSwap):
346:         blas_func = fblas.sswap
347:         dtype = float32
348: except AttributeError:
349:     class TestSswap:
350:         pass
351: 
352: 
353: class TestDswap(BaseSwap):
354:     blas_func = fblas.dswap
355:     dtype = float64
356: 
357: 
358: try:
359:     class TestCswap(BaseSwap):
360:         blas_func = fblas.cswap
361:         dtype = complex64
362: except AttributeError:
363:     class TestCswap:
364:         pass
365: 
366: 
367: class TestZswap(BaseSwap):
368:     blas_func = fblas.zswap
369:     dtype = complex128
370: 
371: ##################################################
372: # Test blas ?gemv
373: # This will be a mess to test all cases.
374: 
375: 
376: class BaseGemv(object):
377:     ''' Mixin class for gemv tests '''
378: 
379:     def get_data(self, x_stride=1, y_stride=1):
380:         mult = array(1, dtype=self.dtype)
381:         if self.dtype in [complex64, complex128]:
382:             mult = array(1+1j, dtype=self.dtype)
383:         from numpy.random import normal, seed
384:         seed(1234)
385:         alpha = array(1., dtype=self.dtype) * mult
386:         beta = array(1., dtype=self.dtype) * mult
387:         a = normal(0., 1., (3, 3)).astype(self.dtype) * mult
388:         x = arange(shape(a)[0]*x_stride, dtype=self.dtype) * mult
389:         y = arange(shape(a)[1]*y_stride, dtype=self.dtype) * mult
390:         return alpha, beta, a, x, y
391: 
392:     def test_simple(self):
393:         alpha, beta, a, x, y = self.get_data()
394:         desired_y = alpha*matrixmultiply(a, x)+beta*y
395:         y = self.blas_func(alpha, a, x, beta, y)
396:         assert_array_almost_equal(desired_y, y)
397: 
398:     def test_default_beta_y(self):
399:         alpha, beta, a, x, y = self.get_data()
400:         desired_y = matrixmultiply(a, x)
401:         y = self.blas_func(1, a, x)
402:         assert_array_almost_equal(desired_y, y)
403: 
404:     def test_simple_transpose(self):
405:         alpha, beta, a, x, y = self.get_data()
406:         desired_y = alpha*matrixmultiply(transpose(a), x)+beta*y
407:         y = self.blas_func(alpha, a, x, beta, y, trans=1)
408:         assert_array_almost_equal(desired_y, y)
409: 
410:     def test_simple_transpose_conj(self):
411:         alpha, beta, a, x, y = self.get_data()
412:         desired_y = alpha*matrixmultiply(transpose(conjugate(a)), x)+beta*y
413:         y = self.blas_func(alpha, a, x, beta, y, trans=2)
414:         assert_array_almost_equal(desired_y, y)
415: 
416:     def test_x_stride(self):
417:         alpha, beta, a, x, y = self.get_data(x_stride=2)
418:         desired_y = alpha*matrixmultiply(a, x[::2])+beta*y
419:         y = self.blas_func(alpha, a, x, beta, y, incx=2)
420:         assert_array_almost_equal(desired_y, y)
421: 
422:     def test_x_stride_transpose(self):
423:         alpha, beta, a, x, y = self.get_data(x_stride=2)
424:         desired_y = alpha*matrixmultiply(transpose(a), x[::2])+beta*y
425:         y = self.blas_func(alpha, a, x, beta, y, trans=1, incx=2)
426:         assert_array_almost_equal(desired_y, y)
427: 
428:     def test_x_stride_assert(self):
429:         # What is the use of this test?
430:         alpha, beta, a, x, y = self.get_data(x_stride=2)
431:         try:
432:             y = self.blas_func(1, a, x, 1, y, trans=0, incx=3)
433:             assert_(0)
434:         except:
435:             pass
436:         try:
437:             y = self.blas_func(1, a, x, 1, y, trans=1, incx=3)
438:             assert_(0)
439:         except:
440:             pass
441: 
442:     def test_y_stride(self):
443:         alpha, beta, a, x, y = self.get_data(y_stride=2)
444:         desired_y = y.copy()
445:         desired_y[::2] = alpha*matrixmultiply(a, x)+beta*y[::2]
446:         y = self.blas_func(alpha, a, x, beta, y, incy=2)
447:         assert_array_almost_equal(desired_y, y)
448: 
449:     def test_y_stride_transpose(self):
450:         alpha, beta, a, x, y = self.get_data(y_stride=2)
451:         desired_y = y.copy()
452:         desired_y[::2] = alpha*matrixmultiply(transpose(a), x)+beta*y[::2]
453:         y = self.blas_func(alpha, a, x, beta, y, trans=1, incy=2)
454:         assert_array_almost_equal(desired_y, y)
455: 
456:     def test_y_stride_assert(self):
457:         # What is the use of this test?
458:         alpha, beta, a, x, y = self.get_data(y_stride=2)
459:         try:
460:             y = self.blas_func(1, a, x, 1, y, trans=0, incy=3)
461:             assert_(0)
462:         except:
463:             pass
464:         try:
465:             y = self.blas_func(1, a, x, 1, y, trans=1, incy=3)
466:             assert_(0)
467:         except:
468:             pass
469: 
470: 
471: try:
472:     class TestSgemv(BaseGemv):
473:         blas_func = fblas.sgemv
474:         dtype = float32
475: 
476:         def test_sgemv_on_osx(self):
477:             from itertools import product
478:             import sys
479:             import numpy as np
480: 
481:             if sys.platform != 'darwin':
482:                 return
483: 
484:             def aligned_array(shape, align, dtype, order='C'):
485:                 # Make array shape `shape` with aligned at `align` bytes
486:                 d = dtype()
487:                 # Make array of correct size with `align` extra bytes
488:                 N = np.prod(shape)
489:                 tmp = np.zeros(N * d.nbytes + align, dtype=np.uint8)
490:                 address = tmp.__array_interface__["data"][0]
491:                 # Find offset into array giving desired alignment
492:                 for offset in range(align):
493:                     if (address + offset) % align == 0:
494:                         break
495:                 tmp = tmp[offset:offset+N*d.nbytes].view(dtype=dtype)
496:                 return tmp.reshape(shape, order=order)
497: 
498:             def as_aligned(arr, align, dtype, order='C'):
499:                 # Copy `arr` into an aligned array with same shape
500:                 aligned = aligned_array(arr.shape, align, dtype, order)
501:                 aligned[:] = arr[:]
502:                 return aligned
503: 
504:             def assert_dot_close(A, X, desired):
505:                 assert_allclose(self.blas_func(1.0, A, X), desired,
506:                                 rtol=1e-5, atol=1e-7)
507: 
508:             testdata = product((15, 32), (10000,), (200, 89), ('C', 'F'))
509:             for align, m, n, a_order in testdata:
510:                 A_d = np.random.rand(m, n)
511:                 X_d = np.random.rand(n)
512:                 desired = np.dot(A_d, X_d)
513:                 # Calculation with aligned single precision
514:                 A_f = as_aligned(A_d, align, np.float32, order=a_order)
515:                 X_f = as_aligned(X_d, align, np.float32, order=a_order)
516:                 assert_dot_close(A_f, X_f, desired)
517: 
518: except AttributeError:
519:     class TestSgemv:
520:         pass
521: 
522: 
523: class TestDgemv(BaseGemv):
524:     blas_func = fblas.dgemv
525:     dtype = float64
526: 
527: 
528: try:
529:     class TestCgemv(BaseGemv):
530:         blas_func = fblas.cgemv
531:         dtype = complex64
532: except AttributeError:
533:     class TestCgemv:
534:         pass
535: 
536: 
537: class TestZgemv(BaseGemv):
538:     blas_func = fblas.zgemv
539:     dtype = complex128
540: 
541: 
542: '''
543: ##################################################
544: ### Test blas ?ger
545: ### This will be a mess to test all cases.
546: 
547: class BaseGer(object):
548:     def get_data(self,x_stride=1,y_stride=1):
549:         from numpy.random import normal, seed
550:         seed(1234)
551:         alpha = array(1., dtype = self.dtype)
552:         a = normal(0.,1.,(3,3)).astype(self.dtype)
553:         x = arange(shape(a)[0]*x_stride,dtype=self.dtype)
554:         y = arange(shape(a)[1]*y_stride,dtype=self.dtype)
555:         return alpha,a,x,y
556:     def test_simple(self):
557:         alpha,a,x,y = self.get_data()
558:         # tranpose takes care of Fortran vs. C(and Python) memory layout
559:         desired_a = alpha*transpose(x[:,newaxis]*y) + a
560:         self.blas_func(x,y,a)
561:         assert_array_almost_equal(desired_a,a)
562:     def test_x_stride(self):
563:         alpha,a,x,y = self.get_data(x_stride=2)
564:         desired_a = alpha*transpose(x[::2,newaxis]*y) + a
565:         self.blas_func(x,y,a,incx=2)
566:         assert_array_almost_equal(desired_a,a)
567:     def test_x_stride_assert(self):
568:         alpha,a,x,y = self.get_data(x_stride=2)
569:         try:
570:             self.blas_func(x,y,a,incx=3)
571:             assert(0)
572:         except:
573:             pass
574:     def test_y_stride(self):
575:         alpha,a,x,y = self.get_data(y_stride=2)
576:         desired_a = alpha*transpose(x[:,newaxis]*y[::2]) + a
577:         self.blas_func(x,y,a,incy=2)
578:         assert_array_almost_equal(desired_a,a)
579: 
580:     def test_y_stride_assert(self):
581:         alpha,a,x,y = self.get_data(y_stride=2)
582:         try:
583:             self.blas_func(a,x,y,incy=3)
584:             assert(0)
585:         except:
586:             pass
587: 
588: class TestSger(BaseGer):
589:     blas_func = fblas.sger
590:     dtype = float32
591: class TestDger(BaseGer):
592:     blas_func = fblas.dger
593:     dtype = float64
594: '''
595: ##################################################
596: # Test blas ?gerc
597: # This will be a mess to test all cases.
598: 
599: '''
600: class BaseGerComplex(BaseGer):
601:     def get_data(self,x_stride=1,y_stride=1):
602:         from numpy.random import normal, seed
603:         seed(1234)
604:         alpha = array(1+1j, dtype = self.dtype)
605:         a = normal(0.,1.,(3,3)).astype(self.dtype)
606:         a = a + normal(0.,1.,(3,3)) * array(1j, dtype = self.dtype)
607:         x = normal(0.,1.,shape(a)[0]*x_stride).astype(self.dtype)
608:         x = x + x * array(1j, dtype = self.dtype)
609:         y = normal(0.,1.,shape(a)[1]*y_stride).astype(self.dtype)
610:         y = y + y * array(1j, dtype = self.dtype)
611:         return alpha,a,x,y
612:     def test_simple(self):
613:         alpha,a,x,y = self.get_data()
614:         # tranpose takes care of Fortran vs. C(and Python) memory layout
615:         a = a * array(0.,dtype = self.dtype)
616:         #desired_a = alpha*transpose(x[:,newaxis]*self.transform(y)) + a
617:         desired_a = alpha*transpose(x[:,newaxis]*y) + a
618:         #self.blas_func(x,y,a,alpha = alpha)
619:         fblas.cgeru(x,y,a,alpha = alpha)
620:         assert_array_almost_equal(desired_a,a)
621: 
622:     #def test_x_stride(self):
623:     #    alpha,a,x,y = self.get_data(x_stride=2)
624:     #    desired_a = alpha*transpose(x[::2,newaxis]*self.transform(y)) + a
625:     #    self.blas_func(x,y,a,incx=2)
626:     #    assert_array_almost_equal(desired_a,a)
627:     #def test_y_stride(self):
628:     #    alpha,a,x,y = self.get_data(y_stride=2)
629:     #    desired_a = alpha*transpose(x[:,newaxis]*self.transform(y[::2])) + a
630:     #    self.blas_func(x,y,a,incy=2)
631:     #    assert_array_almost_equal(desired_a,a)
632: 
633: class TestCgeru(BaseGerComplex):
634:     blas_func = fblas.cgeru
635:     dtype = complex64
636:     def transform(self,x):
637:         return x
638: class TestZgeru(BaseGerComplex):
639:     blas_func = fblas.zgeru
640:     dtype = complex128
641:     def transform(self,x):
642:         return x
643: 
644: class TestCgerc(BaseGerComplex):
645:     blas_func = fblas.cgerc
646:     dtype = complex64
647:     def transform(self,x):
648:         return conjugate(x)
649: 
650: class TestZgerc(BaseGerComplex):
651:     blas_func = fblas.zgerc
652:     dtype = complex128
653:     def transform(self,x):
654:         return conjugate(x)
655: '''
656: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy import float32, float64, complex64, complex128, arange, array, zeros, shape, transpose, newaxis, common_type, conjugate' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_91260 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_91260) is not StypyTypeError):

    if (import_91260 != 'pyd_module'):
        __import__(import_91260)
        sys_modules_91261 = sys.modules[import_91260]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', sys_modules_91261.module_type_store, module_type_store, ['float32', 'float64', 'complex64', 'complex128', 'arange', 'array', 'zeros', 'shape', 'transpose', 'newaxis', 'common_type', 'conjugate'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_91261, sys_modules_91261.module_type_store, module_type_store)
    else:
        from numpy import float32, float64, complex64, complex128, arange, array, zeros, shape, transpose, newaxis, common_type, conjugate

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', None, module_type_store, ['float32', 'float64', 'complex64', 'complex128', 'arange', 'array', 'zeros', 'shape', 'transpose', 'newaxis', 'common_type', 'conjugate'], [float32, float64, complex64, complex128, arange, array, zeros, shape, transpose, newaxis, common_type, conjugate])

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_91260)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.linalg import fblas' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_91262 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg')

if (type(import_91262) is not StypyTypeError):

    if (import_91262 != 'pyd_module'):
        __import__(import_91262)
        sys_modules_91263 = sys.modules[import_91262]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg', sys_modules_91263.module_type_store, module_type_store, ['_fblas'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_91263, sys_modules_91263.module_type_store, module_type_store)
    else:
        from scipy.linalg import _fblas as fblas

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg', None, module_type_store, ['_fblas'], [fblas])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg', import_91262)

# Adding an alias
module_type_store.add_alias('fblas', '_fblas')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy._lib.six import xrange' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_91264 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six')

if (type(import_91264) is not StypyTypeError):

    if (import_91264 != 'pyd_module'):
        __import__(import_91264)
        sys_modules_91265 = sys.modules[import_91264]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six', sys_modules_91265.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_91265, sys_modules_91265.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six', import_91264)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.testing import assert_array_equal, assert_allclose, assert_array_almost_equal, assert_' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_91266 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing')

if (type(import_91266) is not StypyTypeError):

    if (import_91266 != 'pyd_module'):
        __import__(import_91266)
        sys_modules_91267 = sys.modules[import_91266]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing', sys_modules_91267.module_type_store, module_type_store, ['assert_array_equal', 'assert_allclose', 'assert_array_almost_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_91267, sys_modules_91267.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_equal, assert_allclose, assert_array_almost_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing', None, module_type_store, ['assert_array_equal', 'assert_allclose', 'assert_array_almost_equal', 'assert_'], [assert_array_equal, assert_allclose, assert_array_almost_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.testing', import_91266)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')


# Assigning a Num to a Name (line 22):

# Assigning a Num to a Name (line 22):
int_91268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'int')
# Assigning a type to the variable 'accuracy' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'accuracy', int_91268)

@norecursion
def matrixmultiply(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'matrixmultiply'
    module_type_store = module_type_store.open_function_context('matrixmultiply', 28, 0, False)
    
    # Passed parameters checking function
    matrixmultiply.stypy_localization = localization
    matrixmultiply.stypy_type_of_self = None
    matrixmultiply.stypy_type_store = module_type_store
    matrixmultiply.stypy_function_name = 'matrixmultiply'
    matrixmultiply.stypy_param_names_list = ['a', 'b']
    matrixmultiply.stypy_varargs_param_name = None
    matrixmultiply.stypy_kwargs_param_name = None
    matrixmultiply.stypy_call_defaults = defaults
    matrixmultiply.stypy_call_varargs = varargs
    matrixmultiply.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'matrixmultiply', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'matrixmultiply', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'matrixmultiply(...)' code ##################

    
    
    
    # Call to len(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'b' (line 29)
    b_91270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'b', False)
    # Obtaining the member 'shape' of a type (line 29)
    shape_91271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 11), b_91270, 'shape')
    # Processing the call keyword arguments (line 29)
    kwargs_91272 = {}
    # Getting the type of 'len' (line 29)
    len_91269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), 'len', False)
    # Calling len(args, kwargs) (line 29)
    len_call_result_91273 = invoke(stypy.reporting.localization.Localization(__file__, 29, 7), len_91269, *[shape_91271], **kwargs_91272)
    
    int_91274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'int')
    # Applying the binary operator '==' (line 29)
    result_eq_91275 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 7), '==', len_call_result_91273, int_91274)
    
    # Testing the type of an if condition (line 29)
    if_condition_91276 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 4), result_eq_91275)
    # Assigning a type to the variable 'if_condition_91276' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'if_condition_91276', if_condition_91276)
    # SSA begins for if statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 30):
    
    # Assigning a Name to a Name (line 30):
    # Getting the type of 'True' (line 30)
    True_91277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'True')
    # Assigning a type to the variable 'b_is_vector' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'b_is_vector', True_91277)
    
    # Assigning a Subscript to a Name (line 31):
    
    # Assigning a Subscript to a Name (line 31):
    
    # Obtaining the type of the subscript
    slice_91278 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 31, 12), None, None, None)
    # Getting the type of 'newaxis' (line 31)
    newaxis_91279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'newaxis')
    # Getting the type of 'b' (line 31)
    b_91280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'b')
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___91281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), b_91280, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_91282 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), getitem___91281, (slice_91278, newaxis_91279))
    
    # Assigning a type to the variable 'b' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'b', subscript_call_result_91282)
    # SSA branch for the else part of an if statement (line 29)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 33):
    
    # Assigning a Name to a Name (line 33):
    # Getting the type of 'False' (line 33)
    False_91283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'False')
    # Assigning a type to the variable 'b_is_vector' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'b_is_vector', False_91283)
    # SSA join for if statement (line 29)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_(...): (line 34)
    # Processing the call arguments (line 34)
    
    
    # Obtaining the type of the subscript
    int_91285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'int')
    # Getting the type of 'a' (line 34)
    a_91286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'a', False)
    # Obtaining the member 'shape' of a type (line 34)
    shape_91287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), a_91286, 'shape')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___91288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), shape_91287, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_91289 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), getitem___91288, int_91285)
    
    
    # Obtaining the type of the subscript
    int_91290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'int')
    # Getting the type of 'b' (line 34)
    b_91291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'b', False)
    # Obtaining the member 'shape' of a type (line 34)
    shape_91292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 26), b_91291, 'shape')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___91293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 26), shape_91292, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_91294 = invoke(stypy.reporting.localization.Localization(__file__, 34, 26), getitem___91293, int_91290)
    
    # Applying the binary operator '==' (line 34)
    result_eq_91295 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 12), '==', subscript_call_result_91289, subscript_call_result_91294)
    
    # Processing the call keyword arguments (line 34)
    kwargs_91296 = {}
    # Getting the type of 'assert_' (line 34)
    assert__91284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 34)
    assert__call_result_91297 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), assert__91284, *[result_eq_91295], **kwargs_91296)
    
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to zeros(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_91299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    # Adding element type (line 35)
    
    # Obtaining the type of the subscript
    int_91300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'int')
    # Getting the type of 'a' (line 35)
    a_91301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'a', False)
    # Obtaining the member 'shape' of a type (line 35)
    shape_91302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), a_91301, 'shape')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___91303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), shape_91302, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_91304 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), getitem___91303, int_91300)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), tuple_91299, subscript_call_result_91304)
    # Adding element type (line 35)
    
    # Obtaining the type of the subscript
    int_91305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 35), 'int')
    # Getting the type of 'b' (line 35)
    b_91306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 27), 'b', False)
    # Obtaining the member 'shape' of a type (line 35)
    shape_91307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 27), b_91306, 'shape')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___91308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 27), shape_91307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_91309 = invoke(stypy.reporting.localization.Localization(__file__, 35, 27), getitem___91308, int_91305)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), tuple_91299, subscript_call_result_91309)
    
    
    # Call to common_type(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'a' (line 35)
    a_91311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 52), 'a', False)
    # Getting the type of 'b' (line 35)
    b_91312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 55), 'b', False)
    # Processing the call keyword arguments (line 35)
    kwargs_91313 = {}
    # Getting the type of 'common_type' (line 35)
    common_type_91310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 40), 'common_type', False)
    # Calling common_type(args, kwargs) (line 35)
    common_type_call_result_91314 = invoke(stypy.reporting.localization.Localization(__file__, 35, 40), common_type_91310, *[a_91311, b_91312], **kwargs_91313)
    
    # Processing the call keyword arguments (line 35)
    kwargs_91315 = {}
    # Getting the type of 'zeros' (line 35)
    zeros_91298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'zeros', False)
    # Calling zeros(args, kwargs) (line 35)
    zeros_call_result_91316 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), zeros_91298, *[tuple_91299, common_type_call_result_91314], **kwargs_91315)
    
    # Assigning a type to the variable 'c' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'c', zeros_call_result_91316)
    
    
    # Call to xrange(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining the type of the subscript
    int_91318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'int')
    # Getting the type of 'a' (line 36)
    a_91319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'a', False)
    # Obtaining the member 'shape' of a type (line 36)
    shape_91320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 20), a_91319, 'shape')
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___91321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 20), shape_91320, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_91322 = invoke(stypy.reporting.localization.Localization(__file__, 36, 20), getitem___91321, int_91318)
    
    # Processing the call keyword arguments (line 36)
    kwargs_91323 = {}
    # Getting the type of 'xrange' (line 36)
    xrange_91317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 36)
    xrange_call_result_91324 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), xrange_91317, *[subscript_call_result_91322], **kwargs_91323)
    
    # Testing the type of a for loop iterable (line 36)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 36, 4), xrange_call_result_91324)
    # Getting the type of the for loop variable (line 36)
    for_loop_var_91325 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 36, 4), xrange_call_result_91324)
    # Assigning a type to the variable 'i' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'i', for_loop_var_91325)
    # SSA begins for a for statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to xrange(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Obtaining the type of the subscript
    int_91327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 32), 'int')
    # Getting the type of 'b' (line 37)
    b_91328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'b', False)
    # Obtaining the member 'shape' of a type (line 37)
    shape_91329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 24), b_91328, 'shape')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___91330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 24), shape_91329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_91331 = invoke(stypy.reporting.localization.Localization(__file__, 37, 24), getitem___91330, int_91327)
    
    # Processing the call keyword arguments (line 37)
    kwargs_91332 = {}
    # Getting the type of 'xrange' (line 37)
    xrange_91326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'xrange', False)
    # Calling xrange(args, kwargs) (line 37)
    xrange_call_result_91333 = invoke(stypy.reporting.localization.Localization(__file__, 37, 17), xrange_91326, *[subscript_call_result_91331], **kwargs_91332)
    
    # Testing the type of a for loop iterable (line 37)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 8), xrange_call_result_91333)
    # Getting the type of the for loop variable (line 37)
    for_loop_var_91334 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 8), xrange_call_result_91333)
    # Assigning a type to the variable 'j' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'j', for_loop_var_91334)
    # SSA begins for a for statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Num to a Name (line 38):
    
    # Assigning a Num to a Name (line 38):
    int_91335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'int')
    # Assigning a type to the variable 's' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 's', int_91335)
    
    
    # Call to xrange(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Obtaining the type of the subscript
    int_91337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'int')
    # Getting the type of 'a' (line 39)
    a_91338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'a', False)
    # Obtaining the member 'shape' of a type (line 39)
    shape_91339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 28), a_91338, 'shape')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___91340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 28), shape_91339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_91341 = invoke(stypy.reporting.localization.Localization(__file__, 39, 28), getitem___91340, int_91337)
    
    # Processing the call keyword arguments (line 39)
    kwargs_91342 = {}
    # Getting the type of 'xrange' (line 39)
    xrange_91336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'xrange', False)
    # Calling xrange(args, kwargs) (line 39)
    xrange_call_result_91343 = invoke(stypy.reporting.localization.Localization(__file__, 39, 21), xrange_91336, *[subscript_call_result_91341], **kwargs_91342)
    
    # Testing the type of a for loop iterable (line 39)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 12), xrange_call_result_91343)
    # Getting the type of the for loop variable (line 39)
    for_loop_var_91344 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 12), xrange_call_result_91343)
    # Assigning a type to the variable 'k' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'k', for_loop_var_91344)
    # SSA begins for a for statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 's' (line 40)
    s_91345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 's')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 40)
    tuple_91346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 40)
    # Adding element type (line 40)
    # Getting the type of 'i' (line 40)
    i_91347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 23), tuple_91346, i_91347)
    # Adding element type (line 40)
    # Getting the type of 'k' (line 40)
    k_91348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 23), tuple_91346, k_91348)
    
    # Getting the type of 'a' (line 40)
    a_91349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), 'a')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___91350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 21), a_91349, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_91351 = invoke(stypy.reporting.localization.Localization(__file__, 40, 21), getitem___91350, tuple_91346)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 40)
    tuple_91352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 40)
    # Adding element type (line 40)
    # Getting the type of 'k' (line 40)
    k_91353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 33), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 33), tuple_91352, k_91353)
    # Adding element type (line 40)
    # Getting the type of 'j' (line 40)
    j_91354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 36), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 33), tuple_91352, j_91354)
    
    # Getting the type of 'b' (line 40)
    b_91355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 31), 'b')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___91356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 31), b_91355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_91357 = invoke(stypy.reporting.localization.Localization(__file__, 40, 31), getitem___91356, tuple_91352)
    
    # Applying the binary operator '*' (line 40)
    result_mul_91358 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 21), '*', subscript_call_result_91351, subscript_call_result_91357)
    
    # Applying the binary operator '+=' (line 40)
    result_iadd_91359 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 16), '+=', s_91345, result_mul_91358)
    # Assigning a type to the variable 's' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 's', result_iadd_91359)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 41):
    
    # Assigning a Name to a Subscript (line 41):
    # Getting the type of 's' (line 41)
    s_91360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 's')
    # Getting the type of 'c' (line 41)
    c_91361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'c')
    
    # Obtaining an instance of the builtin type 'tuple' (line 41)
    tuple_91362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 41)
    # Adding element type (line 41)
    # Getting the type of 'i' (line 41)
    i_91363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), tuple_91362, i_91363)
    # Adding element type (line 41)
    # Getting the type of 'j' (line 41)
    j_91364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), tuple_91362, j_91364)
    
    # Storing an element on a container (line 41)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 12), c_91361, (tuple_91362, s_91360))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'b_is_vector' (line 42)
    b_is_vector_91365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 7), 'b_is_vector')
    # Testing the type of an if condition (line 42)
    if_condition_91366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 4), b_is_vector_91365)
    # Assigning a type to the variable 'if_condition_91366' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_condition_91366', if_condition_91366)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 43):
    
    # Assigning a Call to a Name (line 43):
    
    # Call to reshape(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Obtaining an instance of the builtin type 'tuple' (line 43)
    tuple_91369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 43)
    # Adding element type (line 43)
    
    # Obtaining the type of the subscript
    int_91370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'int')
    # Getting the type of 'a' (line 43)
    a_91371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'a', False)
    # Obtaining the member 'shape' of a type (line 43)
    shape_91372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 23), a_91371, 'shape')
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___91373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 23), shape_91372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_91374 = invoke(stypy.reporting.localization.Localization(__file__, 43, 23), getitem___91373, int_91370)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), tuple_91369, subscript_call_result_91374)
    
    # Processing the call keyword arguments (line 43)
    kwargs_91375 = {}
    # Getting the type of 'c' (line 43)
    c_91367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'c', False)
    # Obtaining the member 'reshape' of a type (line 43)
    reshape_91368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), c_91367, 'reshape')
    # Calling reshape(args, kwargs) (line 43)
    reshape_call_result_91376 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), reshape_91368, *[tuple_91369], **kwargs_91375)
    
    # Assigning a type to the variable 'c' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'c', reshape_call_result_91376)
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c' (line 44)
    c_91377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type', c_91377)
    
    # ################# End of 'matrixmultiply(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'matrixmultiply' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_91378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_91378)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'matrixmultiply'
    return stypy_return_type_91378

# Assigning a type to the variable 'matrixmultiply' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'matrixmultiply', matrixmultiply)
# Declaration of the 'BaseAxpy' class

class BaseAxpy(object, ):
    str_91379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'str', ' Mixin class for axpy tests ')

    @norecursion
    def test_default_a(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_default_a'
        module_type_store = module_type_store.open_function_context('test_default_a', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseAxpy.test_default_a.__dict__.__setitem__('stypy_localization', localization)
        BaseAxpy.test_default_a.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseAxpy.test_default_a.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseAxpy.test_default_a.__dict__.__setitem__('stypy_function_name', 'BaseAxpy.test_default_a')
        BaseAxpy.test_default_a.__dict__.__setitem__('stypy_param_names_list', [])
        BaseAxpy.test_default_a.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseAxpy.test_default_a.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseAxpy.test_default_a.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseAxpy.test_default_a.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseAxpy.test_default_a.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseAxpy.test_default_a.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseAxpy.test_default_a', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_default_a', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_default_a(...)' code ##################

        
        # Assigning a Call to a Name (line 54):
        
        # Assigning a Call to a Name (line 54):
        
        # Call to arange(...): (line 54)
        # Processing the call arguments (line 54)
        float_91381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 19), 'float')
        # Processing the call keyword arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_91382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 54)
        dtype_91383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 29), self_91382, 'dtype')
        keyword_91384 = dtype_91383
        kwargs_91385 = {'dtype': keyword_91384}
        # Getting the type of 'arange' (line 54)
        arange_91380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 54)
        arange_call_result_91386 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), arange_91380, *[float_91381], **kwargs_91385)
        
        # Assigning a type to the variable 'x' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'x', arange_call_result_91386)
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to arange(...): (line 55)
        # Processing the call arguments (line 55)
        float_91388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'float')
        # Processing the call keyword arguments (line 55)
        # Getting the type of 'x' (line 55)
        x_91389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 29), 'x', False)
        # Obtaining the member 'dtype' of a type (line 55)
        dtype_91390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 29), x_91389, 'dtype')
        keyword_91391 = dtype_91390
        kwargs_91392 = {'dtype': keyword_91391}
        # Getting the type of 'arange' (line 55)
        arange_91387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 55)
        arange_call_result_91393 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), arange_91387, *[float_91388], **kwargs_91392)
        
        # Assigning a type to the variable 'y' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'y', arange_call_result_91393)
        
        # Assigning a BinOp to a Name (line 56):
        
        # Assigning a BinOp to a Name (line 56):
        # Getting the type of 'x' (line 56)
        x_91394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'x')
        float_91395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'float')
        # Applying the binary operator '*' (line 56)
        result_mul_91396 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 17), '*', x_91394, float_91395)
        
        # Getting the type of 'y' (line 56)
        y_91397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'y')
        # Applying the binary operator '+' (line 56)
        result_add_91398 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 17), '+', result_mul_91396, y_91397)
        
        # Assigning a type to the variable 'real_y' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'real_y', result_add_91398)
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to blas_func(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'x' (line 57)
        x_91401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 27), 'x', False)
        # Getting the type of 'y' (line 57)
        y_91402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'y', False)
        # Processing the call keyword arguments (line 57)
        kwargs_91403 = {}
        # Getting the type of 'self' (line 57)
        self_91399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 57)
        blas_func_91400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), self_91399, 'blas_func')
        # Calling blas_func(args, kwargs) (line 57)
        blas_func_call_result_91404 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), blas_func_91400, *[x_91401, y_91402], **kwargs_91403)
        
        # Assigning a type to the variable 'y' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'y', blas_func_call_result_91404)
        
        # Call to assert_array_equal(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'real_y' (line 58)
        real_y_91406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'real_y', False)
        # Getting the type of 'y' (line 58)
        y_91407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 35), 'y', False)
        # Processing the call keyword arguments (line 58)
        kwargs_91408 = {}
        # Getting the type of 'assert_array_equal' (line 58)
        assert_array_equal_91405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 58)
        assert_array_equal_call_result_91409 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assert_array_equal_91405, *[real_y_91406, y_91407], **kwargs_91408)
        
        
        # ################# End of 'test_default_a(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_default_a' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_91410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91410)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_default_a'
        return stypy_return_type_91410


    @norecursion
    def test_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple'
        module_type_store = module_type_store.open_function_context('test_simple', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseAxpy.test_simple.__dict__.__setitem__('stypy_localization', localization)
        BaseAxpy.test_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseAxpy.test_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseAxpy.test_simple.__dict__.__setitem__('stypy_function_name', 'BaseAxpy.test_simple')
        BaseAxpy.test_simple.__dict__.__setitem__('stypy_param_names_list', [])
        BaseAxpy.test_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseAxpy.test_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseAxpy.test_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseAxpy.test_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseAxpy.test_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseAxpy.test_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseAxpy.test_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple(...)' code ##################

        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to arange(...): (line 61)
        # Processing the call arguments (line 61)
        float_91412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'float')
        # Processing the call keyword arguments (line 61)
        # Getting the type of 'self' (line 61)
        self_91413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 61)
        dtype_91414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 29), self_91413, 'dtype')
        keyword_91415 = dtype_91414
        kwargs_91416 = {'dtype': keyword_91415}
        # Getting the type of 'arange' (line 61)
        arange_91411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 61)
        arange_call_result_91417 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), arange_91411, *[float_91412], **kwargs_91416)
        
        # Assigning a type to the variable 'x' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'x', arange_call_result_91417)
        
        # Assigning a Call to a Name (line 62):
        
        # Assigning a Call to a Name (line 62):
        
        # Call to arange(...): (line 62)
        # Processing the call arguments (line 62)
        float_91419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'float')
        # Processing the call keyword arguments (line 62)
        # Getting the type of 'x' (line 62)
        x_91420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 29), 'x', False)
        # Obtaining the member 'dtype' of a type (line 62)
        dtype_91421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 29), x_91420, 'dtype')
        keyword_91422 = dtype_91421
        kwargs_91423 = {'dtype': keyword_91422}
        # Getting the type of 'arange' (line 62)
        arange_91418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 62)
        arange_call_result_91424 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), arange_91418, *[float_91419], **kwargs_91423)
        
        # Assigning a type to the variable 'y' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'y', arange_call_result_91424)
        
        # Assigning a BinOp to a Name (line 63):
        
        # Assigning a BinOp to a Name (line 63):
        # Getting the type of 'x' (line 63)
        x_91425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'x')
        float_91426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'float')
        # Applying the binary operator '*' (line 63)
        result_mul_91427 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 17), '*', x_91425, float_91426)
        
        # Getting the type of 'y' (line 63)
        y_91428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'y')
        # Applying the binary operator '+' (line 63)
        result_add_91429 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 17), '+', result_mul_91427, y_91428)
        
        # Assigning a type to the variable 'real_y' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'real_y', result_add_91429)
        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to blas_func(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'x' (line 64)
        x_91432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'x', False)
        # Getting the type of 'y' (line 64)
        y_91433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'y', False)
        # Processing the call keyword arguments (line 64)
        float_91434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 35), 'float')
        keyword_91435 = float_91434
        kwargs_91436 = {'a': keyword_91435}
        # Getting the type of 'self' (line 64)
        self_91430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 64)
        blas_func_91431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_91430, 'blas_func')
        # Calling blas_func(args, kwargs) (line 64)
        blas_func_call_result_91437 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), blas_func_91431, *[x_91432, y_91433], **kwargs_91436)
        
        # Assigning a type to the variable 'y' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'y', blas_func_call_result_91437)
        
        # Call to assert_array_equal(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'real_y' (line 65)
        real_y_91439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'real_y', False)
        # Getting the type of 'y' (line 65)
        y_91440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 35), 'y', False)
        # Processing the call keyword arguments (line 65)
        kwargs_91441 = {}
        # Getting the type of 'assert_array_equal' (line 65)
        assert_array_equal_91438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 65)
        assert_array_equal_call_result_91442 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), assert_array_equal_91438, *[real_y_91439, y_91440], **kwargs_91441)
        
        
        # ################# End of 'test_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_91443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91443)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple'
        return stypy_return_type_91443


    @norecursion
    def test_x_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_stride'
        module_type_store = module_type_store.open_function_context('test_x_stride', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseAxpy.test_x_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseAxpy.test_x_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseAxpy.test_x_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseAxpy.test_x_stride.__dict__.__setitem__('stypy_function_name', 'BaseAxpy.test_x_stride')
        BaseAxpy.test_x_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseAxpy.test_x_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseAxpy.test_x_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseAxpy.test_x_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseAxpy.test_x_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseAxpy.test_x_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseAxpy.test_x_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseAxpy.test_x_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_stride(...)' code ##################

        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to arange(...): (line 68)
        # Processing the call arguments (line 68)
        float_91445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 19), 'float')
        # Processing the call keyword arguments (line 68)
        # Getting the type of 'self' (line 68)
        self_91446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 68)
        dtype_91447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 29), self_91446, 'dtype')
        keyword_91448 = dtype_91447
        kwargs_91449 = {'dtype': keyword_91448}
        # Getting the type of 'arange' (line 68)
        arange_91444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 68)
        arange_call_result_91450 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), arange_91444, *[float_91445], **kwargs_91449)
        
        # Assigning a type to the variable 'x' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'x', arange_call_result_91450)
        
        # Assigning a Call to a Name (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Call to zeros(...): (line 69)
        # Processing the call arguments (line 69)
        int_91452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 18), 'int')
        # Getting the type of 'x' (line 69)
        x_91453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 69)
        dtype_91454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 21), x_91453, 'dtype')
        # Processing the call keyword arguments (line 69)
        kwargs_91455 = {}
        # Getting the type of 'zeros' (line 69)
        zeros_91451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 69)
        zeros_call_result_91456 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), zeros_91451, *[int_91452, dtype_91454], **kwargs_91455)
        
        # Assigning a type to the variable 'y' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'y', zeros_call_result_91456)
        
        # Assigning a Call to a Name (line 70):
        
        # Assigning a Call to a Name (line 70):
        
        # Call to arange(...): (line 70)
        # Processing the call arguments (line 70)
        float_91458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 19), 'float')
        # Processing the call keyword arguments (line 70)
        # Getting the type of 'x' (line 70)
        x_91459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 29), 'x', False)
        # Obtaining the member 'dtype' of a type (line 70)
        dtype_91460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 29), x_91459, 'dtype')
        keyword_91461 = dtype_91460
        kwargs_91462 = {'dtype': keyword_91461}
        # Getting the type of 'arange' (line 70)
        arange_91457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 70)
        arange_call_result_91463 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), arange_91457, *[float_91458], **kwargs_91462)
        
        # Assigning a type to the variable 'y' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'y', arange_call_result_91463)
        
        # Assigning a BinOp to a Name (line 71):
        
        # Assigning a BinOp to a Name (line 71):
        
        # Obtaining the type of the subscript
        int_91464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'int')
        slice_91465 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 71, 17), None, None, int_91464)
        # Getting the type of 'x' (line 71)
        x_91466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'x')
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___91467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 17), x_91466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_91468 = invoke(stypy.reporting.localization.Localization(__file__, 71, 17), getitem___91467, slice_91465)
        
        float_91469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 24), 'float')
        # Applying the binary operator '*' (line 71)
        result_mul_91470 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 17), '*', subscript_call_result_91468, float_91469)
        
        # Getting the type of 'y' (line 71)
        y_91471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'y')
        # Applying the binary operator '+' (line 71)
        result_add_91472 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 17), '+', result_mul_91470, y_91471)
        
        # Assigning a type to the variable 'real_y' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'real_y', result_add_91472)
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to blas_func(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'x' (line 72)
        x_91475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'x', False)
        # Getting the type of 'y' (line 72)
        y_91476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'y', False)
        # Processing the call keyword arguments (line 72)
        float_91477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 35), 'float')
        keyword_91478 = float_91477
        int_91479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 41), 'int')
        keyword_91480 = int_91479
        int_91481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 49), 'int')
        keyword_91482 = int_91481
        kwargs_91483 = {'a': keyword_91478, 'incx': keyword_91482, 'n': keyword_91480}
        # Getting the type of 'self' (line 72)
        self_91473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 72)
        blas_func_91474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), self_91473, 'blas_func')
        # Calling blas_func(args, kwargs) (line 72)
        blas_func_call_result_91484 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), blas_func_91474, *[x_91475, y_91476], **kwargs_91483)
        
        # Assigning a type to the variable 'y' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'y', blas_func_call_result_91484)
        
        # Call to assert_array_equal(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'real_y' (line 73)
        real_y_91486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 27), 'real_y', False)
        # Getting the type of 'y' (line 73)
        y_91487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 35), 'y', False)
        # Processing the call keyword arguments (line 73)
        kwargs_91488 = {}
        # Getting the type of 'assert_array_equal' (line 73)
        assert_array_equal_91485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 73)
        assert_array_equal_call_result_91489 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), assert_array_equal_91485, *[real_y_91486, y_91487], **kwargs_91488)
        
        
        # ################# End of 'test_x_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_91490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91490)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_stride'
        return stypy_return_type_91490


    @norecursion
    def test_y_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_y_stride'
        module_type_store = module_type_store.open_function_context('test_y_stride', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseAxpy.test_y_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseAxpy.test_y_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseAxpy.test_y_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseAxpy.test_y_stride.__dict__.__setitem__('stypy_function_name', 'BaseAxpy.test_y_stride')
        BaseAxpy.test_y_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseAxpy.test_y_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseAxpy.test_y_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseAxpy.test_y_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseAxpy.test_y_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseAxpy.test_y_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseAxpy.test_y_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseAxpy.test_y_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_y_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_y_stride(...)' code ##################

        
        # Assigning a Call to a Name (line 76):
        
        # Assigning a Call to a Name (line 76):
        
        # Call to arange(...): (line 76)
        # Processing the call arguments (line 76)
        float_91492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 19), 'float')
        # Processing the call keyword arguments (line 76)
        # Getting the type of 'self' (line 76)
        self_91493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 76)
        dtype_91494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 29), self_91493, 'dtype')
        keyword_91495 = dtype_91494
        kwargs_91496 = {'dtype': keyword_91495}
        # Getting the type of 'arange' (line 76)
        arange_91491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 76)
        arange_call_result_91497 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), arange_91491, *[float_91492], **kwargs_91496)
        
        # Assigning a type to the variable 'x' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'x', arange_call_result_91497)
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to zeros(...): (line 77)
        # Processing the call arguments (line 77)
        int_91499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'int')
        # Getting the type of 'x' (line 77)
        x_91500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 77)
        dtype_91501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 21), x_91500, 'dtype')
        # Processing the call keyword arguments (line 77)
        kwargs_91502 = {}
        # Getting the type of 'zeros' (line 77)
        zeros_91498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 77)
        zeros_call_result_91503 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), zeros_91498, *[int_91499, dtype_91501], **kwargs_91502)
        
        # Assigning a type to the variable 'y' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'y', zeros_call_result_91503)
        
        # Assigning a BinOp to a Name (line 78):
        
        # Assigning a BinOp to a Name (line 78):
        # Getting the type of 'x' (line 78)
        x_91504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'x')
        float_91505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 19), 'float')
        # Applying the binary operator '*' (line 78)
        result_mul_91506 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 17), '*', x_91504, float_91505)
        
        
        # Obtaining the type of the subscript
        int_91507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'int')
        slice_91508 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 78, 22), None, None, int_91507)
        # Getting the type of 'y' (line 78)
        y_91509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'y')
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___91510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 22), y_91509, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_91511 = invoke(stypy.reporting.localization.Localization(__file__, 78, 22), getitem___91510, slice_91508)
        
        # Applying the binary operator '+' (line 78)
        result_add_91512 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 17), '+', result_mul_91506, subscript_call_result_91511)
        
        # Assigning a type to the variable 'real_y' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'real_y', result_add_91512)
        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to blas_func(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'x' (line 79)
        x_91515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 27), 'x', False)
        # Getting the type of 'y' (line 79)
        y_91516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'y', False)
        # Processing the call keyword arguments (line 79)
        float_91517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 35), 'float')
        keyword_91518 = float_91517
        int_91519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 41), 'int')
        keyword_91520 = int_91519
        int_91521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 49), 'int')
        keyword_91522 = int_91521
        kwargs_91523 = {'a': keyword_91518, 'incy': keyword_91522, 'n': keyword_91520}
        # Getting the type of 'self' (line 79)
        self_91513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 79)
        blas_func_91514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), self_91513, 'blas_func')
        # Calling blas_func(args, kwargs) (line 79)
        blas_func_call_result_91524 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), blas_func_91514, *[x_91515, y_91516], **kwargs_91523)
        
        # Assigning a type to the variable 'y' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'y', blas_func_call_result_91524)
        
        # Call to assert_array_equal(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'real_y' (line 80)
        real_y_91526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'real_y', False)
        
        # Obtaining the type of the subscript
        int_91527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 39), 'int')
        slice_91528 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 80, 35), None, None, int_91527)
        # Getting the type of 'y' (line 80)
        y_91529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___91530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 35), y_91529, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_91531 = invoke(stypy.reporting.localization.Localization(__file__, 80, 35), getitem___91530, slice_91528)
        
        # Processing the call keyword arguments (line 80)
        kwargs_91532 = {}
        # Getting the type of 'assert_array_equal' (line 80)
        assert_array_equal_91525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 80)
        assert_array_equal_call_result_91533 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assert_array_equal_91525, *[real_y_91526, subscript_call_result_91531], **kwargs_91532)
        
        
        # ################# End of 'test_y_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_y_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_91534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91534)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_y_stride'
        return stypy_return_type_91534


    @norecursion
    def test_x_and_y_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_and_y_stride'
        module_type_store = module_type_store.open_function_context('test_x_and_y_stride', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseAxpy.test_x_and_y_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseAxpy.test_x_and_y_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseAxpy.test_x_and_y_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseAxpy.test_x_and_y_stride.__dict__.__setitem__('stypy_function_name', 'BaseAxpy.test_x_and_y_stride')
        BaseAxpy.test_x_and_y_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseAxpy.test_x_and_y_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseAxpy.test_x_and_y_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseAxpy.test_x_and_y_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseAxpy.test_x_and_y_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseAxpy.test_x_and_y_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseAxpy.test_x_and_y_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseAxpy.test_x_and_y_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_and_y_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_and_y_stride(...)' code ##################

        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to arange(...): (line 83)
        # Processing the call arguments (line 83)
        float_91536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 19), 'float')
        # Processing the call keyword arguments (line 83)
        # Getting the type of 'self' (line 83)
        self_91537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 83)
        dtype_91538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 30), self_91537, 'dtype')
        keyword_91539 = dtype_91538
        kwargs_91540 = {'dtype': keyword_91539}
        # Getting the type of 'arange' (line 83)
        arange_91535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 83)
        arange_call_result_91541 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), arange_91535, *[float_91536], **kwargs_91540)
        
        # Assigning a type to the variable 'x' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'x', arange_call_result_91541)
        
        # Assigning a Call to a Name (line 84):
        
        # Assigning a Call to a Name (line 84):
        
        # Call to zeros(...): (line 84)
        # Processing the call arguments (line 84)
        int_91543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 18), 'int')
        # Getting the type of 'x' (line 84)
        x_91544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 84)
        dtype_91545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), x_91544, 'dtype')
        # Processing the call keyword arguments (line 84)
        kwargs_91546 = {}
        # Getting the type of 'zeros' (line 84)
        zeros_91542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 84)
        zeros_call_result_91547 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), zeros_91542, *[int_91543, dtype_91545], **kwargs_91546)
        
        # Assigning a type to the variable 'y' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'y', zeros_call_result_91547)
        
        # Assigning a BinOp to a Name (line 85):
        
        # Assigning a BinOp to a Name (line 85):
        
        # Obtaining the type of the subscript
        int_91548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'int')
        slice_91549 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 85, 17), None, None, int_91548)
        # Getting the type of 'x' (line 85)
        x_91550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'x')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___91551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 17), x_91550, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_91552 = invoke(stypy.reporting.localization.Localization(__file__, 85, 17), getitem___91551, slice_91549)
        
        float_91553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 24), 'float')
        # Applying the binary operator '*' (line 85)
        result_mul_91554 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 17), '*', subscript_call_result_91552, float_91553)
        
        
        # Obtaining the type of the subscript
        int_91555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 31), 'int')
        slice_91556 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 85, 27), None, None, int_91555)
        # Getting the type of 'y' (line 85)
        y_91557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'y')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___91558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 27), y_91557, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_91559 = invoke(stypy.reporting.localization.Localization(__file__, 85, 27), getitem___91558, slice_91556)
        
        # Applying the binary operator '+' (line 85)
        result_add_91560 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 17), '+', result_mul_91554, subscript_call_result_91559)
        
        # Assigning a type to the variable 'real_y' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'real_y', result_add_91560)
        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to blas_func(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'x' (line 86)
        x_91563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 27), 'x', False)
        # Getting the type of 'y' (line 86)
        y_91564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'y', False)
        # Processing the call keyword arguments (line 86)
        float_91565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 35), 'float')
        keyword_91566 = float_91565
        int_91567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 41), 'int')
        keyword_91568 = int_91567
        int_91569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 49), 'int')
        keyword_91570 = int_91569
        int_91571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 57), 'int')
        keyword_91572 = int_91571
        kwargs_91573 = {'a': keyword_91566, 'incx': keyword_91570, 'incy': keyword_91572, 'n': keyword_91568}
        # Getting the type of 'self' (line 86)
        self_91561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 86)
        blas_func_91562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), self_91561, 'blas_func')
        # Calling blas_func(args, kwargs) (line 86)
        blas_func_call_result_91574 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), blas_func_91562, *[x_91563, y_91564], **kwargs_91573)
        
        # Assigning a type to the variable 'y' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'y', blas_func_call_result_91574)
        
        # Call to assert_array_equal(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'real_y' (line 87)
        real_y_91576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 27), 'real_y', False)
        
        # Obtaining the type of the subscript
        int_91577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 39), 'int')
        slice_91578 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 87, 35), None, None, int_91577)
        # Getting the type of 'y' (line 87)
        y_91579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 35), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___91580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 35), y_91579, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_91581 = invoke(stypy.reporting.localization.Localization(__file__, 87, 35), getitem___91580, slice_91578)
        
        # Processing the call keyword arguments (line 87)
        kwargs_91582 = {}
        # Getting the type of 'assert_array_equal' (line 87)
        assert_array_equal_91575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 87)
        assert_array_equal_call_result_91583 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), assert_array_equal_91575, *[real_y_91576, subscript_call_result_91581], **kwargs_91582)
        
        
        # ################# End of 'test_x_and_y_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_and_y_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_91584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_and_y_stride'
        return stypy_return_type_91584


    @norecursion
    def test_x_bad_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_bad_size'
        module_type_store = module_type_store.open_function_context('test_x_bad_size', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseAxpy.test_x_bad_size.__dict__.__setitem__('stypy_localization', localization)
        BaseAxpy.test_x_bad_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseAxpy.test_x_bad_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseAxpy.test_x_bad_size.__dict__.__setitem__('stypy_function_name', 'BaseAxpy.test_x_bad_size')
        BaseAxpy.test_x_bad_size.__dict__.__setitem__('stypy_param_names_list', [])
        BaseAxpy.test_x_bad_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseAxpy.test_x_bad_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseAxpy.test_x_bad_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseAxpy.test_x_bad_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseAxpy.test_x_bad_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseAxpy.test_x_bad_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseAxpy.test_x_bad_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_bad_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_bad_size(...)' code ##################

        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to arange(...): (line 90)
        # Processing the call arguments (line 90)
        float_91586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 19), 'float')
        # Processing the call keyword arguments (line 90)
        # Getting the type of 'self' (line 90)
        self_91587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 90)
        dtype_91588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 30), self_91587, 'dtype')
        keyword_91589 = dtype_91588
        kwargs_91590 = {'dtype': keyword_91589}
        # Getting the type of 'arange' (line 90)
        arange_91585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 90)
        arange_call_result_91591 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), arange_91585, *[float_91586], **kwargs_91590)
        
        # Assigning a type to the variable 'x' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'x', arange_call_result_91591)
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to zeros(...): (line 91)
        # Processing the call arguments (line 91)
        int_91593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 18), 'int')
        # Getting the type of 'x' (line 91)
        x_91594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 91)
        dtype_91595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 21), x_91594, 'dtype')
        # Processing the call keyword arguments (line 91)
        kwargs_91596 = {}
        # Getting the type of 'zeros' (line 91)
        zeros_91592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 91)
        zeros_call_result_91597 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), zeros_91592, *[int_91593, dtype_91595], **kwargs_91596)
        
        # Assigning a type to the variable 'y' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'y', zeros_call_result_91597)
        
        
        # SSA begins for try-except statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to blas_func(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'x' (line 93)
        x_91600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'x', False)
        # Getting the type of 'y' (line 93)
        y_91601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 30), 'y', False)
        # Processing the call keyword arguments (line 93)
        int_91602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 35), 'int')
        keyword_91603 = int_91602
        int_91604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 43), 'int')
        keyword_91605 = int_91604
        kwargs_91606 = {'incx': keyword_91605, 'n': keyword_91603}
        # Getting the type of 'self' (line 93)
        self_91598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 93)
        blas_func_91599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), self_91598, 'blas_func')
        # Calling blas_func(args, kwargs) (line 93)
        blas_func_call_result_91607 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), blas_func_91599, *[x_91600, y_91601], **kwargs_91606)
        
        # SSA branch for the except part of a try statement (line 92)
        # SSA branch for the except '<any exception>' branch of a try statement (line 92)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 97)
        # Processing the call arguments (line 97)
        int_91609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'int')
        # Processing the call keyword arguments (line 97)
        kwargs_91610 = {}
        # Getting the type of 'assert_' (line 97)
        assert__91608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 97)
        assert__call_result_91611 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assert__91608, *[int_91609], **kwargs_91610)
        
        
        # ################# End of 'test_x_bad_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_bad_size' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_91612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91612)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_bad_size'
        return stypy_return_type_91612


    @norecursion
    def test_y_bad_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_y_bad_size'
        module_type_store = module_type_store.open_function_context('test_y_bad_size', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseAxpy.test_y_bad_size.__dict__.__setitem__('stypy_localization', localization)
        BaseAxpy.test_y_bad_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseAxpy.test_y_bad_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseAxpy.test_y_bad_size.__dict__.__setitem__('stypy_function_name', 'BaseAxpy.test_y_bad_size')
        BaseAxpy.test_y_bad_size.__dict__.__setitem__('stypy_param_names_list', [])
        BaseAxpy.test_y_bad_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseAxpy.test_y_bad_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseAxpy.test_y_bad_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseAxpy.test_y_bad_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseAxpy.test_y_bad_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseAxpy.test_y_bad_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseAxpy.test_y_bad_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_y_bad_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_y_bad_size(...)' code ##################

        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to arange(...): (line 100)
        # Processing the call arguments (line 100)
        float_91614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'float')
        # Processing the call keyword arguments (line 100)
        # Getting the type of 'self' (line 100)
        self_91615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 100)
        dtype_91616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 30), self_91615, 'dtype')
        keyword_91617 = dtype_91616
        kwargs_91618 = {'dtype': keyword_91617}
        # Getting the type of 'arange' (line 100)
        arange_91613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 100)
        arange_call_result_91619 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), arange_91613, *[float_91614], **kwargs_91618)
        
        # Assigning a type to the variable 'x' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'x', arange_call_result_91619)
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to zeros(...): (line 101)
        # Processing the call arguments (line 101)
        int_91621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 18), 'int')
        # Getting the type of 'x' (line 101)
        x_91622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 101)
        dtype_91623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 21), x_91622, 'dtype')
        # Processing the call keyword arguments (line 101)
        kwargs_91624 = {}
        # Getting the type of 'zeros' (line 101)
        zeros_91620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 101)
        zeros_call_result_91625 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), zeros_91620, *[int_91621, dtype_91623], **kwargs_91624)
        
        # Assigning a type to the variable 'y' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'y', zeros_call_result_91625)
        
        
        # SSA begins for try-except statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to blas_func(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'x' (line 103)
        x_91628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'x', False)
        # Getting the type of 'y' (line 103)
        y_91629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'y', False)
        # Processing the call keyword arguments (line 103)
        int_91630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 35), 'int')
        keyword_91631 = int_91630
        int_91632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 43), 'int')
        keyword_91633 = int_91632
        kwargs_91634 = {'incy': keyword_91633, 'n': keyword_91631}
        # Getting the type of 'self' (line 103)
        self_91626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 103)
        blas_func_91627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_91626, 'blas_func')
        # Calling blas_func(args, kwargs) (line 103)
        blas_func_call_result_91635 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), blas_func_91627, *[x_91628, y_91629], **kwargs_91634)
        
        # SSA branch for the except part of a try statement (line 102)
        # SSA branch for the except '<any exception>' branch of a try statement (line 102)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 102)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 107)
        # Processing the call arguments (line 107)
        int_91637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 16), 'int')
        # Processing the call keyword arguments (line 107)
        kwargs_91638 = {}
        # Getting the type of 'assert_' (line 107)
        assert__91636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 107)
        assert__call_result_91639 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), assert__91636, *[int_91637], **kwargs_91638)
        
        
        # ################# End of 'test_y_bad_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_y_bad_size' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_91640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91640)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_y_bad_size'
        return stypy_return_type_91640


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 50, 0, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseAxpy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BaseAxpy' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'BaseAxpy', BaseAxpy)


# SSA begins for try-except statement (line 110)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Declaration of the 'TestSaxpy' class
# Getting the type of 'BaseAxpy' (line 111)
BaseAxpy_91641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'BaseAxpy')

class TestSaxpy(BaseAxpy_91641, ):
    
    # Assigning a Attribute to a Name (line 112):
    
    # Assigning a Name to a Name (line 113):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSaxpy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSaxpy' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'TestSaxpy', TestSaxpy)

# Assigning a Attribute to a Name (line 112):
# Getting the type of 'fblas' (line 112)
fblas_91642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'fblas')
# Obtaining the member 'saxpy' of a type (line 112)
saxpy_91643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), fblas_91642, 'saxpy')
# Getting the type of 'TestSaxpy'
TestSaxpy_91644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestSaxpy')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestSaxpy_91644, 'blas_func', saxpy_91643)

# Assigning a Name to a Name (line 113):
# Getting the type of 'float32' (line 113)
float32_91645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'float32')
# Getting the type of 'TestSaxpy'
TestSaxpy_91646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestSaxpy')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestSaxpy_91646, 'dtype', float32_91645)
# SSA branch for the except part of a try statement (line 110)
# SSA branch for the except 'AttributeError' branch of a try statement (line 110)
module_type_store.open_ssa_branch('except')
# Declaration of the 'TestSaxpy' class

class TestSaxpy:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSaxpy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSaxpy' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'TestSaxpy', TestSaxpy)
# SSA join for try-except statement (line 110)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TestDaxpy' class
# Getting the type of 'BaseAxpy' (line 119)
BaseAxpy_91647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'BaseAxpy')

class TestDaxpy(BaseAxpy_91647, ):
    
    # Assigning a Attribute to a Name (line 120):
    
    # Assigning a Name to a Name (line 121):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 119, 0, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDaxpy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDaxpy' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'TestDaxpy', TestDaxpy)

# Assigning a Attribute to a Name (line 120):
# Getting the type of 'fblas' (line 120)
fblas_91648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'fblas')
# Obtaining the member 'daxpy' of a type (line 120)
daxpy_91649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), fblas_91648, 'daxpy')
# Getting the type of 'TestDaxpy'
TestDaxpy_91650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDaxpy')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDaxpy_91650, 'blas_func', daxpy_91649)

# Assigning a Name to a Name (line 121):
# Getting the type of 'float64' (line 121)
float64_91651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'float64')
# Getting the type of 'TestDaxpy'
TestDaxpy_91652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDaxpy')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDaxpy_91652, 'dtype', float64_91651)


# SSA begins for try-except statement (line 124)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Declaration of the 'TestCaxpy' class
# Getting the type of 'BaseAxpy' (line 125)
BaseAxpy_91653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'BaseAxpy')

class TestCaxpy(BaseAxpy_91653, ):
    
    # Assigning a Attribute to a Name (line 126):
    
    # Assigning a Name to a Name (line 127):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCaxpy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCaxpy' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'TestCaxpy', TestCaxpy)

# Assigning a Attribute to a Name (line 126):
# Getting the type of 'fblas' (line 126)
fblas_91654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'fblas')
# Obtaining the member 'caxpy' of a type (line 126)
caxpy_91655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 20), fblas_91654, 'caxpy')
# Getting the type of 'TestCaxpy'
TestCaxpy_91656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCaxpy')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCaxpy_91656, 'blas_func', caxpy_91655)

# Assigning a Name to a Name (line 127):
# Getting the type of 'complex64' (line 127)
complex64_91657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'complex64')
# Getting the type of 'TestCaxpy'
TestCaxpy_91658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCaxpy')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCaxpy_91658, 'dtype', complex64_91657)
# SSA branch for the except part of a try statement (line 124)
# SSA branch for the except 'AttributeError' branch of a try statement (line 124)
module_type_store.open_ssa_branch('except')
# Declaration of the 'TestCaxpy' class

class TestCaxpy:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCaxpy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCaxpy' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'TestCaxpy', TestCaxpy)
# SSA join for try-except statement (line 124)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TestZaxpy' class
# Getting the type of 'BaseAxpy' (line 133)
BaseAxpy_91659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'BaseAxpy')

class TestZaxpy(BaseAxpy_91659, ):
    
    # Assigning a Attribute to a Name (line 134):
    
    # Assigning a Name to a Name (line 135):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 133, 0, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZaxpy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestZaxpy' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'TestZaxpy', TestZaxpy)

# Assigning a Attribute to a Name (line 134):
# Getting the type of 'fblas' (line 134)
fblas_91660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'fblas')
# Obtaining the member 'zaxpy' of a type (line 134)
zaxpy_91661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), fblas_91660, 'zaxpy')
# Getting the type of 'TestZaxpy'
TestZaxpy_91662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZaxpy')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZaxpy_91662, 'blas_func', zaxpy_91661)

# Assigning a Name to a Name (line 135):
# Getting the type of 'complex128' (line 135)
complex128_91663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'complex128')
# Getting the type of 'TestZaxpy'
TestZaxpy_91664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZaxpy')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZaxpy_91664, 'dtype', complex128_91663)
# Declaration of the 'BaseScal' class

class BaseScal(object, ):
    str_91665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'str', ' Mixin class for scal testing ')

    @norecursion
    def test_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple'
        module_type_store = module_type_store.open_function_context('test_simple', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseScal.test_simple.__dict__.__setitem__('stypy_localization', localization)
        BaseScal.test_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseScal.test_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseScal.test_simple.__dict__.__setitem__('stypy_function_name', 'BaseScal.test_simple')
        BaseScal.test_simple.__dict__.__setitem__('stypy_param_names_list', [])
        BaseScal.test_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseScal.test_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseScal.test_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseScal.test_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseScal.test_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseScal.test_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseScal.test_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple(...)' code ##################

        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to arange(...): (line 145)
        # Processing the call arguments (line 145)
        float_91667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 19), 'float')
        # Processing the call keyword arguments (line 145)
        # Getting the type of 'self' (line 145)
        self_91668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 145)
        dtype_91669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 29), self_91668, 'dtype')
        keyword_91670 = dtype_91669
        kwargs_91671 = {'dtype': keyword_91670}
        # Getting the type of 'arange' (line 145)
        arange_91666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 145)
        arange_call_result_91672 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), arange_91666, *[float_91667], **kwargs_91671)
        
        # Assigning a type to the variable 'x' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'x', arange_call_result_91672)
        
        # Assigning a BinOp to a Name (line 146):
        
        # Assigning a BinOp to a Name (line 146):
        # Getting the type of 'x' (line 146)
        x_91673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'x')
        float_91674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'float')
        # Applying the binary operator '*' (line 146)
        result_mul_91675 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 17), '*', x_91673, float_91674)
        
        # Assigning a type to the variable 'real_x' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'real_x', result_mul_91675)
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to blas_func(...): (line 147)
        # Processing the call arguments (line 147)
        float_91678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 27), 'float')
        # Getting the type of 'x' (line 147)
        x_91679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 31), 'x', False)
        # Processing the call keyword arguments (line 147)
        kwargs_91680 = {}
        # Getting the type of 'self' (line 147)
        self_91676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 147)
        blas_func_91677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), self_91676, 'blas_func')
        # Calling blas_func(args, kwargs) (line 147)
        blas_func_call_result_91681 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), blas_func_91677, *[float_91678, x_91679], **kwargs_91680)
        
        # Assigning a type to the variable 'x' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'x', blas_func_call_result_91681)
        
        # Call to assert_array_equal(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'real_x' (line 148)
        real_x_91683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'real_x', False)
        # Getting the type of 'x' (line 148)
        x_91684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'x', False)
        # Processing the call keyword arguments (line 148)
        kwargs_91685 = {}
        # Getting the type of 'assert_array_equal' (line 148)
        assert_array_equal_91682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 148)
        assert_array_equal_call_result_91686 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), assert_array_equal_91682, *[real_x_91683, x_91684], **kwargs_91685)
        
        
        # ################# End of 'test_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_91687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91687)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple'
        return stypy_return_type_91687


    @norecursion
    def test_x_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_stride'
        module_type_store = module_type_store.open_function_context('test_x_stride', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseScal.test_x_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseScal.test_x_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseScal.test_x_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseScal.test_x_stride.__dict__.__setitem__('stypy_function_name', 'BaseScal.test_x_stride')
        BaseScal.test_x_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseScal.test_x_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseScal.test_x_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseScal.test_x_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseScal.test_x_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseScal.test_x_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseScal.test_x_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseScal.test_x_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_stride(...)' code ##################

        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to arange(...): (line 151)
        # Processing the call arguments (line 151)
        float_91689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 19), 'float')
        # Processing the call keyword arguments (line 151)
        # Getting the type of 'self' (line 151)
        self_91690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 151)
        dtype_91691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 29), self_91690, 'dtype')
        keyword_91692 = dtype_91691
        kwargs_91693 = {'dtype': keyword_91692}
        # Getting the type of 'arange' (line 151)
        arange_91688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 151)
        arange_call_result_91694 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), arange_91688, *[float_91689], **kwargs_91693)
        
        # Assigning a type to the variable 'x' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'x', arange_call_result_91694)
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to copy(...): (line 152)
        # Processing the call keyword arguments (line 152)
        kwargs_91697 = {}
        # Getting the type of 'x' (line 152)
        x_91695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'x', False)
        # Obtaining the member 'copy' of a type (line 152)
        copy_91696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 17), x_91695, 'copy')
        # Calling copy(args, kwargs) (line 152)
        copy_call_result_91698 = invoke(stypy.reporting.localization.Localization(__file__, 152, 17), copy_91696, *[], **kwargs_91697)
        
        # Assigning a type to the variable 'real_x' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'real_x', copy_call_result_91698)
        
        # Assigning a BinOp to a Subscript (line 153):
        
        # Assigning a BinOp to a Subscript (line 153):
        
        # Obtaining the type of the subscript
        int_91699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'int')
        slice_91700 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 153, 22), None, None, int_91699)
        # Getting the type of 'x' (line 153)
        x_91701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 22), 'x')
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___91702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 22), x_91701, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_91703 = invoke(stypy.reporting.localization.Localization(__file__, 153, 22), getitem___91702, slice_91700)
        
        
        # Call to array(...): (line 153)
        # Processing the call arguments (line 153)
        float_91705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 35), 'float')
        # Getting the type of 'self' (line 153)
        self_91706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 39), 'self', False)
        # Obtaining the member 'dtype' of a type (line 153)
        dtype_91707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 39), self_91706, 'dtype')
        # Processing the call keyword arguments (line 153)
        kwargs_91708 = {}
        # Getting the type of 'array' (line 153)
        array_91704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 29), 'array', False)
        # Calling array(args, kwargs) (line 153)
        array_call_result_91709 = invoke(stypy.reporting.localization.Localization(__file__, 153, 29), array_91704, *[float_91705, dtype_91707], **kwargs_91708)
        
        # Applying the binary operator '*' (line 153)
        result_mul_91710 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 22), '*', subscript_call_result_91703, array_call_result_91709)
        
        # Getting the type of 'real_x' (line 153)
        real_x_91711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'real_x')
        int_91712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 17), 'int')
        slice_91713 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 153, 8), None, None, int_91712)
        # Storing an element on a container (line 153)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 8), real_x_91711, (slice_91713, result_mul_91710))
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Call to blas_func(...): (line 154)
        # Processing the call arguments (line 154)
        float_91716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 27), 'float')
        # Getting the type of 'x' (line 154)
        x_91717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 31), 'x', False)
        # Processing the call keyword arguments (line 154)
        int_91718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 36), 'int')
        keyword_91719 = int_91718
        int_91720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 44), 'int')
        keyword_91721 = int_91720
        kwargs_91722 = {'incx': keyword_91721, 'n': keyword_91719}
        # Getting the type of 'self' (line 154)
        self_91714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 154)
        blas_func_91715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), self_91714, 'blas_func')
        # Calling blas_func(args, kwargs) (line 154)
        blas_func_call_result_91723 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), blas_func_91715, *[float_91716, x_91717], **kwargs_91722)
        
        # Assigning a type to the variable 'x' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'x', blas_func_call_result_91723)
        
        # Call to assert_array_equal(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'real_x' (line 155)
        real_x_91725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'real_x', False)
        # Getting the type of 'x' (line 155)
        x_91726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 35), 'x', False)
        # Processing the call keyword arguments (line 155)
        kwargs_91727 = {}
        # Getting the type of 'assert_array_equal' (line 155)
        assert_array_equal_91724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 155)
        assert_array_equal_call_result_91728 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), assert_array_equal_91724, *[real_x_91725, x_91726], **kwargs_91727)
        
        
        # ################# End of 'test_x_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_91729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91729)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_stride'
        return stypy_return_type_91729


    @norecursion
    def test_x_bad_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_bad_size'
        module_type_store = module_type_store.open_function_context('test_x_bad_size', 157, 4, False)
        # Assigning a type to the variable 'self' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseScal.test_x_bad_size.__dict__.__setitem__('stypy_localization', localization)
        BaseScal.test_x_bad_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseScal.test_x_bad_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseScal.test_x_bad_size.__dict__.__setitem__('stypy_function_name', 'BaseScal.test_x_bad_size')
        BaseScal.test_x_bad_size.__dict__.__setitem__('stypy_param_names_list', [])
        BaseScal.test_x_bad_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseScal.test_x_bad_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseScal.test_x_bad_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseScal.test_x_bad_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseScal.test_x_bad_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseScal.test_x_bad_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseScal.test_x_bad_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_bad_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_bad_size(...)' code ##################

        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to arange(...): (line 158)
        # Processing the call arguments (line 158)
        float_91731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 19), 'float')
        # Processing the call keyword arguments (line 158)
        # Getting the type of 'self' (line 158)
        self_91732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 158)
        dtype_91733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 30), self_91732, 'dtype')
        keyword_91734 = dtype_91733
        kwargs_91735 = {'dtype': keyword_91734}
        # Getting the type of 'arange' (line 158)
        arange_91730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 158)
        arange_call_result_91736 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), arange_91730, *[float_91731], **kwargs_91735)
        
        # Assigning a type to the variable 'x' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'x', arange_call_result_91736)
        
        
        # SSA begins for try-except statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to blas_func(...): (line 160)
        # Processing the call arguments (line 160)
        float_91739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 27), 'float')
        # Getting the type of 'x' (line 160)
        x_91740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 31), 'x', False)
        # Processing the call keyword arguments (line 160)
        int_91741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'int')
        keyword_91742 = int_91741
        int_91743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 44), 'int')
        keyword_91744 = int_91743
        kwargs_91745 = {'incx': keyword_91744, 'n': keyword_91742}
        # Getting the type of 'self' (line 160)
        self_91737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 160)
        blas_func_91738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), self_91737, 'blas_func')
        # Calling blas_func(args, kwargs) (line 160)
        blas_func_call_result_91746 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), blas_func_91738, *[float_91739, x_91740], **kwargs_91745)
        
        # SSA branch for the except part of a try statement (line 159)
        # SSA branch for the except '<any exception>' branch of a try statement (line 159)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 164)
        # Processing the call arguments (line 164)
        int_91748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 16), 'int')
        # Processing the call keyword arguments (line 164)
        kwargs_91749 = {}
        # Getting the type of 'assert_' (line 164)
        assert__91747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 164)
        assert__call_result_91750 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), assert__91747, *[int_91748], **kwargs_91749)
        
        
        # ################# End of 'test_x_bad_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_bad_size' in the type store
        # Getting the type of 'stypy_return_type' (line 157)
        stypy_return_type_91751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91751)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_bad_size'
        return stypy_return_type_91751


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 141, 0, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseScal.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BaseScal' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'BaseScal', BaseScal)


# SSA begins for try-except statement (line 167)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Declaration of the 'TestSscal' class
# Getting the type of 'BaseScal' (line 168)
BaseScal_91752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'BaseScal')

class TestSscal(BaseScal_91752, ):
    
    # Assigning a Attribute to a Name (line 169):
    
    # Assigning a Name to a Name (line 170):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSscal.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSscal' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'TestSscal', TestSscal)

# Assigning a Attribute to a Name (line 169):
# Getting the type of 'fblas' (line 169)
fblas_91753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'fblas')
# Obtaining the member 'sscal' of a type (line 169)
sscal_91754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 20), fblas_91753, 'sscal')
# Getting the type of 'TestSscal'
TestSscal_91755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestSscal')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestSscal_91755, 'blas_func', sscal_91754)

# Assigning a Name to a Name (line 170):
# Getting the type of 'float32' (line 170)
float32_91756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'float32')
# Getting the type of 'TestSscal'
TestSscal_91757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestSscal')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestSscal_91757, 'dtype', float32_91756)
# SSA branch for the except part of a try statement (line 167)
# SSA branch for the except 'AttributeError' branch of a try statement (line 167)
module_type_store.open_ssa_branch('except')
# Declaration of the 'TestSscal' class

class TestSscal:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSscal.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSscal' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'TestSscal', TestSscal)
# SSA join for try-except statement (line 167)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TestDscal' class
# Getting the type of 'BaseScal' (line 176)
BaseScal_91758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'BaseScal')

class TestDscal(BaseScal_91758, ):
    
    # Assigning a Attribute to a Name (line 177):
    
    # Assigning a Name to a Name (line 178):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 176, 0, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDscal.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDscal' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'TestDscal', TestDscal)

# Assigning a Attribute to a Name (line 177):
# Getting the type of 'fblas' (line 177)
fblas_91759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'fblas')
# Obtaining the member 'dscal' of a type (line 177)
dscal_91760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 16), fblas_91759, 'dscal')
# Getting the type of 'TestDscal'
TestDscal_91761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDscal')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDscal_91761, 'blas_func', dscal_91760)

# Assigning a Name to a Name (line 178):
# Getting the type of 'float64' (line 178)
float64_91762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'float64')
# Getting the type of 'TestDscal'
TestDscal_91763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDscal')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDscal_91763, 'dtype', float64_91762)


# SSA begins for try-except statement (line 181)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Declaration of the 'TestCscal' class
# Getting the type of 'BaseScal' (line 182)
BaseScal_91764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'BaseScal')

class TestCscal(BaseScal_91764, ):
    
    # Assigning a Attribute to a Name (line 183):
    
    # Assigning a Name to a Name (line 184):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCscal.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCscal' (line 182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'TestCscal', TestCscal)

# Assigning a Attribute to a Name (line 183):
# Getting the type of 'fblas' (line 183)
fblas_91765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'fblas')
# Obtaining the member 'cscal' of a type (line 183)
cscal_91766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 20), fblas_91765, 'cscal')
# Getting the type of 'TestCscal'
TestCscal_91767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCscal')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCscal_91767, 'blas_func', cscal_91766)

# Assigning a Name to a Name (line 184):
# Getting the type of 'complex64' (line 184)
complex64_91768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'complex64')
# Getting the type of 'TestCscal'
TestCscal_91769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCscal')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCscal_91769, 'dtype', complex64_91768)
# SSA branch for the except part of a try statement (line 181)
# SSA branch for the except 'AttributeError' branch of a try statement (line 181)
module_type_store.open_ssa_branch('except')
# Declaration of the 'TestCscal' class

class TestCscal:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCscal.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCscal' (line 186)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'TestCscal', TestCscal)
# SSA join for try-except statement (line 181)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TestZscal' class
# Getting the type of 'BaseScal' (line 190)
BaseScal_91770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'BaseScal')

class TestZscal(BaseScal_91770, ):
    
    # Assigning a Attribute to a Name (line 191):
    
    # Assigning a Name to a Name (line 192):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 190, 0, False)
        # Assigning a type to the variable 'self' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZscal.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestZscal' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'TestZscal', TestZscal)

# Assigning a Attribute to a Name (line 191):
# Getting the type of 'fblas' (line 191)
fblas_91771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'fblas')
# Obtaining the member 'zscal' of a type (line 191)
zscal_91772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 16), fblas_91771, 'zscal')
# Getting the type of 'TestZscal'
TestZscal_91773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZscal')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZscal_91773, 'blas_func', zscal_91772)

# Assigning a Name to a Name (line 192):
# Getting the type of 'complex128' (line 192)
complex128_91774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'complex128')
# Getting the type of 'TestZscal'
TestZscal_91775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZscal')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZscal_91775, 'dtype', complex128_91774)
# Declaration of the 'BaseCopy' class

class BaseCopy(object, ):
    str_91776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 4), 'str', ' Mixin class for copy testing ')

    @norecursion
    def test_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple'
        module_type_store = module_type_store.open_function_context('test_simple', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseCopy.test_simple.__dict__.__setitem__('stypy_localization', localization)
        BaseCopy.test_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseCopy.test_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseCopy.test_simple.__dict__.__setitem__('stypy_function_name', 'BaseCopy.test_simple')
        BaseCopy.test_simple.__dict__.__setitem__('stypy_param_names_list', [])
        BaseCopy.test_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseCopy.test_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseCopy.test_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseCopy.test_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseCopy.test_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseCopy.test_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseCopy.test_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple(...)' code ##################

        
        # Assigning a Call to a Name (line 202):
        
        # Assigning a Call to a Name (line 202):
        
        # Call to arange(...): (line 202)
        # Processing the call arguments (line 202)
        float_91778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 19), 'float')
        # Processing the call keyword arguments (line 202)
        # Getting the type of 'self' (line 202)
        self_91779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 202)
        dtype_91780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 29), self_91779, 'dtype')
        keyword_91781 = dtype_91780
        kwargs_91782 = {'dtype': keyword_91781}
        # Getting the type of 'arange' (line 202)
        arange_91777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 202)
        arange_call_result_91783 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), arange_91777, *[float_91778], **kwargs_91782)
        
        # Assigning a type to the variable 'x' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'x', arange_call_result_91783)
        
        # Assigning a Call to a Name (line 203):
        
        # Assigning a Call to a Name (line 203):
        
        # Call to zeros(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Call to shape(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'x' (line 203)
        x_91786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 24), 'x', False)
        # Processing the call keyword arguments (line 203)
        kwargs_91787 = {}
        # Getting the type of 'shape' (line 203)
        shape_91785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 18), 'shape', False)
        # Calling shape(args, kwargs) (line 203)
        shape_call_result_91788 = invoke(stypy.reporting.localization.Localization(__file__, 203, 18), shape_91785, *[x_91786], **kwargs_91787)
        
        # Getting the type of 'x' (line 203)
        x_91789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 28), 'x', False)
        # Obtaining the member 'dtype' of a type (line 203)
        dtype_91790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 28), x_91789, 'dtype')
        # Processing the call keyword arguments (line 203)
        kwargs_91791 = {}
        # Getting the type of 'zeros' (line 203)
        zeros_91784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 203)
        zeros_call_result_91792 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), zeros_91784, *[shape_call_result_91788, dtype_91790], **kwargs_91791)
        
        # Assigning a type to the variable 'y' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'y', zeros_call_result_91792)
        
        # Assigning a Call to a Name (line 204):
        
        # Assigning a Call to a Name (line 204):
        
        # Call to blas_func(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'x' (line 204)
        x_91795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 27), 'x', False)
        # Getting the type of 'y' (line 204)
        y_91796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 30), 'y', False)
        # Processing the call keyword arguments (line 204)
        kwargs_91797 = {}
        # Getting the type of 'self' (line 204)
        self_91793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 204)
        blas_func_91794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), self_91793, 'blas_func')
        # Calling blas_func(args, kwargs) (line 204)
        blas_func_call_result_91798 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), blas_func_91794, *[x_91795, y_91796], **kwargs_91797)
        
        # Assigning a type to the variable 'y' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'y', blas_func_call_result_91798)
        
        # Call to assert_array_equal(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'x' (line 205)
        x_91800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'x', False)
        # Getting the type of 'y' (line 205)
        y_91801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), 'y', False)
        # Processing the call keyword arguments (line 205)
        kwargs_91802 = {}
        # Getting the type of 'assert_array_equal' (line 205)
        assert_array_equal_91799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 205)
        assert_array_equal_call_result_91803 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), assert_array_equal_91799, *[x_91800, y_91801], **kwargs_91802)
        
        
        # ################# End of 'test_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_91804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91804)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple'
        return stypy_return_type_91804


    @norecursion
    def test_x_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_stride'
        module_type_store = module_type_store.open_function_context('test_x_stride', 207, 4, False)
        # Assigning a type to the variable 'self' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseCopy.test_x_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseCopy.test_x_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseCopy.test_x_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseCopy.test_x_stride.__dict__.__setitem__('stypy_function_name', 'BaseCopy.test_x_stride')
        BaseCopy.test_x_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseCopy.test_x_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseCopy.test_x_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseCopy.test_x_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseCopy.test_x_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseCopy.test_x_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseCopy.test_x_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseCopy.test_x_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_stride(...)' code ##################

        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Call to arange(...): (line 208)
        # Processing the call arguments (line 208)
        float_91806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 19), 'float')
        # Processing the call keyword arguments (line 208)
        # Getting the type of 'self' (line 208)
        self_91807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 208)
        dtype_91808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 29), self_91807, 'dtype')
        keyword_91809 = dtype_91808
        kwargs_91810 = {'dtype': keyword_91809}
        # Getting the type of 'arange' (line 208)
        arange_91805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 208)
        arange_call_result_91811 = invoke(stypy.reporting.localization.Localization(__file__, 208, 12), arange_91805, *[float_91806], **kwargs_91810)
        
        # Assigning a type to the variable 'x' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'x', arange_call_result_91811)
        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to zeros(...): (line 209)
        # Processing the call arguments (line 209)
        int_91813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 18), 'int')
        # Getting the type of 'x' (line 209)
        x_91814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 209)
        dtype_91815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 21), x_91814, 'dtype')
        # Processing the call keyword arguments (line 209)
        kwargs_91816 = {}
        # Getting the type of 'zeros' (line 209)
        zeros_91812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 209)
        zeros_call_result_91817 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), zeros_91812, *[int_91813, dtype_91815], **kwargs_91816)
        
        # Assigning a type to the variable 'y' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'y', zeros_call_result_91817)
        
        # Assigning a Call to a Name (line 210):
        
        # Assigning a Call to a Name (line 210):
        
        # Call to blas_func(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'x' (line 210)
        x_91820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 27), 'x', False)
        # Getting the type of 'y' (line 210)
        y_91821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'y', False)
        # Processing the call keyword arguments (line 210)
        int_91822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 35), 'int')
        keyword_91823 = int_91822
        int_91824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 43), 'int')
        keyword_91825 = int_91824
        kwargs_91826 = {'incx': keyword_91825, 'n': keyword_91823}
        # Getting the type of 'self' (line 210)
        self_91818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 210)
        blas_func_91819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), self_91818, 'blas_func')
        # Calling blas_func(args, kwargs) (line 210)
        blas_func_call_result_91827 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), blas_func_91819, *[x_91820, y_91821], **kwargs_91826)
        
        # Assigning a type to the variable 'y' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'y', blas_func_call_result_91827)
        
        # Call to assert_array_equal(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Obtaining the type of the subscript
        int_91829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 31), 'int')
        slice_91830 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 211, 27), None, None, int_91829)
        # Getting the type of 'x' (line 211)
        x_91831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 27), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___91832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 27), x_91831, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 211)
        subscript_call_result_91833 = invoke(stypy.reporting.localization.Localization(__file__, 211, 27), getitem___91832, slice_91830)
        
        # Getting the type of 'y' (line 211)
        y_91834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 35), 'y', False)
        # Processing the call keyword arguments (line 211)
        kwargs_91835 = {}
        # Getting the type of 'assert_array_equal' (line 211)
        assert_array_equal_91828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 211)
        assert_array_equal_call_result_91836 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), assert_array_equal_91828, *[subscript_call_result_91833, y_91834], **kwargs_91835)
        
        
        # ################# End of 'test_x_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_91837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91837)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_stride'
        return stypy_return_type_91837


    @norecursion
    def test_y_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_y_stride'
        module_type_store = module_type_store.open_function_context('test_y_stride', 213, 4, False)
        # Assigning a type to the variable 'self' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseCopy.test_y_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseCopy.test_y_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseCopy.test_y_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseCopy.test_y_stride.__dict__.__setitem__('stypy_function_name', 'BaseCopy.test_y_stride')
        BaseCopy.test_y_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseCopy.test_y_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseCopy.test_y_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseCopy.test_y_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseCopy.test_y_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseCopy.test_y_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseCopy.test_y_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseCopy.test_y_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_y_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_y_stride(...)' code ##################

        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Call to arange(...): (line 214)
        # Processing the call arguments (line 214)
        float_91839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 19), 'float')
        # Processing the call keyword arguments (line 214)
        # Getting the type of 'self' (line 214)
        self_91840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 214)
        dtype_91841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 29), self_91840, 'dtype')
        keyword_91842 = dtype_91841
        kwargs_91843 = {'dtype': keyword_91842}
        # Getting the type of 'arange' (line 214)
        arange_91838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 214)
        arange_call_result_91844 = invoke(stypy.reporting.localization.Localization(__file__, 214, 12), arange_91838, *[float_91839], **kwargs_91843)
        
        # Assigning a type to the variable 'x' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'x', arange_call_result_91844)
        
        # Assigning a Call to a Name (line 215):
        
        # Assigning a Call to a Name (line 215):
        
        # Call to zeros(...): (line 215)
        # Processing the call arguments (line 215)
        int_91846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 18), 'int')
        # Getting the type of 'x' (line 215)
        x_91847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 215)
        dtype_91848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 21), x_91847, 'dtype')
        # Processing the call keyword arguments (line 215)
        kwargs_91849 = {}
        # Getting the type of 'zeros' (line 215)
        zeros_91845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 215)
        zeros_call_result_91850 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), zeros_91845, *[int_91846, dtype_91848], **kwargs_91849)
        
        # Assigning a type to the variable 'y' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'y', zeros_call_result_91850)
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to blas_func(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'x' (line 216)
        x_91853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 27), 'x', False)
        # Getting the type of 'y' (line 216)
        y_91854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 30), 'y', False)
        # Processing the call keyword arguments (line 216)
        int_91855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 35), 'int')
        keyword_91856 = int_91855
        int_91857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 43), 'int')
        keyword_91858 = int_91857
        kwargs_91859 = {'incy': keyword_91858, 'n': keyword_91856}
        # Getting the type of 'self' (line 216)
        self_91851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 216)
        blas_func_91852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), self_91851, 'blas_func')
        # Calling blas_func(args, kwargs) (line 216)
        blas_func_call_result_91860 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), blas_func_91852, *[x_91853, y_91854], **kwargs_91859)
        
        # Assigning a type to the variable 'y' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'y', blas_func_call_result_91860)
        
        # Call to assert_array_equal(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'x' (line 217)
        x_91862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 27), 'x', False)
        
        # Obtaining the type of the subscript
        int_91863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 34), 'int')
        slice_91864 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 217, 30), None, None, int_91863)
        # Getting the type of 'y' (line 217)
        y_91865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 30), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___91866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 30), y_91865, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_91867 = invoke(stypy.reporting.localization.Localization(__file__, 217, 30), getitem___91866, slice_91864)
        
        # Processing the call keyword arguments (line 217)
        kwargs_91868 = {}
        # Getting the type of 'assert_array_equal' (line 217)
        assert_array_equal_91861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 217)
        assert_array_equal_call_result_91869 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), assert_array_equal_91861, *[x_91862, subscript_call_result_91867], **kwargs_91868)
        
        
        # ################# End of 'test_y_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_y_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_91870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91870)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_y_stride'
        return stypy_return_type_91870


    @norecursion
    def test_x_and_y_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_and_y_stride'
        module_type_store = module_type_store.open_function_context('test_x_and_y_stride', 219, 4, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseCopy.test_x_and_y_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseCopy.test_x_and_y_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseCopy.test_x_and_y_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseCopy.test_x_and_y_stride.__dict__.__setitem__('stypy_function_name', 'BaseCopy.test_x_and_y_stride')
        BaseCopy.test_x_and_y_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseCopy.test_x_and_y_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseCopy.test_x_and_y_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseCopy.test_x_and_y_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseCopy.test_x_and_y_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseCopy.test_x_and_y_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseCopy.test_x_and_y_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseCopy.test_x_and_y_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_and_y_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_and_y_stride(...)' code ##################

        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to arange(...): (line 220)
        # Processing the call arguments (line 220)
        float_91872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 19), 'float')
        # Processing the call keyword arguments (line 220)
        # Getting the type of 'self' (line 220)
        self_91873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 220)
        dtype_91874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 30), self_91873, 'dtype')
        keyword_91875 = dtype_91874
        kwargs_91876 = {'dtype': keyword_91875}
        # Getting the type of 'arange' (line 220)
        arange_91871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 220)
        arange_call_result_91877 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), arange_91871, *[float_91872], **kwargs_91876)
        
        # Assigning a type to the variable 'x' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'x', arange_call_result_91877)
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to zeros(...): (line 221)
        # Processing the call arguments (line 221)
        int_91879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 18), 'int')
        # Getting the type of 'x' (line 221)
        x_91880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 221)
        dtype_91881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 21), x_91880, 'dtype')
        # Processing the call keyword arguments (line 221)
        kwargs_91882 = {}
        # Getting the type of 'zeros' (line 221)
        zeros_91878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 221)
        zeros_call_result_91883 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), zeros_91878, *[int_91879, dtype_91881], **kwargs_91882)
        
        # Assigning a type to the variable 'y' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'y', zeros_call_result_91883)
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to blas_func(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'x' (line 222)
        x_91886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 27), 'x', False)
        # Getting the type of 'y' (line 222)
        y_91887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 30), 'y', False)
        # Processing the call keyword arguments (line 222)
        int_91888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 35), 'int')
        keyword_91889 = int_91888
        int_91890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 43), 'int')
        keyword_91891 = int_91890
        int_91892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 51), 'int')
        keyword_91893 = int_91892
        kwargs_91894 = {'incx': keyword_91891, 'incy': keyword_91893, 'n': keyword_91889}
        # Getting the type of 'self' (line 222)
        self_91884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 222)
        blas_func_91885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), self_91884, 'blas_func')
        # Calling blas_func(args, kwargs) (line 222)
        blas_func_call_result_91895 = invoke(stypy.reporting.localization.Localization(__file__, 222, 12), blas_func_91885, *[x_91886, y_91887], **kwargs_91894)
        
        # Assigning a type to the variable 'y' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'y', blas_func_call_result_91895)
        
        # Call to assert_array_equal(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Obtaining the type of the subscript
        int_91897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 31), 'int')
        slice_91898 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 223, 27), None, None, int_91897)
        # Getting the type of 'x' (line 223)
        x_91899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 27), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___91900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 27), x_91899, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_91901 = invoke(stypy.reporting.localization.Localization(__file__, 223, 27), getitem___91900, slice_91898)
        
        
        # Obtaining the type of the subscript
        int_91902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 39), 'int')
        slice_91903 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 223, 35), None, None, int_91902)
        # Getting the type of 'y' (line 223)
        y_91904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 35), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___91905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 35), y_91904, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_91906 = invoke(stypy.reporting.localization.Localization(__file__, 223, 35), getitem___91905, slice_91903)
        
        # Processing the call keyword arguments (line 223)
        kwargs_91907 = {}
        # Getting the type of 'assert_array_equal' (line 223)
        assert_array_equal_91896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 223)
        assert_array_equal_call_result_91908 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), assert_array_equal_91896, *[subscript_call_result_91901, subscript_call_result_91906], **kwargs_91907)
        
        
        # ################# End of 'test_x_and_y_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_and_y_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_91909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91909)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_and_y_stride'
        return stypy_return_type_91909


    @norecursion
    def test_x_bad_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_bad_size'
        module_type_store = module_type_store.open_function_context('test_x_bad_size', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseCopy.test_x_bad_size.__dict__.__setitem__('stypy_localization', localization)
        BaseCopy.test_x_bad_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseCopy.test_x_bad_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseCopy.test_x_bad_size.__dict__.__setitem__('stypy_function_name', 'BaseCopy.test_x_bad_size')
        BaseCopy.test_x_bad_size.__dict__.__setitem__('stypy_param_names_list', [])
        BaseCopy.test_x_bad_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseCopy.test_x_bad_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseCopy.test_x_bad_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseCopy.test_x_bad_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseCopy.test_x_bad_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseCopy.test_x_bad_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseCopy.test_x_bad_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_bad_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_bad_size(...)' code ##################

        
        # Assigning a Call to a Name (line 226):
        
        # Assigning a Call to a Name (line 226):
        
        # Call to arange(...): (line 226)
        # Processing the call arguments (line 226)
        float_91911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 19), 'float')
        # Processing the call keyword arguments (line 226)
        # Getting the type of 'self' (line 226)
        self_91912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 226)
        dtype_91913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 30), self_91912, 'dtype')
        keyword_91914 = dtype_91913
        kwargs_91915 = {'dtype': keyword_91914}
        # Getting the type of 'arange' (line 226)
        arange_91910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 226)
        arange_call_result_91916 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), arange_91910, *[float_91911], **kwargs_91915)
        
        # Assigning a type to the variable 'x' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'x', arange_call_result_91916)
        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to zeros(...): (line 227)
        # Processing the call arguments (line 227)
        int_91918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 18), 'int')
        # Getting the type of 'x' (line 227)
        x_91919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 227)
        dtype_91920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 21), x_91919, 'dtype')
        # Processing the call keyword arguments (line 227)
        kwargs_91921 = {}
        # Getting the type of 'zeros' (line 227)
        zeros_91917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 227)
        zeros_call_result_91922 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), zeros_91917, *[int_91918, dtype_91920], **kwargs_91921)
        
        # Assigning a type to the variable 'y' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'y', zeros_call_result_91922)
        
        
        # SSA begins for try-except statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to blas_func(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'x' (line 229)
        x_91925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'x', False)
        # Getting the type of 'y' (line 229)
        y_91926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 30), 'y', False)
        # Processing the call keyword arguments (line 229)
        int_91927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 35), 'int')
        keyword_91928 = int_91927
        int_91929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 43), 'int')
        keyword_91930 = int_91929
        kwargs_91931 = {'incx': keyword_91930, 'n': keyword_91928}
        # Getting the type of 'self' (line 229)
        self_91923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 229)
        blas_func_91924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), self_91923, 'blas_func')
        # Calling blas_func(args, kwargs) (line 229)
        blas_func_call_result_91932 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), blas_func_91924, *[x_91925, y_91926], **kwargs_91931)
        
        # SSA branch for the except part of a try statement (line 228)
        # SSA branch for the except '<any exception>' branch of a try statement (line 228)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 233)
        # Processing the call arguments (line 233)
        int_91934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 16), 'int')
        # Processing the call keyword arguments (line 233)
        kwargs_91935 = {}
        # Getting the type of 'assert_' (line 233)
        assert__91933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 233)
        assert__call_result_91936 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), assert__91933, *[int_91934], **kwargs_91935)
        
        
        # ################# End of 'test_x_bad_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_bad_size' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_91937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91937)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_bad_size'
        return stypy_return_type_91937


    @norecursion
    def test_y_bad_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_y_bad_size'
        module_type_store = module_type_store.open_function_context('test_y_bad_size', 235, 4, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseCopy.test_y_bad_size.__dict__.__setitem__('stypy_localization', localization)
        BaseCopy.test_y_bad_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseCopy.test_y_bad_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseCopy.test_y_bad_size.__dict__.__setitem__('stypy_function_name', 'BaseCopy.test_y_bad_size')
        BaseCopy.test_y_bad_size.__dict__.__setitem__('stypy_param_names_list', [])
        BaseCopy.test_y_bad_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseCopy.test_y_bad_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseCopy.test_y_bad_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseCopy.test_y_bad_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseCopy.test_y_bad_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseCopy.test_y_bad_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseCopy.test_y_bad_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_y_bad_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_y_bad_size(...)' code ##################

        
        # Assigning a Call to a Name (line 236):
        
        # Assigning a Call to a Name (line 236):
        
        # Call to arange(...): (line 236)
        # Processing the call arguments (line 236)
        float_91939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 19), 'float')
        # Processing the call keyword arguments (line 236)
        # Getting the type of 'self' (line 236)
        self_91940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 236)
        dtype_91941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 30), self_91940, 'dtype')
        keyword_91942 = dtype_91941
        kwargs_91943 = {'dtype': keyword_91942}
        # Getting the type of 'arange' (line 236)
        arange_91938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 236)
        arange_call_result_91944 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), arange_91938, *[float_91939], **kwargs_91943)
        
        # Assigning a type to the variable 'x' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'x', arange_call_result_91944)
        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to zeros(...): (line 237)
        # Processing the call arguments (line 237)
        int_91946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 18), 'int')
        # Getting the type of 'x' (line 237)
        x_91947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 237)
        dtype_91948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 21), x_91947, 'dtype')
        # Processing the call keyword arguments (line 237)
        kwargs_91949 = {}
        # Getting the type of 'zeros' (line 237)
        zeros_91945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 237)
        zeros_call_result_91950 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), zeros_91945, *[int_91946, dtype_91948], **kwargs_91949)
        
        # Assigning a type to the variable 'y' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'y', zeros_call_result_91950)
        
        
        # SSA begins for try-except statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to blas_func(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'x' (line 239)
        x_91953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'x', False)
        # Getting the type of 'y' (line 239)
        y_91954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 30), 'y', False)
        # Processing the call keyword arguments (line 239)
        int_91955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 35), 'int')
        keyword_91956 = int_91955
        int_91957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 43), 'int')
        keyword_91958 = int_91957
        kwargs_91959 = {'incy': keyword_91958, 'n': keyword_91956}
        # Getting the type of 'self' (line 239)
        self_91951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 239)
        blas_func_91952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), self_91951, 'blas_func')
        # Calling blas_func(args, kwargs) (line 239)
        blas_func_call_result_91960 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), blas_func_91952, *[x_91953, y_91954], **kwargs_91959)
        
        # SSA branch for the except part of a try statement (line 238)
        # SSA branch for the except '<any exception>' branch of a try statement (line 238)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 243)
        # Processing the call arguments (line 243)
        int_91962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 16), 'int')
        # Processing the call keyword arguments (line 243)
        kwargs_91963 = {}
        # Getting the type of 'assert_' (line 243)
        assert__91961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 243)
        assert__call_result_91964 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), assert__91961, *[int_91962], **kwargs_91963)
        
        
        # ################# End of 'test_y_bad_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_y_bad_size' in the type store
        # Getting the type of 'stypy_return_type' (line 235)
        stypy_return_type_91965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91965)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_y_bad_size'
        return stypy_return_type_91965


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseCopy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BaseCopy' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'BaseCopy', BaseCopy)


# SSA begins for try-except statement (line 253)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Declaration of the 'TestScopy' class
# Getting the type of 'BaseCopy' (line 254)
BaseCopy_91966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'BaseCopy')

class TestScopy(BaseCopy_91966, ):
    
    # Assigning a Attribute to a Name (line 255):
    
    # Assigning a Name to a Name (line 256):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScopy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestScopy' (line 254)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'TestScopy', TestScopy)

# Assigning a Attribute to a Name (line 255):
# Getting the type of 'fblas' (line 255)
fblas_91967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'fblas')
# Obtaining the member 'scopy' of a type (line 255)
scopy_91968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 20), fblas_91967, 'scopy')
# Getting the type of 'TestScopy'
TestScopy_91969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestScopy')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestScopy_91969, 'blas_func', scopy_91968)

# Assigning a Name to a Name (line 256):
# Getting the type of 'float32' (line 256)
float32_91970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'float32')
# Getting the type of 'TestScopy'
TestScopy_91971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestScopy')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestScopy_91971, 'dtype', float32_91970)
# SSA branch for the except part of a try statement (line 253)
# SSA branch for the except 'AttributeError' branch of a try statement (line 253)
module_type_store.open_ssa_branch('except')
# Declaration of the 'TestScopy' class

class TestScopy:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 258, 4, False)
        # Assigning a type to the variable 'self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScopy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestScopy' (line 258)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'TestScopy', TestScopy)
# SSA join for try-except statement (line 253)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TestDcopy' class
# Getting the type of 'BaseCopy' (line 262)
BaseCopy_91972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'BaseCopy')

class TestDcopy(BaseCopy_91972, ):
    
    # Assigning a Attribute to a Name (line 263):
    
    # Assigning a Name to a Name (line 264):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 262, 0, False)
        # Assigning a type to the variable 'self' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDcopy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDcopy' (line 262)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'TestDcopy', TestDcopy)

# Assigning a Attribute to a Name (line 263):
# Getting the type of 'fblas' (line 263)
fblas_91973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'fblas')
# Obtaining the member 'dcopy' of a type (line 263)
dcopy_91974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 16), fblas_91973, 'dcopy')
# Getting the type of 'TestDcopy'
TestDcopy_91975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDcopy')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDcopy_91975, 'blas_func', dcopy_91974)

# Assigning a Name to a Name (line 264):
# Getting the type of 'float64' (line 264)
float64_91976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'float64')
# Getting the type of 'TestDcopy'
TestDcopy_91977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDcopy')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDcopy_91977, 'dtype', float64_91976)


# SSA begins for try-except statement (line 267)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Declaration of the 'TestCcopy' class
# Getting the type of 'BaseCopy' (line 268)
BaseCopy_91978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'BaseCopy')

class TestCcopy(BaseCopy_91978, ):
    
    # Assigning a Attribute to a Name (line 269):
    
    # Assigning a Name to a Name (line 270):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCcopy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCcopy' (line 268)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'TestCcopy', TestCcopy)

# Assigning a Attribute to a Name (line 269):
# Getting the type of 'fblas' (line 269)
fblas_91979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'fblas')
# Obtaining the member 'ccopy' of a type (line 269)
ccopy_91980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 20), fblas_91979, 'ccopy')
# Getting the type of 'TestCcopy'
TestCcopy_91981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCcopy')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCcopy_91981, 'blas_func', ccopy_91980)

# Assigning a Name to a Name (line 270):
# Getting the type of 'complex64' (line 270)
complex64_91982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'complex64')
# Getting the type of 'TestCcopy'
TestCcopy_91983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCcopy')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCcopy_91983, 'dtype', complex64_91982)
# SSA branch for the except part of a try statement (line 267)
# SSA branch for the except 'AttributeError' branch of a try statement (line 267)
module_type_store.open_ssa_branch('except')
# Declaration of the 'TestCcopy' class

class TestCcopy:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCcopy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCcopy' (line 272)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'TestCcopy', TestCcopy)
# SSA join for try-except statement (line 267)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TestZcopy' class
# Getting the type of 'BaseCopy' (line 276)
BaseCopy_91984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'BaseCopy')

class TestZcopy(BaseCopy_91984, ):
    
    # Assigning a Attribute to a Name (line 277):
    
    # Assigning a Name to a Name (line 278):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 276, 0, False)
        # Assigning a type to the variable 'self' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZcopy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestZcopy' (line 276)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 0), 'TestZcopy', TestZcopy)

# Assigning a Attribute to a Name (line 277):
# Getting the type of 'fblas' (line 277)
fblas_91985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'fblas')
# Obtaining the member 'zcopy' of a type (line 277)
zcopy_91986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 16), fblas_91985, 'zcopy')
# Getting the type of 'TestZcopy'
TestZcopy_91987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZcopy')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZcopy_91987, 'blas_func', zcopy_91986)

# Assigning a Name to a Name (line 278):
# Getting the type of 'complex128' (line 278)
complex128_91988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'complex128')
# Getting the type of 'TestZcopy'
TestZcopy_91989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZcopy')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZcopy_91989, 'dtype', complex128_91988)
# Declaration of the 'BaseSwap' class

class BaseSwap(object, ):
    str_91990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 4), 'str', ' Mixin class for swap tests ')

    @norecursion
    def test_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple'
        module_type_store = module_type_store.open_function_context('test_simple', 287, 4, False)
        # Assigning a type to the variable 'self' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseSwap.test_simple.__dict__.__setitem__('stypy_localization', localization)
        BaseSwap.test_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseSwap.test_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseSwap.test_simple.__dict__.__setitem__('stypy_function_name', 'BaseSwap.test_simple')
        BaseSwap.test_simple.__dict__.__setitem__('stypy_param_names_list', [])
        BaseSwap.test_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseSwap.test_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseSwap.test_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseSwap.test_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseSwap.test_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseSwap.test_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseSwap.test_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple(...)' code ##################

        
        # Assigning a Call to a Name (line 288):
        
        # Assigning a Call to a Name (line 288):
        
        # Call to arange(...): (line 288)
        # Processing the call arguments (line 288)
        float_91992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 19), 'float')
        # Processing the call keyword arguments (line 288)
        # Getting the type of 'self' (line 288)
        self_91993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 288)
        dtype_91994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 29), self_91993, 'dtype')
        keyword_91995 = dtype_91994
        kwargs_91996 = {'dtype': keyword_91995}
        # Getting the type of 'arange' (line 288)
        arange_91991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 288)
        arange_call_result_91997 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), arange_91991, *[float_91992], **kwargs_91996)
        
        # Assigning a type to the variable 'x' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'x', arange_call_result_91997)
        
        # Assigning a Call to a Name (line 289):
        
        # Assigning a Call to a Name (line 289):
        
        # Call to zeros(...): (line 289)
        # Processing the call arguments (line 289)
        
        # Call to shape(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'x' (line 289)
        x_92000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 24), 'x', False)
        # Processing the call keyword arguments (line 289)
        kwargs_92001 = {}
        # Getting the type of 'shape' (line 289)
        shape_91999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 18), 'shape', False)
        # Calling shape(args, kwargs) (line 289)
        shape_call_result_92002 = invoke(stypy.reporting.localization.Localization(__file__, 289, 18), shape_91999, *[x_92000], **kwargs_92001)
        
        # Getting the type of 'x' (line 289)
        x_92003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 28), 'x', False)
        # Obtaining the member 'dtype' of a type (line 289)
        dtype_92004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 28), x_92003, 'dtype')
        # Processing the call keyword arguments (line 289)
        kwargs_92005 = {}
        # Getting the type of 'zeros' (line 289)
        zeros_91998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 289)
        zeros_call_result_92006 = invoke(stypy.reporting.localization.Localization(__file__, 289, 12), zeros_91998, *[shape_call_result_92002, dtype_92004], **kwargs_92005)
        
        # Assigning a type to the variable 'y' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'y', zeros_call_result_92006)
        
        # Assigning a Call to a Name (line 290):
        
        # Assigning a Call to a Name (line 290):
        
        # Call to copy(...): (line 290)
        # Processing the call keyword arguments (line 290)
        kwargs_92009 = {}
        # Getting the type of 'y' (line 290)
        y_92007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 20), 'y', False)
        # Obtaining the member 'copy' of a type (line 290)
        copy_92008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 20), y_92007, 'copy')
        # Calling copy(args, kwargs) (line 290)
        copy_call_result_92010 = invoke(stypy.reporting.localization.Localization(__file__, 290, 20), copy_92008, *[], **kwargs_92009)
        
        # Assigning a type to the variable 'desired_x' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'desired_x', copy_call_result_92010)
        
        # Assigning a Call to a Name (line 291):
        
        # Assigning a Call to a Name (line 291):
        
        # Call to copy(...): (line 291)
        # Processing the call keyword arguments (line 291)
        kwargs_92013 = {}
        # Getting the type of 'x' (line 291)
        x_92011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'x', False)
        # Obtaining the member 'copy' of a type (line 291)
        copy_92012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 20), x_92011, 'copy')
        # Calling copy(args, kwargs) (line 291)
        copy_call_result_92014 = invoke(stypy.reporting.localization.Localization(__file__, 291, 20), copy_92012, *[], **kwargs_92013)
        
        # Assigning a type to the variable 'desired_y' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'desired_y', copy_call_result_92014)
        
        # Assigning a Call to a Tuple (line 292):
        
        # Assigning a Subscript to a Name (line 292):
        
        # Obtaining the type of the subscript
        int_92015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 8), 'int')
        
        # Call to blas_func(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'x' (line 292)
        x_92018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'x', False)
        # Getting the type of 'y' (line 292)
        y_92019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 33), 'y', False)
        # Processing the call keyword arguments (line 292)
        kwargs_92020 = {}
        # Getting the type of 'self' (line 292)
        self_92016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 292)
        blas_func_92017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 15), self_92016, 'blas_func')
        # Calling blas_func(args, kwargs) (line 292)
        blas_func_call_result_92021 = invoke(stypy.reporting.localization.Localization(__file__, 292, 15), blas_func_92017, *[x_92018, y_92019], **kwargs_92020)
        
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___92022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), blas_func_call_result_92021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_92023 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), getitem___92022, int_92015)
        
        # Assigning a type to the variable 'tuple_var_assignment_91202' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_91202', subscript_call_result_92023)
        
        # Assigning a Subscript to a Name (line 292):
        
        # Obtaining the type of the subscript
        int_92024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 8), 'int')
        
        # Call to blas_func(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'x' (line 292)
        x_92027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'x', False)
        # Getting the type of 'y' (line 292)
        y_92028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 33), 'y', False)
        # Processing the call keyword arguments (line 292)
        kwargs_92029 = {}
        # Getting the type of 'self' (line 292)
        self_92025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 292)
        blas_func_92026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 15), self_92025, 'blas_func')
        # Calling blas_func(args, kwargs) (line 292)
        blas_func_call_result_92030 = invoke(stypy.reporting.localization.Localization(__file__, 292, 15), blas_func_92026, *[x_92027, y_92028], **kwargs_92029)
        
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___92031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), blas_func_call_result_92030, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_92032 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), getitem___92031, int_92024)
        
        # Assigning a type to the variable 'tuple_var_assignment_91203' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_91203', subscript_call_result_92032)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'tuple_var_assignment_91202' (line 292)
        tuple_var_assignment_91202_92033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_91202')
        # Assigning a type to the variable 'x' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'x', tuple_var_assignment_91202_92033)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'tuple_var_assignment_91203' (line 292)
        tuple_var_assignment_91203_92034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_91203')
        # Assigning a type to the variable 'y' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 11), 'y', tuple_var_assignment_91203_92034)
        
        # Call to assert_array_equal(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'desired_x' (line 293)
        desired_x_92036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 27), 'desired_x', False)
        # Getting the type of 'x' (line 293)
        x_92037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 38), 'x', False)
        # Processing the call keyword arguments (line 293)
        kwargs_92038 = {}
        # Getting the type of 'assert_array_equal' (line 293)
        assert_array_equal_92035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 293)
        assert_array_equal_call_result_92039 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), assert_array_equal_92035, *[desired_x_92036, x_92037], **kwargs_92038)
        
        
        # Call to assert_array_equal(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'desired_y' (line 294)
        desired_y_92041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 27), 'desired_y', False)
        # Getting the type of 'y' (line 294)
        y_92042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 38), 'y', False)
        # Processing the call keyword arguments (line 294)
        kwargs_92043 = {}
        # Getting the type of 'assert_array_equal' (line 294)
        assert_array_equal_92040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 294)
        assert_array_equal_call_result_92044 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), assert_array_equal_92040, *[desired_y_92041, y_92042], **kwargs_92043)
        
        
        # ################# End of 'test_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 287)
        stypy_return_type_92045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92045)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple'
        return stypy_return_type_92045


    @norecursion
    def test_x_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_stride'
        module_type_store = module_type_store.open_function_context('test_x_stride', 296, 4, False)
        # Assigning a type to the variable 'self' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseSwap.test_x_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseSwap.test_x_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseSwap.test_x_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseSwap.test_x_stride.__dict__.__setitem__('stypy_function_name', 'BaseSwap.test_x_stride')
        BaseSwap.test_x_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseSwap.test_x_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseSwap.test_x_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseSwap.test_x_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseSwap.test_x_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseSwap.test_x_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseSwap.test_x_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseSwap.test_x_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_stride(...)' code ##################

        
        # Assigning a Call to a Name (line 297):
        
        # Assigning a Call to a Name (line 297):
        
        # Call to arange(...): (line 297)
        # Processing the call arguments (line 297)
        float_92047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 19), 'float')
        # Processing the call keyword arguments (line 297)
        # Getting the type of 'self' (line 297)
        self_92048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 297)
        dtype_92049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 29), self_92048, 'dtype')
        keyword_92050 = dtype_92049
        kwargs_92051 = {'dtype': keyword_92050}
        # Getting the type of 'arange' (line 297)
        arange_92046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 297)
        arange_call_result_92052 = invoke(stypy.reporting.localization.Localization(__file__, 297, 12), arange_92046, *[float_92047], **kwargs_92051)
        
        # Assigning a type to the variable 'x' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'x', arange_call_result_92052)
        
        # Assigning a Call to a Name (line 298):
        
        # Assigning a Call to a Name (line 298):
        
        # Call to zeros(...): (line 298)
        # Processing the call arguments (line 298)
        int_92054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 18), 'int')
        # Getting the type of 'x' (line 298)
        x_92055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 298)
        dtype_92056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 21), x_92055, 'dtype')
        # Processing the call keyword arguments (line 298)
        kwargs_92057 = {}
        # Getting the type of 'zeros' (line 298)
        zeros_92053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 298)
        zeros_call_result_92058 = invoke(stypy.reporting.localization.Localization(__file__, 298, 12), zeros_92053, *[int_92054, dtype_92056], **kwargs_92057)
        
        # Assigning a type to the variable 'y' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'y', zeros_call_result_92058)
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to copy(...): (line 299)
        # Processing the call keyword arguments (line 299)
        kwargs_92061 = {}
        # Getting the type of 'y' (line 299)
        y_92059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 20), 'y', False)
        # Obtaining the member 'copy' of a type (line 299)
        copy_92060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 20), y_92059, 'copy')
        # Calling copy(args, kwargs) (line 299)
        copy_call_result_92062 = invoke(stypy.reporting.localization.Localization(__file__, 299, 20), copy_92060, *[], **kwargs_92061)
        
        # Assigning a type to the variable 'desired_x' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'desired_x', copy_call_result_92062)
        
        # Assigning a Subscript to a Name (line 300):
        
        # Assigning a Subscript to a Name (line 300):
        
        # Obtaining the type of the subscript
        int_92063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 31), 'int')
        slice_92064 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 300, 20), None, None, int_92063)
        
        # Call to copy(...): (line 300)
        # Processing the call keyword arguments (line 300)
        kwargs_92067 = {}
        # Getting the type of 'x' (line 300)
        x_92065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'x', False)
        # Obtaining the member 'copy' of a type (line 300)
        copy_92066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 20), x_92065, 'copy')
        # Calling copy(args, kwargs) (line 300)
        copy_call_result_92068 = invoke(stypy.reporting.localization.Localization(__file__, 300, 20), copy_92066, *[], **kwargs_92067)
        
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___92069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 20), copy_call_result_92068, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
        subscript_call_result_92070 = invoke(stypy.reporting.localization.Localization(__file__, 300, 20), getitem___92069, slice_92064)
        
        # Assigning a type to the variable 'desired_y' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'desired_y', subscript_call_result_92070)
        
        # Assigning a Call to a Tuple (line 301):
        
        # Assigning a Subscript to a Name (line 301):
        
        # Obtaining the type of the subscript
        int_92071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 8), 'int')
        
        # Call to blas_func(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'x' (line 301)
        x_92074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 30), 'x', False)
        # Getting the type of 'y' (line 301)
        y_92075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 33), 'y', False)
        # Processing the call keyword arguments (line 301)
        int_92076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 38), 'int')
        keyword_92077 = int_92076
        int_92078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 46), 'int')
        keyword_92079 = int_92078
        kwargs_92080 = {'incx': keyword_92079, 'n': keyword_92077}
        # Getting the type of 'self' (line 301)
        self_92072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 301)
        blas_func_92073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 15), self_92072, 'blas_func')
        # Calling blas_func(args, kwargs) (line 301)
        blas_func_call_result_92081 = invoke(stypy.reporting.localization.Localization(__file__, 301, 15), blas_func_92073, *[x_92074, y_92075], **kwargs_92080)
        
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___92082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), blas_func_call_result_92081, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 301)
        subscript_call_result_92083 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), getitem___92082, int_92071)
        
        # Assigning a type to the variable 'tuple_var_assignment_91204' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'tuple_var_assignment_91204', subscript_call_result_92083)
        
        # Assigning a Subscript to a Name (line 301):
        
        # Obtaining the type of the subscript
        int_92084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 8), 'int')
        
        # Call to blas_func(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'x' (line 301)
        x_92087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 30), 'x', False)
        # Getting the type of 'y' (line 301)
        y_92088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 33), 'y', False)
        # Processing the call keyword arguments (line 301)
        int_92089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 38), 'int')
        keyword_92090 = int_92089
        int_92091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 46), 'int')
        keyword_92092 = int_92091
        kwargs_92093 = {'incx': keyword_92092, 'n': keyword_92090}
        # Getting the type of 'self' (line 301)
        self_92085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 301)
        blas_func_92086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 15), self_92085, 'blas_func')
        # Calling blas_func(args, kwargs) (line 301)
        blas_func_call_result_92094 = invoke(stypy.reporting.localization.Localization(__file__, 301, 15), blas_func_92086, *[x_92087, y_92088], **kwargs_92093)
        
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___92095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), blas_func_call_result_92094, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 301)
        subscript_call_result_92096 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), getitem___92095, int_92084)
        
        # Assigning a type to the variable 'tuple_var_assignment_91205' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'tuple_var_assignment_91205', subscript_call_result_92096)
        
        # Assigning a Name to a Name (line 301):
        # Getting the type of 'tuple_var_assignment_91204' (line 301)
        tuple_var_assignment_91204_92097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'tuple_var_assignment_91204')
        # Assigning a type to the variable 'x' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'x', tuple_var_assignment_91204_92097)
        
        # Assigning a Name to a Name (line 301):
        # Getting the type of 'tuple_var_assignment_91205' (line 301)
        tuple_var_assignment_91205_92098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'tuple_var_assignment_91205')
        # Assigning a type to the variable 'y' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'y', tuple_var_assignment_91205_92098)
        
        # Call to assert_array_equal(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'desired_x' (line 302)
        desired_x_92100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 27), 'desired_x', False)
        
        # Obtaining the type of the subscript
        int_92101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 42), 'int')
        slice_92102 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 302, 38), None, None, int_92101)
        # Getting the type of 'x' (line 302)
        x_92103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 38), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 302)
        getitem___92104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 38), x_92103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 302)
        subscript_call_result_92105 = invoke(stypy.reporting.localization.Localization(__file__, 302, 38), getitem___92104, slice_92102)
        
        # Processing the call keyword arguments (line 302)
        kwargs_92106 = {}
        # Getting the type of 'assert_array_equal' (line 302)
        assert_array_equal_92099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 302)
        assert_array_equal_call_result_92107 = invoke(stypy.reporting.localization.Localization(__file__, 302, 8), assert_array_equal_92099, *[desired_x_92100, subscript_call_result_92105], **kwargs_92106)
        
        
        # Call to assert_array_equal(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'desired_y' (line 303)
        desired_y_92109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 27), 'desired_y', False)
        # Getting the type of 'y' (line 303)
        y_92110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 38), 'y', False)
        # Processing the call keyword arguments (line 303)
        kwargs_92111 = {}
        # Getting the type of 'assert_array_equal' (line 303)
        assert_array_equal_92108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 303)
        assert_array_equal_call_result_92112 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), assert_array_equal_92108, *[desired_y_92109, y_92110], **kwargs_92111)
        
        
        # ################# End of 'test_x_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 296)
        stypy_return_type_92113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92113)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_stride'
        return stypy_return_type_92113


    @norecursion
    def test_y_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_y_stride'
        module_type_store = module_type_store.open_function_context('test_y_stride', 305, 4, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseSwap.test_y_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseSwap.test_y_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseSwap.test_y_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseSwap.test_y_stride.__dict__.__setitem__('stypy_function_name', 'BaseSwap.test_y_stride')
        BaseSwap.test_y_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseSwap.test_y_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseSwap.test_y_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseSwap.test_y_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseSwap.test_y_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseSwap.test_y_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseSwap.test_y_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseSwap.test_y_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_y_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_y_stride(...)' code ##################

        
        # Assigning a Call to a Name (line 306):
        
        # Assigning a Call to a Name (line 306):
        
        # Call to arange(...): (line 306)
        # Processing the call arguments (line 306)
        float_92115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 19), 'float')
        # Processing the call keyword arguments (line 306)
        # Getting the type of 'self' (line 306)
        self_92116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 29), 'self', False)
        # Obtaining the member 'dtype' of a type (line 306)
        dtype_92117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 29), self_92116, 'dtype')
        keyword_92118 = dtype_92117
        kwargs_92119 = {'dtype': keyword_92118}
        # Getting the type of 'arange' (line 306)
        arange_92114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 306)
        arange_call_result_92120 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), arange_92114, *[float_92115], **kwargs_92119)
        
        # Assigning a type to the variable 'x' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'x', arange_call_result_92120)
        
        # Assigning a Call to a Name (line 307):
        
        # Assigning a Call to a Name (line 307):
        
        # Call to zeros(...): (line 307)
        # Processing the call arguments (line 307)
        int_92122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 18), 'int')
        # Getting the type of 'x' (line 307)
        x_92123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 307)
        dtype_92124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 21), x_92123, 'dtype')
        # Processing the call keyword arguments (line 307)
        kwargs_92125 = {}
        # Getting the type of 'zeros' (line 307)
        zeros_92121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 307)
        zeros_call_result_92126 = invoke(stypy.reporting.localization.Localization(__file__, 307, 12), zeros_92121, *[int_92122, dtype_92124], **kwargs_92125)
        
        # Assigning a type to the variable 'y' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'y', zeros_call_result_92126)
        
        # Assigning a Subscript to a Name (line 308):
        
        # Assigning a Subscript to a Name (line 308):
        
        # Obtaining the type of the subscript
        int_92127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 31), 'int')
        slice_92128 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 308, 20), None, None, int_92127)
        
        # Call to copy(...): (line 308)
        # Processing the call keyword arguments (line 308)
        kwargs_92131 = {}
        # Getting the type of 'y' (line 308)
        y_92129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 20), 'y', False)
        # Obtaining the member 'copy' of a type (line 308)
        copy_92130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 20), y_92129, 'copy')
        # Calling copy(args, kwargs) (line 308)
        copy_call_result_92132 = invoke(stypy.reporting.localization.Localization(__file__, 308, 20), copy_92130, *[], **kwargs_92131)
        
        # Obtaining the member '__getitem__' of a type (line 308)
        getitem___92133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 20), copy_call_result_92132, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 308)
        subscript_call_result_92134 = invoke(stypy.reporting.localization.Localization(__file__, 308, 20), getitem___92133, slice_92128)
        
        # Assigning a type to the variable 'desired_x' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'desired_x', subscript_call_result_92134)
        
        # Assigning a Call to a Name (line 309):
        
        # Assigning a Call to a Name (line 309):
        
        # Call to copy(...): (line 309)
        # Processing the call keyword arguments (line 309)
        kwargs_92137 = {}
        # Getting the type of 'x' (line 309)
        x_92135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 20), 'x', False)
        # Obtaining the member 'copy' of a type (line 309)
        copy_92136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 20), x_92135, 'copy')
        # Calling copy(args, kwargs) (line 309)
        copy_call_result_92138 = invoke(stypy.reporting.localization.Localization(__file__, 309, 20), copy_92136, *[], **kwargs_92137)
        
        # Assigning a type to the variable 'desired_y' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'desired_y', copy_call_result_92138)
        
        # Assigning a Call to a Tuple (line 310):
        
        # Assigning a Subscript to a Name (line 310):
        
        # Obtaining the type of the subscript
        int_92139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 8), 'int')
        
        # Call to blas_func(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'x' (line 310)
        x_92142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 30), 'x', False)
        # Getting the type of 'y' (line 310)
        y_92143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 33), 'y', False)
        # Processing the call keyword arguments (line 310)
        int_92144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 38), 'int')
        keyword_92145 = int_92144
        int_92146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 46), 'int')
        keyword_92147 = int_92146
        kwargs_92148 = {'incy': keyword_92147, 'n': keyword_92145}
        # Getting the type of 'self' (line 310)
        self_92140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 310)
        blas_func_92141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 15), self_92140, 'blas_func')
        # Calling blas_func(args, kwargs) (line 310)
        blas_func_call_result_92149 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), blas_func_92141, *[x_92142, y_92143], **kwargs_92148)
        
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___92150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), blas_func_call_result_92149, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_92151 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), getitem___92150, int_92139)
        
        # Assigning a type to the variable 'tuple_var_assignment_91206' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'tuple_var_assignment_91206', subscript_call_result_92151)
        
        # Assigning a Subscript to a Name (line 310):
        
        # Obtaining the type of the subscript
        int_92152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 8), 'int')
        
        # Call to blas_func(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'x' (line 310)
        x_92155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 30), 'x', False)
        # Getting the type of 'y' (line 310)
        y_92156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 33), 'y', False)
        # Processing the call keyword arguments (line 310)
        int_92157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 38), 'int')
        keyword_92158 = int_92157
        int_92159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 46), 'int')
        keyword_92160 = int_92159
        kwargs_92161 = {'incy': keyword_92160, 'n': keyword_92158}
        # Getting the type of 'self' (line 310)
        self_92153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 310)
        blas_func_92154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 15), self_92153, 'blas_func')
        # Calling blas_func(args, kwargs) (line 310)
        blas_func_call_result_92162 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), blas_func_92154, *[x_92155, y_92156], **kwargs_92161)
        
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___92163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), blas_func_call_result_92162, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_92164 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), getitem___92163, int_92152)
        
        # Assigning a type to the variable 'tuple_var_assignment_91207' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'tuple_var_assignment_91207', subscript_call_result_92164)
        
        # Assigning a Name to a Name (line 310):
        # Getting the type of 'tuple_var_assignment_91206' (line 310)
        tuple_var_assignment_91206_92165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'tuple_var_assignment_91206')
        # Assigning a type to the variable 'x' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'x', tuple_var_assignment_91206_92165)
        
        # Assigning a Name to a Name (line 310):
        # Getting the type of 'tuple_var_assignment_91207' (line 310)
        tuple_var_assignment_91207_92166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'tuple_var_assignment_91207')
        # Assigning a type to the variable 'y' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'y', tuple_var_assignment_91207_92166)
        
        # Call to assert_array_equal(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'desired_x' (line 311)
        desired_x_92168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 27), 'desired_x', False)
        # Getting the type of 'x' (line 311)
        x_92169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 38), 'x', False)
        # Processing the call keyword arguments (line 311)
        kwargs_92170 = {}
        # Getting the type of 'assert_array_equal' (line 311)
        assert_array_equal_92167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 311)
        assert_array_equal_call_result_92171 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), assert_array_equal_92167, *[desired_x_92168, x_92169], **kwargs_92170)
        
        
        # Call to assert_array_equal(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'desired_y' (line 312)
        desired_y_92173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 27), 'desired_y', False)
        
        # Obtaining the type of the subscript
        int_92174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 42), 'int')
        slice_92175 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 312, 38), None, None, int_92174)
        # Getting the type of 'y' (line 312)
        y_92176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 38), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___92177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 38), y_92176, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_92178 = invoke(stypy.reporting.localization.Localization(__file__, 312, 38), getitem___92177, slice_92175)
        
        # Processing the call keyword arguments (line 312)
        kwargs_92179 = {}
        # Getting the type of 'assert_array_equal' (line 312)
        assert_array_equal_92172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 312)
        assert_array_equal_call_result_92180 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), assert_array_equal_92172, *[desired_y_92173, subscript_call_result_92178], **kwargs_92179)
        
        
        # ################# End of 'test_y_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_y_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 305)
        stypy_return_type_92181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92181)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_y_stride'
        return stypy_return_type_92181


    @norecursion
    def test_x_and_y_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_and_y_stride'
        module_type_store = module_type_store.open_function_context('test_x_and_y_stride', 314, 4, False)
        # Assigning a type to the variable 'self' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseSwap.test_x_and_y_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseSwap.test_x_and_y_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseSwap.test_x_and_y_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseSwap.test_x_and_y_stride.__dict__.__setitem__('stypy_function_name', 'BaseSwap.test_x_and_y_stride')
        BaseSwap.test_x_and_y_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseSwap.test_x_and_y_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseSwap.test_x_and_y_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseSwap.test_x_and_y_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseSwap.test_x_and_y_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseSwap.test_x_and_y_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseSwap.test_x_and_y_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseSwap.test_x_and_y_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_and_y_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_and_y_stride(...)' code ##################

        
        # Assigning a Call to a Name (line 315):
        
        # Assigning a Call to a Name (line 315):
        
        # Call to arange(...): (line 315)
        # Processing the call arguments (line 315)
        float_92183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 19), 'float')
        # Processing the call keyword arguments (line 315)
        # Getting the type of 'self' (line 315)
        self_92184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 315)
        dtype_92185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 30), self_92184, 'dtype')
        keyword_92186 = dtype_92185
        kwargs_92187 = {'dtype': keyword_92186}
        # Getting the type of 'arange' (line 315)
        arange_92182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 315)
        arange_call_result_92188 = invoke(stypy.reporting.localization.Localization(__file__, 315, 12), arange_92182, *[float_92183], **kwargs_92187)
        
        # Assigning a type to the variable 'x' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'x', arange_call_result_92188)
        
        # Assigning a Call to a Name (line 316):
        
        # Assigning a Call to a Name (line 316):
        
        # Call to zeros(...): (line 316)
        # Processing the call arguments (line 316)
        int_92190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 18), 'int')
        # Getting the type of 'x' (line 316)
        x_92191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 316)
        dtype_92192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 21), x_92191, 'dtype')
        # Processing the call keyword arguments (line 316)
        kwargs_92193 = {}
        # Getting the type of 'zeros' (line 316)
        zeros_92189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 316)
        zeros_call_result_92194 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), zeros_92189, *[int_92190, dtype_92192], **kwargs_92193)
        
        # Assigning a type to the variable 'y' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'y', zeros_call_result_92194)
        
        # Assigning a Subscript to a Name (line 317):
        
        # Assigning a Subscript to a Name (line 317):
        
        # Obtaining the type of the subscript
        int_92195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 31), 'int')
        slice_92196 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 317, 20), None, None, int_92195)
        
        # Call to copy(...): (line 317)
        # Processing the call keyword arguments (line 317)
        kwargs_92199 = {}
        # Getting the type of 'y' (line 317)
        y_92197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'y', False)
        # Obtaining the member 'copy' of a type (line 317)
        copy_92198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 20), y_92197, 'copy')
        # Calling copy(args, kwargs) (line 317)
        copy_call_result_92200 = invoke(stypy.reporting.localization.Localization(__file__, 317, 20), copy_92198, *[], **kwargs_92199)
        
        # Obtaining the member '__getitem__' of a type (line 317)
        getitem___92201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 20), copy_call_result_92200, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 317)
        subscript_call_result_92202 = invoke(stypy.reporting.localization.Localization(__file__, 317, 20), getitem___92201, slice_92196)
        
        # Assigning a type to the variable 'desired_x' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'desired_x', subscript_call_result_92202)
        
        # Assigning a Subscript to a Name (line 318):
        
        # Assigning a Subscript to a Name (line 318):
        
        # Obtaining the type of the subscript
        int_92203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 31), 'int')
        slice_92204 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 318, 20), None, None, int_92203)
        
        # Call to copy(...): (line 318)
        # Processing the call keyword arguments (line 318)
        kwargs_92207 = {}
        # Getting the type of 'x' (line 318)
        x_92205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'x', False)
        # Obtaining the member 'copy' of a type (line 318)
        copy_92206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 20), x_92205, 'copy')
        # Calling copy(args, kwargs) (line 318)
        copy_call_result_92208 = invoke(stypy.reporting.localization.Localization(__file__, 318, 20), copy_92206, *[], **kwargs_92207)
        
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___92209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 20), copy_call_result_92208, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_92210 = invoke(stypy.reporting.localization.Localization(__file__, 318, 20), getitem___92209, slice_92204)
        
        # Assigning a type to the variable 'desired_y' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'desired_y', subscript_call_result_92210)
        
        # Assigning a Call to a Tuple (line 319):
        
        # Assigning a Subscript to a Name (line 319):
        
        # Obtaining the type of the subscript
        int_92211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 8), 'int')
        
        # Call to blas_func(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'x' (line 319)
        x_92214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 30), 'x', False)
        # Getting the type of 'y' (line 319)
        y_92215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 33), 'y', False)
        # Processing the call keyword arguments (line 319)
        int_92216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 38), 'int')
        keyword_92217 = int_92216
        int_92218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 46), 'int')
        keyword_92219 = int_92218
        int_92220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 54), 'int')
        keyword_92221 = int_92220
        kwargs_92222 = {'incx': keyword_92219, 'incy': keyword_92221, 'n': keyword_92217}
        # Getting the type of 'self' (line 319)
        self_92212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 15), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 319)
        blas_func_92213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 15), self_92212, 'blas_func')
        # Calling blas_func(args, kwargs) (line 319)
        blas_func_call_result_92223 = invoke(stypy.reporting.localization.Localization(__file__, 319, 15), blas_func_92213, *[x_92214, y_92215], **kwargs_92222)
        
        # Obtaining the member '__getitem__' of a type (line 319)
        getitem___92224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), blas_func_call_result_92223, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 319)
        subscript_call_result_92225 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), getitem___92224, int_92211)
        
        # Assigning a type to the variable 'tuple_var_assignment_91208' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'tuple_var_assignment_91208', subscript_call_result_92225)
        
        # Assigning a Subscript to a Name (line 319):
        
        # Obtaining the type of the subscript
        int_92226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 8), 'int')
        
        # Call to blas_func(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'x' (line 319)
        x_92229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 30), 'x', False)
        # Getting the type of 'y' (line 319)
        y_92230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 33), 'y', False)
        # Processing the call keyword arguments (line 319)
        int_92231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 38), 'int')
        keyword_92232 = int_92231
        int_92233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 46), 'int')
        keyword_92234 = int_92233
        int_92235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 54), 'int')
        keyword_92236 = int_92235
        kwargs_92237 = {'incx': keyword_92234, 'incy': keyword_92236, 'n': keyword_92232}
        # Getting the type of 'self' (line 319)
        self_92227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 15), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 319)
        blas_func_92228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 15), self_92227, 'blas_func')
        # Calling blas_func(args, kwargs) (line 319)
        blas_func_call_result_92238 = invoke(stypy.reporting.localization.Localization(__file__, 319, 15), blas_func_92228, *[x_92229, y_92230], **kwargs_92237)
        
        # Obtaining the member '__getitem__' of a type (line 319)
        getitem___92239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), blas_func_call_result_92238, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 319)
        subscript_call_result_92240 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), getitem___92239, int_92226)
        
        # Assigning a type to the variable 'tuple_var_assignment_91209' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'tuple_var_assignment_91209', subscript_call_result_92240)
        
        # Assigning a Name to a Name (line 319):
        # Getting the type of 'tuple_var_assignment_91208' (line 319)
        tuple_var_assignment_91208_92241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'tuple_var_assignment_91208')
        # Assigning a type to the variable 'x' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'x', tuple_var_assignment_91208_92241)
        
        # Assigning a Name to a Name (line 319):
        # Getting the type of 'tuple_var_assignment_91209' (line 319)
        tuple_var_assignment_91209_92242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'tuple_var_assignment_91209')
        # Assigning a type to the variable 'y' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 11), 'y', tuple_var_assignment_91209_92242)
        
        # Call to assert_array_equal(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'desired_x' (line 320)
        desired_x_92244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 27), 'desired_x', False)
        
        # Obtaining the type of the subscript
        int_92245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 42), 'int')
        slice_92246 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 320, 38), None, None, int_92245)
        # Getting the type of 'x' (line 320)
        x_92247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 38), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___92248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 38), x_92247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_92249 = invoke(stypy.reporting.localization.Localization(__file__, 320, 38), getitem___92248, slice_92246)
        
        # Processing the call keyword arguments (line 320)
        kwargs_92250 = {}
        # Getting the type of 'assert_array_equal' (line 320)
        assert_array_equal_92243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 320)
        assert_array_equal_call_result_92251 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), assert_array_equal_92243, *[desired_x_92244, subscript_call_result_92249], **kwargs_92250)
        
        
        # Call to assert_array_equal(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'desired_y' (line 321)
        desired_y_92253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 27), 'desired_y', False)
        
        # Obtaining the type of the subscript
        int_92254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 42), 'int')
        slice_92255 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 321, 38), None, None, int_92254)
        # Getting the type of 'y' (line 321)
        y_92256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 38), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 321)
        getitem___92257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 38), y_92256, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 321)
        subscript_call_result_92258 = invoke(stypy.reporting.localization.Localization(__file__, 321, 38), getitem___92257, slice_92255)
        
        # Processing the call keyword arguments (line 321)
        kwargs_92259 = {}
        # Getting the type of 'assert_array_equal' (line 321)
        assert_array_equal_92252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 321)
        assert_array_equal_call_result_92260 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), assert_array_equal_92252, *[desired_y_92253, subscript_call_result_92258], **kwargs_92259)
        
        
        # ################# End of 'test_x_and_y_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_and_y_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 314)
        stypy_return_type_92261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_and_y_stride'
        return stypy_return_type_92261


    @norecursion
    def test_x_bad_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_bad_size'
        module_type_store = module_type_store.open_function_context('test_x_bad_size', 323, 4, False)
        # Assigning a type to the variable 'self' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseSwap.test_x_bad_size.__dict__.__setitem__('stypy_localization', localization)
        BaseSwap.test_x_bad_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseSwap.test_x_bad_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseSwap.test_x_bad_size.__dict__.__setitem__('stypy_function_name', 'BaseSwap.test_x_bad_size')
        BaseSwap.test_x_bad_size.__dict__.__setitem__('stypy_param_names_list', [])
        BaseSwap.test_x_bad_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseSwap.test_x_bad_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseSwap.test_x_bad_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseSwap.test_x_bad_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseSwap.test_x_bad_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseSwap.test_x_bad_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseSwap.test_x_bad_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_bad_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_bad_size(...)' code ##################

        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to arange(...): (line 324)
        # Processing the call arguments (line 324)
        float_92263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 19), 'float')
        # Processing the call keyword arguments (line 324)
        # Getting the type of 'self' (line 324)
        self_92264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 324)
        dtype_92265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 30), self_92264, 'dtype')
        keyword_92266 = dtype_92265
        kwargs_92267 = {'dtype': keyword_92266}
        # Getting the type of 'arange' (line 324)
        arange_92262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 324)
        arange_call_result_92268 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), arange_92262, *[float_92263], **kwargs_92267)
        
        # Assigning a type to the variable 'x' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'x', arange_call_result_92268)
        
        # Assigning a Call to a Name (line 325):
        
        # Assigning a Call to a Name (line 325):
        
        # Call to zeros(...): (line 325)
        # Processing the call arguments (line 325)
        int_92270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 18), 'int')
        # Getting the type of 'x' (line 325)
        x_92271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 325)
        dtype_92272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 21), x_92271, 'dtype')
        # Processing the call keyword arguments (line 325)
        kwargs_92273 = {}
        # Getting the type of 'zeros' (line 325)
        zeros_92269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 325)
        zeros_call_result_92274 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), zeros_92269, *[int_92270, dtype_92272], **kwargs_92273)
        
        # Assigning a type to the variable 'y' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'y', zeros_call_result_92274)
        
        
        # SSA begins for try-except statement (line 326)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to blas_func(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'x' (line 327)
        x_92277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 27), 'x', False)
        # Getting the type of 'y' (line 327)
        y_92278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 30), 'y', False)
        # Processing the call keyword arguments (line 327)
        int_92279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 35), 'int')
        keyword_92280 = int_92279
        int_92281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 43), 'int')
        keyword_92282 = int_92281
        kwargs_92283 = {'incx': keyword_92282, 'n': keyword_92280}
        # Getting the type of 'self' (line 327)
        self_92275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 327)
        blas_func_92276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 12), self_92275, 'blas_func')
        # Calling blas_func(args, kwargs) (line 327)
        blas_func_call_result_92284 = invoke(stypy.reporting.localization.Localization(__file__, 327, 12), blas_func_92276, *[x_92277, y_92278], **kwargs_92283)
        
        # SSA branch for the except part of a try statement (line 326)
        # SSA branch for the except '<any exception>' branch of a try statement (line 326)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 326)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 331)
        # Processing the call arguments (line 331)
        int_92286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 16), 'int')
        # Processing the call keyword arguments (line 331)
        kwargs_92287 = {}
        # Getting the type of 'assert_' (line 331)
        assert__92285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 331)
        assert__call_result_92288 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), assert__92285, *[int_92286], **kwargs_92287)
        
        
        # ################# End of 'test_x_bad_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_bad_size' in the type store
        # Getting the type of 'stypy_return_type' (line 323)
        stypy_return_type_92289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92289)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_bad_size'
        return stypy_return_type_92289


    @norecursion
    def test_y_bad_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_y_bad_size'
        module_type_store = module_type_store.open_function_context('test_y_bad_size', 333, 4, False)
        # Assigning a type to the variable 'self' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseSwap.test_y_bad_size.__dict__.__setitem__('stypy_localization', localization)
        BaseSwap.test_y_bad_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseSwap.test_y_bad_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseSwap.test_y_bad_size.__dict__.__setitem__('stypy_function_name', 'BaseSwap.test_y_bad_size')
        BaseSwap.test_y_bad_size.__dict__.__setitem__('stypy_param_names_list', [])
        BaseSwap.test_y_bad_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseSwap.test_y_bad_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseSwap.test_y_bad_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseSwap.test_y_bad_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseSwap.test_y_bad_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseSwap.test_y_bad_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseSwap.test_y_bad_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_y_bad_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_y_bad_size(...)' code ##################

        
        # Assigning a Call to a Name (line 334):
        
        # Assigning a Call to a Name (line 334):
        
        # Call to arange(...): (line 334)
        # Processing the call arguments (line 334)
        float_92291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 19), 'float')
        # Processing the call keyword arguments (line 334)
        # Getting the type of 'self' (line 334)
        self_92292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 334)
        dtype_92293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 30), self_92292, 'dtype')
        keyword_92294 = dtype_92293
        kwargs_92295 = {'dtype': keyword_92294}
        # Getting the type of 'arange' (line 334)
        arange_92290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 334)
        arange_call_result_92296 = invoke(stypy.reporting.localization.Localization(__file__, 334, 12), arange_92290, *[float_92291], **kwargs_92295)
        
        # Assigning a type to the variable 'x' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'x', arange_call_result_92296)
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to zeros(...): (line 335)
        # Processing the call arguments (line 335)
        int_92298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 18), 'int')
        # Getting the type of 'x' (line 335)
        x_92299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 335)
        dtype_92300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 21), x_92299, 'dtype')
        # Processing the call keyword arguments (line 335)
        kwargs_92301 = {}
        # Getting the type of 'zeros' (line 335)
        zeros_92297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'zeros', False)
        # Calling zeros(args, kwargs) (line 335)
        zeros_call_result_92302 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), zeros_92297, *[int_92298, dtype_92300], **kwargs_92301)
        
        # Assigning a type to the variable 'y' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'y', zeros_call_result_92302)
        
        
        # SSA begins for try-except statement (line 336)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to blas_func(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'x' (line 337)
        x_92305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 27), 'x', False)
        # Getting the type of 'y' (line 337)
        y_92306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 30), 'y', False)
        # Processing the call keyword arguments (line 337)
        int_92307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 35), 'int')
        keyword_92308 = int_92307
        int_92309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 43), 'int')
        keyword_92310 = int_92309
        kwargs_92311 = {'incy': keyword_92310, 'n': keyword_92308}
        # Getting the type of 'self' (line 337)
        self_92303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 337)
        blas_func_92304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), self_92303, 'blas_func')
        # Calling blas_func(args, kwargs) (line 337)
        blas_func_call_result_92312 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), blas_func_92304, *[x_92305, y_92306], **kwargs_92311)
        
        # SSA branch for the except part of a try statement (line 336)
        # SSA branch for the except '<any exception>' branch of a try statement (line 336)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 336)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 341)
        # Processing the call arguments (line 341)
        int_92314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 16), 'int')
        # Processing the call keyword arguments (line 341)
        kwargs_92315 = {}
        # Getting the type of 'assert_' (line 341)
        assert__92313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 341)
        assert__call_result_92316 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), assert__92313, *[int_92314], **kwargs_92315)
        
        
        # ################# End of 'test_y_bad_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_y_bad_size' in the type store
        # Getting the type of 'stypy_return_type' (line 333)
        stypy_return_type_92317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92317)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_y_bad_size'
        return stypy_return_type_92317


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseSwap.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BaseSwap' (line 284)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'BaseSwap', BaseSwap)


# SSA begins for try-except statement (line 344)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Declaration of the 'TestSswap' class
# Getting the type of 'BaseSwap' (line 345)
BaseSwap_92318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 20), 'BaseSwap')

class TestSswap(BaseSwap_92318, ):
    
    # Assigning a Attribute to a Name (line 346):
    
    # Assigning a Name to a Name (line 347):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 345, 4, False)
        # Assigning a type to the variable 'self' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSswap.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSswap' (line 345)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'TestSswap', TestSswap)

# Assigning a Attribute to a Name (line 346):
# Getting the type of 'fblas' (line 346)
fblas_92319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 20), 'fblas')
# Obtaining the member 'sswap' of a type (line 346)
sswap_92320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 20), fblas_92319, 'sswap')
# Getting the type of 'TestSswap'
TestSswap_92321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestSswap')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestSswap_92321, 'blas_func', sswap_92320)

# Assigning a Name to a Name (line 347):
# Getting the type of 'float32' (line 347)
float32_92322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'float32')
# Getting the type of 'TestSswap'
TestSswap_92323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestSswap')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestSswap_92323, 'dtype', float32_92322)
# SSA branch for the except part of a try statement (line 344)
# SSA branch for the except 'AttributeError' branch of a try statement (line 344)
module_type_store.open_ssa_branch('except')
# Declaration of the 'TestSswap' class

class TestSswap:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 349, 4, False)
        # Assigning a type to the variable 'self' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSswap.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSswap' (line 349)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'TestSswap', TestSswap)
# SSA join for try-except statement (line 344)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TestDswap' class
# Getting the type of 'BaseSwap' (line 353)
BaseSwap_92324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'BaseSwap')

class TestDswap(BaseSwap_92324, ):
    
    # Assigning a Attribute to a Name (line 354):
    
    # Assigning a Name to a Name (line 355):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 353, 0, False)
        # Assigning a type to the variable 'self' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDswap.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDswap' (line 353)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), 'TestDswap', TestDswap)

# Assigning a Attribute to a Name (line 354):
# Getting the type of 'fblas' (line 354)
fblas_92325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'fblas')
# Obtaining the member 'dswap' of a type (line 354)
dswap_92326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 16), fblas_92325, 'dswap')
# Getting the type of 'TestDswap'
TestDswap_92327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDswap')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDswap_92327, 'blas_func', dswap_92326)

# Assigning a Name to a Name (line 355):
# Getting the type of 'float64' (line 355)
float64_92328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'float64')
# Getting the type of 'TestDswap'
TestDswap_92329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDswap')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDswap_92329, 'dtype', float64_92328)


# SSA begins for try-except statement (line 358)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Declaration of the 'TestCswap' class
# Getting the type of 'BaseSwap' (line 359)
BaseSwap_92330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 20), 'BaseSwap')

class TestCswap(BaseSwap_92330, ):
    
    # Assigning a Attribute to a Name (line 360):
    
    # Assigning a Name to a Name (line 361):
    pass

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCswap.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCswap' (line 359)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'TestCswap', TestCswap)

# Assigning a Attribute to a Name (line 360):
# Getting the type of 'fblas' (line 360)
fblas_92331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'fblas')
# Obtaining the member 'cswap' of a type (line 360)
cswap_92332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 20), fblas_92331, 'cswap')
# Getting the type of 'TestCswap'
TestCswap_92333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCswap')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCswap_92333, 'blas_func', cswap_92332)

# Assigning a Name to a Name (line 361):
# Getting the type of 'complex64' (line 361)
complex64_92334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'complex64')
# Getting the type of 'TestCswap'
TestCswap_92335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCswap')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCswap_92335, 'dtype', complex64_92334)
# SSA branch for the except part of a try statement (line 358)
# SSA branch for the except 'AttributeError' branch of a try statement (line 358)
module_type_store.open_ssa_branch('except')
# Declaration of the 'TestCswap' class

class TestCswap:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 363, 4, False)
        # Assigning a type to the variable 'self' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCswap.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCswap' (line 363)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'TestCswap', TestCswap)
# SSA join for try-except statement (line 358)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TestZswap' class
# Getting the type of 'BaseSwap' (line 367)
BaseSwap_92336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'BaseSwap')

class TestZswap(BaseSwap_92336, ):
    
    # Assigning a Attribute to a Name (line 368):
    
    # Assigning a Name to a Name (line 369):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 367, 0, False)
        # Assigning a type to the variable 'self' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZswap.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestZswap' (line 367)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 0), 'TestZswap', TestZswap)

# Assigning a Attribute to a Name (line 368):
# Getting the type of 'fblas' (line 368)
fblas_92337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'fblas')
# Obtaining the member 'zswap' of a type (line 368)
zswap_92338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 16), fblas_92337, 'zswap')
# Getting the type of 'TestZswap'
TestZswap_92339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZswap')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZswap_92339, 'blas_func', zswap_92338)

# Assigning a Name to a Name (line 369):
# Getting the type of 'complex128' (line 369)
complex128_92340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'complex128')
# Getting the type of 'TestZswap'
TestZswap_92341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZswap')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZswap_92341, 'dtype', complex128_92340)
# Declaration of the 'BaseGemv' class

class BaseGemv(object, ):
    str_92342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 4), 'str', ' Mixin class for gemv tests ')

    @norecursion
    def get_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_92343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 32), 'int')
        int_92344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 44), 'int')
        defaults = [int_92343, int_92344]
        # Create a new context for function 'get_data'
        module_type_store = module_type_store.open_function_context('get_data', 379, 4, False)
        # Assigning a type to the variable 'self' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseGemv.get_data.__dict__.__setitem__('stypy_localization', localization)
        BaseGemv.get_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseGemv.get_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseGemv.get_data.__dict__.__setitem__('stypy_function_name', 'BaseGemv.get_data')
        BaseGemv.get_data.__dict__.__setitem__('stypy_param_names_list', ['x_stride', 'y_stride'])
        BaseGemv.get_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseGemv.get_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseGemv.get_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseGemv.get_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseGemv.get_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseGemv.get_data.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.get_data', ['x_stride', 'y_stride'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_data', localization, ['x_stride', 'y_stride'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_data(...)' code ##################

        
        # Assigning a Call to a Name (line 380):
        
        # Assigning a Call to a Name (line 380):
        
        # Call to array(...): (line 380)
        # Processing the call arguments (line 380)
        int_92346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 21), 'int')
        # Processing the call keyword arguments (line 380)
        # Getting the type of 'self' (line 380)
        self_92347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 380)
        dtype_92348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 30), self_92347, 'dtype')
        keyword_92349 = dtype_92348
        kwargs_92350 = {'dtype': keyword_92349}
        # Getting the type of 'array' (line 380)
        array_92345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'array', False)
        # Calling array(args, kwargs) (line 380)
        array_call_result_92351 = invoke(stypy.reporting.localization.Localization(__file__, 380, 15), array_92345, *[int_92346], **kwargs_92350)
        
        # Assigning a type to the variable 'mult' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'mult', array_call_result_92351)
        
        
        # Getting the type of 'self' (line 381)
        self_92352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 11), 'self')
        # Obtaining the member 'dtype' of a type (line 381)
        dtype_92353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 11), self_92352, 'dtype')
        
        # Obtaining an instance of the builtin type 'list' (line 381)
        list_92354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 381)
        # Adding element type (line 381)
        # Getting the type of 'complex64' (line 381)
        complex64_92355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 26), 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 25), list_92354, complex64_92355)
        # Adding element type (line 381)
        # Getting the type of 'complex128' (line 381)
        complex128_92356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 37), 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 25), list_92354, complex128_92356)
        
        # Applying the binary operator 'in' (line 381)
        result_contains_92357 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 11), 'in', dtype_92353, list_92354)
        
        # Testing the type of an if condition (line 381)
        if_condition_92358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 8), result_contains_92357)
        # Assigning a type to the variable 'if_condition_92358' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'if_condition_92358', if_condition_92358)
        # SSA begins for if statement (line 381)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 382):
        
        # Assigning a Call to a Name (line 382):
        
        # Call to array(...): (line 382)
        # Processing the call arguments (line 382)
        int_92360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 25), 'int')
        complex_92361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 27), 'complex')
        # Applying the binary operator '+' (line 382)
        result_add_92362 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 25), '+', int_92360, complex_92361)
        
        # Processing the call keyword arguments (line 382)
        # Getting the type of 'self' (line 382)
        self_92363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 37), 'self', False)
        # Obtaining the member 'dtype' of a type (line 382)
        dtype_92364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 37), self_92363, 'dtype')
        keyword_92365 = dtype_92364
        kwargs_92366 = {'dtype': keyword_92365}
        # Getting the type of 'array' (line 382)
        array_92359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 19), 'array', False)
        # Calling array(args, kwargs) (line 382)
        array_call_result_92367 = invoke(stypy.reporting.localization.Localization(__file__, 382, 19), array_92359, *[result_add_92362], **kwargs_92366)
        
        # Assigning a type to the variable 'mult' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'mult', array_call_result_92367)
        # SSA join for if statement (line 381)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 383, 8))
        
        # 'from numpy.random import normal, seed' statement (line 383)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
        import_92368 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'numpy.random')

        if (type(import_92368) is not StypyTypeError):

            if (import_92368 != 'pyd_module'):
                __import__(import_92368)
                sys_modules_92369 = sys.modules[import_92368]
                import_from_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'numpy.random', sys_modules_92369.module_type_store, module_type_store, ['normal', 'seed'])
                nest_module(stypy.reporting.localization.Localization(__file__, 383, 8), __file__, sys_modules_92369, sys_modules_92369.module_type_store, module_type_store)
            else:
                from numpy.random import normal, seed

                import_from_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'numpy.random', None, module_type_store, ['normal', 'seed'], [normal, seed])

        else:
            # Assigning a type to the variable 'numpy.random' (line 383)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'numpy.random', import_92368)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')
        
        
        # Call to seed(...): (line 384)
        # Processing the call arguments (line 384)
        int_92371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 13), 'int')
        # Processing the call keyword arguments (line 384)
        kwargs_92372 = {}
        # Getting the type of 'seed' (line 384)
        seed_92370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'seed', False)
        # Calling seed(args, kwargs) (line 384)
        seed_call_result_92373 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), seed_92370, *[int_92371], **kwargs_92372)
        
        
        # Assigning a BinOp to a Name (line 385):
        
        # Assigning a BinOp to a Name (line 385):
        
        # Call to array(...): (line 385)
        # Processing the call arguments (line 385)
        float_92375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 22), 'float')
        # Processing the call keyword arguments (line 385)
        # Getting the type of 'self' (line 385)
        self_92376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 32), 'self', False)
        # Obtaining the member 'dtype' of a type (line 385)
        dtype_92377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 32), self_92376, 'dtype')
        keyword_92378 = dtype_92377
        kwargs_92379 = {'dtype': keyword_92378}
        # Getting the type of 'array' (line 385)
        array_92374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'array', False)
        # Calling array(args, kwargs) (line 385)
        array_call_result_92380 = invoke(stypy.reporting.localization.Localization(__file__, 385, 16), array_92374, *[float_92375], **kwargs_92379)
        
        # Getting the type of 'mult' (line 385)
        mult_92381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 46), 'mult')
        # Applying the binary operator '*' (line 385)
        result_mul_92382 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 16), '*', array_call_result_92380, mult_92381)
        
        # Assigning a type to the variable 'alpha' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'alpha', result_mul_92382)
        
        # Assigning a BinOp to a Name (line 386):
        
        # Assigning a BinOp to a Name (line 386):
        
        # Call to array(...): (line 386)
        # Processing the call arguments (line 386)
        float_92384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 21), 'float')
        # Processing the call keyword arguments (line 386)
        # Getting the type of 'self' (line 386)
        self_92385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 31), 'self', False)
        # Obtaining the member 'dtype' of a type (line 386)
        dtype_92386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 31), self_92385, 'dtype')
        keyword_92387 = dtype_92386
        kwargs_92388 = {'dtype': keyword_92387}
        # Getting the type of 'array' (line 386)
        array_92383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'array', False)
        # Calling array(args, kwargs) (line 386)
        array_call_result_92389 = invoke(stypy.reporting.localization.Localization(__file__, 386, 15), array_92383, *[float_92384], **kwargs_92388)
        
        # Getting the type of 'mult' (line 386)
        mult_92390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 45), 'mult')
        # Applying the binary operator '*' (line 386)
        result_mul_92391 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 15), '*', array_call_result_92389, mult_92390)
        
        # Assigning a type to the variable 'beta' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'beta', result_mul_92391)
        
        # Assigning a BinOp to a Name (line 387):
        
        # Assigning a BinOp to a Name (line 387):
        
        # Call to astype(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'self' (line 387)
        self_92401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 42), 'self', False)
        # Obtaining the member 'dtype' of a type (line 387)
        dtype_92402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 42), self_92401, 'dtype')
        # Processing the call keyword arguments (line 387)
        kwargs_92403 = {}
        
        # Call to normal(...): (line 387)
        # Processing the call arguments (line 387)
        float_92393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 19), 'float')
        float_92394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 23), 'float')
        
        # Obtaining an instance of the builtin type 'tuple' (line 387)
        tuple_92395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 387)
        # Adding element type (line 387)
        int_92396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 28), tuple_92395, int_92396)
        # Adding element type (line 387)
        int_92397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 28), tuple_92395, int_92397)
        
        # Processing the call keyword arguments (line 387)
        kwargs_92398 = {}
        # Getting the type of 'normal' (line 387)
        normal_92392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'normal', False)
        # Calling normal(args, kwargs) (line 387)
        normal_call_result_92399 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), normal_92392, *[float_92393, float_92394, tuple_92395], **kwargs_92398)
        
        # Obtaining the member 'astype' of a type (line 387)
        astype_92400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), normal_call_result_92399, 'astype')
        # Calling astype(args, kwargs) (line 387)
        astype_call_result_92404 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), astype_92400, *[dtype_92402], **kwargs_92403)
        
        # Getting the type of 'mult' (line 387)
        mult_92405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 56), 'mult')
        # Applying the binary operator '*' (line 387)
        result_mul_92406 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 12), '*', astype_call_result_92404, mult_92405)
        
        # Assigning a type to the variable 'a' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'a', result_mul_92406)
        
        # Assigning a BinOp to a Name (line 388):
        
        # Assigning a BinOp to a Name (line 388):
        
        # Call to arange(...): (line 388)
        # Processing the call arguments (line 388)
        
        # Obtaining the type of the subscript
        int_92408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 28), 'int')
        
        # Call to shape(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'a' (line 388)
        a_92410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 25), 'a', False)
        # Processing the call keyword arguments (line 388)
        kwargs_92411 = {}
        # Getting the type of 'shape' (line 388)
        shape_92409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 19), 'shape', False)
        # Calling shape(args, kwargs) (line 388)
        shape_call_result_92412 = invoke(stypy.reporting.localization.Localization(__file__, 388, 19), shape_92409, *[a_92410], **kwargs_92411)
        
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___92413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 19), shape_call_result_92412, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 388)
        subscript_call_result_92414 = invoke(stypy.reporting.localization.Localization(__file__, 388, 19), getitem___92413, int_92408)
        
        # Getting the type of 'x_stride' (line 388)
        x_stride_92415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 31), 'x_stride', False)
        # Applying the binary operator '*' (line 388)
        result_mul_92416 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 19), '*', subscript_call_result_92414, x_stride_92415)
        
        # Processing the call keyword arguments (line 388)
        # Getting the type of 'self' (line 388)
        self_92417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 47), 'self', False)
        # Obtaining the member 'dtype' of a type (line 388)
        dtype_92418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 47), self_92417, 'dtype')
        keyword_92419 = dtype_92418
        kwargs_92420 = {'dtype': keyword_92419}
        # Getting the type of 'arange' (line 388)
        arange_92407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 388)
        arange_call_result_92421 = invoke(stypy.reporting.localization.Localization(__file__, 388, 12), arange_92407, *[result_mul_92416], **kwargs_92420)
        
        # Getting the type of 'mult' (line 388)
        mult_92422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 61), 'mult')
        # Applying the binary operator '*' (line 388)
        result_mul_92423 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 12), '*', arange_call_result_92421, mult_92422)
        
        # Assigning a type to the variable 'x' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'x', result_mul_92423)
        
        # Assigning a BinOp to a Name (line 389):
        
        # Assigning a BinOp to a Name (line 389):
        
        # Call to arange(...): (line 389)
        # Processing the call arguments (line 389)
        
        # Obtaining the type of the subscript
        int_92425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 28), 'int')
        
        # Call to shape(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'a' (line 389)
        a_92427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 25), 'a', False)
        # Processing the call keyword arguments (line 389)
        kwargs_92428 = {}
        # Getting the type of 'shape' (line 389)
        shape_92426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'shape', False)
        # Calling shape(args, kwargs) (line 389)
        shape_call_result_92429 = invoke(stypy.reporting.localization.Localization(__file__, 389, 19), shape_92426, *[a_92427], **kwargs_92428)
        
        # Obtaining the member '__getitem__' of a type (line 389)
        getitem___92430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 19), shape_call_result_92429, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 389)
        subscript_call_result_92431 = invoke(stypy.reporting.localization.Localization(__file__, 389, 19), getitem___92430, int_92425)
        
        # Getting the type of 'y_stride' (line 389)
        y_stride_92432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 31), 'y_stride', False)
        # Applying the binary operator '*' (line 389)
        result_mul_92433 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 19), '*', subscript_call_result_92431, y_stride_92432)
        
        # Processing the call keyword arguments (line 389)
        # Getting the type of 'self' (line 389)
        self_92434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 47), 'self', False)
        # Obtaining the member 'dtype' of a type (line 389)
        dtype_92435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 47), self_92434, 'dtype')
        keyword_92436 = dtype_92435
        kwargs_92437 = {'dtype': keyword_92436}
        # Getting the type of 'arange' (line 389)
        arange_92424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 389)
        arange_call_result_92438 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), arange_92424, *[result_mul_92433], **kwargs_92437)
        
        # Getting the type of 'mult' (line 389)
        mult_92439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 61), 'mult')
        # Applying the binary operator '*' (line 389)
        result_mul_92440 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 12), '*', arange_call_result_92438, mult_92439)
        
        # Assigning a type to the variable 'y' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'y', result_mul_92440)
        
        # Obtaining an instance of the builtin type 'tuple' (line 390)
        tuple_92441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 390)
        # Adding element type (line 390)
        # Getting the type of 'alpha' (line 390)
        alpha_92442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'alpha')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 15), tuple_92441, alpha_92442)
        # Adding element type (line 390)
        # Getting the type of 'beta' (line 390)
        beta_92443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 22), 'beta')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 15), tuple_92441, beta_92443)
        # Adding element type (line 390)
        # Getting the type of 'a' (line 390)
        a_92444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 28), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 15), tuple_92441, a_92444)
        # Adding element type (line 390)
        # Getting the type of 'x' (line 390)
        x_92445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 31), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 15), tuple_92441, x_92445)
        # Adding element type (line 390)
        # Getting the type of 'y' (line 390)
        y_92446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 34), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 15), tuple_92441, y_92446)
        
        # Assigning a type to the variable 'stypy_return_type' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'stypy_return_type', tuple_92441)
        
        # ################# End of 'get_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_data' in the type store
        # Getting the type of 'stypy_return_type' (line 379)
        stypy_return_type_92447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92447)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_data'
        return stypy_return_type_92447


    @norecursion
    def test_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple'
        module_type_store = module_type_store.open_function_context('test_simple', 392, 4, False)
        # Assigning a type to the variable 'self' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseGemv.test_simple.__dict__.__setitem__('stypy_localization', localization)
        BaseGemv.test_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseGemv.test_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseGemv.test_simple.__dict__.__setitem__('stypy_function_name', 'BaseGemv.test_simple')
        BaseGemv.test_simple.__dict__.__setitem__('stypy_param_names_list', [])
        BaseGemv.test_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseGemv.test_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseGemv.test_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseGemv.test_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseGemv.test_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseGemv.test_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.test_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple(...)' code ##################

        
        # Assigning a Call to a Tuple (line 393):
        
        # Assigning a Subscript to a Name (line 393):
        
        # Obtaining the type of the subscript
        int_92448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 8), 'int')
        
        # Call to get_data(...): (line 393)
        # Processing the call keyword arguments (line 393)
        kwargs_92451 = {}
        # Getting the type of 'self' (line 393)
        self_92449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 393)
        get_data_92450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 31), self_92449, 'get_data')
        # Calling get_data(args, kwargs) (line 393)
        get_data_call_result_92452 = invoke(stypy.reporting.localization.Localization(__file__, 393, 31), get_data_92450, *[], **kwargs_92451)
        
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___92453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), get_data_call_result_92452, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_92454 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), getitem___92453, int_92448)
        
        # Assigning a type to the variable 'tuple_var_assignment_91210' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_91210', subscript_call_result_92454)
        
        # Assigning a Subscript to a Name (line 393):
        
        # Obtaining the type of the subscript
        int_92455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 8), 'int')
        
        # Call to get_data(...): (line 393)
        # Processing the call keyword arguments (line 393)
        kwargs_92458 = {}
        # Getting the type of 'self' (line 393)
        self_92456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 393)
        get_data_92457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 31), self_92456, 'get_data')
        # Calling get_data(args, kwargs) (line 393)
        get_data_call_result_92459 = invoke(stypy.reporting.localization.Localization(__file__, 393, 31), get_data_92457, *[], **kwargs_92458)
        
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___92460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), get_data_call_result_92459, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_92461 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), getitem___92460, int_92455)
        
        # Assigning a type to the variable 'tuple_var_assignment_91211' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_91211', subscript_call_result_92461)
        
        # Assigning a Subscript to a Name (line 393):
        
        # Obtaining the type of the subscript
        int_92462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 8), 'int')
        
        # Call to get_data(...): (line 393)
        # Processing the call keyword arguments (line 393)
        kwargs_92465 = {}
        # Getting the type of 'self' (line 393)
        self_92463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 393)
        get_data_92464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 31), self_92463, 'get_data')
        # Calling get_data(args, kwargs) (line 393)
        get_data_call_result_92466 = invoke(stypy.reporting.localization.Localization(__file__, 393, 31), get_data_92464, *[], **kwargs_92465)
        
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___92467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), get_data_call_result_92466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_92468 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), getitem___92467, int_92462)
        
        # Assigning a type to the variable 'tuple_var_assignment_91212' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_91212', subscript_call_result_92468)
        
        # Assigning a Subscript to a Name (line 393):
        
        # Obtaining the type of the subscript
        int_92469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 8), 'int')
        
        # Call to get_data(...): (line 393)
        # Processing the call keyword arguments (line 393)
        kwargs_92472 = {}
        # Getting the type of 'self' (line 393)
        self_92470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 393)
        get_data_92471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 31), self_92470, 'get_data')
        # Calling get_data(args, kwargs) (line 393)
        get_data_call_result_92473 = invoke(stypy.reporting.localization.Localization(__file__, 393, 31), get_data_92471, *[], **kwargs_92472)
        
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___92474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), get_data_call_result_92473, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_92475 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), getitem___92474, int_92469)
        
        # Assigning a type to the variable 'tuple_var_assignment_91213' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_91213', subscript_call_result_92475)
        
        # Assigning a Subscript to a Name (line 393):
        
        # Obtaining the type of the subscript
        int_92476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 8), 'int')
        
        # Call to get_data(...): (line 393)
        # Processing the call keyword arguments (line 393)
        kwargs_92479 = {}
        # Getting the type of 'self' (line 393)
        self_92477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 393)
        get_data_92478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 31), self_92477, 'get_data')
        # Calling get_data(args, kwargs) (line 393)
        get_data_call_result_92480 = invoke(stypy.reporting.localization.Localization(__file__, 393, 31), get_data_92478, *[], **kwargs_92479)
        
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___92481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), get_data_call_result_92480, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_92482 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), getitem___92481, int_92476)
        
        # Assigning a type to the variable 'tuple_var_assignment_91214' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_91214', subscript_call_result_92482)
        
        # Assigning a Name to a Name (line 393):
        # Getting the type of 'tuple_var_assignment_91210' (line 393)
        tuple_var_assignment_91210_92483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_91210')
        # Assigning a type to the variable 'alpha' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'alpha', tuple_var_assignment_91210_92483)
        
        # Assigning a Name to a Name (line 393):
        # Getting the type of 'tuple_var_assignment_91211' (line 393)
        tuple_var_assignment_91211_92484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_91211')
        # Assigning a type to the variable 'beta' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'beta', tuple_var_assignment_91211_92484)
        
        # Assigning a Name to a Name (line 393):
        # Getting the type of 'tuple_var_assignment_91212' (line 393)
        tuple_var_assignment_91212_92485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_91212')
        # Assigning a type to the variable 'a' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 21), 'a', tuple_var_assignment_91212_92485)
        
        # Assigning a Name to a Name (line 393):
        # Getting the type of 'tuple_var_assignment_91213' (line 393)
        tuple_var_assignment_91213_92486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_91213')
        # Assigning a type to the variable 'x' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 24), 'x', tuple_var_assignment_91213_92486)
        
        # Assigning a Name to a Name (line 393):
        # Getting the type of 'tuple_var_assignment_91214' (line 393)
        tuple_var_assignment_91214_92487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_91214')
        # Assigning a type to the variable 'y' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 27), 'y', tuple_var_assignment_91214_92487)
        
        # Assigning a BinOp to a Name (line 394):
        
        # Assigning a BinOp to a Name (line 394):
        # Getting the type of 'alpha' (line 394)
        alpha_92488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 20), 'alpha')
        
        # Call to matrixmultiply(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'a' (line 394)
        a_92490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 41), 'a', False)
        # Getting the type of 'x' (line 394)
        x_92491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 44), 'x', False)
        # Processing the call keyword arguments (line 394)
        kwargs_92492 = {}
        # Getting the type of 'matrixmultiply' (line 394)
        matrixmultiply_92489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 26), 'matrixmultiply', False)
        # Calling matrixmultiply(args, kwargs) (line 394)
        matrixmultiply_call_result_92493 = invoke(stypy.reporting.localization.Localization(__file__, 394, 26), matrixmultiply_92489, *[a_92490, x_92491], **kwargs_92492)
        
        # Applying the binary operator '*' (line 394)
        result_mul_92494 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 20), '*', alpha_92488, matrixmultiply_call_result_92493)
        
        # Getting the type of 'beta' (line 394)
        beta_92495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 47), 'beta')
        # Getting the type of 'y' (line 394)
        y_92496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 52), 'y')
        # Applying the binary operator '*' (line 394)
        result_mul_92497 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 47), '*', beta_92495, y_92496)
        
        # Applying the binary operator '+' (line 394)
        result_add_92498 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 20), '+', result_mul_92494, result_mul_92497)
        
        # Assigning a type to the variable 'desired_y' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'desired_y', result_add_92498)
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to blas_func(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'alpha' (line 395)
        alpha_92501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 27), 'alpha', False)
        # Getting the type of 'a' (line 395)
        a_92502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 34), 'a', False)
        # Getting the type of 'x' (line 395)
        x_92503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 37), 'x', False)
        # Getting the type of 'beta' (line 395)
        beta_92504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 40), 'beta', False)
        # Getting the type of 'y' (line 395)
        y_92505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 46), 'y', False)
        # Processing the call keyword arguments (line 395)
        kwargs_92506 = {}
        # Getting the type of 'self' (line 395)
        self_92499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 395)
        blas_func_92500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), self_92499, 'blas_func')
        # Calling blas_func(args, kwargs) (line 395)
        blas_func_call_result_92507 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), blas_func_92500, *[alpha_92501, a_92502, x_92503, beta_92504, y_92505], **kwargs_92506)
        
        # Assigning a type to the variable 'y' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'y', blas_func_call_result_92507)
        
        # Call to assert_array_almost_equal(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'desired_y' (line 396)
        desired_y_92509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 34), 'desired_y', False)
        # Getting the type of 'y' (line 396)
        y_92510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 45), 'y', False)
        # Processing the call keyword arguments (line 396)
        kwargs_92511 = {}
        # Getting the type of 'assert_array_almost_equal' (line 396)
        assert_array_almost_equal_92508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 396)
        assert_array_almost_equal_call_result_92512 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), assert_array_almost_equal_92508, *[desired_y_92509, y_92510], **kwargs_92511)
        
        
        # ################# End of 'test_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 392)
        stypy_return_type_92513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92513)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple'
        return stypy_return_type_92513


    @norecursion
    def test_default_beta_y(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_default_beta_y'
        module_type_store = module_type_store.open_function_context('test_default_beta_y', 398, 4, False)
        # Assigning a type to the variable 'self' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseGemv.test_default_beta_y.__dict__.__setitem__('stypy_localization', localization)
        BaseGemv.test_default_beta_y.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseGemv.test_default_beta_y.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseGemv.test_default_beta_y.__dict__.__setitem__('stypy_function_name', 'BaseGemv.test_default_beta_y')
        BaseGemv.test_default_beta_y.__dict__.__setitem__('stypy_param_names_list', [])
        BaseGemv.test_default_beta_y.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseGemv.test_default_beta_y.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseGemv.test_default_beta_y.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseGemv.test_default_beta_y.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseGemv.test_default_beta_y.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseGemv.test_default_beta_y.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.test_default_beta_y', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_default_beta_y', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_default_beta_y(...)' code ##################

        
        # Assigning a Call to a Tuple (line 399):
        
        # Assigning a Subscript to a Name (line 399):
        
        # Obtaining the type of the subscript
        int_92514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 8), 'int')
        
        # Call to get_data(...): (line 399)
        # Processing the call keyword arguments (line 399)
        kwargs_92517 = {}
        # Getting the type of 'self' (line 399)
        self_92515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 399)
        get_data_92516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 31), self_92515, 'get_data')
        # Calling get_data(args, kwargs) (line 399)
        get_data_call_result_92518 = invoke(stypy.reporting.localization.Localization(__file__, 399, 31), get_data_92516, *[], **kwargs_92517)
        
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___92519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), get_data_call_result_92518, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 399)
        subscript_call_result_92520 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), getitem___92519, int_92514)
        
        # Assigning a type to the variable 'tuple_var_assignment_91215' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'tuple_var_assignment_91215', subscript_call_result_92520)
        
        # Assigning a Subscript to a Name (line 399):
        
        # Obtaining the type of the subscript
        int_92521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 8), 'int')
        
        # Call to get_data(...): (line 399)
        # Processing the call keyword arguments (line 399)
        kwargs_92524 = {}
        # Getting the type of 'self' (line 399)
        self_92522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 399)
        get_data_92523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 31), self_92522, 'get_data')
        # Calling get_data(args, kwargs) (line 399)
        get_data_call_result_92525 = invoke(stypy.reporting.localization.Localization(__file__, 399, 31), get_data_92523, *[], **kwargs_92524)
        
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___92526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), get_data_call_result_92525, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 399)
        subscript_call_result_92527 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), getitem___92526, int_92521)
        
        # Assigning a type to the variable 'tuple_var_assignment_91216' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'tuple_var_assignment_91216', subscript_call_result_92527)
        
        # Assigning a Subscript to a Name (line 399):
        
        # Obtaining the type of the subscript
        int_92528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 8), 'int')
        
        # Call to get_data(...): (line 399)
        # Processing the call keyword arguments (line 399)
        kwargs_92531 = {}
        # Getting the type of 'self' (line 399)
        self_92529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 399)
        get_data_92530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 31), self_92529, 'get_data')
        # Calling get_data(args, kwargs) (line 399)
        get_data_call_result_92532 = invoke(stypy.reporting.localization.Localization(__file__, 399, 31), get_data_92530, *[], **kwargs_92531)
        
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___92533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), get_data_call_result_92532, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 399)
        subscript_call_result_92534 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), getitem___92533, int_92528)
        
        # Assigning a type to the variable 'tuple_var_assignment_91217' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'tuple_var_assignment_91217', subscript_call_result_92534)
        
        # Assigning a Subscript to a Name (line 399):
        
        # Obtaining the type of the subscript
        int_92535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 8), 'int')
        
        # Call to get_data(...): (line 399)
        # Processing the call keyword arguments (line 399)
        kwargs_92538 = {}
        # Getting the type of 'self' (line 399)
        self_92536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 399)
        get_data_92537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 31), self_92536, 'get_data')
        # Calling get_data(args, kwargs) (line 399)
        get_data_call_result_92539 = invoke(stypy.reporting.localization.Localization(__file__, 399, 31), get_data_92537, *[], **kwargs_92538)
        
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___92540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), get_data_call_result_92539, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 399)
        subscript_call_result_92541 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), getitem___92540, int_92535)
        
        # Assigning a type to the variable 'tuple_var_assignment_91218' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'tuple_var_assignment_91218', subscript_call_result_92541)
        
        # Assigning a Subscript to a Name (line 399):
        
        # Obtaining the type of the subscript
        int_92542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 8), 'int')
        
        # Call to get_data(...): (line 399)
        # Processing the call keyword arguments (line 399)
        kwargs_92545 = {}
        # Getting the type of 'self' (line 399)
        self_92543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 399)
        get_data_92544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 31), self_92543, 'get_data')
        # Calling get_data(args, kwargs) (line 399)
        get_data_call_result_92546 = invoke(stypy.reporting.localization.Localization(__file__, 399, 31), get_data_92544, *[], **kwargs_92545)
        
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___92547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), get_data_call_result_92546, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 399)
        subscript_call_result_92548 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), getitem___92547, int_92542)
        
        # Assigning a type to the variable 'tuple_var_assignment_91219' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'tuple_var_assignment_91219', subscript_call_result_92548)
        
        # Assigning a Name to a Name (line 399):
        # Getting the type of 'tuple_var_assignment_91215' (line 399)
        tuple_var_assignment_91215_92549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'tuple_var_assignment_91215')
        # Assigning a type to the variable 'alpha' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'alpha', tuple_var_assignment_91215_92549)
        
        # Assigning a Name to a Name (line 399):
        # Getting the type of 'tuple_var_assignment_91216' (line 399)
        tuple_var_assignment_91216_92550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'tuple_var_assignment_91216')
        # Assigning a type to the variable 'beta' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 15), 'beta', tuple_var_assignment_91216_92550)
        
        # Assigning a Name to a Name (line 399):
        # Getting the type of 'tuple_var_assignment_91217' (line 399)
        tuple_var_assignment_91217_92551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'tuple_var_assignment_91217')
        # Assigning a type to the variable 'a' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 21), 'a', tuple_var_assignment_91217_92551)
        
        # Assigning a Name to a Name (line 399):
        # Getting the type of 'tuple_var_assignment_91218' (line 399)
        tuple_var_assignment_91218_92552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'tuple_var_assignment_91218')
        # Assigning a type to the variable 'x' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 24), 'x', tuple_var_assignment_91218_92552)
        
        # Assigning a Name to a Name (line 399):
        # Getting the type of 'tuple_var_assignment_91219' (line 399)
        tuple_var_assignment_91219_92553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'tuple_var_assignment_91219')
        # Assigning a type to the variable 'y' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 27), 'y', tuple_var_assignment_91219_92553)
        
        # Assigning a Call to a Name (line 400):
        
        # Assigning a Call to a Name (line 400):
        
        # Call to matrixmultiply(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'a' (line 400)
        a_92555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 35), 'a', False)
        # Getting the type of 'x' (line 400)
        x_92556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 38), 'x', False)
        # Processing the call keyword arguments (line 400)
        kwargs_92557 = {}
        # Getting the type of 'matrixmultiply' (line 400)
        matrixmultiply_92554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 20), 'matrixmultiply', False)
        # Calling matrixmultiply(args, kwargs) (line 400)
        matrixmultiply_call_result_92558 = invoke(stypy.reporting.localization.Localization(__file__, 400, 20), matrixmultiply_92554, *[a_92555, x_92556], **kwargs_92557)
        
        # Assigning a type to the variable 'desired_y' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'desired_y', matrixmultiply_call_result_92558)
        
        # Assigning a Call to a Name (line 401):
        
        # Assigning a Call to a Name (line 401):
        
        # Call to blas_func(...): (line 401)
        # Processing the call arguments (line 401)
        int_92561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 27), 'int')
        # Getting the type of 'a' (line 401)
        a_92562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 30), 'a', False)
        # Getting the type of 'x' (line 401)
        x_92563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 33), 'x', False)
        # Processing the call keyword arguments (line 401)
        kwargs_92564 = {}
        # Getting the type of 'self' (line 401)
        self_92559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 401)
        blas_func_92560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 12), self_92559, 'blas_func')
        # Calling blas_func(args, kwargs) (line 401)
        blas_func_call_result_92565 = invoke(stypy.reporting.localization.Localization(__file__, 401, 12), blas_func_92560, *[int_92561, a_92562, x_92563], **kwargs_92564)
        
        # Assigning a type to the variable 'y' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'y', blas_func_call_result_92565)
        
        # Call to assert_array_almost_equal(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'desired_y' (line 402)
        desired_y_92567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 34), 'desired_y', False)
        # Getting the type of 'y' (line 402)
        y_92568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 45), 'y', False)
        # Processing the call keyword arguments (line 402)
        kwargs_92569 = {}
        # Getting the type of 'assert_array_almost_equal' (line 402)
        assert_array_almost_equal_92566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 402)
        assert_array_almost_equal_call_result_92570 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), assert_array_almost_equal_92566, *[desired_y_92567, y_92568], **kwargs_92569)
        
        
        # ################# End of 'test_default_beta_y(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_default_beta_y' in the type store
        # Getting the type of 'stypy_return_type' (line 398)
        stypy_return_type_92571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92571)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_default_beta_y'
        return stypy_return_type_92571


    @norecursion
    def test_simple_transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_transpose'
        module_type_store = module_type_store.open_function_context('test_simple_transpose', 404, 4, False)
        # Assigning a type to the variable 'self' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseGemv.test_simple_transpose.__dict__.__setitem__('stypy_localization', localization)
        BaseGemv.test_simple_transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseGemv.test_simple_transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseGemv.test_simple_transpose.__dict__.__setitem__('stypy_function_name', 'BaseGemv.test_simple_transpose')
        BaseGemv.test_simple_transpose.__dict__.__setitem__('stypy_param_names_list', [])
        BaseGemv.test_simple_transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseGemv.test_simple_transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseGemv.test_simple_transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseGemv.test_simple_transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseGemv.test_simple_transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseGemv.test_simple_transpose.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.test_simple_transpose', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_transpose', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_transpose(...)' code ##################

        
        # Assigning a Call to a Tuple (line 405):
        
        # Assigning a Subscript to a Name (line 405):
        
        # Obtaining the type of the subscript
        int_92572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 8), 'int')
        
        # Call to get_data(...): (line 405)
        # Processing the call keyword arguments (line 405)
        kwargs_92575 = {}
        # Getting the type of 'self' (line 405)
        self_92573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 405)
        get_data_92574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 31), self_92573, 'get_data')
        # Calling get_data(args, kwargs) (line 405)
        get_data_call_result_92576 = invoke(stypy.reporting.localization.Localization(__file__, 405, 31), get_data_92574, *[], **kwargs_92575)
        
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___92577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), get_data_call_result_92576, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_92578 = invoke(stypy.reporting.localization.Localization(__file__, 405, 8), getitem___92577, int_92572)
        
        # Assigning a type to the variable 'tuple_var_assignment_91220' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_91220', subscript_call_result_92578)
        
        # Assigning a Subscript to a Name (line 405):
        
        # Obtaining the type of the subscript
        int_92579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 8), 'int')
        
        # Call to get_data(...): (line 405)
        # Processing the call keyword arguments (line 405)
        kwargs_92582 = {}
        # Getting the type of 'self' (line 405)
        self_92580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 405)
        get_data_92581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 31), self_92580, 'get_data')
        # Calling get_data(args, kwargs) (line 405)
        get_data_call_result_92583 = invoke(stypy.reporting.localization.Localization(__file__, 405, 31), get_data_92581, *[], **kwargs_92582)
        
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___92584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), get_data_call_result_92583, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_92585 = invoke(stypy.reporting.localization.Localization(__file__, 405, 8), getitem___92584, int_92579)
        
        # Assigning a type to the variable 'tuple_var_assignment_91221' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_91221', subscript_call_result_92585)
        
        # Assigning a Subscript to a Name (line 405):
        
        # Obtaining the type of the subscript
        int_92586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 8), 'int')
        
        # Call to get_data(...): (line 405)
        # Processing the call keyword arguments (line 405)
        kwargs_92589 = {}
        # Getting the type of 'self' (line 405)
        self_92587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 405)
        get_data_92588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 31), self_92587, 'get_data')
        # Calling get_data(args, kwargs) (line 405)
        get_data_call_result_92590 = invoke(stypy.reporting.localization.Localization(__file__, 405, 31), get_data_92588, *[], **kwargs_92589)
        
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___92591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), get_data_call_result_92590, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_92592 = invoke(stypy.reporting.localization.Localization(__file__, 405, 8), getitem___92591, int_92586)
        
        # Assigning a type to the variable 'tuple_var_assignment_91222' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_91222', subscript_call_result_92592)
        
        # Assigning a Subscript to a Name (line 405):
        
        # Obtaining the type of the subscript
        int_92593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 8), 'int')
        
        # Call to get_data(...): (line 405)
        # Processing the call keyword arguments (line 405)
        kwargs_92596 = {}
        # Getting the type of 'self' (line 405)
        self_92594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 405)
        get_data_92595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 31), self_92594, 'get_data')
        # Calling get_data(args, kwargs) (line 405)
        get_data_call_result_92597 = invoke(stypy.reporting.localization.Localization(__file__, 405, 31), get_data_92595, *[], **kwargs_92596)
        
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___92598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), get_data_call_result_92597, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_92599 = invoke(stypy.reporting.localization.Localization(__file__, 405, 8), getitem___92598, int_92593)
        
        # Assigning a type to the variable 'tuple_var_assignment_91223' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_91223', subscript_call_result_92599)
        
        # Assigning a Subscript to a Name (line 405):
        
        # Obtaining the type of the subscript
        int_92600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 8), 'int')
        
        # Call to get_data(...): (line 405)
        # Processing the call keyword arguments (line 405)
        kwargs_92603 = {}
        # Getting the type of 'self' (line 405)
        self_92601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 405)
        get_data_92602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 31), self_92601, 'get_data')
        # Calling get_data(args, kwargs) (line 405)
        get_data_call_result_92604 = invoke(stypy.reporting.localization.Localization(__file__, 405, 31), get_data_92602, *[], **kwargs_92603)
        
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___92605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), get_data_call_result_92604, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_92606 = invoke(stypy.reporting.localization.Localization(__file__, 405, 8), getitem___92605, int_92600)
        
        # Assigning a type to the variable 'tuple_var_assignment_91224' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_91224', subscript_call_result_92606)
        
        # Assigning a Name to a Name (line 405):
        # Getting the type of 'tuple_var_assignment_91220' (line 405)
        tuple_var_assignment_91220_92607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_91220')
        # Assigning a type to the variable 'alpha' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'alpha', tuple_var_assignment_91220_92607)
        
        # Assigning a Name to a Name (line 405):
        # Getting the type of 'tuple_var_assignment_91221' (line 405)
        tuple_var_assignment_91221_92608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_91221')
        # Assigning a type to the variable 'beta' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 15), 'beta', tuple_var_assignment_91221_92608)
        
        # Assigning a Name to a Name (line 405):
        # Getting the type of 'tuple_var_assignment_91222' (line 405)
        tuple_var_assignment_91222_92609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_91222')
        # Assigning a type to the variable 'a' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), 'a', tuple_var_assignment_91222_92609)
        
        # Assigning a Name to a Name (line 405):
        # Getting the type of 'tuple_var_assignment_91223' (line 405)
        tuple_var_assignment_91223_92610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_91223')
        # Assigning a type to the variable 'x' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 24), 'x', tuple_var_assignment_91223_92610)
        
        # Assigning a Name to a Name (line 405):
        # Getting the type of 'tuple_var_assignment_91224' (line 405)
        tuple_var_assignment_91224_92611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_91224')
        # Assigning a type to the variable 'y' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 27), 'y', tuple_var_assignment_91224_92611)
        
        # Assigning a BinOp to a Name (line 406):
        
        # Assigning a BinOp to a Name (line 406):
        # Getting the type of 'alpha' (line 406)
        alpha_92612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'alpha')
        
        # Call to matrixmultiply(...): (line 406)
        # Processing the call arguments (line 406)
        
        # Call to transpose(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'a' (line 406)
        a_92615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 51), 'a', False)
        # Processing the call keyword arguments (line 406)
        kwargs_92616 = {}
        # Getting the type of 'transpose' (line 406)
        transpose_92614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 41), 'transpose', False)
        # Calling transpose(args, kwargs) (line 406)
        transpose_call_result_92617 = invoke(stypy.reporting.localization.Localization(__file__, 406, 41), transpose_92614, *[a_92615], **kwargs_92616)
        
        # Getting the type of 'x' (line 406)
        x_92618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 55), 'x', False)
        # Processing the call keyword arguments (line 406)
        kwargs_92619 = {}
        # Getting the type of 'matrixmultiply' (line 406)
        matrixmultiply_92613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 26), 'matrixmultiply', False)
        # Calling matrixmultiply(args, kwargs) (line 406)
        matrixmultiply_call_result_92620 = invoke(stypy.reporting.localization.Localization(__file__, 406, 26), matrixmultiply_92613, *[transpose_call_result_92617, x_92618], **kwargs_92619)
        
        # Applying the binary operator '*' (line 406)
        result_mul_92621 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 20), '*', alpha_92612, matrixmultiply_call_result_92620)
        
        # Getting the type of 'beta' (line 406)
        beta_92622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 58), 'beta')
        # Getting the type of 'y' (line 406)
        y_92623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 63), 'y')
        # Applying the binary operator '*' (line 406)
        result_mul_92624 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 58), '*', beta_92622, y_92623)
        
        # Applying the binary operator '+' (line 406)
        result_add_92625 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 20), '+', result_mul_92621, result_mul_92624)
        
        # Assigning a type to the variable 'desired_y' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'desired_y', result_add_92625)
        
        # Assigning a Call to a Name (line 407):
        
        # Assigning a Call to a Name (line 407):
        
        # Call to blas_func(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'alpha' (line 407)
        alpha_92628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 27), 'alpha', False)
        # Getting the type of 'a' (line 407)
        a_92629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 34), 'a', False)
        # Getting the type of 'x' (line 407)
        x_92630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 37), 'x', False)
        # Getting the type of 'beta' (line 407)
        beta_92631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 40), 'beta', False)
        # Getting the type of 'y' (line 407)
        y_92632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 46), 'y', False)
        # Processing the call keyword arguments (line 407)
        int_92633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 55), 'int')
        keyword_92634 = int_92633
        kwargs_92635 = {'trans': keyword_92634}
        # Getting the type of 'self' (line 407)
        self_92626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 407)
        blas_func_92627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 12), self_92626, 'blas_func')
        # Calling blas_func(args, kwargs) (line 407)
        blas_func_call_result_92636 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), blas_func_92627, *[alpha_92628, a_92629, x_92630, beta_92631, y_92632], **kwargs_92635)
        
        # Assigning a type to the variable 'y' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'y', blas_func_call_result_92636)
        
        # Call to assert_array_almost_equal(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'desired_y' (line 408)
        desired_y_92638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 34), 'desired_y', False)
        # Getting the type of 'y' (line 408)
        y_92639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 45), 'y', False)
        # Processing the call keyword arguments (line 408)
        kwargs_92640 = {}
        # Getting the type of 'assert_array_almost_equal' (line 408)
        assert_array_almost_equal_92637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 408)
        assert_array_almost_equal_call_result_92641 = invoke(stypy.reporting.localization.Localization(__file__, 408, 8), assert_array_almost_equal_92637, *[desired_y_92638, y_92639], **kwargs_92640)
        
        
        # ################# End of 'test_simple_transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 404)
        stypy_return_type_92642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92642)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_transpose'
        return stypy_return_type_92642


    @norecursion
    def test_simple_transpose_conj(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_transpose_conj'
        module_type_store = module_type_store.open_function_context('test_simple_transpose_conj', 410, 4, False)
        # Assigning a type to the variable 'self' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseGemv.test_simple_transpose_conj.__dict__.__setitem__('stypy_localization', localization)
        BaseGemv.test_simple_transpose_conj.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseGemv.test_simple_transpose_conj.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseGemv.test_simple_transpose_conj.__dict__.__setitem__('stypy_function_name', 'BaseGemv.test_simple_transpose_conj')
        BaseGemv.test_simple_transpose_conj.__dict__.__setitem__('stypy_param_names_list', [])
        BaseGemv.test_simple_transpose_conj.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseGemv.test_simple_transpose_conj.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseGemv.test_simple_transpose_conj.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseGemv.test_simple_transpose_conj.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseGemv.test_simple_transpose_conj.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseGemv.test_simple_transpose_conj.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.test_simple_transpose_conj', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_transpose_conj', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_transpose_conj(...)' code ##################

        
        # Assigning a Call to a Tuple (line 411):
        
        # Assigning a Subscript to a Name (line 411):
        
        # Obtaining the type of the subscript
        int_92643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 8), 'int')
        
        # Call to get_data(...): (line 411)
        # Processing the call keyword arguments (line 411)
        kwargs_92646 = {}
        # Getting the type of 'self' (line 411)
        self_92644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 411)
        get_data_92645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 31), self_92644, 'get_data')
        # Calling get_data(args, kwargs) (line 411)
        get_data_call_result_92647 = invoke(stypy.reporting.localization.Localization(__file__, 411, 31), get_data_92645, *[], **kwargs_92646)
        
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___92648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), get_data_call_result_92647, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_92649 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), getitem___92648, int_92643)
        
        # Assigning a type to the variable 'tuple_var_assignment_91225' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_91225', subscript_call_result_92649)
        
        # Assigning a Subscript to a Name (line 411):
        
        # Obtaining the type of the subscript
        int_92650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 8), 'int')
        
        # Call to get_data(...): (line 411)
        # Processing the call keyword arguments (line 411)
        kwargs_92653 = {}
        # Getting the type of 'self' (line 411)
        self_92651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 411)
        get_data_92652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 31), self_92651, 'get_data')
        # Calling get_data(args, kwargs) (line 411)
        get_data_call_result_92654 = invoke(stypy.reporting.localization.Localization(__file__, 411, 31), get_data_92652, *[], **kwargs_92653)
        
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___92655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), get_data_call_result_92654, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_92656 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), getitem___92655, int_92650)
        
        # Assigning a type to the variable 'tuple_var_assignment_91226' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_91226', subscript_call_result_92656)
        
        # Assigning a Subscript to a Name (line 411):
        
        # Obtaining the type of the subscript
        int_92657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 8), 'int')
        
        # Call to get_data(...): (line 411)
        # Processing the call keyword arguments (line 411)
        kwargs_92660 = {}
        # Getting the type of 'self' (line 411)
        self_92658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 411)
        get_data_92659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 31), self_92658, 'get_data')
        # Calling get_data(args, kwargs) (line 411)
        get_data_call_result_92661 = invoke(stypy.reporting.localization.Localization(__file__, 411, 31), get_data_92659, *[], **kwargs_92660)
        
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___92662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), get_data_call_result_92661, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_92663 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), getitem___92662, int_92657)
        
        # Assigning a type to the variable 'tuple_var_assignment_91227' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_91227', subscript_call_result_92663)
        
        # Assigning a Subscript to a Name (line 411):
        
        # Obtaining the type of the subscript
        int_92664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 8), 'int')
        
        # Call to get_data(...): (line 411)
        # Processing the call keyword arguments (line 411)
        kwargs_92667 = {}
        # Getting the type of 'self' (line 411)
        self_92665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 411)
        get_data_92666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 31), self_92665, 'get_data')
        # Calling get_data(args, kwargs) (line 411)
        get_data_call_result_92668 = invoke(stypy.reporting.localization.Localization(__file__, 411, 31), get_data_92666, *[], **kwargs_92667)
        
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___92669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), get_data_call_result_92668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_92670 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), getitem___92669, int_92664)
        
        # Assigning a type to the variable 'tuple_var_assignment_91228' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_91228', subscript_call_result_92670)
        
        # Assigning a Subscript to a Name (line 411):
        
        # Obtaining the type of the subscript
        int_92671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 8), 'int')
        
        # Call to get_data(...): (line 411)
        # Processing the call keyword arguments (line 411)
        kwargs_92674 = {}
        # Getting the type of 'self' (line 411)
        self_92672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 411)
        get_data_92673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 31), self_92672, 'get_data')
        # Calling get_data(args, kwargs) (line 411)
        get_data_call_result_92675 = invoke(stypy.reporting.localization.Localization(__file__, 411, 31), get_data_92673, *[], **kwargs_92674)
        
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___92676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), get_data_call_result_92675, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_92677 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), getitem___92676, int_92671)
        
        # Assigning a type to the variable 'tuple_var_assignment_91229' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_91229', subscript_call_result_92677)
        
        # Assigning a Name to a Name (line 411):
        # Getting the type of 'tuple_var_assignment_91225' (line 411)
        tuple_var_assignment_91225_92678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_91225')
        # Assigning a type to the variable 'alpha' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'alpha', tuple_var_assignment_91225_92678)
        
        # Assigning a Name to a Name (line 411):
        # Getting the type of 'tuple_var_assignment_91226' (line 411)
        tuple_var_assignment_91226_92679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_91226')
        # Assigning a type to the variable 'beta' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 15), 'beta', tuple_var_assignment_91226_92679)
        
        # Assigning a Name to a Name (line 411):
        # Getting the type of 'tuple_var_assignment_91227' (line 411)
        tuple_var_assignment_91227_92680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_91227')
        # Assigning a type to the variable 'a' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 21), 'a', tuple_var_assignment_91227_92680)
        
        # Assigning a Name to a Name (line 411):
        # Getting the type of 'tuple_var_assignment_91228' (line 411)
        tuple_var_assignment_91228_92681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_91228')
        # Assigning a type to the variable 'x' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 24), 'x', tuple_var_assignment_91228_92681)
        
        # Assigning a Name to a Name (line 411):
        # Getting the type of 'tuple_var_assignment_91229' (line 411)
        tuple_var_assignment_91229_92682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_91229')
        # Assigning a type to the variable 'y' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 27), 'y', tuple_var_assignment_91229_92682)
        
        # Assigning a BinOp to a Name (line 412):
        
        # Assigning a BinOp to a Name (line 412):
        # Getting the type of 'alpha' (line 412)
        alpha_92683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 20), 'alpha')
        
        # Call to matrixmultiply(...): (line 412)
        # Processing the call arguments (line 412)
        
        # Call to transpose(...): (line 412)
        # Processing the call arguments (line 412)
        
        # Call to conjugate(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'a' (line 412)
        a_92687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 61), 'a', False)
        # Processing the call keyword arguments (line 412)
        kwargs_92688 = {}
        # Getting the type of 'conjugate' (line 412)
        conjugate_92686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 51), 'conjugate', False)
        # Calling conjugate(args, kwargs) (line 412)
        conjugate_call_result_92689 = invoke(stypy.reporting.localization.Localization(__file__, 412, 51), conjugate_92686, *[a_92687], **kwargs_92688)
        
        # Processing the call keyword arguments (line 412)
        kwargs_92690 = {}
        # Getting the type of 'transpose' (line 412)
        transpose_92685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 41), 'transpose', False)
        # Calling transpose(args, kwargs) (line 412)
        transpose_call_result_92691 = invoke(stypy.reporting.localization.Localization(__file__, 412, 41), transpose_92685, *[conjugate_call_result_92689], **kwargs_92690)
        
        # Getting the type of 'x' (line 412)
        x_92692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 66), 'x', False)
        # Processing the call keyword arguments (line 412)
        kwargs_92693 = {}
        # Getting the type of 'matrixmultiply' (line 412)
        matrixmultiply_92684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 26), 'matrixmultiply', False)
        # Calling matrixmultiply(args, kwargs) (line 412)
        matrixmultiply_call_result_92694 = invoke(stypy.reporting.localization.Localization(__file__, 412, 26), matrixmultiply_92684, *[transpose_call_result_92691, x_92692], **kwargs_92693)
        
        # Applying the binary operator '*' (line 412)
        result_mul_92695 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 20), '*', alpha_92683, matrixmultiply_call_result_92694)
        
        # Getting the type of 'beta' (line 412)
        beta_92696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 69), 'beta')
        # Getting the type of 'y' (line 412)
        y_92697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 74), 'y')
        # Applying the binary operator '*' (line 412)
        result_mul_92698 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 69), '*', beta_92696, y_92697)
        
        # Applying the binary operator '+' (line 412)
        result_add_92699 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 20), '+', result_mul_92695, result_mul_92698)
        
        # Assigning a type to the variable 'desired_y' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'desired_y', result_add_92699)
        
        # Assigning a Call to a Name (line 413):
        
        # Assigning a Call to a Name (line 413):
        
        # Call to blas_func(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'alpha' (line 413)
        alpha_92702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 27), 'alpha', False)
        # Getting the type of 'a' (line 413)
        a_92703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 34), 'a', False)
        # Getting the type of 'x' (line 413)
        x_92704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 37), 'x', False)
        # Getting the type of 'beta' (line 413)
        beta_92705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 40), 'beta', False)
        # Getting the type of 'y' (line 413)
        y_92706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 46), 'y', False)
        # Processing the call keyword arguments (line 413)
        int_92707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 55), 'int')
        keyword_92708 = int_92707
        kwargs_92709 = {'trans': keyword_92708}
        # Getting the type of 'self' (line 413)
        self_92700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 413)
        blas_func_92701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 12), self_92700, 'blas_func')
        # Calling blas_func(args, kwargs) (line 413)
        blas_func_call_result_92710 = invoke(stypy.reporting.localization.Localization(__file__, 413, 12), blas_func_92701, *[alpha_92702, a_92703, x_92704, beta_92705, y_92706], **kwargs_92709)
        
        # Assigning a type to the variable 'y' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'y', blas_func_call_result_92710)
        
        # Call to assert_array_almost_equal(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'desired_y' (line 414)
        desired_y_92712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 34), 'desired_y', False)
        # Getting the type of 'y' (line 414)
        y_92713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 45), 'y', False)
        # Processing the call keyword arguments (line 414)
        kwargs_92714 = {}
        # Getting the type of 'assert_array_almost_equal' (line 414)
        assert_array_almost_equal_92711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 414)
        assert_array_almost_equal_call_result_92715 = invoke(stypy.reporting.localization.Localization(__file__, 414, 8), assert_array_almost_equal_92711, *[desired_y_92712, y_92713], **kwargs_92714)
        
        
        # ################# End of 'test_simple_transpose_conj(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_transpose_conj' in the type store
        # Getting the type of 'stypy_return_type' (line 410)
        stypy_return_type_92716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92716)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_transpose_conj'
        return stypy_return_type_92716


    @norecursion
    def test_x_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_stride'
        module_type_store = module_type_store.open_function_context('test_x_stride', 416, 4, False)
        # Assigning a type to the variable 'self' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseGemv.test_x_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseGemv.test_x_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseGemv.test_x_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseGemv.test_x_stride.__dict__.__setitem__('stypy_function_name', 'BaseGemv.test_x_stride')
        BaseGemv.test_x_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseGemv.test_x_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseGemv.test_x_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseGemv.test_x_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseGemv.test_x_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseGemv.test_x_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseGemv.test_x_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.test_x_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_stride(...)' code ##################

        
        # Assigning a Call to a Tuple (line 417):
        
        # Assigning a Subscript to a Name (line 417):
        
        # Obtaining the type of the subscript
        int_92717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 8), 'int')
        
        # Call to get_data(...): (line 417)
        # Processing the call keyword arguments (line 417)
        int_92720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 54), 'int')
        keyword_92721 = int_92720
        kwargs_92722 = {'x_stride': keyword_92721}
        # Getting the type of 'self' (line 417)
        self_92718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 417)
        get_data_92719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 31), self_92718, 'get_data')
        # Calling get_data(args, kwargs) (line 417)
        get_data_call_result_92723 = invoke(stypy.reporting.localization.Localization(__file__, 417, 31), get_data_92719, *[], **kwargs_92722)
        
        # Obtaining the member '__getitem__' of a type (line 417)
        getitem___92724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 8), get_data_call_result_92723, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 417)
        subscript_call_result_92725 = invoke(stypy.reporting.localization.Localization(__file__, 417, 8), getitem___92724, int_92717)
        
        # Assigning a type to the variable 'tuple_var_assignment_91230' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'tuple_var_assignment_91230', subscript_call_result_92725)
        
        # Assigning a Subscript to a Name (line 417):
        
        # Obtaining the type of the subscript
        int_92726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 8), 'int')
        
        # Call to get_data(...): (line 417)
        # Processing the call keyword arguments (line 417)
        int_92729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 54), 'int')
        keyword_92730 = int_92729
        kwargs_92731 = {'x_stride': keyword_92730}
        # Getting the type of 'self' (line 417)
        self_92727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 417)
        get_data_92728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 31), self_92727, 'get_data')
        # Calling get_data(args, kwargs) (line 417)
        get_data_call_result_92732 = invoke(stypy.reporting.localization.Localization(__file__, 417, 31), get_data_92728, *[], **kwargs_92731)
        
        # Obtaining the member '__getitem__' of a type (line 417)
        getitem___92733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 8), get_data_call_result_92732, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 417)
        subscript_call_result_92734 = invoke(stypy.reporting.localization.Localization(__file__, 417, 8), getitem___92733, int_92726)
        
        # Assigning a type to the variable 'tuple_var_assignment_91231' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'tuple_var_assignment_91231', subscript_call_result_92734)
        
        # Assigning a Subscript to a Name (line 417):
        
        # Obtaining the type of the subscript
        int_92735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 8), 'int')
        
        # Call to get_data(...): (line 417)
        # Processing the call keyword arguments (line 417)
        int_92738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 54), 'int')
        keyword_92739 = int_92738
        kwargs_92740 = {'x_stride': keyword_92739}
        # Getting the type of 'self' (line 417)
        self_92736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 417)
        get_data_92737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 31), self_92736, 'get_data')
        # Calling get_data(args, kwargs) (line 417)
        get_data_call_result_92741 = invoke(stypy.reporting.localization.Localization(__file__, 417, 31), get_data_92737, *[], **kwargs_92740)
        
        # Obtaining the member '__getitem__' of a type (line 417)
        getitem___92742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 8), get_data_call_result_92741, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 417)
        subscript_call_result_92743 = invoke(stypy.reporting.localization.Localization(__file__, 417, 8), getitem___92742, int_92735)
        
        # Assigning a type to the variable 'tuple_var_assignment_91232' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'tuple_var_assignment_91232', subscript_call_result_92743)
        
        # Assigning a Subscript to a Name (line 417):
        
        # Obtaining the type of the subscript
        int_92744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 8), 'int')
        
        # Call to get_data(...): (line 417)
        # Processing the call keyword arguments (line 417)
        int_92747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 54), 'int')
        keyword_92748 = int_92747
        kwargs_92749 = {'x_stride': keyword_92748}
        # Getting the type of 'self' (line 417)
        self_92745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 417)
        get_data_92746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 31), self_92745, 'get_data')
        # Calling get_data(args, kwargs) (line 417)
        get_data_call_result_92750 = invoke(stypy.reporting.localization.Localization(__file__, 417, 31), get_data_92746, *[], **kwargs_92749)
        
        # Obtaining the member '__getitem__' of a type (line 417)
        getitem___92751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 8), get_data_call_result_92750, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 417)
        subscript_call_result_92752 = invoke(stypy.reporting.localization.Localization(__file__, 417, 8), getitem___92751, int_92744)
        
        # Assigning a type to the variable 'tuple_var_assignment_91233' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'tuple_var_assignment_91233', subscript_call_result_92752)
        
        # Assigning a Subscript to a Name (line 417):
        
        # Obtaining the type of the subscript
        int_92753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 8), 'int')
        
        # Call to get_data(...): (line 417)
        # Processing the call keyword arguments (line 417)
        int_92756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 54), 'int')
        keyword_92757 = int_92756
        kwargs_92758 = {'x_stride': keyword_92757}
        # Getting the type of 'self' (line 417)
        self_92754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 417)
        get_data_92755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 31), self_92754, 'get_data')
        # Calling get_data(args, kwargs) (line 417)
        get_data_call_result_92759 = invoke(stypy.reporting.localization.Localization(__file__, 417, 31), get_data_92755, *[], **kwargs_92758)
        
        # Obtaining the member '__getitem__' of a type (line 417)
        getitem___92760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 8), get_data_call_result_92759, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 417)
        subscript_call_result_92761 = invoke(stypy.reporting.localization.Localization(__file__, 417, 8), getitem___92760, int_92753)
        
        # Assigning a type to the variable 'tuple_var_assignment_91234' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'tuple_var_assignment_91234', subscript_call_result_92761)
        
        # Assigning a Name to a Name (line 417):
        # Getting the type of 'tuple_var_assignment_91230' (line 417)
        tuple_var_assignment_91230_92762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'tuple_var_assignment_91230')
        # Assigning a type to the variable 'alpha' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'alpha', tuple_var_assignment_91230_92762)
        
        # Assigning a Name to a Name (line 417):
        # Getting the type of 'tuple_var_assignment_91231' (line 417)
        tuple_var_assignment_91231_92763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'tuple_var_assignment_91231')
        # Assigning a type to the variable 'beta' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 15), 'beta', tuple_var_assignment_91231_92763)
        
        # Assigning a Name to a Name (line 417):
        # Getting the type of 'tuple_var_assignment_91232' (line 417)
        tuple_var_assignment_91232_92764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'tuple_var_assignment_91232')
        # Assigning a type to the variable 'a' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 21), 'a', tuple_var_assignment_91232_92764)
        
        # Assigning a Name to a Name (line 417):
        # Getting the type of 'tuple_var_assignment_91233' (line 417)
        tuple_var_assignment_91233_92765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'tuple_var_assignment_91233')
        # Assigning a type to the variable 'x' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 24), 'x', tuple_var_assignment_91233_92765)
        
        # Assigning a Name to a Name (line 417):
        # Getting the type of 'tuple_var_assignment_91234' (line 417)
        tuple_var_assignment_91234_92766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'tuple_var_assignment_91234')
        # Assigning a type to the variable 'y' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 27), 'y', tuple_var_assignment_91234_92766)
        
        # Assigning a BinOp to a Name (line 418):
        
        # Assigning a BinOp to a Name (line 418):
        # Getting the type of 'alpha' (line 418)
        alpha_92767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 20), 'alpha')
        
        # Call to matrixmultiply(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'a' (line 418)
        a_92769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 41), 'a', False)
        
        # Obtaining the type of the subscript
        int_92770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 48), 'int')
        slice_92771 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 418, 44), None, None, int_92770)
        # Getting the type of 'x' (line 418)
        x_92772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 44), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 418)
        getitem___92773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 44), x_92772, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 418)
        subscript_call_result_92774 = invoke(stypy.reporting.localization.Localization(__file__, 418, 44), getitem___92773, slice_92771)
        
        # Processing the call keyword arguments (line 418)
        kwargs_92775 = {}
        # Getting the type of 'matrixmultiply' (line 418)
        matrixmultiply_92768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 26), 'matrixmultiply', False)
        # Calling matrixmultiply(args, kwargs) (line 418)
        matrixmultiply_call_result_92776 = invoke(stypy.reporting.localization.Localization(__file__, 418, 26), matrixmultiply_92768, *[a_92769, subscript_call_result_92774], **kwargs_92775)
        
        # Applying the binary operator '*' (line 418)
        result_mul_92777 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 20), '*', alpha_92767, matrixmultiply_call_result_92776)
        
        # Getting the type of 'beta' (line 418)
        beta_92778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 52), 'beta')
        # Getting the type of 'y' (line 418)
        y_92779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 57), 'y')
        # Applying the binary operator '*' (line 418)
        result_mul_92780 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 52), '*', beta_92778, y_92779)
        
        # Applying the binary operator '+' (line 418)
        result_add_92781 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 20), '+', result_mul_92777, result_mul_92780)
        
        # Assigning a type to the variable 'desired_y' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'desired_y', result_add_92781)
        
        # Assigning a Call to a Name (line 419):
        
        # Assigning a Call to a Name (line 419):
        
        # Call to blas_func(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'alpha' (line 419)
        alpha_92784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 27), 'alpha', False)
        # Getting the type of 'a' (line 419)
        a_92785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 34), 'a', False)
        # Getting the type of 'x' (line 419)
        x_92786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 37), 'x', False)
        # Getting the type of 'beta' (line 419)
        beta_92787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 40), 'beta', False)
        # Getting the type of 'y' (line 419)
        y_92788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 46), 'y', False)
        # Processing the call keyword arguments (line 419)
        int_92789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 54), 'int')
        keyword_92790 = int_92789
        kwargs_92791 = {'incx': keyword_92790}
        # Getting the type of 'self' (line 419)
        self_92782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 419)
        blas_func_92783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 12), self_92782, 'blas_func')
        # Calling blas_func(args, kwargs) (line 419)
        blas_func_call_result_92792 = invoke(stypy.reporting.localization.Localization(__file__, 419, 12), blas_func_92783, *[alpha_92784, a_92785, x_92786, beta_92787, y_92788], **kwargs_92791)
        
        # Assigning a type to the variable 'y' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'y', blas_func_call_result_92792)
        
        # Call to assert_array_almost_equal(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'desired_y' (line 420)
        desired_y_92794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 34), 'desired_y', False)
        # Getting the type of 'y' (line 420)
        y_92795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 45), 'y', False)
        # Processing the call keyword arguments (line 420)
        kwargs_92796 = {}
        # Getting the type of 'assert_array_almost_equal' (line 420)
        assert_array_almost_equal_92793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 420)
        assert_array_almost_equal_call_result_92797 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), assert_array_almost_equal_92793, *[desired_y_92794, y_92795], **kwargs_92796)
        
        
        # ################# End of 'test_x_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 416)
        stypy_return_type_92798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92798)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_stride'
        return stypy_return_type_92798


    @norecursion
    def test_x_stride_transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_stride_transpose'
        module_type_store = module_type_store.open_function_context('test_x_stride_transpose', 422, 4, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseGemv.test_x_stride_transpose.__dict__.__setitem__('stypy_localization', localization)
        BaseGemv.test_x_stride_transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseGemv.test_x_stride_transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseGemv.test_x_stride_transpose.__dict__.__setitem__('stypy_function_name', 'BaseGemv.test_x_stride_transpose')
        BaseGemv.test_x_stride_transpose.__dict__.__setitem__('stypy_param_names_list', [])
        BaseGemv.test_x_stride_transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseGemv.test_x_stride_transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseGemv.test_x_stride_transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseGemv.test_x_stride_transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseGemv.test_x_stride_transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseGemv.test_x_stride_transpose.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.test_x_stride_transpose', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_stride_transpose', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_stride_transpose(...)' code ##################

        
        # Assigning a Call to a Tuple (line 423):
        
        # Assigning a Subscript to a Name (line 423):
        
        # Obtaining the type of the subscript
        int_92799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 8), 'int')
        
        # Call to get_data(...): (line 423)
        # Processing the call keyword arguments (line 423)
        int_92802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 54), 'int')
        keyword_92803 = int_92802
        kwargs_92804 = {'x_stride': keyword_92803}
        # Getting the type of 'self' (line 423)
        self_92800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 423)
        get_data_92801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 31), self_92800, 'get_data')
        # Calling get_data(args, kwargs) (line 423)
        get_data_call_result_92805 = invoke(stypy.reporting.localization.Localization(__file__, 423, 31), get_data_92801, *[], **kwargs_92804)
        
        # Obtaining the member '__getitem__' of a type (line 423)
        getitem___92806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), get_data_call_result_92805, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
        subscript_call_result_92807 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), getitem___92806, int_92799)
        
        # Assigning a type to the variable 'tuple_var_assignment_91235' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_91235', subscript_call_result_92807)
        
        # Assigning a Subscript to a Name (line 423):
        
        # Obtaining the type of the subscript
        int_92808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 8), 'int')
        
        # Call to get_data(...): (line 423)
        # Processing the call keyword arguments (line 423)
        int_92811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 54), 'int')
        keyword_92812 = int_92811
        kwargs_92813 = {'x_stride': keyword_92812}
        # Getting the type of 'self' (line 423)
        self_92809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 423)
        get_data_92810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 31), self_92809, 'get_data')
        # Calling get_data(args, kwargs) (line 423)
        get_data_call_result_92814 = invoke(stypy.reporting.localization.Localization(__file__, 423, 31), get_data_92810, *[], **kwargs_92813)
        
        # Obtaining the member '__getitem__' of a type (line 423)
        getitem___92815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), get_data_call_result_92814, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
        subscript_call_result_92816 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), getitem___92815, int_92808)
        
        # Assigning a type to the variable 'tuple_var_assignment_91236' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_91236', subscript_call_result_92816)
        
        # Assigning a Subscript to a Name (line 423):
        
        # Obtaining the type of the subscript
        int_92817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 8), 'int')
        
        # Call to get_data(...): (line 423)
        # Processing the call keyword arguments (line 423)
        int_92820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 54), 'int')
        keyword_92821 = int_92820
        kwargs_92822 = {'x_stride': keyword_92821}
        # Getting the type of 'self' (line 423)
        self_92818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 423)
        get_data_92819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 31), self_92818, 'get_data')
        # Calling get_data(args, kwargs) (line 423)
        get_data_call_result_92823 = invoke(stypy.reporting.localization.Localization(__file__, 423, 31), get_data_92819, *[], **kwargs_92822)
        
        # Obtaining the member '__getitem__' of a type (line 423)
        getitem___92824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), get_data_call_result_92823, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
        subscript_call_result_92825 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), getitem___92824, int_92817)
        
        # Assigning a type to the variable 'tuple_var_assignment_91237' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_91237', subscript_call_result_92825)
        
        # Assigning a Subscript to a Name (line 423):
        
        # Obtaining the type of the subscript
        int_92826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 8), 'int')
        
        # Call to get_data(...): (line 423)
        # Processing the call keyword arguments (line 423)
        int_92829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 54), 'int')
        keyword_92830 = int_92829
        kwargs_92831 = {'x_stride': keyword_92830}
        # Getting the type of 'self' (line 423)
        self_92827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 423)
        get_data_92828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 31), self_92827, 'get_data')
        # Calling get_data(args, kwargs) (line 423)
        get_data_call_result_92832 = invoke(stypy.reporting.localization.Localization(__file__, 423, 31), get_data_92828, *[], **kwargs_92831)
        
        # Obtaining the member '__getitem__' of a type (line 423)
        getitem___92833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), get_data_call_result_92832, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
        subscript_call_result_92834 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), getitem___92833, int_92826)
        
        # Assigning a type to the variable 'tuple_var_assignment_91238' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_91238', subscript_call_result_92834)
        
        # Assigning a Subscript to a Name (line 423):
        
        # Obtaining the type of the subscript
        int_92835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 8), 'int')
        
        # Call to get_data(...): (line 423)
        # Processing the call keyword arguments (line 423)
        int_92838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 54), 'int')
        keyword_92839 = int_92838
        kwargs_92840 = {'x_stride': keyword_92839}
        # Getting the type of 'self' (line 423)
        self_92836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 423)
        get_data_92837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 31), self_92836, 'get_data')
        # Calling get_data(args, kwargs) (line 423)
        get_data_call_result_92841 = invoke(stypy.reporting.localization.Localization(__file__, 423, 31), get_data_92837, *[], **kwargs_92840)
        
        # Obtaining the member '__getitem__' of a type (line 423)
        getitem___92842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), get_data_call_result_92841, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
        subscript_call_result_92843 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), getitem___92842, int_92835)
        
        # Assigning a type to the variable 'tuple_var_assignment_91239' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_91239', subscript_call_result_92843)
        
        # Assigning a Name to a Name (line 423):
        # Getting the type of 'tuple_var_assignment_91235' (line 423)
        tuple_var_assignment_91235_92844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_91235')
        # Assigning a type to the variable 'alpha' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'alpha', tuple_var_assignment_91235_92844)
        
        # Assigning a Name to a Name (line 423):
        # Getting the type of 'tuple_var_assignment_91236' (line 423)
        tuple_var_assignment_91236_92845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_91236')
        # Assigning a type to the variable 'beta' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'beta', tuple_var_assignment_91236_92845)
        
        # Assigning a Name to a Name (line 423):
        # Getting the type of 'tuple_var_assignment_91237' (line 423)
        tuple_var_assignment_91237_92846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_91237')
        # Assigning a type to the variable 'a' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 21), 'a', tuple_var_assignment_91237_92846)
        
        # Assigning a Name to a Name (line 423):
        # Getting the type of 'tuple_var_assignment_91238' (line 423)
        tuple_var_assignment_91238_92847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_91238')
        # Assigning a type to the variable 'x' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 24), 'x', tuple_var_assignment_91238_92847)
        
        # Assigning a Name to a Name (line 423):
        # Getting the type of 'tuple_var_assignment_91239' (line 423)
        tuple_var_assignment_91239_92848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_91239')
        # Assigning a type to the variable 'y' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 27), 'y', tuple_var_assignment_91239_92848)
        
        # Assigning a BinOp to a Name (line 424):
        
        # Assigning a BinOp to a Name (line 424):
        # Getting the type of 'alpha' (line 424)
        alpha_92849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 20), 'alpha')
        
        # Call to matrixmultiply(...): (line 424)
        # Processing the call arguments (line 424)
        
        # Call to transpose(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'a' (line 424)
        a_92852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 51), 'a', False)
        # Processing the call keyword arguments (line 424)
        kwargs_92853 = {}
        # Getting the type of 'transpose' (line 424)
        transpose_92851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 41), 'transpose', False)
        # Calling transpose(args, kwargs) (line 424)
        transpose_call_result_92854 = invoke(stypy.reporting.localization.Localization(__file__, 424, 41), transpose_92851, *[a_92852], **kwargs_92853)
        
        
        # Obtaining the type of the subscript
        int_92855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 59), 'int')
        slice_92856 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 424, 55), None, None, int_92855)
        # Getting the type of 'x' (line 424)
        x_92857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 55), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 424)
        getitem___92858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 55), x_92857, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 424)
        subscript_call_result_92859 = invoke(stypy.reporting.localization.Localization(__file__, 424, 55), getitem___92858, slice_92856)
        
        # Processing the call keyword arguments (line 424)
        kwargs_92860 = {}
        # Getting the type of 'matrixmultiply' (line 424)
        matrixmultiply_92850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 26), 'matrixmultiply', False)
        # Calling matrixmultiply(args, kwargs) (line 424)
        matrixmultiply_call_result_92861 = invoke(stypy.reporting.localization.Localization(__file__, 424, 26), matrixmultiply_92850, *[transpose_call_result_92854, subscript_call_result_92859], **kwargs_92860)
        
        # Applying the binary operator '*' (line 424)
        result_mul_92862 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 20), '*', alpha_92849, matrixmultiply_call_result_92861)
        
        # Getting the type of 'beta' (line 424)
        beta_92863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 63), 'beta')
        # Getting the type of 'y' (line 424)
        y_92864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 68), 'y')
        # Applying the binary operator '*' (line 424)
        result_mul_92865 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 63), '*', beta_92863, y_92864)
        
        # Applying the binary operator '+' (line 424)
        result_add_92866 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 20), '+', result_mul_92862, result_mul_92865)
        
        # Assigning a type to the variable 'desired_y' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'desired_y', result_add_92866)
        
        # Assigning a Call to a Name (line 425):
        
        # Assigning a Call to a Name (line 425):
        
        # Call to blas_func(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'alpha' (line 425)
        alpha_92869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 27), 'alpha', False)
        # Getting the type of 'a' (line 425)
        a_92870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 34), 'a', False)
        # Getting the type of 'x' (line 425)
        x_92871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 37), 'x', False)
        # Getting the type of 'beta' (line 425)
        beta_92872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 40), 'beta', False)
        # Getting the type of 'y' (line 425)
        y_92873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 46), 'y', False)
        # Processing the call keyword arguments (line 425)
        int_92874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 55), 'int')
        keyword_92875 = int_92874
        int_92876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 63), 'int')
        keyword_92877 = int_92876
        kwargs_92878 = {'incx': keyword_92877, 'trans': keyword_92875}
        # Getting the type of 'self' (line 425)
        self_92867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 425)
        blas_func_92868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 12), self_92867, 'blas_func')
        # Calling blas_func(args, kwargs) (line 425)
        blas_func_call_result_92879 = invoke(stypy.reporting.localization.Localization(__file__, 425, 12), blas_func_92868, *[alpha_92869, a_92870, x_92871, beta_92872, y_92873], **kwargs_92878)
        
        # Assigning a type to the variable 'y' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'y', blas_func_call_result_92879)
        
        # Call to assert_array_almost_equal(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'desired_y' (line 426)
        desired_y_92881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 34), 'desired_y', False)
        # Getting the type of 'y' (line 426)
        y_92882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 45), 'y', False)
        # Processing the call keyword arguments (line 426)
        kwargs_92883 = {}
        # Getting the type of 'assert_array_almost_equal' (line 426)
        assert_array_almost_equal_92880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 426)
        assert_array_almost_equal_call_result_92884 = invoke(stypy.reporting.localization.Localization(__file__, 426, 8), assert_array_almost_equal_92880, *[desired_y_92881, y_92882], **kwargs_92883)
        
        
        # ################# End of 'test_x_stride_transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_stride_transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_92885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92885)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_stride_transpose'
        return stypy_return_type_92885


    @norecursion
    def test_x_stride_assert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_stride_assert'
        module_type_store = module_type_store.open_function_context('test_x_stride_assert', 428, 4, False)
        # Assigning a type to the variable 'self' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseGemv.test_x_stride_assert.__dict__.__setitem__('stypy_localization', localization)
        BaseGemv.test_x_stride_assert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseGemv.test_x_stride_assert.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseGemv.test_x_stride_assert.__dict__.__setitem__('stypy_function_name', 'BaseGemv.test_x_stride_assert')
        BaseGemv.test_x_stride_assert.__dict__.__setitem__('stypy_param_names_list', [])
        BaseGemv.test_x_stride_assert.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseGemv.test_x_stride_assert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseGemv.test_x_stride_assert.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseGemv.test_x_stride_assert.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseGemv.test_x_stride_assert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseGemv.test_x_stride_assert.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.test_x_stride_assert', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_stride_assert', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_stride_assert(...)' code ##################

        
        # Assigning a Call to a Tuple (line 430):
        
        # Assigning a Subscript to a Name (line 430):
        
        # Obtaining the type of the subscript
        int_92886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 8), 'int')
        
        # Call to get_data(...): (line 430)
        # Processing the call keyword arguments (line 430)
        int_92889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 54), 'int')
        keyword_92890 = int_92889
        kwargs_92891 = {'x_stride': keyword_92890}
        # Getting the type of 'self' (line 430)
        self_92887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 430)
        get_data_92888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 31), self_92887, 'get_data')
        # Calling get_data(args, kwargs) (line 430)
        get_data_call_result_92892 = invoke(stypy.reporting.localization.Localization(__file__, 430, 31), get_data_92888, *[], **kwargs_92891)
        
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___92893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), get_data_call_result_92892, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_92894 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), getitem___92893, int_92886)
        
        # Assigning a type to the variable 'tuple_var_assignment_91240' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tuple_var_assignment_91240', subscript_call_result_92894)
        
        # Assigning a Subscript to a Name (line 430):
        
        # Obtaining the type of the subscript
        int_92895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 8), 'int')
        
        # Call to get_data(...): (line 430)
        # Processing the call keyword arguments (line 430)
        int_92898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 54), 'int')
        keyword_92899 = int_92898
        kwargs_92900 = {'x_stride': keyword_92899}
        # Getting the type of 'self' (line 430)
        self_92896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 430)
        get_data_92897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 31), self_92896, 'get_data')
        # Calling get_data(args, kwargs) (line 430)
        get_data_call_result_92901 = invoke(stypy.reporting.localization.Localization(__file__, 430, 31), get_data_92897, *[], **kwargs_92900)
        
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___92902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), get_data_call_result_92901, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_92903 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), getitem___92902, int_92895)
        
        # Assigning a type to the variable 'tuple_var_assignment_91241' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tuple_var_assignment_91241', subscript_call_result_92903)
        
        # Assigning a Subscript to a Name (line 430):
        
        # Obtaining the type of the subscript
        int_92904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 8), 'int')
        
        # Call to get_data(...): (line 430)
        # Processing the call keyword arguments (line 430)
        int_92907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 54), 'int')
        keyword_92908 = int_92907
        kwargs_92909 = {'x_stride': keyword_92908}
        # Getting the type of 'self' (line 430)
        self_92905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 430)
        get_data_92906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 31), self_92905, 'get_data')
        # Calling get_data(args, kwargs) (line 430)
        get_data_call_result_92910 = invoke(stypy.reporting.localization.Localization(__file__, 430, 31), get_data_92906, *[], **kwargs_92909)
        
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___92911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), get_data_call_result_92910, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_92912 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), getitem___92911, int_92904)
        
        # Assigning a type to the variable 'tuple_var_assignment_91242' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tuple_var_assignment_91242', subscript_call_result_92912)
        
        # Assigning a Subscript to a Name (line 430):
        
        # Obtaining the type of the subscript
        int_92913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 8), 'int')
        
        # Call to get_data(...): (line 430)
        # Processing the call keyword arguments (line 430)
        int_92916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 54), 'int')
        keyword_92917 = int_92916
        kwargs_92918 = {'x_stride': keyword_92917}
        # Getting the type of 'self' (line 430)
        self_92914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 430)
        get_data_92915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 31), self_92914, 'get_data')
        # Calling get_data(args, kwargs) (line 430)
        get_data_call_result_92919 = invoke(stypy.reporting.localization.Localization(__file__, 430, 31), get_data_92915, *[], **kwargs_92918)
        
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___92920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), get_data_call_result_92919, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_92921 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), getitem___92920, int_92913)
        
        # Assigning a type to the variable 'tuple_var_assignment_91243' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tuple_var_assignment_91243', subscript_call_result_92921)
        
        # Assigning a Subscript to a Name (line 430):
        
        # Obtaining the type of the subscript
        int_92922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 8), 'int')
        
        # Call to get_data(...): (line 430)
        # Processing the call keyword arguments (line 430)
        int_92925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 54), 'int')
        keyword_92926 = int_92925
        kwargs_92927 = {'x_stride': keyword_92926}
        # Getting the type of 'self' (line 430)
        self_92923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 430)
        get_data_92924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 31), self_92923, 'get_data')
        # Calling get_data(args, kwargs) (line 430)
        get_data_call_result_92928 = invoke(stypy.reporting.localization.Localization(__file__, 430, 31), get_data_92924, *[], **kwargs_92927)
        
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___92929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), get_data_call_result_92928, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_92930 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), getitem___92929, int_92922)
        
        # Assigning a type to the variable 'tuple_var_assignment_91244' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tuple_var_assignment_91244', subscript_call_result_92930)
        
        # Assigning a Name to a Name (line 430):
        # Getting the type of 'tuple_var_assignment_91240' (line 430)
        tuple_var_assignment_91240_92931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tuple_var_assignment_91240')
        # Assigning a type to the variable 'alpha' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'alpha', tuple_var_assignment_91240_92931)
        
        # Assigning a Name to a Name (line 430):
        # Getting the type of 'tuple_var_assignment_91241' (line 430)
        tuple_var_assignment_91241_92932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tuple_var_assignment_91241')
        # Assigning a type to the variable 'beta' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 15), 'beta', tuple_var_assignment_91241_92932)
        
        # Assigning a Name to a Name (line 430):
        # Getting the type of 'tuple_var_assignment_91242' (line 430)
        tuple_var_assignment_91242_92933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tuple_var_assignment_91242')
        # Assigning a type to the variable 'a' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 21), 'a', tuple_var_assignment_91242_92933)
        
        # Assigning a Name to a Name (line 430):
        # Getting the type of 'tuple_var_assignment_91243' (line 430)
        tuple_var_assignment_91243_92934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tuple_var_assignment_91243')
        # Assigning a type to the variable 'x' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 24), 'x', tuple_var_assignment_91243_92934)
        
        # Assigning a Name to a Name (line 430):
        # Getting the type of 'tuple_var_assignment_91244' (line 430)
        tuple_var_assignment_91244_92935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tuple_var_assignment_91244')
        # Assigning a type to the variable 'y' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 27), 'y', tuple_var_assignment_91244_92935)
        
        
        # SSA begins for try-except statement (line 431)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 432):
        
        # Assigning a Call to a Name (line 432):
        
        # Call to blas_func(...): (line 432)
        # Processing the call arguments (line 432)
        int_92938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 31), 'int')
        # Getting the type of 'a' (line 432)
        a_92939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 34), 'a', False)
        # Getting the type of 'x' (line 432)
        x_92940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 37), 'x', False)
        int_92941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 40), 'int')
        # Getting the type of 'y' (line 432)
        y_92942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 43), 'y', False)
        # Processing the call keyword arguments (line 432)
        int_92943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 52), 'int')
        keyword_92944 = int_92943
        int_92945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 60), 'int')
        keyword_92946 = int_92945
        kwargs_92947 = {'incx': keyword_92946, 'trans': keyword_92944}
        # Getting the type of 'self' (line 432)
        self_92936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 16), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 432)
        blas_func_92937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 16), self_92936, 'blas_func')
        # Calling blas_func(args, kwargs) (line 432)
        blas_func_call_result_92948 = invoke(stypy.reporting.localization.Localization(__file__, 432, 16), blas_func_92937, *[int_92938, a_92939, x_92940, int_92941, y_92942], **kwargs_92947)
        
        # Assigning a type to the variable 'y' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'y', blas_func_call_result_92948)
        
        # Call to assert_(...): (line 433)
        # Processing the call arguments (line 433)
        int_92950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 20), 'int')
        # Processing the call keyword arguments (line 433)
        kwargs_92951 = {}
        # Getting the type of 'assert_' (line 433)
        assert__92949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 433)
        assert__call_result_92952 = invoke(stypy.reporting.localization.Localization(__file__, 433, 12), assert__92949, *[int_92950], **kwargs_92951)
        
        # SSA branch for the except part of a try statement (line 431)
        # SSA branch for the except '<any exception>' branch of a try statement (line 431)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 431)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 436)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 437):
        
        # Assigning a Call to a Name (line 437):
        
        # Call to blas_func(...): (line 437)
        # Processing the call arguments (line 437)
        int_92955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 31), 'int')
        # Getting the type of 'a' (line 437)
        a_92956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 34), 'a', False)
        # Getting the type of 'x' (line 437)
        x_92957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 37), 'x', False)
        int_92958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 40), 'int')
        # Getting the type of 'y' (line 437)
        y_92959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 43), 'y', False)
        # Processing the call keyword arguments (line 437)
        int_92960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 52), 'int')
        keyword_92961 = int_92960
        int_92962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 60), 'int')
        keyword_92963 = int_92962
        kwargs_92964 = {'incx': keyword_92963, 'trans': keyword_92961}
        # Getting the type of 'self' (line 437)
        self_92953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 16), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 437)
        blas_func_92954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 16), self_92953, 'blas_func')
        # Calling blas_func(args, kwargs) (line 437)
        blas_func_call_result_92965 = invoke(stypy.reporting.localization.Localization(__file__, 437, 16), blas_func_92954, *[int_92955, a_92956, x_92957, int_92958, y_92959], **kwargs_92964)
        
        # Assigning a type to the variable 'y' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'y', blas_func_call_result_92965)
        
        # Call to assert_(...): (line 438)
        # Processing the call arguments (line 438)
        int_92967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 20), 'int')
        # Processing the call keyword arguments (line 438)
        kwargs_92968 = {}
        # Getting the type of 'assert_' (line 438)
        assert__92966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 438)
        assert__call_result_92969 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), assert__92966, *[int_92967], **kwargs_92968)
        
        # SSA branch for the except part of a try statement (line 436)
        # SSA branch for the except '<any exception>' branch of a try statement (line 436)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 436)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_x_stride_assert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_stride_assert' in the type store
        # Getting the type of 'stypy_return_type' (line 428)
        stypy_return_type_92970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_92970)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_stride_assert'
        return stypy_return_type_92970


    @norecursion
    def test_y_stride(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_y_stride'
        module_type_store = module_type_store.open_function_context('test_y_stride', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseGemv.test_y_stride.__dict__.__setitem__('stypy_localization', localization)
        BaseGemv.test_y_stride.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseGemv.test_y_stride.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseGemv.test_y_stride.__dict__.__setitem__('stypy_function_name', 'BaseGemv.test_y_stride')
        BaseGemv.test_y_stride.__dict__.__setitem__('stypy_param_names_list', [])
        BaseGemv.test_y_stride.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseGemv.test_y_stride.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseGemv.test_y_stride.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseGemv.test_y_stride.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseGemv.test_y_stride.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseGemv.test_y_stride.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.test_y_stride', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_y_stride', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_y_stride(...)' code ##################

        
        # Assigning a Call to a Tuple (line 443):
        
        # Assigning a Subscript to a Name (line 443):
        
        # Obtaining the type of the subscript
        int_92971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 8), 'int')
        
        # Call to get_data(...): (line 443)
        # Processing the call keyword arguments (line 443)
        int_92974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 54), 'int')
        keyword_92975 = int_92974
        kwargs_92976 = {'y_stride': keyword_92975}
        # Getting the type of 'self' (line 443)
        self_92972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 443)
        get_data_92973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 31), self_92972, 'get_data')
        # Calling get_data(args, kwargs) (line 443)
        get_data_call_result_92977 = invoke(stypy.reporting.localization.Localization(__file__, 443, 31), get_data_92973, *[], **kwargs_92976)
        
        # Obtaining the member '__getitem__' of a type (line 443)
        getitem___92978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), get_data_call_result_92977, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 443)
        subscript_call_result_92979 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), getitem___92978, int_92971)
        
        # Assigning a type to the variable 'tuple_var_assignment_91245' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_91245', subscript_call_result_92979)
        
        # Assigning a Subscript to a Name (line 443):
        
        # Obtaining the type of the subscript
        int_92980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 8), 'int')
        
        # Call to get_data(...): (line 443)
        # Processing the call keyword arguments (line 443)
        int_92983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 54), 'int')
        keyword_92984 = int_92983
        kwargs_92985 = {'y_stride': keyword_92984}
        # Getting the type of 'self' (line 443)
        self_92981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 443)
        get_data_92982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 31), self_92981, 'get_data')
        # Calling get_data(args, kwargs) (line 443)
        get_data_call_result_92986 = invoke(stypy.reporting.localization.Localization(__file__, 443, 31), get_data_92982, *[], **kwargs_92985)
        
        # Obtaining the member '__getitem__' of a type (line 443)
        getitem___92987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), get_data_call_result_92986, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 443)
        subscript_call_result_92988 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), getitem___92987, int_92980)
        
        # Assigning a type to the variable 'tuple_var_assignment_91246' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_91246', subscript_call_result_92988)
        
        # Assigning a Subscript to a Name (line 443):
        
        # Obtaining the type of the subscript
        int_92989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 8), 'int')
        
        # Call to get_data(...): (line 443)
        # Processing the call keyword arguments (line 443)
        int_92992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 54), 'int')
        keyword_92993 = int_92992
        kwargs_92994 = {'y_stride': keyword_92993}
        # Getting the type of 'self' (line 443)
        self_92990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 443)
        get_data_92991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 31), self_92990, 'get_data')
        # Calling get_data(args, kwargs) (line 443)
        get_data_call_result_92995 = invoke(stypy.reporting.localization.Localization(__file__, 443, 31), get_data_92991, *[], **kwargs_92994)
        
        # Obtaining the member '__getitem__' of a type (line 443)
        getitem___92996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), get_data_call_result_92995, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 443)
        subscript_call_result_92997 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), getitem___92996, int_92989)
        
        # Assigning a type to the variable 'tuple_var_assignment_91247' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_91247', subscript_call_result_92997)
        
        # Assigning a Subscript to a Name (line 443):
        
        # Obtaining the type of the subscript
        int_92998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 8), 'int')
        
        # Call to get_data(...): (line 443)
        # Processing the call keyword arguments (line 443)
        int_93001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 54), 'int')
        keyword_93002 = int_93001
        kwargs_93003 = {'y_stride': keyword_93002}
        # Getting the type of 'self' (line 443)
        self_92999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 443)
        get_data_93000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 31), self_92999, 'get_data')
        # Calling get_data(args, kwargs) (line 443)
        get_data_call_result_93004 = invoke(stypy.reporting.localization.Localization(__file__, 443, 31), get_data_93000, *[], **kwargs_93003)
        
        # Obtaining the member '__getitem__' of a type (line 443)
        getitem___93005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), get_data_call_result_93004, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 443)
        subscript_call_result_93006 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), getitem___93005, int_92998)
        
        # Assigning a type to the variable 'tuple_var_assignment_91248' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_91248', subscript_call_result_93006)
        
        # Assigning a Subscript to a Name (line 443):
        
        # Obtaining the type of the subscript
        int_93007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 8), 'int')
        
        # Call to get_data(...): (line 443)
        # Processing the call keyword arguments (line 443)
        int_93010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 54), 'int')
        keyword_93011 = int_93010
        kwargs_93012 = {'y_stride': keyword_93011}
        # Getting the type of 'self' (line 443)
        self_93008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 443)
        get_data_93009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 31), self_93008, 'get_data')
        # Calling get_data(args, kwargs) (line 443)
        get_data_call_result_93013 = invoke(stypy.reporting.localization.Localization(__file__, 443, 31), get_data_93009, *[], **kwargs_93012)
        
        # Obtaining the member '__getitem__' of a type (line 443)
        getitem___93014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), get_data_call_result_93013, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 443)
        subscript_call_result_93015 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), getitem___93014, int_93007)
        
        # Assigning a type to the variable 'tuple_var_assignment_91249' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_91249', subscript_call_result_93015)
        
        # Assigning a Name to a Name (line 443):
        # Getting the type of 'tuple_var_assignment_91245' (line 443)
        tuple_var_assignment_91245_93016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_91245')
        # Assigning a type to the variable 'alpha' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'alpha', tuple_var_assignment_91245_93016)
        
        # Assigning a Name to a Name (line 443):
        # Getting the type of 'tuple_var_assignment_91246' (line 443)
        tuple_var_assignment_91246_93017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_91246')
        # Assigning a type to the variable 'beta' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 15), 'beta', tuple_var_assignment_91246_93017)
        
        # Assigning a Name to a Name (line 443):
        # Getting the type of 'tuple_var_assignment_91247' (line 443)
        tuple_var_assignment_91247_93018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_91247')
        # Assigning a type to the variable 'a' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 21), 'a', tuple_var_assignment_91247_93018)
        
        # Assigning a Name to a Name (line 443):
        # Getting the type of 'tuple_var_assignment_91248' (line 443)
        tuple_var_assignment_91248_93019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_91248')
        # Assigning a type to the variable 'x' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 24), 'x', tuple_var_assignment_91248_93019)
        
        # Assigning a Name to a Name (line 443):
        # Getting the type of 'tuple_var_assignment_91249' (line 443)
        tuple_var_assignment_91249_93020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'tuple_var_assignment_91249')
        # Assigning a type to the variable 'y' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 27), 'y', tuple_var_assignment_91249_93020)
        
        # Assigning a Call to a Name (line 444):
        
        # Assigning a Call to a Name (line 444):
        
        # Call to copy(...): (line 444)
        # Processing the call keyword arguments (line 444)
        kwargs_93023 = {}
        # Getting the type of 'y' (line 444)
        y_93021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'y', False)
        # Obtaining the member 'copy' of a type (line 444)
        copy_93022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 20), y_93021, 'copy')
        # Calling copy(args, kwargs) (line 444)
        copy_call_result_93024 = invoke(stypy.reporting.localization.Localization(__file__, 444, 20), copy_93022, *[], **kwargs_93023)
        
        # Assigning a type to the variable 'desired_y' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'desired_y', copy_call_result_93024)
        
        # Assigning a BinOp to a Subscript (line 445):
        
        # Assigning a BinOp to a Subscript (line 445):
        # Getting the type of 'alpha' (line 445)
        alpha_93025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 25), 'alpha')
        
        # Call to matrixmultiply(...): (line 445)
        # Processing the call arguments (line 445)
        # Getting the type of 'a' (line 445)
        a_93027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 46), 'a', False)
        # Getting the type of 'x' (line 445)
        x_93028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 49), 'x', False)
        # Processing the call keyword arguments (line 445)
        kwargs_93029 = {}
        # Getting the type of 'matrixmultiply' (line 445)
        matrixmultiply_93026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 31), 'matrixmultiply', False)
        # Calling matrixmultiply(args, kwargs) (line 445)
        matrixmultiply_call_result_93030 = invoke(stypy.reporting.localization.Localization(__file__, 445, 31), matrixmultiply_93026, *[a_93027, x_93028], **kwargs_93029)
        
        # Applying the binary operator '*' (line 445)
        result_mul_93031 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 25), '*', alpha_93025, matrixmultiply_call_result_93030)
        
        # Getting the type of 'beta' (line 445)
        beta_93032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 52), 'beta')
        
        # Obtaining the type of the subscript
        int_93033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 61), 'int')
        slice_93034 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 445, 57), None, None, int_93033)
        # Getting the type of 'y' (line 445)
        y_93035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 57), 'y')
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___93036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 57), y_93035, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_93037 = invoke(stypy.reporting.localization.Localization(__file__, 445, 57), getitem___93036, slice_93034)
        
        # Applying the binary operator '*' (line 445)
        result_mul_93038 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 52), '*', beta_93032, subscript_call_result_93037)
        
        # Applying the binary operator '+' (line 445)
        result_add_93039 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 25), '+', result_mul_93031, result_mul_93038)
        
        # Getting the type of 'desired_y' (line 445)
        desired_y_93040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'desired_y')
        int_93041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 20), 'int')
        slice_93042 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 445, 8), None, None, int_93041)
        # Storing an element on a container (line 445)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 8), desired_y_93040, (slice_93042, result_add_93039))
        
        # Assigning a Call to a Name (line 446):
        
        # Assigning a Call to a Name (line 446):
        
        # Call to blas_func(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'alpha' (line 446)
        alpha_93045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 27), 'alpha', False)
        # Getting the type of 'a' (line 446)
        a_93046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 34), 'a', False)
        # Getting the type of 'x' (line 446)
        x_93047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 37), 'x', False)
        # Getting the type of 'beta' (line 446)
        beta_93048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 40), 'beta', False)
        # Getting the type of 'y' (line 446)
        y_93049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 46), 'y', False)
        # Processing the call keyword arguments (line 446)
        int_93050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 54), 'int')
        keyword_93051 = int_93050
        kwargs_93052 = {'incy': keyword_93051}
        # Getting the type of 'self' (line 446)
        self_93043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 446)
        blas_func_93044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 12), self_93043, 'blas_func')
        # Calling blas_func(args, kwargs) (line 446)
        blas_func_call_result_93053 = invoke(stypy.reporting.localization.Localization(__file__, 446, 12), blas_func_93044, *[alpha_93045, a_93046, x_93047, beta_93048, y_93049], **kwargs_93052)
        
        # Assigning a type to the variable 'y' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'y', blas_func_call_result_93053)
        
        # Call to assert_array_almost_equal(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'desired_y' (line 447)
        desired_y_93055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 34), 'desired_y', False)
        # Getting the type of 'y' (line 447)
        y_93056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 45), 'y', False)
        # Processing the call keyword arguments (line 447)
        kwargs_93057 = {}
        # Getting the type of 'assert_array_almost_equal' (line 447)
        assert_array_almost_equal_93054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 447)
        assert_array_almost_equal_call_result_93058 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), assert_array_almost_equal_93054, *[desired_y_93055, y_93056], **kwargs_93057)
        
        
        # ################# End of 'test_y_stride(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_y_stride' in the type store
        # Getting the type of 'stypy_return_type' (line 442)
        stypy_return_type_93059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_93059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_y_stride'
        return stypy_return_type_93059


    @norecursion
    def test_y_stride_transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_y_stride_transpose'
        module_type_store = module_type_store.open_function_context('test_y_stride_transpose', 449, 4, False)
        # Assigning a type to the variable 'self' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseGemv.test_y_stride_transpose.__dict__.__setitem__('stypy_localization', localization)
        BaseGemv.test_y_stride_transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseGemv.test_y_stride_transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseGemv.test_y_stride_transpose.__dict__.__setitem__('stypy_function_name', 'BaseGemv.test_y_stride_transpose')
        BaseGemv.test_y_stride_transpose.__dict__.__setitem__('stypy_param_names_list', [])
        BaseGemv.test_y_stride_transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseGemv.test_y_stride_transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseGemv.test_y_stride_transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseGemv.test_y_stride_transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseGemv.test_y_stride_transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseGemv.test_y_stride_transpose.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.test_y_stride_transpose', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_y_stride_transpose', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_y_stride_transpose(...)' code ##################

        
        # Assigning a Call to a Tuple (line 450):
        
        # Assigning a Subscript to a Name (line 450):
        
        # Obtaining the type of the subscript
        int_93060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 8), 'int')
        
        # Call to get_data(...): (line 450)
        # Processing the call keyword arguments (line 450)
        int_93063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 54), 'int')
        keyword_93064 = int_93063
        kwargs_93065 = {'y_stride': keyword_93064}
        # Getting the type of 'self' (line 450)
        self_93061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 450)
        get_data_93062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 31), self_93061, 'get_data')
        # Calling get_data(args, kwargs) (line 450)
        get_data_call_result_93066 = invoke(stypy.reporting.localization.Localization(__file__, 450, 31), get_data_93062, *[], **kwargs_93065)
        
        # Obtaining the member '__getitem__' of a type (line 450)
        getitem___93067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), get_data_call_result_93066, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 450)
        subscript_call_result_93068 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), getitem___93067, int_93060)
        
        # Assigning a type to the variable 'tuple_var_assignment_91250' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_var_assignment_91250', subscript_call_result_93068)
        
        # Assigning a Subscript to a Name (line 450):
        
        # Obtaining the type of the subscript
        int_93069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 8), 'int')
        
        # Call to get_data(...): (line 450)
        # Processing the call keyword arguments (line 450)
        int_93072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 54), 'int')
        keyword_93073 = int_93072
        kwargs_93074 = {'y_stride': keyword_93073}
        # Getting the type of 'self' (line 450)
        self_93070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 450)
        get_data_93071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 31), self_93070, 'get_data')
        # Calling get_data(args, kwargs) (line 450)
        get_data_call_result_93075 = invoke(stypy.reporting.localization.Localization(__file__, 450, 31), get_data_93071, *[], **kwargs_93074)
        
        # Obtaining the member '__getitem__' of a type (line 450)
        getitem___93076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), get_data_call_result_93075, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 450)
        subscript_call_result_93077 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), getitem___93076, int_93069)
        
        # Assigning a type to the variable 'tuple_var_assignment_91251' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_var_assignment_91251', subscript_call_result_93077)
        
        # Assigning a Subscript to a Name (line 450):
        
        # Obtaining the type of the subscript
        int_93078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 8), 'int')
        
        # Call to get_data(...): (line 450)
        # Processing the call keyword arguments (line 450)
        int_93081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 54), 'int')
        keyword_93082 = int_93081
        kwargs_93083 = {'y_stride': keyword_93082}
        # Getting the type of 'self' (line 450)
        self_93079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 450)
        get_data_93080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 31), self_93079, 'get_data')
        # Calling get_data(args, kwargs) (line 450)
        get_data_call_result_93084 = invoke(stypy.reporting.localization.Localization(__file__, 450, 31), get_data_93080, *[], **kwargs_93083)
        
        # Obtaining the member '__getitem__' of a type (line 450)
        getitem___93085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), get_data_call_result_93084, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 450)
        subscript_call_result_93086 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), getitem___93085, int_93078)
        
        # Assigning a type to the variable 'tuple_var_assignment_91252' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_var_assignment_91252', subscript_call_result_93086)
        
        # Assigning a Subscript to a Name (line 450):
        
        # Obtaining the type of the subscript
        int_93087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 8), 'int')
        
        # Call to get_data(...): (line 450)
        # Processing the call keyword arguments (line 450)
        int_93090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 54), 'int')
        keyword_93091 = int_93090
        kwargs_93092 = {'y_stride': keyword_93091}
        # Getting the type of 'self' (line 450)
        self_93088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 450)
        get_data_93089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 31), self_93088, 'get_data')
        # Calling get_data(args, kwargs) (line 450)
        get_data_call_result_93093 = invoke(stypy.reporting.localization.Localization(__file__, 450, 31), get_data_93089, *[], **kwargs_93092)
        
        # Obtaining the member '__getitem__' of a type (line 450)
        getitem___93094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), get_data_call_result_93093, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 450)
        subscript_call_result_93095 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), getitem___93094, int_93087)
        
        # Assigning a type to the variable 'tuple_var_assignment_91253' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_var_assignment_91253', subscript_call_result_93095)
        
        # Assigning a Subscript to a Name (line 450):
        
        # Obtaining the type of the subscript
        int_93096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 8), 'int')
        
        # Call to get_data(...): (line 450)
        # Processing the call keyword arguments (line 450)
        int_93099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 54), 'int')
        keyword_93100 = int_93099
        kwargs_93101 = {'y_stride': keyword_93100}
        # Getting the type of 'self' (line 450)
        self_93097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 450)
        get_data_93098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 31), self_93097, 'get_data')
        # Calling get_data(args, kwargs) (line 450)
        get_data_call_result_93102 = invoke(stypy.reporting.localization.Localization(__file__, 450, 31), get_data_93098, *[], **kwargs_93101)
        
        # Obtaining the member '__getitem__' of a type (line 450)
        getitem___93103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), get_data_call_result_93102, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 450)
        subscript_call_result_93104 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), getitem___93103, int_93096)
        
        # Assigning a type to the variable 'tuple_var_assignment_91254' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_var_assignment_91254', subscript_call_result_93104)
        
        # Assigning a Name to a Name (line 450):
        # Getting the type of 'tuple_var_assignment_91250' (line 450)
        tuple_var_assignment_91250_93105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_var_assignment_91250')
        # Assigning a type to the variable 'alpha' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'alpha', tuple_var_assignment_91250_93105)
        
        # Assigning a Name to a Name (line 450):
        # Getting the type of 'tuple_var_assignment_91251' (line 450)
        tuple_var_assignment_91251_93106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_var_assignment_91251')
        # Assigning a type to the variable 'beta' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 15), 'beta', tuple_var_assignment_91251_93106)
        
        # Assigning a Name to a Name (line 450):
        # Getting the type of 'tuple_var_assignment_91252' (line 450)
        tuple_var_assignment_91252_93107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_var_assignment_91252')
        # Assigning a type to the variable 'a' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 21), 'a', tuple_var_assignment_91252_93107)
        
        # Assigning a Name to a Name (line 450):
        # Getting the type of 'tuple_var_assignment_91253' (line 450)
        tuple_var_assignment_91253_93108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_var_assignment_91253')
        # Assigning a type to the variable 'x' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 24), 'x', tuple_var_assignment_91253_93108)
        
        # Assigning a Name to a Name (line 450):
        # Getting the type of 'tuple_var_assignment_91254' (line 450)
        tuple_var_assignment_91254_93109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'tuple_var_assignment_91254')
        # Assigning a type to the variable 'y' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 27), 'y', tuple_var_assignment_91254_93109)
        
        # Assigning a Call to a Name (line 451):
        
        # Assigning a Call to a Name (line 451):
        
        # Call to copy(...): (line 451)
        # Processing the call keyword arguments (line 451)
        kwargs_93112 = {}
        # Getting the type of 'y' (line 451)
        y_93110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 20), 'y', False)
        # Obtaining the member 'copy' of a type (line 451)
        copy_93111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 20), y_93110, 'copy')
        # Calling copy(args, kwargs) (line 451)
        copy_call_result_93113 = invoke(stypy.reporting.localization.Localization(__file__, 451, 20), copy_93111, *[], **kwargs_93112)
        
        # Assigning a type to the variable 'desired_y' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'desired_y', copy_call_result_93113)
        
        # Assigning a BinOp to a Subscript (line 452):
        
        # Assigning a BinOp to a Subscript (line 452):
        # Getting the type of 'alpha' (line 452)
        alpha_93114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 25), 'alpha')
        
        # Call to matrixmultiply(...): (line 452)
        # Processing the call arguments (line 452)
        
        # Call to transpose(...): (line 452)
        # Processing the call arguments (line 452)
        # Getting the type of 'a' (line 452)
        a_93117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 56), 'a', False)
        # Processing the call keyword arguments (line 452)
        kwargs_93118 = {}
        # Getting the type of 'transpose' (line 452)
        transpose_93116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 46), 'transpose', False)
        # Calling transpose(args, kwargs) (line 452)
        transpose_call_result_93119 = invoke(stypy.reporting.localization.Localization(__file__, 452, 46), transpose_93116, *[a_93117], **kwargs_93118)
        
        # Getting the type of 'x' (line 452)
        x_93120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 60), 'x', False)
        # Processing the call keyword arguments (line 452)
        kwargs_93121 = {}
        # Getting the type of 'matrixmultiply' (line 452)
        matrixmultiply_93115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 31), 'matrixmultiply', False)
        # Calling matrixmultiply(args, kwargs) (line 452)
        matrixmultiply_call_result_93122 = invoke(stypy.reporting.localization.Localization(__file__, 452, 31), matrixmultiply_93115, *[transpose_call_result_93119, x_93120], **kwargs_93121)
        
        # Applying the binary operator '*' (line 452)
        result_mul_93123 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 25), '*', alpha_93114, matrixmultiply_call_result_93122)
        
        # Getting the type of 'beta' (line 452)
        beta_93124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 63), 'beta')
        
        # Obtaining the type of the subscript
        int_93125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 72), 'int')
        slice_93126 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 452, 68), None, None, int_93125)
        # Getting the type of 'y' (line 452)
        y_93127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 68), 'y')
        # Obtaining the member '__getitem__' of a type (line 452)
        getitem___93128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 68), y_93127, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 452)
        subscript_call_result_93129 = invoke(stypy.reporting.localization.Localization(__file__, 452, 68), getitem___93128, slice_93126)
        
        # Applying the binary operator '*' (line 452)
        result_mul_93130 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 63), '*', beta_93124, subscript_call_result_93129)
        
        # Applying the binary operator '+' (line 452)
        result_add_93131 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 25), '+', result_mul_93123, result_mul_93130)
        
        # Getting the type of 'desired_y' (line 452)
        desired_y_93132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'desired_y')
        int_93133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 20), 'int')
        slice_93134 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 452, 8), None, None, int_93133)
        # Storing an element on a container (line 452)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 8), desired_y_93132, (slice_93134, result_add_93131))
        
        # Assigning a Call to a Name (line 453):
        
        # Assigning a Call to a Name (line 453):
        
        # Call to blas_func(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'alpha' (line 453)
        alpha_93137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 27), 'alpha', False)
        # Getting the type of 'a' (line 453)
        a_93138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 34), 'a', False)
        # Getting the type of 'x' (line 453)
        x_93139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 37), 'x', False)
        # Getting the type of 'beta' (line 453)
        beta_93140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 40), 'beta', False)
        # Getting the type of 'y' (line 453)
        y_93141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 46), 'y', False)
        # Processing the call keyword arguments (line 453)
        int_93142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 55), 'int')
        keyword_93143 = int_93142
        int_93144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 63), 'int')
        keyword_93145 = int_93144
        kwargs_93146 = {'incy': keyword_93145, 'trans': keyword_93143}
        # Getting the type of 'self' (line 453)
        self_93135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 453)
        blas_func_93136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 12), self_93135, 'blas_func')
        # Calling blas_func(args, kwargs) (line 453)
        blas_func_call_result_93147 = invoke(stypy.reporting.localization.Localization(__file__, 453, 12), blas_func_93136, *[alpha_93137, a_93138, x_93139, beta_93140, y_93141], **kwargs_93146)
        
        # Assigning a type to the variable 'y' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'y', blas_func_call_result_93147)
        
        # Call to assert_array_almost_equal(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of 'desired_y' (line 454)
        desired_y_93149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 34), 'desired_y', False)
        # Getting the type of 'y' (line 454)
        y_93150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 45), 'y', False)
        # Processing the call keyword arguments (line 454)
        kwargs_93151 = {}
        # Getting the type of 'assert_array_almost_equal' (line 454)
        assert_array_almost_equal_93148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 454)
        assert_array_almost_equal_call_result_93152 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), assert_array_almost_equal_93148, *[desired_y_93149, y_93150], **kwargs_93151)
        
        
        # ################# End of 'test_y_stride_transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_y_stride_transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 449)
        stypy_return_type_93153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_93153)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_y_stride_transpose'
        return stypy_return_type_93153


    @norecursion
    def test_y_stride_assert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_y_stride_assert'
        module_type_store = module_type_store.open_function_context('test_y_stride_assert', 456, 4, False)
        # Assigning a type to the variable 'self' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseGemv.test_y_stride_assert.__dict__.__setitem__('stypy_localization', localization)
        BaseGemv.test_y_stride_assert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseGemv.test_y_stride_assert.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseGemv.test_y_stride_assert.__dict__.__setitem__('stypy_function_name', 'BaseGemv.test_y_stride_assert')
        BaseGemv.test_y_stride_assert.__dict__.__setitem__('stypy_param_names_list', [])
        BaseGemv.test_y_stride_assert.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseGemv.test_y_stride_assert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseGemv.test_y_stride_assert.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseGemv.test_y_stride_assert.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseGemv.test_y_stride_assert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseGemv.test_y_stride_assert.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.test_y_stride_assert', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_y_stride_assert', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_y_stride_assert(...)' code ##################

        
        # Assigning a Call to a Tuple (line 458):
        
        # Assigning a Subscript to a Name (line 458):
        
        # Obtaining the type of the subscript
        int_93154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 8), 'int')
        
        # Call to get_data(...): (line 458)
        # Processing the call keyword arguments (line 458)
        int_93157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 54), 'int')
        keyword_93158 = int_93157
        kwargs_93159 = {'y_stride': keyword_93158}
        # Getting the type of 'self' (line 458)
        self_93155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 458)
        get_data_93156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 31), self_93155, 'get_data')
        # Calling get_data(args, kwargs) (line 458)
        get_data_call_result_93160 = invoke(stypy.reporting.localization.Localization(__file__, 458, 31), get_data_93156, *[], **kwargs_93159)
        
        # Obtaining the member '__getitem__' of a type (line 458)
        getitem___93161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), get_data_call_result_93160, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 458)
        subscript_call_result_93162 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), getitem___93161, int_93154)
        
        # Assigning a type to the variable 'tuple_var_assignment_91255' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_91255', subscript_call_result_93162)
        
        # Assigning a Subscript to a Name (line 458):
        
        # Obtaining the type of the subscript
        int_93163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 8), 'int')
        
        # Call to get_data(...): (line 458)
        # Processing the call keyword arguments (line 458)
        int_93166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 54), 'int')
        keyword_93167 = int_93166
        kwargs_93168 = {'y_stride': keyword_93167}
        # Getting the type of 'self' (line 458)
        self_93164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 458)
        get_data_93165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 31), self_93164, 'get_data')
        # Calling get_data(args, kwargs) (line 458)
        get_data_call_result_93169 = invoke(stypy.reporting.localization.Localization(__file__, 458, 31), get_data_93165, *[], **kwargs_93168)
        
        # Obtaining the member '__getitem__' of a type (line 458)
        getitem___93170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), get_data_call_result_93169, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 458)
        subscript_call_result_93171 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), getitem___93170, int_93163)
        
        # Assigning a type to the variable 'tuple_var_assignment_91256' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_91256', subscript_call_result_93171)
        
        # Assigning a Subscript to a Name (line 458):
        
        # Obtaining the type of the subscript
        int_93172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 8), 'int')
        
        # Call to get_data(...): (line 458)
        # Processing the call keyword arguments (line 458)
        int_93175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 54), 'int')
        keyword_93176 = int_93175
        kwargs_93177 = {'y_stride': keyword_93176}
        # Getting the type of 'self' (line 458)
        self_93173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 458)
        get_data_93174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 31), self_93173, 'get_data')
        # Calling get_data(args, kwargs) (line 458)
        get_data_call_result_93178 = invoke(stypy.reporting.localization.Localization(__file__, 458, 31), get_data_93174, *[], **kwargs_93177)
        
        # Obtaining the member '__getitem__' of a type (line 458)
        getitem___93179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), get_data_call_result_93178, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 458)
        subscript_call_result_93180 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), getitem___93179, int_93172)
        
        # Assigning a type to the variable 'tuple_var_assignment_91257' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_91257', subscript_call_result_93180)
        
        # Assigning a Subscript to a Name (line 458):
        
        # Obtaining the type of the subscript
        int_93181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 8), 'int')
        
        # Call to get_data(...): (line 458)
        # Processing the call keyword arguments (line 458)
        int_93184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 54), 'int')
        keyword_93185 = int_93184
        kwargs_93186 = {'y_stride': keyword_93185}
        # Getting the type of 'self' (line 458)
        self_93182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 458)
        get_data_93183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 31), self_93182, 'get_data')
        # Calling get_data(args, kwargs) (line 458)
        get_data_call_result_93187 = invoke(stypy.reporting.localization.Localization(__file__, 458, 31), get_data_93183, *[], **kwargs_93186)
        
        # Obtaining the member '__getitem__' of a type (line 458)
        getitem___93188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), get_data_call_result_93187, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 458)
        subscript_call_result_93189 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), getitem___93188, int_93181)
        
        # Assigning a type to the variable 'tuple_var_assignment_91258' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_91258', subscript_call_result_93189)
        
        # Assigning a Subscript to a Name (line 458):
        
        # Obtaining the type of the subscript
        int_93190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 8), 'int')
        
        # Call to get_data(...): (line 458)
        # Processing the call keyword arguments (line 458)
        int_93193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 54), 'int')
        keyword_93194 = int_93193
        kwargs_93195 = {'y_stride': keyword_93194}
        # Getting the type of 'self' (line 458)
        self_93191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 31), 'self', False)
        # Obtaining the member 'get_data' of a type (line 458)
        get_data_93192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 31), self_93191, 'get_data')
        # Calling get_data(args, kwargs) (line 458)
        get_data_call_result_93196 = invoke(stypy.reporting.localization.Localization(__file__, 458, 31), get_data_93192, *[], **kwargs_93195)
        
        # Obtaining the member '__getitem__' of a type (line 458)
        getitem___93197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), get_data_call_result_93196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 458)
        subscript_call_result_93198 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), getitem___93197, int_93190)
        
        # Assigning a type to the variable 'tuple_var_assignment_91259' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_91259', subscript_call_result_93198)
        
        # Assigning a Name to a Name (line 458):
        # Getting the type of 'tuple_var_assignment_91255' (line 458)
        tuple_var_assignment_91255_93199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_91255')
        # Assigning a type to the variable 'alpha' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'alpha', tuple_var_assignment_91255_93199)
        
        # Assigning a Name to a Name (line 458):
        # Getting the type of 'tuple_var_assignment_91256' (line 458)
        tuple_var_assignment_91256_93200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_91256')
        # Assigning a type to the variable 'beta' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 15), 'beta', tuple_var_assignment_91256_93200)
        
        # Assigning a Name to a Name (line 458):
        # Getting the type of 'tuple_var_assignment_91257' (line 458)
        tuple_var_assignment_91257_93201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_91257')
        # Assigning a type to the variable 'a' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 21), 'a', tuple_var_assignment_91257_93201)
        
        # Assigning a Name to a Name (line 458):
        # Getting the type of 'tuple_var_assignment_91258' (line 458)
        tuple_var_assignment_91258_93202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_91258')
        # Assigning a type to the variable 'x' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 24), 'x', tuple_var_assignment_91258_93202)
        
        # Assigning a Name to a Name (line 458):
        # Getting the type of 'tuple_var_assignment_91259' (line 458)
        tuple_var_assignment_91259_93203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_91259')
        # Assigning a type to the variable 'y' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 27), 'y', tuple_var_assignment_91259_93203)
        
        
        # SSA begins for try-except statement (line 459)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 460):
        
        # Assigning a Call to a Name (line 460):
        
        # Call to blas_func(...): (line 460)
        # Processing the call arguments (line 460)
        int_93206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 31), 'int')
        # Getting the type of 'a' (line 460)
        a_93207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'a', False)
        # Getting the type of 'x' (line 460)
        x_93208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 37), 'x', False)
        int_93209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 40), 'int')
        # Getting the type of 'y' (line 460)
        y_93210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 43), 'y', False)
        # Processing the call keyword arguments (line 460)
        int_93211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 52), 'int')
        keyword_93212 = int_93211
        int_93213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 60), 'int')
        keyword_93214 = int_93213
        kwargs_93215 = {'incy': keyword_93214, 'trans': keyword_93212}
        # Getting the type of 'self' (line 460)
        self_93204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 460)
        blas_func_93205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 16), self_93204, 'blas_func')
        # Calling blas_func(args, kwargs) (line 460)
        blas_func_call_result_93216 = invoke(stypy.reporting.localization.Localization(__file__, 460, 16), blas_func_93205, *[int_93206, a_93207, x_93208, int_93209, y_93210], **kwargs_93215)
        
        # Assigning a type to the variable 'y' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'y', blas_func_call_result_93216)
        
        # Call to assert_(...): (line 461)
        # Processing the call arguments (line 461)
        int_93218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 20), 'int')
        # Processing the call keyword arguments (line 461)
        kwargs_93219 = {}
        # Getting the type of 'assert_' (line 461)
        assert__93217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 461)
        assert__call_result_93220 = invoke(stypy.reporting.localization.Localization(__file__, 461, 12), assert__93217, *[int_93218], **kwargs_93219)
        
        # SSA branch for the except part of a try statement (line 459)
        # SSA branch for the except '<any exception>' branch of a try statement (line 459)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 459)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 464)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 465):
        
        # Assigning a Call to a Name (line 465):
        
        # Call to blas_func(...): (line 465)
        # Processing the call arguments (line 465)
        int_93223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 31), 'int')
        # Getting the type of 'a' (line 465)
        a_93224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 34), 'a', False)
        # Getting the type of 'x' (line 465)
        x_93225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 37), 'x', False)
        int_93226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 40), 'int')
        # Getting the type of 'y' (line 465)
        y_93227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 43), 'y', False)
        # Processing the call keyword arguments (line 465)
        int_93228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 52), 'int')
        keyword_93229 = int_93228
        int_93230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 60), 'int')
        keyword_93231 = int_93230
        kwargs_93232 = {'incy': keyword_93231, 'trans': keyword_93229}
        # Getting the type of 'self' (line 465)
        self_93221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), 'self', False)
        # Obtaining the member 'blas_func' of a type (line 465)
        blas_func_93222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 16), self_93221, 'blas_func')
        # Calling blas_func(args, kwargs) (line 465)
        blas_func_call_result_93233 = invoke(stypy.reporting.localization.Localization(__file__, 465, 16), blas_func_93222, *[int_93223, a_93224, x_93225, int_93226, y_93227], **kwargs_93232)
        
        # Assigning a type to the variable 'y' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'y', blas_func_call_result_93233)
        
        # Call to assert_(...): (line 466)
        # Processing the call arguments (line 466)
        int_93235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 20), 'int')
        # Processing the call keyword arguments (line 466)
        kwargs_93236 = {}
        # Getting the type of 'assert_' (line 466)
        assert__93234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 466)
        assert__call_result_93237 = invoke(stypy.reporting.localization.Localization(__file__, 466, 12), assert__93234, *[int_93235], **kwargs_93236)
        
        # SSA branch for the except part of a try statement (line 464)
        # SSA branch for the except '<any exception>' branch of a try statement (line 464)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 464)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_y_stride_assert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_y_stride_assert' in the type store
        # Getting the type of 'stypy_return_type' (line 456)
        stypy_return_type_93238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_93238)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_y_stride_assert'
        return stypy_return_type_93238


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 376, 0, False)
        # Assigning a type to the variable 'self' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseGemv.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BaseGemv' (line 376)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 0), 'BaseGemv', BaseGemv)


# SSA begins for try-except statement (line 471)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Declaration of the 'TestSgemv' class
# Getting the type of 'BaseGemv' (line 472)
BaseGemv_93239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 20), 'BaseGemv')

class TestSgemv(BaseGemv_93239, ):
    
    # Assigning a Attribute to a Name (line 473):
    
    # Assigning a Name to a Name (line 474):

    @norecursion
    def test_sgemv_on_osx(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sgemv_on_osx'
        module_type_store = module_type_store.open_function_context('test_sgemv_on_osx', 476, 8, False)
        # Assigning a type to the variable 'self' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSgemv.test_sgemv_on_osx.__dict__.__setitem__('stypy_localization', localization)
        TestSgemv.test_sgemv_on_osx.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSgemv.test_sgemv_on_osx.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSgemv.test_sgemv_on_osx.__dict__.__setitem__('stypy_function_name', 'TestSgemv.test_sgemv_on_osx')
        TestSgemv.test_sgemv_on_osx.__dict__.__setitem__('stypy_param_names_list', [])
        TestSgemv.test_sgemv_on_osx.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSgemv.test_sgemv_on_osx.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSgemv.test_sgemv_on_osx.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSgemv.test_sgemv_on_osx.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSgemv.test_sgemv_on_osx.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSgemv.test_sgemv_on_osx.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSgemv.test_sgemv_on_osx', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sgemv_on_osx', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sgemv_on_osx(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 477, 12))
        
        # 'from itertools import product' statement (line 477)
        try:
            from itertools import product

        except:
            product = UndefinedType
        import_from_module(stypy.reporting.localization.Localization(__file__, 477, 12), 'itertools', None, module_type_store, ['product'], [product])
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 478, 12))
        
        # 'import sys' statement (line 478)
        import sys

        import_module(stypy.reporting.localization.Localization(__file__, 478, 12), 'sys', sys, module_type_store)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 479, 12))
        
        # 'import numpy' statement (line 479)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
        import_93240 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 479, 12), 'numpy')

        if (type(import_93240) is not StypyTypeError):

            if (import_93240 != 'pyd_module'):
                __import__(import_93240)
                sys_modules_93241 = sys.modules[import_93240]
                import_module(stypy.reporting.localization.Localization(__file__, 479, 12), 'np', sys_modules_93241.module_type_store, module_type_store)
            else:
                import numpy as np

                import_module(stypy.reporting.localization.Localization(__file__, 479, 12), 'np', numpy, module_type_store)

        else:
            # Assigning a type to the variable 'numpy' (line 479)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'numpy', import_93240)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')
        
        
        
        # Getting the type of 'sys' (line 481)
        sys_93242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 15), 'sys')
        # Obtaining the member 'platform' of a type (line 481)
        platform_93243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 15), sys_93242, 'platform')
        str_93244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 31), 'str', 'darwin')
        # Applying the binary operator '!=' (line 481)
        result_ne_93245 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 15), '!=', platform_93243, str_93244)
        
        # Testing the type of an if condition (line 481)
        if_condition_93246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 12), result_ne_93245)
        # Assigning a type to the variable 'if_condition_93246' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'if_condition_93246', if_condition_93246)
        # SSA begins for if statement (line 481)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 481)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def aligned_array(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            str_93247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 57), 'str', 'C')
            defaults = [str_93247]
            # Create a new context for function 'aligned_array'
            module_type_store = module_type_store.open_function_context('aligned_array', 484, 12, False)
            
            # Passed parameters checking function
            aligned_array.stypy_localization = localization
            aligned_array.stypy_type_of_self = None
            aligned_array.stypy_type_store = module_type_store
            aligned_array.stypy_function_name = 'aligned_array'
            aligned_array.stypy_param_names_list = ['shape', 'align', 'dtype', 'order']
            aligned_array.stypy_varargs_param_name = None
            aligned_array.stypy_kwargs_param_name = None
            aligned_array.stypy_call_defaults = defaults
            aligned_array.stypy_call_varargs = varargs
            aligned_array.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'aligned_array', ['shape', 'align', 'dtype', 'order'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'aligned_array', localization, ['shape', 'align', 'dtype', 'order'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'aligned_array(...)' code ##################

            
            # Assigning a Call to a Name (line 486):
            
            # Assigning a Call to a Name (line 486):
            
            # Call to dtype(...): (line 486)
            # Processing the call keyword arguments (line 486)
            kwargs_93249 = {}
            # Getting the type of 'dtype' (line 486)
            dtype_93248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'dtype', False)
            # Calling dtype(args, kwargs) (line 486)
            dtype_call_result_93250 = invoke(stypy.reporting.localization.Localization(__file__, 486, 20), dtype_93248, *[], **kwargs_93249)
            
            # Assigning a type to the variable 'd' (line 486)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 16), 'd', dtype_call_result_93250)
            
            # Assigning a Call to a Name (line 488):
            
            # Assigning a Call to a Name (line 488):
            
            # Call to prod(...): (line 488)
            # Processing the call arguments (line 488)
            # Getting the type of 'shape' (line 488)
            shape_93253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 28), 'shape', False)
            # Processing the call keyword arguments (line 488)
            kwargs_93254 = {}
            # Getting the type of 'np' (line 488)
            np_93251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 20), 'np', False)
            # Obtaining the member 'prod' of a type (line 488)
            prod_93252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 20), np_93251, 'prod')
            # Calling prod(args, kwargs) (line 488)
            prod_call_result_93255 = invoke(stypy.reporting.localization.Localization(__file__, 488, 20), prod_93252, *[shape_93253], **kwargs_93254)
            
            # Assigning a type to the variable 'N' (line 488)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 16), 'N', prod_call_result_93255)
            
            # Assigning a Call to a Name (line 489):
            
            # Assigning a Call to a Name (line 489):
            
            # Call to zeros(...): (line 489)
            # Processing the call arguments (line 489)
            # Getting the type of 'N' (line 489)
            N_93258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 31), 'N', False)
            # Getting the type of 'd' (line 489)
            d_93259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 35), 'd', False)
            # Obtaining the member 'nbytes' of a type (line 489)
            nbytes_93260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 35), d_93259, 'nbytes')
            # Applying the binary operator '*' (line 489)
            result_mul_93261 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 31), '*', N_93258, nbytes_93260)
            
            # Getting the type of 'align' (line 489)
            align_93262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 46), 'align', False)
            # Applying the binary operator '+' (line 489)
            result_add_93263 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 31), '+', result_mul_93261, align_93262)
            
            # Processing the call keyword arguments (line 489)
            # Getting the type of 'np' (line 489)
            np_93264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 59), 'np', False)
            # Obtaining the member 'uint8' of a type (line 489)
            uint8_93265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 59), np_93264, 'uint8')
            keyword_93266 = uint8_93265
            kwargs_93267 = {'dtype': keyword_93266}
            # Getting the type of 'np' (line 489)
            np_93256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 22), 'np', False)
            # Obtaining the member 'zeros' of a type (line 489)
            zeros_93257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 22), np_93256, 'zeros')
            # Calling zeros(args, kwargs) (line 489)
            zeros_call_result_93268 = invoke(stypy.reporting.localization.Localization(__file__, 489, 22), zeros_93257, *[result_add_93263], **kwargs_93267)
            
            # Assigning a type to the variable 'tmp' (line 489)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 16), 'tmp', zeros_call_result_93268)
            
            # Assigning a Subscript to a Name (line 490):
            
            # Assigning a Subscript to a Name (line 490):
            
            # Obtaining the type of the subscript
            int_93269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 58), 'int')
            
            # Obtaining the type of the subscript
            str_93270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 50), 'str', 'data')
            # Getting the type of 'tmp' (line 490)
            tmp_93271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 26), 'tmp')
            # Obtaining the member '__array_interface__' of a type (line 490)
            array_interface___93272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 26), tmp_93271, '__array_interface__')
            # Obtaining the member '__getitem__' of a type (line 490)
            getitem___93273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 26), array_interface___93272, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 490)
            subscript_call_result_93274 = invoke(stypy.reporting.localization.Localization(__file__, 490, 26), getitem___93273, str_93270)
            
            # Obtaining the member '__getitem__' of a type (line 490)
            getitem___93275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 26), subscript_call_result_93274, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 490)
            subscript_call_result_93276 = invoke(stypy.reporting.localization.Localization(__file__, 490, 26), getitem___93275, int_93269)
            
            # Assigning a type to the variable 'address' (line 490)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 16), 'address', subscript_call_result_93276)
            
            
            # Call to range(...): (line 492)
            # Processing the call arguments (line 492)
            # Getting the type of 'align' (line 492)
            align_93278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 36), 'align', False)
            # Processing the call keyword arguments (line 492)
            kwargs_93279 = {}
            # Getting the type of 'range' (line 492)
            range_93277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 30), 'range', False)
            # Calling range(args, kwargs) (line 492)
            range_call_result_93280 = invoke(stypy.reporting.localization.Localization(__file__, 492, 30), range_93277, *[align_93278], **kwargs_93279)
            
            # Testing the type of a for loop iterable (line 492)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 492, 16), range_call_result_93280)
            # Getting the type of the for loop variable (line 492)
            for_loop_var_93281 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 492, 16), range_call_result_93280)
            # Assigning a type to the variable 'offset' (line 492)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 16), 'offset', for_loop_var_93281)
            # SSA begins for a for statement (line 492)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'address' (line 493)
            address_93282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 24), 'address')
            # Getting the type of 'offset' (line 493)
            offset_93283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 34), 'offset')
            # Applying the binary operator '+' (line 493)
            result_add_93284 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 24), '+', address_93282, offset_93283)
            
            # Getting the type of 'align' (line 493)
            align_93285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 44), 'align')
            # Applying the binary operator '%' (line 493)
            result_mod_93286 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 23), '%', result_add_93284, align_93285)
            
            int_93287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 53), 'int')
            # Applying the binary operator '==' (line 493)
            result_eq_93288 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 23), '==', result_mod_93286, int_93287)
            
            # Testing the type of an if condition (line 493)
            if_condition_93289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 493, 20), result_eq_93288)
            # Assigning a type to the variable 'if_condition_93289' (line 493)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 20), 'if_condition_93289', if_condition_93289)
            # SSA begins for if statement (line 493)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 493)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 495):
            
            # Assigning a Call to a Name (line 495):
            
            # Call to view(...): (line 495)
            # Processing the call keyword arguments (line 495)
            # Getting the type of 'dtype' (line 495)
            dtype_93302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 63), 'dtype', False)
            keyword_93303 = dtype_93302
            kwargs_93304 = {'dtype': keyword_93303}
            
            # Obtaining the type of the subscript
            # Getting the type of 'offset' (line 495)
            offset_93290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 26), 'offset', False)
            # Getting the type of 'offset' (line 495)
            offset_93291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 33), 'offset', False)
            # Getting the type of 'N' (line 495)
            N_93292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 40), 'N', False)
            # Getting the type of 'd' (line 495)
            d_93293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 42), 'd', False)
            # Obtaining the member 'nbytes' of a type (line 495)
            nbytes_93294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 42), d_93293, 'nbytes')
            # Applying the binary operator '*' (line 495)
            result_mul_93295 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 40), '*', N_93292, nbytes_93294)
            
            # Applying the binary operator '+' (line 495)
            result_add_93296 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 33), '+', offset_93291, result_mul_93295)
            
            slice_93297 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 495, 22), offset_93290, result_add_93296, None)
            # Getting the type of 'tmp' (line 495)
            tmp_93298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 22), 'tmp', False)
            # Obtaining the member '__getitem__' of a type (line 495)
            getitem___93299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 22), tmp_93298, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 495)
            subscript_call_result_93300 = invoke(stypy.reporting.localization.Localization(__file__, 495, 22), getitem___93299, slice_93297)
            
            # Obtaining the member 'view' of a type (line 495)
            view_93301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 22), subscript_call_result_93300, 'view')
            # Calling view(args, kwargs) (line 495)
            view_call_result_93305 = invoke(stypy.reporting.localization.Localization(__file__, 495, 22), view_93301, *[], **kwargs_93304)
            
            # Assigning a type to the variable 'tmp' (line 495)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 16), 'tmp', view_call_result_93305)
            
            # Call to reshape(...): (line 496)
            # Processing the call arguments (line 496)
            # Getting the type of 'shape' (line 496)
            shape_93308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 35), 'shape', False)
            # Processing the call keyword arguments (line 496)
            # Getting the type of 'order' (line 496)
            order_93309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 48), 'order', False)
            keyword_93310 = order_93309
            kwargs_93311 = {'order': keyword_93310}
            # Getting the type of 'tmp' (line 496)
            tmp_93306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 23), 'tmp', False)
            # Obtaining the member 'reshape' of a type (line 496)
            reshape_93307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 23), tmp_93306, 'reshape')
            # Calling reshape(args, kwargs) (line 496)
            reshape_call_result_93312 = invoke(stypy.reporting.localization.Localization(__file__, 496, 23), reshape_93307, *[shape_93308], **kwargs_93311)
            
            # Assigning a type to the variable 'stypy_return_type' (line 496)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 16), 'stypy_return_type', reshape_call_result_93312)
            
            # ################# End of 'aligned_array(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'aligned_array' in the type store
            # Getting the type of 'stypy_return_type' (line 484)
            stypy_return_type_93313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_93313)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'aligned_array'
            return stypy_return_type_93313

        # Assigning a type to the variable 'aligned_array' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'aligned_array', aligned_array)

        @norecursion
        def as_aligned(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            str_93314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 52), 'str', 'C')
            defaults = [str_93314]
            # Create a new context for function 'as_aligned'
            module_type_store = module_type_store.open_function_context('as_aligned', 498, 12, False)
            
            # Passed parameters checking function
            as_aligned.stypy_localization = localization
            as_aligned.stypy_type_of_self = None
            as_aligned.stypy_type_store = module_type_store
            as_aligned.stypy_function_name = 'as_aligned'
            as_aligned.stypy_param_names_list = ['arr', 'align', 'dtype', 'order']
            as_aligned.stypy_varargs_param_name = None
            as_aligned.stypy_kwargs_param_name = None
            as_aligned.stypy_call_defaults = defaults
            as_aligned.stypy_call_varargs = varargs
            as_aligned.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'as_aligned', ['arr', 'align', 'dtype', 'order'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'as_aligned', localization, ['arr', 'align', 'dtype', 'order'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'as_aligned(...)' code ##################

            
            # Assigning a Call to a Name (line 500):
            
            # Assigning a Call to a Name (line 500):
            
            # Call to aligned_array(...): (line 500)
            # Processing the call arguments (line 500)
            # Getting the type of 'arr' (line 500)
            arr_93316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 40), 'arr', False)
            # Obtaining the member 'shape' of a type (line 500)
            shape_93317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 40), arr_93316, 'shape')
            # Getting the type of 'align' (line 500)
            align_93318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 51), 'align', False)
            # Getting the type of 'dtype' (line 500)
            dtype_93319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 58), 'dtype', False)
            # Getting the type of 'order' (line 500)
            order_93320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 65), 'order', False)
            # Processing the call keyword arguments (line 500)
            kwargs_93321 = {}
            # Getting the type of 'aligned_array' (line 500)
            aligned_array_93315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 26), 'aligned_array', False)
            # Calling aligned_array(args, kwargs) (line 500)
            aligned_array_call_result_93322 = invoke(stypy.reporting.localization.Localization(__file__, 500, 26), aligned_array_93315, *[shape_93317, align_93318, dtype_93319, order_93320], **kwargs_93321)
            
            # Assigning a type to the variable 'aligned' (line 500)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 16), 'aligned', aligned_array_call_result_93322)
            
            # Assigning a Subscript to a Subscript (line 501):
            
            # Assigning a Subscript to a Subscript (line 501):
            
            # Obtaining the type of the subscript
            slice_93323 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 501, 29), None, None, None)
            # Getting the type of 'arr' (line 501)
            arr_93324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 29), 'arr')
            # Obtaining the member '__getitem__' of a type (line 501)
            getitem___93325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 29), arr_93324, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 501)
            subscript_call_result_93326 = invoke(stypy.reporting.localization.Localization(__file__, 501, 29), getitem___93325, slice_93323)
            
            # Getting the type of 'aligned' (line 501)
            aligned_93327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'aligned')
            slice_93328 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 501, 16), None, None, None)
            # Storing an element on a container (line 501)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 16), aligned_93327, (slice_93328, subscript_call_result_93326))
            # Getting the type of 'aligned' (line 502)
            aligned_93329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 23), 'aligned')
            # Assigning a type to the variable 'stypy_return_type' (line 502)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'stypy_return_type', aligned_93329)
            
            # ################# End of 'as_aligned(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'as_aligned' in the type store
            # Getting the type of 'stypy_return_type' (line 498)
            stypy_return_type_93330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_93330)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'as_aligned'
            return stypy_return_type_93330

        # Assigning a type to the variable 'as_aligned' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'as_aligned', as_aligned)

        @norecursion
        def assert_dot_close(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'assert_dot_close'
            module_type_store = module_type_store.open_function_context('assert_dot_close', 504, 12, False)
            
            # Passed parameters checking function
            assert_dot_close.stypy_localization = localization
            assert_dot_close.stypy_type_of_self = None
            assert_dot_close.stypy_type_store = module_type_store
            assert_dot_close.stypy_function_name = 'assert_dot_close'
            assert_dot_close.stypy_param_names_list = ['A', 'X', 'desired']
            assert_dot_close.stypy_varargs_param_name = None
            assert_dot_close.stypy_kwargs_param_name = None
            assert_dot_close.stypy_call_defaults = defaults
            assert_dot_close.stypy_call_varargs = varargs
            assert_dot_close.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'assert_dot_close', ['A', 'X', 'desired'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'assert_dot_close', localization, ['A', 'X', 'desired'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'assert_dot_close(...)' code ##################

            
            # Call to assert_allclose(...): (line 505)
            # Processing the call arguments (line 505)
            
            # Call to blas_func(...): (line 505)
            # Processing the call arguments (line 505)
            float_93334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 47), 'float')
            # Getting the type of 'A' (line 505)
            A_93335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 52), 'A', False)
            # Getting the type of 'X' (line 505)
            X_93336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 55), 'X', False)
            # Processing the call keyword arguments (line 505)
            kwargs_93337 = {}
            # Getting the type of 'self' (line 505)
            self_93332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 32), 'self', False)
            # Obtaining the member 'blas_func' of a type (line 505)
            blas_func_93333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 32), self_93332, 'blas_func')
            # Calling blas_func(args, kwargs) (line 505)
            blas_func_call_result_93338 = invoke(stypy.reporting.localization.Localization(__file__, 505, 32), blas_func_93333, *[float_93334, A_93335, X_93336], **kwargs_93337)
            
            # Getting the type of 'desired' (line 505)
            desired_93339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 59), 'desired', False)
            # Processing the call keyword arguments (line 505)
            float_93340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 37), 'float')
            keyword_93341 = float_93340
            float_93342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 48), 'float')
            keyword_93343 = float_93342
            kwargs_93344 = {'rtol': keyword_93341, 'atol': keyword_93343}
            # Getting the type of 'assert_allclose' (line 505)
            assert_allclose_93331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 16), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 505)
            assert_allclose_call_result_93345 = invoke(stypy.reporting.localization.Localization(__file__, 505, 16), assert_allclose_93331, *[blas_func_call_result_93338, desired_93339], **kwargs_93344)
            
            
            # ################# End of 'assert_dot_close(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'assert_dot_close' in the type store
            # Getting the type of 'stypy_return_type' (line 504)
            stypy_return_type_93346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_93346)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'assert_dot_close'
            return stypy_return_type_93346

        # Assigning a type to the variable 'assert_dot_close' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'assert_dot_close', assert_dot_close)
        
        # Assigning a Call to a Name (line 508):
        
        # Assigning a Call to a Name (line 508):
        
        # Call to product(...): (line 508)
        # Processing the call arguments (line 508)
        
        # Obtaining an instance of the builtin type 'tuple' (line 508)
        tuple_93348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 508)
        # Adding element type (line 508)
        int_93349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 32), tuple_93348, int_93349)
        # Adding element type (line 508)
        int_93350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 32), tuple_93348, int_93350)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 508)
        tuple_93351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 508)
        # Adding element type (line 508)
        int_93352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 42), tuple_93351, int_93352)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 508)
        tuple_93353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 508)
        # Adding element type (line 508)
        int_93354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 52), tuple_93353, int_93354)
        # Adding element type (line 508)
        int_93355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 52), tuple_93353, int_93355)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 508)
        tuple_93356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 508)
        # Adding element type (line 508)
        str_93357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 63), 'str', 'C')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 63), tuple_93356, str_93357)
        # Adding element type (line 508)
        str_93358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 68), 'str', 'F')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 63), tuple_93356, str_93358)
        
        # Processing the call keyword arguments (line 508)
        kwargs_93359 = {}
        # Getting the type of 'product' (line 508)
        product_93347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 23), 'product', False)
        # Calling product(args, kwargs) (line 508)
        product_call_result_93360 = invoke(stypy.reporting.localization.Localization(__file__, 508, 23), product_93347, *[tuple_93348, tuple_93351, tuple_93353, tuple_93356], **kwargs_93359)
        
        # Assigning a type to the variable 'testdata' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'testdata', product_call_result_93360)
        
        # Getting the type of 'testdata' (line 509)
        testdata_93361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 40), 'testdata')
        # Testing the type of a for loop iterable (line 509)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 509, 12), testdata_93361)
        # Getting the type of the for loop variable (line 509)
        for_loop_var_93362 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 509, 12), testdata_93361)
        # Assigning a type to the variable 'align' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'align', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 12), for_loop_var_93362))
        # Assigning a type to the variable 'm' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 12), for_loop_var_93362))
        # Assigning a type to the variable 'n' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 12), for_loop_var_93362))
        # Assigning a type to the variable 'a_order' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'a_order', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 12), for_loop_var_93362))
        # SSA begins for a for statement (line 509)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 510):
        
        # Assigning a Call to a Name (line 510):
        
        # Call to rand(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'm' (line 510)
        m_93366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 37), 'm', False)
        # Getting the type of 'n' (line 510)
        n_93367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 40), 'n', False)
        # Processing the call keyword arguments (line 510)
        kwargs_93368 = {}
        # Getting the type of 'np' (line 510)
        np_93363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 22), 'np', False)
        # Obtaining the member 'random' of a type (line 510)
        random_93364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 22), np_93363, 'random')
        # Obtaining the member 'rand' of a type (line 510)
        rand_93365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 22), random_93364, 'rand')
        # Calling rand(args, kwargs) (line 510)
        rand_call_result_93369 = invoke(stypy.reporting.localization.Localization(__file__, 510, 22), rand_93365, *[m_93366, n_93367], **kwargs_93368)
        
        # Assigning a type to the variable 'A_d' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'A_d', rand_call_result_93369)
        
        # Assigning a Call to a Name (line 511):
        
        # Assigning a Call to a Name (line 511):
        
        # Call to rand(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 'n' (line 511)
        n_93373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 37), 'n', False)
        # Processing the call keyword arguments (line 511)
        kwargs_93374 = {}
        # Getting the type of 'np' (line 511)
        np_93370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 22), 'np', False)
        # Obtaining the member 'random' of a type (line 511)
        random_93371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 22), np_93370, 'random')
        # Obtaining the member 'rand' of a type (line 511)
        rand_93372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 22), random_93371, 'rand')
        # Calling rand(args, kwargs) (line 511)
        rand_call_result_93375 = invoke(stypy.reporting.localization.Localization(__file__, 511, 22), rand_93372, *[n_93373], **kwargs_93374)
        
        # Assigning a type to the variable 'X_d' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 16), 'X_d', rand_call_result_93375)
        
        # Assigning a Call to a Name (line 512):
        
        # Assigning a Call to a Name (line 512):
        
        # Call to dot(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'A_d' (line 512)
        A_d_93378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 33), 'A_d', False)
        # Getting the type of 'X_d' (line 512)
        X_d_93379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 38), 'X_d', False)
        # Processing the call keyword arguments (line 512)
        kwargs_93380 = {}
        # Getting the type of 'np' (line 512)
        np_93376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 26), 'np', False)
        # Obtaining the member 'dot' of a type (line 512)
        dot_93377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 26), np_93376, 'dot')
        # Calling dot(args, kwargs) (line 512)
        dot_call_result_93381 = invoke(stypy.reporting.localization.Localization(__file__, 512, 26), dot_93377, *[A_d_93378, X_d_93379], **kwargs_93380)
        
        # Assigning a type to the variable 'desired' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 16), 'desired', dot_call_result_93381)
        
        # Assigning a Call to a Name (line 514):
        
        # Assigning a Call to a Name (line 514):
        
        # Call to as_aligned(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'A_d' (line 514)
        A_d_93383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 33), 'A_d', False)
        # Getting the type of 'align' (line 514)
        align_93384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 38), 'align', False)
        # Getting the type of 'np' (line 514)
        np_93385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 45), 'np', False)
        # Obtaining the member 'float32' of a type (line 514)
        float32_93386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 45), np_93385, 'float32')
        # Processing the call keyword arguments (line 514)
        # Getting the type of 'a_order' (line 514)
        a_order_93387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 63), 'a_order', False)
        keyword_93388 = a_order_93387
        kwargs_93389 = {'order': keyword_93388}
        # Getting the type of 'as_aligned' (line 514)
        as_aligned_93382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 22), 'as_aligned', False)
        # Calling as_aligned(args, kwargs) (line 514)
        as_aligned_call_result_93390 = invoke(stypy.reporting.localization.Localization(__file__, 514, 22), as_aligned_93382, *[A_d_93383, align_93384, float32_93386], **kwargs_93389)
        
        # Assigning a type to the variable 'A_f' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 16), 'A_f', as_aligned_call_result_93390)
        
        # Assigning a Call to a Name (line 515):
        
        # Assigning a Call to a Name (line 515):
        
        # Call to as_aligned(...): (line 515)
        # Processing the call arguments (line 515)
        # Getting the type of 'X_d' (line 515)
        X_d_93392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 33), 'X_d', False)
        # Getting the type of 'align' (line 515)
        align_93393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 38), 'align', False)
        # Getting the type of 'np' (line 515)
        np_93394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 45), 'np', False)
        # Obtaining the member 'float32' of a type (line 515)
        float32_93395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 45), np_93394, 'float32')
        # Processing the call keyword arguments (line 515)
        # Getting the type of 'a_order' (line 515)
        a_order_93396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 63), 'a_order', False)
        keyword_93397 = a_order_93396
        kwargs_93398 = {'order': keyword_93397}
        # Getting the type of 'as_aligned' (line 515)
        as_aligned_93391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 22), 'as_aligned', False)
        # Calling as_aligned(args, kwargs) (line 515)
        as_aligned_call_result_93399 = invoke(stypy.reporting.localization.Localization(__file__, 515, 22), as_aligned_93391, *[X_d_93392, align_93393, float32_93395], **kwargs_93398)
        
        # Assigning a type to the variable 'X_f' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 16), 'X_f', as_aligned_call_result_93399)
        
        # Call to assert_dot_close(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'A_f' (line 516)
        A_f_93401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 33), 'A_f', False)
        # Getting the type of 'X_f' (line 516)
        X_f_93402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 38), 'X_f', False)
        # Getting the type of 'desired' (line 516)
        desired_93403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 43), 'desired', False)
        # Processing the call keyword arguments (line 516)
        kwargs_93404 = {}
        # Getting the type of 'assert_dot_close' (line 516)
        assert_dot_close_93400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 16), 'assert_dot_close', False)
        # Calling assert_dot_close(args, kwargs) (line 516)
        assert_dot_close_call_result_93405 = invoke(stypy.reporting.localization.Localization(__file__, 516, 16), assert_dot_close_93400, *[A_f_93401, X_f_93402, desired_93403], **kwargs_93404)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_sgemv_on_osx(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sgemv_on_osx' in the type store
        # Getting the type of 'stypy_return_type' (line 476)
        stypy_return_type_93406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_93406)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sgemv_on_osx'
        return stypy_return_type_93406


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 472, 4, False)
        # Assigning a type to the variable 'self' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSgemv.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSgemv' (line 472)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'TestSgemv', TestSgemv)

# Assigning a Attribute to a Name (line 473):
# Getting the type of 'fblas' (line 473)
fblas_93407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 20), 'fblas')
# Obtaining the member 'sgemv' of a type (line 473)
sgemv_93408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 20), fblas_93407, 'sgemv')
# Getting the type of 'TestSgemv'
TestSgemv_93409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestSgemv')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestSgemv_93409, 'blas_func', sgemv_93408)

# Assigning a Name to a Name (line 474):
# Getting the type of 'float32' (line 474)
float32_93410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'float32')
# Getting the type of 'TestSgemv'
TestSgemv_93411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestSgemv')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestSgemv_93411, 'dtype', float32_93410)
# SSA branch for the except part of a try statement (line 471)
# SSA branch for the except 'AttributeError' branch of a try statement (line 471)
module_type_store.open_ssa_branch('except')
# Declaration of the 'TestSgemv' class

class TestSgemv:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 519, 4, False)
        # Assigning a type to the variable 'self' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSgemv.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSgemv' (line 519)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'TestSgemv', TestSgemv)
# SSA join for try-except statement (line 471)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TestDgemv' class
# Getting the type of 'BaseGemv' (line 523)
BaseGemv_93412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'BaseGemv')

class TestDgemv(BaseGemv_93412, ):
    
    # Assigning a Attribute to a Name (line 524):
    
    # Assigning a Name to a Name (line 525):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 523, 0, False)
        # Assigning a type to the variable 'self' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDgemv.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDgemv' (line 523)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 0), 'TestDgemv', TestDgemv)

# Assigning a Attribute to a Name (line 524):
# Getting the type of 'fblas' (line 524)
fblas_93413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'fblas')
# Obtaining the member 'dgemv' of a type (line 524)
dgemv_93414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 16), fblas_93413, 'dgemv')
# Getting the type of 'TestDgemv'
TestDgemv_93415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDgemv')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDgemv_93415, 'blas_func', dgemv_93414)

# Assigning a Name to a Name (line 525):
# Getting the type of 'float64' (line 525)
float64_93416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'float64')
# Getting the type of 'TestDgemv'
TestDgemv_93417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestDgemv')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestDgemv_93417, 'dtype', float64_93416)


# SSA begins for try-except statement (line 528)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Declaration of the 'TestCgemv' class
# Getting the type of 'BaseGemv' (line 529)
BaseGemv_93418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 20), 'BaseGemv')

class TestCgemv(BaseGemv_93418, ):
    
    # Assigning a Attribute to a Name (line 530):
    
    # Assigning a Name to a Name (line 531):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 529, 4, False)
        # Assigning a type to the variable 'self' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCgemv.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCgemv' (line 529)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'TestCgemv', TestCgemv)

# Assigning a Attribute to a Name (line 530):
# Getting the type of 'fblas' (line 530)
fblas_93419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 20), 'fblas')
# Obtaining the member 'cgemv' of a type (line 530)
cgemv_93420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 20), fblas_93419, 'cgemv')
# Getting the type of 'TestCgemv'
TestCgemv_93421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCgemv')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCgemv_93421, 'blas_func', cgemv_93420)

# Assigning a Name to a Name (line 531):
# Getting the type of 'complex64' (line 531)
complex64_93422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 16), 'complex64')
# Getting the type of 'TestCgemv'
TestCgemv_93423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCgemv')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCgemv_93423, 'dtype', complex64_93422)
# SSA branch for the except part of a try statement (line 528)
# SSA branch for the except 'AttributeError' branch of a try statement (line 528)
module_type_store.open_ssa_branch('except')
# Declaration of the 'TestCgemv' class

class TestCgemv:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 533, 4, False)
        # Assigning a type to the variable 'self' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCgemv.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCgemv' (line 533)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'TestCgemv', TestCgemv)
# SSA join for try-except statement (line 528)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TestZgemv' class
# Getting the type of 'BaseGemv' (line 537)
BaseGemv_93424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 16), 'BaseGemv')

class TestZgemv(BaseGemv_93424, ):
    
    # Assigning a Attribute to a Name (line 538):
    
    # Assigning a Name to a Name (line 539):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 537, 0, False)
        # Assigning a type to the variable 'self' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZgemv.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestZgemv' (line 537)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'TestZgemv', TestZgemv)

# Assigning a Attribute to a Name (line 538):
# Getting the type of 'fblas' (line 538)
fblas_93425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 16), 'fblas')
# Obtaining the member 'zgemv' of a type (line 538)
zgemv_93426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 16), fblas_93425, 'zgemv')
# Getting the type of 'TestZgemv'
TestZgemv_93427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZgemv')
# Setting the type of the member 'blas_func' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZgemv_93427, 'blas_func', zgemv_93426)

# Assigning a Name to a Name (line 539):
# Getting the type of 'complex128' (line 539)
complex128_93428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'complex128')
# Getting the type of 'TestZgemv'
TestZgemv_93429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestZgemv')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestZgemv_93429, 'dtype', complex128_93428)
str_93430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, (-1)), 'str', '\n##################################################\n### Test blas ?ger\n### This will be a mess to test all cases.\n\nclass BaseGer(object):\n    def get_data(self,x_stride=1,y_stride=1):\n        from numpy.random import normal, seed\n        seed(1234)\n        alpha = array(1., dtype = self.dtype)\n        a = normal(0.,1.,(3,3)).astype(self.dtype)\n        x = arange(shape(a)[0]*x_stride,dtype=self.dtype)\n        y = arange(shape(a)[1]*y_stride,dtype=self.dtype)\n        return alpha,a,x,y\n    def test_simple(self):\n        alpha,a,x,y = self.get_data()\n        # tranpose takes care of Fortran vs. C(and Python) memory layout\n        desired_a = alpha*transpose(x[:,newaxis]*y) + a\n        self.blas_func(x,y,a)\n        assert_array_almost_equal(desired_a,a)\n    def test_x_stride(self):\n        alpha,a,x,y = self.get_data(x_stride=2)\n        desired_a = alpha*transpose(x[::2,newaxis]*y) + a\n        self.blas_func(x,y,a,incx=2)\n        assert_array_almost_equal(desired_a,a)\n    def test_x_stride_assert(self):\n        alpha,a,x,y = self.get_data(x_stride=2)\n        try:\n            self.blas_func(x,y,a,incx=3)\n            assert(0)\n        except:\n            pass\n    def test_y_stride(self):\n        alpha,a,x,y = self.get_data(y_stride=2)\n        desired_a = alpha*transpose(x[:,newaxis]*y[::2]) + a\n        self.blas_func(x,y,a,incy=2)\n        assert_array_almost_equal(desired_a,a)\n\n    def test_y_stride_assert(self):\n        alpha,a,x,y = self.get_data(y_stride=2)\n        try:\n            self.blas_func(a,x,y,incy=3)\n            assert(0)\n        except:\n            pass\n\nclass TestSger(BaseGer):\n    blas_func = fblas.sger\n    dtype = float32\nclass TestDger(BaseGer):\n    blas_func = fblas.dger\n    dtype = float64\n')
str_93431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, (-1)), 'str', '\nclass BaseGerComplex(BaseGer):\n    def get_data(self,x_stride=1,y_stride=1):\n        from numpy.random import normal, seed\n        seed(1234)\n        alpha = array(1+1j, dtype = self.dtype)\n        a = normal(0.,1.,(3,3)).astype(self.dtype)\n        a = a + normal(0.,1.,(3,3)) * array(1j, dtype = self.dtype)\n        x = normal(0.,1.,shape(a)[0]*x_stride).astype(self.dtype)\n        x = x + x * array(1j, dtype = self.dtype)\n        y = normal(0.,1.,shape(a)[1]*y_stride).astype(self.dtype)\n        y = y + y * array(1j, dtype = self.dtype)\n        return alpha,a,x,y\n    def test_simple(self):\n        alpha,a,x,y = self.get_data()\n        # tranpose takes care of Fortran vs. C(and Python) memory layout\n        a = a * array(0.,dtype = self.dtype)\n        #desired_a = alpha*transpose(x[:,newaxis]*self.transform(y)) + a\n        desired_a = alpha*transpose(x[:,newaxis]*y) + a\n        #self.blas_func(x,y,a,alpha = alpha)\n        fblas.cgeru(x,y,a,alpha = alpha)\n        assert_array_almost_equal(desired_a,a)\n\n    #def test_x_stride(self):\n    #    alpha,a,x,y = self.get_data(x_stride=2)\n    #    desired_a = alpha*transpose(x[::2,newaxis]*self.transform(y)) + a\n    #    self.blas_func(x,y,a,incx=2)\n    #    assert_array_almost_equal(desired_a,a)\n    #def test_y_stride(self):\n    #    alpha,a,x,y = self.get_data(y_stride=2)\n    #    desired_a = alpha*transpose(x[:,newaxis]*self.transform(y[::2])) + a\n    #    self.blas_func(x,y,a,incy=2)\n    #    assert_array_almost_equal(desired_a,a)\n\nclass TestCgeru(BaseGerComplex):\n    blas_func = fblas.cgeru\n    dtype = complex64\n    def transform(self,x):\n        return x\nclass TestZgeru(BaseGerComplex):\n    blas_func = fblas.zgeru\n    dtype = complex128\n    def transform(self,x):\n        return x\n\nclass TestCgerc(BaseGerComplex):\n    blas_func = fblas.cgerc\n    dtype = complex64\n    def transform(self,x):\n        return conjugate(x)\n\nclass TestZgerc(BaseGerComplex):\n    blas_func = fblas.zgerc\n    dtype = complex128\n    def transform(self,x):\n        return conjugate(x)\n')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
