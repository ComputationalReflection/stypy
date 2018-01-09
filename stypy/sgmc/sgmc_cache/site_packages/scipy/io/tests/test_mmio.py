
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from tempfile import mkdtemp, mktemp
4: import os
5: import shutil
6: 
7: import numpy as np
8: from numpy import array, transpose, pi
9: from numpy.testing import (assert_equal,
10:                            assert_array_equal, assert_array_almost_equal)
11: from pytest import raises as assert_raises
12: 
13: import scipy.sparse
14: from scipy.io.mmio import mminfo, mmread, mmwrite
15: 
16: 
17: class TestMMIOArray(object):
18:     def setup_method(self):
19:         self.tmpdir = mkdtemp()
20:         self.fn = os.path.join(self.tmpdir, 'testfile.mtx')
21: 
22:     def teardown_method(self):
23:         shutil.rmtree(self.tmpdir)
24: 
25:     def check(self, a, info):
26:         mmwrite(self.fn, a)
27:         assert_equal(mminfo(self.fn), info)
28:         b = mmread(self.fn)
29:         assert_array_almost_equal(a, b)
30: 
31:     def check_exact(self, a, info):
32:         mmwrite(self.fn, a)
33:         assert_equal(mminfo(self.fn), info)
34:         b = mmread(self.fn)
35:         assert_equal(a, b)
36: 
37:     def test_simple_integer(self):
38:         self.check_exact([[1, 2], [3, 4]],
39:                          (2, 2, 4, 'array', 'integer', 'general'))
40: 
41:     def test_32bit_integer(self):
42:         a = array([[2**31-1, 2**31-2], [2**31-3, 2**31-4]], dtype=np.int32)
43:         self.check_exact(a, (2, 2, 4, 'array', 'integer', 'general'))
44: 
45:     def test_64bit_integer(self):
46:         a = array([[2**31, 2**32], [2**63-2, 2**63-1]], dtype=np.int64)
47:         if (np.intp(0).itemsize < 8):
48:             assert_raises(OverflowError, mmwrite, self.fn, a)
49:         else:
50:             self.check_exact(a, (2, 2, 4, 'array', 'integer', 'general'))
51: 
52:     def test_simple_upper_triangle_integer(self):
53:         self.check_exact([[0, 1], [0, 0]],
54:                          (2, 2, 4, 'array', 'integer', 'general'))
55: 
56:     def test_simple_lower_triangle_integer(self):
57:         self.check_exact([[0, 0], [1, 0]],
58:                          (2, 2, 4, 'array', 'integer', 'general'))
59: 
60:     def test_simple_rectangular_integer(self):
61:         self.check_exact([[1, 2, 3], [4, 5, 6]],
62:                          (2, 3, 6, 'array', 'integer', 'general'))
63: 
64:     def test_simple_rectangular_float(self):
65:         self.check([[1, 2], [3.5, 4], [5, 6]],
66:                    (3, 2, 6, 'array', 'real', 'general'))
67: 
68:     def test_simple_float(self):
69:         self.check([[1, 2], [3, 4.0]],
70:                    (2, 2, 4, 'array', 'real', 'general'))
71: 
72:     def test_simple_complex(self):
73:         self.check([[1, 2], [3, 4j]],
74:                    (2, 2, 4, 'array', 'complex', 'general'))
75: 
76:     def test_simple_symmetric_integer(self):
77:         self.check_exact([[1, 2], [2, 4]],
78:                          (2, 2, 4, 'array', 'integer', 'symmetric'))
79: 
80:     def test_simple_skew_symmetric_integer(self):
81:         self.check_exact([[1, 2], [-2, 4]],
82:                          (2, 2, 4, 'array', 'integer', 'skew-symmetric'))
83: 
84:     def test_simple_skew_symmetric_float(self):
85:         self.check(array([[1, 2], [-2.0, 4]], 'f'),
86:                    (2, 2, 4, 'array', 'real', 'skew-symmetric'))
87: 
88:     def test_simple_hermitian_complex(self):
89:         self.check([[1, 2+3j], [2-3j, 4]],
90:                    (2, 2, 4, 'array', 'complex', 'hermitian'))
91: 
92:     def test_random_symmetric_float(self):
93:         sz = (20, 20)
94:         a = np.random.random(sz)
95:         a = a + transpose(a)
96:         self.check(a, (20, 20, 400, 'array', 'real', 'symmetric'))
97: 
98:     def test_random_rectangular_float(self):
99:         sz = (20, 15)
100:         a = np.random.random(sz)
101:         self.check(a, (20, 15, 300, 'array', 'real', 'general'))
102: 
103: 
104: class TestMMIOSparseCSR(TestMMIOArray):
105:     def setup_method(self):
106:         self.tmpdir = mkdtemp()
107:         self.fn = os.path.join(self.tmpdir, 'testfile.mtx')
108: 
109:     def teardown_method(self):
110:         shutil.rmtree(self.tmpdir)
111: 
112:     def check(self, a, info):
113:         mmwrite(self.fn, a)
114:         assert_equal(mminfo(self.fn), info)
115:         b = mmread(self.fn)
116:         assert_array_almost_equal(a.todense(), b.todense())
117: 
118:     def check_exact(self, a, info):
119:         mmwrite(self.fn, a)
120:         assert_equal(mminfo(self.fn), info)
121:         b = mmread(self.fn)
122:         assert_equal(a.todense(), b.todense())
123: 
124:     def test_simple_integer(self):
125:         self.check_exact(scipy.sparse.csr_matrix([[1, 2], [3, 4]]),
126:                          (2, 2, 4, 'coordinate', 'integer', 'general'))
127: 
128:     def test_32bit_integer(self):
129:         a = scipy.sparse.csr_matrix(array([[2**31-1, -2**31+2],
130:                                            [2**31-3, 2**31-4]],
131:                                           dtype=np.int32))
132:         self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))
133: 
134:     def test_64bit_integer(self):
135:         a = scipy.sparse.csr_matrix(array([[2**32+1, 2**32+1],
136:                                            [-2**63+2, 2**63-2]],
137:                                           dtype=np.int64))
138:         if (np.intp(0).itemsize < 8):
139:             assert_raises(OverflowError, mmwrite, self.fn, a)
140:         else:
141:             self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))
142: 
143:     def test_simple_upper_triangle_integer(self):
144:         self.check_exact(scipy.sparse.csr_matrix([[0, 1], [0, 0]]),
145:                          (2, 2, 1, 'coordinate', 'integer', 'general'))
146: 
147:     def test_simple_lower_triangle_integer(self):
148:         self.check_exact(scipy.sparse.csr_matrix([[0, 0], [1, 0]]),
149:                          (2, 2, 1, 'coordinate', 'integer', 'general'))
150: 
151:     def test_simple_rectangular_integer(self):
152:         self.check_exact(scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6]]),
153:                          (2, 3, 6, 'coordinate', 'integer', 'general'))
154: 
155:     def test_simple_rectangular_float(self):
156:         self.check(scipy.sparse.csr_matrix([[1, 2], [3.5, 4], [5, 6]]),
157:                    (3, 2, 6, 'coordinate', 'real', 'general'))
158: 
159:     def test_simple_float(self):
160:         self.check(scipy.sparse.csr_matrix([[1, 2], [3, 4.0]]),
161:                    (2, 2, 4, 'coordinate', 'real', 'general'))
162: 
163:     def test_simple_complex(self):
164:         self.check(scipy.sparse.csr_matrix([[1, 2], [3, 4j]]),
165:                    (2, 2, 4, 'coordinate', 'complex', 'general'))
166: 
167:     def test_simple_symmetric_integer(self):
168:         self.check_exact(scipy.sparse.csr_matrix([[1, 2], [2, 4]]),
169:                          (2, 2, 3, 'coordinate', 'integer', 'symmetric'))
170: 
171:     def test_simple_skew_symmetric_integer(self):
172:         self.check_exact(scipy.sparse.csr_matrix([[1, 2], [-2, 4]]),
173:                          (2, 2, 3, 'coordinate', 'integer', 'skew-symmetric'))
174: 
175:     def test_simple_skew_symmetric_float(self):
176:         self.check(scipy.sparse.csr_matrix(array([[1, 2], [-2.0, 4]], 'f')),
177:                    (2, 2, 3, 'coordinate', 'real', 'skew-symmetric'))
178: 
179:     def test_simple_hermitian_complex(self):
180:         self.check(scipy.sparse.csr_matrix([[1, 2+3j], [2-3j, 4]]),
181:                    (2, 2, 3, 'coordinate', 'complex', 'hermitian'))
182: 
183:     def test_random_symmetric_float(self):
184:         sz = (20, 20)
185:         a = np.random.random(sz)
186:         a = a + transpose(a)
187:         a = scipy.sparse.csr_matrix(a)
188:         self.check(a, (20, 20, 210, 'coordinate', 'real', 'symmetric'))
189: 
190:     def test_random_rectangular_float(self):
191:         sz = (20, 15)
192:         a = np.random.random(sz)
193:         a = scipy.sparse.csr_matrix(a)
194:         self.check(a, (20, 15, 300, 'coordinate', 'real', 'general'))
195: 
196:     def test_simple_pattern(self):
197:         a = scipy.sparse.csr_matrix([[0, 1.5], [3.0, 2.5]])
198:         p = np.zeros_like(a.todense())
199:         p[a.todense() > 0] = 1
200:         info = (2, 2, 3, 'coordinate', 'pattern', 'general')
201:         mmwrite(self.fn, a, field='pattern')
202:         assert_equal(mminfo(self.fn), info)
203:         b = mmread(self.fn)
204:         assert_array_almost_equal(p, b.todense())
205: 
206: 
207: _32bit_integer_dense_example = '''\
208: %%MatrixMarket matrix array integer general
209: 2  2
210: 2147483647
211: 2147483646
212: 2147483647
213: 2147483646
214: '''
215: 
216: _32bit_integer_sparse_example = '''\
217: %%MatrixMarket matrix coordinate integer symmetric
218: 2  2  2
219: 1  1  2147483647
220: 2  2  2147483646
221: '''
222: 
223: _64bit_integer_dense_example = '''\
224: %%MatrixMarket matrix array integer general
225: 2  2
226:           2147483648
227: -9223372036854775806
228:          -2147483648
229:  9223372036854775807
230: '''
231: 
232: _64bit_integer_sparse_general_example = '''\
233: %%MatrixMarket matrix coordinate integer general
234: 2  2  3
235: 1  1           2147483648
236: 1  2  9223372036854775807
237: 2  2  9223372036854775807
238: '''
239: 
240: _64bit_integer_sparse_symmetric_example = '''\
241: %%MatrixMarket matrix coordinate integer symmetric
242: 2  2  3
243: 1  1            2147483648
244: 1  2  -9223372036854775807
245: 2  2   9223372036854775807
246: '''
247: 
248: _64bit_integer_sparse_skew_example = '''\
249: %%MatrixMarket matrix coordinate integer skew-symmetric
250: 2  2  3
251: 1  1            2147483648
252: 1  2  -9223372036854775807
253: 2  2   9223372036854775807
254: '''
255: 
256: _over64bit_integer_dense_example = '''\
257: %%MatrixMarket matrix array integer general
258: 2  2
259:          2147483648
260: 9223372036854775807
261:          2147483648
262: 9223372036854775808
263: '''
264: 
265: _over64bit_integer_sparse_example = '''\
266: %%MatrixMarket matrix coordinate integer symmetric
267: 2  2  2
268: 1  1            2147483648
269: 2  2  19223372036854775808
270: '''
271: 
272: class TestMMIOReadLargeIntegers(object):
273:     def setup_method(self):
274:         self.tmpdir = mkdtemp()
275:         self.fn = os.path.join(self.tmpdir, 'testfile.mtx')
276: 
277:     def teardown_method(self):
278:         shutil.rmtree(self.tmpdir)
279: 
280:     def check_read(self, example, a, info, dense, over32, over64):
281:         with open(self.fn, 'w') as f:
282:             f.write(example)
283:         assert_equal(mminfo(self.fn), info)
284:         if (over32 and (np.intp(0).itemsize < 8)) or over64:
285:             assert_raises(OverflowError, mmread, self.fn)
286:         else:
287:             b = mmread(self.fn)
288:             if not dense:
289:                 b = b.todense()
290:             assert_equal(a, b)
291: 
292:     def test_read_32bit_integer_dense(self):
293:         a = array([[2**31-1, 2**31-1],
294:                    [2**31-2, 2**31-2]], dtype=np.int64)
295:         self.check_read(_32bit_integer_dense_example,
296:                         a,
297:                         (2, 2, 4, 'array', 'integer', 'general'),
298:                         dense=True,
299:                         over32=False,
300:                         over64=False)
301: 
302:     def test_read_32bit_integer_sparse(self):
303:         a = array([[2**31-1, 0],
304:                    [0, 2**31-2]], dtype=np.int64)
305:         self.check_read(_32bit_integer_sparse_example,
306:                         a,
307:                         (2, 2, 2, 'coordinate', 'integer', 'symmetric'),
308:                         dense=False,
309:                         over32=False,
310:                         over64=False)
311: 
312:     def test_read_64bit_integer_dense(self):
313:         a = array([[2**31, -2**31],
314:                    [-2**63+2, 2**63-1]], dtype=np.int64)
315:         self.check_read(_64bit_integer_dense_example,
316:                         a,
317:                         (2, 2, 4, 'array', 'integer', 'general'),
318:                         dense=True,
319:                         over32=True,
320:                         over64=False)
321: 
322:     def test_read_64bit_integer_sparse_general(self):
323:         a = array([[2**31, 2**63-1],
324:                    [0, 2**63-1]], dtype=np.int64)
325:         self.check_read(_64bit_integer_sparse_general_example,
326:                         a,
327:                         (2, 2, 3, 'coordinate', 'integer', 'general'),
328:                         dense=False,
329:                         over32=True,
330:                         over64=False)
331: 
332:     def test_read_64bit_integer_sparse_symmetric(self):
333:         a = array([[2**31, -2**63+1],
334:                    [-2**63+1, 2**63-1]], dtype=np.int64)
335:         self.check_read(_64bit_integer_sparse_symmetric_example,
336:                         a,
337:                         (2, 2, 3, 'coordinate', 'integer', 'symmetric'),
338:                         dense=False,
339:                         over32=True,
340:                         over64=False)
341: 
342:     def test_read_64bit_integer_sparse_skew(self):
343:         a = array([[2**31, -2**63+1],
344:                    [2**63-1, 2**63-1]], dtype=np.int64)
345:         self.check_read(_64bit_integer_sparse_skew_example,
346:                         a,
347:                         (2, 2, 3, 'coordinate', 'integer', 'skew-symmetric'),
348:                         dense=False,
349:                         over32=True,
350:                         over64=False)
351: 
352:     def test_read_over64bit_integer_dense(self):
353:         self.check_read(_over64bit_integer_dense_example,
354:                         None,
355:                         (2, 2, 4, 'array', 'integer', 'general'),
356:                         dense=True,
357:                         over32=True,
358:                         over64=True)
359: 
360:     def test_read_over64bit_integer_sparse(self):
361:         self.check_read(_over64bit_integer_sparse_example,
362:                         None,
363:                         (2, 2, 2, 'coordinate', 'integer', 'symmetric'),
364:                         dense=False,
365:                         over32=True,
366:                         over64=True)
367: 
368: 
369: _general_example = '''\
370: %%MatrixMarket matrix coordinate real general
371: %=================================================================================
372: %
373: % This ASCII file represents a sparse MxN matrix with L
374: % nonzeros in the following Matrix Market format:
375: %
376: % +----------------------------------------------+
377: % |%%MatrixMarket matrix coordinate real general | <--- header line
378: % |%                                             | <--+
379: % |% comments                                    |    |-- 0 or more comment lines
380: % |%                                             | <--+
381: % |    M  N  L                                   | <--- rows, columns, entries
382: % |    I1  J1  A(I1, J1)                         | <--+
383: % |    I2  J2  A(I2, J2)                         |    |
384: % |    I3  J3  A(I3, J3)                         |    |-- L lines
385: % |        . . .                                 |    |
386: % |    IL JL  A(IL, JL)                          | <--+
387: % +----------------------------------------------+
388: %
389: % Indices are 1-based, i.e. A(1,1) is the first element.
390: %
391: %=================================================================================
392:   5  5  8
393:     1     1   1.000e+00
394:     2     2   1.050e+01
395:     3     3   1.500e-02
396:     1     4   6.000e+00
397:     4     2   2.505e+02
398:     4     4  -2.800e+02
399:     4     5   3.332e+01
400:     5     5   1.200e+01
401: '''
402: 
403: _hermitian_example = '''\
404: %%MatrixMarket matrix coordinate complex hermitian
405:   5  5  7
406:     1     1     1.0      0
407:     2     2    10.5      0
408:     4     2   250.5     22.22
409:     3     3     1.5e-2   0
410:     4     4    -2.8e2    0
411:     5     5    12.       0
412:     5     4     0       33.32
413: '''
414: 
415: _skew_example = '''\
416: %%MatrixMarket matrix coordinate real skew-symmetric
417:   5  5  7
418:     1     1     1.0
419:     2     2    10.5
420:     4     2   250.5
421:     3     3     1.5e-2
422:     4     4    -2.8e2
423:     5     5    12.
424:     5     4     0
425: '''
426: 
427: _symmetric_example = '''\
428: %%MatrixMarket matrix coordinate real symmetric
429:   5  5  7
430:     1     1     1.0
431:     2     2    10.5
432:     4     2   250.5
433:     3     3     1.5e-2
434:     4     4    -2.8e2
435:     5     5    12.
436:     5     4     8
437: '''
438: 
439: _symmetric_pattern_example = '''\
440: %%MatrixMarket matrix coordinate pattern symmetric
441:   5  5  7
442:     1     1
443:     2     2
444:     4     2
445:     3     3
446:     4     4
447:     5     5
448:     5     4
449: '''
450: 
451: 
452: class TestMMIOCoordinate(object):
453:     def setup_method(self):
454:         self.tmpdir = mkdtemp()
455:         self.fn = os.path.join(self.tmpdir, 'testfile.mtx')
456: 
457:     def teardown_method(self):
458:         shutil.rmtree(self.tmpdir)
459: 
460:     def check_read(self, example, a, info):
461:         f = open(self.fn, 'w')
462:         f.write(example)
463:         f.close()
464:         assert_equal(mminfo(self.fn), info)
465:         b = mmread(self.fn).todense()
466:         assert_array_almost_equal(a, b)
467: 
468:     def test_read_general(self):
469:         a = [[1, 0, 0, 6, 0],
470:              [0, 10.5, 0, 0, 0],
471:              [0, 0, .015, 0, 0],
472:              [0, 250.5, 0, -280, 33.32],
473:              [0, 0, 0, 0, 12]]
474:         self.check_read(_general_example, a,
475:                         (5, 5, 8, 'coordinate', 'real', 'general'))
476: 
477:     def test_read_hermitian(self):
478:         a = [[1, 0, 0, 0, 0],
479:              [0, 10.5, 0, 250.5 - 22.22j, 0],
480:              [0, 0, .015, 0, 0],
481:              [0, 250.5 + 22.22j, 0, -280, -33.32j],
482:              [0, 0, 0, 33.32j, 12]]
483:         self.check_read(_hermitian_example, a,
484:                         (5, 5, 7, 'coordinate', 'complex', 'hermitian'))
485: 
486:     def test_read_skew(self):
487:         a = [[1, 0, 0, 0, 0],
488:              [0, 10.5, 0, -250.5, 0],
489:              [0, 0, .015, 0, 0],
490:              [0, 250.5, 0, -280, 0],
491:              [0, 0, 0, 0, 12]]
492:         self.check_read(_skew_example, a,
493:                         (5, 5, 7, 'coordinate', 'real', 'skew-symmetric'))
494: 
495:     def test_read_symmetric(self):
496:         a = [[1, 0, 0, 0, 0],
497:              [0, 10.5, 0, 250.5, 0],
498:              [0, 0, .015, 0, 0],
499:              [0, 250.5, 0, -280, 8],
500:              [0, 0, 0, 8, 12]]
501:         self.check_read(_symmetric_example, a,
502:                         (5, 5, 7, 'coordinate', 'real', 'symmetric'))
503: 
504:     def test_read_symmetric_pattern(self):
505:         a = [[1, 0, 0, 0, 0],
506:              [0, 1, 0, 1, 0],
507:              [0, 0, 1, 0, 0],
508:              [0, 1, 0, 1, 1],
509:              [0, 0, 0, 1, 1]]
510:         self.check_read(_symmetric_pattern_example, a,
511:                         (5, 5, 7, 'coordinate', 'pattern', 'symmetric'))
512: 
513:     def test_empty_write_read(self):
514:         # http://projects.scipy.org/scipy/ticket/883
515: 
516:         b = scipy.sparse.coo_matrix((10, 10))
517:         mmwrite(self.fn, b)
518: 
519:         assert_equal(mminfo(self.fn),
520:                      (10, 10, 0, 'coordinate', 'real', 'symmetric'))
521:         a = b.todense()
522:         b = mmread(self.fn).todense()
523:         assert_array_almost_equal(a, b)
524: 
525:     def test_bzip2_py3(self):
526:         # test if fix for #2152 works
527:         try:
528:             # bz2 module isn't always built when building Python.
529:             import bz2
530:         except:
531:             return
532:         I = array([0, 0, 1, 2, 3, 3, 3, 4])
533:         J = array([0, 3, 1, 2, 1, 3, 4, 4])
534:         V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])
535: 
536:         b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))
537: 
538:         mmwrite(self.fn, b)
539: 
540:         fn_bzip2 = "%s.bz2" % self.fn
541:         with open(self.fn, 'rb') as f_in:
542:             f_out = bz2.BZ2File(fn_bzip2, 'wb')
543:             f_out.write(f_in.read())
544:             f_out.close()
545: 
546:         a = mmread(fn_bzip2).todense()
547:         assert_array_almost_equal(a, b.todense())
548: 
549:     def test_gzip_py3(self):
550:         # test if fix for #2152 works
551:         try:
552:             # gzip module can be missing from Python installation
553:             import gzip
554:         except:
555:             return
556:         I = array([0, 0, 1, 2, 3, 3, 3, 4])
557:         J = array([0, 3, 1, 2, 1, 3, 4, 4])
558:         V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])
559: 
560:         b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))
561: 
562:         mmwrite(self.fn, b)
563: 
564:         fn_gzip = "%s.gz" % self.fn
565:         with open(self.fn, 'rb') as f_in:
566:             f_out = gzip.open(fn_gzip, 'wb')
567:             f_out.write(f_in.read())
568:             f_out.close()
569: 
570:         a = mmread(fn_gzip).todense()
571:         assert_array_almost_equal(a, b.todense())
572: 
573:     def test_real_write_read(self):
574:         I = array([0, 0, 1, 2, 3, 3, 3, 4])
575:         J = array([0, 3, 1, 2, 1, 3, 4, 4])
576:         V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])
577: 
578:         b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))
579: 
580:         mmwrite(self.fn, b)
581: 
582:         assert_equal(mminfo(self.fn),
583:                      (5, 5, 8, 'coordinate', 'real', 'general'))
584:         a = b.todense()
585:         b = mmread(self.fn).todense()
586:         assert_array_almost_equal(a, b)
587: 
588:     def test_complex_write_read(self):
589:         I = array([0, 0, 1, 2, 3, 3, 3, 4])
590:         J = array([0, 3, 1, 2, 1, 3, 4, 4])
591:         V = array([1.0 + 3j, 6.0 + 2j, 10.50 + 0.9j, 0.015 + -4.4j,
592:                    250.5 + 0j, -280.0 + 5j, 33.32 + 6.4j, 12.00 + 0.8j])
593: 
594:         b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))
595: 
596:         mmwrite(self.fn, b)
597: 
598:         assert_equal(mminfo(self.fn),
599:                      (5, 5, 8, 'coordinate', 'complex', 'general'))
600:         a = b.todense()
601:         b = mmread(self.fn).todense()
602:         assert_array_almost_equal(a, b)
603: 
604:     def test_sparse_formats(self):
605:         mats = []
606: 
607:         I = array([0, 0, 1, 2, 3, 3, 3, 4])
608:         J = array([0, 3, 1, 2, 1, 3, 4, 4])
609: 
610:         V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])
611:         mats.append(scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5)))
612: 
613:         V = array([1.0 + 3j, 6.0 + 2j, 10.50 + 0.9j, 0.015 + -4.4j,
614:                    250.5 + 0j, -280.0 + 5j, 33.32 + 6.4j, 12.00 + 0.8j])
615:         mats.append(scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5)))
616: 
617:         for mat in mats:
618:             expected = mat.todense()
619:             for fmt in ['csr', 'csc', 'coo']:
620:                 fn = mktemp(dir=self.tmpdir)  # safe, we own tmpdir
621:                 mmwrite(fn, mat.asformat(fmt))
622: 
623:                 result = mmread(fn).todense()
624:                 assert_array_almost_equal(result, expected)
625: 
626:     def test_precision(self):
627:         test_values = [pi] + [10**(i) for i in range(0, -10, -1)]
628:         test_precisions = range(1, 10)
629:         for value in test_values:
630:             for precision in test_precisions:
631:                 # construct sparse matrix with test value at last main diagonal
632:                 n = 10**precision + 1
633:                 A = scipy.sparse.dok_matrix((n, n))
634:                 A[n-1, n-1] = value
635:                 # write matrix with test precision and read again
636:                 mmwrite(self.fn, A, precision=precision)
637:                 A = scipy.io.mmread(self.fn)
638:                 # check for right entries in matrix
639:                 assert_array_equal(A.row, [n-1])
640:                 assert_array_equal(A.col, [n-1])
641:                 assert_array_almost_equal(A.data,
642:                     [float('%%.%dg' % precision % value)])
643: 
644: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from tempfile import mkdtemp, mktemp' statement (line 3)
try:
    from tempfile import mkdtemp, mktemp

except:
    mkdtemp = UndefinedType
    mktemp = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'tempfile', None, module_type_store, ['mkdtemp', 'mktemp'], [mkdtemp, mktemp])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import shutil' statement (line 5)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_4738 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_4738) is not StypyTypeError):

    if (import_4738 != 'pyd_module'):
        __import__(import_4738)
        sys_modules_4739 = sys.modules[import_4738]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_4739.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_4738)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy import array, transpose, pi' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_4740 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_4740) is not StypyTypeError):

    if (import_4740 != 'pyd_module'):
        __import__(import_4740)
        sys_modules_4741 = sys.modules[import_4740]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', sys_modules_4741.module_type_store, module_type_store, ['array', 'transpose', 'pi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_4741, sys_modules_4741.module_type_store, module_type_store)
    else:
        from numpy import array, transpose, pi

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', None, module_type_store, ['array', 'transpose', 'pi'], [array, transpose, pi])

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_4740)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_4742 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing')

if (type(import_4742) is not StypyTypeError):

    if (import_4742 != 'pyd_module'):
        __import__(import_4742)
        sys_modules_4743 = sys.modules[import_4742]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', sys_modules_4743.module_type_store, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_4743, sys_modules_4743.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_array_almost_equal'], [assert_equal, assert_array_equal, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', import_4742)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from pytest import assert_raises' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_4744 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest')

if (type(import_4744) is not StypyTypeError):

    if (import_4744 != 'pyd_module'):
        __import__(import_4744)
        sys_modules_4745 = sys.modules[import_4744]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', sys_modules_4745.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_4745, sys_modules_4745.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', import_4744)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import scipy.sparse' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_4746 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse')

if (type(import_4746) is not StypyTypeError):

    if (import_4746 != 'pyd_module'):
        __import__(import_4746)
        sys_modules_4747 = sys.modules[import_4746]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', sys_modules_4747.module_type_store, module_type_store)
    else:
        import scipy.sparse

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', import_4746)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.io.mmio import mminfo, mmread, mmwrite' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_4748 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.mmio')

if (type(import_4748) is not StypyTypeError):

    if (import_4748 != 'pyd_module'):
        __import__(import_4748)
        sys_modules_4749 = sys.modules[import_4748]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.mmio', sys_modules_4749.module_type_store, module_type_store, ['mminfo', 'mmread', 'mmwrite'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_4749, sys_modules_4749.module_type_store, module_type_store)
    else:
        from scipy.io.mmio import mminfo, mmread, mmwrite

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.mmio', None, module_type_store, ['mminfo', 'mmread', 'mmwrite'], [mminfo, mmread, mmwrite])

else:
    # Assigning a type to the variable 'scipy.io.mmio' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.mmio', import_4748)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

# Declaration of the 'TestMMIOArray' class

class TestMMIOArray(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.setup_method.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.setup_method')
        TestMMIOArray.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 19):
        
        # Call to mkdtemp(...): (line 19)
        # Processing the call keyword arguments (line 19)
        kwargs_4751 = {}
        # Getting the type of 'mkdtemp' (line 19)
        mkdtemp_4750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'mkdtemp', False)
        # Calling mkdtemp(args, kwargs) (line 19)
        mkdtemp_call_result_4752 = invoke(stypy.reporting.localization.Localization(__file__, 19, 22), mkdtemp_4750, *[], **kwargs_4751)
        
        # Getting the type of 'self' (line 19)
        self_4753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self')
        # Setting the type of the member 'tmpdir' of a type (line 19)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_4753, 'tmpdir', mkdtemp_call_result_4752)
        
        # Assigning a Call to a Attribute (line 20):
        
        # Call to join(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'self' (line 20)
        self_4757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 31), 'self', False)
        # Obtaining the member 'tmpdir' of a type (line 20)
        tmpdir_4758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 31), self_4757, 'tmpdir')
        str_4759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 44), 'str', 'testfile.mtx')
        # Processing the call keyword arguments (line 20)
        kwargs_4760 = {}
        # Getting the type of 'os' (line 20)
        os_4754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 20)
        path_4755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 18), os_4754, 'path')
        # Obtaining the member 'join' of a type (line 20)
        join_4756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 18), path_4755, 'join')
        # Calling join(args, kwargs) (line 20)
        join_call_result_4761 = invoke(stypy.reporting.localization.Localization(__file__, 20, 18), join_4756, *[tmpdir_4758, str_4759], **kwargs_4760)
        
        # Getting the type of 'self' (line 20)
        self_4762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member 'fn' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_4762, 'fn', join_call_result_4761)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_4763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4763)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_4763


    @norecursion
    def teardown_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'teardown_method'
        module_type_store = module_type_store.open_function_context('teardown_method', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.teardown_method.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.teardown_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.teardown_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.teardown_method.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.teardown_method')
        TestMMIOArray.teardown_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.teardown_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.teardown_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.teardown_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.teardown_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.teardown_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.teardown_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.teardown_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'teardown_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'teardown_method(...)' code ##################

        
        # Call to rmtree(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'self' (line 23)
        self_4766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 22), 'self', False)
        # Obtaining the member 'tmpdir' of a type (line 23)
        tmpdir_4767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 22), self_4766, 'tmpdir')
        # Processing the call keyword arguments (line 23)
        kwargs_4768 = {}
        # Getting the type of 'shutil' (line 23)
        shutil_4764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'shutil', False)
        # Obtaining the member 'rmtree' of a type (line 23)
        rmtree_4765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), shutil_4764, 'rmtree')
        # Calling rmtree(args, kwargs) (line 23)
        rmtree_call_result_4769 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), rmtree_4765, *[tmpdir_4767], **kwargs_4768)
        
        
        # ################# End of 'teardown_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'teardown_method' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_4770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4770)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'teardown_method'
        return stypy_return_type_4770


    @norecursion
    def check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.check.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.check.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.check.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.check')
        TestMMIOArray.check.__dict__.__setitem__('stypy_param_names_list', ['a', 'info'])
        TestMMIOArray.check.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.check.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.check.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.check.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.check.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.check', ['a', 'info'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['a', 'info'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Call to mmwrite(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'self' (line 26)
        self_4772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'self', False)
        # Obtaining the member 'fn' of a type (line 26)
        fn_4773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), self_4772, 'fn')
        # Getting the type of 'a' (line 26)
        a_4774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'a', False)
        # Processing the call keyword arguments (line 26)
        kwargs_4775 = {}
        # Getting the type of 'mmwrite' (line 26)
        mmwrite_4771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 26)
        mmwrite_call_result_4776 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), mmwrite_4771, *[fn_4773, a_4774], **kwargs_4775)
        
        
        # Call to assert_equal(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Call to mminfo(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'self' (line 27)
        self_4779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 28), 'self', False)
        # Obtaining the member 'fn' of a type (line 27)
        fn_4780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 28), self_4779, 'fn')
        # Processing the call keyword arguments (line 27)
        kwargs_4781 = {}
        # Getting the type of 'mminfo' (line 27)
        mminfo_4778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'mminfo', False)
        # Calling mminfo(args, kwargs) (line 27)
        mminfo_call_result_4782 = invoke(stypy.reporting.localization.Localization(__file__, 27, 21), mminfo_4778, *[fn_4780], **kwargs_4781)
        
        # Getting the type of 'info' (line 27)
        info_4783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 38), 'info', False)
        # Processing the call keyword arguments (line 27)
        kwargs_4784 = {}
        # Getting the type of 'assert_equal' (line 27)
        assert_equal_4777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 27)
        assert_equal_call_result_4785 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_equal_4777, *[mminfo_call_result_4782, info_4783], **kwargs_4784)
        
        
        # Assigning a Call to a Name (line 28):
        
        # Call to mmread(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'self' (line 28)
        self_4787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'self', False)
        # Obtaining the member 'fn' of a type (line 28)
        fn_4788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 19), self_4787, 'fn')
        # Processing the call keyword arguments (line 28)
        kwargs_4789 = {}
        # Getting the type of 'mmread' (line 28)
        mmread_4786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'mmread', False)
        # Calling mmread(args, kwargs) (line 28)
        mmread_call_result_4790 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), mmread_4786, *[fn_4788], **kwargs_4789)
        
        # Assigning a type to the variable 'b' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'b', mmread_call_result_4790)
        
        # Call to assert_array_almost_equal(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'a' (line 29)
        a_4792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 34), 'a', False)
        # Getting the type of 'b' (line 29)
        b_4793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 37), 'b', False)
        # Processing the call keyword arguments (line 29)
        kwargs_4794 = {}
        # Getting the type of 'assert_array_almost_equal' (line 29)
        assert_array_almost_equal_4791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 29)
        assert_array_almost_equal_call_result_4795 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assert_array_almost_equal_4791, *[a_4792, b_4793], **kwargs_4794)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_4796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4796)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_4796


    @norecursion
    def check_exact(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_exact'
        module_type_store = module_type_store.open_function_context('check_exact', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.check_exact.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.check_exact.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.check_exact.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.check_exact.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.check_exact')
        TestMMIOArray.check_exact.__dict__.__setitem__('stypy_param_names_list', ['a', 'info'])
        TestMMIOArray.check_exact.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.check_exact.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.check_exact.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.check_exact.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.check_exact.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.check_exact.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.check_exact', ['a', 'info'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_exact', localization, ['a', 'info'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_exact(...)' code ##################

        
        # Call to mmwrite(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'self' (line 32)
        self_4798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'self', False)
        # Obtaining the member 'fn' of a type (line 32)
        fn_4799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), self_4798, 'fn')
        # Getting the type of 'a' (line 32)
        a_4800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'a', False)
        # Processing the call keyword arguments (line 32)
        kwargs_4801 = {}
        # Getting the type of 'mmwrite' (line 32)
        mmwrite_4797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 32)
        mmwrite_call_result_4802 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), mmwrite_4797, *[fn_4799, a_4800], **kwargs_4801)
        
        
        # Call to assert_equal(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to mminfo(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'self' (line 33)
        self_4805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 28), 'self', False)
        # Obtaining the member 'fn' of a type (line 33)
        fn_4806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 28), self_4805, 'fn')
        # Processing the call keyword arguments (line 33)
        kwargs_4807 = {}
        # Getting the type of 'mminfo' (line 33)
        mminfo_4804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'mminfo', False)
        # Calling mminfo(args, kwargs) (line 33)
        mminfo_call_result_4808 = invoke(stypy.reporting.localization.Localization(__file__, 33, 21), mminfo_4804, *[fn_4806], **kwargs_4807)
        
        # Getting the type of 'info' (line 33)
        info_4809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 38), 'info', False)
        # Processing the call keyword arguments (line 33)
        kwargs_4810 = {}
        # Getting the type of 'assert_equal' (line 33)
        assert_equal_4803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 33)
        assert_equal_call_result_4811 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), assert_equal_4803, *[mminfo_call_result_4808, info_4809], **kwargs_4810)
        
        
        # Assigning a Call to a Name (line 34):
        
        # Call to mmread(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'self' (line 34)
        self_4813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'self', False)
        # Obtaining the member 'fn' of a type (line 34)
        fn_4814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 19), self_4813, 'fn')
        # Processing the call keyword arguments (line 34)
        kwargs_4815 = {}
        # Getting the type of 'mmread' (line 34)
        mmread_4812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'mmread', False)
        # Calling mmread(args, kwargs) (line 34)
        mmread_call_result_4816 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), mmread_4812, *[fn_4814], **kwargs_4815)
        
        # Assigning a type to the variable 'b' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'b', mmread_call_result_4816)
        
        # Call to assert_equal(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'a' (line 35)
        a_4818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'a', False)
        # Getting the type of 'b' (line 35)
        b_4819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'b', False)
        # Processing the call keyword arguments (line 35)
        kwargs_4820 = {}
        # Getting the type of 'assert_equal' (line 35)
        assert_equal_4817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 35)
        assert_equal_call_result_4821 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assert_equal_4817, *[a_4818, b_4819], **kwargs_4820)
        
        
        # ################# End of 'check_exact(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_exact' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_4822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_exact'
        return stypy_return_type_4822


    @norecursion
    def test_simple_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_integer'
        module_type_store = module_type_store.open_function_context('test_simple_integer', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_simple_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_simple_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_simple_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_simple_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_simple_integer')
        TestMMIOArray.test_simple_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_simple_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_simple_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_simple_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_simple_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_simple_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_simple_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_simple_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_integer(...)' code ##################

        
        # Call to check_exact(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_4825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_4826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        # Adding element type (line 38)
        int_4827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 26), list_4826, int_4827)
        # Adding element type (line 38)
        int_4828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 26), list_4826, int_4828)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 25), list_4825, list_4826)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_4829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        # Adding element type (line 38)
        int_4830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 34), list_4829, int_4830)
        # Adding element type (line 38)
        int_4831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 34), list_4829, int_4831)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 25), list_4825, list_4829)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_4832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_4833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 26), tuple_4832, int_4833)
        # Adding element type (line 39)
        int_4834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 26), tuple_4832, int_4834)
        # Adding element type (line 39)
        int_4835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 26), tuple_4832, int_4835)
        # Adding element type (line 39)
        str_4836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 35), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 26), tuple_4832, str_4836)
        # Adding element type (line 39)
        str_4837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 44), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 26), tuple_4832, str_4837)
        # Adding element type (line 39)
        str_4838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 55), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 26), tuple_4832, str_4838)
        
        # Processing the call keyword arguments (line 38)
        kwargs_4839 = {}
        # Getting the type of 'self' (line 38)
        self_4823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 38)
        check_exact_4824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_4823, 'check_exact')
        # Calling check_exact(args, kwargs) (line 38)
        check_exact_call_result_4840 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), check_exact_4824, *[list_4825, tuple_4832], **kwargs_4839)
        
        
        # ################# End of 'test_simple_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_4841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4841)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_integer'
        return stypy_return_type_4841


    @norecursion
    def test_32bit_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_32bit_integer'
        module_type_store = module_type_store.open_function_context('test_32bit_integer', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_32bit_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_32bit_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_32bit_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_32bit_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_32bit_integer')
        TestMMIOArray.test_32bit_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_32bit_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_32bit_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_32bit_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_32bit_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_32bit_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_32bit_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_32bit_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_32bit_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_32bit_integer(...)' code ##################

        
        # Assigning a Call to a Name (line 42):
        
        # Call to array(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_4843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_4844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        int_4845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 20), 'int')
        int_4846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 23), 'int')
        # Applying the binary operator '**' (line 42)
        result_pow_4847 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 20), '**', int_4845, int_4846)
        
        int_4848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 26), 'int')
        # Applying the binary operator '-' (line 42)
        result_sub_4849 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 20), '-', result_pow_4847, int_4848)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 19), list_4844, result_sub_4849)
        # Adding element type (line 42)
        int_4850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 29), 'int')
        int_4851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 32), 'int')
        # Applying the binary operator '**' (line 42)
        result_pow_4852 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 29), '**', int_4850, int_4851)
        
        int_4853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 35), 'int')
        # Applying the binary operator '-' (line 42)
        result_sub_4854 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 29), '-', result_pow_4852, int_4853)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 19), list_4844, result_sub_4854)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 18), list_4843, list_4844)
        # Adding element type (line 42)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_4855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        int_4856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 40), 'int')
        int_4857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 43), 'int')
        # Applying the binary operator '**' (line 42)
        result_pow_4858 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 40), '**', int_4856, int_4857)
        
        int_4859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 46), 'int')
        # Applying the binary operator '-' (line 42)
        result_sub_4860 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 40), '-', result_pow_4858, int_4859)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 39), list_4855, result_sub_4860)
        # Adding element type (line 42)
        int_4861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 49), 'int')
        int_4862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 52), 'int')
        # Applying the binary operator '**' (line 42)
        result_pow_4863 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 49), '**', int_4861, int_4862)
        
        int_4864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 55), 'int')
        # Applying the binary operator '-' (line 42)
        result_sub_4865 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 49), '-', result_pow_4863, int_4864)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 39), list_4855, result_sub_4865)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 18), list_4843, list_4855)
        
        # Processing the call keyword arguments (line 42)
        # Getting the type of 'np' (line 42)
        np_4866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 66), 'np', False)
        # Obtaining the member 'int32' of a type (line 42)
        int32_4867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 66), np_4866, 'int32')
        keyword_4868 = int32_4867
        kwargs_4869 = {'dtype': keyword_4868}
        # Getting the type of 'array' (line 42)
        array_4842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'array', False)
        # Calling array(args, kwargs) (line 42)
        array_call_result_4870 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), array_4842, *[list_4843], **kwargs_4869)
        
        # Assigning a type to the variable 'a' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'a', array_call_result_4870)
        
        # Call to check_exact(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'a' (line 43)
        a_4873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 43)
        tuple_4874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 43)
        # Adding element type (line 43)
        int_4875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 29), tuple_4874, int_4875)
        # Adding element type (line 43)
        int_4876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 29), tuple_4874, int_4876)
        # Adding element type (line 43)
        int_4877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 29), tuple_4874, int_4877)
        # Adding element type (line 43)
        str_4878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 38), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 29), tuple_4874, str_4878)
        # Adding element type (line 43)
        str_4879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 47), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 29), tuple_4874, str_4879)
        # Adding element type (line 43)
        str_4880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 58), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 29), tuple_4874, str_4880)
        
        # Processing the call keyword arguments (line 43)
        kwargs_4881 = {}
        # Getting the type of 'self' (line 43)
        self_4871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 43)
        check_exact_4872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_4871, 'check_exact')
        # Calling check_exact(args, kwargs) (line 43)
        check_exact_call_result_4882 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), check_exact_4872, *[a_4873, tuple_4874], **kwargs_4881)
        
        
        # ################# End of 'test_32bit_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_32bit_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_4883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4883)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_32bit_integer'
        return stypy_return_type_4883


    @norecursion
    def test_64bit_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_64bit_integer'
        module_type_store = module_type_store.open_function_context('test_64bit_integer', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_64bit_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_64bit_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_64bit_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_64bit_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_64bit_integer')
        TestMMIOArray.test_64bit_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_64bit_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_64bit_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_64bit_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_64bit_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_64bit_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_64bit_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_64bit_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_64bit_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_64bit_integer(...)' code ##################

        
        # Assigning a Call to a Name (line 46):
        
        # Call to array(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_4885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_4886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        int_4887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'int')
        int_4888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'int')
        # Applying the binary operator '**' (line 46)
        result_pow_4889 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 20), '**', int_4887, int_4888)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_4886, result_pow_4889)
        # Adding element type (line 46)
        int_4890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 27), 'int')
        int_4891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 30), 'int')
        # Applying the binary operator '**' (line 46)
        result_pow_4892 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 27), '**', int_4890, int_4891)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_4886, result_pow_4892)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 18), list_4885, list_4886)
        # Adding element type (line 46)
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_4893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        int_4894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 36), 'int')
        int_4895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 39), 'int')
        # Applying the binary operator '**' (line 46)
        result_pow_4896 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 36), '**', int_4894, int_4895)
        
        int_4897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 42), 'int')
        # Applying the binary operator '-' (line 46)
        result_sub_4898 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 36), '-', result_pow_4896, int_4897)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 35), list_4893, result_sub_4898)
        # Adding element type (line 46)
        int_4899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 45), 'int')
        int_4900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 48), 'int')
        # Applying the binary operator '**' (line 46)
        result_pow_4901 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 45), '**', int_4899, int_4900)
        
        int_4902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 51), 'int')
        # Applying the binary operator '-' (line 46)
        result_sub_4903 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 45), '-', result_pow_4901, int_4902)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 35), list_4893, result_sub_4903)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 18), list_4885, list_4893)
        
        # Processing the call keyword arguments (line 46)
        # Getting the type of 'np' (line 46)
        np_4904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 62), 'np', False)
        # Obtaining the member 'int64' of a type (line 46)
        int64_4905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 62), np_4904, 'int64')
        keyword_4906 = int64_4905
        kwargs_4907 = {'dtype': keyword_4906}
        # Getting the type of 'array' (line 46)
        array_4884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'array', False)
        # Calling array(args, kwargs) (line 46)
        array_call_result_4908 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), array_4884, *[list_4885], **kwargs_4907)
        
        # Assigning a type to the variable 'a' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'a', array_call_result_4908)
        
        
        
        # Call to intp(...): (line 47)
        # Processing the call arguments (line 47)
        int_4911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 20), 'int')
        # Processing the call keyword arguments (line 47)
        kwargs_4912 = {}
        # Getting the type of 'np' (line 47)
        np_4909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'np', False)
        # Obtaining the member 'intp' of a type (line 47)
        intp_4910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), np_4909, 'intp')
        # Calling intp(args, kwargs) (line 47)
        intp_call_result_4913 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), intp_4910, *[int_4911], **kwargs_4912)
        
        # Obtaining the member 'itemsize' of a type (line 47)
        itemsize_4914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), intp_call_result_4913, 'itemsize')
        int_4915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 34), 'int')
        # Applying the binary operator '<' (line 47)
        result_lt_4916 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 12), '<', itemsize_4914, int_4915)
        
        # Testing the type of an if condition (line 47)
        if_condition_4917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), result_lt_4916)
        # Assigning a type to the variable 'if_condition_4917' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_4917', if_condition_4917)
        # SSA begins for if statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_raises(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'OverflowError' (line 48)
        OverflowError_4919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'OverflowError', False)
        # Getting the type of 'mmwrite' (line 48)
        mmwrite_4920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 41), 'mmwrite', False)
        # Getting the type of 'self' (line 48)
        self_4921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 50), 'self', False)
        # Obtaining the member 'fn' of a type (line 48)
        fn_4922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 50), self_4921, 'fn')
        # Getting the type of 'a' (line 48)
        a_4923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 59), 'a', False)
        # Processing the call keyword arguments (line 48)
        kwargs_4924 = {}
        # Getting the type of 'assert_raises' (line 48)
        assert_raises_4918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 48)
        assert_raises_call_result_4925 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), assert_raises_4918, *[OverflowError_4919, mmwrite_4920, fn_4922, a_4923], **kwargs_4924)
        
        # SSA branch for the else part of an if statement (line 47)
        module_type_store.open_ssa_branch('else')
        
        # Call to check_exact(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'a' (line 50)
        a_4928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 29), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 50)
        tuple_4929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 50)
        # Adding element type (line 50)
        int_4930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_4929, int_4930)
        # Adding element type (line 50)
        int_4931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_4929, int_4931)
        # Adding element type (line 50)
        int_4932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_4929, int_4932)
        # Adding element type (line 50)
        str_4933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 42), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_4929, str_4933)
        # Adding element type (line 50)
        str_4934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 51), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_4929, str_4934)
        # Adding element type (line 50)
        str_4935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 62), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_4929, str_4935)
        
        # Processing the call keyword arguments (line 50)
        kwargs_4936 = {}
        # Getting the type of 'self' (line 50)
        self_4926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 50)
        check_exact_4927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_4926, 'check_exact')
        # Calling check_exact(args, kwargs) (line 50)
        check_exact_call_result_4937 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), check_exact_4927, *[a_4928, tuple_4929], **kwargs_4936)
        
        # SSA join for if statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_64bit_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_64bit_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_4938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_64bit_integer'
        return stypy_return_type_4938


    @norecursion
    def test_simple_upper_triangle_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_upper_triangle_integer'
        module_type_store = module_type_store.open_function_context('test_simple_upper_triangle_integer', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_simple_upper_triangle_integer')
        TestMMIOArray.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_simple_upper_triangle_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_upper_triangle_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_upper_triangle_integer(...)' code ##################

        
        # Call to check_exact(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_4941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_4942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        int_4943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 26), list_4942, int_4943)
        # Adding element type (line 53)
        int_4944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 26), list_4942, int_4944)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 25), list_4941, list_4942)
        # Adding element type (line 53)
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_4945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        int_4946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 34), list_4945, int_4946)
        # Adding element type (line 53)
        int_4947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 34), list_4945, int_4947)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 25), list_4941, list_4945)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_4948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        int_4949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 26), tuple_4948, int_4949)
        # Adding element type (line 54)
        int_4950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 26), tuple_4948, int_4950)
        # Adding element type (line 54)
        int_4951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 26), tuple_4948, int_4951)
        # Adding element type (line 54)
        str_4952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 35), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 26), tuple_4948, str_4952)
        # Adding element type (line 54)
        str_4953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 44), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 26), tuple_4948, str_4953)
        # Adding element type (line 54)
        str_4954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 55), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 26), tuple_4948, str_4954)
        
        # Processing the call keyword arguments (line 53)
        kwargs_4955 = {}
        # Getting the type of 'self' (line 53)
        self_4939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 53)
        check_exact_4940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_4939, 'check_exact')
        # Calling check_exact(args, kwargs) (line 53)
        check_exact_call_result_4956 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), check_exact_4940, *[list_4941, tuple_4948], **kwargs_4955)
        
        
        # ################# End of 'test_simple_upper_triangle_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_upper_triangle_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_4957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4957)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_upper_triangle_integer'
        return stypy_return_type_4957


    @norecursion
    def test_simple_lower_triangle_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_lower_triangle_integer'
        module_type_store = module_type_store.open_function_context('test_simple_lower_triangle_integer', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_simple_lower_triangle_integer')
        TestMMIOArray.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_simple_lower_triangle_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_lower_triangle_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_lower_triangle_integer(...)' code ##################

        
        # Call to check_exact(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_4960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_4961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_4962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 26), list_4961, int_4962)
        # Adding element type (line 57)
        int_4963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 26), list_4961, int_4963)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 25), list_4960, list_4961)
        # Adding element type (line 57)
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_4964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_4965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 34), list_4964, int_4965)
        # Adding element type (line 57)
        int_4966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 34), list_4964, int_4966)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 25), list_4960, list_4964)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 58)
        tuple_4967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 58)
        # Adding element type (line 58)
        int_4968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 26), tuple_4967, int_4968)
        # Adding element type (line 58)
        int_4969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 26), tuple_4967, int_4969)
        # Adding element type (line 58)
        int_4970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 26), tuple_4967, int_4970)
        # Adding element type (line 58)
        str_4971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 35), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 26), tuple_4967, str_4971)
        # Adding element type (line 58)
        str_4972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 44), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 26), tuple_4967, str_4972)
        # Adding element type (line 58)
        str_4973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 55), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 26), tuple_4967, str_4973)
        
        # Processing the call keyword arguments (line 57)
        kwargs_4974 = {}
        # Getting the type of 'self' (line 57)
        self_4958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 57)
        check_exact_4959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_4958, 'check_exact')
        # Calling check_exact(args, kwargs) (line 57)
        check_exact_call_result_4975 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), check_exact_4959, *[list_4960, tuple_4967], **kwargs_4974)
        
        
        # ################# End of 'test_simple_lower_triangle_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_lower_triangle_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_4976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4976)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_lower_triangle_integer'
        return stypy_return_type_4976


    @norecursion
    def test_simple_rectangular_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_rectangular_integer'
        module_type_store = module_type_store.open_function_context('test_simple_rectangular_integer', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_simple_rectangular_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_simple_rectangular_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_simple_rectangular_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_simple_rectangular_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_simple_rectangular_integer')
        TestMMIOArray.test_simple_rectangular_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_simple_rectangular_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_simple_rectangular_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_simple_rectangular_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_simple_rectangular_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_simple_rectangular_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_simple_rectangular_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_simple_rectangular_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_rectangular_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_rectangular_integer(...)' code ##################

        
        # Call to check_exact(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_4979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_4980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_4981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 26), list_4980, int_4981)
        # Adding element type (line 61)
        int_4982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 26), list_4980, int_4982)
        # Adding element type (line 61)
        int_4983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 26), list_4980, int_4983)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 25), list_4979, list_4980)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_4984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_4985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 37), list_4984, int_4985)
        # Adding element type (line 61)
        int_4986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 37), list_4984, int_4986)
        # Adding element type (line 61)
        int_4987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 37), list_4984, int_4987)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 25), list_4979, list_4984)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 62)
        tuple_4988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 62)
        # Adding element type (line 62)
        int_4989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 26), tuple_4988, int_4989)
        # Adding element type (line 62)
        int_4990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 26), tuple_4988, int_4990)
        # Adding element type (line 62)
        int_4991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 26), tuple_4988, int_4991)
        # Adding element type (line 62)
        str_4992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 35), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 26), tuple_4988, str_4992)
        # Adding element type (line 62)
        str_4993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 44), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 26), tuple_4988, str_4993)
        # Adding element type (line 62)
        str_4994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 55), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 26), tuple_4988, str_4994)
        
        # Processing the call keyword arguments (line 61)
        kwargs_4995 = {}
        # Getting the type of 'self' (line 61)
        self_4977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 61)
        check_exact_4978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_4977, 'check_exact')
        # Calling check_exact(args, kwargs) (line 61)
        check_exact_call_result_4996 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), check_exact_4978, *[list_4979, tuple_4988], **kwargs_4995)
        
        
        # ################# End of 'test_simple_rectangular_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_rectangular_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_4997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4997)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_rectangular_integer'
        return stypy_return_type_4997


    @norecursion
    def test_simple_rectangular_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_rectangular_float'
        module_type_store = module_type_store.open_function_context('test_simple_rectangular_float', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_simple_rectangular_float.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_simple_rectangular_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_simple_rectangular_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_simple_rectangular_float.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_simple_rectangular_float')
        TestMMIOArray.test_simple_rectangular_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_simple_rectangular_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_simple_rectangular_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_simple_rectangular_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_simple_rectangular_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_simple_rectangular_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_simple_rectangular_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_simple_rectangular_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_rectangular_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_rectangular_float(...)' code ##################

        
        # Call to check(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_5000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_5001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        int_5002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 20), list_5001, int_5002)
        # Adding element type (line 65)
        int_5003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 20), list_5001, int_5003)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), list_5000, list_5001)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_5004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        float_5005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 28), list_5004, float_5005)
        # Adding element type (line 65)
        int_5006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 28), list_5004, int_5006)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), list_5000, list_5004)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_5007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        int_5008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 38), list_5007, int_5008)
        # Adding element type (line 65)
        int_5009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 38), list_5007, int_5009)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), list_5000, list_5007)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_5010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        int_5011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 20), tuple_5010, int_5011)
        # Adding element type (line 66)
        int_5012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 20), tuple_5010, int_5012)
        # Adding element type (line 66)
        int_5013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 20), tuple_5010, int_5013)
        # Adding element type (line 66)
        str_5014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 29), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 20), tuple_5010, str_5014)
        # Adding element type (line 66)
        str_5015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 38), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 20), tuple_5010, str_5015)
        # Adding element type (line 66)
        str_5016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 46), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 20), tuple_5010, str_5016)
        
        # Processing the call keyword arguments (line 65)
        kwargs_5017 = {}
        # Getting the type of 'self' (line 65)
        self_4998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 65)
        check_4999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_4998, 'check')
        # Calling check(args, kwargs) (line 65)
        check_call_result_5018 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), check_4999, *[list_5000, tuple_5010], **kwargs_5017)
        
        
        # ################# End of 'test_simple_rectangular_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_rectangular_float' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_5019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5019)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_rectangular_float'
        return stypy_return_type_5019


    @norecursion
    def test_simple_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_float'
        module_type_store = module_type_store.open_function_context('test_simple_float', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_simple_float.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_simple_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_simple_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_simple_float.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_simple_float')
        TestMMIOArray.test_simple_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_simple_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_simple_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_simple_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_simple_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_simple_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_simple_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_simple_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_float(...)' code ##################

        
        # Call to check(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_5022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_5023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        int_5024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 20), list_5023, int_5024)
        # Adding element type (line 69)
        int_5025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 20), list_5023, int_5025)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 19), list_5022, list_5023)
        # Adding element type (line 69)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_5026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        int_5027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 28), list_5026, int_5027)
        # Adding element type (line 69)
        float_5028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 28), list_5026, float_5028)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 19), list_5022, list_5026)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 70)
        tuple_5029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 70)
        # Adding element type (line 70)
        int_5030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 20), tuple_5029, int_5030)
        # Adding element type (line 70)
        int_5031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 20), tuple_5029, int_5031)
        # Adding element type (line 70)
        int_5032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 20), tuple_5029, int_5032)
        # Adding element type (line 70)
        str_5033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 29), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 20), tuple_5029, str_5033)
        # Adding element type (line 70)
        str_5034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 38), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 20), tuple_5029, str_5034)
        # Adding element type (line 70)
        str_5035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 46), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 20), tuple_5029, str_5035)
        
        # Processing the call keyword arguments (line 69)
        kwargs_5036 = {}
        # Getting the type of 'self' (line 69)
        self_5020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 69)
        check_5021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_5020, 'check')
        # Calling check(args, kwargs) (line 69)
        check_call_result_5037 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), check_5021, *[list_5022, tuple_5029], **kwargs_5036)
        
        
        # ################# End of 'test_simple_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_float' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_5038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5038)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_float'
        return stypy_return_type_5038


    @norecursion
    def test_simple_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_complex'
        module_type_store = module_type_store.open_function_context('test_simple_complex', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_simple_complex.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_simple_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_simple_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_simple_complex.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_simple_complex')
        TestMMIOArray.test_simple_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_simple_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_simple_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_simple_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_simple_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_simple_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_simple_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_simple_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_complex(...)' code ##################

        
        # Call to check(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_5041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_5042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        int_5043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 20), list_5042, int_5043)
        # Adding element type (line 73)
        int_5044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 20), list_5042, int_5044)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 19), list_5041, list_5042)
        # Adding element type (line 73)
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_5045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        int_5046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), list_5045, int_5046)
        # Adding element type (line 73)
        complex_5047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 32), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), list_5045, complex_5047)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 19), list_5041, list_5045)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 74)
        tuple_5048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 74)
        # Adding element type (line 74)
        int_5049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 20), tuple_5048, int_5049)
        # Adding element type (line 74)
        int_5050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 20), tuple_5048, int_5050)
        # Adding element type (line 74)
        int_5051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 20), tuple_5048, int_5051)
        # Adding element type (line 74)
        str_5052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 29), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 20), tuple_5048, str_5052)
        # Adding element type (line 74)
        str_5053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 38), 'str', 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 20), tuple_5048, str_5053)
        # Adding element type (line 74)
        str_5054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 49), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 20), tuple_5048, str_5054)
        
        # Processing the call keyword arguments (line 73)
        kwargs_5055 = {}
        # Getting the type of 'self' (line 73)
        self_5039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 73)
        check_5040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_5039, 'check')
        # Calling check(args, kwargs) (line 73)
        check_call_result_5056 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), check_5040, *[list_5041, tuple_5048], **kwargs_5055)
        
        
        # ################# End of 'test_simple_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_5057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5057)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_complex'
        return stypy_return_type_5057


    @norecursion
    def test_simple_symmetric_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_symmetric_integer'
        module_type_store = module_type_store.open_function_context('test_simple_symmetric_integer', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_simple_symmetric_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_simple_symmetric_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_simple_symmetric_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_simple_symmetric_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_simple_symmetric_integer')
        TestMMIOArray.test_simple_symmetric_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_simple_symmetric_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_simple_symmetric_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_simple_symmetric_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_simple_symmetric_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_simple_symmetric_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_simple_symmetric_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_simple_symmetric_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_symmetric_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_symmetric_integer(...)' code ##################

        
        # Call to check_exact(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_5060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_5061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        int_5062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 26), list_5061, int_5062)
        # Adding element type (line 77)
        int_5063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 26), list_5061, int_5063)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_5060, list_5061)
        # Adding element type (line 77)
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_5064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        int_5065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), list_5064, int_5065)
        # Adding element type (line 77)
        int_5066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), list_5064, int_5066)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_5060, list_5064)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 78)
        tuple_5067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 78)
        # Adding element type (line 78)
        int_5068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), tuple_5067, int_5068)
        # Adding element type (line 78)
        int_5069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), tuple_5067, int_5069)
        # Adding element type (line 78)
        int_5070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), tuple_5067, int_5070)
        # Adding element type (line 78)
        str_5071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 35), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), tuple_5067, str_5071)
        # Adding element type (line 78)
        str_5072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 44), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), tuple_5067, str_5072)
        # Adding element type (line 78)
        str_5073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 55), 'str', 'symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), tuple_5067, str_5073)
        
        # Processing the call keyword arguments (line 77)
        kwargs_5074 = {}
        # Getting the type of 'self' (line 77)
        self_5058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 77)
        check_exact_5059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_5058, 'check_exact')
        # Calling check_exact(args, kwargs) (line 77)
        check_exact_call_result_5075 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), check_exact_5059, *[list_5060, tuple_5067], **kwargs_5074)
        
        
        # ################# End of 'test_simple_symmetric_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_symmetric_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_5076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5076)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_symmetric_integer'
        return stypy_return_type_5076


    @norecursion
    def test_simple_skew_symmetric_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_skew_symmetric_integer'
        module_type_store = module_type_store.open_function_context('test_simple_skew_symmetric_integer', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_simple_skew_symmetric_integer')
        TestMMIOArray.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_simple_skew_symmetric_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_skew_symmetric_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_skew_symmetric_integer(...)' code ##################

        
        # Call to check_exact(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_5079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_5080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        int_5081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 26), list_5080, int_5081)
        # Adding element type (line 81)
        int_5082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 26), list_5080, int_5082)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), list_5079, list_5080)
        # Adding element type (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_5083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        int_5084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 34), list_5083, int_5084)
        # Adding element type (line 81)
        int_5085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 34), list_5083, int_5085)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), list_5079, list_5083)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_5086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        int_5087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 26), tuple_5086, int_5087)
        # Adding element type (line 82)
        int_5088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 26), tuple_5086, int_5088)
        # Adding element type (line 82)
        int_5089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 26), tuple_5086, int_5089)
        # Adding element type (line 82)
        str_5090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 35), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 26), tuple_5086, str_5090)
        # Adding element type (line 82)
        str_5091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 44), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 26), tuple_5086, str_5091)
        # Adding element type (line 82)
        str_5092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 55), 'str', 'skew-symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 26), tuple_5086, str_5092)
        
        # Processing the call keyword arguments (line 81)
        kwargs_5093 = {}
        # Getting the type of 'self' (line 81)
        self_5077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 81)
        check_exact_5078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_5077, 'check_exact')
        # Calling check_exact(args, kwargs) (line 81)
        check_exact_call_result_5094 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), check_exact_5078, *[list_5079, tuple_5086], **kwargs_5093)
        
        
        # ################# End of 'test_simple_skew_symmetric_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_skew_symmetric_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_5095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5095)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_skew_symmetric_integer'
        return stypy_return_type_5095


    @norecursion
    def test_simple_skew_symmetric_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_skew_symmetric_float'
        module_type_store = module_type_store.open_function_context('test_simple_skew_symmetric_float', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_simple_skew_symmetric_float')
        TestMMIOArray.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_simple_skew_symmetric_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_skew_symmetric_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_skew_symmetric_float(...)' code ##################

        
        # Call to check(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to array(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_5099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_5100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        int_5101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 26), list_5100, int_5101)
        # Adding element type (line 85)
        int_5102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 26), list_5100, int_5102)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 25), list_5099, list_5100)
        # Adding element type (line 85)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_5103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        float_5104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 34), list_5103, float_5104)
        # Adding element type (line 85)
        int_5105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 34), list_5103, int_5105)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 25), list_5099, list_5103)
        
        str_5106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 46), 'str', 'f')
        # Processing the call keyword arguments (line 85)
        kwargs_5107 = {}
        # Getting the type of 'array' (line 85)
        array_5098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'array', False)
        # Calling array(args, kwargs) (line 85)
        array_call_result_5108 = invoke(stypy.reporting.localization.Localization(__file__, 85, 19), array_5098, *[list_5099, str_5106], **kwargs_5107)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_5109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        int_5110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 20), tuple_5109, int_5110)
        # Adding element type (line 86)
        int_5111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 20), tuple_5109, int_5111)
        # Adding element type (line 86)
        int_5112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 20), tuple_5109, int_5112)
        # Adding element type (line 86)
        str_5113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 20), tuple_5109, str_5113)
        # Adding element type (line 86)
        str_5114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 38), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 20), tuple_5109, str_5114)
        # Adding element type (line 86)
        str_5115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 46), 'str', 'skew-symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 20), tuple_5109, str_5115)
        
        # Processing the call keyword arguments (line 85)
        kwargs_5116 = {}
        # Getting the type of 'self' (line 85)
        self_5096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 85)
        check_5097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_5096, 'check')
        # Calling check(args, kwargs) (line 85)
        check_call_result_5117 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), check_5097, *[array_call_result_5108, tuple_5109], **kwargs_5116)
        
        
        # ################# End of 'test_simple_skew_symmetric_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_skew_symmetric_float' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_5118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5118)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_skew_symmetric_float'
        return stypy_return_type_5118


    @norecursion
    def test_simple_hermitian_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_hermitian_complex'
        module_type_store = module_type_store.open_function_context('test_simple_hermitian_complex', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_simple_hermitian_complex.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_simple_hermitian_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_simple_hermitian_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_simple_hermitian_complex.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_simple_hermitian_complex')
        TestMMIOArray.test_simple_hermitian_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_simple_hermitian_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_simple_hermitian_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_simple_hermitian_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_simple_hermitian_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_simple_hermitian_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_simple_hermitian_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_simple_hermitian_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_hermitian_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_hermitian_complex(...)' code ##################

        
        # Call to check(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_5121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_5122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        int_5123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 20), list_5122, int_5123)
        # Adding element type (line 89)
        int_5124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 24), 'int')
        complex_5125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 26), 'complex')
        # Applying the binary operator '+' (line 89)
        result_add_5126 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 24), '+', int_5124, complex_5125)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 20), list_5122, result_add_5126)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 19), list_5121, list_5122)
        # Adding element type (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_5127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        int_5128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 32), 'int')
        complex_5129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 34), 'complex')
        # Applying the binary operator '-' (line 89)
        result_sub_5130 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 32), '-', int_5128, complex_5129)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 31), list_5127, result_sub_5130)
        # Adding element type (line 89)
        int_5131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 31), list_5127, int_5131)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 19), list_5121, list_5127)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 90)
        tuple_5132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 90)
        # Adding element type (line 90)
        int_5133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 20), tuple_5132, int_5133)
        # Adding element type (line 90)
        int_5134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 20), tuple_5132, int_5134)
        # Adding element type (line 90)
        int_5135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 20), tuple_5132, int_5135)
        # Adding element type (line 90)
        str_5136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 29), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 20), tuple_5132, str_5136)
        # Adding element type (line 90)
        str_5137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 38), 'str', 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 20), tuple_5132, str_5137)
        # Adding element type (line 90)
        str_5138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 49), 'str', 'hermitian')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 20), tuple_5132, str_5138)
        
        # Processing the call keyword arguments (line 89)
        kwargs_5139 = {}
        # Getting the type of 'self' (line 89)
        self_5119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 89)
        check_5120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_5119, 'check')
        # Calling check(args, kwargs) (line 89)
        check_call_result_5140 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), check_5120, *[list_5121, tuple_5132], **kwargs_5139)
        
        
        # ################# End of 'test_simple_hermitian_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_hermitian_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_5141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5141)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_hermitian_complex'
        return stypy_return_type_5141


    @norecursion
    def test_random_symmetric_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_symmetric_float'
        module_type_store = module_type_store.open_function_context('test_random_symmetric_float', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_random_symmetric_float.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_random_symmetric_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_random_symmetric_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_random_symmetric_float.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_random_symmetric_float')
        TestMMIOArray.test_random_symmetric_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_random_symmetric_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_random_symmetric_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_random_symmetric_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_random_symmetric_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_random_symmetric_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_random_symmetric_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_random_symmetric_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_symmetric_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_symmetric_float(...)' code ##################

        
        # Assigning a Tuple to a Name (line 93):
        
        # Obtaining an instance of the builtin type 'tuple' (line 93)
        tuple_5142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 93)
        # Adding element type (line 93)
        int_5143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 14), tuple_5142, int_5143)
        # Adding element type (line 93)
        int_5144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 14), tuple_5142, int_5144)
        
        # Assigning a type to the variable 'sz' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'sz', tuple_5142)
        
        # Assigning a Call to a Name (line 94):
        
        # Call to random(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'sz' (line 94)
        sz_5148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 29), 'sz', False)
        # Processing the call keyword arguments (line 94)
        kwargs_5149 = {}
        # Getting the type of 'np' (line 94)
        np_5145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 94)
        random_5146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), np_5145, 'random')
        # Obtaining the member 'random' of a type (line 94)
        random_5147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), random_5146, 'random')
        # Calling random(args, kwargs) (line 94)
        random_call_result_5150 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), random_5147, *[sz_5148], **kwargs_5149)
        
        # Assigning a type to the variable 'a' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'a', random_call_result_5150)
        
        # Assigning a BinOp to a Name (line 95):
        # Getting the type of 'a' (line 95)
        a_5151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'a')
        
        # Call to transpose(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'a' (line 95)
        a_5153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'a', False)
        # Processing the call keyword arguments (line 95)
        kwargs_5154 = {}
        # Getting the type of 'transpose' (line 95)
        transpose_5152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'transpose', False)
        # Calling transpose(args, kwargs) (line 95)
        transpose_call_result_5155 = invoke(stypy.reporting.localization.Localization(__file__, 95, 16), transpose_5152, *[a_5153], **kwargs_5154)
        
        # Applying the binary operator '+' (line 95)
        result_add_5156 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 12), '+', a_5151, transpose_call_result_5155)
        
        # Assigning a type to the variable 'a' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'a', result_add_5156)
        
        # Call to check(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'a' (line 96)
        a_5159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 96)
        tuple_5160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 96)
        # Adding element type (line 96)
        int_5161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), tuple_5160, int_5161)
        # Adding element type (line 96)
        int_5162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), tuple_5160, int_5162)
        # Adding element type (line 96)
        int_5163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), tuple_5160, int_5163)
        # Adding element type (line 96)
        str_5164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 36), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), tuple_5160, str_5164)
        # Adding element type (line 96)
        str_5165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 45), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), tuple_5160, str_5165)
        # Adding element type (line 96)
        str_5166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 53), 'str', 'symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), tuple_5160, str_5166)
        
        # Processing the call keyword arguments (line 96)
        kwargs_5167 = {}
        # Getting the type of 'self' (line 96)
        self_5157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 96)
        check_5158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_5157, 'check')
        # Calling check(args, kwargs) (line 96)
        check_call_result_5168 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), check_5158, *[a_5159, tuple_5160], **kwargs_5167)
        
        
        # ################# End of 'test_random_symmetric_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_symmetric_float' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_5169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_symmetric_float'
        return stypy_return_type_5169


    @norecursion
    def test_random_rectangular_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_rectangular_float'
        module_type_store = module_type_store.open_function_context('test_random_rectangular_float', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOArray.test_random_rectangular_float.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOArray.test_random_rectangular_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOArray.test_random_rectangular_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOArray.test_random_rectangular_float.__dict__.__setitem__('stypy_function_name', 'TestMMIOArray.test_random_rectangular_float')
        TestMMIOArray.test_random_rectangular_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOArray.test_random_rectangular_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOArray.test_random_rectangular_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOArray.test_random_rectangular_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOArray.test_random_rectangular_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOArray.test_random_rectangular_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOArray.test_random_rectangular_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.test_random_rectangular_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_rectangular_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_rectangular_float(...)' code ##################

        
        # Assigning a Tuple to a Name (line 99):
        
        # Obtaining an instance of the builtin type 'tuple' (line 99)
        tuple_5170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 99)
        # Adding element type (line 99)
        int_5171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), tuple_5170, int_5171)
        # Adding element type (line 99)
        int_5172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), tuple_5170, int_5172)
        
        # Assigning a type to the variable 'sz' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'sz', tuple_5170)
        
        # Assigning a Call to a Name (line 100):
        
        # Call to random(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'sz' (line 100)
        sz_5176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 29), 'sz', False)
        # Processing the call keyword arguments (line 100)
        kwargs_5177 = {}
        # Getting the type of 'np' (line 100)
        np_5173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 100)
        random_5174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), np_5173, 'random')
        # Obtaining the member 'random' of a type (line 100)
        random_5175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), random_5174, 'random')
        # Calling random(args, kwargs) (line 100)
        random_call_result_5178 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), random_5175, *[sz_5176], **kwargs_5177)
        
        # Assigning a type to the variable 'a' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'a', random_call_result_5178)
        
        # Call to check(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'a' (line 101)
        a_5181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 101)
        tuple_5182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 101)
        # Adding element type (line 101)
        int_5183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 23), tuple_5182, int_5183)
        # Adding element type (line 101)
        int_5184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 23), tuple_5182, int_5184)
        # Adding element type (line 101)
        int_5185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 23), tuple_5182, int_5185)
        # Adding element type (line 101)
        str_5186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 36), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 23), tuple_5182, str_5186)
        # Adding element type (line 101)
        str_5187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 45), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 23), tuple_5182, str_5187)
        # Adding element type (line 101)
        str_5188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 53), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 23), tuple_5182, str_5188)
        
        # Processing the call keyword arguments (line 101)
        kwargs_5189 = {}
        # Getting the type of 'self' (line 101)
        self_5179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 101)
        check_5180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_5179, 'check')
        # Calling check(args, kwargs) (line 101)
        check_call_result_5190 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), check_5180, *[a_5181, tuple_5182], **kwargs_5189)
        
        
        # ################# End of 'test_random_rectangular_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_rectangular_float' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_5191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5191)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_rectangular_float'
        return stypy_return_type_5191


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 0, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOArray.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestMMIOArray' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'TestMMIOArray', TestMMIOArray)
# Declaration of the 'TestMMIOSparseCSR' class
# Getting the type of 'TestMMIOArray' (line 104)
TestMMIOArray_5192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'TestMMIOArray')

class TestMMIOSparseCSR(TestMMIOArray_5192, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.setup_method.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.setup_method')
        TestMMIOSparseCSR.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 106):
        
        # Call to mkdtemp(...): (line 106)
        # Processing the call keyword arguments (line 106)
        kwargs_5194 = {}
        # Getting the type of 'mkdtemp' (line 106)
        mkdtemp_5193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'mkdtemp', False)
        # Calling mkdtemp(args, kwargs) (line 106)
        mkdtemp_call_result_5195 = invoke(stypy.reporting.localization.Localization(__file__, 106, 22), mkdtemp_5193, *[], **kwargs_5194)
        
        # Getting the type of 'self' (line 106)
        self_5196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self')
        # Setting the type of the member 'tmpdir' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_5196, 'tmpdir', mkdtemp_call_result_5195)
        
        # Assigning a Call to a Attribute (line 107):
        
        # Call to join(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'self' (line 107)
        self_5200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 31), 'self', False)
        # Obtaining the member 'tmpdir' of a type (line 107)
        tmpdir_5201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 31), self_5200, 'tmpdir')
        str_5202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 44), 'str', 'testfile.mtx')
        # Processing the call keyword arguments (line 107)
        kwargs_5203 = {}
        # Getting the type of 'os' (line 107)
        os_5197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 107)
        path_5198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 18), os_5197, 'path')
        # Obtaining the member 'join' of a type (line 107)
        join_5199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 18), path_5198, 'join')
        # Calling join(args, kwargs) (line 107)
        join_call_result_5204 = invoke(stypy.reporting.localization.Localization(__file__, 107, 18), join_5199, *[tmpdir_5201, str_5202], **kwargs_5203)
        
        # Getting the type of 'self' (line 107)
        self_5205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'self')
        # Setting the type of the member 'fn' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), self_5205, 'fn', join_call_result_5204)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_5206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5206)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_5206


    @norecursion
    def teardown_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'teardown_method'
        module_type_store = module_type_store.open_function_context('teardown_method', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.teardown_method.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.teardown_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.teardown_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.teardown_method.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.teardown_method')
        TestMMIOSparseCSR.teardown_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.teardown_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.teardown_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.teardown_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.teardown_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.teardown_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.teardown_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.teardown_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'teardown_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'teardown_method(...)' code ##################

        
        # Call to rmtree(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'self' (line 110)
        self_5209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'self', False)
        # Obtaining the member 'tmpdir' of a type (line 110)
        tmpdir_5210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 22), self_5209, 'tmpdir')
        # Processing the call keyword arguments (line 110)
        kwargs_5211 = {}
        # Getting the type of 'shutil' (line 110)
        shutil_5207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'shutil', False)
        # Obtaining the member 'rmtree' of a type (line 110)
        rmtree_5208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), shutil_5207, 'rmtree')
        # Calling rmtree(args, kwargs) (line 110)
        rmtree_call_result_5212 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), rmtree_5208, *[tmpdir_5210], **kwargs_5211)
        
        
        # ################# End of 'teardown_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'teardown_method' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_5213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5213)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'teardown_method'
        return stypy_return_type_5213


    @norecursion
    def check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.check.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.check.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.check.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.check')
        TestMMIOSparseCSR.check.__dict__.__setitem__('stypy_param_names_list', ['a', 'info'])
        TestMMIOSparseCSR.check.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.check.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.check.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.check.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.check.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.check', ['a', 'info'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['a', 'info'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Call to mmwrite(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'self' (line 113)
        self_5215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'self', False)
        # Obtaining the member 'fn' of a type (line 113)
        fn_5216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), self_5215, 'fn')
        # Getting the type of 'a' (line 113)
        a_5217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'a', False)
        # Processing the call keyword arguments (line 113)
        kwargs_5218 = {}
        # Getting the type of 'mmwrite' (line 113)
        mmwrite_5214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 113)
        mmwrite_call_result_5219 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), mmwrite_5214, *[fn_5216, a_5217], **kwargs_5218)
        
        
        # Call to assert_equal(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to mminfo(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'self' (line 114)
        self_5222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'self', False)
        # Obtaining the member 'fn' of a type (line 114)
        fn_5223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 28), self_5222, 'fn')
        # Processing the call keyword arguments (line 114)
        kwargs_5224 = {}
        # Getting the type of 'mminfo' (line 114)
        mminfo_5221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'mminfo', False)
        # Calling mminfo(args, kwargs) (line 114)
        mminfo_call_result_5225 = invoke(stypy.reporting.localization.Localization(__file__, 114, 21), mminfo_5221, *[fn_5223], **kwargs_5224)
        
        # Getting the type of 'info' (line 114)
        info_5226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 38), 'info', False)
        # Processing the call keyword arguments (line 114)
        kwargs_5227 = {}
        # Getting the type of 'assert_equal' (line 114)
        assert_equal_5220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 114)
        assert_equal_call_result_5228 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), assert_equal_5220, *[mminfo_call_result_5225, info_5226], **kwargs_5227)
        
        
        # Assigning a Call to a Name (line 115):
        
        # Call to mmread(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'self' (line 115)
        self_5230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'self', False)
        # Obtaining the member 'fn' of a type (line 115)
        fn_5231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 19), self_5230, 'fn')
        # Processing the call keyword arguments (line 115)
        kwargs_5232 = {}
        # Getting the type of 'mmread' (line 115)
        mmread_5229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'mmread', False)
        # Calling mmread(args, kwargs) (line 115)
        mmread_call_result_5233 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), mmread_5229, *[fn_5231], **kwargs_5232)
        
        # Assigning a type to the variable 'b' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'b', mmread_call_result_5233)
        
        # Call to assert_array_almost_equal(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Call to todense(...): (line 116)
        # Processing the call keyword arguments (line 116)
        kwargs_5237 = {}
        # Getting the type of 'a' (line 116)
        a_5235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'a', False)
        # Obtaining the member 'todense' of a type (line 116)
        todense_5236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 34), a_5235, 'todense')
        # Calling todense(args, kwargs) (line 116)
        todense_call_result_5238 = invoke(stypy.reporting.localization.Localization(__file__, 116, 34), todense_5236, *[], **kwargs_5237)
        
        
        # Call to todense(...): (line 116)
        # Processing the call keyword arguments (line 116)
        kwargs_5241 = {}
        # Getting the type of 'b' (line 116)
        b_5239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 47), 'b', False)
        # Obtaining the member 'todense' of a type (line 116)
        todense_5240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 47), b_5239, 'todense')
        # Calling todense(args, kwargs) (line 116)
        todense_call_result_5242 = invoke(stypy.reporting.localization.Localization(__file__, 116, 47), todense_5240, *[], **kwargs_5241)
        
        # Processing the call keyword arguments (line 116)
        kwargs_5243 = {}
        # Getting the type of 'assert_array_almost_equal' (line 116)
        assert_array_almost_equal_5234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 116)
        assert_array_almost_equal_call_result_5244 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), assert_array_almost_equal_5234, *[todense_call_result_5238, todense_call_result_5242], **kwargs_5243)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_5245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5245)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_5245


    @norecursion
    def check_exact(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_exact'
        module_type_store = module_type_store.open_function_context('check_exact', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.check_exact.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.check_exact.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.check_exact.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.check_exact.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.check_exact')
        TestMMIOSparseCSR.check_exact.__dict__.__setitem__('stypy_param_names_list', ['a', 'info'])
        TestMMIOSparseCSR.check_exact.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.check_exact.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.check_exact.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.check_exact.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.check_exact.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.check_exact.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.check_exact', ['a', 'info'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_exact', localization, ['a', 'info'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_exact(...)' code ##################

        
        # Call to mmwrite(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'self' (line 119)
        self_5247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'self', False)
        # Obtaining the member 'fn' of a type (line 119)
        fn_5248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 16), self_5247, 'fn')
        # Getting the type of 'a' (line 119)
        a_5249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'a', False)
        # Processing the call keyword arguments (line 119)
        kwargs_5250 = {}
        # Getting the type of 'mmwrite' (line 119)
        mmwrite_5246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 119)
        mmwrite_call_result_5251 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), mmwrite_5246, *[fn_5248, a_5249], **kwargs_5250)
        
        
        # Call to assert_equal(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to mminfo(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'self' (line 120)
        self_5254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'self', False)
        # Obtaining the member 'fn' of a type (line 120)
        fn_5255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 28), self_5254, 'fn')
        # Processing the call keyword arguments (line 120)
        kwargs_5256 = {}
        # Getting the type of 'mminfo' (line 120)
        mminfo_5253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'mminfo', False)
        # Calling mminfo(args, kwargs) (line 120)
        mminfo_call_result_5257 = invoke(stypy.reporting.localization.Localization(__file__, 120, 21), mminfo_5253, *[fn_5255], **kwargs_5256)
        
        # Getting the type of 'info' (line 120)
        info_5258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 38), 'info', False)
        # Processing the call keyword arguments (line 120)
        kwargs_5259 = {}
        # Getting the type of 'assert_equal' (line 120)
        assert_equal_5252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 120)
        assert_equal_call_result_5260 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), assert_equal_5252, *[mminfo_call_result_5257, info_5258], **kwargs_5259)
        
        
        # Assigning a Call to a Name (line 121):
        
        # Call to mmread(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'self' (line 121)
        self_5262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'self', False)
        # Obtaining the member 'fn' of a type (line 121)
        fn_5263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 19), self_5262, 'fn')
        # Processing the call keyword arguments (line 121)
        kwargs_5264 = {}
        # Getting the type of 'mmread' (line 121)
        mmread_5261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'mmread', False)
        # Calling mmread(args, kwargs) (line 121)
        mmread_call_result_5265 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), mmread_5261, *[fn_5263], **kwargs_5264)
        
        # Assigning a type to the variable 'b' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'b', mmread_call_result_5265)
        
        # Call to assert_equal(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Call to todense(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_5269 = {}
        # Getting the type of 'a' (line 122)
        a_5267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'a', False)
        # Obtaining the member 'todense' of a type (line 122)
        todense_5268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 21), a_5267, 'todense')
        # Calling todense(args, kwargs) (line 122)
        todense_call_result_5270 = invoke(stypy.reporting.localization.Localization(__file__, 122, 21), todense_5268, *[], **kwargs_5269)
        
        
        # Call to todense(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_5273 = {}
        # Getting the type of 'b' (line 122)
        b_5271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 34), 'b', False)
        # Obtaining the member 'todense' of a type (line 122)
        todense_5272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 34), b_5271, 'todense')
        # Calling todense(args, kwargs) (line 122)
        todense_call_result_5274 = invoke(stypy.reporting.localization.Localization(__file__, 122, 34), todense_5272, *[], **kwargs_5273)
        
        # Processing the call keyword arguments (line 122)
        kwargs_5275 = {}
        # Getting the type of 'assert_equal' (line 122)
        assert_equal_5266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 122)
        assert_equal_call_result_5276 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), assert_equal_5266, *[todense_call_result_5270, todense_call_result_5274], **kwargs_5275)
        
        
        # ################# End of 'check_exact(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_exact' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_5277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5277)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_exact'
        return stypy_return_type_5277


    @norecursion
    def test_simple_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_integer'
        module_type_store = module_type_store.open_function_context('test_simple_integer', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_integer')
        TestMMIOSparseCSR.test_simple_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_integer(...)' code ##################

        
        # Call to check_exact(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to csr_matrix(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_5283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        # Adding element type (line 125)
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_5284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        # Adding element type (line 125)
        int_5285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 50), list_5284, int_5285)
        # Adding element type (line 125)
        int_5286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 50), list_5284, int_5286)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 49), list_5283, list_5284)
        # Adding element type (line 125)
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_5287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        # Adding element type (line 125)
        int_5288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 58), list_5287, int_5288)
        # Adding element type (line 125)
        int_5289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 58), list_5287, int_5289)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 49), list_5283, list_5287)
        
        # Processing the call keyword arguments (line 125)
        kwargs_5290 = {}
        # Getting the type of 'scipy' (line 125)
        scipy_5280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 125)
        sparse_5281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 25), scipy_5280, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 125)
        csr_matrix_5282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 25), sparse_5281, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 125)
        csr_matrix_call_result_5291 = invoke(stypy.reporting.localization.Localization(__file__, 125, 25), csr_matrix_5282, *[list_5283], **kwargs_5290)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 126)
        tuple_5292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 126)
        # Adding element type (line 126)
        int_5293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 26), tuple_5292, int_5293)
        # Adding element type (line 126)
        int_5294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 26), tuple_5292, int_5294)
        # Adding element type (line 126)
        int_5295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 26), tuple_5292, int_5295)
        # Adding element type (line 126)
        str_5296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 35), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 26), tuple_5292, str_5296)
        # Adding element type (line 126)
        str_5297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 49), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 26), tuple_5292, str_5297)
        # Adding element type (line 126)
        str_5298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 60), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 26), tuple_5292, str_5298)
        
        # Processing the call keyword arguments (line 125)
        kwargs_5299 = {}
        # Getting the type of 'self' (line 125)
        self_5278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 125)
        check_exact_5279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_5278, 'check_exact')
        # Calling check_exact(args, kwargs) (line 125)
        check_exact_call_result_5300 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), check_exact_5279, *[csr_matrix_call_result_5291, tuple_5292], **kwargs_5299)
        
        
        # ################# End of 'test_simple_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_5301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5301)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_integer'
        return stypy_return_type_5301


    @norecursion
    def test_32bit_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_32bit_integer'
        module_type_store = module_type_store.open_function_context('test_32bit_integer', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_32bit_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_32bit_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_32bit_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_32bit_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_32bit_integer')
        TestMMIOSparseCSR.test_32bit_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_32bit_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_32bit_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_32bit_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_32bit_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_32bit_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_32bit_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_32bit_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_32bit_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_32bit_integer(...)' code ##################

        
        # Assigning a Call to a Name (line 129):
        
        # Call to csr_matrix(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Call to array(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_5306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_5307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        int_5308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 44), 'int')
        int_5309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 47), 'int')
        # Applying the binary operator '**' (line 129)
        result_pow_5310 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 44), '**', int_5308, int_5309)
        
        int_5311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 50), 'int')
        # Applying the binary operator '-' (line 129)
        result_sub_5312 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 44), '-', result_pow_5310, int_5311)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 43), list_5307, result_sub_5312)
        # Adding element type (line 129)
        
        int_5313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 54), 'int')
        int_5314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 57), 'int')
        # Applying the binary operator '**' (line 129)
        result_pow_5315 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 54), '**', int_5313, int_5314)
        
        # Applying the 'usub' unary operator (line 129)
        result___neg___5316 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 53), 'usub', result_pow_5315)
        
        int_5317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 60), 'int')
        # Applying the binary operator '+' (line 129)
        result_add_5318 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 53), '+', result___neg___5316, int_5317)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 43), list_5307, result_add_5318)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 42), list_5306, list_5307)
        # Adding element type (line 129)
        
        # Obtaining an instance of the builtin type 'list' (line 130)
        list_5319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 130)
        # Adding element type (line 130)
        int_5320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 44), 'int')
        int_5321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 47), 'int')
        # Applying the binary operator '**' (line 130)
        result_pow_5322 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 44), '**', int_5320, int_5321)
        
        int_5323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 50), 'int')
        # Applying the binary operator '-' (line 130)
        result_sub_5324 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 44), '-', result_pow_5322, int_5323)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 43), list_5319, result_sub_5324)
        # Adding element type (line 130)
        int_5325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 53), 'int')
        int_5326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 56), 'int')
        # Applying the binary operator '**' (line 130)
        result_pow_5327 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 53), '**', int_5325, int_5326)
        
        int_5328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 59), 'int')
        # Applying the binary operator '-' (line 130)
        result_sub_5329 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 53), '-', result_pow_5327, int_5328)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 43), list_5319, result_sub_5329)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 42), list_5306, list_5319)
        
        # Processing the call keyword arguments (line 129)
        # Getting the type of 'np' (line 131)
        np_5330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 48), 'np', False)
        # Obtaining the member 'int32' of a type (line 131)
        int32_5331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 48), np_5330, 'int32')
        keyword_5332 = int32_5331
        kwargs_5333 = {'dtype': keyword_5332}
        # Getting the type of 'array' (line 129)
        array_5305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 36), 'array', False)
        # Calling array(args, kwargs) (line 129)
        array_call_result_5334 = invoke(stypy.reporting.localization.Localization(__file__, 129, 36), array_5305, *[list_5306], **kwargs_5333)
        
        # Processing the call keyword arguments (line 129)
        kwargs_5335 = {}
        # Getting the type of 'scipy' (line 129)
        scipy_5302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 129)
        sparse_5303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), scipy_5302, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 129)
        csr_matrix_5304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), sparse_5303, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 129)
        csr_matrix_call_result_5336 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), csr_matrix_5304, *[array_call_result_5334], **kwargs_5335)
        
        # Assigning a type to the variable 'a' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'a', csr_matrix_call_result_5336)
        
        # Call to check_exact(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'a' (line 132)
        a_5339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 25), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 132)
        tuple_5340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 132)
        # Adding element type (line 132)
        int_5341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 29), tuple_5340, int_5341)
        # Adding element type (line 132)
        int_5342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 29), tuple_5340, int_5342)
        # Adding element type (line 132)
        int_5343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 29), tuple_5340, int_5343)
        # Adding element type (line 132)
        str_5344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 38), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 29), tuple_5340, str_5344)
        # Adding element type (line 132)
        str_5345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 52), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 29), tuple_5340, str_5345)
        # Adding element type (line 132)
        str_5346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 63), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 29), tuple_5340, str_5346)
        
        # Processing the call keyword arguments (line 132)
        kwargs_5347 = {}
        # Getting the type of 'self' (line 132)
        self_5337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 132)
        check_exact_5338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_5337, 'check_exact')
        # Calling check_exact(args, kwargs) (line 132)
        check_exact_call_result_5348 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), check_exact_5338, *[a_5339, tuple_5340], **kwargs_5347)
        
        
        # ################# End of 'test_32bit_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_32bit_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_5349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5349)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_32bit_integer'
        return stypy_return_type_5349


    @norecursion
    def test_64bit_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_64bit_integer'
        module_type_store = module_type_store.open_function_context('test_64bit_integer', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_64bit_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_64bit_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_64bit_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_64bit_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_64bit_integer')
        TestMMIOSparseCSR.test_64bit_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_64bit_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_64bit_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_64bit_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_64bit_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_64bit_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_64bit_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_64bit_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_64bit_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_64bit_integer(...)' code ##################

        
        # Assigning a Call to a Name (line 135):
        
        # Call to csr_matrix(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Call to array(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_5354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_5355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        int_5356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 44), 'int')
        int_5357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 47), 'int')
        # Applying the binary operator '**' (line 135)
        result_pow_5358 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 44), '**', int_5356, int_5357)
        
        int_5359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 50), 'int')
        # Applying the binary operator '+' (line 135)
        result_add_5360 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 44), '+', result_pow_5358, int_5359)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 43), list_5355, result_add_5360)
        # Adding element type (line 135)
        int_5361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 53), 'int')
        int_5362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 56), 'int')
        # Applying the binary operator '**' (line 135)
        result_pow_5363 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 53), '**', int_5361, int_5362)
        
        int_5364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 59), 'int')
        # Applying the binary operator '+' (line 135)
        result_add_5365 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 53), '+', result_pow_5363, int_5364)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 43), list_5355, result_add_5365)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 42), list_5354, list_5355)
        # Adding element type (line 135)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_5366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        
        int_5367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 45), 'int')
        int_5368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 48), 'int')
        # Applying the binary operator '**' (line 136)
        result_pow_5369 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 45), '**', int_5367, int_5368)
        
        # Applying the 'usub' unary operator (line 136)
        result___neg___5370 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 44), 'usub', result_pow_5369)
        
        int_5371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 51), 'int')
        # Applying the binary operator '+' (line 136)
        result_add_5372 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 44), '+', result___neg___5370, int_5371)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 43), list_5366, result_add_5372)
        # Adding element type (line 136)
        int_5373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 54), 'int')
        int_5374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 57), 'int')
        # Applying the binary operator '**' (line 136)
        result_pow_5375 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 54), '**', int_5373, int_5374)
        
        int_5376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 60), 'int')
        # Applying the binary operator '-' (line 136)
        result_sub_5377 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 54), '-', result_pow_5375, int_5376)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 43), list_5366, result_sub_5377)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 42), list_5354, list_5366)
        
        # Processing the call keyword arguments (line 135)
        # Getting the type of 'np' (line 137)
        np_5378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 48), 'np', False)
        # Obtaining the member 'int64' of a type (line 137)
        int64_5379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 48), np_5378, 'int64')
        keyword_5380 = int64_5379
        kwargs_5381 = {'dtype': keyword_5380}
        # Getting the type of 'array' (line 135)
        array_5353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), 'array', False)
        # Calling array(args, kwargs) (line 135)
        array_call_result_5382 = invoke(stypy.reporting.localization.Localization(__file__, 135, 36), array_5353, *[list_5354], **kwargs_5381)
        
        # Processing the call keyword arguments (line 135)
        kwargs_5383 = {}
        # Getting the type of 'scipy' (line 135)
        scipy_5350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 135)
        sparse_5351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), scipy_5350, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 135)
        csr_matrix_5352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), sparse_5351, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 135)
        csr_matrix_call_result_5384 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), csr_matrix_5352, *[array_call_result_5382], **kwargs_5383)
        
        # Assigning a type to the variable 'a' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'a', csr_matrix_call_result_5384)
        
        
        
        # Call to intp(...): (line 138)
        # Processing the call arguments (line 138)
        int_5387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 20), 'int')
        # Processing the call keyword arguments (line 138)
        kwargs_5388 = {}
        # Getting the type of 'np' (line 138)
        np_5385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'np', False)
        # Obtaining the member 'intp' of a type (line 138)
        intp_5386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), np_5385, 'intp')
        # Calling intp(args, kwargs) (line 138)
        intp_call_result_5389 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), intp_5386, *[int_5387], **kwargs_5388)
        
        # Obtaining the member 'itemsize' of a type (line 138)
        itemsize_5390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), intp_call_result_5389, 'itemsize')
        int_5391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 34), 'int')
        # Applying the binary operator '<' (line 138)
        result_lt_5392 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 12), '<', itemsize_5390, int_5391)
        
        # Testing the type of an if condition (line 138)
        if_condition_5393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 8), result_lt_5392)
        # Assigning a type to the variable 'if_condition_5393' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'if_condition_5393', if_condition_5393)
        # SSA begins for if statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_raises(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'OverflowError' (line 139)
        OverflowError_5395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 26), 'OverflowError', False)
        # Getting the type of 'mmwrite' (line 139)
        mmwrite_5396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 41), 'mmwrite', False)
        # Getting the type of 'self' (line 139)
        self_5397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'self', False)
        # Obtaining the member 'fn' of a type (line 139)
        fn_5398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 50), self_5397, 'fn')
        # Getting the type of 'a' (line 139)
        a_5399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 59), 'a', False)
        # Processing the call keyword arguments (line 139)
        kwargs_5400 = {}
        # Getting the type of 'assert_raises' (line 139)
        assert_raises_5394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 139)
        assert_raises_call_result_5401 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), assert_raises_5394, *[OverflowError_5395, mmwrite_5396, fn_5398, a_5399], **kwargs_5400)
        
        # SSA branch for the else part of an if statement (line 138)
        module_type_store.open_ssa_branch('else')
        
        # Call to check_exact(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'a' (line 141)
        a_5404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 29), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 141)
        tuple_5405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 141)
        # Adding element type (line 141)
        int_5406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 33), tuple_5405, int_5406)
        # Adding element type (line 141)
        int_5407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 33), tuple_5405, int_5407)
        # Adding element type (line 141)
        int_5408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 33), tuple_5405, int_5408)
        # Adding element type (line 141)
        str_5409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 42), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 33), tuple_5405, str_5409)
        # Adding element type (line 141)
        str_5410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 56), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 33), tuple_5405, str_5410)
        # Adding element type (line 141)
        str_5411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 67), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 33), tuple_5405, str_5411)
        
        # Processing the call keyword arguments (line 141)
        kwargs_5412 = {}
        # Getting the type of 'self' (line 141)
        self_5402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 141)
        check_exact_5403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), self_5402, 'check_exact')
        # Calling check_exact(args, kwargs) (line 141)
        check_exact_call_result_5413 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), check_exact_5403, *[a_5404, tuple_5405], **kwargs_5412)
        
        # SSA join for if statement (line 138)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_64bit_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_64bit_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_5414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5414)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_64bit_integer'
        return stypy_return_type_5414


    @norecursion
    def test_simple_upper_triangle_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_upper_triangle_integer'
        module_type_store = module_type_store.open_function_context('test_simple_upper_triangle_integer', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_upper_triangle_integer')
        TestMMIOSparseCSR.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_upper_triangle_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_upper_triangle_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_upper_triangle_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_upper_triangle_integer(...)' code ##################

        
        # Call to check_exact(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to csr_matrix(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_5420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_5421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        int_5422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 50), list_5421, int_5422)
        # Adding element type (line 144)
        int_5423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 50), list_5421, int_5423)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 49), list_5420, list_5421)
        # Adding element type (line 144)
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_5424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        int_5425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 58), list_5424, int_5425)
        # Adding element type (line 144)
        int_5426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 58), list_5424, int_5426)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 49), list_5420, list_5424)
        
        # Processing the call keyword arguments (line 144)
        kwargs_5427 = {}
        # Getting the type of 'scipy' (line 144)
        scipy_5417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 144)
        sparse_5418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 25), scipy_5417, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 144)
        csr_matrix_5419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 25), sparse_5418, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 144)
        csr_matrix_call_result_5428 = invoke(stypy.reporting.localization.Localization(__file__, 144, 25), csr_matrix_5419, *[list_5420], **kwargs_5427)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 145)
        tuple_5429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 145)
        # Adding element type (line 145)
        int_5430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 26), tuple_5429, int_5430)
        # Adding element type (line 145)
        int_5431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 26), tuple_5429, int_5431)
        # Adding element type (line 145)
        int_5432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 26), tuple_5429, int_5432)
        # Adding element type (line 145)
        str_5433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 35), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 26), tuple_5429, str_5433)
        # Adding element type (line 145)
        str_5434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 49), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 26), tuple_5429, str_5434)
        # Adding element type (line 145)
        str_5435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 60), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 26), tuple_5429, str_5435)
        
        # Processing the call keyword arguments (line 144)
        kwargs_5436 = {}
        # Getting the type of 'self' (line 144)
        self_5415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 144)
        check_exact_5416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_5415, 'check_exact')
        # Calling check_exact(args, kwargs) (line 144)
        check_exact_call_result_5437 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), check_exact_5416, *[csr_matrix_call_result_5428, tuple_5429], **kwargs_5436)
        
        
        # ################# End of 'test_simple_upper_triangle_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_upper_triangle_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_5438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5438)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_upper_triangle_integer'
        return stypy_return_type_5438


    @norecursion
    def test_simple_lower_triangle_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_lower_triangle_integer'
        module_type_store = module_type_store.open_function_context('test_simple_lower_triangle_integer', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_lower_triangle_integer')
        TestMMIOSparseCSR.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_lower_triangle_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_lower_triangle_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_lower_triangle_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_lower_triangle_integer(...)' code ##################

        
        # Call to check_exact(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Call to csr_matrix(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Obtaining an instance of the builtin type 'list' (line 148)
        list_5444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 148)
        # Adding element type (line 148)
        
        # Obtaining an instance of the builtin type 'list' (line 148)
        list_5445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 148)
        # Adding element type (line 148)
        int_5446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 50), list_5445, int_5446)
        # Adding element type (line 148)
        int_5447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 50), list_5445, int_5447)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 49), list_5444, list_5445)
        # Adding element type (line 148)
        
        # Obtaining an instance of the builtin type 'list' (line 148)
        list_5448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 148)
        # Adding element type (line 148)
        int_5449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 58), list_5448, int_5449)
        # Adding element type (line 148)
        int_5450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 58), list_5448, int_5450)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 49), list_5444, list_5448)
        
        # Processing the call keyword arguments (line 148)
        kwargs_5451 = {}
        # Getting the type of 'scipy' (line 148)
        scipy_5441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 148)
        sparse_5442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 25), scipy_5441, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 148)
        csr_matrix_5443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 25), sparse_5442, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 148)
        csr_matrix_call_result_5452 = invoke(stypy.reporting.localization.Localization(__file__, 148, 25), csr_matrix_5443, *[list_5444], **kwargs_5451)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 149)
        tuple_5453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 149)
        # Adding element type (line 149)
        int_5454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 26), tuple_5453, int_5454)
        # Adding element type (line 149)
        int_5455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 26), tuple_5453, int_5455)
        # Adding element type (line 149)
        int_5456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 26), tuple_5453, int_5456)
        # Adding element type (line 149)
        str_5457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 35), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 26), tuple_5453, str_5457)
        # Adding element type (line 149)
        str_5458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 49), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 26), tuple_5453, str_5458)
        # Adding element type (line 149)
        str_5459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 60), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 26), tuple_5453, str_5459)
        
        # Processing the call keyword arguments (line 148)
        kwargs_5460 = {}
        # Getting the type of 'self' (line 148)
        self_5439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 148)
        check_exact_5440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_5439, 'check_exact')
        # Calling check_exact(args, kwargs) (line 148)
        check_exact_call_result_5461 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), check_exact_5440, *[csr_matrix_call_result_5452, tuple_5453], **kwargs_5460)
        
        
        # ################# End of 'test_simple_lower_triangle_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_lower_triangle_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_5462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5462)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_lower_triangle_integer'
        return stypy_return_type_5462


    @norecursion
    def test_simple_rectangular_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_rectangular_integer'
        module_type_store = module_type_store.open_function_context('test_simple_rectangular_integer', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_rectangular_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_rectangular_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_rectangular_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_rectangular_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_rectangular_integer')
        TestMMIOSparseCSR.test_simple_rectangular_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_rectangular_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_rectangular_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_rectangular_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_rectangular_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_rectangular_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_rectangular_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_rectangular_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_rectangular_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_rectangular_integer(...)' code ##################

        
        # Call to check_exact(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Call to csr_matrix(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_5468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_5469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        int_5470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 50), list_5469, int_5470)
        # Adding element type (line 152)
        int_5471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 50), list_5469, int_5471)
        # Adding element type (line 152)
        int_5472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 50), list_5469, int_5472)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 49), list_5468, list_5469)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_5473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        int_5474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 61), list_5473, int_5474)
        # Adding element type (line 152)
        int_5475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 61), list_5473, int_5475)
        # Adding element type (line 152)
        int_5476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 61), list_5473, int_5476)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 49), list_5468, list_5473)
        
        # Processing the call keyword arguments (line 152)
        kwargs_5477 = {}
        # Getting the type of 'scipy' (line 152)
        scipy_5465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 152)
        sparse_5466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 25), scipy_5465, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 152)
        csr_matrix_5467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 25), sparse_5466, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 152)
        csr_matrix_call_result_5478 = invoke(stypy.reporting.localization.Localization(__file__, 152, 25), csr_matrix_5467, *[list_5468], **kwargs_5477)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 153)
        tuple_5479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 153)
        # Adding element type (line 153)
        int_5480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 26), tuple_5479, int_5480)
        # Adding element type (line 153)
        int_5481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 26), tuple_5479, int_5481)
        # Adding element type (line 153)
        int_5482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 26), tuple_5479, int_5482)
        # Adding element type (line 153)
        str_5483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 35), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 26), tuple_5479, str_5483)
        # Adding element type (line 153)
        str_5484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 49), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 26), tuple_5479, str_5484)
        # Adding element type (line 153)
        str_5485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 60), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 26), tuple_5479, str_5485)
        
        # Processing the call keyword arguments (line 152)
        kwargs_5486 = {}
        # Getting the type of 'self' (line 152)
        self_5463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 152)
        check_exact_5464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), self_5463, 'check_exact')
        # Calling check_exact(args, kwargs) (line 152)
        check_exact_call_result_5487 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), check_exact_5464, *[csr_matrix_call_result_5478, tuple_5479], **kwargs_5486)
        
        
        # ################# End of 'test_simple_rectangular_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_rectangular_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_5488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5488)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_rectangular_integer'
        return stypy_return_type_5488


    @norecursion
    def test_simple_rectangular_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_rectangular_float'
        module_type_store = module_type_store.open_function_context('test_simple_rectangular_float', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_rectangular_float.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_rectangular_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_rectangular_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_rectangular_float.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_rectangular_float')
        TestMMIOSparseCSR.test_simple_rectangular_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_rectangular_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_rectangular_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_rectangular_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_rectangular_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_rectangular_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_rectangular_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_rectangular_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_rectangular_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_rectangular_float(...)' code ##################

        
        # Call to check(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Call to csr_matrix(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_5494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_5495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        int_5496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 44), list_5495, int_5496)
        # Adding element type (line 156)
        int_5497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 44), list_5495, int_5497)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 43), list_5494, list_5495)
        # Adding element type (line 156)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_5498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        float_5499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 52), list_5498, float_5499)
        # Adding element type (line 156)
        int_5500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 52), list_5498, int_5500)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 43), list_5494, list_5498)
        # Adding element type (line 156)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_5501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        int_5502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 62), list_5501, int_5502)
        # Adding element type (line 156)
        int_5503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 62), list_5501, int_5503)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 43), list_5494, list_5501)
        
        # Processing the call keyword arguments (line 156)
        kwargs_5504 = {}
        # Getting the type of 'scipy' (line 156)
        scipy_5491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 156)
        sparse_5492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 19), scipy_5491, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 156)
        csr_matrix_5493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 19), sparse_5492, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 156)
        csr_matrix_call_result_5505 = invoke(stypy.reporting.localization.Localization(__file__, 156, 19), csr_matrix_5493, *[list_5494], **kwargs_5504)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 157)
        tuple_5506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 157)
        # Adding element type (line 157)
        int_5507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), tuple_5506, int_5507)
        # Adding element type (line 157)
        int_5508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), tuple_5506, int_5508)
        # Adding element type (line 157)
        int_5509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), tuple_5506, int_5509)
        # Adding element type (line 157)
        str_5510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 29), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), tuple_5506, str_5510)
        # Adding element type (line 157)
        str_5511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 43), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), tuple_5506, str_5511)
        # Adding element type (line 157)
        str_5512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 51), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 20), tuple_5506, str_5512)
        
        # Processing the call keyword arguments (line 156)
        kwargs_5513 = {}
        # Getting the type of 'self' (line 156)
        self_5489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 156)
        check_5490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), self_5489, 'check')
        # Calling check(args, kwargs) (line 156)
        check_call_result_5514 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), check_5490, *[csr_matrix_call_result_5505, tuple_5506], **kwargs_5513)
        
        
        # ################# End of 'test_simple_rectangular_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_rectangular_float' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_5515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5515)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_rectangular_float'
        return stypy_return_type_5515


    @norecursion
    def test_simple_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_float'
        module_type_store = module_type_store.open_function_context('test_simple_float', 159, 4, False)
        # Assigning a type to the variable 'self' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_float.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_float.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_float')
        TestMMIOSparseCSR.test_simple_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_float(...)' code ##################

        
        # Call to check(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Call to csr_matrix(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_5521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        # Adding element type (line 160)
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_5522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        # Adding element type (line 160)
        int_5523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 44), list_5522, int_5523)
        # Adding element type (line 160)
        int_5524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 44), list_5522, int_5524)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 43), list_5521, list_5522)
        # Adding element type (line 160)
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_5525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        # Adding element type (line 160)
        int_5526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 52), list_5525, int_5526)
        # Adding element type (line 160)
        float_5527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 52), list_5525, float_5527)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 43), list_5521, list_5525)
        
        # Processing the call keyword arguments (line 160)
        kwargs_5528 = {}
        # Getting the type of 'scipy' (line 160)
        scipy_5518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 19), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 160)
        sparse_5519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), scipy_5518, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 160)
        csr_matrix_5520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), sparse_5519, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 160)
        csr_matrix_call_result_5529 = invoke(stypy.reporting.localization.Localization(__file__, 160, 19), csr_matrix_5520, *[list_5521], **kwargs_5528)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 161)
        tuple_5530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 161)
        # Adding element type (line 161)
        int_5531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 20), tuple_5530, int_5531)
        # Adding element type (line 161)
        int_5532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 20), tuple_5530, int_5532)
        # Adding element type (line 161)
        int_5533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 20), tuple_5530, int_5533)
        # Adding element type (line 161)
        str_5534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 29), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 20), tuple_5530, str_5534)
        # Adding element type (line 161)
        str_5535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 43), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 20), tuple_5530, str_5535)
        # Adding element type (line 161)
        str_5536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 51), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 20), tuple_5530, str_5536)
        
        # Processing the call keyword arguments (line 160)
        kwargs_5537 = {}
        # Getting the type of 'self' (line 160)
        self_5516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 160)
        check_5517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_5516, 'check')
        # Calling check(args, kwargs) (line 160)
        check_call_result_5538 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), check_5517, *[csr_matrix_call_result_5529, tuple_5530], **kwargs_5537)
        
        
        # ################# End of 'test_simple_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_float' in the type store
        # Getting the type of 'stypy_return_type' (line 159)
        stypy_return_type_5539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5539)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_float'
        return stypy_return_type_5539


    @norecursion
    def test_simple_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_complex'
        module_type_store = module_type_store.open_function_context('test_simple_complex', 163, 4, False)
        # Assigning a type to the variable 'self' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_complex.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_complex.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_complex')
        TestMMIOSparseCSR.test_simple_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_complex(...)' code ##################

        
        # Call to check(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Call to csr_matrix(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_5545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        # Adding element type (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_5546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        # Adding element type (line 164)
        int_5547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 44), list_5546, int_5547)
        # Adding element type (line 164)
        int_5548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 44), list_5546, int_5548)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 43), list_5545, list_5546)
        # Adding element type (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_5549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        # Adding element type (line 164)
        int_5550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 52), list_5549, int_5550)
        # Adding element type (line 164)
        complex_5551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 56), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 52), list_5549, complex_5551)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 43), list_5545, list_5549)
        
        # Processing the call keyword arguments (line 164)
        kwargs_5552 = {}
        # Getting the type of 'scipy' (line 164)
        scipy_5542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 19), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 164)
        sparse_5543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 19), scipy_5542, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 164)
        csr_matrix_5544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 19), sparse_5543, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 164)
        csr_matrix_call_result_5553 = invoke(stypy.reporting.localization.Localization(__file__, 164, 19), csr_matrix_5544, *[list_5545], **kwargs_5552)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 165)
        tuple_5554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 165)
        # Adding element type (line 165)
        int_5555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), tuple_5554, int_5555)
        # Adding element type (line 165)
        int_5556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), tuple_5554, int_5556)
        # Adding element type (line 165)
        int_5557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), tuple_5554, int_5557)
        # Adding element type (line 165)
        str_5558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 29), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), tuple_5554, str_5558)
        # Adding element type (line 165)
        str_5559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 43), 'str', 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), tuple_5554, str_5559)
        # Adding element type (line 165)
        str_5560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 54), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), tuple_5554, str_5560)
        
        # Processing the call keyword arguments (line 164)
        kwargs_5561 = {}
        # Getting the type of 'self' (line 164)
        self_5540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 164)
        check_5541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_5540, 'check')
        # Calling check(args, kwargs) (line 164)
        check_call_result_5562 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), check_5541, *[csr_matrix_call_result_5553, tuple_5554], **kwargs_5561)
        
        
        # ################# End of 'test_simple_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 163)
        stypy_return_type_5563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5563)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_complex'
        return stypy_return_type_5563


    @norecursion
    def test_simple_symmetric_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_symmetric_integer'
        module_type_store = module_type_store.open_function_context('test_simple_symmetric_integer', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_symmetric_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_symmetric_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_symmetric_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_symmetric_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_symmetric_integer')
        TestMMIOSparseCSR.test_simple_symmetric_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_symmetric_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_symmetric_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_symmetric_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_symmetric_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_symmetric_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_symmetric_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_symmetric_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_symmetric_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_symmetric_integer(...)' code ##################

        
        # Call to check_exact(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Call to csr_matrix(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Obtaining an instance of the builtin type 'list' (line 168)
        list_5569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 168)
        # Adding element type (line 168)
        
        # Obtaining an instance of the builtin type 'list' (line 168)
        list_5570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 168)
        # Adding element type (line 168)
        int_5571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 50), list_5570, int_5571)
        # Adding element type (line 168)
        int_5572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 50), list_5570, int_5572)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 49), list_5569, list_5570)
        # Adding element type (line 168)
        
        # Obtaining an instance of the builtin type 'list' (line 168)
        list_5573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 168)
        # Adding element type (line 168)
        int_5574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 58), list_5573, int_5574)
        # Adding element type (line 168)
        int_5575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 58), list_5573, int_5575)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 49), list_5569, list_5573)
        
        # Processing the call keyword arguments (line 168)
        kwargs_5576 = {}
        # Getting the type of 'scipy' (line 168)
        scipy_5566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 168)
        sparse_5567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 25), scipy_5566, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 168)
        csr_matrix_5568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 25), sparse_5567, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 168)
        csr_matrix_call_result_5577 = invoke(stypy.reporting.localization.Localization(__file__, 168, 25), csr_matrix_5568, *[list_5569], **kwargs_5576)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_5578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        int_5579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 26), tuple_5578, int_5579)
        # Adding element type (line 169)
        int_5580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 26), tuple_5578, int_5580)
        # Adding element type (line 169)
        int_5581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 26), tuple_5578, int_5581)
        # Adding element type (line 169)
        str_5582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 35), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 26), tuple_5578, str_5582)
        # Adding element type (line 169)
        str_5583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 49), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 26), tuple_5578, str_5583)
        # Adding element type (line 169)
        str_5584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 60), 'str', 'symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 26), tuple_5578, str_5584)
        
        # Processing the call keyword arguments (line 168)
        kwargs_5585 = {}
        # Getting the type of 'self' (line 168)
        self_5564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 168)
        check_exact_5565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_5564, 'check_exact')
        # Calling check_exact(args, kwargs) (line 168)
        check_exact_call_result_5586 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), check_exact_5565, *[csr_matrix_call_result_5577, tuple_5578], **kwargs_5585)
        
        
        # ################# End of 'test_simple_symmetric_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_symmetric_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_5587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5587)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_symmetric_integer'
        return stypy_return_type_5587


    @norecursion
    def test_simple_skew_symmetric_integer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_skew_symmetric_integer'
        module_type_store = module_type_store.open_function_context('test_simple_skew_symmetric_integer', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_skew_symmetric_integer')
        TestMMIOSparseCSR.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_skew_symmetric_integer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_skew_symmetric_integer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_skew_symmetric_integer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_skew_symmetric_integer(...)' code ##################

        
        # Call to check_exact(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Call to csr_matrix(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Obtaining an instance of the builtin type 'list' (line 172)
        list_5593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 172)
        # Adding element type (line 172)
        
        # Obtaining an instance of the builtin type 'list' (line 172)
        list_5594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 172)
        # Adding element type (line 172)
        int_5595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 50), list_5594, int_5595)
        # Adding element type (line 172)
        int_5596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 50), list_5594, int_5596)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 49), list_5593, list_5594)
        # Adding element type (line 172)
        
        # Obtaining an instance of the builtin type 'list' (line 172)
        list_5597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 172)
        # Adding element type (line 172)
        int_5598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 58), list_5597, int_5598)
        # Adding element type (line 172)
        int_5599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 58), list_5597, int_5599)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 49), list_5593, list_5597)
        
        # Processing the call keyword arguments (line 172)
        kwargs_5600 = {}
        # Getting the type of 'scipy' (line 172)
        scipy_5590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 172)
        sparse_5591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), scipy_5590, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 172)
        csr_matrix_5592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), sparse_5591, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 172)
        csr_matrix_call_result_5601 = invoke(stypy.reporting.localization.Localization(__file__, 172, 25), csr_matrix_5592, *[list_5593], **kwargs_5600)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 173)
        tuple_5602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 173)
        # Adding element type (line 173)
        int_5603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 26), tuple_5602, int_5603)
        # Adding element type (line 173)
        int_5604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 26), tuple_5602, int_5604)
        # Adding element type (line 173)
        int_5605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 26), tuple_5602, int_5605)
        # Adding element type (line 173)
        str_5606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 35), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 26), tuple_5602, str_5606)
        # Adding element type (line 173)
        str_5607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 49), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 26), tuple_5602, str_5607)
        # Adding element type (line 173)
        str_5608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 60), 'str', 'skew-symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 26), tuple_5602, str_5608)
        
        # Processing the call keyword arguments (line 172)
        kwargs_5609 = {}
        # Getting the type of 'self' (line 172)
        self_5588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self', False)
        # Obtaining the member 'check_exact' of a type (line 172)
        check_exact_5589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_5588, 'check_exact')
        # Calling check_exact(args, kwargs) (line 172)
        check_exact_call_result_5610 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), check_exact_5589, *[csr_matrix_call_result_5601, tuple_5602], **kwargs_5609)
        
        
        # ################# End of 'test_simple_skew_symmetric_integer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_skew_symmetric_integer' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_5611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5611)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_skew_symmetric_integer'
        return stypy_return_type_5611


    @norecursion
    def test_simple_skew_symmetric_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_skew_symmetric_float'
        module_type_store = module_type_store.open_function_context('test_simple_skew_symmetric_float', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_skew_symmetric_float')
        TestMMIOSparseCSR.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_skew_symmetric_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_skew_symmetric_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_skew_symmetric_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_skew_symmetric_float(...)' code ##################

        
        # Call to check(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Call to csr_matrix(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Call to array(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_5618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_5619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        int_5620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 50), list_5619, int_5620)
        # Adding element type (line 176)
        int_5621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 50), list_5619, int_5621)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 49), list_5618, list_5619)
        # Adding element type (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_5622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        float_5623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 58), list_5622, float_5623)
        # Adding element type (line 176)
        int_5624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 58), list_5622, int_5624)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 49), list_5618, list_5622)
        
        str_5625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 70), 'str', 'f')
        # Processing the call keyword arguments (line 176)
        kwargs_5626 = {}
        # Getting the type of 'array' (line 176)
        array_5617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'array', False)
        # Calling array(args, kwargs) (line 176)
        array_call_result_5627 = invoke(stypy.reporting.localization.Localization(__file__, 176, 43), array_5617, *[list_5618, str_5625], **kwargs_5626)
        
        # Processing the call keyword arguments (line 176)
        kwargs_5628 = {}
        # Getting the type of 'scipy' (line 176)
        scipy_5614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 176)
        sparse_5615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 19), scipy_5614, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 176)
        csr_matrix_5616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 19), sparse_5615, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 176)
        csr_matrix_call_result_5629 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), csr_matrix_5616, *[array_call_result_5627], **kwargs_5628)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 177)
        tuple_5630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 177)
        # Adding element type (line 177)
        int_5631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 20), tuple_5630, int_5631)
        # Adding element type (line 177)
        int_5632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 20), tuple_5630, int_5632)
        # Adding element type (line 177)
        int_5633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 20), tuple_5630, int_5633)
        # Adding element type (line 177)
        str_5634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 29), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 20), tuple_5630, str_5634)
        # Adding element type (line 177)
        str_5635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 43), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 20), tuple_5630, str_5635)
        # Adding element type (line 177)
        str_5636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 51), 'str', 'skew-symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 20), tuple_5630, str_5636)
        
        # Processing the call keyword arguments (line 176)
        kwargs_5637 = {}
        # Getting the type of 'self' (line 176)
        self_5612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 176)
        check_5613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_5612, 'check')
        # Calling check(args, kwargs) (line 176)
        check_call_result_5638 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), check_5613, *[csr_matrix_call_result_5629, tuple_5630], **kwargs_5637)
        
        
        # ################# End of 'test_simple_skew_symmetric_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_skew_symmetric_float' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_5639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5639)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_skew_symmetric_float'
        return stypy_return_type_5639


    @norecursion
    def test_simple_hermitian_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_hermitian_complex'
        module_type_store = module_type_store.open_function_context('test_simple_hermitian_complex', 179, 4, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_hermitian_complex.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_hermitian_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_hermitian_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_hermitian_complex.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_hermitian_complex')
        TestMMIOSparseCSR.test_simple_hermitian_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_hermitian_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_hermitian_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_hermitian_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_hermitian_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_hermitian_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_hermitian_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_hermitian_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_hermitian_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_hermitian_complex(...)' code ##################

        
        # Call to check(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Call to csr_matrix(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Obtaining an instance of the builtin type 'list' (line 180)
        list_5645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 180)
        # Adding element type (line 180)
        
        # Obtaining an instance of the builtin type 'list' (line 180)
        list_5646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 180)
        # Adding element type (line 180)
        int_5647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 44), list_5646, int_5647)
        # Adding element type (line 180)
        int_5648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 48), 'int')
        complex_5649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 50), 'complex')
        # Applying the binary operator '+' (line 180)
        result_add_5650 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 48), '+', int_5648, complex_5649)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 44), list_5646, result_add_5650)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 43), list_5645, list_5646)
        # Adding element type (line 180)
        
        # Obtaining an instance of the builtin type 'list' (line 180)
        list_5651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 180)
        # Adding element type (line 180)
        int_5652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 56), 'int')
        complex_5653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 58), 'complex')
        # Applying the binary operator '-' (line 180)
        result_sub_5654 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 56), '-', int_5652, complex_5653)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 55), list_5651, result_sub_5654)
        # Adding element type (line 180)
        int_5655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 55), list_5651, int_5655)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 43), list_5645, list_5651)
        
        # Processing the call keyword arguments (line 180)
        kwargs_5656 = {}
        # Getting the type of 'scipy' (line 180)
        scipy_5642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 180)
        sparse_5643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 19), scipy_5642, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 180)
        csr_matrix_5644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 19), sparse_5643, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 180)
        csr_matrix_call_result_5657 = invoke(stypy.reporting.localization.Localization(__file__, 180, 19), csr_matrix_5644, *[list_5645], **kwargs_5656)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 181)
        tuple_5658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 181)
        # Adding element type (line 181)
        int_5659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 20), tuple_5658, int_5659)
        # Adding element type (line 181)
        int_5660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 20), tuple_5658, int_5660)
        # Adding element type (line 181)
        int_5661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 20), tuple_5658, int_5661)
        # Adding element type (line 181)
        str_5662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 29), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 20), tuple_5658, str_5662)
        # Adding element type (line 181)
        str_5663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 43), 'str', 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 20), tuple_5658, str_5663)
        # Adding element type (line 181)
        str_5664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 54), 'str', 'hermitian')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 20), tuple_5658, str_5664)
        
        # Processing the call keyword arguments (line 180)
        kwargs_5665 = {}
        # Getting the type of 'self' (line 180)
        self_5640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 180)
        check_5641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_5640, 'check')
        # Calling check(args, kwargs) (line 180)
        check_call_result_5666 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), check_5641, *[csr_matrix_call_result_5657, tuple_5658], **kwargs_5665)
        
        
        # ################# End of 'test_simple_hermitian_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_hermitian_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 179)
        stypy_return_type_5667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5667)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_hermitian_complex'
        return stypy_return_type_5667


    @norecursion
    def test_random_symmetric_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_symmetric_float'
        module_type_store = module_type_store.open_function_context('test_random_symmetric_float', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_random_symmetric_float.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_random_symmetric_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_random_symmetric_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_random_symmetric_float.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_random_symmetric_float')
        TestMMIOSparseCSR.test_random_symmetric_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_random_symmetric_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_random_symmetric_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_random_symmetric_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_random_symmetric_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_random_symmetric_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_random_symmetric_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_random_symmetric_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_symmetric_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_symmetric_float(...)' code ##################

        
        # Assigning a Tuple to a Name (line 184):
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_5668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        int_5669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 14), tuple_5668, int_5669)
        # Adding element type (line 184)
        int_5670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 14), tuple_5668, int_5670)
        
        # Assigning a type to the variable 'sz' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'sz', tuple_5668)
        
        # Assigning a Call to a Name (line 185):
        
        # Call to random(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'sz' (line 185)
        sz_5674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 29), 'sz', False)
        # Processing the call keyword arguments (line 185)
        kwargs_5675 = {}
        # Getting the type of 'np' (line 185)
        np_5671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 185)
        random_5672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 12), np_5671, 'random')
        # Obtaining the member 'random' of a type (line 185)
        random_5673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 12), random_5672, 'random')
        # Calling random(args, kwargs) (line 185)
        random_call_result_5676 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), random_5673, *[sz_5674], **kwargs_5675)
        
        # Assigning a type to the variable 'a' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'a', random_call_result_5676)
        
        # Assigning a BinOp to a Name (line 186):
        # Getting the type of 'a' (line 186)
        a_5677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'a')
        
        # Call to transpose(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'a' (line 186)
        a_5679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 26), 'a', False)
        # Processing the call keyword arguments (line 186)
        kwargs_5680 = {}
        # Getting the type of 'transpose' (line 186)
        transpose_5678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'transpose', False)
        # Calling transpose(args, kwargs) (line 186)
        transpose_call_result_5681 = invoke(stypy.reporting.localization.Localization(__file__, 186, 16), transpose_5678, *[a_5679], **kwargs_5680)
        
        # Applying the binary operator '+' (line 186)
        result_add_5682 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 12), '+', a_5677, transpose_call_result_5681)
        
        # Assigning a type to the variable 'a' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'a', result_add_5682)
        
        # Assigning a Call to a Name (line 187):
        
        # Call to csr_matrix(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'a' (line 187)
        a_5686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 36), 'a', False)
        # Processing the call keyword arguments (line 187)
        kwargs_5687 = {}
        # Getting the type of 'scipy' (line 187)
        scipy_5683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 187)
        sparse_5684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 12), scipy_5683, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 187)
        csr_matrix_5685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 12), sparse_5684, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 187)
        csr_matrix_call_result_5688 = invoke(stypy.reporting.localization.Localization(__file__, 187, 12), csr_matrix_5685, *[a_5686], **kwargs_5687)
        
        # Assigning a type to the variable 'a' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'a', csr_matrix_call_result_5688)
        
        # Call to check(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'a' (line 188)
        a_5691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 188)
        tuple_5692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 188)
        # Adding element type (line 188)
        int_5693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 23), tuple_5692, int_5693)
        # Adding element type (line 188)
        int_5694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 23), tuple_5692, int_5694)
        # Adding element type (line 188)
        int_5695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 23), tuple_5692, int_5695)
        # Adding element type (line 188)
        str_5696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 36), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 23), tuple_5692, str_5696)
        # Adding element type (line 188)
        str_5697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 50), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 23), tuple_5692, str_5697)
        # Adding element type (line 188)
        str_5698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 58), 'str', 'symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 23), tuple_5692, str_5698)
        
        # Processing the call keyword arguments (line 188)
        kwargs_5699 = {}
        # Getting the type of 'self' (line 188)
        self_5689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 188)
        check_5690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_5689, 'check')
        # Calling check(args, kwargs) (line 188)
        check_call_result_5700 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), check_5690, *[a_5691, tuple_5692], **kwargs_5699)
        
        
        # ################# End of 'test_random_symmetric_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_symmetric_float' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_5701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5701)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_symmetric_float'
        return stypy_return_type_5701


    @norecursion
    def test_random_rectangular_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_rectangular_float'
        module_type_store = module_type_store.open_function_context('test_random_rectangular_float', 190, 4, False)
        # Assigning a type to the variable 'self' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_random_rectangular_float.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_random_rectangular_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_random_rectangular_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_random_rectangular_float.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_random_rectangular_float')
        TestMMIOSparseCSR.test_random_rectangular_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_random_rectangular_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_random_rectangular_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_random_rectangular_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_random_rectangular_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_random_rectangular_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_random_rectangular_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_random_rectangular_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_rectangular_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_rectangular_float(...)' code ##################

        
        # Assigning a Tuple to a Name (line 191):
        
        # Obtaining an instance of the builtin type 'tuple' (line 191)
        tuple_5702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 191)
        # Adding element type (line 191)
        int_5703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 14), tuple_5702, int_5703)
        # Adding element type (line 191)
        int_5704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 14), tuple_5702, int_5704)
        
        # Assigning a type to the variable 'sz' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'sz', tuple_5702)
        
        # Assigning a Call to a Name (line 192):
        
        # Call to random(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'sz' (line 192)
        sz_5708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 29), 'sz', False)
        # Processing the call keyword arguments (line 192)
        kwargs_5709 = {}
        # Getting the type of 'np' (line 192)
        np_5705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 192)
        random_5706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), np_5705, 'random')
        # Obtaining the member 'random' of a type (line 192)
        random_5707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), random_5706, 'random')
        # Calling random(args, kwargs) (line 192)
        random_call_result_5710 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), random_5707, *[sz_5708], **kwargs_5709)
        
        # Assigning a type to the variable 'a' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'a', random_call_result_5710)
        
        # Assigning a Call to a Name (line 193):
        
        # Call to csr_matrix(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'a' (line 193)
        a_5714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 36), 'a', False)
        # Processing the call keyword arguments (line 193)
        kwargs_5715 = {}
        # Getting the type of 'scipy' (line 193)
        scipy_5711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 193)
        sparse_5712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), scipy_5711, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 193)
        csr_matrix_5713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), sparse_5712, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 193)
        csr_matrix_call_result_5716 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), csr_matrix_5713, *[a_5714], **kwargs_5715)
        
        # Assigning a type to the variable 'a' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'a', csr_matrix_call_result_5716)
        
        # Call to check(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'a' (line 194)
        a_5719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 19), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 194)
        tuple_5720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 194)
        # Adding element type (line 194)
        int_5721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 23), tuple_5720, int_5721)
        # Adding element type (line 194)
        int_5722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 23), tuple_5720, int_5722)
        # Adding element type (line 194)
        int_5723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 23), tuple_5720, int_5723)
        # Adding element type (line 194)
        str_5724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 36), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 23), tuple_5720, str_5724)
        # Adding element type (line 194)
        str_5725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 50), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 23), tuple_5720, str_5725)
        # Adding element type (line 194)
        str_5726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 58), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 23), tuple_5720, str_5726)
        
        # Processing the call keyword arguments (line 194)
        kwargs_5727 = {}
        # Getting the type of 'self' (line 194)
        self_5717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'self', False)
        # Obtaining the member 'check' of a type (line 194)
        check_5718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), self_5717, 'check')
        # Calling check(args, kwargs) (line 194)
        check_call_result_5728 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), check_5718, *[a_5719, tuple_5720], **kwargs_5727)
        
        
        # ################# End of 'test_random_rectangular_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_rectangular_float' in the type store
        # Getting the type of 'stypy_return_type' (line 190)
        stypy_return_type_5729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5729)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_rectangular_float'
        return stypy_return_type_5729


    @norecursion
    def test_simple_pattern(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_pattern'
        module_type_store = module_type_store.open_function_context('test_simple_pattern', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOSparseCSR.test_simple_pattern.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOSparseCSR.test_simple_pattern.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOSparseCSR.test_simple_pattern.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOSparseCSR.test_simple_pattern.__dict__.__setitem__('stypy_function_name', 'TestMMIOSparseCSR.test_simple_pattern')
        TestMMIOSparseCSR.test_simple_pattern.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOSparseCSR.test_simple_pattern.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOSparseCSR.test_simple_pattern.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOSparseCSR.test_simple_pattern.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOSparseCSR.test_simple_pattern.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOSparseCSR.test_simple_pattern.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOSparseCSR.test_simple_pattern.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.test_simple_pattern', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_pattern', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_pattern(...)' code ##################

        
        # Assigning a Call to a Name (line 197):
        
        # Call to csr_matrix(...): (line 197)
        # Processing the call arguments (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_5733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_5734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        int_5735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 37), list_5734, int_5735)
        # Adding element type (line 197)
        float_5736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 37), list_5734, float_5736)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 36), list_5733, list_5734)
        # Adding element type (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_5737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        float_5738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 47), list_5737, float_5738)
        # Adding element type (line 197)
        float_5739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 47), list_5737, float_5739)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 36), list_5733, list_5737)
        
        # Processing the call keyword arguments (line 197)
        kwargs_5740 = {}
        # Getting the type of 'scipy' (line 197)
        scipy_5730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 197)
        sparse_5731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), scipy_5730, 'sparse')
        # Obtaining the member 'csr_matrix' of a type (line 197)
        csr_matrix_5732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), sparse_5731, 'csr_matrix')
        # Calling csr_matrix(args, kwargs) (line 197)
        csr_matrix_call_result_5741 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), csr_matrix_5732, *[list_5733], **kwargs_5740)
        
        # Assigning a type to the variable 'a' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'a', csr_matrix_call_result_5741)
        
        # Assigning a Call to a Name (line 198):
        
        # Call to zeros_like(...): (line 198)
        # Processing the call arguments (line 198)
        
        # Call to todense(...): (line 198)
        # Processing the call keyword arguments (line 198)
        kwargs_5746 = {}
        # Getting the type of 'a' (line 198)
        a_5744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 26), 'a', False)
        # Obtaining the member 'todense' of a type (line 198)
        todense_5745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 26), a_5744, 'todense')
        # Calling todense(args, kwargs) (line 198)
        todense_call_result_5747 = invoke(stypy.reporting.localization.Localization(__file__, 198, 26), todense_5745, *[], **kwargs_5746)
        
        # Processing the call keyword arguments (line 198)
        kwargs_5748 = {}
        # Getting the type of 'np' (line 198)
        np_5742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'np', False)
        # Obtaining the member 'zeros_like' of a type (line 198)
        zeros_like_5743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), np_5742, 'zeros_like')
        # Calling zeros_like(args, kwargs) (line 198)
        zeros_like_call_result_5749 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), zeros_like_5743, *[todense_call_result_5747], **kwargs_5748)
        
        # Assigning a type to the variable 'p' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'p', zeros_like_call_result_5749)
        
        # Assigning a Num to a Subscript (line 199):
        int_5750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'int')
        # Getting the type of 'p' (line 199)
        p_5751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'p')
        
        
        # Call to todense(...): (line 199)
        # Processing the call keyword arguments (line 199)
        kwargs_5754 = {}
        # Getting the type of 'a' (line 199)
        a_5752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 10), 'a', False)
        # Obtaining the member 'todense' of a type (line 199)
        todense_5753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 10), a_5752, 'todense')
        # Calling todense(args, kwargs) (line 199)
        todense_call_result_5755 = invoke(stypy.reporting.localization.Localization(__file__, 199, 10), todense_5753, *[], **kwargs_5754)
        
        int_5756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 24), 'int')
        # Applying the binary operator '>' (line 199)
        result_gt_5757 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 10), '>', todense_call_result_5755, int_5756)
        
        # Storing an element on a container (line 199)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 8), p_5751, (result_gt_5757, int_5750))
        
        # Assigning a Tuple to a Name (line 200):
        
        # Obtaining an instance of the builtin type 'tuple' (line 200)
        tuple_5758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 200)
        # Adding element type (line 200)
        int_5759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 16), tuple_5758, int_5759)
        # Adding element type (line 200)
        int_5760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 16), tuple_5758, int_5760)
        # Adding element type (line 200)
        int_5761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 16), tuple_5758, int_5761)
        # Adding element type (line 200)
        str_5762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 25), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 16), tuple_5758, str_5762)
        # Adding element type (line 200)
        str_5763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 39), 'str', 'pattern')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 16), tuple_5758, str_5763)
        # Adding element type (line 200)
        str_5764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 50), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 16), tuple_5758, str_5764)
        
        # Assigning a type to the variable 'info' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'info', tuple_5758)
        
        # Call to mmwrite(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'self' (line 201)
        self_5766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'self', False)
        # Obtaining the member 'fn' of a type (line 201)
        fn_5767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), self_5766, 'fn')
        # Getting the type of 'a' (line 201)
        a_5768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 25), 'a', False)
        # Processing the call keyword arguments (line 201)
        str_5769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 34), 'str', 'pattern')
        keyword_5770 = str_5769
        kwargs_5771 = {'field': keyword_5770}
        # Getting the type of 'mmwrite' (line 201)
        mmwrite_5765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 201)
        mmwrite_call_result_5772 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), mmwrite_5765, *[fn_5767, a_5768], **kwargs_5771)
        
        
        # Call to assert_equal(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Call to mminfo(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'self' (line 202)
        self_5775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'self', False)
        # Obtaining the member 'fn' of a type (line 202)
        fn_5776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 28), self_5775, 'fn')
        # Processing the call keyword arguments (line 202)
        kwargs_5777 = {}
        # Getting the type of 'mminfo' (line 202)
        mminfo_5774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'mminfo', False)
        # Calling mminfo(args, kwargs) (line 202)
        mminfo_call_result_5778 = invoke(stypy.reporting.localization.Localization(__file__, 202, 21), mminfo_5774, *[fn_5776], **kwargs_5777)
        
        # Getting the type of 'info' (line 202)
        info_5779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 38), 'info', False)
        # Processing the call keyword arguments (line 202)
        kwargs_5780 = {}
        # Getting the type of 'assert_equal' (line 202)
        assert_equal_5773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 202)
        assert_equal_call_result_5781 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), assert_equal_5773, *[mminfo_call_result_5778, info_5779], **kwargs_5780)
        
        
        # Assigning a Call to a Name (line 203):
        
        # Call to mmread(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'self' (line 203)
        self_5783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 19), 'self', False)
        # Obtaining the member 'fn' of a type (line 203)
        fn_5784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 19), self_5783, 'fn')
        # Processing the call keyword arguments (line 203)
        kwargs_5785 = {}
        # Getting the type of 'mmread' (line 203)
        mmread_5782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'mmread', False)
        # Calling mmread(args, kwargs) (line 203)
        mmread_call_result_5786 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), mmread_5782, *[fn_5784], **kwargs_5785)
        
        # Assigning a type to the variable 'b' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'b', mmread_call_result_5786)
        
        # Call to assert_array_almost_equal(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'p' (line 204)
        p_5788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 34), 'p', False)
        
        # Call to todense(...): (line 204)
        # Processing the call keyword arguments (line 204)
        kwargs_5791 = {}
        # Getting the type of 'b' (line 204)
        b_5789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 37), 'b', False)
        # Obtaining the member 'todense' of a type (line 204)
        todense_5790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 37), b_5789, 'todense')
        # Calling todense(args, kwargs) (line 204)
        todense_call_result_5792 = invoke(stypy.reporting.localization.Localization(__file__, 204, 37), todense_5790, *[], **kwargs_5791)
        
        # Processing the call keyword arguments (line 204)
        kwargs_5793 = {}
        # Getting the type of 'assert_array_almost_equal' (line 204)
        assert_array_almost_equal_5787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 204)
        assert_array_almost_equal_call_result_5794 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), assert_array_almost_equal_5787, *[p_5788, todense_call_result_5792], **kwargs_5793)
        
        
        # ################# End of 'test_simple_pattern(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_pattern' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_5795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5795)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_pattern'
        return stypy_return_type_5795


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 104, 0, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOSparseCSR.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestMMIOSparseCSR' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'TestMMIOSparseCSR', TestMMIOSparseCSR)

# Assigning a Str to a Name (line 207):
str_5796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, (-1)), 'str', '%%MatrixMarket matrix array integer general\n2  2\n2147483647\n2147483646\n2147483647\n2147483646\n')
# Assigning a type to the variable '_32bit_integer_dense_example' (line 207)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), '_32bit_integer_dense_example', str_5796)

# Assigning a Str to a Name (line 216):
str_5797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, (-1)), 'str', '%%MatrixMarket matrix coordinate integer symmetric\n2  2  2\n1  1  2147483647\n2  2  2147483646\n')
# Assigning a type to the variable '_32bit_integer_sparse_example' (line 216)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 0), '_32bit_integer_sparse_example', str_5797)

# Assigning a Str to a Name (line 223):
str_5798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, (-1)), 'str', '%%MatrixMarket matrix array integer general\n2  2\n          2147483648\n-9223372036854775806\n         -2147483648\n 9223372036854775807\n')
# Assigning a type to the variable '_64bit_integer_dense_example' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), '_64bit_integer_dense_example', str_5798)

# Assigning a Str to a Name (line 232):
str_5799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, (-1)), 'str', '%%MatrixMarket matrix coordinate integer general\n2  2  3\n1  1           2147483648\n1  2  9223372036854775807\n2  2  9223372036854775807\n')
# Assigning a type to the variable '_64bit_integer_sparse_general_example' (line 232)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 0), '_64bit_integer_sparse_general_example', str_5799)

# Assigning a Str to a Name (line 240):
str_5800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, (-1)), 'str', '%%MatrixMarket matrix coordinate integer symmetric\n2  2  3\n1  1            2147483648\n1  2  -9223372036854775807\n2  2   9223372036854775807\n')
# Assigning a type to the variable '_64bit_integer_sparse_symmetric_example' (line 240)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), '_64bit_integer_sparse_symmetric_example', str_5800)

# Assigning a Str to a Name (line 248):
str_5801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, (-1)), 'str', '%%MatrixMarket matrix coordinate integer skew-symmetric\n2  2  3\n1  1            2147483648\n1  2  -9223372036854775807\n2  2   9223372036854775807\n')
# Assigning a type to the variable '_64bit_integer_sparse_skew_example' (line 248)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), '_64bit_integer_sparse_skew_example', str_5801)

# Assigning a Str to a Name (line 256):
str_5802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, (-1)), 'str', '%%MatrixMarket matrix array integer general\n2  2\n         2147483648\n9223372036854775807\n         2147483648\n9223372036854775808\n')
# Assigning a type to the variable '_over64bit_integer_dense_example' (line 256)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 0), '_over64bit_integer_dense_example', str_5802)

# Assigning a Str to a Name (line 265):
str_5803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, (-1)), 'str', '%%MatrixMarket matrix coordinate integer symmetric\n2  2  2\n1  1            2147483648\n2  2  19223372036854775808\n')
# Assigning a type to the variable '_over64bit_integer_sparse_example' (line 265)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), '_over64bit_integer_sparse_example', str_5803)
# Declaration of the 'TestMMIOReadLargeIntegers' class

class TestMMIOReadLargeIntegers(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOReadLargeIntegers.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOReadLargeIntegers.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOReadLargeIntegers.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOReadLargeIntegers.setup_method.__dict__.__setitem__('stypy_function_name', 'TestMMIOReadLargeIntegers.setup_method')
        TestMMIOReadLargeIntegers.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOReadLargeIntegers.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOReadLargeIntegers.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOReadLargeIntegers.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOReadLargeIntegers.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOReadLargeIntegers.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOReadLargeIntegers.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 274):
        
        # Call to mkdtemp(...): (line 274)
        # Processing the call keyword arguments (line 274)
        kwargs_5805 = {}
        # Getting the type of 'mkdtemp' (line 274)
        mkdtemp_5804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'mkdtemp', False)
        # Calling mkdtemp(args, kwargs) (line 274)
        mkdtemp_call_result_5806 = invoke(stypy.reporting.localization.Localization(__file__, 274, 22), mkdtemp_5804, *[], **kwargs_5805)
        
        # Getting the type of 'self' (line 274)
        self_5807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'self')
        # Setting the type of the member 'tmpdir' of a type (line 274)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), self_5807, 'tmpdir', mkdtemp_call_result_5806)
        
        # Assigning a Call to a Attribute (line 275):
        
        # Call to join(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'self' (line 275)
        self_5811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 31), 'self', False)
        # Obtaining the member 'tmpdir' of a type (line 275)
        tmpdir_5812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 31), self_5811, 'tmpdir')
        str_5813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 44), 'str', 'testfile.mtx')
        # Processing the call keyword arguments (line 275)
        kwargs_5814 = {}
        # Getting the type of 'os' (line 275)
        os_5808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 275)
        path_5809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 18), os_5808, 'path')
        # Obtaining the member 'join' of a type (line 275)
        join_5810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 18), path_5809, 'join')
        # Calling join(args, kwargs) (line 275)
        join_call_result_5815 = invoke(stypy.reporting.localization.Localization(__file__, 275, 18), join_5810, *[tmpdir_5812, str_5813], **kwargs_5814)
        
        # Getting the type of 'self' (line 275)
        self_5816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self')
        # Setting the type of the member 'fn' of a type (line 275)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_5816, 'fn', join_call_result_5815)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 273)
        stypy_return_type_5817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5817)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_5817


    @norecursion
    def teardown_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'teardown_method'
        module_type_store = module_type_store.open_function_context('teardown_method', 277, 4, False)
        # Assigning a type to the variable 'self' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOReadLargeIntegers.teardown_method.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOReadLargeIntegers.teardown_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOReadLargeIntegers.teardown_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOReadLargeIntegers.teardown_method.__dict__.__setitem__('stypy_function_name', 'TestMMIOReadLargeIntegers.teardown_method')
        TestMMIOReadLargeIntegers.teardown_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOReadLargeIntegers.teardown_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOReadLargeIntegers.teardown_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOReadLargeIntegers.teardown_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOReadLargeIntegers.teardown_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOReadLargeIntegers.teardown_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOReadLargeIntegers.teardown_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.teardown_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'teardown_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'teardown_method(...)' code ##################

        
        # Call to rmtree(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'self' (line 278)
        self_5820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 22), 'self', False)
        # Obtaining the member 'tmpdir' of a type (line 278)
        tmpdir_5821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 22), self_5820, 'tmpdir')
        # Processing the call keyword arguments (line 278)
        kwargs_5822 = {}
        # Getting the type of 'shutil' (line 278)
        shutil_5818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'shutil', False)
        # Obtaining the member 'rmtree' of a type (line 278)
        rmtree_5819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), shutil_5818, 'rmtree')
        # Calling rmtree(args, kwargs) (line 278)
        rmtree_call_result_5823 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), rmtree_5819, *[tmpdir_5821], **kwargs_5822)
        
        
        # ################# End of 'teardown_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'teardown_method' in the type store
        # Getting the type of 'stypy_return_type' (line 277)
        stypy_return_type_5824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5824)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'teardown_method'
        return stypy_return_type_5824


    @norecursion
    def check_read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_read'
        module_type_store = module_type_store.open_function_context('check_read', 280, 4, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOReadLargeIntegers.check_read.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOReadLargeIntegers.check_read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOReadLargeIntegers.check_read.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOReadLargeIntegers.check_read.__dict__.__setitem__('stypy_function_name', 'TestMMIOReadLargeIntegers.check_read')
        TestMMIOReadLargeIntegers.check_read.__dict__.__setitem__('stypy_param_names_list', ['example', 'a', 'info', 'dense', 'over32', 'over64'])
        TestMMIOReadLargeIntegers.check_read.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOReadLargeIntegers.check_read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOReadLargeIntegers.check_read.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOReadLargeIntegers.check_read.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOReadLargeIntegers.check_read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOReadLargeIntegers.check_read.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.check_read', ['example', 'a', 'info', 'dense', 'over32', 'over64'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_read', localization, ['example', 'a', 'info', 'dense', 'over32', 'over64'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_read(...)' code ##################

        
        # Call to open(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'self' (line 281)
        self_5826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 18), 'self', False)
        # Obtaining the member 'fn' of a type (line 281)
        fn_5827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 18), self_5826, 'fn')
        str_5828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 27), 'str', 'w')
        # Processing the call keyword arguments (line 281)
        kwargs_5829 = {}
        # Getting the type of 'open' (line 281)
        open_5825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 13), 'open', False)
        # Calling open(args, kwargs) (line 281)
        open_call_result_5830 = invoke(stypy.reporting.localization.Localization(__file__, 281, 13), open_5825, *[fn_5827, str_5828], **kwargs_5829)
        
        with_5831 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 281, 13), open_call_result_5830, 'with parameter', '__enter__', '__exit__')

        if with_5831:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 281)
            enter___5832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 13), open_call_result_5830, '__enter__')
            with_enter_5833 = invoke(stypy.reporting.localization.Localization(__file__, 281, 13), enter___5832)
            # Assigning a type to the variable 'f' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 13), 'f', with_enter_5833)
            
            # Call to write(...): (line 282)
            # Processing the call arguments (line 282)
            # Getting the type of 'example' (line 282)
            example_5836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'example', False)
            # Processing the call keyword arguments (line 282)
            kwargs_5837 = {}
            # Getting the type of 'f' (line 282)
            f_5834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'f', False)
            # Obtaining the member 'write' of a type (line 282)
            write_5835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), f_5834, 'write')
            # Calling write(args, kwargs) (line 282)
            write_call_result_5838 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), write_5835, *[example_5836], **kwargs_5837)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 281)
            exit___5839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 13), open_call_result_5830, '__exit__')
            with_exit_5840 = invoke(stypy.reporting.localization.Localization(__file__, 281, 13), exit___5839, None, None, None)

        
        # Call to assert_equal(...): (line 283)
        # Processing the call arguments (line 283)
        
        # Call to mminfo(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'self' (line 283)
        self_5843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 28), 'self', False)
        # Obtaining the member 'fn' of a type (line 283)
        fn_5844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 28), self_5843, 'fn')
        # Processing the call keyword arguments (line 283)
        kwargs_5845 = {}
        # Getting the type of 'mminfo' (line 283)
        mminfo_5842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 21), 'mminfo', False)
        # Calling mminfo(args, kwargs) (line 283)
        mminfo_call_result_5846 = invoke(stypy.reporting.localization.Localization(__file__, 283, 21), mminfo_5842, *[fn_5844], **kwargs_5845)
        
        # Getting the type of 'info' (line 283)
        info_5847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 38), 'info', False)
        # Processing the call keyword arguments (line 283)
        kwargs_5848 = {}
        # Getting the type of 'assert_equal' (line 283)
        assert_equal_5841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 283)
        assert_equal_call_result_5849 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), assert_equal_5841, *[mminfo_call_result_5846, info_5847], **kwargs_5848)
        
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'over32' (line 284)
        over32_5850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'over32')
        
        
        # Call to intp(...): (line 284)
        # Processing the call arguments (line 284)
        int_5853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 32), 'int')
        # Processing the call keyword arguments (line 284)
        kwargs_5854 = {}
        # Getting the type of 'np' (line 284)
        np_5851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'np', False)
        # Obtaining the member 'intp' of a type (line 284)
        intp_5852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 24), np_5851, 'intp')
        # Calling intp(args, kwargs) (line 284)
        intp_call_result_5855 = invoke(stypy.reporting.localization.Localization(__file__, 284, 24), intp_5852, *[int_5853], **kwargs_5854)
        
        # Obtaining the member 'itemsize' of a type (line 284)
        itemsize_5856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 24), intp_call_result_5855, 'itemsize')
        int_5857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 46), 'int')
        # Applying the binary operator '<' (line 284)
        result_lt_5858 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 24), '<', itemsize_5856, int_5857)
        
        # Applying the binary operator 'and' (line 284)
        result_and_keyword_5859 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 12), 'and', over32_5850, result_lt_5858)
        
        # Getting the type of 'over64' (line 284)
        over64_5860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 53), 'over64')
        # Applying the binary operator 'or' (line 284)
        result_or_keyword_5861 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), 'or', result_and_keyword_5859, over64_5860)
        
        # Testing the type of an if condition (line 284)
        if_condition_5862 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 8), result_or_keyword_5861)
        # Assigning a type to the variable 'if_condition_5862' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'if_condition_5862', if_condition_5862)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_raises(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'OverflowError' (line 285)
        OverflowError_5864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 26), 'OverflowError', False)
        # Getting the type of 'mmread' (line 285)
        mmread_5865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 41), 'mmread', False)
        # Getting the type of 'self' (line 285)
        self_5866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 49), 'self', False)
        # Obtaining the member 'fn' of a type (line 285)
        fn_5867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 49), self_5866, 'fn')
        # Processing the call keyword arguments (line 285)
        kwargs_5868 = {}
        # Getting the type of 'assert_raises' (line 285)
        assert_raises_5863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 285)
        assert_raises_call_result_5869 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), assert_raises_5863, *[OverflowError_5864, mmread_5865, fn_5867], **kwargs_5868)
        
        # SSA branch for the else part of an if statement (line 284)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 287):
        
        # Call to mmread(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'self' (line 287)
        self_5871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 23), 'self', False)
        # Obtaining the member 'fn' of a type (line 287)
        fn_5872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 23), self_5871, 'fn')
        # Processing the call keyword arguments (line 287)
        kwargs_5873 = {}
        # Getting the type of 'mmread' (line 287)
        mmread_5870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'mmread', False)
        # Calling mmread(args, kwargs) (line 287)
        mmread_call_result_5874 = invoke(stypy.reporting.localization.Localization(__file__, 287, 16), mmread_5870, *[fn_5872], **kwargs_5873)
        
        # Assigning a type to the variable 'b' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'b', mmread_call_result_5874)
        
        
        # Getting the type of 'dense' (line 288)
        dense_5875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 'dense')
        # Applying the 'not' unary operator (line 288)
        result_not__5876 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 15), 'not', dense_5875)
        
        # Testing the type of an if condition (line 288)
        if_condition_5877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 12), result_not__5876)
        # Assigning a type to the variable 'if_condition_5877' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'if_condition_5877', if_condition_5877)
        # SSA begins for if statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 289):
        
        # Call to todense(...): (line 289)
        # Processing the call keyword arguments (line 289)
        kwargs_5880 = {}
        # Getting the type of 'b' (line 289)
        b_5878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'b', False)
        # Obtaining the member 'todense' of a type (line 289)
        todense_5879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 20), b_5878, 'todense')
        # Calling todense(args, kwargs) (line 289)
        todense_call_result_5881 = invoke(stypy.reporting.localization.Localization(__file__, 289, 20), todense_5879, *[], **kwargs_5880)
        
        # Assigning a type to the variable 'b' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'b', todense_call_result_5881)
        # SSA join for if statement (line 288)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'a' (line 290)
        a_5883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 25), 'a', False)
        # Getting the type of 'b' (line 290)
        b_5884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 28), 'b', False)
        # Processing the call keyword arguments (line 290)
        kwargs_5885 = {}
        # Getting the type of 'assert_equal' (line 290)
        assert_equal_5882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 290)
        assert_equal_call_result_5886 = invoke(stypy.reporting.localization.Localization(__file__, 290, 12), assert_equal_5882, *[a_5883, b_5884], **kwargs_5885)
        
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_read' in the type store
        # Getting the type of 'stypy_return_type' (line 280)
        stypy_return_type_5887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5887)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_read'
        return stypy_return_type_5887


    @norecursion
    def test_read_32bit_integer_dense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_32bit_integer_dense'
        module_type_store = module_type_store.open_function_context('test_read_32bit_integer_dense', 292, 4, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOReadLargeIntegers.test_read_32bit_integer_dense.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_dense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_dense.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_dense.__dict__.__setitem__('stypy_function_name', 'TestMMIOReadLargeIntegers.test_read_32bit_integer_dense')
        TestMMIOReadLargeIntegers.test_read_32bit_integer_dense.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOReadLargeIntegers.test_read_32bit_integer_dense.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_dense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_dense.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_dense.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_dense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_dense.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.test_read_32bit_integer_dense', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_32bit_integer_dense', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_32bit_integer_dense(...)' code ##################

        
        # Assigning a Call to a Name (line 293):
        
        # Call to array(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_5889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_5890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        int_5891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 20), 'int')
        int_5892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 23), 'int')
        # Applying the binary operator '**' (line 293)
        result_pow_5893 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 20), '**', int_5891, int_5892)
        
        int_5894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 26), 'int')
        # Applying the binary operator '-' (line 293)
        result_sub_5895 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 20), '-', result_pow_5893, int_5894)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 19), list_5890, result_sub_5895)
        # Adding element type (line 293)
        int_5896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 29), 'int')
        int_5897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 32), 'int')
        # Applying the binary operator '**' (line 293)
        result_pow_5898 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 29), '**', int_5896, int_5897)
        
        int_5899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 35), 'int')
        # Applying the binary operator '-' (line 293)
        result_sub_5900 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 29), '-', result_pow_5898, int_5899)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 19), list_5890, result_sub_5900)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 18), list_5889, list_5890)
        # Adding element type (line 293)
        
        # Obtaining an instance of the builtin type 'list' (line 294)
        list_5901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 294)
        # Adding element type (line 294)
        int_5902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 20), 'int')
        int_5903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 23), 'int')
        # Applying the binary operator '**' (line 294)
        result_pow_5904 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 20), '**', int_5902, int_5903)
        
        int_5905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 26), 'int')
        # Applying the binary operator '-' (line 294)
        result_sub_5906 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 20), '-', result_pow_5904, int_5905)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 19), list_5901, result_sub_5906)
        # Adding element type (line 294)
        int_5907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 29), 'int')
        int_5908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 32), 'int')
        # Applying the binary operator '**' (line 294)
        result_pow_5909 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 29), '**', int_5907, int_5908)
        
        int_5910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 35), 'int')
        # Applying the binary operator '-' (line 294)
        result_sub_5911 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 29), '-', result_pow_5909, int_5910)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 19), list_5901, result_sub_5911)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 18), list_5889, list_5901)
        
        # Processing the call keyword arguments (line 293)
        # Getting the type of 'np' (line 294)
        np_5912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 46), 'np', False)
        # Obtaining the member 'int64' of a type (line 294)
        int64_5913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 46), np_5912, 'int64')
        keyword_5914 = int64_5913
        kwargs_5915 = {'dtype': keyword_5914}
        # Getting the type of 'array' (line 293)
        array_5888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'array', False)
        # Calling array(args, kwargs) (line 293)
        array_call_result_5916 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), array_5888, *[list_5889], **kwargs_5915)
        
        # Assigning a type to the variable 'a' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'a', array_call_result_5916)
        
        # Call to check_read(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of '_32bit_integer_dense_example' (line 295)
        _32bit_integer_dense_example_5919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), '_32bit_integer_dense_example', False)
        # Getting the type of 'a' (line 296)
        a_5920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 24), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 297)
        tuple_5921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 297)
        # Adding element type (line 297)
        int_5922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 25), tuple_5921, int_5922)
        # Adding element type (line 297)
        int_5923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 25), tuple_5921, int_5923)
        # Adding element type (line 297)
        int_5924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 25), tuple_5921, int_5924)
        # Adding element type (line 297)
        str_5925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 34), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 25), tuple_5921, str_5925)
        # Adding element type (line 297)
        str_5926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 43), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 25), tuple_5921, str_5926)
        # Adding element type (line 297)
        str_5927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 54), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 25), tuple_5921, str_5927)
        
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'True' (line 298)
        True_5928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 30), 'True', False)
        keyword_5929 = True_5928
        # Getting the type of 'False' (line 299)
        False_5930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 31), 'False', False)
        keyword_5931 = False_5930
        # Getting the type of 'False' (line 300)
        False_5932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 31), 'False', False)
        keyword_5933 = False_5932
        kwargs_5934 = {'dense': keyword_5929, 'over64': keyword_5933, 'over32': keyword_5931}
        # Getting the type of 'self' (line 295)
        self_5917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 295)
        check_read_5918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), self_5917, 'check_read')
        # Calling check_read(args, kwargs) (line 295)
        check_read_call_result_5935 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), check_read_5918, *[_32bit_integer_dense_example_5919, a_5920, tuple_5921], **kwargs_5934)
        
        
        # ################# End of 'test_read_32bit_integer_dense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_32bit_integer_dense' in the type store
        # Getting the type of 'stypy_return_type' (line 292)
        stypy_return_type_5936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5936)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_32bit_integer_dense'
        return stypy_return_type_5936


    @norecursion
    def test_read_32bit_integer_sparse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_32bit_integer_sparse'
        module_type_store = module_type_store.open_function_context('test_read_32bit_integer_sparse', 302, 4, False)
        # Assigning a type to the variable 'self' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse.__dict__.__setitem__('stypy_function_name', 'TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse')
        TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.test_read_32bit_integer_sparse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_32bit_integer_sparse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_32bit_integer_sparse(...)' code ##################

        
        # Assigning a Call to a Name (line 303):
        
        # Call to array(...): (line 303)
        # Processing the call arguments (line 303)
        
        # Obtaining an instance of the builtin type 'list' (line 303)
        list_5938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 303)
        # Adding element type (line 303)
        
        # Obtaining an instance of the builtin type 'list' (line 303)
        list_5939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 303)
        # Adding element type (line 303)
        int_5940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 20), 'int')
        int_5941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 23), 'int')
        # Applying the binary operator '**' (line 303)
        result_pow_5942 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 20), '**', int_5940, int_5941)
        
        int_5943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 26), 'int')
        # Applying the binary operator '-' (line 303)
        result_sub_5944 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 20), '-', result_pow_5942, int_5943)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 19), list_5939, result_sub_5944)
        # Adding element type (line 303)
        int_5945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 19), list_5939, int_5945)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 18), list_5938, list_5939)
        # Adding element type (line 303)
        
        # Obtaining an instance of the builtin type 'list' (line 304)
        list_5946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 304)
        # Adding element type (line 304)
        int_5947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 19), list_5946, int_5947)
        # Adding element type (line 304)
        int_5948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 23), 'int')
        int_5949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 26), 'int')
        # Applying the binary operator '**' (line 304)
        result_pow_5950 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 23), '**', int_5948, int_5949)
        
        int_5951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 29), 'int')
        # Applying the binary operator '-' (line 304)
        result_sub_5952 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 23), '-', result_pow_5950, int_5951)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 19), list_5946, result_sub_5952)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 18), list_5938, list_5946)
        
        # Processing the call keyword arguments (line 303)
        # Getting the type of 'np' (line 304)
        np_5953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 40), 'np', False)
        # Obtaining the member 'int64' of a type (line 304)
        int64_5954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 40), np_5953, 'int64')
        keyword_5955 = int64_5954
        kwargs_5956 = {'dtype': keyword_5955}
        # Getting the type of 'array' (line 303)
        array_5937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'array', False)
        # Calling array(args, kwargs) (line 303)
        array_call_result_5957 = invoke(stypy.reporting.localization.Localization(__file__, 303, 12), array_5937, *[list_5938], **kwargs_5956)
        
        # Assigning a type to the variable 'a' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'a', array_call_result_5957)
        
        # Call to check_read(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of '_32bit_integer_sparse_example' (line 305)
        _32bit_integer_sparse_example_5960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 24), '_32bit_integer_sparse_example', False)
        # Getting the type of 'a' (line 306)
        a_5961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 24), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 307)
        tuple_5962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 307)
        # Adding element type (line 307)
        int_5963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 25), tuple_5962, int_5963)
        # Adding element type (line 307)
        int_5964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 25), tuple_5962, int_5964)
        # Adding element type (line 307)
        int_5965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 25), tuple_5962, int_5965)
        # Adding element type (line 307)
        str_5966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 34), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 25), tuple_5962, str_5966)
        # Adding element type (line 307)
        str_5967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 48), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 25), tuple_5962, str_5967)
        # Adding element type (line 307)
        str_5968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 59), 'str', 'symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 25), tuple_5962, str_5968)
        
        # Processing the call keyword arguments (line 305)
        # Getting the type of 'False' (line 308)
        False_5969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 30), 'False', False)
        keyword_5970 = False_5969
        # Getting the type of 'False' (line 309)
        False_5971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 31), 'False', False)
        keyword_5972 = False_5971
        # Getting the type of 'False' (line 310)
        False_5973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 31), 'False', False)
        keyword_5974 = False_5973
        kwargs_5975 = {'dense': keyword_5970, 'over64': keyword_5974, 'over32': keyword_5972}
        # Getting the type of 'self' (line 305)
        self_5958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 305)
        check_read_5959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), self_5958, 'check_read')
        # Calling check_read(args, kwargs) (line 305)
        check_read_call_result_5976 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), check_read_5959, *[_32bit_integer_sparse_example_5960, a_5961, tuple_5962], **kwargs_5975)
        
        
        # ################# End of 'test_read_32bit_integer_sparse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_32bit_integer_sparse' in the type store
        # Getting the type of 'stypy_return_type' (line 302)
        stypy_return_type_5977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5977)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_32bit_integer_sparse'
        return stypy_return_type_5977


    @norecursion
    def test_read_64bit_integer_dense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_64bit_integer_dense'
        module_type_store = module_type_store.open_function_context('test_read_64bit_integer_dense', 312, 4, False)
        # Assigning a type to the variable 'self' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOReadLargeIntegers.test_read_64bit_integer_dense.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_dense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_dense.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_dense.__dict__.__setitem__('stypy_function_name', 'TestMMIOReadLargeIntegers.test_read_64bit_integer_dense')
        TestMMIOReadLargeIntegers.test_read_64bit_integer_dense.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOReadLargeIntegers.test_read_64bit_integer_dense.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_dense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_dense.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_dense.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_dense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_dense.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.test_read_64bit_integer_dense', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_64bit_integer_dense', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_64bit_integer_dense(...)' code ##################

        
        # Assigning a Call to a Name (line 313):
        
        # Call to array(...): (line 313)
        # Processing the call arguments (line 313)
        
        # Obtaining an instance of the builtin type 'list' (line 313)
        list_5979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 313)
        # Adding element type (line 313)
        
        # Obtaining an instance of the builtin type 'list' (line 313)
        list_5980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 313)
        # Adding element type (line 313)
        int_5981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 20), 'int')
        int_5982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 23), 'int')
        # Applying the binary operator '**' (line 313)
        result_pow_5983 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 20), '**', int_5981, int_5982)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 19), list_5980, result_pow_5983)
        # Adding element type (line 313)
        
        int_5984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 28), 'int')
        int_5985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 31), 'int')
        # Applying the binary operator '**' (line 313)
        result_pow_5986 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 28), '**', int_5984, int_5985)
        
        # Applying the 'usub' unary operator (line 313)
        result___neg___5987 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 27), 'usub', result_pow_5986)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 19), list_5980, result___neg___5987)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 18), list_5979, list_5980)
        # Adding element type (line 313)
        
        # Obtaining an instance of the builtin type 'list' (line 314)
        list_5988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 314)
        # Adding element type (line 314)
        
        int_5989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 21), 'int')
        int_5990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 24), 'int')
        # Applying the binary operator '**' (line 314)
        result_pow_5991 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 21), '**', int_5989, int_5990)
        
        # Applying the 'usub' unary operator (line 314)
        result___neg___5992 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 20), 'usub', result_pow_5991)
        
        int_5993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 27), 'int')
        # Applying the binary operator '+' (line 314)
        result_add_5994 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 20), '+', result___neg___5992, int_5993)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 19), list_5988, result_add_5994)
        # Adding element type (line 314)
        int_5995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 30), 'int')
        int_5996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 33), 'int')
        # Applying the binary operator '**' (line 314)
        result_pow_5997 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 30), '**', int_5995, int_5996)
        
        int_5998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 36), 'int')
        # Applying the binary operator '-' (line 314)
        result_sub_5999 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 30), '-', result_pow_5997, int_5998)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 19), list_5988, result_sub_5999)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 18), list_5979, list_5988)
        
        # Processing the call keyword arguments (line 313)
        # Getting the type of 'np' (line 314)
        np_6000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 47), 'np', False)
        # Obtaining the member 'int64' of a type (line 314)
        int64_6001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 47), np_6000, 'int64')
        keyword_6002 = int64_6001
        kwargs_6003 = {'dtype': keyword_6002}
        # Getting the type of 'array' (line 313)
        array_5978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'array', False)
        # Calling array(args, kwargs) (line 313)
        array_call_result_6004 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), array_5978, *[list_5979], **kwargs_6003)
        
        # Assigning a type to the variable 'a' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'a', array_call_result_6004)
        
        # Call to check_read(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of '_64bit_integer_dense_example' (line 315)
        _64bit_integer_dense_example_6007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 24), '_64bit_integer_dense_example', False)
        # Getting the type of 'a' (line 316)
        a_6008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 24), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 317)
        tuple_6009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 317)
        # Adding element type (line 317)
        int_6010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 25), tuple_6009, int_6010)
        # Adding element type (line 317)
        int_6011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 25), tuple_6009, int_6011)
        # Adding element type (line 317)
        int_6012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 25), tuple_6009, int_6012)
        # Adding element type (line 317)
        str_6013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 34), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 25), tuple_6009, str_6013)
        # Adding element type (line 317)
        str_6014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 43), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 25), tuple_6009, str_6014)
        # Adding element type (line 317)
        str_6015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 54), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 25), tuple_6009, str_6015)
        
        # Processing the call keyword arguments (line 315)
        # Getting the type of 'True' (line 318)
        True_6016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 30), 'True', False)
        keyword_6017 = True_6016
        # Getting the type of 'True' (line 319)
        True_6018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 31), 'True', False)
        keyword_6019 = True_6018
        # Getting the type of 'False' (line 320)
        False_6020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 31), 'False', False)
        keyword_6021 = False_6020
        kwargs_6022 = {'dense': keyword_6017, 'over64': keyword_6021, 'over32': keyword_6019}
        # Getting the type of 'self' (line 315)
        self_6005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 315)
        check_read_6006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_6005, 'check_read')
        # Calling check_read(args, kwargs) (line 315)
        check_read_call_result_6023 = invoke(stypy.reporting.localization.Localization(__file__, 315, 8), check_read_6006, *[_64bit_integer_dense_example_6007, a_6008, tuple_6009], **kwargs_6022)
        
        
        # ################# End of 'test_read_64bit_integer_dense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_64bit_integer_dense' in the type store
        # Getting the type of 'stypy_return_type' (line 312)
        stypy_return_type_6024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6024)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_64bit_integer_dense'
        return stypy_return_type_6024


    @norecursion
    def test_read_64bit_integer_sparse_general(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_64bit_integer_sparse_general'
        module_type_store = module_type_store.open_function_context('test_read_64bit_integer_sparse_general', 322, 4, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general.__dict__.__setitem__('stypy_function_name', 'TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general')
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_general', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_64bit_integer_sparse_general', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_64bit_integer_sparse_general(...)' code ##################

        
        # Assigning a Call to a Name (line 323):
        
        # Call to array(...): (line 323)
        # Processing the call arguments (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_6026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_6027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        int_6028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 20), 'int')
        int_6029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 23), 'int')
        # Applying the binary operator '**' (line 323)
        result_pow_6030 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 20), '**', int_6028, int_6029)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 19), list_6027, result_pow_6030)
        # Adding element type (line 323)
        int_6031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 27), 'int')
        int_6032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 30), 'int')
        # Applying the binary operator '**' (line 323)
        result_pow_6033 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 27), '**', int_6031, int_6032)
        
        int_6034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 33), 'int')
        # Applying the binary operator '-' (line 323)
        result_sub_6035 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 27), '-', result_pow_6033, int_6034)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 19), list_6027, result_sub_6035)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 18), list_6026, list_6027)
        # Adding element type (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_6036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)
        int_6037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 19), list_6036, int_6037)
        # Adding element type (line 324)
        int_6038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 23), 'int')
        int_6039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 26), 'int')
        # Applying the binary operator '**' (line 324)
        result_pow_6040 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 23), '**', int_6038, int_6039)
        
        int_6041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 29), 'int')
        # Applying the binary operator '-' (line 324)
        result_sub_6042 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 23), '-', result_pow_6040, int_6041)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 19), list_6036, result_sub_6042)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 18), list_6026, list_6036)
        
        # Processing the call keyword arguments (line 323)
        # Getting the type of 'np' (line 324)
        np_6043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 40), 'np', False)
        # Obtaining the member 'int64' of a type (line 324)
        int64_6044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 40), np_6043, 'int64')
        keyword_6045 = int64_6044
        kwargs_6046 = {'dtype': keyword_6045}
        # Getting the type of 'array' (line 323)
        array_6025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'array', False)
        # Calling array(args, kwargs) (line 323)
        array_call_result_6047 = invoke(stypy.reporting.localization.Localization(__file__, 323, 12), array_6025, *[list_6026], **kwargs_6046)
        
        # Assigning a type to the variable 'a' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'a', array_call_result_6047)
        
        # Call to check_read(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of '_64bit_integer_sparse_general_example' (line 325)
        _64bit_integer_sparse_general_example_6050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 24), '_64bit_integer_sparse_general_example', False)
        # Getting the type of 'a' (line 326)
        a_6051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 24), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 327)
        tuple_6052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 327)
        # Adding element type (line 327)
        int_6053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 25), tuple_6052, int_6053)
        # Adding element type (line 327)
        int_6054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 25), tuple_6052, int_6054)
        # Adding element type (line 327)
        int_6055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 25), tuple_6052, int_6055)
        # Adding element type (line 327)
        str_6056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 34), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 25), tuple_6052, str_6056)
        # Adding element type (line 327)
        str_6057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 48), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 25), tuple_6052, str_6057)
        # Adding element type (line 327)
        str_6058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 59), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 25), tuple_6052, str_6058)
        
        # Processing the call keyword arguments (line 325)
        # Getting the type of 'False' (line 328)
        False_6059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 30), 'False', False)
        keyword_6060 = False_6059
        # Getting the type of 'True' (line 329)
        True_6061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 31), 'True', False)
        keyword_6062 = True_6061
        # Getting the type of 'False' (line 330)
        False_6063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 31), 'False', False)
        keyword_6064 = False_6063
        kwargs_6065 = {'dense': keyword_6060, 'over64': keyword_6064, 'over32': keyword_6062}
        # Getting the type of 'self' (line 325)
        self_6048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 325)
        check_read_6049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), self_6048, 'check_read')
        # Calling check_read(args, kwargs) (line 325)
        check_read_call_result_6066 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), check_read_6049, *[_64bit_integer_sparse_general_example_6050, a_6051, tuple_6052], **kwargs_6065)
        
        
        # ################# End of 'test_read_64bit_integer_sparse_general(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_64bit_integer_sparse_general' in the type store
        # Getting the type of 'stypy_return_type' (line 322)
        stypy_return_type_6067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6067)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_64bit_integer_sparse_general'
        return stypy_return_type_6067


    @norecursion
    def test_read_64bit_integer_sparse_symmetric(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_64bit_integer_sparse_symmetric'
        module_type_store = module_type_store.open_function_context('test_read_64bit_integer_sparse_symmetric', 332, 4, False)
        # Assigning a type to the variable 'self' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric.__dict__.__setitem__('stypy_function_name', 'TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric')
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_symmetric', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_64bit_integer_sparse_symmetric', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_64bit_integer_sparse_symmetric(...)' code ##################

        
        # Assigning a Call to a Name (line 333):
        
        # Call to array(...): (line 333)
        # Processing the call arguments (line 333)
        
        # Obtaining an instance of the builtin type 'list' (line 333)
        list_6069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 333)
        # Adding element type (line 333)
        
        # Obtaining an instance of the builtin type 'list' (line 333)
        list_6070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 333)
        # Adding element type (line 333)
        int_6071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 20), 'int')
        int_6072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 23), 'int')
        # Applying the binary operator '**' (line 333)
        result_pow_6073 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 20), '**', int_6071, int_6072)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 19), list_6070, result_pow_6073)
        # Adding element type (line 333)
        
        int_6074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 28), 'int')
        int_6075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 31), 'int')
        # Applying the binary operator '**' (line 333)
        result_pow_6076 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 28), '**', int_6074, int_6075)
        
        # Applying the 'usub' unary operator (line 333)
        result___neg___6077 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 27), 'usub', result_pow_6076)
        
        int_6078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 34), 'int')
        # Applying the binary operator '+' (line 333)
        result_add_6079 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 27), '+', result___neg___6077, int_6078)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 19), list_6070, result_add_6079)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 18), list_6069, list_6070)
        # Adding element type (line 333)
        
        # Obtaining an instance of the builtin type 'list' (line 334)
        list_6080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 334)
        # Adding element type (line 334)
        
        int_6081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 21), 'int')
        int_6082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 24), 'int')
        # Applying the binary operator '**' (line 334)
        result_pow_6083 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 21), '**', int_6081, int_6082)
        
        # Applying the 'usub' unary operator (line 334)
        result___neg___6084 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 20), 'usub', result_pow_6083)
        
        int_6085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 27), 'int')
        # Applying the binary operator '+' (line 334)
        result_add_6086 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 20), '+', result___neg___6084, int_6085)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 19), list_6080, result_add_6086)
        # Adding element type (line 334)
        int_6087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 30), 'int')
        int_6088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 33), 'int')
        # Applying the binary operator '**' (line 334)
        result_pow_6089 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 30), '**', int_6087, int_6088)
        
        int_6090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 36), 'int')
        # Applying the binary operator '-' (line 334)
        result_sub_6091 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 30), '-', result_pow_6089, int_6090)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 19), list_6080, result_sub_6091)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 18), list_6069, list_6080)
        
        # Processing the call keyword arguments (line 333)
        # Getting the type of 'np' (line 334)
        np_6092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 47), 'np', False)
        # Obtaining the member 'int64' of a type (line 334)
        int64_6093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 47), np_6092, 'int64')
        keyword_6094 = int64_6093
        kwargs_6095 = {'dtype': keyword_6094}
        # Getting the type of 'array' (line 333)
        array_6068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'array', False)
        # Calling array(args, kwargs) (line 333)
        array_call_result_6096 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), array_6068, *[list_6069], **kwargs_6095)
        
        # Assigning a type to the variable 'a' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'a', array_call_result_6096)
        
        # Call to check_read(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of '_64bit_integer_sparse_symmetric_example' (line 335)
        _64bit_integer_sparse_symmetric_example_6099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 24), '_64bit_integer_sparse_symmetric_example', False)
        # Getting the type of 'a' (line 336)
        a_6100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 337)
        tuple_6101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 337)
        # Adding element type (line 337)
        int_6102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 25), tuple_6101, int_6102)
        # Adding element type (line 337)
        int_6103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 25), tuple_6101, int_6103)
        # Adding element type (line 337)
        int_6104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 25), tuple_6101, int_6104)
        # Adding element type (line 337)
        str_6105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 34), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 25), tuple_6101, str_6105)
        # Adding element type (line 337)
        str_6106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 48), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 25), tuple_6101, str_6106)
        # Adding element type (line 337)
        str_6107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 59), 'str', 'symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 25), tuple_6101, str_6107)
        
        # Processing the call keyword arguments (line 335)
        # Getting the type of 'False' (line 338)
        False_6108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 30), 'False', False)
        keyword_6109 = False_6108
        # Getting the type of 'True' (line 339)
        True_6110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'True', False)
        keyword_6111 = True_6110
        # Getting the type of 'False' (line 340)
        False_6112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 31), 'False', False)
        keyword_6113 = False_6112
        kwargs_6114 = {'dense': keyword_6109, 'over64': keyword_6113, 'over32': keyword_6111}
        # Getting the type of 'self' (line 335)
        self_6097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 335)
        check_read_6098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), self_6097, 'check_read')
        # Calling check_read(args, kwargs) (line 335)
        check_read_call_result_6115 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), check_read_6098, *[_64bit_integer_sparse_symmetric_example_6099, a_6100, tuple_6101], **kwargs_6114)
        
        
        # ################# End of 'test_read_64bit_integer_sparse_symmetric(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_64bit_integer_sparse_symmetric' in the type store
        # Getting the type of 'stypy_return_type' (line 332)
        stypy_return_type_6116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6116)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_64bit_integer_sparse_symmetric'
        return stypy_return_type_6116


    @norecursion
    def test_read_64bit_integer_sparse_skew(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_64bit_integer_sparse_skew'
        module_type_store = module_type_store.open_function_context('test_read_64bit_integer_sparse_skew', 342, 4, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew.__dict__.__setitem__('stypy_function_name', 'TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew')
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.test_read_64bit_integer_sparse_skew', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_64bit_integer_sparse_skew', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_64bit_integer_sparse_skew(...)' code ##################

        
        # Assigning a Call to a Name (line 343):
        
        # Call to array(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_6118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_6119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        int_6120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 20), 'int')
        int_6121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 23), 'int')
        # Applying the binary operator '**' (line 343)
        result_pow_6122 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 20), '**', int_6120, int_6121)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 19), list_6119, result_pow_6122)
        # Adding element type (line 343)
        
        int_6123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 28), 'int')
        int_6124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 31), 'int')
        # Applying the binary operator '**' (line 343)
        result_pow_6125 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 28), '**', int_6123, int_6124)
        
        # Applying the 'usub' unary operator (line 343)
        result___neg___6126 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 27), 'usub', result_pow_6125)
        
        int_6127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 34), 'int')
        # Applying the binary operator '+' (line 343)
        result_add_6128 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 27), '+', result___neg___6126, int_6127)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 19), list_6119, result_add_6128)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 18), list_6118, list_6119)
        # Adding element type (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_6129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        # Adding element type (line 344)
        int_6130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 20), 'int')
        int_6131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 23), 'int')
        # Applying the binary operator '**' (line 344)
        result_pow_6132 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 20), '**', int_6130, int_6131)
        
        int_6133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 26), 'int')
        # Applying the binary operator '-' (line 344)
        result_sub_6134 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 20), '-', result_pow_6132, int_6133)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 19), list_6129, result_sub_6134)
        # Adding element type (line 344)
        int_6135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 29), 'int')
        int_6136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 32), 'int')
        # Applying the binary operator '**' (line 344)
        result_pow_6137 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 29), '**', int_6135, int_6136)
        
        int_6138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 35), 'int')
        # Applying the binary operator '-' (line 344)
        result_sub_6139 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 29), '-', result_pow_6137, int_6138)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 19), list_6129, result_sub_6139)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 18), list_6118, list_6129)
        
        # Processing the call keyword arguments (line 343)
        # Getting the type of 'np' (line 344)
        np_6140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 46), 'np', False)
        # Obtaining the member 'int64' of a type (line 344)
        int64_6141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 46), np_6140, 'int64')
        keyword_6142 = int64_6141
        kwargs_6143 = {'dtype': keyword_6142}
        # Getting the type of 'array' (line 343)
        array_6117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'array', False)
        # Calling array(args, kwargs) (line 343)
        array_call_result_6144 = invoke(stypy.reporting.localization.Localization(__file__, 343, 12), array_6117, *[list_6118], **kwargs_6143)
        
        # Assigning a type to the variable 'a' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'a', array_call_result_6144)
        
        # Call to check_read(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of '_64bit_integer_sparse_skew_example' (line 345)
        _64bit_integer_sparse_skew_example_6147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), '_64bit_integer_sparse_skew_example', False)
        # Getting the type of 'a' (line 346)
        a_6148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 24), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 347)
        tuple_6149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 347)
        # Adding element type (line 347)
        int_6150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 25), tuple_6149, int_6150)
        # Adding element type (line 347)
        int_6151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 25), tuple_6149, int_6151)
        # Adding element type (line 347)
        int_6152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 25), tuple_6149, int_6152)
        # Adding element type (line 347)
        str_6153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 34), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 25), tuple_6149, str_6153)
        # Adding element type (line 347)
        str_6154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 48), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 25), tuple_6149, str_6154)
        # Adding element type (line 347)
        str_6155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 59), 'str', 'skew-symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 25), tuple_6149, str_6155)
        
        # Processing the call keyword arguments (line 345)
        # Getting the type of 'False' (line 348)
        False_6156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 30), 'False', False)
        keyword_6157 = False_6156
        # Getting the type of 'True' (line 349)
        True_6158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 31), 'True', False)
        keyword_6159 = True_6158
        # Getting the type of 'False' (line 350)
        False_6160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 31), 'False', False)
        keyword_6161 = False_6160
        kwargs_6162 = {'dense': keyword_6157, 'over64': keyword_6161, 'over32': keyword_6159}
        # Getting the type of 'self' (line 345)
        self_6145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 345)
        check_read_6146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), self_6145, 'check_read')
        # Calling check_read(args, kwargs) (line 345)
        check_read_call_result_6163 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), check_read_6146, *[_64bit_integer_sparse_skew_example_6147, a_6148, tuple_6149], **kwargs_6162)
        
        
        # ################# End of 'test_read_64bit_integer_sparse_skew(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_64bit_integer_sparse_skew' in the type store
        # Getting the type of 'stypy_return_type' (line 342)
        stypy_return_type_6164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6164)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_64bit_integer_sparse_skew'
        return stypy_return_type_6164


    @norecursion
    def test_read_over64bit_integer_dense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_over64bit_integer_dense'
        module_type_store = module_type_store.open_function_context('test_read_over64bit_integer_dense', 352, 4, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense.__dict__.__setitem__('stypy_function_name', 'TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense')
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.test_read_over64bit_integer_dense', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_over64bit_integer_dense', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_over64bit_integer_dense(...)' code ##################

        
        # Call to check_read(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of '_over64bit_integer_dense_example' (line 353)
        _over64bit_integer_dense_example_6167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 24), '_over64bit_integer_dense_example', False)
        # Getting the type of 'None' (line 354)
        None_6168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'None', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 355)
        tuple_6169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 355)
        # Adding element type (line 355)
        int_6170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 25), tuple_6169, int_6170)
        # Adding element type (line 355)
        int_6171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 25), tuple_6169, int_6171)
        # Adding element type (line 355)
        int_6172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 25), tuple_6169, int_6172)
        # Adding element type (line 355)
        str_6173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 34), 'str', 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 25), tuple_6169, str_6173)
        # Adding element type (line 355)
        str_6174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 43), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 25), tuple_6169, str_6174)
        # Adding element type (line 355)
        str_6175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 54), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 25), tuple_6169, str_6175)
        
        # Processing the call keyword arguments (line 353)
        # Getting the type of 'True' (line 356)
        True_6176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 30), 'True', False)
        keyword_6177 = True_6176
        # Getting the type of 'True' (line 357)
        True_6178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 31), 'True', False)
        keyword_6179 = True_6178
        # Getting the type of 'True' (line 358)
        True_6180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 31), 'True', False)
        keyword_6181 = True_6180
        kwargs_6182 = {'dense': keyword_6177, 'over64': keyword_6181, 'over32': keyword_6179}
        # Getting the type of 'self' (line 353)
        self_6165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 353)
        check_read_6166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), self_6165, 'check_read')
        # Calling check_read(args, kwargs) (line 353)
        check_read_call_result_6183 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), check_read_6166, *[_over64bit_integer_dense_example_6167, None_6168, tuple_6169], **kwargs_6182)
        
        
        # ################# End of 'test_read_over64bit_integer_dense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_over64bit_integer_dense' in the type store
        # Getting the type of 'stypy_return_type' (line 352)
        stypy_return_type_6184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6184)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_over64bit_integer_dense'
        return stypy_return_type_6184


    @norecursion
    def test_read_over64bit_integer_sparse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_over64bit_integer_sparse'
        module_type_store = module_type_store.open_function_context('test_read_over64bit_integer_sparse', 360, 4, False)
        # Assigning a type to the variable 'self' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse.__dict__.__setitem__('stypy_function_name', 'TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse')
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.test_read_over64bit_integer_sparse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_over64bit_integer_sparse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_over64bit_integer_sparse(...)' code ##################

        
        # Call to check_read(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of '_over64bit_integer_sparse_example' (line 361)
        _over64bit_integer_sparse_example_6187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 24), '_over64bit_integer_sparse_example', False)
        # Getting the type of 'None' (line 362)
        None_6188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 24), 'None', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 363)
        tuple_6189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 363)
        # Adding element type (line 363)
        int_6190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 25), tuple_6189, int_6190)
        # Adding element type (line 363)
        int_6191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 25), tuple_6189, int_6191)
        # Adding element type (line 363)
        int_6192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 25), tuple_6189, int_6192)
        # Adding element type (line 363)
        str_6193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 34), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 25), tuple_6189, str_6193)
        # Adding element type (line 363)
        str_6194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 48), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 25), tuple_6189, str_6194)
        # Adding element type (line 363)
        str_6195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 59), 'str', 'symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 25), tuple_6189, str_6195)
        
        # Processing the call keyword arguments (line 361)
        # Getting the type of 'False' (line 364)
        False_6196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'False', False)
        keyword_6197 = False_6196
        # Getting the type of 'True' (line 365)
        True_6198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 31), 'True', False)
        keyword_6199 = True_6198
        # Getting the type of 'True' (line 366)
        True_6200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 31), 'True', False)
        keyword_6201 = True_6200
        kwargs_6202 = {'dense': keyword_6197, 'over64': keyword_6201, 'over32': keyword_6199}
        # Getting the type of 'self' (line 361)
        self_6185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 361)
        check_read_6186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), self_6185, 'check_read')
        # Calling check_read(args, kwargs) (line 361)
        check_read_call_result_6203 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), check_read_6186, *[_over64bit_integer_sparse_example_6187, None_6188, tuple_6189], **kwargs_6202)
        
        
        # ################# End of 'test_read_over64bit_integer_sparse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_over64bit_integer_sparse' in the type store
        # Getting the type of 'stypy_return_type' (line 360)
        stypy_return_type_6204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6204)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_over64bit_integer_sparse'
        return stypy_return_type_6204


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 272, 0, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOReadLargeIntegers.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestMMIOReadLargeIntegers' (line 272)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 0), 'TestMMIOReadLargeIntegers', TestMMIOReadLargeIntegers)

# Assigning a Str to a Name (line 369):
str_6205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, (-1)), 'str', '%%MatrixMarket matrix coordinate real general\n%=================================================================================\n%\n% This ASCII file represents a sparse MxN matrix with L\n% nonzeros in the following Matrix Market format:\n%\n% +----------------------------------------------+\n% |%%MatrixMarket matrix coordinate real general | <--- header line\n% |%                                             | <--+\n% |% comments                                    |    |-- 0 or more comment lines\n% |%                                             | <--+\n% |    M  N  L                                   | <--- rows, columns, entries\n% |    I1  J1  A(I1, J1)                         | <--+\n% |    I2  J2  A(I2, J2)                         |    |\n% |    I3  J3  A(I3, J3)                         |    |-- L lines\n% |        . . .                                 |    |\n% |    IL JL  A(IL, JL)                          | <--+\n% +----------------------------------------------+\n%\n% Indices are 1-based, i.e. A(1,1) is the first element.\n%\n%=================================================================================\n  5  5  8\n    1     1   1.000e+00\n    2     2   1.050e+01\n    3     3   1.500e-02\n    1     4   6.000e+00\n    4     2   2.505e+02\n    4     4  -2.800e+02\n    4     5   3.332e+01\n    5     5   1.200e+01\n')
# Assigning a type to the variable '_general_example' (line 369)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 0), '_general_example', str_6205)

# Assigning a Str to a Name (line 403):
str_6206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, (-1)), 'str', '%%MatrixMarket matrix coordinate complex hermitian\n  5  5  7\n    1     1     1.0      0\n    2     2    10.5      0\n    4     2   250.5     22.22\n    3     3     1.5e-2   0\n    4     4    -2.8e2    0\n    5     5    12.       0\n    5     4     0       33.32\n')
# Assigning a type to the variable '_hermitian_example' (line 403)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 0), '_hermitian_example', str_6206)

# Assigning a Str to a Name (line 415):
str_6207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, (-1)), 'str', '%%MatrixMarket matrix coordinate real skew-symmetric\n  5  5  7\n    1     1     1.0\n    2     2    10.5\n    4     2   250.5\n    3     3     1.5e-2\n    4     4    -2.8e2\n    5     5    12.\n    5     4     0\n')
# Assigning a type to the variable '_skew_example' (line 415)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 0), '_skew_example', str_6207)

# Assigning a Str to a Name (line 427):
str_6208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, (-1)), 'str', '%%MatrixMarket matrix coordinate real symmetric\n  5  5  7\n    1     1     1.0\n    2     2    10.5\n    4     2   250.5\n    3     3     1.5e-2\n    4     4    -2.8e2\n    5     5    12.\n    5     4     8\n')
# Assigning a type to the variable '_symmetric_example' (line 427)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 0), '_symmetric_example', str_6208)

# Assigning a Str to a Name (line 439):
str_6209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, (-1)), 'str', '%%MatrixMarket matrix coordinate pattern symmetric\n  5  5  7\n    1     1\n    2     2\n    4     2\n    3     3\n    4     4\n    5     5\n    5     4\n')
# Assigning a type to the variable '_symmetric_pattern_example' (line 439)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 0), '_symmetric_pattern_example', str_6209)
# Declaration of the 'TestMMIOCoordinate' class

class TestMMIOCoordinate(object, ):

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
        TestMMIOCoordinate.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.setup_method.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.setup_method')
        TestMMIOCoordinate.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 454):
        
        # Call to mkdtemp(...): (line 454)
        # Processing the call keyword arguments (line 454)
        kwargs_6211 = {}
        # Getting the type of 'mkdtemp' (line 454)
        mkdtemp_6210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 22), 'mkdtemp', False)
        # Calling mkdtemp(args, kwargs) (line 454)
        mkdtemp_call_result_6212 = invoke(stypy.reporting.localization.Localization(__file__, 454, 22), mkdtemp_6210, *[], **kwargs_6211)
        
        # Getting the type of 'self' (line 454)
        self_6213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'self')
        # Setting the type of the member 'tmpdir' of a type (line 454)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), self_6213, 'tmpdir', mkdtemp_call_result_6212)
        
        # Assigning a Call to a Attribute (line 455):
        
        # Call to join(...): (line 455)
        # Processing the call arguments (line 455)
        # Getting the type of 'self' (line 455)
        self_6217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 31), 'self', False)
        # Obtaining the member 'tmpdir' of a type (line 455)
        tmpdir_6218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 31), self_6217, 'tmpdir')
        str_6219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 44), 'str', 'testfile.mtx')
        # Processing the call keyword arguments (line 455)
        kwargs_6220 = {}
        # Getting the type of 'os' (line 455)
        os_6214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 455)
        path_6215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 18), os_6214, 'path')
        # Obtaining the member 'join' of a type (line 455)
        join_6216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 18), path_6215, 'join')
        # Calling join(args, kwargs) (line 455)
        join_call_result_6221 = invoke(stypy.reporting.localization.Localization(__file__, 455, 18), join_6216, *[tmpdir_6218, str_6219], **kwargs_6220)
        
        # Getting the type of 'self' (line 455)
        self_6222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'self')
        # Setting the type of the member 'fn' of a type (line 455)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), self_6222, 'fn', join_call_result_6221)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 453)
        stypy_return_type_6223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6223)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_6223


    @norecursion
    def teardown_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'teardown_method'
        module_type_store = module_type_store.open_function_context('teardown_method', 457, 4, False)
        # Assigning a type to the variable 'self' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.teardown_method.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.teardown_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.teardown_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.teardown_method.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.teardown_method')
        TestMMIOCoordinate.teardown_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.teardown_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.teardown_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.teardown_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.teardown_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.teardown_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.teardown_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.teardown_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'teardown_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'teardown_method(...)' code ##################

        
        # Call to rmtree(...): (line 458)
        # Processing the call arguments (line 458)
        # Getting the type of 'self' (line 458)
        self_6226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 22), 'self', False)
        # Obtaining the member 'tmpdir' of a type (line 458)
        tmpdir_6227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 22), self_6226, 'tmpdir')
        # Processing the call keyword arguments (line 458)
        kwargs_6228 = {}
        # Getting the type of 'shutil' (line 458)
        shutil_6224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'shutil', False)
        # Obtaining the member 'rmtree' of a type (line 458)
        rmtree_6225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), shutil_6224, 'rmtree')
        # Calling rmtree(args, kwargs) (line 458)
        rmtree_call_result_6229 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), rmtree_6225, *[tmpdir_6227], **kwargs_6228)
        
        
        # ################# End of 'teardown_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'teardown_method' in the type store
        # Getting the type of 'stypy_return_type' (line 457)
        stypy_return_type_6230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6230)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'teardown_method'
        return stypy_return_type_6230


    @norecursion
    def check_read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_read'
        module_type_store = module_type_store.open_function_context('check_read', 460, 4, False)
        # Assigning a type to the variable 'self' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.check_read.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.check_read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.check_read.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.check_read.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.check_read')
        TestMMIOCoordinate.check_read.__dict__.__setitem__('stypy_param_names_list', ['example', 'a', 'info'])
        TestMMIOCoordinate.check_read.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.check_read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.check_read.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.check_read.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.check_read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.check_read.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.check_read', ['example', 'a', 'info'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_read', localization, ['example', 'a', 'info'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_read(...)' code ##################

        
        # Assigning a Call to a Name (line 461):
        
        # Call to open(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'self' (line 461)
        self_6232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 17), 'self', False)
        # Obtaining the member 'fn' of a type (line 461)
        fn_6233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 17), self_6232, 'fn')
        str_6234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 26), 'str', 'w')
        # Processing the call keyword arguments (line 461)
        kwargs_6235 = {}
        # Getting the type of 'open' (line 461)
        open_6231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'open', False)
        # Calling open(args, kwargs) (line 461)
        open_call_result_6236 = invoke(stypy.reporting.localization.Localization(__file__, 461, 12), open_6231, *[fn_6233, str_6234], **kwargs_6235)
        
        # Assigning a type to the variable 'f' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'f', open_call_result_6236)
        
        # Call to write(...): (line 462)
        # Processing the call arguments (line 462)
        # Getting the type of 'example' (line 462)
        example_6239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 16), 'example', False)
        # Processing the call keyword arguments (line 462)
        kwargs_6240 = {}
        # Getting the type of 'f' (line 462)
        f_6237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 462)
        write_6238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), f_6237, 'write')
        # Calling write(args, kwargs) (line 462)
        write_call_result_6241 = invoke(stypy.reporting.localization.Localization(__file__, 462, 8), write_6238, *[example_6239], **kwargs_6240)
        
        
        # Call to close(...): (line 463)
        # Processing the call keyword arguments (line 463)
        kwargs_6244 = {}
        # Getting the type of 'f' (line 463)
        f_6242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'f', False)
        # Obtaining the member 'close' of a type (line 463)
        close_6243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 8), f_6242, 'close')
        # Calling close(args, kwargs) (line 463)
        close_call_result_6245 = invoke(stypy.reporting.localization.Localization(__file__, 463, 8), close_6243, *[], **kwargs_6244)
        
        
        # Call to assert_equal(...): (line 464)
        # Processing the call arguments (line 464)
        
        # Call to mminfo(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'self' (line 464)
        self_6248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 28), 'self', False)
        # Obtaining the member 'fn' of a type (line 464)
        fn_6249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 28), self_6248, 'fn')
        # Processing the call keyword arguments (line 464)
        kwargs_6250 = {}
        # Getting the type of 'mminfo' (line 464)
        mminfo_6247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 21), 'mminfo', False)
        # Calling mminfo(args, kwargs) (line 464)
        mminfo_call_result_6251 = invoke(stypy.reporting.localization.Localization(__file__, 464, 21), mminfo_6247, *[fn_6249], **kwargs_6250)
        
        # Getting the type of 'info' (line 464)
        info_6252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 38), 'info', False)
        # Processing the call keyword arguments (line 464)
        kwargs_6253 = {}
        # Getting the type of 'assert_equal' (line 464)
        assert_equal_6246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 464)
        assert_equal_call_result_6254 = invoke(stypy.reporting.localization.Localization(__file__, 464, 8), assert_equal_6246, *[mminfo_call_result_6251, info_6252], **kwargs_6253)
        
        
        # Assigning a Call to a Name (line 465):
        
        # Call to todense(...): (line 465)
        # Processing the call keyword arguments (line 465)
        kwargs_6261 = {}
        
        # Call to mmread(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'self' (line 465)
        self_6256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 19), 'self', False)
        # Obtaining the member 'fn' of a type (line 465)
        fn_6257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 19), self_6256, 'fn')
        # Processing the call keyword arguments (line 465)
        kwargs_6258 = {}
        # Getting the type of 'mmread' (line 465)
        mmread_6255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'mmread', False)
        # Calling mmread(args, kwargs) (line 465)
        mmread_call_result_6259 = invoke(stypy.reporting.localization.Localization(__file__, 465, 12), mmread_6255, *[fn_6257], **kwargs_6258)
        
        # Obtaining the member 'todense' of a type (line 465)
        todense_6260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 12), mmread_call_result_6259, 'todense')
        # Calling todense(args, kwargs) (line 465)
        todense_call_result_6262 = invoke(stypy.reporting.localization.Localization(__file__, 465, 12), todense_6260, *[], **kwargs_6261)
        
        # Assigning a type to the variable 'b' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'b', todense_call_result_6262)
        
        # Call to assert_array_almost_equal(...): (line 466)
        # Processing the call arguments (line 466)
        # Getting the type of 'a' (line 466)
        a_6264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 34), 'a', False)
        # Getting the type of 'b' (line 466)
        b_6265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 37), 'b', False)
        # Processing the call keyword arguments (line 466)
        kwargs_6266 = {}
        # Getting the type of 'assert_array_almost_equal' (line 466)
        assert_array_almost_equal_6263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 466)
        assert_array_almost_equal_call_result_6267 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), assert_array_almost_equal_6263, *[a_6264, b_6265], **kwargs_6266)
        
        
        # ################# End of 'check_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_read' in the type store
        # Getting the type of 'stypy_return_type' (line 460)
        stypy_return_type_6268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6268)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_read'
        return stypy_return_type_6268


    @norecursion
    def test_read_general(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_general'
        module_type_store = module_type_store.open_function_context('test_read_general', 468, 4, False)
        # Assigning a type to the variable 'self' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_read_general.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_read_general.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_read_general.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_read_general.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_read_general')
        TestMMIOCoordinate.test_read_general.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_read_general.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_read_general.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_read_general.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_read_general.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_read_general.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_read_general.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_read_general', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_general', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_general(...)' code ##################

        
        # Assigning a List to a Name (line 469):
        
        # Obtaining an instance of the builtin type 'list' (line 469)
        list_6269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 469)
        # Adding element type (line 469)
        
        # Obtaining an instance of the builtin type 'list' (line 469)
        list_6270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 469)
        # Adding element type (line 469)
        int_6271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 13), list_6270, int_6271)
        # Adding element type (line 469)
        int_6272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 13), list_6270, int_6272)
        # Adding element type (line 469)
        int_6273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 13), list_6270, int_6273)
        # Adding element type (line 469)
        int_6274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 13), list_6270, int_6274)
        # Adding element type (line 469)
        int_6275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 13), list_6270, int_6275)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 12), list_6269, list_6270)
        # Adding element type (line 469)
        
        # Obtaining an instance of the builtin type 'list' (line 470)
        list_6276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 470)
        # Adding element type (line 470)
        int_6277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 13), list_6276, int_6277)
        # Adding element type (line 470)
        float_6278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 13), list_6276, float_6278)
        # Adding element type (line 470)
        int_6279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 13), list_6276, int_6279)
        # Adding element type (line 470)
        int_6280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 13), list_6276, int_6280)
        # Adding element type (line 470)
        int_6281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 13), list_6276, int_6281)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 12), list_6269, list_6276)
        # Adding element type (line 469)
        
        # Obtaining an instance of the builtin type 'list' (line 471)
        list_6282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 471)
        # Adding element type (line 471)
        int_6283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 13), list_6282, int_6283)
        # Adding element type (line 471)
        int_6284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 13), list_6282, int_6284)
        # Adding element type (line 471)
        float_6285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 13), list_6282, float_6285)
        # Adding element type (line 471)
        int_6286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 13), list_6282, int_6286)
        # Adding element type (line 471)
        int_6287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 13), list_6282, int_6287)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 12), list_6269, list_6282)
        # Adding element type (line 469)
        
        # Obtaining an instance of the builtin type 'list' (line 472)
        list_6288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 472)
        # Adding element type (line 472)
        int_6289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 13), list_6288, int_6289)
        # Adding element type (line 472)
        float_6290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 13), list_6288, float_6290)
        # Adding element type (line 472)
        int_6291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 13), list_6288, int_6291)
        # Adding element type (line 472)
        int_6292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 13), list_6288, int_6292)
        # Adding element type (line 472)
        float_6293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 13), list_6288, float_6293)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 12), list_6269, list_6288)
        # Adding element type (line 469)
        
        # Obtaining an instance of the builtin type 'list' (line 473)
        list_6294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 473)
        # Adding element type (line 473)
        int_6295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 13), list_6294, int_6295)
        # Adding element type (line 473)
        int_6296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 13), list_6294, int_6296)
        # Adding element type (line 473)
        int_6297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 13), list_6294, int_6297)
        # Adding element type (line 473)
        int_6298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 13), list_6294, int_6298)
        # Adding element type (line 473)
        int_6299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 13), list_6294, int_6299)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 12), list_6269, list_6294)
        
        # Assigning a type to the variable 'a' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'a', list_6269)
        
        # Call to check_read(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of '_general_example' (line 474)
        _general_example_6302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 24), '_general_example', False)
        # Getting the type of 'a' (line 474)
        a_6303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 42), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 475)
        tuple_6304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 475)
        # Adding element type (line 475)
        int_6305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 25), tuple_6304, int_6305)
        # Adding element type (line 475)
        int_6306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 25), tuple_6304, int_6306)
        # Adding element type (line 475)
        int_6307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 25), tuple_6304, int_6307)
        # Adding element type (line 475)
        str_6308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 34), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 25), tuple_6304, str_6308)
        # Adding element type (line 475)
        str_6309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 48), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 25), tuple_6304, str_6309)
        # Adding element type (line 475)
        str_6310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 56), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 25), tuple_6304, str_6310)
        
        # Processing the call keyword arguments (line 474)
        kwargs_6311 = {}
        # Getting the type of 'self' (line 474)
        self_6300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 474)
        check_read_6301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), self_6300, 'check_read')
        # Calling check_read(args, kwargs) (line 474)
        check_read_call_result_6312 = invoke(stypy.reporting.localization.Localization(__file__, 474, 8), check_read_6301, *[_general_example_6302, a_6303, tuple_6304], **kwargs_6311)
        
        
        # ################# End of 'test_read_general(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_general' in the type store
        # Getting the type of 'stypy_return_type' (line 468)
        stypy_return_type_6313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6313)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_general'
        return stypy_return_type_6313


    @norecursion
    def test_read_hermitian(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_hermitian'
        module_type_store = module_type_store.open_function_context('test_read_hermitian', 477, 4, False)
        # Assigning a type to the variable 'self' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_read_hermitian.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_read_hermitian.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_read_hermitian.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_read_hermitian.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_read_hermitian')
        TestMMIOCoordinate.test_read_hermitian.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_read_hermitian.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_read_hermitian.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_read_hermitian.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_read_hermitian.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_read_hermitian.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_read_hermitian.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_read_hermitian', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_hermitian', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_hermitian(...)' code ##################

        
        # Assigning a List to a Name (line 478):
        
        # Obtaining an instance of the builtin type 'list' (line 478)
        list_6314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 478)
        # Adding element type (line 478)
        
        # Obtaining an instance of the builtin type 'list' (line 478)
        list_6315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 478)
        # Adding element type (line 478)
        int_6316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 13), list_6315, int_6316)
        # Adding element type (line 478)
        int_6317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 13), list_6315, int_6317)
        # Adding element type (line 478)
        int_6318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 13), list_6315, int_6318)
        # Adding element type (line 478)
        int_6319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 13), list_6315, int_6319)
        # Adding element type (line 478)
        int_6320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 13), list_6315, int_6320)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_6314, list_6315)
        # Adding element type (line 478)
        
        # Obtaining an instance of the builtin type 'list' (line 479)
        list_6321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 479)
        # Adding element type (line 479)
        int_6322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 13), list_6321, int_6322)
        # Adding element type (line 479)
        float_6323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 13), list_6321, float_6323)
        # Adding element type (line 479)
        int_6324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 13), list_6321, int_6324)
        # Adding element type (line 479)
        float_6325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 26), 'float')
        complex_6326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 34), 'complex')
        # Applying the binary operator '-' (line 479)
        result_sub_6327 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 26), '-', float_6325, complex_6326)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 13), list_6321, result_sub_6327)
        # Adding element type (line 479)
        int_6328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 13), list_6321, int_6328)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_6314, list_6321)
        # Adding element type (line 478)
        
        # Obtaining an instance of the builtin type 'list' (line 480)
        list_6329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 480)
        # Adding element type (line 480)
        int_6330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 13), list_6329, int_6330)
        # Adding element type (line 480)
        int_6331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 13), list_6329, int_6331)
        # Adding element type (line 480)
        float_6332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 13), list_6329, float_6332)
        # Adding element type (line 480)
        int_6333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 13), list_6329, int_6333)
        # Adding element type (line 480)
        int_6334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 13), list_6329, int_6334)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_6314, list_6329)
        # Adding element type (line 478)
        
        # Obtaining an instance of the builtin type 'list' (line 481)
        list_6335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 481)
        # Adding element type (line 481)
        int_6336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 13), list_6335, int_6336)
        # Adding element type (line 481)
        float_6337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 17), 'float')
        complex_6338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 25), 'complex')
        # Applying the binary operator '+' (line 481)
        result_add_6339 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 17), '+', float_6337, complex_6338)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 13), list_6335, result_add_6339)
        # Adding element type (line 481)
        int_6340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 13), list_6335, int_6340)
        # Adding element type (line 481)
        int_6341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 13), list_6335, int_6341)
        # Adding element type (line 481)
        complex_6342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 42), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 13), list_6335, complex_6342)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_6314, list_6335)
        # Adding element type (line 478)
        
        # Obtaining an instance of the builtin type 'list' (line 482)
        list_6343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 482)
        # Adding element type (line 482)
        int_6344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 13), list_6343, int_6344)
        # Adding element type (line 482)
        int_6345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 13), list_6343, int_6345)
        # Adding element type (line 482)
        int_6346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 13), list_6343, int_6346)
        # Adding element type (line 482)
        complex_6347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 23), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 13), list_6343, complex_6347)
        # Adding element type (line 482)
        int_6348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 13), list_6343, int_6348)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_6314, list_6343)
        
        # Assigning a type to the variable 'a' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'a', list_6314)
        
        # Call to check_read(...): (line 483)
        # Processing the call arguments (line 483)
        # Getting the type of '_hermitian_example' (line 483)
        _hermitian_example_6351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), '_hermitian_example', False)
        # Getting the type of 'a' (line 483)
        a_6352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 44), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 484)
        tuple_6353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 484)
        # Adding element type (line 484)
        int_6354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 25), tuple_6353, int_6354)
        # Adding element type (line 484)
        int_6355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 25), tuple_6353, int_6355)
        # Adding element type (line 484)
        int_6356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 25), tuple_6353, int_6356)
        # Adding element type (line 484)
        str_6357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 34), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 25), tuple_6353, str_6357)
        # Adding element type (line 484)
        str_6358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 48), 'str', 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 25), tuple_6353, str_6358)
        # Adding element type (line 484)
        str_6359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 59), 'str', 'hermitian')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 25), tuple_6353, str_6359)
        
        # Processing the call keyword arguments (line 483)
        kwargs_6360 = {}
        # Getting the type of 'self' (line 483)
        self_6349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 483)
        check_read_6350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), self_6349, 'check_read')
        # Calling check_read(args, kwargs) (line 483)
        check_read_call_result_6361 = invoke(stypy.reporting.localization.Localization(__file__, 483, 8), check_read_6350, *[_hermitian_example_6351, a_6352, tuple_6353], **kwargs_6360)
        
        
        # ################# End of 'test_read_hermitian(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_hermitian' in the type store
        # Getting the type of 'stypy_return_type' (line 477)
        stypy_return_type_6362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6362)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_hermitian'
        return stypy_return_type_6362


    @norecursion
    def test_read_skew(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_skew'
        module_type_store = module_type_store.open_function_context('test_read_skew', 486, 4, False)
        # Assigning a type to the variable 'self' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_read_skew.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_read_skew.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_read_skew.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_read_skew.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_read_skew')
        TestMMIOCoordinate.test_read_skew.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_read_skew.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_read_skew.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_read_skew.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_read_skew.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_read_skew.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_read_skew.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_read_skew', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_skew', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_skew(...)' code ##################

        
        # Assigning a List to a Name (line 487):
        
        # Obtaining an instance of the builtin type 'list' (line 487)
        list_6363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 487)
        # Adding element type (line 487)
        
        # Obtaining an instance of the builtin type 'list' (line 487)
        list_6364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 487)
        # Adding element type (line 487)
        int_6365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 13), list_6364, int_6365)
        # Adding element type (line 487)
        int_6366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 13), list_6364, int_6366)
        # Adding element type (line 487)
        int_6367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 13), list_6364, int_6367)
        # Adding element type (line 487)
        int_6368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 13), list_6364, int_6368)
        # Adding element type (line 487)
        int_6369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 13), list_6364, int_6369)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), list_6363, list_6364)
        # Adding element type (line 487)
        
        # Obtaining an instance of the builtin type 'list' (line 488)
        list_6370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 488)
        # Adding element type (line 488)
        int_6371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 13), list_6370, int_6371)
        # Adding element type (line 488)
        float_6372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 13), list_6370, float_6372)
        # Adding element type (line 488)
        int_6373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 13), list_6370, int_6373)
        # Adding element type (line 488)
        float_6374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 13), list_6370, float_6374)
        # Adding element type (line 488)
        int_6375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 13), list_6370, int_6375)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), list_6363, list_6370)
        # Adding element type (line 487)
        
        # Obtaining an instance of the builtin type 'list' (line 489)
        list_6376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 489)
        # Adding element type (line 489)
        int_6377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 13), list_6376, int_6377)
        # Adding element type (line 489)
        int_6378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 13), list_6376, int_6378)
        # Adding element type (line 489)
        float_6379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 13), list_6376, float_6379)
        # Adding element type (line 489)
        int_6380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 13), list_6376, int_6380)
        # Adding element type (line 489)
        int_6381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 13), list_6376, int_6381)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), list_6363, list_6376)
        # Adding element type (line 487)
        
        # Obtaining an instance of the builtin type 'list' (line 490)
        list_6382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 490)
        # Adding element type (line 490)
        int_6383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 13), list_6382, int_6383)
        # Adding element type (line 490)
        float_6384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 13), list_6382, float_6384)
        # Adding element type (line 490)
        int_6385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 13), list_6382, int_6385)
        # Adding element type (line 490)
        int_6386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 13), list_6382, int_6386)
        # Adding element type (line 490)
        int_6387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 13), list_6382, int_6387)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), list_6363, list_6382)
        # Adding element type (line 487)
        
        # Obtaining an instance of the builtin type 'list' (line 491)
        list_6388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 491)
        # Adding element type (line 491)
        int_6389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 13), list_6388, int_6389)
        # Adding element type (line 491)
        int_6390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 13), list_6388, int_6390)
        # Adding element type (line 491)
        int_6391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 13), list_6388, int_6391)
        # Adding element type (line 491)
        int_6392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 13), list_6388, int_6392)
        # Adding element type (line 491)
        int_6393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 13), list_6388, int_6393)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 12), list_6363, list_6388)
        
        # Assigning a type to the variable 'a' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'a', list_6363)
        
        # Call to check_read(...): (line 492)
        # Processing the call arguments (line 492)
        # Getting the type of '_skew_example' (line 492)
        _skew_example_6396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 24), '_skew_example', False)
        # Getting the type of 'a' (line 492)
        a_6397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 39), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 493)
        tuple_6398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 493)
        # Adding element type (line 493)
        int_6399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 25), tuple_6398, int_6399)
        # Adding element type (line 493)
        int_6400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 25), tuple_6398, int_6400)
        # Adding element type (line 493)
        int_6401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 25), tuple_6398, int_6401)
        # Adding element type (line 493)
        str_6402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 34), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 25), tuple_6398, str_6402)
        # Adding element type (line 493)
        str_6403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 48), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 25), tuple_6398, str_6403)
        # Adding element type (line 493)
        str_6404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 56), 'str', 'skew-symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 25), tuple_6398, str_6404)
        
        # Processing the call keyword arguments (line 492)
        kwargs_6405 = {}
        # Getting the type of 'self' (line 492)
        self_6394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 492)
        check_read_6395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 8), self_6394, 'check_read')
        # Calling check_read(args, kwargs) (line 492)
        check_read_call_result_6406 = invoke(stypy.reporting.localization.Localization(__file__, 492, 8), check_read_6395, *[_skew_example_6396, a_6397, tuple_6398], **kwargs_6405)
        
        
        # ################# End of 'test_read_skew(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_skew' in the type store
        # Getting the type of 'stypy_return_type' (line 486)
        stypy_return_type_6407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6407)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_skew'
        return stypy_return_type_6407


    @norecursion
    def test_read_symmetric(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_symmetric'
        module_type_store = module_type_store.open_function_context('test_read_symmetric', 495, 4, False)
        # Assigning a type to the variable 'self' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_read_symmetric.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_read_symmetric.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_read_symmetric.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_read_symmetric.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_read_symmetric')
        TestMMIOCoordinate.test_read_symmetric.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_read_symmetric.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_read_symmetric.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_read_symmetric.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_read_symmetric.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_read_symmetric.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_read_symmetric.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_read_symmetric', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_symmetric', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_symmetric(...)' code ##################

        
        # Assigning a List to a Name (line 496):
        
        # Obtaining an instance of the builtin type 'list' (line 496)
        list_6408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 496)
        # Adding element type (line 496)
        
        # Obtaining an instance of the builtin type 'list' (line 496)
        list_6409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 496)
        # Adding element type (line 496)
        int_6410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 13), list_6409, int_6410)
        # Adding element type (line 496)
        int_6411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 13), list_6409, int_6411)
        # Adding element type (line 496)
        int_6412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 13), list_6409, int_6412)
        # Adding element type (line 496)
        int_6413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 13), list_6409, int_6413)
        # Adding element type (line 496)
        int_6414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 13), list_6409, int_6414)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), list_6408, list_6409)
        # Adding element type (line 496)
        
        # Obtaining an instance of the builtin type 'list' (line 497)
        list_6415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 497)
        # Adding element type (line 497)
        int_6416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 13), list_6415, int_6416)
        # Adding element type (line 497)
        float_6417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 13), list_6415, float_6417)
        # Adding element type (line 497)
        int_6418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 13), list_6415, int_6418)
        # Adding element type (line 497)
        float_6419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 13), list_6415, float_6419)
        # Adding element type (line 497)
        int_6420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 13), list_6415, int_6420)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), list_6408, list_6415)
        # Adding element type (line 496)
        
        # Obtaining an instance of the builtin type 'list' (line 498)
        list_6421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 498)
        # Adding element type (line 498)
        int_6422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 13), list_6421, int_6422)
        # Adding element type (line 498)
        int_6423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 13), list_6421, int_6423)
        # Adding element type (line 498)
        float_6424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 13), list_6421, float_6424)
        # Adding element type (line 498)
        int_6425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 13), list_6421, int_6425)
        # Adding element type (line 498)
        int_6426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 13), list_6421, int_6426)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), list_6408, list_6421)
        # Adding element type (line 496)
        
        # Obtaining an instance of the builtin type 'list' (line 499)
        list_6427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 499)
        # Adding element type (line 499)
        int_6428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 13), list_6427, int_6428)
        # Adding element type (line 499)
        float_6429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 13), list_6427, float_6429)
        # Adding element type (line 499)
        int_6430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 13), list_6427, int_6430)
        # Adding element type (line 499)
        int_6431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 13), list_6427, int_6431)
        # Adding element type (line 499)
        int_6432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 13), list_6427, int_6432)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), list_6408, list_6427)
        # Adding element type (line 496)
        
        # Obtaining an instance of the builtin type 'list' (line 500)
        list_6433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 500)
        # Adding element type (line 500)
        int_6434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 13), list_6433, int_6434)
        # Adding element type (line 500)
        int_6435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 13), list_6433, int_6435)
        # Adding element type (line 500)
        int_6436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 13), list_6433, int_6436)
        # Adding element type (line 500)
        int_6437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 13), list_6433, int_6437)
        # Adding element type (line 500)
        int_6438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 13), list_6433, int_6438)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), list_6408, list_6433)
        
        # Assigning a type to the variable 'a' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'a', list_6408)
        
        # Call to check_read(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of '_symmetric_example' (line 501)
        _symmetric_example_6441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 24), '_symmetric_example', False)
        # Getting the type of 'a' (line 501)
        a_6442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 44), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 502)
        tuple_6443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 502)
        # Adding element type (line 502)
        int_6444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 25), tuple_6443, int_6444)
        # Adding element type (line 502)
        int_6445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 25), tuple_6443, int_6445)
        # Adding element type (line 502)
        int_6446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 25), tuple_6443, int_6446)
        # Adding element type (line 502)
        str_6447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 34), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 25), tuple_6443, str_6447)
        # Adding element type (line 502)
        str_6448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 48), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 25), tuple_6443, str_6448)
        # Adding element type (line 502)
        str_6449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 56), 'str', 'symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 25), tuple_6443, str_6449)
        
        # Processing the call keyword arguments (line 501)
        kwargs_6450 = {}
        # Getting the type of 'self' (line 501)
        self_6439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 501)
        check_read_6440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), self_6439, 'check_read')
        # Calling check_read(args, kwargs) (line 501)
        check_read_call_result_6451 = invoke(stypy.reporting.localization.Localization(__file__, 501, 8), check_read_6440, *[_symmetric_example_6441, a_6442, tuple_6443], **kwargs_6450)
        
        
        # ################# End of 'test_read_symmetric(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_symmetric' in the type store
        # Getting the type of 'stypy_return_type' (line 495)
        stypy_return_type_6452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6452)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_symmetric'
        return stypy_return_type_6452


    @norecursion
    def test_read_symmetric_pattern(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_symmetric_pattern'
        module_type_store = module_type_store.open_function_context('test_read_symmetric_pattern', 504, 4, False)
        # Assigning a type to the variable 'self' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_read_symmetric_pattern.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_read_symmetric_pattern.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_read_symmetric_pattern.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_read_symmetric_pattern.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_read_symmetric_pattern')
        TestMMIOCoordinate.test_read_symmetric_pattern.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_read_symmetric_pattern.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_read_symmetric_pattern.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_read_symmetric_pattern.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_read_symmetric_pattern.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_read_symmetric_pattern.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_read_symmetric_pattern.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_read_symmetric_pattern', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_symmetric_pattern', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_symmetric_pattern(...)' code ##################

        
        # Assigning a List to a Name (line 505):
        
        # Obtaining an instance of the builtin type 'list' (line 505)
        list_6453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 505)
        # Adding element type (line 505)
        
        # Obtaining an instance of the builtin type 'list' (line 505)
        list_6454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 505)
        # Adding element type (line 505)
        int_6455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 13), list_6454, int_6455)
        # Adding element type (line 505)
        int_6456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 13), list_6454, int_6456)
        # Adding element type (line 505)
        int_6457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 13), list_6454, int_6457)
        # Adding element type (line 505)
        int_6458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 13), list_6454, int_6458)
        # Adding element type (line 505)
        int_6459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 13), list_6454, int_6459)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 12), list_6453, list_6454)
        # Adding element type (line 505)
        
        # Obtaining an instance of the builtin type 'list' (line 506)
        list_6460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 506)
        # Adding element type (line 506)
        int_6461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 13), list_6460, int_6461)
        # Adding element type (line 506)
        int_6462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 13), list_6460, int_6462)
        # Adding element type (line 506)
        int_6463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 13), list_6460, int_6463)
        # Adding element type (line 506)
        int_6464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 13), list_6460, int_6464)
        # Adding element type (line 506)
        int_6465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 13), list_6460, int_6465)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 12), list_6453, list_6460)
        # Adding element type (line 505)
        
        # Obtaining an instance of the builtin type 'list' (line 507)
        list_6466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 507)
        # Adding element type (line 507)
        int_6467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), list_6466, int_6467)
        # Adding element type (line 507)
        int_6468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), list_6466, int_6468)
        # Adding element type (line 507)
        int_6469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), list_6466, int_6469)
        # Adding element type (line 507)
        int_6470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), list_6466, int_6470)
        # Adding element type (line 507)
        int_6471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), list_6466, int_6471)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 12), list_6453, list_6466)
        # Adding element type (line 505)
        
        # Obtaining an instance of the builtin type 'list' (line 508)
        list_6472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 508)
        # Adding element type (line 508)
        int_6473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 13), list_6472, int_6473)
        # Adding element type (line 508)
        int_6474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 13), list_6472, int_6474)
        # Adding element type (line 508)
        int_6475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 13), list_6472, int_6475)
        # Adding element type (line 508)
        int_6476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 13), list_6472, int_6476)
        # Adding element type (line 508)
        int_6477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 13), list_6472, int_6477)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 12), list_6453, list_6472)
        # Adding element type (line 505)
        
        # Obtaining an instance of the builtin type 'list' (line 509)
        list_6478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 509)
        # Adding element type (line 509)
        int_6479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 13), list_6478, int_6479)
        # Adding element type (line 509)
        int_6480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 13), list_6478, int_6480)
        # Adding element type (line 509)
        int_6481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 13), list_6478, int_6481)
        # Adding element type (line 509)
        int_6482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 13), list_6478, int_6482)
        # Adding element type (line 509)
        int_6483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 13), list_6478, int_6483)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 12), list_6453, list_6478)
        
        # Assigning a type to the variable 'a' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'a', list_6453)
        
        # Call to check_read(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of '_symmetric_pattern_example' (line 510)
        _symmetric_pattern_example_6486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 24), '_symmetric_pattern_example', False)
        # Getting the type of 'a' (line 510)
        a_6487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 52), 'a', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 511)
        tuple_6488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 511)
        # Adding element type (line 511)
        int_6489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 25), tuple_6488, int_6489)
        # Adding element type (line 511)
        int_6490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 25), tuple_6488, int_6490)
        # Adding element type (line 511)
        int_6491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 25), tuple_6488, int_6491)
        # Adding element type (line 511)
        str_6492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 34), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 25), tuple_6488, str_6492)
        # Adding element type (line 511)
        str_6493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 48), 'str', 'pattern')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 25), tuple_6488, str_6493)
        # Adding element type (line 511)
        str_6494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 59), 'str', 'symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 25), tuple_6488, str_6494)
        
        # Processing the call keyword arguments (line 510)
        kwargs_6495 = {}
        # Getting the type of 'self' (line 510)
        self_6484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'self', False)
        # Obtaining the member 'check_read' of a type (line 510)
        check_read_6485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), self_6484, 'check_read')
        # Calling check_read(args, kwargs) (line 510)
        check_read_call_result_6496 = invoke(stypy.reporting.localization.Localization(__file__, 510, 8), check_read_6485, *[_symmetric_pattern_example_6486, a_6487, tuple_6488], **kwargs_6495)
        
        
        # ################# End of 'test_read_symmetric_pattern(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_symmetric_pattern' in the type store
        # Getting the type of 'stypy_return_type' (line 504)
        stypy_return_type_6497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6497)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_symmetric_pattern'
        return stypy_return_type_6497


    @norecursion
    def test_empty_write_read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_empty_write_read'
        module_type_store = module_type_store.open_function_context('test_empty_write_read', 513, 4, False)
        # Assigning a type to the variable 'self' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_empty_write_read.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_empty_write_read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_empty_write_read.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_empty_write_read.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_empty_write_read')
        TestMMIOCoordinate.test_empty_write_read.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_empty_write_read.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_empty_write_read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_empty_write_read.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_empty_write_read.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_empty_write_read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_empty_write_read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_empty_write_read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_empty_write_read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_empty_write_read(...)' code ##################

        
        # Assigning a Call to a Name (line 516):
        
        # Call to coo_matrix(...): (line 516)
        # Processing the call arguments (line 516)
        
        # Obtaining an instance of the builtin type 'tuple' (line 516)
        tuple_6501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 516)
        # Adding element type (line 516)
        int_6502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 37), tuple_6501, int_6502)
        # Adding element type (line 516)
        int_6503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 37), tuple_6501, int_6503)
        
        # Processing the call keyword arguments (line 516)
        kwargs_6504 = {}
        # Getting the type of 'scipy' (line 516)
        scipy_6498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 516)
        sparse_6499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), scipy_6498, 'sparse')
        # Obtaining the member 'coo_matrix' of a type (line 516)
        coo_matrix_6500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), sparse_6499, 'coo_matrix')
        # Calling coo_matrix(args, kwargs) (line 516)
        coo_matrix_call_result_6505 = invoke(stypy.reporting.localization.Localization(__file__, 516, 12), coo_matrix_6500, *[tuple_6501], **kwargs_6504)
        
        # Assigning a type to the variable 'b' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'b', coo_matrix_call_result_6505)
        
        # Call to mmwrite(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 'self' (line 517)
        self_6507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 16), 'self', False)
        # Obtaining the member 'fn' of a type (line 517)
        fn_6508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 16), self_6507, 'fn')
        # Getting the type of 'b' (line 517)
        b_6509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 25), 'b', False)
        # Processing the call keyword arguments (line 517)
        kwargs_6510 = {}
        # Getting the type of 'mmwrite' (line 517)
        mmwrite_6506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 517)
        mmwrite_call_result_6511 = invoke(stypy.reporting.localization.Localization(__file__, 517, 8), mmwrite_6506, *[fn_6508, b_6509], **kwargs_6510)
        
        
        # Call to assert_equal(...): (line 519)
        # Processing the call arguments (line 519)
        
        # Call to mminfo(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'self' (line 519)
        self_6514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 28), 'self', False)
        # Obtaining the member 'fn' of a type (line 519)
        fn_6515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 28), self_6514, 'fn')
        # Processing the call keyword arguments (line 519)
        kwargs_6516 = {}
        # Getting the type of 'mminfo' (line 519)
        mminfo_6513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 21), 'mminfo', False)
        # Calling mminfo(args, kwargs) (line 519)
        mminfo_call_result_6517 = invoke(stypy.reporting.localization.Localization(__file__, 519, 21), mminfo_6513, *[fn_6515], **kwargs_6516)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 520)
        tuple_6518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 520)
        # Adding element type (line 520)
        int_6519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 22), tuple_6518, int_6519)
        # Adding element type (line 520)
        int_6520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 22), tuple_6518, int_6520)
        # Adding element type (line 520)
        int_6521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 22), tuple_6518, int_6521)
        # Adding element type (line 520)
        str_6522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 33), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 22), tuple_6518, str_6522)
        # Adding element type (line 520)
        str_6523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 47), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 22), tuple_6518, str_6523)
        # Adding element type (line 520)
        str_6524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 55), 'str', 'symmetric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 22), tuple_6518, str_6524)
        
        # Processing the call keyword arguments (line 519)
        kwargs_6525 = {}
        # Getting the type of 'assert_equal' (line 519)
        assert_equal_6512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 519)
        assert_equal_call_result_6526 = invoke(stypy.reporting.localization.Localization(__file__, 519, 8), assert_equal_6512, *[mminfo_call_result_6517, tuple_6518], **kwargs_6525)
        
        
        # Assigning a Call to a Name (line 521):
        
        # Call to todense(...): (line 521)
        # Processing the call keyword arguments (line 521)
        kwargs_6529 = {}
        # Getting the type of 'b' (line 521)
        b_6527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'b', False)
        # Obtaining the member 'todense' of a type (line 521)
        todense_6528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 12), b_6527, 'todense')
        # Calling todense(args, kwargs) (line 521)
        todense_call_result_6530 = invoke(stypy.reporting.localization.Localization(__file__, 521, 12), todense_6528, *[], **kwargs_6529)
        
        # Assigning a type to the variable 'a' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'a', todense_call_result_6530)
        
        # Assigning a Call to a Name (line 522):
        
        # Call to todense(...): (line 522)
        # Processing the call keyword arguments (line 522)
        kwargs_6537 = {}
        
        # Call to mmread(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'self' (line 522)
        self_6532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 19), 'self', False)
        # Obtaining the member 'fn' of a type (line 522)
        fn_6533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 19), self_6532, 'fn')
        # Processing the call keyword arguments (line 522)
        kwargs_6534 = {}
        # Getting the type of 'mmread' (line 522)
        mmread_6531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'mmread', False)
        # Calling mmread(args, kwargs) (line 522)
        mmread_call_result_6535 = invoke(stypy.reporting.localization.Localization(__file__, 522, 12), mmread_6531, *[fn_6533], **kwargs_6534)
        
        # Obtaining the member 'todense' of a type (line 522)
        todense_6536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 12), mmread_call_result_6535, 'todense')
        # Calling todense(args, kwargs) (line 522)
        todense_call_result_6538 = invoke(stypy.reporting.localization.Localization(__file__, 522, 12), todense_6536, *[], **kwargs_6537)
        
        # Assigning a type to the variable 'b' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'b', todense_call_result_6538)
        
        # Call to assert_array_almost_equal(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 'a' (line 523)
        a_6540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 34), 'a', False)
        # Getting the type of 'b' (line 523)
        b_6541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 37), 'b', False)
        # Processing the call keyword arguments (line 523)
        kwargs_6542 = {}
        # Getting the type of 'assert_array_almost_equal' (line 523)
        assert_array_almost_equal_6539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 523)
        assert_array_almost_equal_call_result_6543 = invoke(stypy.reporting.localization.Localization(__file__, 523, 8), assert_array_almost_equal_6539, *[a_6540, b_6541], **kwargs_6542)
        
        
        # ################# End of 'test_empty_write_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_empty_write_read' in the type store
        # Getting the type of 'stypy_return_type' (line 513)
        stypy_return_type_6544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6544)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_empty_write_read'
        return stypy_return_type_6544


    @norecursion
    def test_bzip2_py3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bzip2_py3'
        module_type_store = module_type_store.open_function_context('test_bzip2_py3', 525, 4, False)
        # Assigning a type to the variable 'self' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_bzip2_py3.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_bzip2_py3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_bzip2_py3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_bzip2_py3.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_bzip2_py3')
        TestMMIOCoordinate.test_bzip2_py3.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_bzip2_py3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_bzip2_py3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_bzip2_py3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_bzip2_py3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_bzip2_py3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_bzip2_py3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_bzip2_py3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bzip2_py3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bzip2_py3(...)' code ##################

        
        
        # SSA begins for try-except statement (line 527)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 529, 12))
        
        # 'import bz2' statement (line 529)
        import bz2

        import_module(stypy.reporting.localization.Localization(__file__, 529, 12), 'bz2', bz2, module_type_store)
        
        # SSA branch for the except part of a try statement (line 527)
        # SSA branch for the except '<any exception>' branch of a try statement (line 527)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 527)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 532):
        
        # Call to array(...): (line 532)
        # Processing the call arguments (line 532)
        
        # Obtaining an instance of the builtin type 'list' (line 532)
        list_6546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 532)
        # Adding element type (line 532)
        int_6547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 18), list_6546, int_6547)
        # Adding element type (line 532)
        int_6548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 18), list_6546, int_6548)
        # Adding element type (line 532)
        int_6549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 18), list_6546, int_6549)
        # Adding element type (line 532)
        int_6550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 18), list_6546, int_6550)
        # Adding element type (line 532)
        int_6551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 18), list_6546, int_6551)
        # Adding element type (line 532)
        int_6552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 18), list_6546, int_6552)
        # Adding element type (line 532)
        int_6553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 18), list_6546, int_6553)
        # Adding element type (line 532)
        int_6554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 18), list_6546, int_6554)
        
        # Processing the call keyword arguments (line 532)
        kwargs_6555 = {}
        # Getting the type of 'array' (line 532)
        array_6545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'array', False)
        # Calling array(args, kwargs) (line 532)
        array_call_result_6556 = invoke(stypy.reporting.localization.Localization(__file__, 532, 12), array_6545, *[list_6546], **kwargs_6555)
        
        # Assigning a type to the variable 'I' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'I', array_call_result_6556)
        
        # Assigning a Call to a Name (line 533):
        
        # Call to array(...): (line 533)
        # Processing the call arguments (line 533)
        
        # Obtaining an instance of the builtin type 'list' (line 533)
        list_6558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 533)
        # Adding element type (line 533)
        int_6559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 18), list_6558, int_6559)
        # Adding element type (line 533)
        int_6560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 18), list_6558, int_6560)
        # Adding element type (line 533)
        int_6561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 18), list_6558, int_6561)
        # Adding element type (line 533)
        int_6562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 18), list_6558, int_6562)
        # Adding element type (line 533)
        int_6563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 18), list_6558, int_6563)
        # Adding element type (line 533)
        int_6564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 18), list_6558, int_6564)
        # Adding element type (line 533)
        int_6565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 18), list_6558, int_6565)
        # Adding element type (line 533)
        int_6566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 18), list_6558, int_6566)
        
        # Processing the call keyword arguments (line 533)
        kwargs_6567 = {}
        # Getting the type of 'array' (line 533)
        array_6557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'array', False)
        # Calling array(args, kwargs) (line 533)
        array_call_result_6568 = invoke(stypy.reporting.localization.Localization(__file__, 533, 12), array_6557, *[list_6558], **kwargs_6567)
        
        # Assigning a type to the variable 'J' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'J', array_call_result_6568)
        
        # Assigning a Call to a Name (line 534):
        
        # Call to array(...): (line 534)
        # Processing the call arguments (line 534)
        
        # Obtaining an instance of the builtin type 'list' (line 534)
        list_6570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 534)
        # Adding element type (line 534)
        float_6571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_6570, float_6571)
        # Adding element type (line 534)
        float_6572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_6570, float_6572)
        # Adding element type (line 534)
        float_6573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_6570, float_6573)
        # Adding element type (line 534)
        float_6574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_6570, float_6574)
        # Adding element type (line 534)
        float_6575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_6570, float_6575)
        # Adding element type (line 534)
        float_6576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_6570, float_6576)
        # Adding element type (line 534)
        float_6577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_6570, float_6577)
        # Adding element type (line 534)
        float_6578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 18), list_6570, float_6578)
        
        # Processing the call keyword arguments (line 534)
        kwargs_6579 = {}
        # Getting the type of 'array' (line 534)
        array_6569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'array', False)
        # Calling array(args, kwargs) (line 534)
        array_call_result_6580 = invoke(stypy.reporting.localization.Localization(__file__, 534, 12), array_6569, *[list_6570], **kwargs_6579)
        
        # Assigning a type to the variable 'V' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'V', array_call_result_6580)
        
        # Assigning a Call to a Name (line 536):
        
        # Call to coo_matrix(...): (line 536)
        # Processing the call arguments (line 536)
        
        # Obtaining an instance of the builtin type 'tuple' (line 536)
        tuple_6584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 536)
        # Adding element type (line 536)
        # Getting the type of 'V' (line 536)
        V_6585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 37), 'V', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 37), tuple_6584, V_6585)
        # Adding element type (line 536)
        
        # Obtaining an instance of the builtin type 'tuple' (line 536)
        tuple_6586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 536)
        # Adding element type (line 536)
        # Getting the type of 'I' (line 536)
        I_6587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 41), 'I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 41), tuple_6586, I_6587)
        # Adding element type (line 536)
        # Getting the type of 'J' (line 536)
        J_6588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 44), 'J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 41), tuple_6586, J_6588)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 37), tuple_6584, tuple_6586)
        
        # Processing the call keyword arguments (line 536)
        
        # Obtaining an instance of the builtin type 'tuple' (line 536)
        tuple_6589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 536)
        # Adding element type (line 536)
        int_6590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 56), tuple_6589, int_6590)
        # Adding element type (line 536)
        int_6591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 56), tuple_6589, int_6591)
        
        keyword_6592 = tuple_6589
        kwargs_6593 = {'shape': keyword_6592}
        # Getting the type of 'scipy' (line 536)
        scipy_6581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 536)
        sparse_6582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 12), scipy_6581, 'sparse')
        # Obtaining the member 'coo_matrix' of a type (line 536)
        coo_matrix_6583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 12), sparse_6582, 'coo_matrix')
        # Calling coo_matrix(args, kwargs) (line 536)
        coo_matrix_call_result_6594 = invoke(stypy.reporting.localization.Localization(__file__, 536, 12), coo_matrix_6583, *[tuple_6584], **kwargs_6593)
        
        # Assigning a type to the variable 'b' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'b', coo_matrix_call_result_6594)
        
        # Call to mmwrite(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'self' (line 538)
        self_6596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 16), 'self', False)
        # Obtaining the member 'fn' of a type (line 538)
        fn_6597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 16), self_6596, 'fn')
        # Getting the type of 'b' (line 538)
        b_6598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 25), 'b', False)
        # Processing the call keyword arguments (line 538)
        kwargs_6599 = {}
        # Getting the type of 'mmwrite' (line 538)
        mmwrite_6595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 538)
        mmwrite_call_result_6600 = invoke(stypy.reporting.localization.Localization(__file__, 538, 8), mmwrite_6595, *[fn_6597, b_6598], **kwargs_6599)
        
        
        # Assigning a BinOp to a Name (line 540):
        str_6601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 19), 'str', '%s.bz2')
        # Getting the type of 'self' (line 540)
        self_6602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 30), 'self')
        # Obtaining the member 'fn' of a type (line 540)
        fn_6603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 30), self_6602, 'fn')
        # Applying the binary operator '%' (line 540)
        result_mod_6604 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 19), '%', str_6601, fn_6603)
        
        # Assigning a type to the variable 'fn_bzip2' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'fn_bzip2', result_mod_6604)
        
        # Call to open(...): (line 541)
        # Processing the call arguments (line 541)
        # Getting the type of 'self' (line 541)
        self_6606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 18), 'self', False)
        # Obtaining the member 'fn' of a type (line 541)
        fn_6607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 18), self_6606, 'fn')
        str_6608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 27), 'str', 'rb')
        # Processing the call keyword arguments (line 541)
        kwargs_6609 = {}
        # Getting the type of 'open' (line 541)
        open_6605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 13), 'open', False)
        # Calling open(args, kwargs) (line 541)
        open_call_result_6610 = invoke(stypy.reporting.localization.Localization(__file__, 541, 13), open_6605, *[fn_6607, str_6608], **kwargs_6609)
        
        with_6611 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 541, 13), open_call_result_6610, 'with parameter', '__enter__', '__exit__')

        if with_6611:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 541)
            enter___6612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 13), open_call_result_6610, '__enter__')
            with_enter_6613 = invoke(stypy.reporting.localization.Localization(__file__, 541, 13), enter___6612)
            # Assigning a type to the variable 'f_in' (line 541)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 13), 'f_in', with_enter_6613)
            
            # Assigning a Call to a Name (line 542):
            
            # Call to BZ2File(...): (line 542)
            # Processing the call arguments (line 542)
            # Getting the type of 'fn_bzip2' (line 542)
            fn_bzip2_6616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 32), 'fn_bzip2', False)
            str_6617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 42), 'str', 'wb')
            # Processing the call keyword arguments (line 542)
            kwargs_6618 = {}
            # Getting the type of 'bz2' (line 542)
            bz2_6614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 20), 'bz2', False)
            # Obtaining the member 'BZ2File' of a type (line 542)
            BZ2File_6615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 20), bz2_6614, 'BZ2File')
            # Calling BZ2File(args, kwargs) (line 542)
            BZ2File_call_result_6619 = invoke(stypy.reporting.localization.Localization(__file__, 542, 20), BZ2File_6615, *[fn_bzip2_6616, str_6617], **kwargs_6618)
            
            # Assigning a type to the variable 'f_out' (line 542)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'f_out', BZ2File_call_result_6619)
            
            # Call to write(...): (line 543)
            # Processing the call arguments (line 543)
            
            # Call to read(...): (line 543)
            # Processing the call keyword arguments (line 543)
            kwargs_6624 = {}
            # Getting the type of 'f_in' (line 543)
            f_in_6622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 24), 'f_in', False)
            # Obtaining the member 'read' of a type (line 543)
            read_6623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 24), f_in_6622, 'read')
            # Calling read(args, kwargs) (line 543)
            read_call_result_6625 = invoke(stypy.reporting.localization.Localization(__file__, 543, 24), read_6623, *[], **kwargs_6624)
            
            # Processing the call keyword arguments (line 543)
            kwargs_6626 = {}
            # Getting the type of 'f_out' (line 543)
            f_out_6620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'f_out', False)
            # Obtaining the member 'write' of a type (line 543)
            write_6621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 12), f_out_6620, 'write')
            # Calling write(args, kwargs) (line 543)
            write_call_result_6627 = invoke(stypy.reporting.localization.Localization(__file__, 543, 12), write_6621, *[read_call_result_6625], **kwargs_6626)
            
            
            # Call to close(...): (line 544)
            # Processing the call keyword arguments (line 544)
            kwargs_6630 = {}
            # Getting the type of 'f_out' (line 544)
            f_out_6628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'f_out', False)
            # Obtaining the member 'close' of a type (line 544)
            close_6629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 12), f_out_6628, 'close')
            # Calling close(args, kwargs) (line 544)
            close_call_result_6631 = invoke(stypy.reporting.localization.Localization(__file__, 544, 12), close_6629, *[], **kwargs_6630)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 541)
            exit___6632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 13), open_call_result_6610, '__exit__')
            with_exit_6633 = invoke(stypy.reporting.localization.Localization(__file__, 541, 13), exit___6632, None, None, None)

        
        # Assigning a Call to a Name (line 546):
        
        # Call to todense(...): (line 546)
        # Processing the call keyword arguments (line 546)
        kwargs_6639 = {}
        
        # Call to mmread(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'fn_bzip2' (line 546)
        fn_bzip2_6635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 19), 'fn_bzip2', False)
        # Processing the call keyword arguments (line 546)
        kwargs_6636 = {}
        # Getting the type of 'mmread' (line 546)
        mmread_6634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'mmread', False)
        # Calling mmread(args, kwargs) (line 546)
        mmread_call_result_6637 = invoke(stypy.reporting.localization.Localization(__file__, 546, 12), mmread_6634, *[fn_bzip2_6635], **kwargs_6636)
        
        # Obtaining the member 'todense' of a type (line 546)
        todense_6638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 12), mmread_call_result_6637, 'todense')
        # Calling todense(args, kwargs) (line 546)
        todense_call_result_6640 = invoke(stypy.reporting.localization.Localization(__file__, 546, 12), todense_6638, *[], **kwargs_6639)
        
        # Assigning a type to the variable 'a' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'a', todense_call_result_6640)
        
        # Call to assert_array_almost_equal(...): (line 547)
        # Processing the call arguments (line 547)
        # Getting the type of 'a' (line 547)
        a_6642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 34), 'a', False)
        
        # Call to todense(...): (line 547)
        # Processing the call keyword arguments (line 547)
        kwargs_6645 = {}
        # Getting the type of 'b' (line 547)
        b_6643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 37), 'b', False)
        # Obtaining the member 'todense' of a type (line 547)
        todense_6644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 37), b_6643, 'todense')
        # Calling todense(args, kwargs) (line 547)
        todense_call_result_6646 = invoke(stypy.reporting.localization.Localization(__file__, 547, 37), todense_6644, *[], **kwargs_6645)
        
        # Processing the call keyword arguments (line 547)
        kwargs_6647 = {}
        # Getting the type of 'assert_array_almost_equal' (line 547)
        assert_array_almost_equal_6641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 547)
        assert_array_almost_equal_call_result_6648 = invoke(stypy.reporting.localization.Localization(__file__, 547, 8), assert_array_almost_equal_6641, *[a_6642, todense_call_result_6646], **kwargs_6647)
        
        
        # ################# End of 'test_bzip2_py3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bzip2_py3' in the type store
        # Getting the type of 'stypy_return_type' (line 525)
        stypy_return_type_6649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6649)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bzip2_py3'
        return stypy_return_type_6649


    @norecursion
    def test_gzip_py3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gzip_py3'
        module_type_store = module_type_store.open_function_context('test_gzip_py3', 549, 4, False)
        # Assigning a type to the variable 'self' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_gzip_py3.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_gzip_py3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_gzip_py3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_gzip_py3.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_gzip_py3')
        TestMMIOCoordinate.test_gzip_py3.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_gzip_py3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_gzip_py3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_gzip_py3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_gzip_py3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_gzip_py3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_gzip_py3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_gzip_py3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gzip_py3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gzip_py3(...)' code ##################

        
        
        # SSA begins for try-except statement (line 551)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 553, 12))
        
        # 'import gzip' statement (line 553)
        import gzip

        import_module(stypy.reporting.localization.Localization(__file__, 553, 12), 'gzip', gzip, module_type_store)
        
        # SSA branch for the except part of a try statement (line 551)
        # SSA branch for the except '<any exception>' branch of a try statement (line 551)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 551)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 556):
        
        # Call to array(...): (line 556)
        # Processing the call arguments (line 556)
        
        # Obtaining an instance of the builtin type 'list' (line 556)
        list_6651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 556)
        # Adding element type (line 556)
        int_6652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 18), list_6651, int_6652)
        # Adding element type (line 556)
        int_6653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 18), list_6651, int_6653)
        # Adding element type (line 556)
        int_6654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 18), list_6651, int_6654)
        # Adding element type (line 556)
        int_6655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 18), list_6651, int_6655)
        # Adding element type (line 556)
        int_6656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 18), list_6651, int_6656)
        # Adding element type (line 556)
        int_6657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 18), list_6651, int_6657)
        # Adding element type (line 556)
        int_6658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 18), list_6651, int_6658)
        # Adding element type (line 556)
        int_6659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 18), list_6651, int_6659)
        
        # Processing the call keyword arguments (line 556)
        kwargs_6660 = {}
        # Getting the type of 'array' (line 556)
        array_6650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'array', False)
        # Calling array(args, kwargs) (line 556)
        array_call_result_6661 = invoke(stypy.reporting.localization.Localization(__file__, 556, 12), array_6650, *[list_6651], **kwargs_6660)
        
        # Assigning a type to the variable 'I' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'I', array_call_result_6661)
        
        # Assigning a Call to a Name (line 557):
        
        # Call to array(...): (line 557)
        # Processing the call arguments (line 557)
        
        # Obtaining an instance of the builtin type 'list' (line 557)
        list_6663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 557)
        # Adding element type (line 557)
        int_6664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 18), list_6663, int_6664)
        # Adding element type (line 557)
        int_6665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 18), list_6663, int_6665)
        # Adding element type (line 557)
        int_6666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 18), list_6663, int_6666)
        # Adding element type (line 557)
        int_6667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 18), list_6663, int_6667)
        # Adding element type (line 557)
        int_6668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 18), list_6663, int_6668)
        # Adding element type (line 557)
        int_6669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 18), list_6663, int_6669)
        # Adding element type (line 557)
        int_6670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 18), list_6663, int_6670)
        # Adding element type (line 557)
        int_6671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 18), list_6663, int_6671)
        
        # Processing the call keyword arguments (line 557)
        kwargs_6672 = {}
        # Getting the type of 'array' (line 557)
        array_6662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'array', False)
        # Calling array(args, kwargs) (line 557)
        array_call_result_6673 = invoke(stypy.reporting.localization.Localization(__file__, 557, 12), array_6662, *[list_6663], **kwargs_6672)
        
        # Assigning a type to the variable 'J' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'J', array_call_result_6673)
        
        # Assigning a Call to a Name (line 558):
        
        # Call to array(...): (line 558)
        # Processing the call arguments (line 558)
        
        # Obtaining an instance of the builtin type 'list' (line 558)
        list_6675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 558)
        # Adding element type (line 558)
        float_6676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 18), list_6675, float_6676)
        # Adding element type (line 558)
        float_6677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 18), list_6675, float_6677)
        # Adding element type (line 558)
        float_6678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 18), list_6675, float_6678)
        # Adding element type (line 558)
        float_6679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 18), list_6675, float_6679)
        # Adding element type (line 558)
        float_6680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 18), list_6675, float_6680)
        # Adding element type (line 558)
        float_6681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 18), list_6675, float_6681)
        # Adding element type (line 558)
        float_6682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 18), list_6675, float_6682)
        # Adding element type (line 558)
        float_6683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 18), list_6675, float_6683)
        
        # Processing the call keyword arguments (line 558)
        kwargs_6684 = {}
        # Getting the type of 'array' (line 558)
        array_6674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'array', False)
        # Calling array(args, kwargs) (line 558)
        array_call_result_6685 = invoke(stypy.reporting.localization.Localization(__file__, 558, 12), array_6674, *[list_6675], **kwargs_6684)
        
        # Assigning a type to the variable 'V' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'V', array_call_result_6685)
        
        # Assigning a Call to a Name (line 560):
        
        # Call to coo_matrix(...): (line 560)
        # Processing the call arguments (line 560)
        
        # Obtaining an instance of the builtin type 'tuple' (line 560)
        tuple_6689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 560)
        # Adding element type (line 560)
        # Getting the type of 'V' (line 560)
        V_6690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 37), 'V', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 37), tuple_6689, V_6690)
        # Adding element type (line 560)
        
        # Obtaining an instance of the builtin type 'tuple' (line 560)
        tuple_6691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 560)
        # Adding element type (line 560)
        # Getting the type of 'I' (line 560)
        I_6692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 41), 'I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 41), tuple_6691, I_6692)
        # Adding element type (line 560)
        # Getting the type of 'J' (line 560)
        J_6693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 44), 'J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 41), tuple_6691, J_6693)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 37), tuple_6689, tuple_6691)
        
        # Processing the call keyword arguments (line 560)
        
        # Obtaining an instance of the builtin type 'tuple' (line 560)
        tuple_6694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 560)
        # Adding element type (line 560)
        int_6695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 56), tuple_6694, int_6695)
        # Adding element type (line 560)
        int_6696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 56), tuple_6694, int_6696)
        
        keyword_6697 = tuple_6694
        kwargs_6698 = {'shape': keyword_6697}
        # Getting the type of 'scipy' (line 560)
        scipy_6686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 560)
        sparse_6687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 12), scipy_6686, 'sparse')
        # Obtaining the member 'coo_matrix' of a type (line 560)
        coo_matrix_6688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 12), sparse_6687, 'coo_matrix')
        # Calling coo_matrix(args, kwargs) (line 560)
        coo_matrix_call_result_6699 = invoke(stypy.reporting.localization.Localization(__file__, 560, 12), coo_matrix_6688, *[tuple_6689], **kwargs_6698)
        
        # Assigning a type to the variable 'b' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'b', coo_matrix_call_result_6699)
        
        # Call to mmwrite(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'self' (line 562)
        self_6701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 16), 'self', False)
        # Obtaining the member 'fn' of a type (line 562)
        fn_6702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 16), self_6701, 'fn')
        # Getting the type of 'b' (line 562)
        b_6703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 25), 'b', False)
        # Processing the call keyword arguments (line 562)
        kwargs_6704 = {}
        # Getting the type of 'mmwrite' (line 562)
        mmwrite_6700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 562)
        mmwrite_call_result_6705 = invoke(stypy.reporting.localization.Localization(__file__, 562, 8), mmwrite_6700, *[fn_6702, b_6703], **kwargs_6704)
        
        
        # Assigning a BinOp to a Name (line 564):
        str_6706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 18), 'str', '%s.gz')
        # Getting the type of 'self' (line 564)
        self_6707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 28), 'self')
        # Obtaining the member 'fn' of a type (line 564)
        fn_6708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 28), self_6707, 'fn')
        # Applying the binary operator '%' (line 564)
        result_mod_6709 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 18), '%', str_6706, fn_6708)
        
        # Assigning a type to the variable 'fn_gzip' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'fn_gzip', result_mod_6709)
        
        # Call to open(...): (line 565)
        # Processing the call arguments (line 565)
        # Getting the type of 'self' (line 565)
        self_6711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 18), 'self', False)
        # Obtaining the member 'fn' of a type (line 565)
        fn_6712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 18), self_6711, 'fn')
        str_6713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 27), 'str', 'rb')
        # Processing the call keyword arguments (line 565)
        kwargs_6714 = {}
        # Getting the type of 'open' (line 565)
        open_6710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 13), 'open', False)
        # Calling open(args, kwargs) (line 565)
        open_call_result_6715 = invoke(stypy.reporting.localization.Localization(__file__, 565, 13), open_6710, *[fn_6712, str_6713], **kwargs_6714)
        
        with_6716 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 565, 13), open_call_result_6715, 'with parameter', '__enter__', '__exit__')

        if with_6716:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 565)
            enter___6717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 13), open_call_result_6715, '__enter__')
            with_enter_6718 = invoke(stypy.reporting.localization.Localization(__file__, 565, 13), enter___6717)
            # Assigning a type to the variable 'f_in' (line 565)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 13), 'f_in', with_enter_6718)
            
            # Assigning a Call to a Name (line 566):
            
            # Call to open(...): (line 566)
            # Processing the call arguments (line 566)
            # Getting the type of 'fn_gzip' (line 566)
            fn_gzip_6721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 30), 'fn_gzip', False)
            str_6722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 39), 'str', 'wb')
            # Processing the call keyword arguments (line 566)
            kwargs_6723 = {}
            # Getting the type of 'gzip' (line 566)
            gzip_6719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 20), 'gzip', False)
            # Obtaining the member 'open' of a type (line 566)
            open_6720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 20), gzip_6719, 'open')
            # Calling open(args, kwargs) (line 566)
            open_call_result_6724 = invoke(stypy.reporting.localization.Localization(__file__, 566, 20), open_6720, *[fn_gzip_6721, str_6722], **kwargs_6723)
            
            # Assigning a type to the variable 'f_out' (line 566)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'f_out', open_call_result_6724)
            
            # Call to write(...): (line 567)
            # Processing the call arguments (line 567)
            
            # Call to read(...): (line 567)
            # Processing the call keyword arguments (line 567)
            kwargs_6729 = {}
            # Getting the type of 'f_in' (line 567)
            f_in_6727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 24), 'f_in', False)
            # Obtaining the member 'read' of a type (line 567)
            read_6728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 24), f_in_6727, 'read')
            # Calling read(args, kwargs) (line 567)
            read_call_result_6730 = invoke(stypy.reporting.localization.Localization(__file__, 567, 24), read_6728, *[], **kwargs_6729)
            
            # Processing the call keyword arguments (line 567)
            kwargs_6731 = {}
            # Getting the type of 'f_out' (line 567)
            f_out_6725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'f_out', False)
            # Obtaining the member 'write' of a type (line 567)
            write_6726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 12), f_out_6725, 'write')
            # Calling write(args, kwargs) (line 567)
            write_call_result_6732 = invoke(stypy.reporting.localization.Localization(__file__, 567, 12), write_6726, *[read_call_result_6730], **kwargs_6731)
            
            
            # Call to close(...): (line 568)
            # Processing the call keyword arguments (line 568)
            kwargs_6735 = {}
            # Getting the type of 'f_out' (line 568)
            f_out_6733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'f_out', False)
            # Obtaining the member 'close' of a type (line 568)
            close_6734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 12), f_out_6733, 'close')
            # Calling close(args, kwargs) (line 568)
            close_call_result_6736 = invoke(stypy.reporting.localization.Localization(__file__, 568, 12), close_6734, *[], **kwargs_6735)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 565)
            exit___6737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 13), open_call_result_6715, '__exit__')
            with_exit_6738 = invoke(stypy.reporting.localization.Localization(__file__, 565, 13), exit___6737, None, None, None)

        
        # Assigning a Call to a Name (line 570):
        
        # Call to todense(...): (line 570)
        # Processing the call keyword arguments (line 570)
        kwargs_6744 = {}
        
        # Call to mmread(...): (line 570)
        # Processing the call arguments (line 570)
        # Getting the type of 'fn_gzip' (line 570)
        fn_gzip_6740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 19), 'fn_gzip', False)
        # Processing the call keyword arguments (line 570)
        kwargs_6741 = {}
        # Getting the type of 'mmread' (line 570)
        mmread_6739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'mmread', False)
        # Calling mmread(args, kwargs) (line 570)
        mmread_call_result_6742 = invoke(stypy.reporting.localization.Localization(__file__, 570, 12), mmread_6739, *[fn_gzip_6740], **kwargs_6741)
        
        # Obtaining the member 'todense' of a type (line 570)
        todense_6743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 12), mmread_call_result_6742, 'todense')
        # Calling todense(args, kwargs) (line 570)
        todense_call_result_6745 = invoke(stypy.reporting.localization.Localization(__file__, 570, 12), todense_6743, *[], **kwargs_6744)
        
        # Assigning a type to the variable 'a' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'a', todense_call_result_6745)
        
        # Call to assert_array_almost_equal(...): (line 571)
        # Processing the call arguments (line 571)
        # Getting the type of 'a' (line 571)
        a_6747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 34), 'a', False)
        
        # Call to todense(...): (line 571)
        # Processing the call keyword arguments (line 571)
        kwargs_6750 = {}
        # Getting the type of 'b' (line 571)
        b_6748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 37), 'b', False)
        # Obtaining the member 'todense' of a type (line 571)
        todense_6749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 37), b_6748, 'todense')
        # Calling todense(args, kwargs) (line 571)
        todense_call_result_6751 = invoke(stypy.reporting.localization.Localization(__file__, 571, 37), todense_6749, *[], **kwargs_6750)
        
        # Processing the call keyword arguments (line 571)
        kwargs_6752 = {}
        # Getting the type of 'assert_array_almost_equal' (line 571)
        assert_array_almost_equal_6746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 571)
        assert_array_almost_equal_call_result_6753 = invoke(stypy.reporting.localization.Localization(__file__, 571, 8), assert_array_almost_equal_6746, *[a_6747, todense_call_result_6751], **kwargs_6752)
        
        
        # ################# End of 'test_gzip_py3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gzip_py3' in the type store
        # Getting the type of 'stypy_return_type' (line 549)
        stypy_return_type_6754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6754)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gzip_py3'
        return stypy_return_type_6754


    @norecursion
    def test_real_write_read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_real_write_read'
        module_type_store = module_type_store.open_function_context('test_real_write_read', 573, 4, False)
        # Assigning a type to the variable 'self' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_real_write_read.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_real_write_read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_real_write_read.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_real_write_read.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_real_write_read')
        TestMMIOCoordinate.test_real_write_read.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_real_write_read.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_real_write_read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_real_write_read.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_real_write_read.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_real_write_read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_real_write_read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_real_write_read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_real_write_read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_real_write_read(...)' code ##################

        
        # Assigning a Call to a Name (line 574):
        
        # Call to array(...): (line 574)
        # Processing the call arguments (line 574)
        
        # Obtaining an instance of the builtin type 'list' (line 574)
        list_6756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 574)
        # Adding element type (line 574)
        int_6757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 18), list_6756, int_6757)
        # Adding element type (line 574)
        int_6758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 18), list_6756, int_6758)
        # Adding element type (line 574)
        int_6759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 18), list_6756, int_6759)
        # Adding element type (line 574)
        int_6760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 18), list_6756, int_6760)
        # Adding element type (line 574)
        int_6761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 18), list_6756, int_6761)
        # Adding element type (line 574)
        int_6762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 18), list_6756, int_6762)
        # Adding element type (line 574)
        int_6763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 18), list_6756, int_6763)
        # Adding element type (line 574)
        int_6764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 18), list_6756, int_6764)
        
        # Processing the call keyword arguments (line 574)
        kwargs_6765 = {}
        # Getting the type of 'array' (line 574)
        array_6755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'array', False)
        # Calling array(args, kwargs) (line 574)
        array_call_result_6766 = invoke(stypy.reporting.localization.Localization(__file__, 574, 12), array_6755, *[list_6756], **kwargs_6765)
        
        # Assigning a type to the variable 'I' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'I', array_call_result_6766)
        
        # Assigning a Call to a Name (line 575):
        
        # Call to array(...): (line 575)
        # Processing the call arguments (line 575)
        
        # Obtaining an instance of the builtin type 'list' (line 575)
        list_6768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 575)
        # Adding element type (line 575)
        int_6769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 18), list_6768, int_6769)
        # Adding element type (line 575)
        int_6770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 18), list_6768, int_6770)
        # Adding element type (line 575)
        int_6771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 18), list_6768, int_6771)
        # Adding element type (line 575)
        int_6772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 18), list_6768, int_6772)
        # Adding element type (line 575)
        int_6773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 18), list_6768, int_6773)
        # Adding element type (line 575)
        int_6774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 18), list_6768, int_6774)
        # Adding element type (line 575)
        int_6775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 18), list_6768, int_6775)
        # Adding element type (line 575)
        int_6776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 18), list_6768, int_6776)
        
        # Processing the call keyword arguments (line 575)
        kwargs_6777 = {}
        # Getting the type of 'array' (line 575)
        array_6767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'array', False)
        # Calling array(args, kwargs) (line 575)
        array_call_result_6778 = invoke(stypy.reporting.localization.Localization(__file__, 575, 12), array_6767, *[list_6768], **kwargs_6777)
        
        # Assigning a type to the variable 'J' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'J', array_call_result_6778)
        
        # Assigning a Call to a Name (line 576):
        
        # Call to array(...): (line 576)
        # Processing the call arguments (line 576)
        
        # Obtaining an instance of the builtin type 'list' (line 576)
        list_6780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 576)
        # Adding element type (line 576)
        float_6781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 18), list_6780, float_6781)
        # Adding element type (line 576)
        float_6782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 18), list_6780, float_6782)
        # Adding element type (line 576)
        float_6783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 18), list_6780, float_6783)
        # Adding element type (line 576)
        float_6784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 18), list_6780, float_6784)
        # Adding element type (line 576)
        float_6785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 18), list_6780, float_6785)
        # Adding element type (line 576)
        float_6786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 18), list_6780, float_6786)
        # Adding element type (line 576)
        float_6787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 18), list_6780, float_6787)
        # Adding element type (line 576)
        float_6788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 18), list_6780, float_6788)
        
        # Processing the call keyword arguments (line 576)
        kwargs_6789 = {}
        # Getting the type of 'array' (line 576)
        array_6779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'array', False)
        # Calling array(args, kwargs) (line 576)
        array_call_result_6790 = invoke(stypy.reporting.localization.Localization(__file__, 576, 12), array_6779, *[list_6780], **kwargs_6789)
        
        # Assigning a type to the variable 'V' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'V', array_call_result_6790)
        
        # Assigning a Call to a Name (line 578):
        
        # Call to coo_matrix(...): (line 578)
        # Processing the call arguments (line 578)
        
        # Obtaining an instance of the builtin type 'tuple' (line 578)
        tuple_6794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 578)
        # Adding element type (line 578)
        # Getting the type of 'V' (line 578)
        V_6795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 37), 'V', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 37), tuple_6794, V_6795)
        # Adding element type (line 578)
        
        # Obtaining an instance of the builtin type 'tuple' (line 578)
        tuple_6796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 578)
        # Adding element type (line 578)
        # Getting the type of 'I' (line 578)
        I_6797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 41), 'I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 41), tuple_6796, I_6797)
        # Adding element type (line 578)
        # Getting the type of 'J' (line 578)
        J_6798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 44), 'J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 41), tuple_6796, J_6798)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 37), tuple_6794, tuple_6796)
        
        # Processing the call keyword arguments (line 578)
        
        # Obtaining an instance of the builtin type 'tuple' (line 578)
        tuple_6799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 578)
        # Adding element type (line 578)
        int_6800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 56), tuple_6799, int_6800)
        # Adding element type (line 578)
        int_6801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 56), tuple_6799, int_6801)
        
        keyword_6802 = tuple_6799
        kwargs_6803 = {'shape': keyword_6802}
        # Getting the type of 'scipy' (line 578)
        scipy_6791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 578)
        sparse_6792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 12), scipy_6791, 'sparse')
        # Obtaining the member 'coo_matrix' of a type (line 578)
        coo_matrix_6793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 12), sparse_6792, 'coo_matrix')
        # Calling coo_matrix(args, kwargs) (line 578)
        coo_matrix_call_result_6804 = invoke(stypy.reporting.localization.Localization(__file__, 578, 12), coo_matrix_6793, *[tuple_6794], **kwargs_6803)
        
        # Assigning a type to the variable 'b' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'b', coo_matrix_call_result_6804)
        
        # Call to mmwrite(...): (line 580)
        # Processing the call arguments (line 580)
        # Getting the type of 'self' (line 580)
        self_6806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 16), 'self', False)
        # Obtaining the member 'fn' of a type (line 580)
        fn_6807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 16), self_6806, 'fn')
        # Getting the type of 'b' (line 580)
        b_6808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 25), 'b', False)
        # Processing the call keyword arguments (line 580)
        kwargs_6809 = {}
        # Getting the type of 'mmwrite' (line 580)
        mmwrite_6805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 580)
        mmwrite_call_result_6810 = invoke(stypy.reporting.localization.Localization(__file__, 580, 8), mmwrite_6805, *[fn_6807, b_6808], **kwargs_6809)
        
        
        # Call to assert_equal(...): (line 582)
        # Processing the call arguments (line 582)
        
        # Call to mminfo(...): (line 582)
        # Processing the call arguments (line 582)
        # Getting the type of 'self' (line 582)
        self_6813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 28), 'self', False)
        # Obtaining the member 'fn' of a type (line 582)
        fn_6814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 28), self_6813, 'fn')
        # Processing the call keyword arguments (line 582)
        kwargs_6815 = {}
        # Getting the type of 'mminfo' (line 582)
        mminfo_6812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 21), 'mminfo', False)
        # Calling mminfo(args, kwargs) (line 582)
        mminfo_call_result_6816 = invoke(stypy.reporting.localization.Localization(__file__, 582, 21), mminfo_6812, *[fn_6814], **kwargs_6815)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 583)
        tuple_6817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 583)
        # Adding element type (line 583)
        int_6818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 22), tuple_6817, int_6818)
        # Adding element type (line 583)
        int_6819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 22), tuple_6817, int_6819)
        # Adding element type (line 583)
        int_6820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 22), tuple_6817, int_6820)
        # Adding element type (line 583)
        str_6821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 31), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 22), tuple_6817, str_6821)
        # Adding element type (line 583)
        str_6822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 45), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 22), tuple_6817, str_6822)
        # Adding element type (line 583)
        str_6823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 53), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 22), tuple_6817, str_6823)
        
        # Processing the call keyword arguments (line 582)
        kwargs_6824 = {}
        # Getting the type of 'assert_equal' (line 582)
        assert_equal_6811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 582)
        assert_equal_call_result_6825 = invoke(stypy.reporting.localization.Localization(__file__, 582, 8), assert_equal_6811, *[mminfo_call_result_6816, tuple_6817], **kwargs_6824)
        
        
        # Assigning a Call to a Name (line 584):
        
        # Call to todense(...): (line 584)
        # Processing the call keyword arguments (line 584)
        kwargs_6828 = {}
        # Getting the type of 'b' (line 584)
        b_6826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'b', False)
        # Obtaining the member 'todense' of a type (line 584)
        todense_6827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 12), b_6826, 'todense')
        # Calling todense(args, kwargs) (line 584)
        todense_call_result_6829 = invoke(stypy.reporting.localization.Localization(__file__, 584, 12), todense_6827, *[], **kwargs_6828)
        
        # Assigning a type to the variable 'a' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'a', todense_call_result_6829)
        
        # Assigning a Call to a Name (line 585):
        
        # Call to todense(...): (line 585)
        # Processing the call keyword arguments (line 585)
        kwargs_6836 = {}
        
        # Call to mmread(...): (line 585)
        # Processing the call arguments (line 585)
        # Getting the type of 'self' (line 585)
        self_6831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 19), 'self', False)
        # Obtaining the member 'fn' of a type (line 585)
        fn_6832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 19), self_6831, 'fn')
        # Processing the call keyword arguments (line 585)
        kwargs_6833 = {}
        # Getting the type of 'mmread' (line 585)
        mmread_6830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'mmread', False)
        # Calling mmread(args, kwargs) (line 585)
        mmread_call_result_6834 = invoke(stypy.reporting.localization.Localization(__file__, 585, 12), mmread_6830, *[fn_6832], **kwargs_6833)
        
        # Obtaining the member 'todense' of a type (line 585)
        todense_6835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 12), mmread_call_result_6834, 'todense')
        # Calling todense(args, kwargs) (line 585)
        todense_call_result_6837 = invoke(stypy.reporting.localization.Localization(__file__, 585, 12), todense_6835, *[], **kwargs_6836)
        
        # Assigning a type to the variable 'b' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'b', todense_call_result_6837)
        
        # Call to assert_array_almost_equal(...): (line 586)
        # Processing the call arguments (line 586)
        # Getting the type of 'a' (line 586)
        a_6839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 34), 'a', False)
        # Getting the type of 'b' (line 586)
        b_6840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 37), 'b', False)
        # Processing the call keyword arguments (line 586)
        kwargs_6841 = {}
        # Getting the type of 'assert_array_almost_equal' (line 586)
        assert_array_almost_equal_6838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 586)
        assert_array_almost_equal_call_result_6842 = invoke(stypy.reporting.localization.Localization(__file__, 586, 8), assert_array_almost_equal_6838, *[a_6839, b_6840], **kwargs_6841)
        
        
        # ################# End of 'test_real_write_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_real_write_read' in the type store
        # Getting the type of 'stypy_return_type' (line 573)
        stypy_return_type_6843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6843)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_real_write_read'
        return stypy_return_type_6843


    @norecursion
    def test_complex_write_read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_complex_write_read'
        module_type_store = module_type_store.open_function_context('test_complex_write_read', 588, 4, False)
        # Assigning a type to the variable 'self' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_complex_write_read.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_complex_write_read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_complex_write_read.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_complex_write_read.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_complex_write_read')
        TestMMIOCoordinate.test_complex_write_read.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_complex_write_read.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_complex_write_read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_complex_write_read.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_complex_write_read.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_complex_write_read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_complex_write_read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_complex_write_read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_complex_write_read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_complex_write_read(...)' code ##################

        
        # Assigning a Call to a Name (line 589):
        
        # Call to array(...): (line 589)
        # Processing the call arguments (line 589)
        
        # Obtaining an instance of the builtin type 'list' (line 589)
        list_6845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 589)
        # Adding element type (line 589)
        int_6846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 18), list_6845, int_6846)
        # Adding element type (line 589)
        int_6847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 18), list_6845, int_6847)
        # Adding element type (line 589)
        int_6848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 18), list_6845, int_6848)
        # Adding element type (line 589)
        int_6849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 18), list_6845, int_6849)
        # Adding element type (line 589)
        int_6850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 18), list_6845, int_6850)
        # Adding element type (line 589)
        int_6851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 18), list_6845, int_6851)
        # Adding element type (line 589)
        int_6852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 18), list_6845, int_6852)
        # Adding element type (line 589)
        int_6853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 18), list_6845, int_6853)
        
        # Processing the call keyword arguments (line 589)
        kwargs_6854 = {}
        # Getting the type of 'array' (line 589)
        array_6844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'array', False)
        # Calling array(args, kwargs) (line 589)
        array_call_result_6855 = invoke(stypy.reporting.localization.Localization(__file__, 589, 12), array_6844, *[list_6845], **kwargs_6854)
        
        # Assigning a type to the variable 'I' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'I', array_call_result_6855)
        
        # Assigning a Call to a Name (line 590):
        
        # Call to array(...): (line 590)
        # Processing the call arguments (line 590)
        
        # Obtaining an instance of the builtin type 'list' (line 590)
        list_6857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 590)
        # Adding element type (line 590)
        int_6858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 18), list_6857, int_6858)
        # Adding element type (line 590)
        int_6859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 18), list_6857, int_6859)
        # Adding element type (line 590)
        int_6860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 18), list_6857, int_6860)
        # Adding element type (line 590)
        int_6861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 18), list_6857, int_6861)
        # Adding element type (line 590)
        int_6862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 18), list_6857, int_6862)
        # Adding element type (line 590)
        int_6863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 18), list_6857, int_6863)
        # Adding element type (line 590)
        int_6864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 18), list_6857, int_6864)
        # Adding element type (line 590)
        int_6865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 18), list_6857, int_6865)
        
        # Processing the call keyword arguments (line 590)
        kwargs_6866 = {}
        # Getting the type of 'array' (line 590)
        array_6856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'array', False)
        # Calling array(args, kwargs) (line 590)
        array_call_result_6867 = invoke(stypy.reporting.localization.Localization(__file__, 590, 12), array_6856, *[list_6857], **kwargs_6866)
        
        # Assigning a type to the variable 'J' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'J', array_call_result_6867)
        
        # Assigning a Call to a Name (line 591):
        
        # Call to array(...): (line 591)
        # Processing the call arguments (line 591)
        
        # Obtaining an instance of the builtin type 'list' (line 591)
        list_6869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 591)
        # Adding element type (line 591)
        float_6870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 19), 'float')
        complex_6871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 25), 'complex')
        # Applying the binary operator '+' (line 591)
        result_add_6872 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 19), '+', float_6870, complex_6871)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 18), list_6869, result_add_6872)
        # Adding element type (line 591)
        float_6873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 29), 'float')
        complex_6874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 35), 'complex')
        # Applying the binary operator '+' (line 591)
        result_add_6875 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 29), '+', float_6873, complex_6874)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 18), list_6869, result_add_6875)
        # Adding element type (line 591)
        float_6876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 39), 'float')
        complex_6877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 47), 'complex')
        # Applying the binary operator '+' (line 591)
        result_add_6878 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 39), '+', float_6876, complex_6877)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 18), list_6869, result_add_6878)
        # Adding element type (line 591)
        float_6879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 53), 'float')
        complex_6880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 61), 'complex')
        # Applying the binary operator '+' (line 591)
        result_add_6881 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 53), '+', float_6879, complex_6880)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 18), list_6869, result_add_6881)
        # Adding element type (line 591)
        float_6882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 19), 'float')
        complex_6883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 27), 'complex')
        # Applying the binary operator '+' (line 592)
        result_add_6884 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 19), '+', float_6882, complex_6883)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 18), list_6869, result_add_6884)
        # Adding element type (line 591)
        float_6885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 31), 'float')
        complex_6886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 40), 'complex')
        # Applying the binary operator '+' (line 592)
        result_add_6887 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 31), '+', float_6885, complex_6886)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 18), list_6869, result_add_6887)
        # Adding element type (line 591)
        float_6888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 44), 'float')
        complex_6889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 52), 'complex')
        # Applying the binary operator '+' (line 592)
        result_add_6890 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 44), '+', float_6888, complex_6889)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 18), list_6869, result_add_6890)
        # Adding element type (line 591)
        float_6891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 58), 'float')
        complex_6892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 66), 'complex')
        # Applying the binary operator '+' (line 592)
        result_add_6893 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 58), '+', float_6891, complex_6892)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 18), list_6869, result_add_6893)
        
        # Processing the call keyword arguments (line 591)
        kwargs_6894 = {}
        # Getting the type of 'array' (line 591)
        array_6868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 12), 'array', False)
        # Calling array(args, kwargs) (line 591)
        array_call_result_6895 = invoke(stypy.reporting.localization.Localization(__file__, 591, 12), array_6868, *[list_6869], **kwargs_6894)
        
        # Assigning a type to the variable 'V' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'V', array_call_result_6895)
        
        # Assigning a Call to a Name (line 594):
        
        # Call to coo_matrix(...): (line 594)
        # Processing the call arguments (line 594)
        
        # Obtaining an instance of the builtin type 'tuple' (line 594)
        tuple_6899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 594)
        # Adding element type (line 594)
        # Getting the type of 'V' (line 594)
        V_6900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 37), 'V', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 37), tuple_6899, V_6900)
        # Adding element type (line 594)
        
        # Obtaining an instance of the builtin type 'tuple' (line 594)
        tuple_6901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 594)
        # Adding element type (line 594)
        # Getting the type of 'I' (line 594)
        I_6902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 41), 'I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 41), tuple_6901, I_6902)
        # Adding element type (line 594)
        # Getting the type of 'J' (line 594)
        J_6903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 44), 'J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 41), tuple_6901, J_6903)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 37), tuple_6899, tuple_6901)
        
        # Processing the call keyword arguments (line 594)
        
        # Obtaining an instance of the builtin type 'tuple' (line 594)
        tuple_6904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 594)
        # Adding element type (line 594)
        int_6905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 56), tuple_6904, int_6905)
        # Adding element type (line 594)
        int_6906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 56), tuple_6904, int_6906)
        
        keyword_6907 = tuple_6904
        kwargs_6908 = {'shape': keyword_6907}
        # Getting the type of 'scipy' (line 594)
        scipy_6896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 594)
        sparse_6897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 12), scipy_6896, 'sparse')
        # Obtaining the member 'coo_matrix' of a type (line 594)
        coo_matrix_6898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 12), sparse_6897, 'coo_matrix')
        # Calling coo_matrix(args, kwargs) (line 594)
        coo_matrix_call_result_6909 = invoke(stypy.reporting.localization.Localization(__file__, 594, 12), coo_matrix_6898, *[tuple_6899], **kwargs_6908)
        
        # Assigning a type to the variable 'b' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'b', coo_matrix_call_result_6909)
        
        # Call to mmwrite(...): (line 596)
        # Processing the call arguments (line 596)
        # Getting the type of 'self' (line 596)
        self_6911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'self', False)
        # Obtaining the member 'fn' of a type (line 596)
        fn_6912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 16), self_6911, 'fn')
        # Getting the type of 'b' (line 596)
        b_6913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 25), 'b', False)
        # Processing the call keyword arguments (line 596)
        kwargs_6914 = {}
        # Getting the type of 'mmwrite' (line 596)
        mmwrite_6910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 596)
        mmwrite_call_result_6915 = invoke(stypy.reporting.localization.Localization(__file__, 596, 8), mmwrite_6910, *[fn_6912, b_6913], **kwargs_6914)
        
        
        # Call to assert_equal(...): (line 598)
        # Processing the call arguments (line 598)
        
        # Call to mminfo(...): (line 598)
        # Processing the call arguments (line 598)
        # Getting the type of 'self' (line 598)
        self_6918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 28), 'self', False)
        # Obtaining the member 'fn' of a type (line 598)
        fn_6919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 28), self_6918, 'fn')
        # Processing the call keyword arguments (line 598)
        kwargs_6920 = {}
        # Getting the type of 'mminfo' (line 598)
        mminfo_6917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 21), 'mminfo', False)
        # Calling mminfo(args, kwargs) (line 598)
        mminfo_call_result_6921 = invoke(stypy.reporting.localization.Localization(__file__, 598, 21), mminfo_6917, *[fn_6919], **kwargs_6920)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 599)
        tuple_6922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 599)
        # Adding element type (line 599)
        int_6923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 22), tuple_6922, int_6923)
        # Adding element type (line 599)
        int_6924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 22), tuple_6922, int_6924)
        # Adding element type (line 599)
        int_6925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 22), tuple_6922, int_6925)
        # Adding element type (line 599)
        str_6926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 31), 'str', 'coordinate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 22), tuple_6922, str_6926)
        # Adding element type (line 599)
        str_6927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 45), 'str', 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 22), tuple_6922, str_6927)
        # Adding element type (line 599)
        str_6928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 56), 'str', 'general')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 22), tuple_6922, str_6928)
        
        # Processing the call keyword arguments (line 598)
        kwargs_6929 = {}
        # Getting the type of 'assert_equal' (line 598)
        assert_equal_6916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 598)
        assert_equal_call_result_6930 = invoke(stypy.reporting.localization.Localization(__file__, 598, 8), assert_equal_6916, *[mminfo_call_result_6921, tuple_6922], **kwargs_6929)
        
        
        # Assigning a Call to a Name (line 600):
        
        # Call to todense(...): (line 600)
        # Processing the call keyword arguments (line 600)
        kwargs_6933 = {}
        # Getting the type of 'b' (line 600)
        b_6931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'b', False)
        # Obtaining the member 'todense' of a type (line 600)
        todense_6932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 12), b_6931, 'todense')
        # Calling todense(args, kwargs) (line 600)
        todense_call_result_6934 = invoke(stypy.reporting.localization.Localization(__file__, 600, 12), todense_6932, *[], **kwargs_6933)
        
        # Assigning a type to the variable 'a' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'a', todense_call_result_6934)
        
        # Assigning a Call to a Name (line 601):
        
        # Call to todense(...): (line 601)
        # Processing the call keyword arguments (line 601)
        kwargs_6941 = {}
        
        # Call to mmread(...): (line 601)
        # Processing the call arguments (line 601)
        # Getting the type of 'self' (line 601)
        self_6936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 19), 'self', False)
        # Obtaining the member 'fn' of a type (line 601)
        fn_6937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 19), self_6936, 'fn')
        # Processing the call keyword arguments (line 601)
        kwargs_6938 = {}
        # Getting the type of 'mmread' (line 601)
        mmread_6935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'mmread', False)
        # Calling mmread(args, kwargs) (line 601)
        mmread_call_result_6939 = invoke(stypy.reporting.localization.Localization(__file__, 601, 12), mmread_6935, *[fn_6937], **kwargs_6938)
        
        # Obtaining the member 'todense' of a type (line 601)
        todense_6940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 12), mmread_call_result_6939, 'todense')
        # Calling todense(args, kwargs) (line 601)
        todense_call_result_6942 = invoke(stypy.reporting.localization.Localization(__file__, 601, 12), todense_6940, *[], **kwargs_6941)
        
        # Assigning a type to the variable 'b' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'b', todense_call_result_6942)
        
        # Call to assert_array_almost_equal(...): (line 602)
        # Processing the call arguments (line 602)
        # Getting the type of 'a' (line 602)
        a_6944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 34), 'a', False)
        # Getting the type of 'b' (line 602)
        b_6945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 37), 'b', False)
        # Processing the call keyword arguments (line 602)
        kwargs_6946 = {}
        # Getting the type of 'assert_array_almost_equal' (line 602)
        assert_array_almost_equal_6943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 602)
        assert_array_almost_equal_call_result_6947 = invoke(stypy.reporting.localization.Localization(__file__, 602, 8), assert_array_almost_equal_6943, *[a_6944, b_6945], **kwargs_6946)
        
        
        # ################# End of 'test_complex_write_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_complex_write_read' in the type store
        # Getting the type of 'stypy_return_type' (line 588)
        stypy_return_type_6948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6948)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_complex_write_read'
        return stypy_return_type_6948


    @norecursion
    def test_sparse_formats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sparse_formats'
        module_type_store = module_type_store.open_function_context('test_sparse_formats', 604, 4, False)
        # Assigning a type to the variable 'self' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_sparse_formats.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_sparse_formats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_sparse_formats.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_sparse_formats.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_sparse_formats')
        TestMMIOCoordinate.test_sparse_formats.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_sparse_formats.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_sparse_formats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_sparse_formats.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_sparse_formats.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_sparse_formats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_sparse_formats.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_sparse_formats', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sparse_formats', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sparse_formats(...)' code ##################

        
        # Assigning a List to a Name (line 605):
        
        # Obtaining an instance of the builtin type 'list' (line 605)
        list_6949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 605)
        
        # Assigning a type to the variable 'mats' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'mats', list_6949)
        
        # Assigning a Call to a Name (line 607):
        
        # Call to array(...): (line 607)
        # Processing the call arguments (line 607)
        
        # Obtaining an instance of the builtin type 'list' (line 607)
        list_6951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 607)
        # Adding element type (line 607)
        int_6952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 18), list_6951, int_6952)
        # Adding element type (line 607)
        int_6953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 18), list_6951, int_6953)
        # Adding element type (line 607)
        int_6954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 18), list_6951, int_6954)
        # Adding element type (line 607)
        int_6955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 18), list_6951, int_6955)
        # Adding element type (line 607)
        int_6956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 18), list_6951, int_6956)
        # Adding element type (line 607)
        int_6957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 18), list_6951, int_6957)
        # Adding element type (line 607)
        int_6958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 18), list_6951, int_6958)
        # Adding element type (line 607)
        int_6959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 18), list_6951, int_6959)
        
        # Processing the call keyword arguments (line 607)
        kwargs_6960 = {}
        # Getting the type of 'array' (line 607)
        array_6950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'array', False)
        # Calling array(args, kwargs) (line 607)
        array_call_result_6961 = invoke(stypy.reporting.localization.Localization(__file__, 607, 12), array_6950, *[list_6951], **kwargs_6960)
        
        # Assigning a type to the variable 'I' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'I', array_call_result_6961)
        
        # Assigning a Call to a Name (line 608):
        
        # Call to array(...): (line 608)
        # Processing the call arguments (line 608)
        
        # Obtaining an instance of the builtin type 'list' (line 608)
        list_6963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 608)
        # Adding element type (line 608)
        int_6964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 18), list_6963, int_6964)
        # Adding element type (line 608)
        int_6965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 18), list_6963, int_6965)
        # Adding element type (line 608)
        int_6966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 18), list_6963, int_6966)
        # Adding element type (line 608)
        int_6967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 18), list_6963, int_6967)
        # Adding element type (line 608)
        int_6968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 18), list_6963, int_6968)
        # Adding element type (line 608)
        int_6969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 18), list_6963, int_6969)
        # Adding element type (line 608)
        int_6970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 18), list_6963, int_6970)
        # Adding element type (line 608)
        int_6971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 18), list_6963, int_6971)
        
        # Processing the call keyword arguments (line 608)
        kwargs_6972 = {}
        # Getting the type of 'array' (line 608)
        array_6962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'array', False)
        # Calling array(args, kwargs) (line 608)
        array_call_result_6973 = invoke(stypy.reporting.localization.Localization(__file__, 608, 12), array_6962, *[list_6963], **kwargs_6972)
        
        # Assigning a type to the variable 'J' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'J', array_call_result_6973)
        
        # Assigning a Call to a Name (line 610):
        
        # Call to array(...): (line 610)
        # Processing the call arguments (line 610)
        
        # Obtaining an instance of the builtin type 'list' (line 610)
        list_6975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 610)
        # Adding element type (line 610)
        float_6976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 18), list_6975, float_6976)
        # Adding element type (line 610)
        float_6977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 18), list_6975, float_6977)
        # Adding element type (line 610)
        float_6978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 18), list_6975, float_6978)
        # Adding element type (line 610)
        float_6979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 18), list_6975, float_6979)
        # Adding element type (line 610)
        float_6980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 18), list_6975, float_6980)
        # Adding element type (line 610)
        float_6981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 18), list_6975, float_6981)
        # Adding element type (line 610)
        float_6982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 18), list_6975, float_6982)
        # Adding element type (line 610)
        float_6983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 18), list_6975, float_6983)
        
        # Processing the call keyword arguments (line 610)
        kwargs_6984 = {}
        # Getting the type of 'array' (line 610)
        array_6974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'array', False)
        # Calling array(args, kwargs) (line 610)
        array_call_result_6985 = invoke(stypy.reporting.localization.Localization(__file__, 610, 12), array_6974, *[list_6975], **kwargs_6984)
        
        # Assigning a type to the variable 'V' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'V', array_call_result_6985)
        
        # Call to append(...): (line 611)
        # Processing the call arguments (line 611)
        
        # Call to coo_matrix(...): (line 611)
        # Processing the call arguments (line 611)
        
        # Obtaining an instance of the builtin type 'tuple' (line 611)
        tuple_6991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 611)
        # Adding element type (line 611)
        # Getting the type of 'V' (line 611)
        V_6992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 45), 'V', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 45), tuple_6991, V_6992)
        # Adding element type (line 611)
        
        # Obtaining an instance of the builtin type 'tuple' (line 611)
        tuple_6993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 611)
        # Adding element type (line 611)
        # Getting the type of 'I' (line 611)
        I_6994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 49), 'I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 49), tuple_6993, I_6994)
        # Adding element type (line 611)
        # Getting the type of 'J' (line 611)
        J_6995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 52), 'J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 49), tuple_6993, J_6995)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 45), tuple_6991, tuple_6993)
        
        # Processing the call keyword arguments (line 611)
        
        # Obtaining an instance of the builtin type 'tuple' (line 611)
        tuple_6996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 64), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 611)
        # Adding element type (line 611)
        int_6997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 64), tuple_6996, int_6997)
        # Adding element type (line 611)
        int_6998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 64), tuple_6996, int_6998)
        
        keyword_6999 = tuple_6996
        kwargs_7000 = {'shape': keyword_6999}
        # Getting the type of 'scipy' (line 611)
        scipy_6988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 20), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 611)
        sparse_6989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 20), scipy_6988, 'sparse')
        # Obtaining the member 'coo_matrix' of a type (line 611)
        coo_matrix_6990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 20), sparse_6989, 'coo_matrix')
        # Calling coo_matrix(args, kwargs) (line 611)
        coo_matrix_call_result_7001 = invoke(stypy.reporting.localization.Localization(__file__, 611, 20), coo_matrix_6990, *[tuple_6991], **kwargs_7000)
        
        # Processing the call keyword arguments (line 611)
        kwargs_7002 = {}
        # Getting the type of 'mats' (line 611)
        mats_6986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 611)
        append_6987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 8), mats_6986, 'append')
        # Calling append(args, kwargs) (line 611)
        append_call_result_7003 = invoke(stypy.reporting.localization.Localization(__file__, 611, 8), append_6987, *[coo_matrix_call_result_7001], **kwargs_7002)
        
        
        # Assigning a Call to a Name (line 613):
        
        # Call to array(...): (line 613)
        # Processing the call arguments (line 613)
        
        # Obtaining an instance of the builtin type 'list' (line 613)
        list_7005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 613)
        # Adding element type (line 613)
        float_7006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 19), 'float')
        complex_7007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 25), 'complex')
        # Applying the binary operator '+' (line 613)
        result_add_7008 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 19), '+', float_7006, complex_7007)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 18), list_7005, result_add_7008)
        # Adding element type (line 613)
        float_7009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 29), 'float')
        complex_7010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 35), 'complex')
        # Applying the binary operator '+' (line 613)
        result_add_7011 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 29), '+', float_7009, complex_7010)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 18), list_7005, result_add_7011)
        # Adding element type (line 613)
        float_7012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 39), 'float')
        complex_7013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 47), 'complex')
        # Applying the binary operator '+' (line 613)
        result_add_7014 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 39), '+', float_7012, complex_7013)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 18), list_7005, result_add_7014)
        # Adding element type (line 613)
        float_7015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 53), 'float')
        complex_7016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 61), 'complex')
        # Applying the binary operator '+' (line 613)
        result_add_7017 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 53), '+', float_7015, complex_7016)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 18), list_7005, result_add_7017)
        # Adding element type (line 613)
        float_7018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 19), 'float')
        complex_7019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 27), 'complex')
        # Applying the binary operator '+' (line 614)
        result_add_7020 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 19), '+', float_7018, complex_7019)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 18), list_7005, result_add_7020)
        # Adding element type (line 613)
        float_7021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 31), 'float')
        complex_7022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 40), 'complex')
        # Applying the binary operator '+' (line 614)
        result_add_7023 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 31), '+', float_7021, complex_7022)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 18), list_7005, result_add_7023)
        # Adding element type (line 613)
        float_7024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 44), 'float')
        complex_7025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 52), 'complex')
        # Applying the binary operator '+' (line 614)
        result_add_7026 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 44), '+', float_7024, complex_7025)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 18), list_7005, result_add_7026)
        # Adding element type (line 613)
        float_7027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 58), 'float')
        complex_7028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 66), 'complex')
        # Applying the binary operator '+' (line 614)
        result_add_7029 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 58), '+', float_7027, complex_7028)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 18), list_7005, result_add_7029)
        
        # Processing the call keyword arguments (line 613)
        kwargs_7030 = {}
        # Getting the type of 'array' (line 613)
        array_7004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'array', False)
        # Calling array(args, kwargs) (line 613)
        array_call_result_7031 = invoke(stypy.reporting.localization.Localization(__file__, 613, 12), array_7004, *[list_7005], **kwargs_7030)
        
        # Assigning a type to the variable 'V' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'V', array_call_result_7031)
        
        # Call to append(...): (line 615)
        # Processing the call arguments (line 615)
        
        # Call to coo_matrix(...): (line 615)
        # Processing the call arguments (line 615)
        
        # Obtaining an instance of the builtin type 'tuple' (line 615)
        tuple_7037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 615)
        # Adding element type (line 615)
        # Getting the type of 'V' (line 615)
        V_7038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 45), 'V', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 45), tuple_7037, V_7038)
        # Adding element type (line 615)
        
        # Obtaining an instance of the builtin type 'tuple' (line 615)
        tuple_7039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 615)
        # Adding element type (line 615)
        # Getting the type of 'I' (line 615)
        I_7040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 49), 'I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 49), tuple_7039, I_7040)
        # Adding element type (line 615)
        # Getting the type of 'J' (line 615)
        J_7041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 52), 'J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 49), tuple_7039, J_7041)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 45), tuple_7037, tuple_7039)
        
        # Processing the call keyword arguments (line 615)
        
        # Obtaining an instance of the builtin type 'tuple' (line 615)
        tuple_7042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 64), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 615)
        # Adding element type (line 615)
        int_7043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 64), tuple_7042, int_7043)
        # Adding element type (line 615)
        int_7044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 64), tuple_7042, int_7044)
        
        keyword_7045 = tuple_7042
        kwargs_7046 = {'shape': keyword_7045}
        # Getting the type of 'scipy' (line 615)
        scipy_7034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 20), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 615)
        sparse_7035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 20), scipy_7034, 'sparse')
        # Obtaining the member 'coo_matrix' of a type (line 615)
        coo_matrix_7036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 20), sparse_7035, 'coo_matrix')
        # Calling coo_matrix(args, kwargs) (line 615)
        coo_matrix_call_result_7047 = invoke(stypy.reporting.localization.Localization(__file__, 615, 20), coo_matrix_7036, *[tuple_7037], **kwargs_7046)
        
        # Processing the call keyword arguments (line 615)
        kwargs_7048 = {}
        # Getting the type of 'mats' (line 615)
        mats_7032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'mats', False)
        # Obtaining the member 'append' of a type (line 615)
        append_7033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 8), mats_7032, 'append')
        # Calling append(args, kwargs) (line 615)
        append_call_result_7049 = invoke(stypy.reporting.localization.Localization(__file__, 615, 8), append_7033, *[coo_matrix_call_result_7047], **kwargs_7048)
        
        
        # Getting the type of 'mats' (line 617)
        mats_7050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 19), 'mats')
        # Testing the type of a for loop iterable (line 617)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 617, 8), mats_7050)
        # Getting the type of the for loop variable (line 617)
        for_loop_var_7051 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 617, 8), mats_7050)
        # Assigning a type to the variable 'mat' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'mat', for_loop_var_7051)
        # SSA begins for a for statement (line 617)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 618):
        
        # Call to todense(...): (line 618)
        # Processing the call keyword arguments (line 618)
        kwargs_7054 = {}
        # Getting the type of 'mat' (line 618)
        mat_7052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 23), 'mat', False)
        # Obtaining the member 'todense' of a type (line 618)
        todense_7053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 23), mat_7052, 'todense')
        # Calling todense(args, kwargs) (line 618)
        todense_call_result_7055 = invoke(stypy.reporting.localization.Localization(__file__, 618, 23), todense_7053, *[], **kwargs_7054)
        
        # Assigning a type to the variable 'expected' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'expected', todense_call_result_7055)
        
        
        # Obtaining an instance of the builtin type 'list' (line 619)
        list_7056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 619)
        # Adding element type (line 619)
        str_7057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 24), 'str', 'csr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 23), list_7056, str_7057)
        # Adding element type (line 619)
        str_7058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 31), 'str', 'csc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 23), list_7056, str_7058)
        # Adding element type (line 619)
        str_7059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 38), 'str', 'coo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 23), list_7056, str_7059)
        
        # Testing the type of a for loop iterable (line 619)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 619, 12), list_7056)
        # Getting the type of the for loop variable (line 619)
        for_loop_var_7060 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 619, 12), list_7056)
        # Assigning a type to the variable 'fmt' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'fmt', for_loop_var_7060)
        # SSA begins for a for statement (line 619)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 620):
        
        # Call to mktemp(...): (line 620)
        # Processing the call keyword arguments (line 620)
        # Getting the type of 'self' (line 620)
        self_7062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 32), 'self', False)
        # Obtaining the member 'tmpdir' of a type (line 620)
        tmpdir_7063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 32), self_7062, 'tmpdir')
        keyword_7064 = tmpdir_7063
        kwargs_7065 = {'dir': keyword_7064}
        # Getting the type of 'mktemp' (line 620)
        mktemp_7061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 21), 'mktemp', False)
        # Calling mktemp(args, kwargs) (line 620)
        mktemp_call_result_7066 = invoke(stypy.reporting.localization.Localization(__file__, 620, 21), mktemp_7061, *[], **kwargs_7065)
        
        # Assigning a type to the variable 'fn' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 16), 'fn', mktemp_call_result_7066)
        
        # Call to mmwrite(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'fn' (line 621)
        fn_7068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 24), 'fn', False)
        
        # Call to asformat(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'fmt' (line 621)
        fmt_7071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 41), 'fmt', False)
        # Processing the call keyword arguments (line 621)
        kwargs_7072 = {}
        # Getting the type of 'mat' (line 621)
        mat_7069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 28), 'mat', False)
        # Obtaining the member 'asformat' of a type (line 621)
        asformat_7070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 28), mat_7069, 'asformat')
        # Calling asformat(args, kwargs) (line 621)
        asformat_call_result_7073 = invoke(stypy.reporting.localization.Localization(__file__, 621, 28), asformat_7070, *[fmt_7071], **kwargs_7072)
        
        # Processing the call keyword arguments (line 621)
        kwargs_7074 = {}
        # Getting the type of 'mmwrite' (line 621)
        mmwrite_7067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 16), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 621)
        mmwrite_call_result_7075 = invoke(stypy.reporting.localization.Localization(__file__, 621, 16), mmwrite_7067, *[fn_7068, asformat_call_result_7073], **kwargs_7074)
        
        
        # Assigning a Call to a Name (line 623):
        
        # Call to todense(...): (line 623)
        # Processing the call keyword arguments (line 623)
        kwargs_7081 = {}
        
        # Call to mmread(...): (line 623)
        # Processing the call arguments (line 623)
        # Getting the type of 'fn' (line 623)
        fn_7077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 32), 'fn', False)
        # Processing the call keyword arguments (line 623)
        kwargs_7078 = {}
        # Getting the type of 'mmread' (line 623)
        mmread_7076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 25), 'mmread', False)
        # Calling mmread(args, kwargs) (line 623)
        mmread_call_result_7079 = invoke(stypy.reporting.localization.Localization(__file__, 623, 25), mmread_7076, *[fn_7077], **kwargs_7078)
        
        # Obtaining the member 'todense' of a type (line 623)
        todense_7080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 25), mmread_call_result_7079, 'todense')
        # Calling todense(args, kwargs) (line 623)
        todense_call_result_7082 = invoke(stypy.reporting.localization.Localization(__file__, 623, 25), todense_7080, *[], **kwargs_7081)
        
        # Assigning a type to the variable 'result' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'result', todense_call_result_7082)
        
        # Call to assert_array_almost_equal(...): (line 624)
        # Processing the call arguments (line 624)
        # Getting the type of 'result' (line 624)
        result_7084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 42), 'result', False)
        # Getting the type of 'expected' (line 624)
        expected_7085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 50), 'expected', False)
        # Processing the call keyword arguments (line 624)
        kwargs_7086 = {}
        # Getting the type of 'assert_array_almost_equal' (line 624)
        assert_array_almost_equal_7083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 624)
        assert_array_almost_equal_call_result_7087 = invoke(stypy.reporting.localization.Localization(__file__, 624, 16), assert_array_almost_equal_7083, *[result_7084, expected_7085], **kwargs_7086)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_sparse_formats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sparse_formats' in the type store
        # Getting the type of 'stypy_return_type' (line 604)
        stypy_return_type_7088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7088)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sparse_formats'
        return stypy_return_type_7088


    @norecursion
    def test_precision(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_precision'
        module_type_store = module_type_store.open_function_context('test_precision', 626, 4, False)
        # Assigning a type to the variable 'self' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMMIOCoordinate.test_precision.__dict__.__setitem__('stypy_localization', localization)
        TestMMIOCoordinate.test_precision.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMMIOCoordinate.test_precision.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMMIOCoordinate.test_precision.__dict__.__setitem__('stypy_function_name', 'TestMMIOCoordinate.test_precision')
        TestMMIOCoordinate.test_precision.__dict__.__setitem__('stypy_param_names_list', [])
        TestMMIOCoordinate.test_precision.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMMIOCoordinate.test_precision.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMMIOCoordinate.test_precision.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMMIOCoordinate.test_precision.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMMIOCoordinate.test_precision.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMMIOCoordinate.test_precision.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.test_precision', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_precision', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_precision(...)' code ##################

        
        # Assigning a BinOp to a Name (line 627):
        
        # Obtaining an instance of the builtin type 'list' (line 627)
        list_7089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 627)
        # Adding element type (line 627)
        # Getting the type of 'pi' (line 627)
        pi_7090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 23), 'pi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 22), list_7089, pi_7090)
        
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 627)
        # Processing the call arguments (line 627)
        int_7095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 53), 'int')
        int_7096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 56), 'int')
        int_7097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 61), 'int')
        # Processing the call keyword arguments (line 627)
        kwargs_7098 = {}
        # Getting the type of 'range' (line 627)
        range_7094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 47), 'range', False)
        # Calling range(args, kwargs) (line 627)
        range_call_result_7099 = invoke(stypy.reporting.localization.Localization(__file__, 627, 47), range_7094, *[int_7095, int_7096, int_7097], **kwargs_7098)
        
        comprehension_7100 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 30), range_call_result_7099)
        # Assigning a type to the variable 'i' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 30), 'i', comprehension_7100)
        int_7091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 30), 'int')
        # Getting the type of 'i' (line 627)
        i_7092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 35), 'i')
        # Applying the binary operator '**' (line 627)
        result_pow_7093 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 30), '**', int_7091, i_7092)
        
        list_7101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 30), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 30), list_7101, result_pow_7093)
        # Applying the binary operator '+' (line 627)
        result_add_7102 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 22), '+', list_7089, list_7101)
        
        # Assigning a type to the variable 'test_values' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'test_values', result_add_7102)
        
        # Assigning a Call to a Name (line 628):
        
        # Call to range(...): (line 628)
        # Processing the call arguments (line 628)
        int_7104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 32), 'int')
        int_7105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 35), 'int')
        # Processing the call keyword arguments (line 628)
        kwargs_7106 = {}
        # Getting the type of 'range' (line 628)
        range_7103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 26), 'range', False)
        # Calling range(args, kwargs) (line 628)
        range_call_result_7107 = invoke(stypy.reporting.localization.Localization(__file__, 628, 26), range_7103, *[int_7104, int_7105], **kwargs_7106)
        
        # Assigning a type to the variable 'test_precisions' (line 628)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'test_precisions', range_call_result_7107)
        
        # Getting the type of 'test_values' (line 629)
        test_values_7108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 21), 'test_values')
        # Testing the type of a for loop iterable (line 629)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 629, 8), test_values_7108)
        # Getting the type of the for loop variable (line 629)
        for_loop_var_7109 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 629, 8), test_values_7108)
        # Assigning a type to the variable 'value' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'value', for_loop_var_7109)
        # SSA begins for a for statement (line 629)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'test_precisions' (line 630)
        test_precisions_7110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 29), 'test_precisions')
        # Testing the type of a for loop iterable (line 630)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 630, 12), test_precisions_7110)
        # Getting the type of the for loop variable (line 630)
        for_loop_var_7111 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 630, 12), test_precisions_7110)
        # Assigning a type to the variable 'precision' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'precision', for_loop_var_7111)
        # SSA begins for a for statement (line 630)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 632):
        int_7112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 20), 'int')
        # Getting the type of 'precision' (line 632)
        precision_7113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 24), 'precision')
        # Applying the binary operator '**' (line 632)
        result_pow_7114 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 20), '**', int_7112, precision_7113)
        
        int_7115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 36), 'int')
        # Applying the binary operator '+' (line 632)
        result_add_7116 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 20), '+', result_pow_7114, int_7115)
        
        # Assigning a type to the variable 'n' (line 632)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 16), 'n', result_add_7116)
        
        # Assigning a Call to a Name (line 633):
        
        # Call to dok_matrix(...): (line 633)
        # Processing the call arguments (line 633)
        
        # Obtaining an instance of the builtin type 'tuple' (line 633)
        tuple_7120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 633)
        # Adding element type (line 633)
        # Getting the type of 'n' (line 633)
        n_7121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 45), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 633, 45), tuple_7120, n_7121)
        # Adding element type (line 633)
        # Getting the type of 'n' (line 633)
        n_7122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 48), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 633, 45), tuple_7120, n_7122)
        
        # Processing the call keyword arguments (line 633)
        kwargs_7123 = {}
        # Getting the type of 'scipy' (line 633)
        scipy_7117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 20), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 633)
        sparse_7118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 20), scipy_7117, 'sparse')
        # Obtaining the member 'dok_matrix' of a type (line 633)
        dok_matrix_7119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 20), sparse_7118, 'dok_matrix')
        # Calling dok_matrix(args, kwargs) (line 633)
        dok_matrix_call_result_7124 = invoke(stypy.reporting.localization.Localization(__file__, 633, 20), dok_matrix_7119, *[tuple_7120], **kwargs_7123)
        
        # Assigning a type to the variable 'A' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 16), 'A', dok_matrix_call_result_7124)
        
        # Assigning a Name to a Subscript (line 634):
        # Getting the type of 'value' (line 634)
        value_7125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 30), 'value')
        # Getting the type of 'A' (line 634)
        A_7126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 16), 'A')
        
        # Obtaining an instance of the builtin type 'tuple' (line 634)
        tuple_7127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 634)
        # Adding element type (line 634)
        # Getting the type of 'n' (line 634)
        n_7128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 18), 'n')
        int_7129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 20), 'int')
        # Applying the binary operator '-' (line 634)
        result_sub_7130 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 18), '-', n_7128, int_7129)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 18), tuple_7127, result_sub_7130)
        # Adding element type (line 634)
        # Getting the type of 'n' (line 634)
        n_7131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 23), 'n')
        int_7132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 25), 'int')
        # Applying the binary operator '-' (line 634)
        result_sub_7133 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 23), '-', n_7131, int_7132)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 18), tuple_7127, result_sub_7133)
        
        # Storing an element on a container (line 634)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 16), A_7126, (tuple_7127, value_7125))
        
        # Call to mmwrite(...): (line 636)
        # Processing the call arguments (line 636)
        # Getting the type of 'self' (line 636)
        self_7135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 24), 'self', False)
        # Obtaining the member 'fn' of a type (line 636)
        fn_7136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 24), self_7135, 'fn')
        # Getting the type of 'A' (line 636)
        A_7137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 33), 'A', False)
        # Processing the call keyword arguments (line 636)
        # Getting the type of 'precision' (line 636)
        precision_7138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 46), 'precision', False)
        keyword_7139 = precision_7138
        kwargs_7140 = {'precision': keyword_7139}
        # Getting the type of 'mmwrite' (line 636)
        mmwrite_7134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'mmwrite', False)
        # Calling mmwrite(args, kwargs) (line 636)
        mmwrite_call_result_7141 = invoke(stypy.reporting.localization.Localization(__file__, 636, 16), mmwrite_7134, *[fn_7136, A_7137], **kwargs_7140)
        
        
        # Assigning a Call to a Name (line 637):
        
        # Call to mmread(...): (line 637)
        # Processing the call arguments (line 637)
        # Getting the type of 'self' (line 637)
        self_7145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 36), 'self', False)
        # Obtaining the member 'fn' of a type (line 637)
        fn_7146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 36), self_7145, 'fn')
        # Processing the call keyword arguments (line 637)
        kwargs_7147 = {}
        # Getting the type of 'scipy' (line 637)
        scipy_7142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 20), 'scipy', False)
        # Obtaining the member 'io' of a type (line 637)
        io_7143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 20), scipy_7142, 'io')
        # Obtaining the member 'mmread' of a type (line 637)
        mmread_7144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 20), io_7143, 'mmread')
        # Calling mmread(args, kwargs) (line 637)
        mmread_call_result_7148 = invoke(stypy.reporting.localization.Localization(__file__, 637, 20), mmread_7144, *[fn_7146], **kwargs_7147)
        
        # Assigning a type to the variable 'A' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 16), 'A', mmread_call_result_7148)
        
        # Call to assert_array_equal(...): (line 639)
        # Processing the call arguments (line 639)
        # Getting the type of 'A' (line 639)
        A_7150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 35), 'A', False)
        # Obtaining the member 'row' of a type (line 639)
        row_7151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 35), A_7150, 'row')
        
        # Obtaining an instance of the builtin type 'list' (line 639)
        list_7152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 639)
        # Adding element type (line 639)
        # Getting the type of 'n' (line 639)
        n_7153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 43), 'n', False)
        int_7154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 45), 'int')
        # Applying the binary operator '-' (line 639)
        result_sub_7155 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 43), '-', n_7153, int_7154)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 639, 42), list_7152, result_sub_7155)
        
        # Processing the call keyword arguments (line 639)
        kwargs_7156 = {}
        # Getting the type of 'assert_array_equal' (line 639)
        assert_array_equal_7149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 16), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 639)
        assert_array_equal_call_result_7157 = invoke(stypy.reporting.localization.Localization(__file__, 639, 16), assert_array_equal_7149, *[row_7151, list_7152], **kwargs_7156)
        
        
        # Call to assert_array_equal(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 'A' (line 640)
        A_7159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 35), 'A', False)
        # Obtaining the member 'col' of a type (line 640)
        col_7160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 35), A_7159, 'col')
        
        # Obtaining an instance of the builtin type 'list' (line 640)
        list_7161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 640)
        # Adding element type (line 640)
        # Getting the type of 'n' (line 640)
        n_7162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 43), 'n', False)
        int_7163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 45), 'int')
        # Applying the binary operator '-' (line 640)
        result_sub_7164 = python_operator(stypy.reporting.localization.Localization(__file__, 640, 43), '-', n_7162, int_7163)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 42), list_7161, result_sub_7164)
        
        # Processing the call keyword arguments (line 640)
        kwargs_7165 = {}
        # Getting the type of 'assert_array_equal' (line 640)
        assert_array_equal_7158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 16), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 640)
        assert_array_equal_call_result_7166 = invoke(stypy.reporting.localization.Localization(__file__, 640, 16), assert_array_equal_7158, *[col_7160, list_7161], **kwargs_7165)
        
        
        # Call to assert_array_almost_equal(...): (line 641)
        # Processing the call arguments (line 641)
        # Getting the type of 'A' (line 641)
        A_7168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 42), 'A', False)
        # Obtaining the member 'data' of a type (line 641)
        data_7169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 42), A_7168, 'data')
        
        # Obtaining an instance of the builtin type 'list' (line 642)
        list_7170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 642)
        # Adding element type (line 642)
        
        # Call to float(...): (line 642)
        # Processing the call arguments (line 642)
        str_7172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 27), 'str', '%%.%dg')
        # Getting the type of 'precision' (line 642)
        precision_7173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 38), 'precision', False)
        # Applying the binary operator '%' (line 642)
        result_mod_7174 = python_operator(stypy.reporting.localization.Localization(__file__, 642, 27), '%', str_7172, precision_7173)
        
        # Getting the type of 'value' (line 642)
        value_7175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 50), 'value', False)
        # Applying the binary operator '%' (line 642)
        result_mod_7176 = python_operator(stypy.reporting.localization.Localization(__file__, 642, 48), '%', result_mod_7174, value_7175)
        
        # Processing the call keyword arguments (line 642)
        kwargs_7177 = {}
        # Getting the type of 'float' (line 642)
        float_7171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 21), 'float', False)
        # Calling float(args, kwargs) (line 642)
        float_call_result_7178 = invoke(stypy.reporting.localization.Localization(__file__, 642, 21), float_7171, *[result_mod_7176], **kwargs_7177)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 20), list_7170, float_call_result_7178)
        
        # Processing the call keyword arguments (line 641)
        kwargs_7179 = {}
        # Getting the type of 'assert_array_almost_equal' (line 641)
        assert_array_almost_equal_7167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 641)
        assert_array_almost_equal_call_result_7180 = invoke(stypy.reporting.localization.Localization(__file__, 641, 16), assert_array_almost_equal_7167, *[data_7169, list_7170], **kwargs_7179)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_precision(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_precision' in the type store
        # Getting the type of 'stypy_return_type' (line 626)
        stypy_return_type_7181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7181)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_precision'
        return stypy_return_type_7181


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMMIOCoordinate.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestMMIOCoordinate' (line 452)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 0), 'TestMMIOCoordinate', TestMMIOCoordinate)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
