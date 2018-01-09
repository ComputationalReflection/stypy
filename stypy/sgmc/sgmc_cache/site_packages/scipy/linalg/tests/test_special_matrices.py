
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for functions in special_matrices.py.'''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: import numpy as np
6: from numpy import arange, add, array, eye, copy, sqrt
7: from numpy.testing import (assert_equal, assert_array_equal,
8:                            assert_array_almost_equal, assert_allclose)
9: from pytest import raises as assert_raises
10: 
11: from scipy._lib.six import xrange
12: 
13: from scipy import fftpack
14: from scipy.special import comb
15: from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie,
16:                           companion, tri, triu, tril, kron, block_diag,
17:                           helmert, hilbert, invhilbert, pascal, invpascal, dft)
18: from numpy.linalg import cond
19: 
20: 
21: def get_mat(n):
22:     data = arange(n)
23:     data = add.outer(data,data)
24:     return data
25: 
26: 
27: class TestTri(object):
28:     def test_basic(self):
29:         assert_equal(tri(4),array([[1,0,0,0],
30:                                    [1,1,0,0],
31:                                    [1,1,1,0],
32:                                    [1,1,1,1]]))
33:         assert_equal(tri(4,dtype='f'),array([[1,0,0,0],
34:                                                 [1,1,0,0],
35:                                                 [1,1,1,0],
36:                                                 [1,1,1,1]],'f'))
37: 
38:     def test_diag(self):
39:         assert_equal(tri(4,k=1),array([[1,1,0,0],
40:                                        [1,1,1,0],
41:                                        [1,1,1,1],
42:                                        [1,1,1,1]]))
43:         assert_equal(tri(4,k=-1),array([[0,0,0,0],
44:                                         [1,0,0,0],
45:                                         [1,1,0,0],
46:                                         [1,1,1,0]]))
47: 
48:     def test_2d(self):
49:         assert_equal(tri(4,3),array([[1,0,0],
50:                                      [1,1,0],
51:                                      [1,1,1],
52:                                      [1,1,1]]))
53:         assert_equal(tri(3,4),array([[1,0,0,0],
54:                                      [1,1,0,0],
55:                                      [1,1,1,0]]))
56: 
57:     def test_diag2d(self):
58:         assert_equal(tri(3,4,k=2),array([[1,1,1,0],
59:                                          [1,1,1,1],
60:                                          [1,1,1,1]]))
61:         assert_equal(tri(4,3,k=-2),array([[0,0,0],
62:                                           [0,0,0],
63:                                           [1,0,0],
64:                                           [1,1,0]]))
65: 
66: 
67: class TestTril(object):
68:     def test_basic(self):
69:         a = (100*get_mat(5)).astype('l')
70:         b = a.copy()
71:         for k in range(5):
72:             for l in range(k+1,5):
73:                 b[k,l] = 0
74:         assert_equal(tril(a),b)
75: 
76:     def test_diag(self):
77:         a = (100*get_mat(5)).astype('f')
78:         b = a.copy()
79:         for k in range(5):
80:             for l in range(k+3,5):
81:                 b[k,l] = 0
82:         assert_equal(tril(a,k=2),b)
83:         b = a.copy()
84:         for k in range(5):
85:             for l in range(max((k-1,0)),5):
86:                 b[k,l] = 0
87:         assert_equal(tril(a,k=-2),b)
88: 
89: 
90: class TestTriu(object):
91:     def test_basic(self):
92:         a = (100*get_mat(5)).astype('l')
93:         b = a.copy()
94:         for k in range(5):
95:             for l in range(k+1,5):
96:                 b[l,k] = 0
97:         assert_equal(triu(a),b)
98: 
99:     def test_diag(self):
100:         a = (100*get_mat(5)).astype('f')
101:         b = a.copy()
102:         for k in range(5):
103:             for l in range(max((k-1,0)),5):
104:                 b[l,k] = 0
105:         assert_equal(triu(a,k=2),b)
106:         b = a.copy()
107:         for k in range(5):
108:             for l in range(k+3,5):
109:                 b[l,k] = 0
110:         assert_equal(triu(a,k=-2),b)
111: 
112: 
113: class TestToeplitz(object):
114: 
115:     def test_basic(self):
116:         y = toeplitz([1,2,3])
117:         assert_array_equal(y,[[1,2,3],[2,1,2],[3,2,1]])
118:         y = toeplitz([1,2,3],[1,4,5])
119:         assert_array_equal(y,[[1,4,5],[2,1,4],[3,2,1]])
120: 
121:     def test_complex_01(self):
122:         data = (1.0 + arange(3.0)) * (1.0 + 1.0j)
123:         x = copy(data)
124:         t = toeplitz(x)
125:         # Calling toeplitz should not change x.
126:         assert_array_equal(x, data)
127:         # According to the docstring, x should be the first column of t.
128:         col0 = t[:,0]
129:         assert_array_equal(col0, data)
130:         assert_array_equal(t[0,1:], data[1:].conj())
131: 
132:     def test_scalar_00(self):
133:         '''Scalar arguments still produce a 2D array.'''
134:         t = toeplitz(10)
135:         assert_array_equal(t, [[10]])
136:         t = toeplitz(10, 20)
137:         assert_array_equal(t, [[10]])
138: 
139:     def test_scalar_01(self):
140:         c = array([1,2,3])
141:         t = toeplitz(c, 1)
142:         assert_array_equal(t, [[1],[2],[3]])
143: 
144:     def test_scalar_02(self):
145:         c = array([1,2,3])
146:         t = toeplitz(c, array(1))
147:         assert_array_equal(t, [[1],[2],[3]])
148: 
149:     def test_scalar_03(self):
150:         c = array([1,2,3])
151:         t = toeplitz(c, array([1]))
152:         assert_array_equal(t, [[1],[2],[3]])
153: 
154:     def test_scalar_04(self):
155:         r = array([10,2,3])
156:         t = toeplitz(1, r)
157:         assert_array_equal(t, [[1,2,3]])
158: 
159: 
160: class TestHankel(object):
161:     def test_basic(self):
162:         y = hankel([1,2,3])
163:         assert_array_equal(y, [[1,2,3], [2,3,0], [3,0,0]])
164:         y = hankel([1,2,3], [3,4,5])
165:         assert_array_equal(y, [[1,2,3], [2,3,4], [3,4,5]])
166: 
167: 
168: class TestCirculant(object):
169:     def test_basic(self):
170:         y = circulant([1,2,3])
171:         assert_array_equal(y, [[1,3,2], [2,1,3], [3,2,1]])
172: 
173: 
174: class TestHadamard(object):
175: 
176:     def test_basic(self):
177: 
178:         y = hadamard(1)
179:         assert_array_equal(y, [[1]])
180: 
181:         y = hadamard(2, dtype=float)
182:         assert_array_equal(y, [[1.0, 1.0], [1.0, -1.0]])
183: 
184:         y = hadamard(4)
185:         assert_array_equal(y, [[1,1,1,1], [1,-1,1,-1], [1,1,-1,-1], [1,-1,-1,1]])
186: 
187:         assert_raises(ValueError, hadamard, 0)
188:         assert_raises(ValueError, hadamard, 5)
189: 
190: 
191: class TestLeslie(object):
192: 
193:     def test_bad_shapes(self):
194:         assert_raises(ValueError, leslie, [[1,1],[2,2]], [3,4,5])
195:         assert_raises(ValueError, leslie, [3,4,5], [[1,1],[2,2]])
196:         assert_raises(ValueError, leslie, [1,2], [1,2])
197:         assert_raises(ValueError, leslie, [1], [])
198: 
199:     def test_basic(self):
200:         a = leslie([1, 2, 3], [0.25, 0.5])
201:         expected = array([
202:             [1.0, 2.0, 3.0],
203:             [0.25, 0.0, 0.0],
204:             [0.0, 0.5, 0.0]])
205:         assert_array_equal(a, expected)
206: 
207: 
208: class TestCompanion(object):
209: 
210:     def test_bad_shapes(self):
211:         assert_raises(ValueError, companion, [[1,1],[2,2]])
212:         assert_raises(ValueError, companion, [0,4,5])
213:         assert_raises(ValueError, companion, [1])
214:         assert_raises(ValueError, companion, [])
215: 
216:     def test_basic(self):
217:         c = companion([1, 2, 3])
218:         expected = array([
219:             [-2.0, -3.0],
220:             [1.0, 0.0]])
221:         assert_array_equal(c, expected)
222: 
223:         c = companion([2.0, 5.0, -10.0])
224:         expected = array([
225:             [-2.5, 5.0],
226:             [1.0, 0.0]])
227:         assert_array_equal(c, expected)
228: 
229: 
230: class TestBlockDiag:
231:     def test_basic(self):
232:         x = block_diag(eye(2), [[1,2], [3,4], [5,6]], [[1, 2, 3]])
233:         assert_array_equal(x, [[1, 0, 0, 0, 0, 0, 0],
234:                                [0, 1, 0, 0, 0, 0, 0],
235:                                [0, 0, 1, 2, 0, 0, 0],
236:                                [0, 0, 3, 4, 0, 0, 0],
237:                                [0, 0, 5, 6, 0, 0, 0],
238:                                [0, 0, 0, 0, 1, 2, 3]])
239: 
240:     def test_dtype(self):
241:         x = block_diag([[1.5]])
242:         assert_equal(x.dtype, float)
243: 
244:         x = block_diag([[True]])
245:         assert_equal(x.dtype, bool)
246: 
247:     def test_mixed_dtypes(self):
248:         actual = block_diag([[1]], [[1j]])
249:         desired = np.array([[1, 0], [0, 1j]])
250:         assert_array_equal(actual, desired)
251: 
252:     def test_scalar_and_1d_args(self):
253:         a = block_diag(1)
254:         assert_equal(a.shape, (1,1))
255:         assert_array_equal(a, [[1]])
256: 
257:         a = block_diag([2,3], 4)
258:         assert_array_equal(a, [[2, 3, 0], [0, 0, 4]])
259: 
260:     def test_bad_arg(self):
261:         assert_raises(ValueError, block_diag, [[[1]]])
262: 
263:     def test_no_args(self):
264:         a = block_diag()
265:         assert_equal(a.ndim, 2)
266:         assert_equal(a.nbytes, 0)
267:     
268:     def test_empty_matrix_arg(self):
269:         # regression test for gh-4596: check the shape of the result
270:         # for empty matrix inputs. Empty matrices are no longer ignored
271:         # (gh-4908) it is viewed as a shape (1, 0) matrix.
272:         a = block_diag([[1, 0], [0, 1]],
273:                        [],
274:                        [[2, 3], [4, 5], [6, 7]])
275:         assert_array_equal(a, [[1, 0, 0, 0],
276:                                [0, 1, 0, 0],
277:                                [0, 0, 0, 0],
278:                                [0, 0, 2, 3],
279:                                [0, 0, 4, 5],
280:                                [0, 0, 6, 7]])
281: 
282:     def test_zerosized_matrix_arg(self):
283:         # test for gh-4908: check the shape of the result for 
284:         # zero-sized matrix inputs, i.e. matrices with shape (0,n) or (n,0).
285:         # note that [[]] takes shape (1,0)
286:         a = block_diag([[1, 0], [0, 1]],
287:                        [[]],
288:                        [[2, 3], [4, 5], [6, 7]],
289:                        np.zeros([0,2],dtype='int32'))
290:         assert_array_equal(a, [[1, 0, 0, 0, 0, 0],
291:                                [0, 1, 0, 0, 0, 0],
292:                                [0, 0, 0, 0, 0, 0],
293:                                [0, 0, 2, 3, 0, 0],
294:                                [0, 0, 4, 5, 0, 0],
295:                                [0, 0, 6, 7, 0, 0]])
296: 
297: class TestKron:
298: 
299:     def test_basic(self):
300: 
301:         a = kron(array([[1, 2], [3, 4]]), array([[1, 1, 1]]))
302:         assert_array_equal(a, array([[1, 1, 1, 2, 2, 2],
303:                                      [3, 3, 3, 4, 4, 4]]))
304: 
305:         m1 = array([[1, 2], [3, 4]])
306:         m2 = array([[10], [11]])
307:         a = kron(m1, m2)
308:         expected = array([[10, 20],
309:                           [11, 22],
310:                           [30, 40],
311:                           [33, 44]])
312:         assert_array_equal(a, expected)
313: 
314: 
315: class TestHelmert(object):
316: 
317:     def test_orthogonality(self):
318:         for n in range(1, 7):
319:             H = helmert(n, full=True)
320:             I = np.eye(n)
321:             assert_allclose(H.dot(H.T), I, atol=1e-12)
322:             assert_allclose(H.T.dot(H), I, atol=1e-12)
323: 
324:     def test_subspace(self):
325:         for n in range(2, 7):
326:             H_full = helmert(n, full=True)
327:             H_partial = helmert(n)
328:             for U in H_full[1:, :].T, H_partial.T:
329:                 C = np.eye(n) - np.ones((n, n)) / n
330:                 assert_allclose(U.dot(U.T), C)
331:                 assert_allclose(U.T.dot(U), np.eye(n-1), atol=1e-12)
332: 
333: 
334: class TestHilbert(object):
335: 
336:     def test_basic(self):
337:         h3 = array([[1.0, 1/2., 1/3.],
338:                     [1/2., 1/3., 1/4.],
339:                     [1/3., 1/4., 1/5.]])
340:         assert_array_almost_equal(hilbert(3), h3)
341: 
342:         assert_array_equal(hilbert(1), [[1.0]])
343: 
344:         h0 = hilbert(0)
345:         assert_equal(h0.shape, (0,0))
346: 
347: 
348: class TestInvHilbert(object):
349: 
350:     def test_basic(self):
351:         invh1 = array([[1]])
352:         assert_array_equal(invhilbert(1, exact=True), invh1)
353:         assert_array_equal(invhilbert(1), invh1)
354: 
355:         invh2 = array([[4, -6],
356:                        [-6, 12]])
357:         assert_array_equal(invhilbert(2, exact=True), invh2)
358:         assert_array_almost_equal(invhilbert(2), invh2)
359: 
360:         invh3 = array([[9, -36, 30],
361:                        [-36, 192, -180],
362:                         [30, -180, 180]])
363:         assert_array_equal(invhilbert(3, exact=True), invh3)
364:         assert_array_almost_equal(invhilbert(3), invh3)
365: 
366:         invh4 = array([[16, -120, 240, -140],
367:                        [-120, 1200, -2700, 1680],
368:                        [240, -2700, 6480, -4200],
369:                        [-140, 1680, -4200, 2800]])
370:         assert_array_equal(invhilbert(4, exact=True), invh4)
371:         assert_array_almost_equal(invhilbert(4), invh4)
372: 
373:         invh5 = array([[25, -300, 1050, -1400, 630],
374:                        [-300, 4800, -18900, 26880, -12600],
375:                        [1050, -18900, 79380, -117600, 56700],
376:                        [-1400, 26880, -117600, 179200, -88200],
377:                        [630, -12600, 56700, -88200, 44100]])
378:         assert_array_equal(invhilbert(5, exact=True), invh5)
379:         assert_array_almost_equal(invhilbert(5), invh5)
380: 
381:         invh17 = array([
382:             [289, -41616, 1976760, -46124400, 629598060, -5540462928,
383:              33374693352, -143034400080, 446982500250, -1033026222800,
384:              1774926873720, -2258997839280, 2099709530100, -1384423866000,
385:              613101997800, -163493866080, 19835652870],
386:             [-41616, 7990272, -426980160, 10627061760, -151103534400, 1367702848512,
387:              -8410422724704, 36616806420480, -115857864064800, 270465047424000,
388:              -468580694662080, 600545887119360, -561522320049600, 372133135180800,
389:              -165537539406000, 44316454993920, -5395297580640],
390:             [1976760, -426980160, 24337869120, -630981792000, 9228108708000,
391:              -85267724461920, 532660105897920, -2348052711713280, 7504429831470000,
392:              -17664748409880000, 30818191841236800, -39732544853164800,
393:              37341234283298400, -24857330514030000, 11100752642520000,
394:              -2982128117299200, 364182586693200],
395:             [-46124400, 10627061760, -630981792000, 16826181120000,
396:              -251209625940000, 2358021022156800, -14914482965141760,
397:              66409571644416000, -214015221119700000, 507295338950400000,
398:              -890303319857952000, 1153715376477081600, -1089119333262870000,
399:              727848632044800000, -326170262829600000, 87894302404608000,
400:              -10763618673376800],
401:             [629598060, -151103534400, 9228108708000,
402:              -251209625940000, 3810012660090000, -36210360321495360,
403:              231343968720664800, -1038687206500944000, 3370739732635275000,
404:              -8037460526495400000, 14178080368737885600, -18454939322943942000,
405:              17489975175339030000, -11728977435138600000, 5272370630081100000,
406:              -1424711708039692800, 174908803442373000],
407:             [-5540462928, 1367702848512, -85267724461920, 2358021022156800,
408:              -36210360321495360, 347619459086355456, -2239409617216035264,
409:              10124803292907663360, -33052510749726468000, 79217210949138662400,
410:              -140362995650505067440, 183420385176741672960, -174433352415381259200,
411:              117339159519533952000, -52892422160973595200, 14328529177999196160,
412:              -1763080738699119840],
413:             [33374693352, -8410422724704, 532660105897920,
414:              -14914482965141760, 231343968720664800, -2239409617216035264,
415:              14527452132196331328, -66072377044391477760, 216799987176909536400,
416:              -521925895055522958000, 928414062734059661760, -1217424500995626443520,
417:              1161358898976091015200, -783401860847777371200, 354015418167362952000,
418:              -96120549902411274240, 11851820521255194480],
419:             [-143034400080, 36616806420480, -2348052711713280, 66409571644416000,
420:              -1038687206500944000, 10124803292907663360, -66072377044391477760,
421:              302045152202932469760, -995510145200094810000, 2405996923185123840000,
422:              -4294704507885446054400, 5649058909023744614400,
423:              -5403874060541811254400, 3654352703663101440000,
424:              -1655137020003255360000, 450325202737117593600, -55630994283442749600],
425:             [446982500250, -115857864064800, 7504429831470000, -214015221119700000,
426:              3370739732635275000, -33052510749726468000, 216799987176909536400,
427:              -995510145200094810000, 3293967392206196062500,
428:              -7988661659013106500000, 14303908928401362270000,
429:              -18866974090684772052000, 18093328327706957325000,
430:              -12263364009096700500000, 5565847995255512250000,
431:              -1517208935002984080000, 187754605706619279900],
432:             [-1033026222800, 270465047424000, -17664748409880000,
433:              507295338950400000, -8037460526495400000, 79217210949138662400,
434:              -521925895055522958000, 2405996923185123840000,
435:              -7988661659013106500000, 19434404971634224000000,
436:              -34894474126569249192000, 46141453390504792320000,
437:              -44349976506971935800000, 30121928988527376000000,
438:              -13697025107665828500000, 3740200989399948902400,
439:              -463591619028689580000],
440:             [1774926873720, -468580694662080,
441:              30818191841236800, -890303319857952000, 14178080368737885600,
442:              -140362995650505067440, 928414062734059661760, -4294704507885446054400,
443:              14303908928401362270000, -34894474126569249192000,
444:              62810053427824648545600, -83243376594051600326400,
445:              80177044485212743068000, -54558343880470209780000,
446:              24851882355348879230400, -6797096028813368678400, 843736746632215035600],
447:             [-2258997839280, 600545887119360, -39732544853164800,
448:              1153715376477081600, -18454939322943942000, 183420385176741672960,
449:              -1217424500995626443520, 5649058909023744614400,
450:              -18866974090684772052000, 46141453390504792320000,
451:              -83243376594051600326400, 110552468520163390156800,
452:              -106681852579497947388000, 72720410752415168870400,
453:              -33177973900974346080000, 9087761081682520473600,
454:              -1129631016152221783200],
455:             [2099709530100, -561522320049600, 37341234283298400,
456:              -1089119333262870000, 17489975175339030000, -174433352415381259200,
457:              1161358898976091015200, -5403874060541811254400,
458:              18093328327706957325000, -44349976506971935800000,
459:              80177044485212743068000, -106681852579497947388000,
460:              103125790826848015808400, -70409051543137015800000,
461:              32171029219823375700000, -8824053728865840192000,
462:              1098252376814660067000],
463:             [-1384423866000, 372133135180800,
464:              -24857330514030000, 727848632044800000, -11728977435138600000,
465:              117339159519533952000, -783401860847777371200, 3654352703663101440000,
466:              -12263364009096700500000, 30121928988527376000000,
467:              -54558343880470209780000, 72720410752415168870400,
468:              -70409051543137015800000, 48142941226076592000000,
469:              -22027500987368499000000, 6049545098753157120000,
470:              -753830033789944188000],
471:             [613101997800, -165537539406000,
472:              11100752642520000, -326170262829600000, 5272370630081100000,
473:              -52892422160973595200, 354015418167362952000, -1655137020003255360000,
474:              5565847995255512250000, -13697025107665828500000,
475:              24851882355348879230400, -33177973900974346080000,
476:              32171029219823375700000, -22027500987368499000000,
477:              10091416708498869000000, -2774765838662800128000, 346146444087219270000],
478:             [-163493866080, 44316454993920, -2982128117299200, 87894302404608000,
479:              -1424711708039692800, 14328529177999196160, -96120549902411274240,
480:              450325202737117593600, -1517208935002984080000, 3740200989399948902400,
481:              -6797096028813368678400, 9087761081682520473600,
482:              -8824053728865840192000, 6049545098753157120000,
483:              -2774765838662800128000, 763806510427609497600, -95382575704033754400],
484:             [19835652870, -5395297580640, 364182586693200, -10763618673376800,
485:              174908803442373000, -1763080738699119840, 11851820521255194480,
486:              -55630994283442749600, 187754605706619279900, -463591619028689580000,
487:              843736746632215035600, -1129631016152221783200, 1098252376814660067000,
488:              -753830033789944188000, 346146444087219270000, -95382575704033754400,
489:              11922821963004219300]
490:             ])
491:         assert_array_equal(invhilbert(17, exact=True), invh17)
492:         assert_allclose(invhilbert(17), invh17.astype(float), rtol=1e-12)
493: 
494:     def test_inverse(self):
495:         for n in xrange(1, 10):
496:             a = hilbert(n)
497:             b = invhilbert(n)
498:             # The Hilbert matrix is increasingly badly conditioned,
499:             # so take that into account in the test
500:             c = cond(a)
501:             assert_allclose(a.dot(b), eye(n), atol=1e-15*c, rtol=1e-15*c)
502: 
503: 
504: class TestPascal(object):
505: 
506:     cases = [
507:         (1, array([[1]]), array([[1]])),
508:         (2, array([[1, 1],
509:                    [1, 2]]),
510:             array([[1, 0],
511:                    [1, 1]])),
512:         (3, array([[1, 1, 1],
513:                    [1, 2, 3],
514:                    [1, 3, 6]]),
515:             array([[1, 0, 0],
516:                    [1, 1, 0],
517:                    [1, 2, 1]])),
518:         (4, array([[1, 1, 1, 1],
519:                    [1, 2, 3, 4],
520:                    [1, 3, 6, 10],
521:                    [1, 4, 10, 20]]),
522:             array([[1, 0, 0, 0],
523:                    [1, 1, 0, 0],
524:                    [1, 2, 1, 0],
525:                    [1, 3, 3, 1]])),
526:     ]
527: 
528:     def check_case(self, n, sym, low):
529:         assert_array_equal(pascal(n), sym)
530:         assert_array_equal(pascal(n, kind='lower'), low)
531:         assert_array_equal(pascal(n, kind='upper'), low.T)
532:         assert_array_almost_equal(pascal(n, exact=False), sym)
533:         assert_array_almost_equal(pascal(n, exact=False, kind='lower'), low)
534:         assert_array_almost_equal(pascal(n, exact=False, kind='upper'), low.T)
535: 
536:     def test_cases(self):
537:         for n, sym, low in self.cases:
538:             self.check_case(n, sym, low)
539: 
540:     def test_big(self):
541:         p = pascal(50)
542:         assert_equal(p[-1, -1], comb(98, 49, exact=True))
543: 
544:     def test_threshold(self):
545:         # Regression test.  An early version of `pascal` returned an
546:         # array of type np.uint64 for n=35, but that data type is too small
547:         # to hold p[-1, -1].  The second assert_equal below would fail
548:         # because p[-1, -1] overflowed.
549:         p = pascal(34)
550:         assert_equal(2*p.item(-1, -2), p.item(-1, -1), err_msg="n = 34")
551:         p = pascal(35)
552:         assert_equal(2*p.item(-1, -2), p.item(-1, -1), err_msg="n = 35")
553: 
554: 
555: def test_invpascal():
556: 
557:     def check_invpascal(n, kind, exact):
558:         ip = invpascal(n, kind=kind, exact=exact)
559:         p = pascal(n, kind=kind, exact=exact)
560:         # Matrix-multiply ip and p, and check that we get the identity matrix.
561:         # We can't use the simple expression e = ip.dot(p), because when
562:         # n < 35 and exact is True, p.dtype is np.uint64 and ip.dtype is
563:         # np.int64. The product of those dtypes is np.float64, which loses
564:         # precision when n is greater than 18.  Instead we'll cast both to
565:         # object arrays, and then multiply.
566:         e = ip.astype(object).dot(p.astype(object))
567:         assert_array_equal(e, eye(n), err_msg="n=%d  kind=%r exact=%r" %
568:                                               (n, kind, exact))
569: 
570:     kinds = ['symmetric', 'lower', 'upper']
571: 
572:     ns = [1, 2, 5, 18]
573:     for n in ns:
574:         for kind in kinds:
575:             for exact in [True, False]:
576:                 check_invpascal(n, kind, exact)
577: 
578:     ns = [19, 34, 35, 50]
579:     for n in ns:
580:         for kind in kinds:
581:             check_invpascal(n, kind, True)
582: 
583: 
584: def test_dft():
585:     m = dft(2)
586:     expected = array([[1.0, 1.0], [1.0, -1.0]])
587:     assert_array_almost_equal(m, expected)
588:     m = dft(2, scale='n')
589:     assert_array_almost_equal(m, expected/2.0)
590:     m = dft(2, scale='sqrtn')
591:     assert_array_almost_equal(m, expected/sqrt(2.0))
592: 
593:     x = array([0, 1, 2, 3, 4, 5, 0, 1])
594:     m = dft(8)
595:     mx = m.dot(x)
596:     fx = fftpack.fft(x)
597:     assert_array_almost_equal(mx, fx)
598: 
599: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_110969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for functions in special_matrices.py.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110970 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_110970) is not StypyTypeError):

    if (import_110970 != 'pyd_module'):
        __import__(import_110970)
        sys_modules_110971 = sys.modules[import_110970]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_110971.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_110970)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy import arange, add, array, eye, copy, sqrt' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110972 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_110972) is not StypyTypeError):

    if (import_110972 != 'pyd_module'):
        __import__(import_110972)
        sys_modules_110973 = sys.modules[import_110972]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', sys_modules_110973.module_type_store, module_type_store, ['arange', 'add', 'array', 'eye', 'copy', 'sqrt'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_110973, sys_modules_110973.module_type_store, module_type_store)
    else:
        from numpy import arange, add, array, eye, copy, sqrt

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', None, module_type_store, ['arange', 'add', 'array', 'eye', 'copy', 'sqrt'], [arange, add, array, eye, copy, sqrt])

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_110972)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal, assert_allclose' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110974 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_110974) is not StypyTypeError):

    if (import_110974 != 'pyd_module'):
        __import__(import_110974)
        sys_modules_110975 = sys.modules[import_110974]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_110975.module_type_store, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_array_almost_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_110975, sys_modules_110975.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_array_almost_equal', 'assert_allclose'], [assert_equal, assert_array_equal, assert_array_almost_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_110974)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from pytest import assert_raises' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110976 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_110976) is not StypyTypeError):

    if (import_110976 != 'pyd_module'):
        __import__(import_110976)
        sys_modules_110977 = sys.modules[import_110976]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_110977.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_110977, sys_modules_110977.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_110976)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib.six import xrange' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110978 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six')

if (type(import_110978) is not StypyTypeError):

    if (import_110978 != 'pyd_module'):
        __import__(import_110978)
        sys_modules_110979 = sys.modules[import_110978]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', sys_modules_110979.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_110979, sys_modules_110979.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', import_110978)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy import fftpack' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110980 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy')

if (type(import_110980) is not StypyTypeError):

    if (import_110980 != 'pyd_module'):
        __import__(import_110980)
        sys_modules_110981 = sys.modules[import_110980]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy', sys_modules_110981.module_type_store, module_type_store, ['fftpack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_110981, sys_modules_110981.module_type_store, module_type_store)
    else:
        from scipy import fftpack

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy', None, module_type_store, ['fftpack'], [fftpack])

else:
    # Assigning a type to the variable 'scipy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy', import_110980)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.special import comb' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110982 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.special')

if (type(import_110982) is not StypyTypeError):

    if (import_110982 != 'pyd_module'):
        __import__(import_110982)
        sys_modules_110983 = sys.modules[import_110982]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.special', sys_modules_110983.module_type_store, module_type_store, ['comb'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_110983, sys_modules_110983.module_type_store, module_type_store)
    else:
        from scipy.special import comb

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.special', None, module_type_store, ['comb'], [comb])

else:
    # Assigning a type to the variable 'scipy.special' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.special', import_110982)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.linalg import toeplitz, hankel, circulant, hadamard, leslie, companion, tri, triu, tril, kron, block_diag, helmert, hilbert, invhilbert, pascal, invpascal, dft' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110984 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg')

if (type(import_110984) is not StypyTypeError):

    if (import_110984 != 'pyd_module'):
        __import__(import_110984)
        sys_modules_110985 = sys.modules[import_110984]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg', sys_modules_110985.module_type_store, module_type_store, ['toeplitz', 'hankel', 'circulant', 'hadamard', 'leslie', 'companion', 'tri', 'triu', 'tril', 'kron', 'block_diag', 'helmert', 'hilbert', 'invhilbert', 'pascal', 'invpascal', 'dft'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_110985, sys_modules_110985.module_type_store, module_type_store)
    else:
        from scipy.linalg import toeplitz, hankel, circulant, hadamard, leslie, companion, tri, triu, tril, kron, block_diag, helmert, hilbert, invhilbert, pascal, invpascal, dft

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg', None, module_type_store, ['toeplitz', 'hankel', 'circulant', 'hadamard', 'leslie', 'companion', 'tri', 'triu', 'tril', 'kron', 'block_diag', 'helmert', 'hilbert', 'invhilbert', 'pascal', 'invpascal', 'dft'], [toeplitz, hankel, circulant, hadamard, leslie, companion, tri, triu, tril, kron, block_diag, helmert, hilbert, invhilbert, pascal, invpascal, dft])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg', import_110984)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.linalg import cond' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_110986 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.linalg')

if (type(import_110986) is not StypyTypeError):

    if (import_110986 != 'pyd_module'):
        __import__(import_110986)
        sys_modules_110987 = sys.modules[import_110986]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.linalg', sys_modules_110987.module_type_store, module_type_store, ['cond'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_110987, sys_modules_110987.module_type_store, module_type_store)
    else:
        from numpy.linalg import cond

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.linalg', None, module_type_store, ['cond'], [cond])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.linalg', import_110986)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')


@norecursion
def get_mat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_mat'
    module_type_store = module_type_store.open_function_context('get_mat', 21, 0, False)
    
    # Passed parameters checking function
    get_mat.stypy_localization = localization
    get_mat.stypy_type_of_self = None
    get_mat.stypy_type_store = module_type_store
    get_mat.stypy_function_name = 'get_mat'
    get_mat.stypy_param_names_list = ['n']
    get_mat.stypy_varargs_param_name = None
    get_mat.stypy_kwargs_param_name = None
    get_mat.stypy_call_defaults = defaults
    get_mat.stypy_call_varargs = varargs
    get_mat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_mat', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_mat', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_mat(...)' code ##################

    
    # Assigning a Call to a Name (line 22):
    
    # Call to arange(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'n' (line 22)
    n_110989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'n', False)
    # Processing the call keyword arguments (line 22)
    kwargs_110990 = {}
    # Getting the type of 'arange' (line 22)
    arange_110988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'arange', False)
    # Calling arange(args, kwargs) (line 22)
    arange_call_result_110991 = invoke(stypy.reporting.localization.Localization(__file__, 22, 11), arange_110988, *[n_110989], **kwargs_110990)
    
    # Assigning a type to the variable 'data' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'data', arange_call_result_110991)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to outer(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'data' (line 23)
    data_110994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'data', False)
    # Getting the type of 'data' (line 23)
    data_110995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'data', False)
    # Processing the call keyword arguments (line 23)
    kwargs_110996 = {}
    # Getting the type of 'add' (line 23)
    add_110992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'add', False)
    # Obtaining the member 'outer' of a type (line 23)
    outer_110993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 11), add_110992, 'outer')
    # Calling outer(args, kwargs) (line 23)
    outer_call_result_110997 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), outer_110993, *[data_110994, data_110995], **kwargs_110996)
    
    # Assigning a type to the variable 'data' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'data', outer_call_result_110997)
    # Getting the type of 'data' (line 24)
    data_110998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'data')
    # Assigning a type to the variable 'stypy_return_type' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type', data_110998)
    
    # ################# End of 'get_mat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_mat' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_110999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_110999)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_mat'
    return stypy_return_type_110999

# Assigning a type to the variable 'get_mat' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'get_mat', get_mat)
# Declaration of the 'TestTri' class

class TestTri(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTri.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestTri.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTri.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTri.test_basic.__dict__.__setitem__('stypy_function_name', 'TestTri.test_basic')
        TestTri.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestTri.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTri.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTri.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTri.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTri.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTri.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTri.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Call to assert_equal(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Call to tri(...): (line 29)
        # Processing the call arguments (line 29)
        int_111002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'int')
        # Processing the call keyword arguments (line 29)
        kwargs_111003 = {}
        # Getting the type of 'tri' (line 29)
        tri_111001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), 'tri', False)
        # Calling tri(args, kwargs) (line 29)
        tri_call_result_111004 = invoke(stypy.reporting.localization.Localization(__file__, 29, 21), tri_111001, *[int_111002], **kwargs_111003)
        
        
        # Call to array(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_111006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_111007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        int_111008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 35), list_111007, int_111008)
        # Adding element type (line 29)
        int_111009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 35), list_111007, int_111009)
        # Adding element type (line 29)
        int_111010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 35), list_111007, int_111010)
        # Adding element type (line 29)
        int_111011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 35), list_111007, int_111011)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 34), list_111006, list_111007)
        # Adding element type (line 29)
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_111012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        int_111013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 35), list_111012, int_111013)
        # Adding element type (line 30)
        int_111014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 35), list_111012, int_111014)
        # Adding element type (line 30)
        int_111015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 35), list_111012, int_111015)
        # Adding element type (line 30)
        int_111016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 35), list_111012, int_111016)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 34), list_111006, list_111012)
        # Adding element type (line 29)
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_111017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        int_111018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 35), list_111017, int_111018)
        # Adding element type (line 31)
        int_111019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 35), list_111017, int_111019)
        # Adding element type (line 31)
        int_111020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 35), list_111017, int_111020)
        # Adding element type (line 31)
        int_111021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 35), list_111017, int_111021)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 34), list_111006, list_111017)
        # Adding element type (line 29)
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_111022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        # Adding element type (line 32)
        int_111023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 35), list_111022, int_111023)
        # Adding element type (line 32)
        int_111024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 35), list_111022, int_111024)
        # Adding element type (line 32)
        int_111025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 35), list_111022, int_111025)
        # Adding element type (line 32)
        int_111026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 35), list_111022, int_111026)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 34), list_111006, list_111022)
        
        # Processing the call keyword arguments (line 29)
        kwargs_111027 = {}
        # Getting the type of 'array' (line 29)
        array_111005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 28), 'array', False)
        # Calling array(args, kwargs) (line 29)
        array_call_result_111028 = invoke(stypy.reporting.localization.Localization(__file__, 29, 28), array_111005, *[list_111006], **kwargs_111027)
        
        # Processing the call keyword arguments (line 29)
        kwargs_111029 = {}
        # Getting the type of 'assert_equal' (line 29)
        assert_equal_111000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 29)
        assert_equal_call_result_111030 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assert_equal_111000, *[tri_call_result_111004, array_call_result_111028], **kwargs_111029)
        
        
        # Call to assert_equal(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to tri(...): (line 33)
        # Processing the call arguments (line 33)
        int_111033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'int')
        # Processing the call keyword arguments (line 33)
        str_111034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 33), 'str', 'f')
        keyword_111035 = str_111034
        kwargs_111036 = {'dtype': keyword_111035}
        # Getting the type of 'tri' (line 33)
        tri_111032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'tri', False)
        # Calling tri(args, kwargs) (line 33)
        tri_call_result_111037 = invoke(stypy.reporting.localization.Localization(__file__, 33, 21), tri_111032, *[int_111033], **kwargs_111036)
        
        
        # Call to array(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_111039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_111040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        int_111041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 45), list_111040, int_111041)
        # Adding element type (line 33)
        int_111042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 45), list_111040, int_111042)
        # Adding element type (line 33)
        int_111043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 45), list_111040, int_111043)
        # Adding element type (line 33)
        int_111044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 45), list_111040, int_111044)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 44), list_111039, list_111040)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_111045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        int_111046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 48), list_111045, int_111046)
        # Adding element type (line 34)
        int_111047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 48), list_111045, int_111047)
        # Adding element type (line 34)
        int_111048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 48), list_111045, int_111048)
        # Adding element type (line 34)
        int_111049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 48), list_111045, int_111049)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 44), list_111039, list_111045)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_111050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        int_111051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 48), list_111050, int_111051)
        # Adding element type (line 35)
        int_111052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 48), list_111050, int_111052)
        # Adding element type (line 35)
        int_111053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 48), list_111050, int_111053)
        # Adding element type (line 35)
        int_111054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 48), list_111050, int_111054)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 44), list_111039, list_111050)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 36)
        list_111055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 36)
        # Adding element type (line 36)
        int_111056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 48), list_111055, int_111056)
        # Adding element type (line 36)
        int_111057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 48), list_111055, int_111057)
        # Adding element type (line 36)
        int_111058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 48), list_111055, int_111058)
        # Adding element type (line 36)
        int_111059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 48), list_111055, int_111059)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 44), list_111039, list_111055)
        
        str_111060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 59), 'str', 'f')
        # Processing the call keyword arguments (line 33)
        kwargs_111061 = {}
        # Getting the type of 'array' (line 33)
        array_111038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 38), 'array', False)
        # Calling array(args, kwargs) (line 33)
        array_call_result_111062 = invoke(stypy.reporting.localization.Localization(__file__, 33, 38), array_111038, *[list_111039, str_111060], **kwargs_111061)
        
        # Processing the call keyword arguments (line 33)
        kwargs_111063 = {}
        # Getting the type of 'assert_equal' (line 33)
        assert_equal_111031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 33)
        assert_equal_call_result_111064 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), assert_equal_111031, *[tri_call_result_111037, array_call_result_111062], **kwargs_111063)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_111065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111065)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_111065


    @norecursion
    def test_diag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_diag'
        module_type_store = module_type_store.open_function_context('test_diag', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTri.test_diag.__dict__.__setitem__('stypy_localization', localization)
        TestTri.test_diag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTri.test_diag.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTri.test_diag.__dict__.__setitem__('stypy_function_name', 'TestTri.test_diag')
        TestTri.test_diag.__dict__.__setitem__('stypy_param_names_list', [])
        TestTri.test_diag.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTri.test_diag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTri.test_diag.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTri.test_diag.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTri.test_diag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTri.test_diag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTri.test_diag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_diag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_diag(...)' code ##################

        
        # Call to assert_equal(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to tri(...): (line 39)
        # Processing the call arguments (line 39)
        int_111068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'int')
        # Processing the call keyword arguments (line 39)
        int_111069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 29), 'int')
        keyword_111070 = int_111069
        kwargs_111071 = {'k': keyword_111070}
        # Getting the type of 'tri' (line 39)
        tri_111067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'tri', False)
        # Calling tri(args, kwargs) (line 39)
        tri_call_result_111072 = invoke(stypy.reporting.localization.Localization(__file__, 39, 21), tri_111067, *[int_111068], **kwargs_111071)
        
        
        # Call to array(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_111074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_111075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        int_111076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 39), list_111075, int_111076)
        # Adding element type (line 39)
        int_111077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 39), list_111075, int_111077)
        # Adding element type (line 39)
        int_111078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 39), list_111075, int_111078)
        # Adding element type (line 39)
        int_111079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 39), list_111075, int_111079)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 38), list_111074, list_111075)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_111080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        int_111081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 39), list_111080, int_111081)
        # Adding element type (line 40)
        int_111082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 39), list_111080, int_111082)
        # Adding element type (line 40)
        int_111083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 39), list_111080, int_111083)
        # Adding element type (line 40)
        int_111084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 39), list_111080, int_111084)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 38), list_111074, list_111080)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_111085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        int_111086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 39), list_111085, int_111086)
        # Adding element type (line 41)
        int_111087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 39), list_111085, int_111087)
        # Adding element type (line 41)
        int_111088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 39), list_111085, int_111088)
        # Adding element type (line 41)
        int_111089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 39), list_111085, int_111089)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 38), list_111074, list_111085)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_111090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        int_111091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 39), list_111090, int_111091)
        # Adding element type (line 42)
        int_111092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 39), list_111090, int_111092)
        # Adding element type (line 42)
        int_111093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 39), list_111090, int_111093)
        # Adding element type (line 42)
        int_111094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 39), list_111090, int_111094)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 38), list_111074, list_111090)
        
        # Processing the call keyword arguments (line 39)
        kwargs_111095 = {}
        # Getting the type of 'array' (line 39)
        array_111073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 32), 'array', False)
        # Calling array(args, kwargs) (line 39)
        array_call_result_111096 = invoke(stypy.reporting.localization.Localization(__file__, 39, 32), array_111073, *[list_111074], **kwargs_111095)
        
        # Processing the call keyword arguments (line 39)
        kwargs_111097 = {}
        # Getting the type of 'assert_equal' (line 39)
        assert_equal_111066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 39)
        assert_equal_call_result_111098 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), assert_equal_111066, *[tri_call_result_111072, array_call_result_111096], **kwargs_111097)
        
        
        # Call to assert_equal(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Call to tri(...): (line 43)
        # Processing the call arguments (line 43)
        int_111101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'int')
        # Processing the call keyword arguments (line 43)
        int_111102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 29), 'int')
        keyword_111103 = int_111102
        kwargs_111104 = {'k': keyword_111103}
        # Getting the type of 'tri' (line 43)
        tri_111100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'tri', False)
        # Calling tri(args, kwargs) (line 43)
        tri_call_result_111105 = invoke(stypy.reporting.localization.Localization(__file__, 43, 21), tri_111100, *[int_111101], **kwargs_111104)
        
        
        # Call to array(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_111107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_111108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        int_111109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 40), list_111108, int_111109)
        # Adding element type (line 43)
        int_111110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 40), list_111108, int_111110)
        # Adding element type (line 43)
        int_111111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 40), list_111108, int_111111)
        # Adding element type (line 43)
        int_111112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 40), list_111108, int_111112)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 39), list_111107, list_111108)
        # Adding element type (line 43)
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_111113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        # Adding element type (line 44)
        int_111114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 40), list_111113, int_111114)
        # Adding element type (line 44)
        int_111115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 40), list_111113, int_111115)
        # Adding element type (line 44)
        int_111116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 40), list_111113, int_111116)
        # Adding element type (line 44)
        int_111117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 40), list_111113, int_111117)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 39), list_111107, list_111113)
        # Adding element type (line 43)
        
        # Obtaining an instance of the builtin type 'list' (line 45)
        list_111118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 45)
        # Adding element type (line 45)
        int_111119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 40), list_111118, int_111119)
        # Adding element type (line 45)
        int_111120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 40), list_111118, int_111120)
        # Adding element type (line 45)
        int_111121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 40), list_111118, int_111121)
        # Adding element type (line 45)
        int_111122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 40), list_111118, int_111122)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 39), list_111107, list_111118)
        # Adding element type (line 43)
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_111123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        int_111124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 40), list_111123, int_111124)
        # Adding element type (line 46)
        int_111125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 40), list_111123, int_111125)
        # Adding element type (line 46)
        int_111126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 40), list_111123, int_111126)
        # Adding element type (line 46)
        int_111127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 40), list_111123, int_111127)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 39), list_111107, list_111123)
        
        # Processing the call keyword arguments (line 43)
        kwargs_111128 = {}
        # Getting the type of 'array' (line 43)
        array_111106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 33), 'array', False)
        # Calling array(args, kwargs) (line 43)
        array_call_result_111129 = invoke(stypy.reporting.localization.Localization(__file__, 43, 33), array_111106, *[list_111107], **kwargs_111128)
        
        # Processing the call keyword arguments (line 43)
        kwargs_111130 = {}
        # Getting the type of 'assert_equal' (line 43)
        assert_equal_111099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 43)
        assert_equal_call_result_111131 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), assert_equal_111099, *[tri_call_result_111105, array_call_result_111129], **kwargs_111130)
        
        
        # ################# End of 'test_diag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_diag' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_111132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111132)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_diag'
        return stypy_return_type_111132


    @norecursion
    def test_2d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_2d'
        module_type_store = module_type_store.open_function_context('test_2d', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTri.test_2d.__dict__.__setitem__('stypy_localization', localization)
        TestTri.test_2d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTri.test_2d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTri.test_2d.__dict__.__setitem__('stypy_function_name', 'TestTri.test_2d')
        TestTri.test_2d.__dict__.__setitem__('stypy_param_names_list', [])
        TestTri.test_2d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTri.test_2d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTri.test_2d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTri.test_2d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTri.test_2d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTri.test_2d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTri.test_2d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_2d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_2d(...)' code ##################

        
        # Call to assert_equal(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Call to tri(...): (line 49)
        # Processing the call arguments (line 49)
        int_111135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 25), 'int')
        int_111136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 27), 'int')
        # Processing the call keyword arguments (line 49)
        kwargs_111137 = {}
        # Getting the type of 'tri' (line 49)
        tri_111134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'tri', False)
        # Calling tri(args, kwargs) (line 49)
        tri_call_result_111138 = invoke(stypy.reporting.localization.Localization(__file__, 49, 21), tri_111134, *[int_111135, int_111136], **kwargs_111137)
        
        
        # Call to array(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_111140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_111141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        int_111142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 37), list_111141, int_111142)
        # Adding element type (line 49)
        int_111143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 37), list_111141, int_111143)
        # Adding element type (line 49)
        int_111144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 37), list_111141, int_111144)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 36), list_111140, list_111141)
        # Adding element type (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 50)
        list_111145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 50)
        # Adding element type (line 50)
        int_111146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 37), list_111145, int_111146)
        # Adding element type (line 50)
        int_111147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 37), list_111145, int_111147)
        # Adding element type (line 50)
        int_111148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 37), list_111145, int_111148)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 36), list_111140, list_111145)
        # Adding element type (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_111149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        int_111150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 37), list_111149, int_111150)
        # Adding element type (line 51)
        int_111151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 37), list_111149, int_111151)
        # Adding element type (line 51)
        int_111152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 37), list_111149, int_111152)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 36), list_111140, list_111149)
        # Adding element type (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 52)
        list_111153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 52)
        # Adding element type (line 52)
        int_111154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 37), list_111153, int_111154)
        # Adding element type (line 52)
        int_111155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 37), list_111153, int_111155)
        # Adding element type (line 52)
        int_111156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 37), list_111153, int_111156)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 36), list_111140, list_111153)
        
        # Processing the call keyword arguments (line 49)
        kwargs_111157 = {}
        # Getting the type of 'array' (line 49)
        array_111139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'array', False)
        # Calling array(args, kwargs) (line 49)
        array_call_result_111158 = invoke(stypy.reporting.localization.Localization(__file__, 49, 30), array_111139, *[list_111140], **kwargs_111157)
        
        # Processing the call keyword arguments (line 49)
        kwargs_111159 = {}
        # Getting the type of 'assert_equal' (line 49)
        assert_equal_111133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 49)
        assert_equal_call_result_111160 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assert_equal_111133, *[tri_call_result_111138, array_call_result_111158], **kwargs_111159)
        
        
        # Call to assert_equal(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to tri(...): (line 53)
        # Processing the call arguments (line 53)
        int_111163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'int')
        int_111164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 27), 'int')
        # Processing the call keyword arguments (line 53)
        kwargs_111165 = {}
        # Getting the type of 'tri' (line 53)
        tri_111162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'tri', False)
        # Calling tri(args, kwargs) (line 53)
        tri_call_result_111166 = invoke(stypy.reporting.localization.Localization(__file__, 53, 21), tri_111162, *[int_111163, int_111164], **kwargs_111165)
        
        
        # Call to array(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_111168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_111169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        int_111170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 37), list_111169, int_111170)
        # Adding element type (line 53)
        int_111171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 37), list_111169, int_111171)
        # Adding element type (line 53)
        int_111172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 37), list_111169, int_111172)
        # Adding element type (line 53)
        int_111173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 37), list_111169, int_111173)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 36), list_111168, list_111169)
        # Adding element type (line 53)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_111174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_111175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 37), list_111174, int_111175)
        # Adding element type (line 54)
        int_111176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 37), list_111174, int_111176)
        # Adding element type (line 54)
        int_111177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 37), list_111174, int_111177)
        # Adding element type (line 54)
        int_111178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 37), list_111174, int_111178)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 36), list_111168, list_111174)
        # Adding element type (line 53)
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_111179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_111180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 37), list_111179, int_111180)
        # Adding element type (line 55)
        int_111181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 37), list_111179, int_111181)
        # Adding element type (line 55)
        int_111182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 37), list_111179, int_111182)
        # Adding element type (line 55)
        int_111183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 37), list_111179, int_111183)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 36), list_111168, list_111179)
        
        # Processing the call keyword arguments (line 53)
        kwargs_111184 = {}
        # Getting the type of 'array' (line 53)
        array_111167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'array', False)
        # Calling array(args, kwargs) (line 53)
        array_call_result_111185 = invoke(stypy.reporting.localization.Localization(__file__, 53, 30), array_111167, *[list_111168], **kwargs_111184)
        
        # Processing the call keyword arguments (line 53)
        kwargs_111186 = {}
        # Getting the type of 'assert_equal' (line 53)
        assert_equal_111161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 53)
        assert_equal_call_result_111187 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assert_equal_111161, *[tri_call_result_111166, array_call_result_111185], **kwargs_111186)
        
        
        # ################# End of 'test_2d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_2d' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_111188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111188)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_2d'
        return stypy_return_type_111188


    @norecursion
    def test_diag2d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_diag2d'
        module_type_store = module_type_store.open_function_context('test_diag2d', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTri.test_diag2d.__dict__.__setitem__('stypy_localization', localization)
        TestTri.test_diag2d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTri.test_diag2d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTri.test_diag2d.__dict__.__setitem__('stypy_function_name', 'TestTri.test_diag2d')
        TestTri.test_diag2d.__dict__.__setitem__('stypy_param_names_list', [])
        TestTri.test_diag2d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTri.test_diag2d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTri.test_diag2d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTri.test_diag2d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTri.test_diag2d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTri.test_diag2d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTri.test_diag2d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_diag2d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_diag2d(...)' code ##################

        
        # Call to assert_equal(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to tri(...): (line 58)
        # Processing the call arguments (line 58)
        int_111191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'int')
        int_111192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'int')
        # Processing the call keyword arguments (line 58)
        int_111193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'int')
        keyword_111194 = int_111193
        kwargs_111195 = {'k': keyword_111194}
        # Getting the type of 'tri' (line 58)
        tri_111190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'tri', False)
        # Calling tri(args, kwargs) (line 58)
        tri_call_result_111196 = invoke(stypy.reporting.localization.Localization(__file__, 58, 21), tri_111190, *[int_111191, int_111192], **kwargs_111195)
        
        
        # Call to array(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_111198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_111199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        int_111200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 41), list_111199, int_111200)
        # Adding element type (line 58)
        int_111201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 41), list_111199, int_111201)
        # Adding element type (line 58)
        int_111202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 41), list_111199, int_111202)
        # Adding element type (line 58)
        int_111203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 41), list_111199, int_111203)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 40), list_111198, list_111199)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_111204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        int_111205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 41), list_111204, int_111205)
        # Adding element type (line 59)
        int_111206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 41), list_111204, int_111206)
        # Adding element type (line 59)
        int_111207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 41), list_111204, int_111207)
        # Adding element type (line 59)
        int_111208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 41), list_111204, int_111208)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 40), list_111198, list_111204)
        # Adding element type (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_111209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        int_111210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 41), list_111209, int_111210)
        # Adding element type (line 60)
        int_111211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 41), list_111209, int_111211)
        # Adding element type (line 60)
        int_111212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 41), list_111209, int_111212)
        # Adding element type (line 60)
        int_111213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 41), list_111209, int_111213)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 40), list_111198, list_111209)
        
        # Processing the call keyword arguments (line 58)
        kwargs_111214 = {}
        # Getting the type of 'array' (line 58)
        array_111197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'array', False)
        # Calling array(args, kwargs) (line 58)
        array_call_result_111215 = invoke(stypy.reporting.localization.Localization(__file__, 58, 34), array_111197, *[list_111198], **kwargs_111214)
        
        # Processing the call keyword arguments (line 58)
        kwargs_111216 = {}
        # Getting the type of 'assert_equal' (line 58)
        assert_equal_111189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 58)
        assert_equal_call_result_111217 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assert_equal_111189, *[tri_call_result_111196, array_call_result_111215], **kwargs_111216)
        
        
        # Call to assert_equal(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to tri(...): (line 61)
        # Processing the call arguments (line 61)
        int_111220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 25), 'int')
        int_111221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'int')
        # Processing the call keyword arguments (line 61)
        int_111222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 31), 'int')
        keyword_111223 = int_111222
        kwargs_111224 = {'k': keyword_111223}
        # Getting the type of 'tri' (line 61)
        tri_111219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'tri', False)
        # Calling tri(args, kwargs) (line 61)
        tri_call_result_111225 = invoke(stypy.reporting.localization.Localization(__file__, 61, 21), tri_111219, *[int_111220, int_111221], **kwargs_111224)
        
        
        # Call to array(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_111227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_111228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        int_111229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 42), list_111228, int_111229)
        # Adding element type (line 61)
        int_111230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 42), list_111228, int_111230)
        # Adding element type (line 61)
        int_111231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 42), list_111228, int_111231)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 41), list_111227, list_111228)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_111232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        int_111233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 42), list_111232, int_111233)
        # Adding element type (line 62)
        int_111234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 42), list_111232, int_111234)
        # Adding element type (line 62)
        int_111235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 42), list_111232, int_111235)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 41), list_111227, list_111232)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_111236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        int_111237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 42), list_111236, int_111237)
        # Adding element type (line 63)
        int_111238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 42), list_111236, int_111238)
        # Adding element type (line 63)
        int_111239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 42), list_111236, int_111239)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 41), list_111227, list_111236)
        # Adding element type (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_111240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        int_111241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 42), list_111240, int_111241)
        # Adding element type (line 64)
        int_111242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 42), list_111240, int_111242)
        # Adding element type (line 64)
        int_111243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 42), list_111240, int_111243)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 41), list_111227, list_111240)
        
        # Processing the call keyword arguments (line 61)
        kwargs_111244 = {}
        # Getting the type of 'array' (line 61)
        array_111226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 35), 'array', False)
        # Calling array(args, kwargs) (line 61)
        array_call_result_111245 = invoke(stypy.reporting.localization.Localization(__file__, 61, 35), array_111226, *[list_111227], **kwargs_111244)
        
        # Processing the call keyword arguments (line 61)
        kwargs_111246 = {}
        # Getting the type of 'assert_equal' (line 61)
        assert_equal_111218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 61)
        assert_equal_call_result_111247 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert_equal_111218, *[tri_call_result_111225, array_call_result_111245], **kwargs_111246)
        
        
        # ################# End of 'test_diag2d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_diag2d' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_111248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111248)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_diag2d'
        return stypy_return_type_111248


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 27, 0, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTri.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTri' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'TestTri', TestTri)
# Declaration of the 'TestTril' class

class TestTril(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTril.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestTril.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTril.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTril.test_basic.__dict__.__setitem__('stypy_function_name', 'TestTril.test_basic')
        TestTril.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestTril.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTril.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTril.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTril.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTril.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTril.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTril.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 69):
        
        # Call to astype(...): (line 69)
        # Processing the call arguments (line 69)
        str_111256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 36), 'str', 'l')
        # Processing the call keyword arguments (line 69)
        kwargs_111257 = {}
        int_111249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 13), 'int')
        
        # Call to get_mat(...): (line 69)
        # Processing the call arguments (line 69)
        int_111251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'int')
        # Processing the call keyword arguments (line 69)
        kwargs_111252 = {}
        # Getting the type of 'get_mat' (line 69)
        get_mat_111250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'get_mat', False)
        # Calling get_mat(args, kwargs) (line 69)
        get_mat_call_result_111253 = invoke(stypy.reporting.localization.Localization(__file__, 69, 17), get_mat_111250, *[int_111251], **kwargs_111252)
        
        # Applying the binary operator '*' (line 69)
        result_mul_111254 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '*', int_111249, get_mat_call_result_111253)
        
        # Obtaining the member 'astype' of a type (line 69)
        astype_111255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), result_mul_111254, 'astype')
        # Calling astype(args, kwargs) (line 69)
        astype_call_result_111258 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), astype_111255, *[str_111256], **kwargs_111257)
        
        # Assigning a type to the variable 'a' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'a', astype_call_result_111258)
        
        # Assigning a Call to a Name (line 70):
        
        # Call to copy(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_111261 = {}
        # Getting the type of 'a' (line 70)
        a_111259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'a', False)
        # Obtaining the member 'copy' of a type (line 70)
        copy_111260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), a_111259, 'copy')
        # Calling copy(args, kwargs) (line 70)
        copy_call_result_111262 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), copy_111260, *[], **kwargs_111261)
        
        # Assigning a type to the variable 'b' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'b', copy_call_result_111262)
        
        
        # Call to range(...): (line 71)
        # Processing the call arguments (line 71)
        int_111264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'int')
        # Processing the call keyword arguments (line 71)
        kwargs_111265 = {}
        # Getting the type of 'range' (line 71)
        range_111263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'range', False)
        # Calling range(args, kwargs) (line 71)
        range_call_result_111266 = invoke(stypy.reporting.localization.Localization(__file__, 71, 17), range_111263, *[int_111264], **kwargs_111265)
        
        # Testing the type of a for loop iterable (line 71)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 71, 8), range_call_result_111266)
        # Getting the type of the for loop variable (line 71)
        for_loop_var_111267 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 71, 8), range_call_result_111266)
        # Assigning a type to the variable 'k' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'k', for_loop_var_111267)
        # SSA begins for a for statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'k' (line 72)
        k_111269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'k', False)
        int_111270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'int')
        # Applying the binary operator '+' (line 72)
        result_add_111271 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 27), '+', k_111269, int_111270)
        
        int_111272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 31), 'int')
        # Processing the call keyword arguments (line 72)
        kwargs_111273 = {}
        # Getting the type of 'range' (line 72)
        range_111268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'range', False)
        # Calling range(args, kwargs) (line 72)
        range_call_result_111274 = invoke(stypy.reporting.localization.Localization(__file__, 72, 21), range_111268, *[result_add_111271, int_111272], **kwargs_111273)
        
        # Testing the type of a for loop iterable (line 72)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 72, 12), range_call_result_111274)
        # Getting the type of the for loop variable (line 72)
        for_loop_var_111275 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 72, 12), range_call_result_111274)
        # Assigning a type to the variable 'l' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'l', for_loop_var_111275)
        # SSA begins for a for statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 73):
        int_111276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 25), 'int')
        # Getting the type of 'b' (line 73)
        b_111277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'b')
        
        # Obtaining an instance of the builtin type 'tuple' (line 73)
        tuple_111278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 73)
        # Adding element type (line 73)
        # Getting the type of 'k' (line 73)
        k_111279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 18), tuple_111278, k_111279)
        # Adding element type (line 73)
        # Getting the type of 'l' (line 73)
        l_111280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 18), tuple_111278, l_111280)
        
        # Storing an element on a container (line 73)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), b_111277, (tuple_111278, int_111276))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Call to tril(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'a' (line 74)
        a_111283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'a', False)
        # Processing the call keyword arguments (line 74)
        kwargs_111284 = {}
        # Getting the type of 'tril' (line 74)
        tril_111282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 21), 'tril', False)
        # Calling tril(args, kwargs) (line 74)
        tril_call_result_111285 = invoke(stypy.reporting.localization.Localization(__file__, 74, 21), tril_111282, *[a_111283], **kwargs_111284)
        
        # Getting the type of 'b' (line 74)
        b_111286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 29), 'b', False)
        # Processing the call keyword arguments (line 74)
        kwargs_111287 = {}
        # Getting the type of 'assert_equal' (line 74)
        assert_equal_111281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 74)
        assert_equal_call_result_111288 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), assert_equal_111281, *[tril_call_result_111285, b_111286], **kwargs_111287)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_111289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111289)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_111289


    @norecursion
    def test_diag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_diag'
        module_type_store = module_type_store.open_function_context('test_diag', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTril.test_diag.__dict__.__setitem__('stypy_localization', localization)
        TestTril.test_diag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTril.test_diag.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTril.test_diag.__dict__.__setitem__('stypy_function_name', 'TestTril.test_diag')
        TestTril.test_diag.__dict__.__setitem__('stypy_param_names_list', [])
        TestTril.test_diag.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTril.test_diag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTril.test_diag.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTril.test_diag.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTril.test_diag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTril.test_diag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTril.test_diag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_diag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_diag(...)' code ##################

        
        # Assigning a Call to a Name (line 77):
        
        # Call to astype(...): (line 77)
        # Processing the call arguments (line 77)
        str_111297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 36), 'str', 'f')
        # Processing the call keyword arguments (line 77)
        kwargs_111298 = {}
        int_111290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 13), 'int')
        
        # Call to get_mat(...): (line 77)
        # Processing the call arguments (line 77)
        int_111292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'int')
        # Processing the call keyword arguments (line 77)
        kwargs_111293 = {}
        # Getting the type of 'get_mat' (line 77)
        get_mat_111291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'get_mat', False)
        # Calling get_mat(args, kwargs) (line 77)
        get_mat_call_result_111294 = invoke(stypy.reporting.localization.Localization(__file__, 77, 17), get_mat_111291, *[int_111292], **kwargs_111293)
        
        # Applying the binary operator '*' (line 77)
        result_mul_111295 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 13), '*', int_111290, get_mat_call_result_111294)
        
        # Obtaining the member 'astype' of a type (line 77)
        astype_111296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), result_mul_111295, 'astype')
        # Calling astype(args, kwargs) (line 77)
        astype_call_result_111299 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), astype_111296, *[str_111297], **kwargs_111298)
        
        # Assigning a type to the variable 'a' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'a', astype_call_result_111299)
        
        # Assigning a Call to a Name (line 78):
        
        # Call to copy(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_111302 = {}
        # Getting the type of 'a' (line 78)
        a_111300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'a', False)
        # Obtaining the member 'copy' of a type (line 78)
        copy_111301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), a_111300, 'copy')
        # Calling copy(args, kwargs) (line 78)
        copy_call_result_111303 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), copy_111301, *[], **kwargs_111302)
        
        # Assigning a type to the variable 'b' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'b', copy_call_result_111303)
        
        
        # Call to range(...): (line 79)
        # Processing the call arguments (line 79)
        int_111305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 23), 'int')
        # Processing the call keyword arguments (line 79)
        kwargs_111306 = {}
        # Getting the type of 'range' (line 79)
        range_111304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'range', False)
        # Calling range(args, kwargs) (line 79)
        range_call_result_111307 = invoke(stypy.reporting.localization.Localization(__file__, 79, 17), range_111304, *[int_111305], **kwargs_111306)
        
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 8), range_call_result_111307)
        # Getting the type of the for loop variable (line 79)
        for_loop_var_111308 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 8), range_call_result_111307)
        # Assigning a type to the variable 'k' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'k', for_loop_var_111308)
        # SSA begins for a for statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'k' (line 80)
        k_111310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'k', False)
        int_111311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 29), 'int')
        # Applying the binary operator '+' (line 80)
        result_add_111312 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 27), '+', k_111310, int_111311)
        
        int_111313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 31), 'int')
        # Processing the call keyword arguments (line 80)
        kwargs_111314 = {}
        # Getting the type of 'range' (line 80)
        range_111309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 21), 'range', False)
        # Calling range(args, kwargs) (line 80)
        range_call_result_111315 = invoke(stypy.reporting.localization.Localization(__file__, 80, 21), range_111309, *[result_add_111312, int_111313], **kwargs_111314)
        
        # Testing the type of a for loop iterable (line 80)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 80, 12), range_call_result_111315)
        # Getting the type of the for loop variable (line 80)
        for_loop_var_111316 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 80, 12), range_call_result_111315)
        # Assigning a type to the variable 'l' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'l', for_loop_var_111316)
        # SSA begins for a for statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 81):
        int_111317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'int')
        # Getting the type of 'b' (line 81)
        b_111318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'b')
        
        # Obtaining an instance of the builtin type 'tuple' (line 81)
        tuple_111319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 81)
        # Adding element type (line 81)
        # Getting the type of 'k' (line 81)
        k_111320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 18), 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 18), tuple_111319, k_111320)
        # Adding element type (line 81)
        # Getting the type of 'l' (line 81)
        l_111321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 18), tuple_111319, l_111321)
        
        # Storing an element on a container (line 81)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 16), b_111318, (tuple_111319, int_111317))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Call to tril(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'a' (line 82)
        a_111324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'a', False)
        # Processing the call keyword arguments (line 82)
        int_111325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 30), 'int')
        keyword_111326 = int_111325
        kwargs_111327 = {'k': keyword_111326}
        # Getting the type of 'tril' (line 82)
        tril_111323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'tril', False)
        # Calling tril(args, kwargs) (line 82)
        tril_call_result_111328 = invoke(stypy.reporting.localization.Localization(__file__, 82, 21), tril_111323, *[a_111324], **kwargs_111327)
        
        # Getting the type of 'b' (line 82)
        b_111329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 33), 'b', False)
        # Processing the call keyword arguments (line 82)
        kwargs_111330 = {}
        # Getting the type of 'assert_equal' (line 82)
        assert_equal_111322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 82)
        assert_equal_call_result_111331 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), assert_equal_111322, *[tril_call_result_111328, b_111329], **kwargs_111330)
        
        
        # Assigning a Call to a Name (line 83):
        
        # Call to copy(...): (line 83)
        # Processing the call keyword arguments (line 83)
        kwargs_111334 = {}
        # Getting the type of 'a' (line 83)
        a_111332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'a', False)
        # Obtaining the member 'copy' of a type (line 83)
        copy_111333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), a_111332, 'copy')
        # Calling copy(args, kwargs) (line 83)
        copy_call_result_111335 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), copy_111333, *[], **kwargs_111334)
        
        # Assigning a type to the variable 'b' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'b', copy_call_result_111335)
        
        
        # Call to range(...): (line 84)
        # Processing the call arguments (line 84)
        int_111337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'int')
        # Processing the call keyword arguments (line 84)
        kwargs_111338 = {}
        # Getting the type of 'range' (line 84)
        range_111336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'range', False)
        # Calling range(args, kwargs) (line 84)
        range_call_result_111339 = invoke(stypy.reporting.localization.Localization(__file__, 84, 17), range_111336, *[int_111337], **kwargs_111338)
        
        # Testing the type of a for loop iterable (line 84)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 8), range_call_result_111339)
        # Getting the type of the for loop variable (line 84)
        for_loop_var_111340 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 8), range_call_result_111339)
        # Assigning a type to the variable 'k' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'k', for_loop_var_111340)
        # SSA begins for a for statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to max(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Obtaining an instance of the builtin type 'tuple' (line 85)
        tuple_111343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 85)
        # Adding element type (line 85)
        # Getting the type of 'k' (line 85)
        k_111344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 32), 'k', False)
        int_111345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 34), 'int')
        # Applying the binary operator '-' (line 85)
        result_sub_111346 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 32), '-', k_111344, int_111345)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 32), tuple_111343, result_sub_111346)
        # Adding element type (line 85)
        int_111347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 32), tuple_111343, int_111347)
        
        # Processing the call keyword arguments (line 85)
        kwargs_111348 = {}
        # Getting the type of 'max' (line 85)
        max_111342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'max', False)
        # Calling max(args, kwargs) (line 85)
        max_call_result_111349 = invoke(stypy.reporting.localization.Localization(__file__, 85, 27), max_111342, *[tuple_111343], **kwargs_111348)
        
        int_111350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 40), 'int')
        # Processing the call keyword arguments (line 85)
        kwargs_111351 = {}
        # Getting the type of 'range' (line 85)
        range_111341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'range', False)
        # Calling range(args, kwargs) (line 85)
        range_call_result_111352 = invoke(stypy.reporting.localization.Localization(__file__, 85, 21), range_111341, *[max_call_result_111349, int_111350], **kwargs_111351)
        
        # Testing the type of a for loop iterable (line 85)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 85, 12), range_call_result_111352)
        # Getting the type of the for loop variable (line 85)
        for_loop_var_111353 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 85, 12), range_call_result_111352)
        # Assigning a type to the variable 'l' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'l', for_loop_var_111353)
        # SSA begins for a for statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 86):
        int_111354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'int')
        # Getting the type of 'b' (line 86)
        b_111355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'b')
        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_111356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        # Getting the type of 'k' (line 86)
        k_111357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 18), tuple_111356, k_111357)
        # Adding element type (line 86)
        # Getting the type of 'l' (line 86)
        l_111358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 18), tuple_111356, l_111358)
        
        # Storing an element on a container (line 86)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 16), b_111355, (tuple_111356, int_111354))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to tril(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'a' (line 87)
        a_111361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'a', False)
        # Processing the call keyword arguments (line 87)
        int_111362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 30), 'int')
        keyword_111363 = int_111362
        kwargs_111364 = {'k': keyword_111363}
        # Getting the type of 'tril' (line 87)
        tril_111360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'tril', False)
        # Calling tril(args, kwargs) (line 87)
        tril_call_result_111365 = invoke(stypy.reporting.localization.Localization(__file__, 87, 21), tril_111360, *[a_111361], **kwargs_111364)
        
        # Getting the type of 'b' (line 87)
        b_111366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 34), 'b', False)
        # Processing the call keyword arguments (line 87)
        kwargs_111367 = {}
        # Getting the type of 'assert_equal' (line 87)
        assert_equal_111359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 87)
        assert_equal_call_result_111368 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), assert_equal_111359, *[tril_call_result_111365, b_111366], **kwargs_111367)
        
        
        # ################# End of 'test_diag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_diag' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_111369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111369)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_diag'
        return stypy_return_type_111369


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 67, 0, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTril.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTril' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'TestTril', TestTril)
# Declaration of the 'TestTriu' class

class TestTriu(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTriu.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestTriu.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTriu.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTriu.test_basic.__dict__.__setitem__('stypy_function_name', 'TestTriu.test_basic')
        TestTriu.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestTriu.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTriu.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTriu.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTriu.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTriu.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTriu.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTriu.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 92):
        
        # Call to astype(...): (line 92)
        # Processing the call arguments (line 92)
        str_111377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 36), 'str', 'l')
        # Processing the call keyword arguments (line 92)
        kwargs_111378 = {}
        int_111370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 13), 'int')
        
        # Call to get_mat(...): (line 92)
        # Processing the call arguments (line 92)
        int_111372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'int')
        # Processing the call keyword arguments (line 92)
        kwargs_111373 = {}
        # Getting the type of 'get_mat' (line 92)
        get_mat_111371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'get_mat', False)
        # Calling get_mat(args, kwargs) (line 92)
        get_mat_call_result_111374 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), get_mat_111371, *[int_111372], **kwargs_111373)
        
        # Applying the binary operator '*' (line 92)
        result_mul_111375 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 13), '*', int_111370, get_mat_call_result_111374)
        
        # Obtaining the member 'astype' of a type (line 92)
        astype_111376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), result_mul_111375, 'astype')
        # Calling astype(args, kwargs) (line 92)
        astype_call_result_111379 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), astype_111376, *[str_111377], **kwargs_111378)
        
        # Assigning a type to the variable 'a' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'a', astype_call_result_111379)
        
        # Assigning a Call to a Name (line 93):
        
        # Call to copy(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_111382 = {}
        # Getting the type of 'a' (line 93)
        a_111380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'a', False)
        # Obtaining the member 'copy' of a type (line 93)
        copy_111381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), a_111380, 'copy')
        # Calling copy(args, kwargs) (line 93)
        copy_call_result_111383 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), copy_111381, *[], **kwargs_111382)
        
        # Assigning a type to the variable 'b' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'b', copy_call_result_111383)
        
        
        # Call to range(...): (line 94)
        # Processing the call arguments (line 94)
        int_111385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 23), 'int')
        # Processing the call keyword arguments (line 94)
        kwargs_111386 = {}
        # Getting the type of 'range' (line 94)
        range_111384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'range', False)
        # Calling range(args, kwargs) (line 94)
        range_call_result_111387 = invoke(stypy.reporting.localization.Localization(__file__, 94, 17), range_111384, *[int_111385], **kwargs_111386)
        
        # Testing the type of a for loop iterable (line 94)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 94, 8), range_call_result_111387)
        # Getting the type of the for loop variable (line 94)
        for_loop_var_111388 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 94, 8), range_call_result_111387)
        # Assigning a type to the variable 'k' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'k', for_loop_var_111388)
        # SSA begins for a for statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'k' (line 95)
        k_111390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 27), 'k', False)
        int_111391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 29), 'int')
        # Applying the binary operator '+' (line 95)
        result_add_111392 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 27), '+', k_111390, int_111391)
        
        int_111393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 31), 'int')
        # Processing the call keyword arguments (line 95)
        kwargs_111394 = {}
        # Getting the type of 'range' (line 95)
        range_111389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'range', False)
        # Calling range(args, kwargs) (line 95)
        range_call_result_111395 = invoke(stypy.reporting.localization.Localization(__file__, 95, 21), range_111389, *[result_add_111392, int_111393], **kwargs_111394)
        
        # Testing the type of a for loop iterable (line 95)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 95, 12), range_call_result_111395)
        # Getting the type of the for loop variable (line 95)
        for_loop_var_111396 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 95, 12), range_call_result_111395)
        # Assigning a type to the variable 'l' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'l', for_loop_var_111396)
        # SSA begins for a for statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 96):
        int_111397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 25), 'int')
        # Getting the type of 'b' (line 96)
        b_111398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'b')
        
        # Obtaining an instance of the builtin type 'tuple' (line 96)
        tuple_111399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 96)
        # Adding element type (line 96)
        # Getting the type of 'l' (line 96)
        l_111400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 18), tuple_111399, l_111400)
        # Adding element type (line 96)
        # Getting the type of 'k' (line 96)
        k_111401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 18), tuple_111399, k_111401)
        
        # Storing an element on a container (line 96)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 16), b_111398, (tuple_111399, int_111397))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to triu(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'a' (line 97)
        a_111404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'a', False)
        # Processing the call keyword arguments (line 97)
        kwargs_111405 = {}
        # Getting the type of 'triu' (line 97)
        triu_111403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'triu', False)
        # Calling triu(args, kwargs) (line 97)
        triu_call_result_111406 = invoke(stypy.reporting.localization.Localization(__file__, 97, 21), triu_111403, *[a_111404], **kwargs_111405)
        
        # Getting the type of 'b' (line 97)
        b_111407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'b', False)
        # Processing the call keyword arguments (line 97)
        kwargs_111408 = {}
        # Getting the type of 'assert_equal' (line 97)
        assert_equal_111402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 97)
        assert_equal_call_result_111409 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assert_equal_111402, *[triu_call_result_111406, b_111407], **kwargs_111408)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_111410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111410)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_111410


    @norecursion
    def test_diag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_diag'
        module_type_store = module_type_store.open_function_context('test_diag', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTriu.test_diag.__dict__.__setitem__('stypy_localization', localization)
        TestTriu.test_diag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTriu.test_diag.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTriu.test_diag.__dict__.__setitem__('stypy_function_name', 'TestTriu.test_diag')
        TestTriu.test_diag.__dict__.__setitem__('stypy_param_names_list', [])
        TestTriu.test_diag.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTriu.test_diag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTriu.test_diag.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTriu.test_diag.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTriu.test_diag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTriu.test_diag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTriu.test_diag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_diag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_diag(...)' code ##################

        
        # Assigning a Call to a Name (line 100):
        
        # Call to astype(...): (line 100)
        # Processing the call arguments (line 100)
        str_111418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 36), 'str', 'f')
        # Processing the call keyword arguments (line 100)
        kwargs_111419 = {}
        int_111411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 13), 'int')
        
        # Call to get_mat(...): (line 100)
        # Processing the call arguments (line 100)
        int_111413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 25), 'int')
        # Processing the call keyword arguments (line 100)
        kwargs_111414 = {}
        # Getting the type of 'get_mat' (line 100)
        get_mat_111412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 'get_mat', False)
        # Calling get_mat(args, kwargs) (line 100)
        get_mat_call_result_111415 = invoke(stypy.reporting.localization.Localization(__file__, 100, 17), get_mat_111412, *[int_111413], **kwargs_111414)
        
        # Applying the binary operator '*' (line 100)
        result_mul_111416 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 13), '*', int_111411, get_mat_call_result_111415)
        
        # Obtaining the member 'astype' of a type (line 100)
        astype_111417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), result_mul_111416, 'astype')
        # Calling astype(args, kwargs) (line 100)
        astype_call_result_111420 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), astype_111417, *[str_111418], **kwargs_111419)
        
        # Assigning a type to the variable 'a' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'a', astype_call_result_111420)
        
        # Assigning a Call to a Name (line 101):
        
        # Call to copy(...): (line 101)
        # Processing the call keyword arguments (line 101)
        kwargs_111423 = {}
        # Getting the type of 'a' (line 101)
        a_111421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'a', False)
        # Obtaining the member 'copy' of a type (line 101)
        copy_111422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), a_111421, 'copy')
        # Calling copy(args, kwargs) (line 101)
        copy_call_result_111424 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), copy_111422, *[], **kwargs_111423)
        
        # Assigning a type to the variable 'b' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'b', copy_call_result_111424)
        
        
        # Call to range(...): (line 102)
        # Processing the call arguments (line 102)
        int_111426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'int')
        # Processing the call keyword arguments (line 102)
        kwargs_111427 = {}
        # Getting the type of 'range' (line 102)
        range_111425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'range', False)
        # Calling range(args, kwargs) (line 102)
        range_call_result_111428 = invoke(stypy.reporting.localization.Localization(__file__, 102, 17), range_111425, *[int_111426], **kwargs_111427)
        
        # Testing the type of a for loop iterable (line 102)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 102, 8), range_call_result_111428)
        # Getting the type of the for loop variable (line 102)
        for_loop_var_111429 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 102, 8), range_call_result_111428)
        # Assigning a type to the variable 'k' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'k', for_loop_var_111429)
        # SSA begins for a for statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to max(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_111432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        # Getting the type of 'k' (line 103)
        k_111433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 32), 'k', False)
        int_111434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 34), 'int')
        # Applying the binary operator '-' (line 103)
        result_sub_111435 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 32), '-', k_111433, int_111434)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 32), tuple_111432, result_sub_111435)
        # Adding element type (line 103)
        int_111436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 32), tuple_111432, int_111436)
        
        # Processing the call keyword arguments (line 103)
        kwargs_111437 = {}
        # Getting the type of 'max' (line 103)
        max_111431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'max', False)
        # Calling max(args, kwargs) (line 103)
        max_call_result_111438 = invoke(stypy.reporting.localization.Localization(__file__, 103, 27), max_111431, *[tuple_111432], **kwargs_111437)
        
        int_111439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 40), 'int')
        # Processing the call keyword arguments (line 103)
        kwargs_111440 = {}
        # Getting the type of 'range' (line 103)
        range_111430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 21), 'range', False)
        # Calling range(args, kwargs) (line 103)
        range_call_result_111441 = invoke(stypy.reporting.localization.Localization(__file__, 103, 21), range_111430, *[max_call_result_111438, int_111439], **kwargs_111440)
        
        # Testing the type of a for loop iterable (line 103)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 103, 12), range_call_result_111441)
        # Getting the type of the for loop variable (line 103)
        for_loop_var_111442 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 103, 12), range_call_result_111441)
        # Assigning a type to the variable 'l' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'l', for_loop_var_111442)
        # SSA begins for a for statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 104):
        int_111443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 25), 'int')
        # Getting the type of 'b' (line 104)
        b_111444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'b')
        
        # Obtaining an instance of the builtin type 'tuple' (line 104)
        tuple_111445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 104)
        # Adding element type (line 104)
        # Getting the type of 'l' (line 104)
        l_111446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 18), tuple_111445, l_111446)
        # Adding element type (line 104)
        # Getting the type of 'k' (line 104)
        k_111447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 18), tuple_111445, k_111447)
        
        # Storing an element on a container (line 104)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 16), b_111444, (tuple_111445, int_111443))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to triu(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'a' (line 105)
        a_111450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'a', False)
        # Processing the call keyword arguments (line 105)
        int_111451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 30), 'int')
        keyword_111452 = int_111451
        kwargs_111453 = {'k': keyword_111452}
        # Getting the type of 'triu' (line 105)
        triu_111449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'triu', False)
        # Calling triu(args, kwargs) (line 105)
        triu_call_result_111454 = invoke(stypy.reporting.localization.Localization(__file__, 105, 21), triu_111449, *[a_111450], **kwargs_111453)
        
        # Getting the type of 'b' (line 105)
        b_111455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'b', False)
        # Processing the call keyword arguments (line 105)
        kwargs_111456 = {}
        # Getting the type of 'assert_equal' (line 105)
        assert_equal_111448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 105)
        assert_equal_call_result_111457 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), assert_equal_111448, *[triu_call_result_111454, b_111455], **kwargs_111456)
        
        
        # Assigning a Call to a Name (line 106):
        
        # Call to copy(...): (line 106)
        # Processing the call keyword arguments (line 106)
        kwargs_111460 = {}
        # Getting the type of 'a' (line 106)
        a_111458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'a', False)
        # Obtaining the member 'copy' of a type (line 106)
        copy_111459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), a_111458, 'copy')
        # Calling copy(args, kwargs) (line 106)
        copy_call_result_111461 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), copy_111459, *[], **kwargs_111460)
        
        # Assigning a type to the variable 'b' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'b', copy_call_result_111461)
        
        
        # Call to range(...): (line 107)
        # Processing the call arguments (line 107)
        int_111463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 23), 'int')
        # Processing the call keyword arguments (line 107)
        kwargs_111464 = {}
        # Getting the type of 'range' (line 107)
        range_111462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'range', False)
        # Calling range(args, kwargs) (line 107)
        range_call_result_111465 = invoke(stypy.reporting.localization.Localization(__file__, 107, 17), range_111462, *[int_111463], **kwargs_111464)
        
        # Testing the type of a for loop iterable (line 107)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 107, 8), range_call_result_111465)
        # Getting the type of the for loop variable (line 107)
        for_loop_var_111466 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 107, 8), range_call_result_111465)
        # Assigning a type to the variable 'k' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'k', for_loop_var_111466)
        # SSA begins for a for statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'k' (line 108)
        k_111468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'k', False)
        int_111469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'int')
        # Applying the binary operator '+' (line 108)
        result_add_111470 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 27), '+', k_111468, int_111469)
        
        int_111471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 31), 'int')
        # Processing the call keyword arguments (line 108)
        kwargs_111472 = {}
        # Getting the type of 'range' (line 108)
        range_111467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'range', False)
        # Calling range(args, kwargs) (line 108)
        range_call_result_111473 = invoke(stypy.reporting.localization.Localization(__file__, 108, 21), range_111467, *[result_add_111470, int_111471], **kwargs_111472)
        
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 12), range_call_result_111473)
        # Getting the type of the for loop variable (line 108)
        for_loop_var_111474 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 12), range_call_result_111473)
        # Assigning a type to the variable 'l' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'l', for_loop_var_111474)
        # SSA begins for a for statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 109):
        int_111475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'int')
        # Getting the type of 'b' (line 109)
        b_111476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'b')
        
        # Obtaining an instance of the builtin type 'tuple' (line 109)
        tuple_111477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 109)
        # Adding element type (line 109)
        # Getting the type of 'l' (line 109)
        l_111478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), tuple_111477, l_111478)
        # Adding element type (line 109)
        # Getting the type of 'k' (line 109)
        k_111479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'k')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), tuple_111477, k_111479)
        
        # Storing an element on a container (line 109)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 16), b_111476, (tuple_111477, int_111475))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to triu(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'a' (line 110)
        a_111482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'a', False)
        # Processing the call keyword arguments (line 110)
        int_111483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
        keyword_111484 = int_111483
        kwargs_111485 = {'k': keyword_111484}
        # Getting the type of 'triu' (line 110)
        triu_111481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'triu', False)
        # Calling triu(args, kwargs) (line 110)
        triu_call_result_111486 = invoke(stypy.reporting.localization.Localization(__file__, 110, 21), triu_111481, *[a_111482], **kwargs_111485)
        
        # Getting the type of 'b' (line 110)
        b_111487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 34), 'b', False)
        # Processing the call keyword arguments (line 110)
        kwargs_111488 = {}
        # Getting the type of 'assert_equal' (line 110)
        assert_equal_111480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 110)
        assert_equal_call_result_111489 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), assert_equal_111480, *[triu_call_result_111486, b_111487], **kwargs_111488)
        
        
        # ################# End of 'test_diag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_diag' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_111490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111490)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_diag'
        return stypy_return_type_111490


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 90, 0, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTriu.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTriu' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'TestTriu', TestTriu)
# Declaration of the 'TestToeplitz' class

class TestToeplitz(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestToeplitz.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestToeplitz.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestToeplitz.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestToeplitz.test_basic.__dict__.__setitem__('stypy_function_name', 'TestToeplitz.test_basic')
        TestToeplitz.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestToeplitz.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestToeplitz.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestToeplitz.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestToeplitz.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestToeplitz.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestToeplitz.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestToeplitz.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 116):
        
        # Call to toeplitz(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_111492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        int_111493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 21), list_111492, int_111493)
        # Adding element type (line 116)
        int_111494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 21), list_111492, int_111494)
        # Adding element type (line 116)
        int_111495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 21), list_111492, int_111495)
        
        # Processing the call keyword arguments (line 116)
        kwargs_111496 = {}
        # Getting the type of 'toeplitz' (line 116)
        toeplitz_111491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'toeplitz', False)
        # Calling toeplitz(args, kwargs) (line 116)
        toeplitz_call_result_111497 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), toeplitz_111491, *[list_111492], **kwargs_111496)
        
        # Assigning a type to the variable 'y' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'y', toeplitz_call_result_111497)
        
        # Call to assert_array_equal(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'y' (line 117)
        y_111499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'y', False)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_111500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_111501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        int_111502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 30), list_111501, int_111502)
        # Adding element type (line 117)
        int_111503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 30), list_111501, int_111503)
        # Adding element type (line 117)
        int_111504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 30), list_111501, int_111504)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 29), list_111500, list_111501)
        # Adding element type (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_111505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        int_111506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_111505, int_111506)
        # Adding element type (line 117)
        int_111507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_111505, int_111507)
        # Adding element type (line 117)
        int_111508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_111505, int_111508)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 29), list_111500, list_111505)
        # Adding element type (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_111509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        int_111510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 46), list_111509, int_111510)
        # Adding element type (line 117)
        int_111511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 46), list_111509, int_111511)
        # Adding element type (line 117)
        int_111512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 46), list_111509, int_111512)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 29), list_111500, list_111509)
        
        # Processing the call keyword arguments (line 117)
        kwargs_111513 = {}
        # Getting the type of 'assert_array_equal' (line 117)
        assert_array_equal_111498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 117)
        assert_array_equal_call_result_111514 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), assert_array_equal_111498, *[y_111499, list_111500], **kwargs_111513)
        
        
        # Assigning a Call to a Name (line 118):
        
        # Call to toeplitz(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_111516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        # Adding element type (line 118)
        int_111517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 21), list_111516, int_111517)
        # Adding element type (line 118)
        int_111518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 21), list_111516, int_111518)
        # Adding element type (line 118)
        int_111519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 21), list_111516, int_111519)
        
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_111520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        # Adding element type (line 118)
        int_111521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 29), list_111520, int_111521)
        # Adding element type (line 118)
        int_111522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 29), list_111520, int_111522)
        # Adding element type (line 118)
        int_111523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 29), list_111520, int_111523)
        
        # Processing the call keyword arguments (line 118)
        kwargs_111524 = {}
        # Getting the type of 'toeplitz' (line 118)
        toeplitz_111515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'toeplitz', False)
        # Calling toeplitz(args, kwargs) (line 118)
        toeplitz_call_result_111525 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), toeplitz_111515, *[list_111516, list_111520], **kwargs_111524)
        
        # Assigning a type to the variable 'y' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'y', toeplitz_call_result_111525)
        
        # Call to assert_array_equal(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'y' (line 119)
        y_111527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'y', False)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_111528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_111529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_111530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 30), list_111529, int_111530)
        # Adding element type (line 119)
        int_111531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 30), list_111529, int_111531)
        # Adding element type (line 119)
        int_111532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 30), list_111529, int_111532)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 29), list_111528, list_111529)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_111533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_111534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 38), list_111533, int_111534)
        # Adding element type (line 119)
        int_111535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 38), list_111533, int_111535)
        # Adding element type (line 119)
        int_111536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 38), list_111533, int_111536)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 29), list_111528, list_111533)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_111537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_111538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 46), list_111537, int_111538)
        # Adding element type (line 119)
        int_111539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 46), list_111537, int_111539)
        # Adding element type (line 119)
        int_111540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 46), list_111537, int_111540)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 29), list_111528, list_111537)
        
        # Processing the call keyword arguments (line 119)
        kwargs_111541 = {}
        # Getting the type of 'assert_array_equal' (line 119)
        assert_array_equal_111526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 119)
        assert_array_equal_call_result_111542 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), assert_array_equal_111526, *[y_111527, list_111528], **kwargs_111541)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_111543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111543)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_111543


    @norecursion
    def test_complex_01(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_complex_01'
        module_type_store = module_type_store.open_function_context('test_complex_01', 121, 4, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestToeplitz.test_complex_01.__dict__.__setitem__('stypy_localization', localization)
        TestToeplitz.test_complex_01.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestToeplitz.test_complex_01.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestToeplitz.test_complex_01.__dict__.__setitem__('stypy_function_name', 'TestToeplitz.test_complex_01')
        TestToeplitz.test_complex_01.__dict__.__setitem__('stypy_param_names_list', [])
        TestToeplitz.test_complex_01.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestToeplitz.test_complex_01.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestToeplitz.test_complex_01.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestToeplitz.test_complex_01.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestToeplitz.test_complex_01.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestToeplitz.test_complex_01.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestToeplitz.test_complex_01', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_complex_01', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_complex_01(...)' code ##################

        
        # Assigning a BinOp to a Name (line 122):
        float_111544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 16), 'float')
        
        # Call to arange(...): (line 122)
        # Processing the call arguments (line 122)
        float_111546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 29), 'float')
        # Processing the call keyword arguments (line 122)
        kwargs_111547 = {}
        # Getting the type of 'arange' (line 122)
        arange_111545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'arange', False)
        # Calling arange(args, kwargs) (line 122)
        arange_call_result_111548 = invoke(stypy.reporting.localization.Localization(__file__, 122, 22), arange_111545, *[float_111546], **kwargs_111547)
        
        # Applying the binary operator '+' (line 122)
        result_add_111549 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 16), '+', float_111544, arange_call_result_111548)
        
        float_111550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 38), 'float')
        complex_111551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 44), 'complex')
        # Applying the binary operator '+' (line 122)
        result_add_111552 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 38), '+', float_111550, complex_111551)
        
        # Applying the binary operator '*' (line 122)
        result_mul_111553 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 15), '*', result_add_111549, result_add_111552)
        
        # Assigning a type to the variable 'data' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'data', result_mul_111553)
        
        # Assigning a Call to a Name (line 123):
        
        # Call to copy(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'data' (line 123)
        data_111555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'data', False)
        # Processing the call keyword arguments (line 123)
        kwargs_111556 = {}
        # Getting the type of 'copy' (line 123)
        copy_111554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'copy', False)
        # Calling copy(args, kwargs) (line 123)
        copy_call_result_111557 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), copy_111554, *[data_111555], **kwargs_111556)
        
        # Assigning a type to the variable 'x' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'x', copy_call_result_111557)
        
        # Assigning a Call to a Name (line 124):
        
        # Call to toeplitz(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'x' (line 124)
        x_111559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'x', False)
        # Processing the call keyword arguments (line 124)
        kwargs_111560 = {}
        # Getting the type of 'toeplitz' (line 124)
        toeplitz_111558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'toeplitz', False)
        # Calling toeplitz(args, kwargs) (line 124)
        toeplitz_call_result_111561 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), toeplitz_111558, *[x_111559], **kwargs_111560)
        
        # Assigning a type to the variable 't' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 't', toeplitz_call_result_111561)
        
        # Call to assert_array_equal(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'x' (line 126)
        x_111563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'x', False)
        # Getting the type of 'data' (line 126)
        data_111564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'data', False)
        # Processing the call keyword arguments (line 126)
        kwargs_111565 = {}
        # Getting the type of 'assert_array_equal' (line 126)
        assert_array_equal_111562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 126)
        assert_array_equal_call_result_111566 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), assert_array_equal_111562, *[x_111563, data_111564], **kwargs_111565)
        
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        slice_111567 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 128, 15), None, None, None)
        int_111568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 19), 'int')
        # Getting the type of 't' (line 128)
        t_111569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 't')
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___111570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 15), t_111569, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_111571 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), getitem___111570, (slice_111567, int_111568))
        
        # Assigning a type to the variable 'col0' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'col0', subscript_call_result_111571)
        
        # Call to assert_array_equal(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'col0' (line 129)
        col0_111573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'col0', False)
        # Getting the type of 'data' (line 129)
        data_111574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 33), 'data', False)
        # Processing the call keyword arguments (line 129)
        kwargs_111575 = {}
        # Getting the type of 'assert_array_equal' (line 129)
        assert_array_equal_111572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 129)
        assert_array_equal_call_result_111576 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), assert_array_equal_111572, *[col0_111573, data_111574], **kwargs_111575)
        
        
        # Call to assert_array_equal(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Obtaining the type of the subscript
        int_111578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 29), 'int')
        int_111579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 31), 'int')
        slice_111580 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 130, 27), int_111579, None, None)
        # Getting the type of 't' (line 130)
        t_111581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), 't', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___111582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 27), t_111581, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_111583 = invoke(stypy.reporting.localization.Localization(__file__, 130, 27), getitem___111582, (int_111578, slice_111580))
        
        
        # Call to conj(...): (line 130)
        # Processing the call keyword arguments (line 130)
        kwargs_111590 = {}
        
        # Obtaining the type of the subscript
        int_111584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 41), 'int')
        slice_111585 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 130, 36), int_111584, None, None)
        # Getting the type of 'data' (line 130)
        data_111586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 36), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___111587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 36), data_111586, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_111588 = invoke(stypy.reporting.localization.Localization(__file__, 130, 36), getitem___111587, slice_111585)
        
        # Obtaining the member 'conj' of a type (line 130)
        conj_111589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 36), subscript_call_result_111588, 'conj')
        # Calling conj(args, kwargs) (line 130)
        conj_call_result_111591 = invoke(stypy.reporting.localization.Localization(__file__, 130, 36), conj_111589, *[], **kwargs_111590)
        
        # Processing the call keyword arguments (line 130)
        kwargs_111592 = {}
        # Getting the type of 'assert_array_equal' (line 130)
        assert_array_equal_111577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 130)
        assert_array_equal_call_result_111593 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), assert_array_equal_111577, *[subscript_call_result_111583, conj_call_result_111591], **kwargs_111592)
        
        
        # ################# End of 'test_complex_01(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_complex_01' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_111594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111594)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_complex_01'
        return stypy_return_type_111594


    @norecursion
    def test_scalar_00(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_00'
        module_type_store = module_type_store.open_function_context('test_scalar_00', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestToeplitz.test_scalar_00.__dict__.__setitem__('stypy_localization', localization)
        TestToeplitz.test_scalar_00.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestToeplitz.test_scalar_00.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestToeplitz.test_scalar_00.__dict__.__setitem__('stypy_function_name', 'TestToeplitz.test_scalar_00')
        TestToeplitz.test_scalar_00.__dict__.__setitem__('stypy_param_names_list', [])
        TestToeplitz.test_scalar_00.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestToeplitz.test_scalar_00.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestToeplitz.test_scalar_00.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestToeplitz.test_scalar_00.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestToeplitz.test_scalar_00.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestToeplitz.test_scalar_00.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestToeplitz.test_scalar_00', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_00', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_00(...)' code ##################

        str_111595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'str', 'Scalar arguments still produce a 2D array.')
        
        # Assigning a Call to a Name (line 134):
        
        # Call to toeplitz(...): (line 134)
        # Processing the call arguments (line 134)
        int_111597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 21), 'int')
        # Processing the call keyword arguments (line 134)
        kwargs_111598 = {}
        # Getting the type of 'toeplitz' (line 134)
        toeplitz_111596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'toeplitz', False)
        # Calling toeplitz(args, kwargs) (line 134)
        toeplitz_call_result_111599 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), toeplitz_111596, *[int_111597], **kwargs_111598)
        
        # Assigning a type to the variable 't' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 't', toeplitz_call_result_111599)
        
        # Call to assert_array_equal(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 't' (line 135)
        t_111601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 27), 't', False)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_111602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_111603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        int_111604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 31), list_111603, int_111604)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 30), list_111602, list_111603)
        
        # Processing the call keyword arguments (line 135)
        kwargs_111605 = {}
        # Getting the type of 'assert_array_equal' (line 135)
        assert_array_equal_111600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 135)
        assert_array_equal_call_result_111606 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), assert_array_equal_111600, *[t_111601, list_111602], **kwargs_111605)
        
        
        # Assigning a Call to a Name (line 136):
        
        # Call to toeplitz(...): (line 136)
        # Processing the call arguments (line 136)
        int_111608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 21), 'int')
        int_111609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 25), 'int')
        # Processing the call keyword arguments (line 136)
        kwargs_111610 = {}
        # Getting the type of 'toeplitz' (line 136)
        toeplitz_111607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'toeplitz', False)
        # Calling toeplitz(args, kwargs) (line 136)
        toeplitz_call_result_111611 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), toeplitz_111607, *[int_111608, int_111609], **kwargs_111610)
        
        # Assigning a type to the variable 't' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 't', toeplitz_call_result_111611)
        
        # Call to assert_array_equal(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 't' (line 137)
        t_111613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 't', False)
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_111614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_111615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        int_111616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 31), list_111615, int_111616)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 30), list_111614, list_111615)
        
        # Processing the call keyword arguments (line 137)
        kwargs_111617 = {}
        # Getting the type of 'assert_array_equal' (line 137)
        assert_array_equal_111612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 137)
        assert_array_equal_call_result_111618 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assert_array_equal_111612, *[t_111613, list_111614], **kwargs_111617)
        
        
        # ################# End of 'test_scalar_00(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_00' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_111619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111619)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_00'
        return stypy_return_type_111619


    @norecursion
    def test_scalar_01(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_01'
        module_type_store = module_type_store.open_function_context('test_scalar_01', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestToeplitz.test_scalar_01.__dict__.__setitem__('stypy_localization', localization)
        TestToeplitz.test_scalar_01.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestToeplitz.test_scalar_01.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestToeplitz.test_scalar_01.__dict__.__setitem__('stypy_function_name', 'TestToeplitz.test_scalar_01')
        TestToeplitz.test_scalar_01.__dict__.__setitem__('stypy_param_names_list', [])
        TestToeplitz.test_scalar_01.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestToeplitz.test_scalar_01.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestToeplitz.test_scalar_01.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestToeplitz.test_scalar_01.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestToeplitz.test_scalar_01.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestToeplitz.test_scalar_01.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestToeplitz.test_scalar_01', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_01', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_01(...)' code ##################

        
        # Assigning a Call to a Name (line 140):
        
        # Call to array(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_111621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        int_111622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 18), list_111621, int_111622)
        # Adding element type (line 140)
        int_111623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 18), list_111621, int_111623)
        # Adding element type (line 140)
        int_111624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 18), list_111621, int_111624)
        
        # Processing the call keyword arguments (line 140)
        kwargs_111625 = {}
        # Getting the type of 'array' (line 140)
        array_111620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'array', False)
        # Calling array(args, kwargs) (line 140)
        array_call_result_111626 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), array_111620, *[list_111621], **kwargs_111625)
        
        # Assigning a type to the variable 'c' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'c', array_call_result_111626)
        
        # Assigning a Call to a Name (line 141):
        
        # Call to toeplitz(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'c' (line 141)
        c_111628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 'c', False)
        int_111629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 24), 'int')
        # Processing the call keyword arguments (line 141)
        kwargs_111630 = {}
        # Getting the type of 'toeplitz' (line 141)
        toeplitz_111627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'toeplitz', False)
        # Calling toeplitz(args, kwargs) (line 141)
        toeplitz_call_result_111631 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), toeplitz_111627, *[c_111628, int_111629], **kwargs_111630)
        
        # Assigning a type to the variable 't' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 't', toeplitz_call_result_111631)
        
        # Call to assert_array_equal(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 't' (line 142)
        t_111633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 't', False)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_111634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_111635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        int_111636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 31), list_111635, int_111636)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 30), list_111634, list_111635)
        # Adding element type (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_111637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        int_111638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 35), list_111637, int_111638)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 30), list_111634, list_111637)
        # Adding element type (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_111639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        int_111640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 39), list_111639, int_111640)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 30), list_111634, list_111639)
        
        # Processing the call keyword arguments (line 142)
        kwargs_111641 = {}
        # Getting the type of 'assert_array_equal' (line 142)
        assert_array_equal_111632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 142)
        assert_array_equal_call_result_111642 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), assert_array_equal_111632, *[t_111633, list_111634], **kwargs_111641)
        
        
        # ################# End of 'test_scalar_01(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_01' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_111643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111643)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_01'
        return stypy_return_type_111643


    @norecursion
    def test_scalar_02(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_02'
        module_type_store = module_type_store.open_function_context('test_scalar_02', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestToeplitz.test_scalar_02.__dict__.__setitem__('stypy_localization', localization)
        TestToeplitz.test_scalar_02.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestToeplitz.test_scalar_02.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestToeplitz.test_scalar_02.__dict__.__setitem__('stypy_function_name', 'TestToeplitz.test_scalar_02')
        TestToeplitz.test_scalar_02.__dict__.__setitem__('stypy_param_names_list', [])
        TestToeplitz.test_scalar_02.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestToeplitz.test_scalar_02.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestToeplitz.test_scalar_02.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestToeplitz.test_scalar_02.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestToeplitz.test_scalar_02.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestToeplitz.test_scalar_02.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestToeplitz.test_scalar_02', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_02', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_02(...)' code ##################

        
        # Assigning a Call to a Name (line 145):
        
        # Call to array(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Obtaining an instance of the builtin type 'list' (line 145)
        list_111645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 145)
        # Adding element type (line 145)
        int_111646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), list_111645, int_111646)
        # Adding element type (line 145)
        int_111647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), list_111645, int_111647)
        # Adding element type (line 145)
        int_111648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 18), list_111645, int_111648)
        
        # Processing the call keyword arguments (line 145)
        kwargs_111649 = {}
        # Getting the type of 'array' (line 145)
        array_111644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'array', False)
        # Calling array(args, kwargs) (line 145)
        array_call_result_111650 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), array_111644, *[list_111645], **kwargs_111649)
        
        # Assigning a type to the variable 'c' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'c', array_call_result_111650)
        
        # Assigning a Call to a Name (line 146):
        
        # Call to toeplitz(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'c' (line 146)
        c_111652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 21), 'c', False)
        
        # Call to array(...): (line 146)
        # Processing the call arguments (line 146)
        int_111654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 30), 'int')
        # Processing the call keyword arguments (line 146)
        kwargs_111655 = {}
        # Getting the type of 'array' (line 146)
        array_111653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'array', False)
        # Calling array(args, kwargs) (line 146)
        array_call_result_111656 = invoke(stypy.reporting.localization.Localization(__file__, 146, 24), array_111653, *[int_111654], **kwargs_111655)
        
        # Processing the call keyword arguments (line 146)
        kwargs_111657 = {}
        # Getting the type of 'toeplitz' (line 146)
        toeplitz_111651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'toeplitz', False)
        # Calling toeplitz(args, kwargs) (line 146)
        toeplitz_call_result_111658 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), toeplitz_111651, *[c_111652, array_call_result_111656], **kwargs_111657)
        
        # Assigning a type to the variable 't' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 't', toeplitz_call_result_111658)
        
        # Call to assert_array_equal(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 't' (line 147)
        t_111660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 't', False)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_111661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_111662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        int_111663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 31), list_111662, int_111663)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 30), list_111661, list_111662)
        # Adding element type (line 147)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_111664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        int_111665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 35), list_111664, int_111665)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 30), list_111661, list_111664)
        # Adding element type (line 147)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_111666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        int_111667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 39), list_111666, int_111667)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 30), list_111661, list_111666)
        
        # Processing the call keyword arguments (line 147)
        kwargs_111668 = {}
        # Getting the type of 'assert_array_equal' (line 147)
        assert_array_equal_111659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 147)
        assert_array_equal_call_result_111669 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), assert_array_equal_111659, *[t_111660, list_111661], **kwargs_111668)
        
        
        # ################# End of 'test_scalar_02(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_02' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_111670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111670)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_02'
        return stypy_return_type_111670


    @norecursion
    def test_scalar_03(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_03'
        module_type_store = module_type_store.open_function_context('test_scalar_03', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestToeplitz.test_scalar_03.__dict__.__setitem__('stypy_localization', localization)
        TestToeplitz.test_scalar_03.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestToeplitz.test_scalar_03.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestToeplitz.test_scalar_03.__dict__.__setitem__('stypy_function_name', 'TestToeplitz.test_scalar_03')
        TestToeplitz.test_scalar_03.__dict__.__setitem__('stypy_param_names_list', [])
        TestToeplitz.test_scalar_03.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestToeplitz.test_scalar_03.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestToeplitz.test_scalar_03.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestToeplitz.test_scalar_03.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestToeplitz.test_scalar_03.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestToeplitz.test_scalar_03.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestToeplitz.test_scalar_03', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_03', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_03(...)' code ##################

        
        # Assigning a Call to a Name (line 150):
        
        # Call to array(...): (line 150)
        # Processing the call arguments (line 150)
        
        # Obtaining an instance of the builtin type 'list' (line 150)
        list_111672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 150)
        # Adding element type (line 150)
        int_111673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), list_111672, int_111673)
        # Adding element type (line 150)
        int_111674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), list_111672, int_111674)
        # Adding element type (line 150)
        int_111675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), list_111672, int_111675)
        
        # Processing the call keyword arguments (line 150)
        kwargs_111676 = {}
        # Getting the type of 'array' (line 150)
        array_111671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'array', False)
        # Calling array(args, kwargs) (line 150)
        array_call_result_111677 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), array_111671, *[list_111672], **kwargs_111676)
        
        # Assigning a type to the variable 'c' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'c', array_call_result_111677)
        
        # Assigning a Call to a Name (line 151):
        
        # Call to toeplitz(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'c' (line 151)
        c_111679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'c', False)
        
        # Call to array(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_111681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        int_111682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 30), list_111681, int_111682)
        
        # Processing the call keyword arguments (line 151)
        kwargs_111683 = {}
        # Getting the type of 'array' (line 151)
        array_111680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'array', False)
        # Calling array(args, kwargs) (line 151)
        array_call_result_111684 = invoke(stypy.reporting.localization.Localization(__file__, 151, 24), array_111680, *[list_111681], **kwargs_111683)
        
        # Processing the call keyword arguments (line 151)
        kwargs_111685 = {}
        # Getting the type of 'toeplitz' (line 151)
        toeplitz_111678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'toeplitz', False)
        # Calling toeplitz(args, kwargs) (line 151)
        toeplitz_call_result_111686 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), toeplitz_111678, *[c_111679, array_call_result_111684], **kwargs_111685)
        
        # Assigning a type to the variable 't' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 't', toeplitz_call_result_111686)
        
        # Call to assert_array_equal(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 't' (line 152)
        t_111688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 27), 't', False)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_111689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_111690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        int_111691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 31), list_111690, int_111691)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 30), list_111689, list_111690)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_111692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        int_111693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 35), list_111692, int_111693)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 30), list_111689, list_111692)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_111694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        int_111695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 39), list_111694, int_111695)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 30), list_111689, list_111694)
        
        # Processing the call keyword arguments (line 152)
        kwargs_111696 = {}
        # Getting the type of 'assert_array_equal' (line 152)
        assert_array_equal_111687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 152)
        assert_array_equal_call_result_111697 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), assert_array_equal_111687, *[t_111688, list_111689], **kwargs_111696)
        
        
        # ################# End of 'test_scalar_03(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_03' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_111698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111698)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_03'
        return stypy_return_type_111698


    @norecursion
    def test_scalar_04(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_04'
        module_type_store = module_type_store.open_function_context('test_scalar_04', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestToeplitz.test_scalar_04.__dict__.__setitem__('stypy_localization', localization)
        TestToeplitz.test_scalar_04.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestToeplitz.test_scalar_04.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestToeplitz.test_scalar_04.__dict__.__setitem__('stypy_function_name', 'TestToeplitz.test_scalar_04')
        TestToeplitz.test_scalar_04.__dict__.__setitem__('stypy_param_names_list', [])
        TestToeplitz.test_scalar_04.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestToeplitz.test_scalar_04.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestToeplitz.test_scalar_04.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestToeplitz.test_scalar_04.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestToeplitz.test_scalar_04.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestToeplitz.test_scalar_04.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestToeplitz.test_scalar_04', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_04', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_04(...)' code ##################

        
        # Assigning a Call to a Name (line 155):
        
        # Call to array(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Obtaining an instance of the builtin type 'list' (line 155)
        list_111700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 155)
        # Adding element type (line 155)
        int_111701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 18), list_111700, int_111701)
        # Adding element type (line 155)
        int_111702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 18), list_111700, int_111702)
        # Adding element type (line 155)
        int_111703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 18), list_111700, int_111703)
        
        # Processing the call keyword arguments (line 155)
        kwargs_111704 = {}
        # Getting the type of 'array' (line 155)
        array_111699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'array', False)
        # Calling array(args, kwargs) (line 155)
        array_call_result_111705 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), array_111699, *[list_111700], **kwargs_111704)
        
        # Assigning a type to the variable 'r' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'r', array_call_result_111705)
        
        # Assigning a Call to a Name (line 156):
        
        # Call to toeplitz(...): (line 156)
        # Processing the call arguments (line 156)
        int_111707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 21), 'int')
        # Getting the type of 'r' (line 156)
        r_111708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 24), 'r', False)
        # Processing the call keyword arguments (line 156)
        kwargs_111709 = {}
        # Getting the type of 'toeplitz' (line 156)
        toeplitz_111706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'toeplitz', False)
        # Calling toeplitz(args, kwargs) (line 156)
        toeplitz_call_result_111710 = invoke(stypy.reporting.localization.Localization(__file__, 156, 12), toeplitz_111706, *[int_111707, r_111708], **kwargs_111709)
        
        # Assigning a type to the variable 't' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 't', toeplitz_call_result_111710)
        
        # Call to assert_array_equal(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 't' (line 157)
        t_111712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 't', False)
        
        # Obtaining an instance of the builtin type 'list' (line 157)
        list_111713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 157)
        # Adding element type (line 157)
        
        # Obtaining an instance of the builtin type 'list' (line 157)
        list_111714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 157)
        # Adding element type (line 157)
        int_111715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 31), list_111714, int_111715)
        # Adding element type (line 157)
        int_111716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 31), list_111714, int_111716)
        # Adding element type (line 157)
        int_111717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 31), list_111714, int_111717)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 30), list_111713, list_111714)
        
        # Processing the call keyword arguments (line 157)
        kwargs_111718 = {}
        # Getting the type of 'assert_array_equal' (line 157)
        assert_array_equal_111711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 157)
        assert_array_equal_call_result_111719 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), assert_array_equal_111711, *[t_111712, list_111713], **kwargs_111718)
        
        
        # ################# End of 'test_scalar_04(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_04' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_111720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111720)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_04'
        return stypy_return_type_111720


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 113, 0, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestToeplitz.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestToeplitz' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'TestToeplitz', TestToeplitz)
# Declaration of the 'TestHankel' class

class TestHankel(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHankel.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestHankel.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHankel.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHankel.test_basic.__dict__.__setitem__('stypy_function_name', 'TestHankel.test_basic')
        TestHankel.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestHankel.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHankel.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHankel.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHankel.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHankel.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHankel.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHankel.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 162):
        
        # Call to hankel(...): (line 162)
        # Processing the call arguments (line 162)
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_111722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        # Adding element type (line 162)
        int_111723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 19), list_111722, int_111723)
        # Adding element type (line 162)
        int_111724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 19), list_111722, int_111724)
        # Adding element type (line 162)
        int_111725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 19), list_111722, int_111725)
        
        # Processing the call keyword arguments (line 162)
        kwargs_111726 = {}
        # Getting the type of 'hankel' (line 162)
        hankel_111721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'hankel', False)
        # Calling hankel(args, kwargs) (line 162)
        hankel_call_result_111727 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), hankel_111721, *[list_111722], **kwargs_111726)
        
        # Assigning a type to the variable 'y' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'y', hankel_call_result_111727)
        
        # Call to assert_array_equal(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'y' (line 163)
        y_111729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'y', False)
        
        # Obtaining an instance of the builtin type 'list' (line 163)
        list_111730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 163)
        # Adding element type (line 163)
        
        # Obtaining an instance of the builtin type 'list' (line 163)
        list_111731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 163)
        # Adding element type (line 163)
        int_111732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 31), list_111731, int_111732)
        # Adding element type (line 163)
        int_111733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 31), list_111731, int_111733)
        # Adding element type (line 163)
        int_111734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 31), list_111731, int_111734)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 30), list_111730, list_111731)
        # Adding element type (line 163)
        
        # Obtaining an instance of the builtin type 'list' (line 163)
        list_111735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 163)
        # Adding element type (line 163)
        int_111736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 40), list_111735, int_111736)
        # Adding element type (line 163)
        int_111737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 40), list_111735, int_111737)
        # Adding element type (line 163)
        int_111738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 40), list_111735, int_111738)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 30), list_111730, list_111735)
        # Adding element type (line 163)
        
        # Obtaining an instance of the builtin type 'list' (line 163)
        list_111739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 163)
        # Adding element type (line 163)
        int_111740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 49), list_111739, int_111740)
        # Adding element type (line 163)
        int_111741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 49), list_111739, int_111741)
        # Adding element type (line 163)
        int_111742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 49), list_111739, int_111742)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 30), list_111730, list_111739)
        
        # Processing the call keyword arguments (line 163)
        kwargs_111743 = {}
        # Getting the type of 'assert_array_equal' (line 163)
        assert_array_equal_111728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 163)
        assert_array_equal_call_result_111744 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), assert_array_equal_111728, *[y_111729, list_111730], **kwargs_111743)
        
        
        # Assigning a Call to a Name (line 164):
        
        # Call to hankel(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_111746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        # Adding element type (line 164)
        int_111747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 19), list_111746, int_111747)
        # Adding element type (line 164)
        int_111748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 19), list_111746, int_111748)
        # Adding element type (line 164)
        int_111749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 19), list_111746, int_111749)
        
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_111750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        # Adding element type (line 164)
        int_111751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), list_111750, int_111751)
        # Adding element type (line 164)
        int_111752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), list_111750, int_111752)
        # Adding element type (line 164)
        int_111753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), list_111750, int_111753)
        
        # Processing the call keyword arguments (line 164)
        kwargs_111754 = {}
        # Getting the type of 'hankel' (line 164)
        hankel_111745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'hankel', False)
        # Calling hankel(args, kwargs) (line 164)
        hankel_call_result_111755 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), hankel_111745, *[list_111746, list_111750], **kwargs_111754)
        
        # Assigning a type to the variable 'y' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'y', hankel_call_result_111755)
        
        # Call to assert_array_equal(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'y' (line 165)
        y_111757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 27), 'y', False)
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_111758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_111759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        int_111760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 31), list_111759, int_111760)
        # Adding element type (line 165)
        int_111761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 31), list_111759, int_111761)
        # Adding element type (line 165)
        int_111762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 31), list_111759, int_111762)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 30), list_111758, list_111759)
        # Adding element type (line 165)
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_111763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        int_111764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 40), list_111763, int_111764)
        # Adding element type (line 165)
        int_111765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 40), list_111763, int_111765)
        # Adding element type (line 165)
        int_111766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 40), list_111763, int_111766)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 30), list_111758, list_111763)
        # Adding element type (line 165)
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_111767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        int_111768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 49), list_111767, int_111768)
        # Adding element type (line 165)
        int_111769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 49), list_111767, int_111769)
        # Adding element type (line 165)
        int_111770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 49), list_111767, int_111770)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 30), list_111758, list_111767)
        
        # Processing the call keyword arguments (line 165)
        kwargs_111771 = {}
        # Getting the type of 'assert_array_equal' (line 165)
        assert_array_equal_111756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 165)
        assert_array_equal_call_result_111772 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), assert_array_equal_111756, *[y_111757, list_111758], **kwargs_111771)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_111773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111773)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_111773


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 160, 0, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHankel.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestHankel' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'TestHankel', TestHankel)
# Declaration of the 'TestCirculant' class

class TestCirculant(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCirculant.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestCirculant.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCirculant.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCirculant.test_basic.__dict__.__setitem__('stypy_function_name', 'TestCirculant.test_basic')
        TestCirculant.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestCirculant.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCirculant.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCirculant.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCirculant.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCirculant.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCirculant.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCirculant.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 170):
        
        # Call to circulant(...): (line 170)
        # Processing the call arguments (line 170)
        
        # Obtaining an instance of the builtin type 'list' (line 170)
        list_111775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 170)
        # Adding element type (line 170)
        int_111776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 22), list_111775, int_111776)
        # Adding element type (line 170)
        int_111777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 22), list_111775, int_111777)
        # Adding element type (line 170)
        int_111778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 22), list_111775, int_111778)
        
        # Processing the call keyword arguments (line 170)
        kwargs_111779 = {}
        # Getting the type of 'circulant' (line 170)
        circulant_111774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'circulant', False)
        # Calling circulant(args, kwargs) (line 170)
        circulant_call_result_111780 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), circulant_111774, *[list_111775], **kwargs_111779)
        
        # Assigning a type to the variable 'y' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'y', circulant_call_result_111780)
        
        # Call to assert_array_equal(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'y' (line 171)
        y_111782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 27), 'y', False)
        
        # Obtaining an instance of the builtin type 'list' (line 171)
        list_111783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 171)
        # Adding element type (line 171)
        
        # Obtaining an instance of the builtin type 'list' (line 171)
        list_111784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 171)
        # Adding element type (line 171)
        int_111785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 31), list_111784, int_111785)
        # Adding element type (line 171)
        int_111786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 31), list_111784, int_111786)
        # Adding element type (line 171)
        int_111787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 31), list_111784, int_111787)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 30), list_111783, list_111784)
        # Adding element type (line 171)
        
        # Obtaining an instance of the builtin type 'list' (line 171)
        list_111788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 171)
        # Adding element type (line 171)
        int_111789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 40), list_111788, int_111789)
        # Adding element type (line 171)
        int_111790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 40), list_111788, int_111790)
        # Adding element type (line 171)
        int_111791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 40), list_111788, int_111791)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 30), list_111783, list_111788)
        # Adding element type (line 171)
        
        # Obtaining an instance of the builtin type 'list' (line 171)
        list_111792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 171)
        # Adding element type (line 171)
        int_111793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 49), list_111792, int_111793)
        # Adding element type (line 171)
        int_111794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 49), list_111792, int_111794)
        # Adding element type (line 171)
        int_111795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 49), list_111792, int_111795)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 30), list_111783, list_111792)
        
        # Processing the call keyword arguments (line 171)
        kwargs_111796 = {}
        # Getting the type of 'assert_array_equal' (line 171)
        assert_array_equal_111781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 171)
        assert_array_equal_call_result_111797 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), assert_array_equal_111781, *[y_111782, list_111783], **kwargs_111796)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 169)
        stypy_return_type_111798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111798)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_111798


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 168, 0, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCirculant.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCirculant' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'TestCirculant', TestCirculant)
# Declaration of the 'TestHadamard' class

class TestHadamard(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 176, 4, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHadamard.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestHadamard.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHadamard.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHadamard.test_basic.__dict__.__setitem__('stypy_function_name', 'TestHadamard.test_basic')
        TestHadamard.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestHadamard.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHadamard.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHadamard.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHadamard.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHadamard.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHadamard.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHadamard.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 178):
        
        # Call to hadamard(...): (line 178)
        # Processing the call arguments (line 178)
        int_111800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 21), 'int')
        # Processing the call keyword arguments (line 178)
        kwargs_111801 = {}
        # Getting the type of 'hadamard' (line 178)
        hadamard_111799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'hadamard', False)
        # Calling hadamard(args, kwargs) (line 178)
        hadamard_call_result_111802 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), hadamard_111799, *[int_111800], **kwargs_111801)
        
        # Assigning a type to the variable 'y' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'y', hadamard_call_result_111802)
        
        # Call to assert_array_equal(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'y' (line 179)
        y_111804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'y', False)
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_111805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_111806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        int_111807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 31), list_111806, int_111807)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 30), list_111805, list_111806)
        
        # Processing the call keyword arguments (line 179)
        kwargs_111808 = {}
        # Getting the type of 'assert_array_equal' (line 179)
        assert_array_equal_111803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 179)
        assert_array_equal_call_result_111809 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), assert_array_equal_111803, *[y_111804, list_111805], **kwargs_111808)
        
        
        # Assigning a Call to a Name (line 181):
        
        # Call to hadamard(...): (line 181)
        # Processing the call arguments (line 181)
        int_111811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 21), 'int')
        # Processing the call keyword arguments (line 181)
        # Getting the type of 'float' (line 181)
        float_111812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), 'float', False)
        keyword_111813 = float_111812
        kwargs_111814 = {'dtype': keyword_111813}
        # Getting the type of 'hadamard' (line 181)
        hadamard_111810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'hadamard', False)
        # Calling hadamard(args, kwargs) (line 181)
        hadamard_call_result_111815 = invoke(stypy.reporting.localization.Localization(__file__, 181, 12), hadamard_111810, *[int_111811], **kwargs_111814)
        
        # Assigning a type to the variable 'y' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'y', hadamard_call_result_111815)
        
        # Call to assert_array_equal(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'y' (line 182)
        y_111817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 27), 'y', False)
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_111818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        # Adding element type (line 182)
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_111819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        # Adding element type (line 182)
        float_111820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 31), list_111819, float_111820)
        # Adding element type (line 182)
        float_111821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 31), list_111819, float_111821)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 30), list_111818, list_111819)
        # Adding element type (line 182)
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_111822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        # Adding element type (line 182)
        float_111823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 43), list_111822, float_111823)
        # Adding element type (line 182)
        float_111824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 43), list_111822, float_111824)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 30), list_111818, list_111822)
        
        # Processing the call keyword arguments (line 182)
        kwargs_111825 = {}
        # Getting the type of 'assert_array_equal' (line 182)
        assert_array_equal_111816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 182)
        assert_array_equal_call_result_111826 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), assert_array_equal_111816, *[y_111817, list_111818], **kwargs_111825)
        
        
        # Assigning a Call to a Name (line 184):
        
        # Call to hadamard(...): (line 184)
        # Processing the call arguments (line 184)
        int_111828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 21), 'int')
        # Processing the call keyword arguments (line 184)
        kwargs_111829 = {}
        # Getting the type of 'hadamard' (line 184)
        hadamard_111827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'hadamard', False)
        # Calling hadamard(args, kwargs) (line 184)
        hadamard_call_result_111830 = invoke(stypy.reporting.localization.Localization(__file__, 184, 12), hadamard_111827, *[int_111828], **kwargs_111829)
        
        # Assigning a type to the variable 'y' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'y', hadamard_call_result_111830)
        
        # Call to assert_array_equal(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'y' (line 185)
        y_111832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 27), 'y', False)
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_111833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_111834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        int_111835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 31), list_111834, int_111835)
        # Adding element type (line 185)
        int_111836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 31), list_111834, int_111836)
        # Adding element type (line 185)
        int_111837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 31), list_111834, int_111837)
        # Adding element type (line 185)
        int_111838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 31), list_111834, int_111838)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 30), list_111833, list_111834)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_111839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        int_111840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 42), list_111839, int_111840)
        # Adding element type (line 185)
        int_111841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 42), list_111839, int_111841)
        # Adding element type (line 185)
        int_111842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 42), list_111839, int_111842)
        # Adding element type (line 185)
        int_111843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 42), list_111839, int_111843)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 30), list_111833, list_111839)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_111844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        int_111845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 55), list_111844, int_111845)
        # Adding element type (line 185)
        int_111846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 55), list_111844, int_111846)
        # Adding element type (line 185)
        int_111847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 55), list_111844, int_111847)
        # Adding element type (line 185)
        int_111848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 55), list_111844, int_111848)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 30), list_111833, list_111844)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_111849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 68), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        int_111850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 68), list_111849, int_111850)
        # Adding element type (line 185)
        int_111851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 71), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 68), list_111849, int_111851)
        # Adding element type (line 185)
        int_111852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 74), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 68), list_111849, int_111852)
        # Adding element type (line 185)
        int_111853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 77), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 68), list_111849, int_111853)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 30), list_111833, list_111849)
        
        # Processing the call keyword arguments (line 185)
        kwargs_111854 = {}
        # Getting the type of 'assert_array_equal' (line 185)
        assert_array_equal_111831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 185)
        assert_array_equal_call_result_111855 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), assert_array_equal_111831, *[y_111832, list_111833], **kwargs_111854)
        
        
        # Call to assert_raises(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'ValueError' (line 187)
        ValueError_111857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 22), 'ValueError', False)
        # Getting the type of 'hadamard' (line 187)
        hadamard_111858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 34), 'hadamard', False)
        int_111859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 44), 'int')
        # Processing the call keyword arguments (line 187)
        kwargs_111860 = {}
        # Getting the type of 'assert_raises' (line 187)
        assert_raises_111856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 187)
        assert_raises_call_result_111861 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), assert_raises_111856, *[ValueError_111857, hadamard_111858, int_111859], **kwargs_111860)
        
        
        # Call to assert_raises(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'ValueError' (line 188)
        ValueError_111863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 22), 'ValueError', False)
        # Getting the type of 'hadamard' (line 188)
        hadamard_111864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 34), 'hadamard', False)
        int_111865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 44), 'int')
        # Processing the call keyword arguments (line 188)
        kwargs_111866 = {}
        # Getting the type of 'assert_raises' (line 188)
        assert_raises_111862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 188)
        assert_raises_call_result_111867 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), assert_raises_111862, *[ValueError_111863, hadamard_111864, int_111865], **kwargs_111866)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_111868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111868)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_111868


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 174, 0, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHadamard.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestHadamard' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'TestHadamard', TestHadamard)
# Declaration of the 'TestLeslie' class

class TestLeslie(object, ):

    @norecursion
    def test_bad_shapes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bad_shapes'
        module_type_store = module_type_store.open_function_context('test_bad_shapes', 193, 4, False)
        # Assigning a type to the variable 'self' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLeslie.test_bad_shapes.__dict__.__setitem__('stypy_localization', localization)
        TestLeslie.test_bad_shapes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLeslie.test_bad_shapes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLeslie.test_bad_shapes.__dict__.__setitem__('stypy_function_name', 'TestLeslie.test_bad_shapes')
        TestLeslie.test_bad_shapes.__dict__.__setitem__('stypy_param_names_list', [])
        TestLeslie.test_bad_shapes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLeslie.test_bad_shapes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLeslie.test_bad_shapes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLeslie.test_bad_shapes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLeslie.test_bad_shapes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLeslie.test_bad_shapes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLeslie.test_bad_shapes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bad_shapes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bad_shapes(...)' code ##################

        
        # Call to assert_raises(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'ValueError' (line 194)
        ValueError_111870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 22), 'ValueError', False)
        # Getting the type of 'leslie' (line 194)
        leslie_111871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 34), 'leslie', False)
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_111872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_111873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        int_111874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 43), list_111873, int_111874)
        # Adding element type (line 194)
        int_111875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 43), list_111873, int_111875)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 42), list_111872, list_111873)
        # Adding element type (line 194)
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_111876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        int_111877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 49), list_111876, int_111877)
        # Adding element type (line 194)
        int_111878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 49), list_111876, int_111878)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 42), list_111872, list_111876)
        
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_111879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        int_111880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 57), list_111879, int_111880)
        # Adding element type (line 194)
        int_111881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 57), list_111879, int_111881)
        # Adding element type (line 194)
        int_111882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 57), list_111879, int_111882)
        
        # Processing the call keyword arguments (line 194)
        kwargs_111883 = {}
        # Getting the type of 'assert_raises' (line 194)
        assert_raises_111869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 194)
        assert_raises_call_result_111884 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), assert_raises_111869, *[ValueError_111870, leslie_111871, list_111872, list_111879], **kwargs_111883)
        
        
        # Call to assert_raises(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'ValueError' (line 195)
        ValueError_111886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'ValueError', False)
        # Getting the type of 'leslie' (line 195)
        leslie_111887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 34), 'leslie', False)
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_111888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        int_111889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 42), list_111888, int_111889)
        # Adding element type (line 195)
        int_111890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 42), list_111888, int_111890)
        # Adding element type (line 195)
        int_111891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 42), list_111888, int_111891)
        
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_111892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_111893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        int_111894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 52), list_111893, int_111894)
        # Adding element type (line 195)
        int_111895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 52), list_111893, int_111895)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 51), list_111892, list_111893)
        # Adding element type (line 195)
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_111896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        int_111897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 58), list_111896, int_111897)
        # Adding element type (line 195)
        int_111898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 58), list_111896, int_111898)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 51), list_111892, list_111896)
        
        # Processing the call keyword arguments (line 195)
        kwargs_111899 = {}
        # Getting the type of 'assert_raises' (line 195)
        assert_raises_111885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 195)
        assert_raises_call_result_111900 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), assert_raises_111885, *[ValueError_111886, leslie_111887, list_111888, list_111892], **kwargs_111899)
        
        
        # Call to assert_raises(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'ValueError' (line 196)
        ValueError_111902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 22), 'ValueError', False)
        # Getting the type of 'leslie' (line 196)
        leslie_111903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 34), 'leslie', False)
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_111904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        int_111905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 42), list_111904, int_111905)
        # Adding element type (line 196)
        int_111906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 42), list_111904, int_111906)
        
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_111907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        int_111908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 49), list_111907, int_111908)
        # Adding element type (line 196)
        int_111909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 49), list_111907, int_111909)
        
        # Processing the call keyword arguments (line 196)
        kwargs_111910 = {}
        # Getting the type of 'assert_raises' (line 196)
        assert_raises_111901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 196)
        assert_raises_call_result_111911 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), assert_raises_111901, *[ValueError_111902, leslie_111903, list_111904, list_111907], **kwargs_111910)
        
        
        # Call to assert_raises(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'ValueError' (line 197)
        ValueError_111913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 22), 'ValueError', False)
        # Getting the type of 'leslie' (line 197)
        leslie_111914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 34), 'leslie', False)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_111915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        int_111916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 42), list_111915, int_111916)
        
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_111917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        
        # Processing the call keyword arguments (line 197)
        kwargs_111918 = {}
        # Getting the type of 'assert_raises' (line 197)
        assert_raises_111912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 197)
        assert_raises_call_result_111919 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), assert_raises_111912, *[ValueError_111913, leslie_111914, list_111915, list_111917], **kwargs_111918)
        
        
        # ################# End of 'test_bad_shapes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bad_shapes' in the type store
        # Getting the type of 'stypy_return_type' (line 193)
        stypy_return_type_111920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111920)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bad_shapes'
        return stypy_return_type_111920


    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 199, 4, False)
        # Assigning a type to the variable 'self' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLeslie.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestLeslie.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLeslie.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLeslie.test_basic.__dict__.__setitem__('stypy_function_name', 'TestLeslie.test_basic')
        TestLeslie.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestLeslie.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLeslie.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLeslie.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLeslie.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLeslie.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLeslie.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLeslie.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 200):
        
        # Call to leslie(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Obtaining an instance of the builtin type 'list' (line 200)
        list_111922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 200)
        # Adding element type (line 200)
        int_111923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 19), list_111922, int_111923)
        # Adding element type (line 200)
        int_111924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 19), list_111922, int_111924)
        # Adding element type (line 200)
        int_111925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 19), list_111922, int_111925)
        
        
        # Obtaining an instance of the builtin type 'list' (line 200)
        list_111926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 200)
        # Adding element type (line 200)
        float_111927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 30), list_111926, float_111927)
        # Adding element type (line 200)
        float_111928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 30), list_111926, float_111928)
        
        # Processing the call keyword arguments (line 200)
        kwargs_111929 = {}
        # Getting the type of 'leslie' (line 200)
        leslie_111921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'leslie', False)
        # Calling leslie(args, kwargs) (line 200)
        leslie_call_result_111930 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), leslie_111921, *[list_111922, list_111926], **kwargs_111929)
        
        # Assigning a type to the variable 'a' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'a', leslie_call_result_111930)
        
        # Assigning a Call to a Name (line 201):
        
        # Call to array(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 201)
        list_111932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 201)
        # Adding element type (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_111933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        float_111934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 12), list_111933, float_111934)
        # Adding element type (line 202)
        float_111935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 12), list_111933, float_111935)
        # Adding element type (line 202)
        float_111936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 12), list_111933, float_111936)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 25), list_111932, list_111933)
        # Adding element type (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_111937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        # Adding element type (line 203)
        float_111938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 12), list_111937, float_111938)
        # Adding element type (line 203)
        float_111939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 12), list_111937, float_111939)
        # Adding element type (line 203)
        float_111940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 12), list_111937, float_111940)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 25), list_111932, list_111937)
        # Adding element type (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_111941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        # Adding element type (line 204)
        float_111942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 12), list_111941, float_111942)
        # Adding element type (line 204)
        float_111943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 12), list_111941, float_111943)
        # Adding element type (line 204)
        float_111944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 12), list_111941, float_111944)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 25), list_111932, list_111941)
        
        # Processing the call keyword arguments (line 201)
        kwargs_111945 = {}
        # Getting the type of 'array' (line 201)
        array_111931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'array', False)
        # Calling array(args, kwargs) (line 201)
        array_call_result_111946 = invoke(stypy.reporting.localization.Localization(__file__, 201, 19), array_111931, *[list_111932], **kwargs_111945)
        
        # Assigning a type to the variable 'expected' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'expected', array_call_result_111946)
        
        # Call to assert_array_equal(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'a' (line 205)
        a_111948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'a', False)
        # Getting the type of 'expected' (line 205)
        expected_111949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), 'expected', False)
        # Processing the call keyword arguments (line 205)
        kwargs_111950 = {}
        # Getting the type of 'assert_array_equal' (line 205)
        assert_array_equal_111947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 205)
        assert_array_equal_call_result_111951 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), assert_array_equal_111947, *[a_111948, expected_111949], **kwargs_111950)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 199)
        stypy_return_type_111952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111952)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_111952


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLeslie.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLeslie' (line 191)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'TestLeslie', TestLeslie)
# Declaration of the 'TestCompanion' class

class TestCompanion(object, ):

    @norecursion
    def test_bad_shapes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bad_shapes'
        module_type_store = module_type_store.open_function_context('test_bad_shapes', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCompanion.test_bad_shapes.__dict__.__setitem__('stypy_localization', localization)
        TestCompanion.test_bad_shapes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCompanion.test_bad_shapes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCompanion.test_bad_shapes.__dict__.__setitem__('stypy_function_name', 'TestCompanion.test_bad_shapes')
        TestCompanion.test_bad_shapes.__dict__.__setitem__('stypy_param_names_list', [])
        TestCompanion.test_bad_shapes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCompanion.test_bad_shapes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCompanion.test_bad_shapes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCompanion.test_bad_shapes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCompanion.test_bad_shapes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCompanion.test_bad_shapes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCompanion.test_bad_shapes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bad_shapes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bad_shapes(...)' code ##################

        
        # Call to assert_raises(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'ValueError' (line 211)
        ValueError_111954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 'ValueError', False)
        # Getting the type of 'companion' (line 211)
        companion_111955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 34), 'companion', False)
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_111956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        # Adding element type (line 211)
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_111957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        # Adding element type (line 211)
        int_111958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 46), list_111957, int_111958)
        # Adding element type (line 211)
        int_111959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 46), list_111957, int_111959)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 45), list_111956, list_111957)
        # Adding element type (line 211)
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_111960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        # Adding element type (line 211)
        int_111961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 52), list_111960, int_111961)
        # Adding element type (line 211)
        int_111962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 52), list_111960, int_111962)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 45), list_111956, list_111960)
        
        # Processing the call keyword arguments (line 211)
        kwargs_111963 = {}
        # Getting the type of 'assert_raises' (line 211)
        assert_raises_111953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 211)
        assert_raises_call_result_111964 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), assert_raises_111953, *[ValueError_111954, companion_111955, list_111956], **kwargs_111963)
        
        
        # Call to assert_raises(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'ValueError' (line 212)
        ValueError_111966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 22), 'ValueError', False)
        # Getting the type of 'companion' (line 212)
        companion_111967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 34), 'companion', False)
        
        # Obtaining an instance of the builtin type 'list' (line 212)
        list_111968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 212)
        # Adding element type (line 212)
        int_111969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 45), list_111968, int_111969)
        # Adding element type (line 212)
        int_111970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 45), list_111968, int_111970)
        # Adding element type (line 212)
        int_111971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 45), list_111968, int_111971)
        
        # Processing the call keyword arguments (line 212)
        kwargs_111972 = {}
        # Getting the type of 'assert_raises' (line 212)
        assert_raises_111965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 212)
        assert_raises_call_result_111973 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), assert_raises_111965, *[ValueError_111966, companion_111967, list_111968], **kwargs_111972)
        
        
        # Call to assert_raises(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'ValueError' (line 213)
        ValueError_111975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 22), 'ValueError', False)
        # Getting the type of 'companion' (line 213)
        companion_111976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 34), 'companion', False)
        
        # Obtaining an instance of the builtin type 'list' (line 213)
        list_111977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 213)
        # Adding element type (line 213)
        int_111978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 45), list_111977, int_111978)
        
        # Processing the call keyword arguments (line 213)
        kwargs_111979 = {}
        # Getting the type of 'assert_raises' (line 213)
        assert_raises_111974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 213)
        assert_raises_call_result_111980 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), assert_raises_111974, *[ValueError_111975, companion_111976, list_111977], **kwargs_111979)
        
        
        # Call to assert_raises(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'ValueError' (line 214)
        ValueError_111982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 22), 'ValueError', False)
        # Getting the type of 'companion' (line 214)
        companion_111983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 34), 'companion', False)
        
        # Obtaining an instance of the builtin type 'list' (line 214)
        list_111984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 214)
        
        # Processing the call keyword arguments (line 214)
        kwargs_111985 = {}
        # Getting the type of 'assert_raises' (line 214)
        assert_raises_111981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 214)
        assert_raises_call_result_111986 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), assert_raises_111981, *[ValueError_111982, companion_111983, list_111984], **kwargs_111985)
        
        
        # ################# End of 'test_bad_shapes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bad_shapes' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_111987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111987)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bad_shapes'
        return stypy_return_type_111987


    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCompanion.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestCompanion.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCompanion.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCompanion.test_basic.__dict__.__setitem__('stypy_function_name', 'TestCompanion.test_basic')
        TestCompanion.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestCompanion.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCompanion.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCompanion.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCompanion.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCompanion.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCompanion.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCompanion.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 217):
        
        # Call to companion(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_111989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        int_111990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 22), list_111989, int_111990)
        # Adding element type (line 217)
        int_111991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 22), list_111989, int_111991)
        # Adding element type (line 217)
        int_111992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 22), list_111989, int_111992)
        
        # Processing the call keyword arguments (line 217)
        kwargs_111993 = {}
        # Getting the type of 'companion' (line 217)
        companion_111988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'companion', False)
        # Calling companion(args, kwargs) (line 217)
        companion_call_result_111994 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), companion_111988, *[list_111989], **kwargs_111993)
        
        # Assigning a type to the variable 'c' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'c', companion_call_result_111994)
        
        # Assigning a Call to a Name (line 218):
        
        # Call to array(...): (line 218)
        # Processing the call arguments (line 218)
        
        # Obtaining an instance of the builtin type 'list' (line 218)
        list_111996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 218)
        # Adding element type (line 218)
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_111997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        # Adding element type (line 219)
        float_111998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 12), list_111997, float_111998)
        # Adding element type (line 219)
        float_111999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 12), list_111997, float_111999)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 25), list_111996, list_111997)
        # Adding element type (line 218)
        
        # Obtaining an instance of the builtin type 'list' (line 220)
        list_112000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 220)
        # Adding element type (line 220)
        float_112001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 12), list_112000, float_112001)
        # Adding element type (line 220)
        float_112002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 12), list_112000, float_112002)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 25), list_111996, list_112000)
        
        # Processing the call keyword arguments (line 218)
        kwargs_112003 = {}
        # Getting the type of 'array' (line 218)
        array_111995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 19), 'array', False)
        # Calling array(args, kwargs) (line 218)
        array_call_result_112004 = invoke(stypy.reporting.localization.Localization(__file__, 218, 19), array_111995, *[list_111996], **kwargs_112003)
        
        # Assigning a type to the variable 'expected' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'expected', array_call_result_112004)
        
        # Call to assert_array_equal(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'c' (line 221)
        c_112006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 27), 'c', False)
        # Getting the type of 'expected' (line 221)
        expected_112007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 30), 'expected', False)
        # Processing the call keyword arguments (line 221)
        kwargs_112008 = {}
        # Getting the type of 'assert_array_equal' (line 221)
        assert_array_equal_112005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 221)
        assert_array_equal_call_result_112009 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), assert_array_equal_112005, *[c_112006, expected_112007], **kwargs_112008)
        
        
        # Assigning a Call to a Name (line 223):
        
        # Call to companion(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_112011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        float_112012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 22), list_112011, float_112012)
        # Adding element type (line 223)
        float_112013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 22), list_112011, float_112013)
        # Adding element type (line 223)
        float_112014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 22), list_112011, float_112014)
        
        # Processing the call keyword arguments (line 223)
        kwargs_112015 = {}
        # Getting the type of 'companion' (line 223)
        companion_112010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'companion', False)
        # Calling companion(args, kwargs) (line 223)
        companion_call_result_112016 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), companion_112010, *[list_112011], **kwargs_112015)
        
        # Assigning a type to the variable 'c' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'c', companion_call_result_112016)
        
        # Assigning a Call to a Name (line 224):
        
        # Call to array(...): (line 224)
        # Processing the call arguments (line 224)
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_112018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        
        # Obtaining an instance of the builtin type 'list' (line 225)
        list_112019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 225)
        # Adding element type (line 225)
        float_112020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 12), list_112019, float_112020)
        # Adding element type (line 225)
        float_112021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 12), list_112019, float_112021)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 25), list_112018, list_112019)
        # Adding element type (line 224)
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_112022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        float_112023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 12), list_112022, float_112023)
        # Adding element type (line 226)
        float_112024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 12), list_112022, float_112024)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 25), list_112018, list_112022)
        
        # Processing the call keyword arguments (line 224)
        kwargs_112025 = {}
        # Getting the type of 'array' (line 224)
        array_112017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'array', False)
        # Calling array(args, kwargs) (line 224)
        array_call_result_112026 = invoke(stypy.reporting.localization.Localization(__file__, 224, 19), array_112017, *[list_112018], **kwargs_112025)
        
        # Assigning a type to the variable 'expected' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'expected', array_call_result_112026)
        
        # Call to assert_array_equal(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'c' (line 227)
        c_112028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 27), 'c', False)
        # Getting the type of 'expected' (line 227)
        expected_112029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 30), 'expected', False)
        # Processing the call keyword arguments (line 227)
        kwargs_112030 = {}
        # Getting the type of 'assert_array_equal' (line 227)
        assert_array_equal_112027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 227)
        assert_array_equal_call_result_112031 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), assert_array_equal_112027, *[c_112028, expected_112029], **kwargs_112030)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_112032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112032)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_112032


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 208, 0, False)
        # Assigning a type to the variable 'self' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCompanion.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCompanion' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'TestCompanion', TestCompanion)
# Declaration of the 'TestBlockDiag' class

class TestBlockDiag:

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBlockDiag.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestBlockDiag.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBlockDiag.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBlockDiag.test_basic.__dict__.__setitem__('stypy_function_name', 'TestBlockDiag.test_basic')
        TestBlockDiag.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestBlockDiag.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBlockDiag.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBlockDiag.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBlockDiag.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBlockDiag.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBlockDiag.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlockDiag.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 232):
        
        # Call to block_diag(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to eye(...): (line 232)
        # Processing the call arguments (line 232)
        int_112035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 27), 'int')
        # Processing the call keyword arguments (line 232)
        kwargs_112036 = {}
        # Getting the type of 'eye' (line 232)
        eye_112034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'eye', False)
        # Calling eye(args, kwargs) (line 232)
        eye_call_result_112037 = invoke(stypy.reporting.localization.Localization(__file__, 232, 23), eye_112034, *[int_112035], **kwargs_112036)
        
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_112038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_112039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        int_112040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 32), list_112039, int_112040)
        # Adding element type (line 232)
        int_112041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 32), list_112039, int_112041)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 31), list_112038, list_112039)
        # Adding element type (line 232)
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_112042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        int_112043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 39), list_112042, int_112043)
        # Adding element type (line 232)
        int_112044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 39), list_112042, int_112044)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 31), list_112038, list_112042)
        # Adding element type (line 232)
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_112045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        int_112046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 46), list_112045, int_112046)
        # Adding element type (line 232)
        int_112047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 46), list_112045, int_112047)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 31), list_112038, list_112045)
        
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_112048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_112049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        int_112050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 55), list_112049, int_112050)
        # Adding element type (line 232)
        int_112051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 55), list_112049, int_112051)
        # Adding element type (line 232)
        int_112052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 55), list_112049, int_112052)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 54), list_112048, list_112049)
        
        # Processing the call keyword arguments (line 232)
        kwargs_112053 = {}
        # Getting the type of 'block_diag' (line 232)
        block_diag_112033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'block_diag', False)
        # Calling block_diag(args, kwargs) (line 232)
        block_diag_call_result_112054 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), block_diag_112033, *[eye_call_result_112037, list_112038, list_112048], **kwargs_112053)
        
        # Assigning a type to the variable 'x' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'x', block_diag_call_result_112054)
        
        # Call to assert_array_equal(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'x' (line 233)
        x_112056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'x', False)
        
        # Obtaining an instance of the builtin type 'list' (line 233)
        list_112057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 233)
        # Adding element type (line 233)
        
        # Obtaining an instance of the builtin type 'list' (line 233)
        list_112058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 233)
        # Adding element type (line 233)
        int_112059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 31), list_112058, int_112059)
        # Adding element type (line 233)
        int_112060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 31), list_112058, int_112060)
        # Adding element type (line 233)
        int_112061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 31), list_112058, int_112061)
        # Adding element type (line 233)
        int_112062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 31), list_112058, int_112062)
        # Adding element type (line 233)
        int_112063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 31), list_112058, int_112063)
        # Adding element type (line 233)
        int_112064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 31), list_112058, int_112064)
        # Adding element type (line 233)
        int_112065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 31), list_112058, int_112065)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 30), list_112057, list_112058)
        # Adding element type (line 233)
        
        # Obtaining an instance of the builtin type 'list' (line 234)
        list_112066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 234)
        # Adding element type (line 234)
        int_112067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 31), list_112066, int_112067)
        # Adding element type (line 234)
        int_112068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 31), list_112066, int_112068)
        # Adding element type (line 234)
        int_112069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 31), list_112066, int_112069)
        # Adding element type (line 234)
        int_112070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 31), list_112066, int_112070)
        # Adding element type (line 234)
        int_112071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 31), list_112066, int_112071)
        # Adding element type (line 234)
        int_112072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 31), list_112066, int_112072)
        # Adding element type (line 234)
        int_112073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 31), list_112066, int_112073)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 30), list_112057, list_112066)
        # Adding element type (line 233)
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_112074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        int_112075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 31), list_112074, int_112075)
        # Adding element type (line 235)
        int_112076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 31), list_112074, int_112076)
        # Adding element type (line 235)
        int_112077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 31), list_112074, int_112077)
        # Adding element type (line 235)
        int_112078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 31), list_112074, int_112078)
        # Adding element type (line 235)
        int_112079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 31), list_112074, int_112079)
        # Adding element type (line 235)
        int_112080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 31), list_112074, int_112080)
        # Adding element type (line 235)
        int_112081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 31), list_112074, int_112081)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 30), list_112057, list_112074)
        # Adding element type (line 233)
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_112082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        # Adding element type (line 236)
        int_112083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 31), list_112082, int_112083)
        # Adding element type (line 236)
        int_112084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 31), list_112082, int_112084)
        # Adding element type (line 236)
        int_112085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 31), list_112082, int_112085)
        # Adding element type (line 236)
        int_112086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 31), list_112082, int_112086)
        # Adding element type (line 236)
        int_112087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 31), list_112082, int_112087)
        # Adding element type (line 236)
        int_112088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 31), list_112082, int_112088)
        # Adding element type (line 236)
        int_112089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 31), list_112082, int_112089)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 30), list_112057, list_112082)
        # Adding element type (line 233)
        
        # Obtaining an instance of the builtin type 'list' (line 237)
        list_112090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 237)
        # Adding element type (line 237)
        int_112091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 31), list_112090, int_112091)
        # Adding element type (line 237)
        int_112092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 31), list_112090, int_112092)
        # Adding element type (line 237)
        int_112093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 31), list_112090, int_112093)
        # Adding element type (line 237)
        int_112094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 31), list_112090, int_112094)
        # Adding element type (line 237)
        int_112095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 31), list_112090, int_112095)
        # Adding element type (line 237)
        int_112096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 31), list_112090, int_112096)
        # Adding element type (line 237)
        int_112097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 31), list_112090, int_112097)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 30), list_112057, list_112090)
        # Adding element type (line 233)
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_112098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        int_112099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 31), list_112098, int_112099)
        # Adding element type (line 238)
        int_112100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 31), list_112098, int_112100)
        # Adding element type (line 238)
        int_112101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 31), list_112098, int_112101)
        # Adding element type (line 238)
        int_112102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 31), list_112098, int_112102)
        # Adding element type (line 238)
        int_112103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 31), list_112098, int_112103)
        # Adding element type (line 238)
        int_112104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 31), list_112098, int_112104)
        # Adding element type (line 238)
        int_112105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 31), list_112098, int_112105)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 30), list_112057, list_112098)
        
        # Processing the call keyword arguments (line 233)
        kwargs_112106 = {}
        # Getting the type of 'assert_array_equal' (line 233)
        assert_array_equal_112055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 233)
        assert_array_equal_call_result_112107 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), assert_array_equal_112055, *[x_112056, list_112057], **kwargs_112106)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_112108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112108)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_112108


    @norecursion
    def test_dtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dtype'
        module_type_store = module_type_store.open_function_context('test_dtype', 240, 4, False)
        # Assigning a type to the variable 'self' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBlockDiag.test_dtype.__dict__.__setitem__('stypy_localization', localization)
        TestBlockDiag.test_dtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBlockDiag.test_dtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBlockDiag.test_dtype.__dict__.__setitem__('stypy_function_name', 'TestBlockDiag.test_dtype')
        TestBlockDiag.test_dtype.__dict__.__setitem__('stypy_param_names_list', [])
        TestBlockDiag.test_dtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBlockDiag.test_dtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBlockDiag.test_dtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBlockDiag.test_dtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBlockDiag.test_dtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBlockDiag.test_dtype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlockDiag.test_dtype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dtype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dtype(...)' code ##################

        
        # Assigning a Call to a Name (line 241):
        
        # Call to block_diag(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Obtaining an instance of the builtin type 'list' (line 241)
        list_112110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 241)
        # Adding element type (line 241)
        
        # Obtaining an instance of the builtin type 'list' (line 241)
        list_112111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 241)
        # Adding element type (line 241)
        float_112112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 24), list_112111, float_112112)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 23), list_112110, list_112111)
        
        # Processing the call keyword arguments (line 241)
        kwargs_112113 = {}
        # Getting the type of 'block_diag' (line 241)
        block_diag_112109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'block_diag', False)
        # Calling block_diag(args, kwargs) (line 241)
        block_diag_call_result_112114 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), block_diag_112109, *[list_112110], **kwargs_112113)
        
        # Assigning a type to the variable 'x' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'x', block_diag_call_result_112114)
        
        # Call to assert_equal(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'x' (line 242)
        x_112116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 242)
        dtype_112117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 21), x_112116, 'dtype')
        # Getting the type of 'float' (line 242)
        float_112118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 30), 'float', False)
        # Processing the call keyword arguments (line 242)
        kwargs_112119 = {}
        # Getting the type of 'assert_equal' (line 242)
        assert_equal_112115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 242)
        assert_equal_call_result_112120 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), assert_equal_112115, *[dtype_112117, float_112118], **kwargs_112119)
        
        
        # Assigning a Call to a Name (line 244):
        
        # Call to block_diag(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Obtaining an instance of the builtin type 'list' (line 244)
        list_112122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 244)
        # Adding element type (line 244)
        
        # Obtaining an instance of the builtin type 'list' (line 244)
        list_112123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 244)
        # Adding element type (line 244)
        # Getting the type of 'True' (line 244)
        True_112124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 25), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), list_112123, True_112124)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 23), list_112122, list_112123)
        
        # Processing the call keyword arguments (line 244)
        kwargs_112125 = {}
        # Getting the type of 'block_diag' (line 244)
        block_diag_112121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'block_diag', False)
        # Calling block_diag(args, kwargs) (line 244)
        block_diag_call_result_112126 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), block_diag_112121, *[list_112122], **kwargs_112125)
        
        # Assigning a type to the variable 'x' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'x', block_diag_call_result_112126)
        
        # Call to assert_equal(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'x' (line 245)
        x_112128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 245)
        dtype_112129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 21), x_112128, 'dtype')
        # Getting the type of 'bool' (line 245)
        bool_112130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 30), 'bool', False)
        # Processing the call keyword arguments (line 245)
        kwargs_112131 = {}
        # Getting the type of 'assert_equal' (line 245)
        assert_equal_112127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 245)
        assert_equal_call_result_112132 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), assert_equal_112127, *[dtype_112129, bool_112130], **kwargs_112131)
        
        
        # ################# End of 'test_dtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dtype' in the type store
        # Getting the type of 'stypy_return_type' (line 240)
        stypy_return_type_112133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112133)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dtype'
        return stypy_return_type_112133


    @norecursion
    def test_mixed_dtypes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_mixed_dtypes'
        module_type_store = module_type_store.open_function_context('test_mixed_dtypes', 247, 4, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBlockDiag.test_mixed_dtypes.__dict__.__setitem__('stypy_localization', localization)
        TestBlockDiag.test_mixed_dtypes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBlockDiag.test_mixed_dtypes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBlockDiag.test_mixed_dtypes.__dict__.__setitem__('stypy_function_name', 'TestBlockDiag.test_mixed_dtypes')
        TestBlockDiag.test_mixed_dtypes.__dict__.__setitem__('stypy_param_names_list', [])
        TestBlockDiag.test_mixed_dtypes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBlockDiag.test_mixed_dtypes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBlockDiag.test_mixed_dtypes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBlockDiag.test_mixed_dtypes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBlockDiag.test_mixed_dtypes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBlockDiag.test_mixed_dtypes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlockDiag.test_mixed_dtypes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_mixed_dtypes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_mixed_dtypes(...)' code ##################

        
        # Assigning a Call to a Name (line 248):
        
        # Call to block_diag(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Obtaining an instance of the builtin type 'list' (line 248)
        list_112135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 248)
        # Adding element type (line 248)
        
        # Obtaining an instance of the builtin type 'list' (line 248)
        list_112136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 248)
        # Adding element type (line 248)
        int_112137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 29), list_112136, int_112137)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 28), list_112135, list_112136)
        
        
        # Obtaining an instance of the builtin type 'list' (line 248)
        list_112138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 248)
        # Adding element type (line 248)
        
        # Obtaining an instance of the builtin type 'list' (line 248)
        list_112139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 248)
        # Adding element type (line 248)
        complex_112140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 37), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 36), list_112139, complex_112140)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 35), list_112138, list_112139)
        
        # Processing the call keyword arguments (line 248)
        kwargs_112141 = {}
        # Getting the type of 'block_diag' (line 248)
        block_diag_112134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 17), 'block_diag', False)
        # Calling block_diag(args, kwargs) (line 248)
        block_diag_call_result_112142 = invoke(stypy.reporting.localization.Localization(__file__, 248, 17), block_diag_112134, *[list_112135, list_112138], **kwargs_112141)
        
        # Assigning a type to the variable 'actual' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'actual', block_diag_call_result_112142)
        
        # Assigning a Call to a Name (line 249):
        
        # Call to array(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Obtaining an instance of the builtin type 'list' (line 249)
        list_112145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 249)
        # Adding element type (line 249)
        
        # Obtaining an instance of the builtin type 'list' (line 249)
        list_112146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 249)
        # Adding element type (line 249)
        int_112147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 28), list_112146, int_112147)
        # Adding element type (line 249)
        int_112148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 28), list_112146, int_112148)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 27), list_112145, list_112146)
        # Adding element type (line 249)
        
        # Obtaining an instance of the builtin type 'list' (line 249)
        list_112149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 249)
        # Adding element type (line 249)
        int_112150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 36), list_112149, int_112150)
        # Adding element type (line 249)
        complex_112151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 40), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 36), list_112149, complex_112151)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 27), list_112145, list_112149)
        
        # Processing the call keyword arguments (line 249)
        kwargs_112152 = {}
        # Getting the type of 'np' (line 249)
        np_112143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 249)
        array_112144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 18), np_112143, 'array')
        # Calling array(args, kwargs) (line 249)
        array_call_result_112153 = invoke(stypy.reporting.localization.Localization(__file__, 249, 18), array_112144, *[list_112145], **kwargs_112152)
        
        # Assigning a type to the variable 'desired' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'desired', array_call_result_112153)
        
        # Call to assert_array_equal(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'actual' (line 250)
        actual_112155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'actual', False)
        # Getting the type of 'desired' (line 250)
        desired_112156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 35), 'desired', False)
        # Processing the call keyword arguments (line 250)
        kwargs_112157 = {}
        # Getting the type of 'assert_array_equal' (line 250)
        assert_array_equal_112154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 250)
        assert_array_equal_call_result_112158 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), assert_array_equal_112154, *[actual_112155, desired_112156], **kwargs_112157)
        
        
        # ################# End of 'test_mixed_dtypes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_mixed_dtypes' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_112159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112159)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_mixed_dtypes'
        return stypy_return_type_112159


    @norecursion
    def test_scalar_and_1d_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_and_1d_args'
        module_type_store = module_type_store.open_function_context('test_scalar_and_1d_args', 252, 4, False)
        # Assigning a type to the variable 'self' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBlockDiag.test_scalar_and_1d_args.__dict__.__setitem__('stypy_localization', localization)
        TestBlockDiag.test_scalar_and_1d_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBlockDiag.test_scalar_and_1d_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBlockDiag.test_scalar_and_1d_args.__dict__.__setitem__('stypy_function_name', 'TestBlockDiag.test_scalar_and_1d_args')
        TestBlockDiag.test_scalar_and_1d_args.__dict__.__setitem__('stypy_param_names_list', [])
        TestBlockDiag.test_scalar_and_1d_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBlockDiag.test_scalar_and_1d_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBlockDiag.test_scalar_and_1d_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBlockDiag.test_scalar_and_1d_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBlockDiag.test_scalar_and_1d_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBlockDiag.test_scalar_and_1d_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlockDiag.test_scalar_and_1d_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_and_1d_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_and_1d_args(...)' code ##################

        
        # Assigning a Call to a Name (line 253):
        
        # Call to block_diag(...): (line 253)
        # Processing the call arguments (line 253)
        int_112161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 23), 'int')
        # Processing the call keyword arguments (line 253)
        kwargs_112162 = {}
        # Getting the type of 'block_diag' (line 253)
        block_diag_112160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'block_diag', False)
        # Calling block_diag(args, kwargs) (line 253)
        block_diag_call_result_112163 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), block_diag_112160, *[int_112161], **kwargs_112162)
        
        # Assigning a type to the variable 'a' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'a', block_diag_call_result_112163)
        
        # Call to assert_equal(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'a' (line 254)
        a_112165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 21), 'a', False)
        # Obtaining the member 'shape' of a type (line 254)
        shape_112166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 21), a_112165, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 254)
        tuple_112167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 254)
        # Adding element type (line 254)
        int_112168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_112167, int_112168)
        # Adding element type (line 254)
        int_112169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_112167, int_112169)
        
        # Processing the call keyword arguments (line 254)
        kwargs_112170 = {}
        # Getting the type of 'assert_equal' (line 254)
        assert_equal_112164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 254)
        assert_equal_call_result_112171 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), assert_equal_112164, *[shape_112166, tuple_112167], **kwargs_112170)
        
        
        # Call to assert_array_equal(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'a' (line 255)
        a_112173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'a', False)
        
        # Obtaining an instance of the builtin type 'list' (line 255)
        list_112174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 255)
        # Adding element type (line 255)
        
        # Obtaining an instance of the builtin type 'list' (line 255)
        list_112175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 255)
        # Adding element type (line 255)
        int_112176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 31), list_112175, int_112176)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 30), list_112174, list_112175)
        
        # Processing the call keyword arguments (line 255)
        kwargs_112177 = {}
        # Getting the type of 'assert_array_equal' (line 255)
        assert_array_equal_112172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 255)
        assert_array_equal_call_result_112178 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), assert_array_equal_112172, *[a_112173, list_112174], **kwargs_112177)
        
        
        # Assigning a Call to a Name (line 257):
        
        # Call to block_diag(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Obtaining an instance of the builtin type 'list' (line 257)
        list_112180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 257)
        # Adding element type (line 257)
        int_112181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 23), list_112180, int_112181)
        # Adding element type (line 257)
        int_112182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 23), list_112180, int_112182)
        
        int_112183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 30), 'int')
        # Processing the call keyword arguments (line 257)
        kwargs_112184 = {}
        # Getting the type of 'block_diag' (line 257)
        block_diag_112179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'block_diag', False)
        # Calling block_diag(args, kwargs) (line 257)
        block_diag_call_result_112185 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), block_diag_112179, *[list_112180, int_112183], **kwargs_112184)
        
        # Assigning a type to the variable 'a' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'a', block_diag_call_result_112185)
        
        # Call to assert_array_equal(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'a' (line 258)
        a_112187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 27), 'a', False)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_112188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_112189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        int_112190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 31), list_112189, int_112190)
        # Adding element type (line 258)
        int_112191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 31), list_112189, int_112191)
        # Adding element type (line 258)
        int_112192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 31), list_112189, int_112192)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 30), list_112188, list_112189)
        # Adding element type (line 258)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_112193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        int_112194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 42), list_112193, int_112194)
        # Adding element type (line 258)
        int_112195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 42), list_112193, int_112195)
        # Adding element type (line 258)
        int_112196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 42), list_112193, int_112196)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 30), list_112188, list_112193)
        
        # Processing the call keyword arguments (line 258)
        kwargs_112197 = {}
        # Getting the type of 'assert_array_equal' (line 258)
        assert_array_equal_112186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 258)
        assert_array_equal_call_result_112198 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), assert_array_equal_112186, *[a_112187, list_112188], **kwargs_112197)
        
        
        # ################# End of 'test_scalar_and_1d_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_and_1d_args' in the type store
        # Getting the type of 'stypy_return_type' (line 252)
        stypy_return_type_112199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112199)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_and_1d_args'
        return stypy_return_type_112199


    @norecursion
    def test_bad_arg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bad_arg'
        module_type_store = module_type_store.open_function_context('test_bad_arg', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBlockDiag.test_bad_arg.__dict__.__setitem__('stypy_localization', localization)
        TestBlockDiag.test_bad_arg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBlockDiag.test_bad_arg.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBlockDiag.test_bad_arg.__dict__.__setitem__('stypy_function_name', 'TestBlockDiag.test_bad_arg')
        TestBlockDiag.test_bad_arg.__dict__.__setitem__('stypy_param_names_list', [])
        TestBlockDiag.test_bad_arg.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBlockDiag.test_bad_arg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBlockDiag.test_bad_arg.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBlockDiag.test_bad_arg.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBlockDiag.test_bad_arg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBlockDiag.test_bad_arg.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlockDiag.test_bad_arg', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bad_arg', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bad_arg(...)' code ##################

        
        # Call to assert_raises(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'ValueError' (line 261)
        ValueError_112201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 22), 'ValueError', False)
        # Getting the type of 'block_diag' (line 261)
        block_diag_112202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 34), 'block_diag', False)
        
        # Obtaining an instance of the builtin type 'list' (line 261)
        list_112203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 261)
        # Adding element type (line 261)
        
        # Obtaining an instance of the builtin type 'list' (line 261)
        list_112204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 261)
        # Adding element type (line 261)
        
        # Obtaining an instance of the builtin type 'list' (line 261)
        list_112205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 261)
        # Adding element type (line 261)
        int_112206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 48), list_112205, int_112206)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 47), list_112204, list_112205)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 46), list_112203, list_112204)
        
        # Processing the call keyword arguments (line 261)
        kwargs_112207 = {}
        # Getting the type of 'assert_raises' (line 261)
        assert_raises_112200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 261)
        assert_raises_call_result_112208 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), assert_raises_112200, *[ValueError_112201, block_diag_112202, list_112203], **kwargs_112207)
        
        
        # ################# End of 'test_bad_arg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bad_arg' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_112209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112209)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bad_arg'
        return stypy_return_type_112209


    @norecursion
    def test_no_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_no_args'
        module_type_store = module_type_store.open_function_context('test_no_args', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBlockDiag.test_no_args.__dict__.__setitem__('stypy_localization', localization)
        TestBlockDiag.test_no_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBlockDiag.test_no_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBlockDiag.test_no_args.__dict__.__setitem__('stypy_function_name', 'TestBlockDiag.test_no_args')
        TestBlockDiag.test_no_args.__dict__.__setitem__('stypy_param_names_list', [])
        TestBlockDiag.test_no_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBlockDiag.test_no_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBlockDiag.test_no_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBlockDiag.test_no_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBlockDiag.test_no_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBlockDiag.test_no_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlockDiag.test_no_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_no_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_no_args(...)' code ##################

        
        # Assigning a Call to a Name (line 264):
        
        # Call to block_diag(...): (line 264)
        # Processing the call keyword arguments (line 264)
        kwargs_112211 = {}
        # Getting the type of 'block_diag' (line 264)
        block_diag_112210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'block_diag', False)
        # Calling block_diag(args, kwargs) (line 264)
        block_diag_call_result_112212 = invoke(stypy.reporting.localization.Localization(__file__, 264, 12), block_diag_112210, *[], **kwargs_112211)
        
        # Assigning a type to the variable 'a' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'a', block_diag_call_result_112212)
        
        # Call to assert_equal(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'a' (line 265)
        a_112214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 21), 'a', False)
        # Obtaining the member 'ndim' of a type (line 265)
        ndim_112215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 21), a_112214, 'ndim')
        int_112216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 29), 'int')
        # Processing the call keyword arguments (line 265)
        kwargs_112217 = {}
        # Getting the type of 'assert_equal' (line 265)
        assert_equal_112213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 265)
        assert_equal_call_result_112218 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), assert_equal_112213, *[ndim_112215, int_112216], **kwargs_112217)
        
        
        # Call to assert_equal(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'a' (line 266)
        a_112220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 21), 'a', False)
        # Obtaining the member 'nbytes' of a type (line 266)
        nbytes_112221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 21), a_112220, 'nbytes')
        int_112222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 31), 'int')
        # Processing the call keyword arguments (line 266)
        kwargs_112223 = {}
        # Getting the type of 'assert_equal' (line 266)
        assert_equal_112219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 266)
        assert_equal_call_result_112224 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), assert_equal_112219, *[nbytes_112221, int_112222], **kwargs_112223)
        
        
        # ################# End of 'test_no_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_no_args' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_112225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112225)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_no_args'
        return stypy_return_type_112225


    @norecursion
    def test_empty_matrix_arg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_empty_matrix_arg'
        module_type_store = module_type_store.open_function_context('test_empty_matrix_arg', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBlockDiag.test_empty_matrix_arg.__dict__.__setitem__('stypy_localization', localization)
        TestBlockDiag.test_empty_matrix_arg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBlockDiag.test_empty_matrix_arg.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBlockDiag.test_empty_matrix_arg.__dict__.__setitem__('stypy_function_name', 'TestBlockDiag.test_empty_matrix_arg')
        TestBlockDiag.test_empty_matrix_arg.__dict__.__setitem__('stypy_param_names_list', [])
        TestBlockDiag.test_empty_matrix_arg.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBlockDiag.test_empty_matrix_arg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBlockDiag.test_empty_matrix_arg.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBlockDiag.test_empty_matrix_arg.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBlockDiag.test_empty_matrix_arg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBlockDiag.test_empty_matrix_arg.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlockDiag.test_empty_matrix_arg', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_empty_matrix_arg', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_empty_matrix_arg(...)' code ##################

        
        # Assigning a Call to a Name (line 272):
        
        # Call to block_diag(...): (line 272)
        # Processing the call arguments (line 272)
        
        # Obtaining an instance of the builtin type 'list' (line 272)
        list_112227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 272)
        # Adding element type (line 272)
        
        # Obtaining an instance of the builtin type 'list' (line 272)
        list_112228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 272)
        # Adding element type (line 272)
        int_112229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 24), list_112228, int_112229)
        # Adding element type (line 272)
        int_112230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 24), list_112228, int_112230)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 23), list_112227, list_112228)
        # Adding element type (line 272)
        
        # Obtaining an instance of the builtin type 'list' (line 272)
        list_112231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 272)
        # Adding element type (line 272)
        int_112232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 32), list_112231, int_112232)
        # Adding element type (line 272)
        int_112233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 32), list_112231, int_112233)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 23), list_112227, list_112231)
        
        
        # Obtaining an instance of the builtin type 'list' (line 273)
        list_112234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 273)
        
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_112235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_112236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        int_112237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 24), list_112236, int_112237)
        # Adding element type (line 274)
        int_112238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 24), list_112236, int_112238)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 23), list_112235, list_112236)
        # Adding element type (line 274)
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_112239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        int_112240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 32), list_112239, int_112240)
        # Adding element type (line 274)
        int_112241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 32), list_112239, int_112241)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 23), list_112235, list_112239)
        # Adding element type (line 274)
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_112242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        int_112243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 40), list_112242, int_112243)
        # Adding element type (line 274)
        int_112244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 40), list_112242, int_112244)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 23), list_112235, list_112242)
        
        # Processing the call keyword arguments (line 272)
        kwargs_112245 = {}
        # Getting the type of 'block_diag' (line 272)
        block_diag_112226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'block_diag', False)
        # Calling block_diag(args, kwargs) (line 272)
        block_diag_call_result_112246 = invoke(stypy.reporting.localization.Localization(__file__, 272, 12), block_diag_112226, *[list_112227, list_112234, list_112235], **kwargs_112245)
        
        # Assigning a type to the variable 'a' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'a', block_diag_call_result_112246)
        
        # Call to assert_array_equal(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'a' (line 275)
        a_112248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'a', False)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_112249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_112250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        int_112251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 31), list_112250, int_112251)
        # Adding element type (line 275)
        int_112252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 31), list_112250, int_112252)
        # Adding element type (line 275)
        int_112253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 31), list_112250, int_112253)
        # Adding element type (line 275)
        int_112254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 31), list_112250, int_112254)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 30), list_112249, list_112250)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_112255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        # Adding element type (line 276)
        int_112256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 31), list_112255, int_112256)
        # Adding element type (line 276)
        int_112257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 31), list_112255, int_112257)
        # Adding element type (line 276)
        int_112258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 31), list_112255, int_112258)
        # Adding element type (line 276)
        int_112259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 31), list_112255, int_112259)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 30), list_112249, list_112255)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 277)
        list_112260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 277)
        # Adding element type (line 277)
        int_112261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 31), list_112260, int_112261)
        # Adding element type (line 277)
        int_112262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 31), list_112260, int_112262)
        # Adding element type (line 277)
        int_112263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 31), list_112260, int_112263)
        # Adding element type (line 277)
        int_112264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 31), list_112260, int_112264)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 30), list_112249, list_112260)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_112265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        # Adding element type (line 278)
        int_112266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 31), list_112265, int_112266)
        # Adding element type (line 278)
        int_112267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 31), list_112265, int_112267)
        # Adding element type (line 278)
        int_112268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 31), list_112265, int_112268)
        # Adding element type (line 278)
        int_112269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 31), list_112265, int_112269)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 30), list_112249, list_112265)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_112270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        # Adding element type (line 279)
        int_112271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 31), list_112270, int_112271)
        # Adding element type (line 279)
        int_112272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 31), list_112270, int_112272)
        # Adding element type (line 279)
        int_112273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 31), list_112270, int_112273)
        # Adding element type (line 279)
        int_112274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 31), list_112270, int_112274)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 30), list_112249, list_112270)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_112275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        int_112276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 31), list_112275, int_112276)
        # Adding element type (line 280)
        int_112277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 31), list_112275, int_112277)
        # Adding element type (line 280)
        int_112278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 31), list_112275, int_112278)
        # Adding element type (line 280)
        int_112279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 31), list_112275, int_112279)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 30), list_112249, list_112275)
        
        # Processing the call keyword arguments (line 275)
        kwargs_112280 = {}
        # Getting the type of 'assert_array_equal' (line 275)
        assert_array_equal_112247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 275)
        assert_array_equal_call_result_112281 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), assert_array_equal_112247, *[a_112248, list_112249], **kwargs_112280)
        
        
        # ################# End of 'test_empty_matrix_arg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_empty_matrix_arg' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_112282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112282)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_empty_matrix_arg'
        return stypy_return_type_112282


    @norecursion
    def test_zerosized_matrix_arg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_zerosized_matrix_arg'
        module_type_store = module_type_store.open_function_context('test_zerosized_matrix_arg', 282, 4, False)
        # Assigning a type to the variable 'self' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBlockDiag.test_zerosized_matrix_arg.__dict__.__setitem__('stypy_localization', localization)
        TestBlockDiag.test_zerosized_matrix_arg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBlockDiag.test_zerosized_matrix_arg.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBlockDiag.test_zerosized_matrix_arg.__dict__.__setitem__('stypy_function_name', 'TestBlockDiag.test_zerosized_matrix_arg')
        TestBlockDiag.test_zerosized_matrix_arg.__dict__.__setitem__('stypy_param_names_list', [])
        TestBlockDiag.test_zerosized_matrix_arg.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBlockDiag.test_zerosized_matrix_arg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBlockDiag.test_zerosized_matrix_arg.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBlockDiag.test_zerosized_matrix_arg.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBlockDiag.test_zerosized_matrix_arg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBlockDiag.test_zerosized_matrix_arg.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlockDiag.test_zerosized_matrix_arg', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_zerosized_matrix_arg', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_zerosized_matrix_arg(...)' code ##################

        
        # Assigning a Call to a Name (line 286):
        
        # Call to block_diag(...): (line 286)
        # Processing the call arguments (line 286)
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_112284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        # Adding element type (line 286)
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_112285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        # Adding element type (line 286)
        int_112286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 24), list_112285, int_112286)
        # Adding element type (line 286)
        int_112287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 24), list_112285, int_112287)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 23), list_112284, list_112285)
        # Adding element type (line 286)
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_112288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        # Adding element type (line 286)
        int_112289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 32), list_112288, int_112289)
        # Adding element type (line 286)
        int_112290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 32), list_112288, int_112290)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 23), list_112284, list_112288)
        
        
        # Obtaining an instance of the builtin type 'list' (line 287)
        list_112291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 287)
        # Adding element type (line 287)
        
        # Obtaining an instance of the builtin type 'list' (line 287)
        list_112292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 287)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 23), list_112291, list_112292)
        
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_112293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        # Adding element type (line 288)
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_112294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        # Adding element type (line 288)
        int_112295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 24), list_112294, int_112295)
        # Adding element type (line 288)
        int_112296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 24), list_112294, int_112296)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 23), list_112293, list_112294)
        # Adding element type (line 288)
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_112297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        # Adding element type (line 288)
        int_112298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 32), list_112297, int_112298)
        # Adding element type (line 288)
        int_112299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 32), list_112297, int_112299)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 23), list_112293, list_112297)
        # Adding element type (line 288)
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_112300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        # Adding element type (line 288)
        int_112301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 40), list_112300, int_112301)
        # Adding element type (line 288)
        int_112302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 40), list_112300, int_112302)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 23), list_112293, list_112300)
        
        
        # Call to zeros(...): (line 289)
        # Processing the call arguments (line 289)
        
        # Obtaining an instance of the builtin type 'list' (line 289)
        list_112305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 289)
        # Adding element type (line 289)
        int_112306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 32), list_112305, int_112306)
        # Adding element type (line 289)
        int_112307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 32), list_112305, int_112307)
        
        # Processing the call keyword arguments (line 289)
        str_112308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 44), 'str', 'int32')
        keyword_112309 = str_112308
        kwargs_112310 = {'dtype': keyword_112309}
        # Getting the type of 'np' (line 289)
        np_112303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 23), 'np', False)
        # Obtaining the member 'zeros' of a type (line 289)
        zeros_112304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 23), np_112303, 'zeros')
        # Calling zeros(args, kwargs) (line 289)
        zeros_call_result_112311 = invoke(stypy.reporting.localization.Localization(__file__, 289, 23), zeros_112304, *[list_112305], **kwargs_112310)
        
        # Processing the call keyword arguments (line 286)
        kwargs_112312 = {}
        # Getting the type of 'block_diag' (line 286)
        block_diag_112283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'block_diag', False)
        # Calling block_diag(args, kwargs) (line 286)
        block_diag_call_result_112313 = invoke(stypy.reporting.localization.Localization(__file__, 286, 12), block_diag_112283, *[list_112284, list_112291, list_112293, zeros_call_result_112311], **kwargs_112312)
        
        # Assigning a type to the variable 'a' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'a', block_diag_call_result_112313)
        
        # Call to assert_array_equal(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'a' (line 290)
        a_112315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 27), 'a', False)
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_112316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        # Adding element type (line 290)
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_112317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        # Adding element type (line 290)
        int_112318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 31), list_112317, int_112318)
        # Adding element type (line 290)
        int_112319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 31), list_112317, int_112319)
        # Adding element type (line 290)
        int_112320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 31), list_112317, int_112320)
        # Adding element type (line 290)
        int_112321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 31), list_112317, int_112321)
        # Adding element type (line 290)
        int_112322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 31), list_112317, int_112322)
        # Adding element type (line 290)
        int_112323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 31), list_112317, int_112323)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 30), list_112316, list_112317)
        # Adding element type (line 290)
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_112324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        int_112325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 31), list_112324, int_112325)
        # Adding element type (line 291)
        int_112326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 31), list_112324, int_112326)
        # Adding element type (line 291)
        int_112327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 31), list_112324, int_112327)
        # Adding element type (line 291)
        int_112328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 31), list_112324, int_112328)
        # Adding element type (line 291)
        int_112329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 31), list_112324, int_112329)
        # Adding element type (line 291)
        int_112330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 31), list_112324, int_112330)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 30), list_112316, list_112324)
        # Adding element type (line 290)
        
        # Obtaining an instance of the builtin type 'list' (line 292)
        list_112331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 292)
        # Adding element type (line 292)
        int_112332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 31), list_112331, int_112332)
        # Adding element type (line 292)
        int_112333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 31), list_112331, int_112333)
        # Adding element type (line 292)
        int_112334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 31), list_112331, int_112334)
        # Adding element type (line 292)
        int_112335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 31), list_112331, int_112335)
        # Adding element type (line 292)
        int_112336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 31), list_112331, int_112336)
        # Adding element type (line 292)
        int_112337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 31), list_112331, int_112337)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 30), list_112316, list_112331)
        # Adding element type (line 290)
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_112338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        int_112339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 31), list_112338, int_112339)
        # Adding element type (line 293)
        int_112340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 31), list_112338, int_112340)
        # Adding element type (line 293)
        int_112341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 31), list_112338, int_112341)
        # Adding element type (line 293)
        int_112342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 31), list_112338, int_112342)
        # Adding element type (line 293)
        int_112343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 31), list_112338, int_112343)
        # Adding element type (line 293)
        int_112344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 31), list_112338, int_112344)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 30), list_112316, list_112338)
        # Adding element type (line 290)
        
        # Obtaining an instance of the builtin type 'list' (line 294)
        list_112345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 294)
        # Adding element type (line 294)
        int_112346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 31), list_112345, int_112346)
        # Adding element type (line 294)
        int_112347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 31), list_112345, int_112347)
        # Adding element type (line 294)
        int_112348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 31), list_112345, int_112348)
        # Adding element type (line 294)
        int_112349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 31), list_112345, int_112349)
        # Adding element type (line 294)
        int_112350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 31), list_112345, int_112350)
        # Adding element type (line 294)
        int_112351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 31), list_112345, int_112351)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 30), list_112316, list_112345)
        # Adding element type (line 290)
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_112352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        int_112353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 31), list_112352, int_112353)
        # Adding element type (line 295)
        int_112354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 31), list_112352, int_112354)
        # Adding element type (line 295)
        int_112355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 31), list_112352, int_112355)
        # Adding element type (line 295)
        int_112356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 31), list_112352, int_112356)
        # Adding element type (line 295)
        int_112357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 31), list_112352, int_112357)
        # Adding element type (line 295)
        int_112358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 31), list_112352, int_112358)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 30), list_112316, list_112352)
        
        # Processing the call keyword arguments (line 290)
        kwargs_112359 = {}
        # Getting the type of 'assert_array_equal' (line 290)
        assert_array_equal_112314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 290)
        assert_array_equal_call_result_112360 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), assert_array_equal_112314, *[a_112315, list_112316], **kwargs_112359)
        
        
        # ################# End of 'test_zerosized_matrix_arg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_zerosized_matrix_arg' in the type store
        # Getting the type of 'stypy_return_type' (line 282)
        stypy_return_type_112361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112361)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_zerosized_matrix_arg'
        return stypy_return_type_112361


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 230, 0, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlockDiag.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBlockDiag' (line 230)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'TestBlockDiag', TestBlockDiag)
# Declaration of the 'TestKron' class

class TestKron:

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 299, 4, False)
        # Assigning a type to the variable 'self' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKron.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestKron.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKron.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKron.test_basic.__dict__.__setitem__('stypy_function_name', 'TestKron.test_basic')
        TestKron.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestKron.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKron.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKron.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKron.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKron.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKron.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKron.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 301):
        
        # Call to kron(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Call to array(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_112364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        # Adding element type (line 301)
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_112365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        # Adding element type (line 301)
        int_112366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 24), list_112365, int_112366)
        # Adding element type (line 301)
        int_112367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 24), list_112365, int_112367)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 23), list_112364, list_112365)
        # Adding element type (line 301)
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_112368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        # Adding element type (line 301)
        int_112369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 32), list_112368, int_112369)
        # Adding element type (line 301)
        int_112370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 32), list_112368, int_112370)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 23), list_112364, list_112368)
        
        # Processing the call keyword arguments (line 301)
        kwargs_112371 = {}
        # Getting the type of 'array' (line 301)
        array_112363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 17), 'array', False)
        # Calling array(args, kwargs) (line 301)
        array_call_result_112372 = invoke(stypy.reporting.localization.Localization(__file__, 301, 17), array_112363, *[list_112364], **kwargs_112371)
        
        
        # Call to array(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_112374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        # Adding element type (line 301)
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_112375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        # Adding element type (line 301)
        int_112376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 49), list_112375, int_112376)
        # Adding element type (line 301)
        int_112377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 49), list_112375, int_112377)
        # Adding element type (line 301)
        int_112378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 49), list_112375, int_112378)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 48), list_112374, list_112375)
        
        # Processing the call keyword arguments (line 301)
        kwargs_112379 = {}
        # Getting the type of 'array' (line 301)
        array_112373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 42), 'array', False)
        # Calling array(args, kwargs) (line 301)
        array_call_result_112380 = invoke(stypy.reporting.localization.Localization(__file__, 301, 42), array_112373, *[list_112374], **kwargs_112379)
        
        # Processing the call keyword arguments (line 301)
        kwargs_112381 = {}
        # Getting the type of 'kron' (line 301)
        kron_112362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'kron', False)
        # Calling kron(args, kwargs) (line 301)
        kron_call_result_112382 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), kron_112362, *[array_call_result_112372, array_call_result_112380], **kwargs_112381)
        
        # Assigning a type to the variable 'a' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'a', kron_call_result_112382)
        
        # Call to assert_array_equal(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'a' (line 302)
        a_112384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 27), 'a', False)
        
        # Call to array(...): (line 302)
        # Processing the call arguments (line 302)
        
        # Obtaining an instance of the builtin type 'list' (line 302)
        list_112386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 302)
        # Adding element type (line 302)
        
        # Obtaining an instance of the builtin type 'list' (line 302)
        list_112387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 302)
        # Adding element type (line 302)
        int_112388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 37), list_112387, int_112388)
        # Adding element type (line 302)
        int_112389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 37), list_112387, int_112389)
        # Adding element type (line 302)
        int_112390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 37), list_112387, int_112390)
        # Adding element type (line 302)
        int_112391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 37), list_112387, int_112391)
        # Adding element type (line 302)
        int_112392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 37), list_112387, int_112392)
        # Adding element type (line 302)
        int_112393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 37), list_112387, int_112393)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 36), list_112386, list_112387)
        # Adding element type (line 302)
        
        # Obtaining an instance of the builtin type 'list' (line 303)
        list_112394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 303)
        # Adding element type (line 303)
        int_112395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 37), list_112394, int_112395)
        # Adding element type (line 303)
        int_112396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 37), list_112394, int_112396)
        # Adding element type (line 303)
        int_112397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 37), list_112394, int_112397)
        # Adding element type (line 303)
        int_112398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 37), list_112394, int_112398)
        # Adding element type (line 303)
        int_112399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 37), list_112394, int_112399)
        # Adding element type (line 303)
        int_112400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 37), list_112394, int_112400)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 36), list_112386, list_112394)
        
        # Processing the call keyword arguments (line 302)
        kwargs_112401 = {}
        # Getting the type of 'array' (line 302)
        array_112385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 30), 'array', False)
        # Calling array(args, kwargs) (line 302)
        array_call_result_112402 = invoke(stypy.reporting.localization.Localization(__file__, 302, 30), array_112385, *[list_112386], **kwargs_112401)
        
        # Processing the call keyword arguments (line 302)
        kwargs_112403 = {}
        # Getting the type of 'assert_array_equal' (line 302)
        assert_array_equal_112383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 302)
        assert_array_equal_call_result_112404 = invoke(stypy.reporting.localization.Localization(__file__, 302, 8), assert_array_equal_112383, *[a_112384, array_call_result_112402], **kwargs_112403)
        
        
        # Assigning a Call to a Name (line 305):
        
        # Call to array(...): (line 305)
        # Processing the call arguments (line 305)
        
        # Obtaining an instance of the builtin type 'list' (line 305)
        list_112406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 305)
        # Adding element type (line 305)
        
        # Obtaining an instance of the builtin type 'list' (line 305)
        list_112407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 305)
        # Adding element type (line 305)
        int_112408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 20), list_112407, int_112408)
        # Adding element type (line 305)
        int_112409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 20), list_112407, int_112409)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 19), list_112406, list_112407)
        # Adding element type (line 305)
        
        # Obtaining an instance of the builtin type 'list' (line 305)
        list_112410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 305)
        # Adding element type (line 305)
        int_112411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 28), list_112410, int_112411)
        # Adding element type (line 305)
        int_112412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 28), list_112410, int_112412)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 19), list_112406, list_112410)
        
        # Processing the call keyword arguments (line 305)
        kwargs_112413 = {}
        # Getting the type of 'array' (line 305)
        array_112405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 13), 'array', False)
        # Calling array(args, kwargs) (line 305)
        array_call_result_112414 = invoke(stypy.reporting.localization.Localization(__file__, 305, 13), array_112405, *[list_112406], **kwargs_112413)
        
        # Assigning a type to the variable 'm1' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'm1', array_call_result_112414)
        
        # Assigning a Call to a Name (line 306):
        
        # Call to array(...): (line 306)
        # Processing the call arguments (line 306)
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_112416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_112417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        int_112418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 20), list_112417, int_112418)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 19), list_112416, list_112417)
        # Adding element type (line 306)
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_112419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        int_112420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 26), list_112419, int_112420)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 19), list_112416, list_112419)
        
        # Processing the call keyword arguments (line 306)
        kwargs_112421 = {}
        # Getting the type of 'array' (line 306)
        array_112415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 13), 'array', False)
        # Calling array(args, kwargs) (line 306)
        array_call_result_112422 = invoke(stypy.reporting.localization.Localization(__file__, 306, 13), array_112415, *[list_112416], **kwargs_112421)
        
        # Assigning a type to the variable 'm2' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'm2', array_call_result_112422)
        
        # Assigning a Call to a Name (line 307):
        
        # Call to kron(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'm1' (line 307)
        m1_112424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 17), 'm1', False)
        # Getting the type of 'm2' (line 307)
        m2_112425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 21), 'm2', False)
        # Processing the call keyword arguments (line 307)
        kwargs_112426 = {}
        # Getting the type of 'kron' (line 307)
        kron_112423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'kron', False)
        # Calling kron(args, kwargs) (line 307)
        kron_call_result_112427 = invoke(stypy.reporting.localization.Localization(__file__, 307, 12), kron_112423, *[m1_112424, m2_112425], **kwargs_112426)
        
        # Assigning a type to the variable 'a' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'a', kron_call_result_112427)
        
        # Assigning a Call to a Name (line 308):
        
        # Call to array(...): (line 308)
        # Processing the call arguments (line 308)
        
        # Obtaining an instance of the builtin type 'list' (line 308)
        list_112429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 308)
        # Adding element type (line 308)
        
        # Obtaining an instance of the builtin type 'list' (line 308)
        list_112430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 308)
        # Adding element type (line 308)
        int_112431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 26), list_112430, int_112431)
        # Adding element type (line 308)
        int_112432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 26), list_112430, int_112432)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 25), list_112429, list_112430)
        # Adding element type (line 308)
        
        # Obtaining an instance of the builtin type 'list' (line 309)
        list_112433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 309)
        # Adding element type (line 309)
        int_112434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 26), list_112433, int_112434)
        # Adding element type (line 309)
        int_112435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 26), list_112433, int_112435)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 25), list_112429, list_112433)
        # Adding element type (line 308)
        
        # Obtaining an instance of the builtin type 'list' (line 310)
        list_112436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 310)
        # Adding element type (line 310)
        int_112437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 26), list_112436, int_112437)
        # Adding element type (line 310)
        int_112438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 26), list_112436, int_112438)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 25), list_112429, list_112436)
        # Adding element type (line 308)
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_112439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        int_112440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 26), list_112439, int_112440)
        # Adding element type (line 311)
        int_112441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 26), list_112439, int_112441)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 25), list_112429, list_112439)
        
        # Processing the call keyword arguments (line 308)
        kwargs_112442 = {}
        # Getting the type of 'array' (line 308)
        array_112428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 19), 'array', False)
        # Calling array(args, kwargs) (line 308)
        array_call_result_112443 = invoke(stypy.reporting.localization.Localization(__file__, 308, 19), array_112428, *[list_112429], **kwargs_112442)
        
        # Assigning a type to the variable 'expected' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'expected', array_call_result_112443)
        
        # Call to assert_array_equal(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'a' (line 312)
        a_112445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 27), 'a', False)
        # Getting the type of 'expected' (line 312)
        expected_112446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 30), 'expected', False)
        # Processing the call keyword arguments (line 312)
        kwargs_112447 = {}
        # Getting the type of 'assert_array_equal' (line 312)
        assert_array_equal_112444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 312)
        assert_array_equal_call_result_112448 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), assert_array_equal_112444, *[a_112445, expected_112446], **kwargs_112447)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 299)
        stypy_return_type_112449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112449)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_112449


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 297, 0, False)
        # Assigning a type to the variable 'self' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKron.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestKron' (line 297)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'TestKron', TestKron)
# Declaration of the 'TestHelmert' class

class TestHelmert(object, ):

    @norecursion
    def test_orthogonality(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_orthogonality'
        module_type_store = module_type_store.open_function_context('test_orthogonality', 317, 4, False)
        # Assigning a type to the variable 'self' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHelmert.test_orthogonality.__dict__.__setitem__('stypy_localization', localization)
        TestHelmert.test_orthogonality.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHelmert.test_orthogonality.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHelmert.test_orthogonality.__dict__.__setitem__('stypy_function_name', 'TestHelmert.test_orthogonality')
        TestHelmert.test_orthogonality.__dict__.__setitem__('stypy_param_names_list', [])
        TestHelmert.test_orthogonality.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHelmert.test_orthogonality.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHelmert.test_orthogonality.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHelmert.test_orthogonality.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHelmert.test_orthogonality.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHelmert.test_orthogonality.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHelmert.test_orthogonality', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_orthogonality', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_orthogonality(...)' code ##################

        
        
        # Call to range(...): (line 318)
        # Processing the call arguments (line 318)
        int_112451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 23), 'int')
        int_112452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 26), 'int')
        # Processing the call keyword arguments (line 318)
        kwargs_112453 = {}
        # Getting the type of 'range' (line 318)
        range_112450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 17), 'range', False)
        # Calling range(args, kwargs) (line 318)
        range_call_result_112454 = invoke(stypy.reporting.localization.Localization(__file__, 318, 17), range_112450, *[int_112451, int_112452], **kwargs_112453)
        
        # Testing the type of a for loop iterable (line 318)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 318, 8), range_call_result_112454)
        # Getting the type of the for loop variable (line 318)
        for_loop_var_112455 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 318, 8), range_call_result_112454)
        # Assigning a type to the variable 'n' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'n', for_loop_var_112455)
        # SSA begins for a for statement (line 318)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 319):
        
        # Call to helmert(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'n' (line 319)
        n_112457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 24), 'n', False)
        # Processing the call keyword arguments (line 319)
        # Getting the type of 'True' (line 319)
        True_112458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 32), 'True', False)
        keyword_112459 = True_112458
        kwargs_112460 = {'full': keyword_112459}
        # Getting the type of 'helmert' (line 319)
        helmert_112456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 16), 'helmert', False)
        # Calling helmert(args, kwargs) (line 319)
        helmert_call_result_112461 = invoke(stypy.reporting.localization.Localization(__file__, 319, 16), helmert_112456, *[n_112457], **kwargs_112460)
        
        # Assigning a type to the variable 'H' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'H', helmert_call_result_112461)
        
        # Assigning a Call to a Name (line 320):
        
        # Call to eye(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'n' (line 320)
        n_112464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 'n', False)
        # Processing the call keyword arguments (line 320)
        kwargs_112465 = {}
        # Getting the type of 'np' (line 320)
        np_112462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'np', False)
        # Obtaining the member 'eye' of a type (line 320)
        eye_112463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 16), np_112462, 'eye')
        # Calling eye(args, kwargs) (line 320)
        eye_call_result_112466 = invoke(stypy.reporting.localization.Localization(__file__, 320, 16), eye_112463, *[n_112464], **kwargs_112465)
        
        # Assigning a type to the variable 'I' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'I', eye_call_result_112466)
        
        # Call to assert_allclose(...): (line 321)
        # Processing the call arguments (line 321)
        
        # Call to dot(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'H' (line 321)
        H_112470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 34), 'H', False)
        # Obtaining the member 'T' of a type (line 321)
        T_112471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 34), H_112470, 'T')
        # Processing the call keyword arguments (line 321)
        kwargs_112472 = {}
        # Getting the type of 'H' (line 321)
        H_112468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 28), 'H', False)
        # Obtaining the member 'dot' of a type (line 321)
        dot_112469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 28), H_112468, 'dot')
        # Calling dot(args, kwargs) (line 321)
        dot_call_result_112473 = invoke(stypy.reporting.localization.Localization(__file__, 321, 28), dot_112469, *[T_112471], **kwargs_112472)
        
        # Getting the type of 'I' (line 321)
        I_112474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 40), 'I', False)
        # Processing the call keyword arguments (line 321)
        float_112475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 48), 'float')
        keyword_112476 = float_112475
        kwargs_112477 = {'atol': keyword_112476}
        # Getting the type of 'assert_allclose' (line 321)
        assert_allclose_112467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 321)
        assert_allclose_call_result_112478 = invoke(stypy.reporting.localization.Localization(__file__, 321, 12), assert_allclose_112467, *[dot_call_result_112473, I_112474], **kwargs_112477)
        
        
        # Call to assert_allclose(...): (line 322)
        # Processing the call arguments (line 322)
        
        # Call to dot(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'H' (line 322)
        H_112483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 36), 'H', False)
        # Processing the call keyword arguments (line 322)
        kwargs_112484 = {}
        # Getting the type of 'H' (line 322)
        H_112480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 28), 'H', False)
        # Obtaining the member 'T' of a type (line 322)
        T_112481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 28), H_112480, 'T')
        # Obtaining the member 'dot' of a type (line 322)
        dot_112482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 28), T_112481, 'dot')
        # Calling dot(args, kwargs) (line 322)
        dot_call_result_112485 = invoke(stypy.reporting.localization.Localization(__file__, 322, 28), dot_112482, *[H_112483], **kwargs_112484)
        
        # Getting the type of 'I' (line 322)
        I_112486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 40), 'I', False)
        # Processing the call keyword arguments (line 322)
        float_112487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 48), 'float')
        keyword_112488 = float_112487
        kwargs_112489 = {'atol': keyword_112488}
        # Getting the type of 'assert_allclose' (line 322)
        assert_allclose_112479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 322)
        assert_allclose_call_result_112490 = invoke(stypy.reporting.localization.Localization(__file__, 322, 12), assert_allclose_112479, *[dot_call_result_112485, I_112486], **kwargs_112489)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_orthogonality(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_orthogonality' in the type store
        # Getting the type of 'stypy_return_type' (line 317)
        stypy_return_type_112491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112491)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_orthogonality'
        return stypy_return_type_112491


    @norecursion
    def test_subspace(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_subspace'
        module_type_store = module_type_store.open_function_context('test_subspace', 324, 4, False)
        # Assigning a type to the variable 'self' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHelmert.test_subspace.__dict__.__setitem__('stypy_localization', localization)
        TestHelmert.test_subspace.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHelmert.test_subspace.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHelmert.test_subspace.__dict__.__setitem__('stypy_function_name', 'TestHelmert.test_subspace')
        TestHelmert.test_subspace.__dict__.__setitem__('stypy_param_names_list', [])
        TestHelmert.test_subspace.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHelmert.test_subspace.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHelmert.test_subspace.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHelmert.test_subspace.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHelmert.test_subspace.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHelmert.test_subspace.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHelmert.test_subspace', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_subspace', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_subspace(...)' code ##################

        
        
        # Call to range(...): (line 325)
        # Processing the call arguments (line 325)
        int_112493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 23), 'int')
        int_112494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 26), 'int')
        # Processing the call keyword arguments (line 325)
        kwargs_112495 = {}
        # Getting the type of 'range' (line 325)
        range_112492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 17), 'range', False)
        # Calling range(args, kwargs) (line 325)
        range_call_result_112496 = invoke(stypy.reporting.localization.Localization(__file__, 325, 17), range_112492, *[int_112493, int_112494], **kwargs_112495)
        
        # Testing the type of a for loop iterable (line 325)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 325, 8), range_call_result_112496)
        # Getting the type of the for loop variable (line 325)
        for_loop_var_112497 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 325, 8), range_call_result_112496)
        # Assigning a type to the variable 'n' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'n', for_loop_var_112497)
        # SSA begins for a for statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 326):
        
        # Call to helmert(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'n' (line 326)
        n_112499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 29), 'n', False)
        # Processing the call keyword arguments (line 326)
        # Getting the type of 'True' (line 326)
        True_112500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 37), 'True', False)
        keyword_112501 = True_112500
        kwargs_112502 = {'full': keyword_112501}
        # Getting the type of 'helmert' (line 326)
        helmert_112498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 21), 'helmert', False)
        # Calling helmert(args, kwargs) (line 326)
        helmert_call_result_112503 = invoke(stypy.reporting.localization.Localization(__file__, 326, 21), helmert_112498, *[n_112499], **kwargs_112502)
        
        # Assigning a type to the variable 'H_full' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'H_full', helmert_call_result_112503)
        
        # Assigning a Call to a Name (line 327):
        
        # Call to helmert(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'n' (line 327)
        n_112505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 32), 'n', False)
        # Processing the call keyword arguments (line 327)
        kwargs_112506 = {}
        # Getting the type of 'helmert' (line 327)
        helmert_112504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 24), 'helmert', False)
        # Calling helmert(args, kwargs) (line 327)
        helmert_call_result_112507 = invoke(stypy.reporting.localization.Localization(__file__, 327, 24), helmert_112504, *[n_112505], **kwargs_112506)
        
        # Assigning a type to the variable 'H_partial' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'H_partial', helmert_call_result_112507)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 328)
        tuple_112508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 328)
        # Adding element type (line 328)
        
        # Obtaining the type of the subscript
        int_112509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 28), 'int')
        slice_112510 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 328, 21), int_112509, None, None)
        slice_112511 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 328, 21), None, None, None)
        # Getting the type of 'H_full' (line 328)
        H_full_112512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 21), 'H_full')
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___112513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 21), H_full_112512, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_112514 = invoke(stypy.reporting.localization.Localization(__file__, 328, 21), getitem___112513, (slice_112510, slice_112511))
        
        # Obtaining the member 'T' of a type (line 328)
        T_112515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 21), subscript_call_result_112514, 'T')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 21), tuple_112508, T_112515)
        # Adding element type (line 328)
        # Getting the type of 'H_partial' (line 328)
        H_partial_112516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 38), 'H_partial')
        # Obtaining the member 'T' of a type (line 328)
        T_112517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 38), H_partial_112516, 'T')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 21), tuple_112508, T_112517)
        
        # Testing the type of a for loop iterable (line 328)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 328, 12), tuple_112508)
        # Getting the type of the for loop variable (line 328)
        for_loop_var_112518 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 328, 12), tuple_112508)
        # Assigning a type to the variable 'U' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'U', for_loop_var_112518)
        # SSA begins for a for statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 329):
        
        # Call to eye(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'n' (line 329)
        n_112521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 27), 'n', False)
        # Processing the call keyword arguments (line 329)
        kwargs_112522 = {}
        # Getting the type of 'np' (line 329)
        np_112519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'np', False)
        # Obtaining the member 'eye' of a type (line 329)
        eye_112520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 20), np_112519, 'eye')
        # Calling eye(args, kwargs) (line 329)
        eye_call_result_112523 = invoke(stypy.reporting.localization.Localization(__file__, 329, 20), eye_112520, *[n_112521], **kwargs_112522)
        
        
        # Call to ones(...): (line 329)
        # Processing the call arguments (line 329)
        
        # Obtaining an instance of the builtin type 'tuple' (line 329)
        tuple_112526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 329)
        # Adding element type (line 329)
        # Getting the type of 'n' (line 329)
        n_112527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 41), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 41), tuple_112526, n_112527)
        # Adding element type (line 329)
        # Getting the type of 'n' (line 329)
        n_112528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 44), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 41), tuple_112526, n_112528)
        
        # Processing the call keyword arguments (line 329)
        kwargs_112529 = {}
        # Getting the type of 'np' (line 329)
        np_112524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 32), 'np', False)
        # Obtaining the member 'ones' of a type (line 329)
        ones_112525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 32), np_112524, 'ones')
        # Calling ones(args, kwargs) (line 329)
        ones_call_result_112530 = invoke(stypy.reporting.localization.Localization(__file__, 329, 32), ones_112525, *[tuple_112526], **kwargs_112529)
        
        # Getting the type of 'n' (line 329)
        n_112531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 50), 'n')
        # Applying the binary operator 'div' (line 329)
        result_div_112532 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 32), 'div', ones_call_result_112530, n_112531)
        
        # Applying the binary operator '-' (line 329)
        result_sub_112533 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 20), '-', eye_call_result_112523, result_div_112532)
        
        # Assigning a type to the variable 'C' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'C', result_sub_112533)
        
        # Call to assert_allclose(...): (line 330)
        # Processing the call arguments (line 330)
        
        # Call to dot(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'U' (line 330)
        U_112537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 38), 'U', False)
        # Obtaining the member 'T' of a type (line 330)
        T_112538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 38), U_112537, 'T')
        # Processing the call keyword arguments (line 330)
        kwargs_112539 = {}
        # Getting the type of 'U' (line 330)
        U_112535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 32), 'U', False)
        # Obtaining the member 'dot' of a type (line 330)
        dot_112536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 32), U_112535, 'dot')
        # Calling dot(args, kwargs) (line 330)
        dot_call_result_112540 = invoke(stypy.reporting.localization.Localization(__file__, 330, 32), dot_112536, *[T_112538], **kwargs_112539)
        
        # Getting the type of 'C' (line 330)
        C_112541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 44), 'C', False)
        # Processing the call keyword arguments (line 330)
        kwargs_112542 = {}
        # Getting the type of 'assert_allclose' (line 330)
        assert_allclose_112534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 330)
        assert_allclose_call_result_112543 = invoke(stypy.reporting.localization.Localization(__file__, 330, 16), assert_allclose_112534, *[dot_call_result_112540, C_112541], **kwargs_112542)
        
        
        # Call to assert_allclose(...): (line 331)
        # Processing the call arguments (line 331)
        
        # Call to dot(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'U' (line 331)
        U_112548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 40), 'U', False)
        # Processing the call keyword arguments (line 331)
        kwargs_112549 = {}
        # Getting the type of 'U' (line 331)
        U_112545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 32), 'U', False)
        # Obtaining the member 'T' of a type (line 331)
        T_112546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 32), U_112545, 'T')
        # Obtaining the member 'dot' of a type (line 331)
        dot_112547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 32), T_112546, 'dot')
        # Calling dot(args, kwargs) (line 331)
        dot_call_result_112550 = invoke(stypy.reporting.localization.Localization(__file__, 331, 32), dot_112547, *[U_112548], **kwargs_112549)
        
        
        # Call to eye(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'n' (line 331)
        n_112553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 51), 'n', False)
        int_112554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 53), 'int')
        # Applying the binary operator '-' (line 331)
        result_sub_112555 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 51), '-', n_112553, int_112554)
        
        # Processing the call keyword arguments (line 331)
        kwargs_112556 = {}
        # Getting the type of 'np' (line 331)
        np_112551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 44), 'np', False)
        # Obtaining the member 'eye' of a type (line 331)
        eye_112552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 44), np_112551, 'eye')
        # Calling eye(args, kwargs) (line 331)
        eye_call_result_112557 = invoke(stypy.reporting.localization.Localization(__file__, 331, 44), eye_112552, *[result_sub_112555], **kwargs_112556)
        
        # Processing the call keyword arguments (line 331)
        float_112558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 62), 'float')
        keyword_112559 = float_112558
        kwargs_112560 = {'atol': keyword_112559}
        # Getting the type of 'assert_allclose' (line 331)
        assert_allclose_112544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 331)
        assert_allclose_call_result_112561 = invoke(stypy.reporting.localization.Localization(__file__, 331, 16), assert_allclose_112544, *[dot_call_result_112550, eye_call_result_112557], **kwargs_112560)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_subspace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_subspace' in the type store
        # Getting the type of 'stypy_return_type' (line 324)
        stypy_return_type_112562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112562)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_subspace'
        return stypy_return_type_112562


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 315, 0, False)
        # Assigning a type to the variable 'self' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHelmert.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestHelmert' (line 315)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), 'TestHelmert', TestHelmert)
# Declaration of the 'TestHilbert' class

class TestHilbert(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 336, 4, False)
        # Assigning a type to the variable 'self' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHilbert.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestHilbert.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHilbert.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHilbert.test_basic.__dict__.__setitem__('stypy_function_name', 'TestHilbert.test_basic')
        TestHilbert.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestHilbert.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHilbert.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHilbert.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHilbert.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHilbert.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHilbert.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHilbert.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 337):
        
        # Call to array(...): (line 337)
        # Processing the call arguments (line 337)
        
        # Obtaining an instance of the builtin type 'list' (line 337)
        list_112564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 337)
        # Adding element type (line 337)
        
        # Obtaining an instance of the builtin type 'list' (line 337)
        list_112565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 337)
        # Adding element type (line 337)
        float_112566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 20), list_112565, float_112566)
        # Adding element type (line 337)
        int_112567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 26), 'int')
        float_112568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 28), 'float')
        # Applying the binary operator 'div' (line 337)
        result_div_112569 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 26), 'div', int_112567, float_112568)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 20), list_112565, result_div_112569)
        # Adding element type (line 337)
        int_112570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 32), 'int')
        float_112571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 34), 'float')
        # Applying the binary operator 'div' (line 337)
        result_div_112572 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 32), 'div', int_112570, float_112571)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 20), list_112565, result_div_112572)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 19), list_112564, list_112565)
        # Adding element type (line 337)
        
        # Obtaining an instance of the builtin type 'list' (line 338)
        list_112573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 338)
        # Adding element type (line 338)
        int_112574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 21), 'int')
        float_112575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 23), 'float')
        # Applying the binary operator 'div' (line 338)
        result_div_112576 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 21), 'div', int_112574, float_112575)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 20), list_112573, result_div_112576)
        # Adding element type (line 338)
        int_112577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 27), 'int')
        float_112578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 29), 'float')
        # Applying the binary operator 'div' (line 338)
        result_div_112579 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 27), 'div', int_112577, float_112578)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 20), list_112573, result_div_112579)
        # Adding element type (line 338)
        int_112580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 33), 'int')
        float_112581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 35), 'float')
        # Applying the binary operator 'div' (line 338)
        result_div_112582 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 33), 'div', int_112580, float_112581)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 20), list_112573, result_div_112582)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 19), list_112564, list_112573)
        # Adding element type (line 337)
        
        # Obtaining an instance of the builtin type 'list' (line 339)
        list_112583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 339)
        # Adding element type (line 339)
        int_112584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 21), 'int')
        float_112585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 23), 'float')
        # Applying the binary operator 'div' (line 339)
        result_div_112586 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 21), 'div', int_112584, float_112585)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 20), list_112583, result_div_112586)
        # Adding element type (line 339)
        int_112587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 27), 'int')
        float_112588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 29), 'float')
        # Applying the binary operator 'div' (line 339)
        result_div_112589 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 27), 'div', int_112587, float_112588)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 20), list_112583, result_div_112589)
        # Adding element type (line 339)
        int_112590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 33), 'int')
        float_112591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 35), 'float')
        # Applying the binary operator 'div' (line 339)
        result_div_112592 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 33), 'div', int_112590, float_112591)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 20), list_112583, result_div_112592)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 19), list_112564, list_112583)
        
        # Processing the call keyword arguments (line 337)
        kwargs_112593 = {}
        # Getting the type of 'array' (line 337)
        array_112563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 13), 'array', False)
        # Calling array(args, kwargs) (line 337)
        array_call_result_112594 = invoke(stypy.reporting.localization.Localization(__file__, 337, 13), array_112563, *[list_112564], **kwargs_112593)
        
        # Assigning a type to the variable 'h3' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'h3', array_call_result_112594)
        
        # Call to assert_array_almost_equal(...): (line 340)
        # Processing the call arguments (line 340)
        
        # Call to hilbert(...): (line 340)
        # Processing the call arguments (line 340)
        int_112597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 42), 'int')
        # Processing the call keyword arguments (line 340)
        kwargs_112598 = {}
        # Getting the type of 'hilbert' (line 340)
        hilbert_112596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 34), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 340)
        hilbert_call_result_112599 = invoke(stypy.reporting.localization.Localization(__file__, 340, 34), hilbert_112596, *[int_112597], **kwargs_112598)
        
        # Getting the type of 'h3' (line 340)
        h3_112600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 46), 'h3', False)
        # Processing the call keyword arguments (line 340)
        kwargs_112601 = {}
        # Getting the type of 'assert_array_almost_equal' (line 340)
        assert_array_almost_equal_112595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 340)
        assert_array_almost_equal_call_result_112602 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), assert_array_almost_equal_112595, *[hilbert_call_result_112599, h3_112600], **kwargs_112601)
        
        
        # Call to assert_array_equal(...): (line 342)
        # Processing the call arguments (line 342)
        
        # Call to hilbert(...): (line 342)
        # Processing the call arguments (line 342)
        int_112605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 35), 'int')
        # Processing the call keyword arguments (line 342)
        kwargs_112606 = {}
        # Getting the type of 'hilbert' (line 342)
        hilbert_112604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 27), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 342)
        hilbert_call_result_112607 = invoke(stypy.reporting.localization.Localization(__file__, 342, 27), hilbert_112604, *[int_112605], **kwargs_112606)
        
        
        # Obtaining an instance of the builtin type 'list' (line 342)
        list_112608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 342)
        # Adding element type (line 342)
        
        # Obtaining an instance of the builtin type 'list' (line 342)
        list_112609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 342)
        # Adding element type (line 342)
        float_112610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 40), list_112609, float_112610)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 39), list_112608, list_112609)
        
        # Processing the call keyword arguments (line 342)
        kwargs_112611 = {}
        # Getting the type of 'assert_array_equal' (line 342)
        assert_array_equal_112603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 342)
        assert_array_equal_call_result_112612 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), assert_array_equal_112603, *[hilbert_call_result_112607, list_112608], **kwargs_112611)
        
        
        # Assigning a Call to a Name (line 344):
        
        # Call to hilbert(...): (line 344)
        # Processing the call arguments (line 344)
        int_112614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 21), 'int')
        # Processing the call keyword arguments (line 344)
        kwargs_112615 = {}
        # Getting the type of 'hilbert' (line 344)
        hilbert_112613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 13), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 344)
        hilbert_call_result_112616 = invoke(stypy.reporting.localization.Localization(__file__, 344, 13), hilbert_112613, *[int_112614], **kwargs_112615)
        
        # Assigning a type to the variable 'h0' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'h0', hilbert_call_result_112616)
        
        # Call to assert_equal(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'h0' (line 345)
        h0_112618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 21), 'h0', False)
        # Obtaining the member 'shape' of a type (line 345)
        shape_112619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 21), h0_112618, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 345)
        tuple_112620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 345)
        # Adding element type (line 345)
        int_112621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 32), tuple_112620, int_112621)
        # Adding element type (line 345)
        int_112622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 32), tuple_112620, int_112622)
        
        # Processing the call keyword arguments (line 345)
        kwargs_112623 = {}
        # Getting the type of 'assert_equal' (line 345)
        assert_equal_112617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 345)
        assert_equal_call_result_112624 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), assert_equal_112617, *[shape_112619, tuple_112620], **kwargs_112623)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 336)
        stypy_return_type_112625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112625)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_112625


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 334, 0, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHilbert.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestHilbert' (line 334)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 0), 'TestHilbert', TestHilbert)
# Declaration of the 'TestInvHilbert' class

class TestInvHilbert(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 350, 4, False)
        # Assigning a type to the variable 'self' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInvHilbert.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestInvHilbert.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInvHilbert.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInvHilbert.test_basic.__dict__.__setitem__('stypy_function_name', 'TestInvHilbert.test_basic')
        TestInvHilbert.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestInvHilbert.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInvHilbert.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInvHilbert.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInvHilbert.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInvHilbert.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInvHilbert.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInvHilbert.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 351):
        
        # Call to array(...): (line 351)
        # Processing the call arguments (line 351)
        
        # Obtaining an instance of the builtin type 'list' (line 351)
        list_112627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 351)
        # Adding element type (line 351)
        
        # Obtaining an instance of the builtin type 'list' (line 351)
        list_112628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 351)
        # Adding element type (line 351)
        int_112629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 23), list_112628, int_112629)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 22), list_112627, list_112628)
        
        # Processing the call keyword arguments (line 351)
        kwargs_112630 = {}
        # Getting the type of 'array' (line 351)
        array_112626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 16), 'array', False)
        # Calling array(args, kwargs) (line 351)
        array_call_result_112631 = invoke(stypy.reporting.localization.Localization(__file__, 351, 16), array_112626, *[list_112627], **kwargs_112630)
        
        # Assigning a type to the variable 'invh1' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'invh1', array_call_result_112631)
        
        # Call to assert_array_equal(...): (line 352)
        # Processing the call arguments (line 352)
        
        # Call to invhilbert(...): (line 352)
        # Processing the call arguments (line 352)
        int_112634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 38), 'int')
        # Processing the call keyword arguments (line 352)
        # Getting the type of 'True' (line 352)
        True_112635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 47), 'True', False)
        keyword_112636 = True_112635
        kwargs_112637 = {'exact': keyword_112636}
        # Getting the type of 'invhilbert' (line 352)
        invhilbert_112633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 27), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 352)
        invhilbert_call_result_112638 = invoke(stypy.reporting.localization.Localization(__file__, 352, 27), invhilbert_112633, *[int_112634], **kwargs_112637)
        
        # Getting the type of 'invh1' (line 352)
        invh1_112639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 54), 'invh1', False)
        # Processing the call keyword arguments (line 352)
        kwargs_112640 = {}
        # Getting the type of 'assert_array_equal' (line 352)
        assert_array_equal_112632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 352)
        assert_array_equal_call_result_112641 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), assert_array_equal_112632, *[invhilbert_call_result_112638, invh1_112639], **kwargs_112640)
        
        
        # Call to assert_array_equal(...): (line 353)
        # Processing the call arguments (line 353)
        
        # Call to invhilbert(...): (line 353)
        # Processing the call arguments (line 353)
        int_112644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 38), 'int')
        # Processing the call keyword arguments (line 353)
        kwargs_112645 = {}
        # Getting the type of 'invhilbert' (line 353)
        invhilbert_112643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 27), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 353)
        invhilbert_call_result_112646 = invoke(stypy.reporting.localization.Localization(__file__, 353, 27), invhilbert_112643, *[int_112644], **kwargs_112645)
        
        # Getting the type of 'invh1' (line 353)
        invh1_112647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 42), 'invh1', False)
        # Processing the call keyword arguments (line 353)
        kwargs_112648 = {}
        # Getting the type of 'assert_array_equal' (line 353)
        assert_array_equal_112642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 353)
        assert_array_equal_call_result_112649 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), assert_array_equal_112642, *[invhilbert_call_result_112646, invh1_112647], **kwargs_112648)
        
        
        # Assigning a Call to a Name (line 355):
        
        # Call to array(...): (line 355)
        # Processing the call arguments (line 355)
        
        # Obtaining an instance of the builtin type 'list' (line 355)
        list_112651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 355)
        # Adding element type (line 355)
        
        # Obtaining an instance of the builtin type 'list' (line 355)
        list_112652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 355)
        # Adding element type (line 355)
        int_112653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 23), list_112652, int_112653)
        # Adding element type (line 355)
        int_112654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 23), list_112652, int_112654)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), list_112651, list_112652)
        # Adding element type (line 355)
        
        # Obtaining an instance of the builtin type 'list' (line 356)
        list_112655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 356)
        # Adding element type (line 356)
        int_112656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 23), list_112655, int_112656)
        # Adding element type (line 356)
        int_112657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 23), list_112655, int_112657)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), list_112651, list_112655)
        
        # Processing the call keyword arguments (line 355)
        kwargs_112658 = {}
        # Getting the type of 'array' (line 355)
        array_112650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'array', False)
        # Calling array(args, kwargs) (line 355)
        array_call_result_112659 = invoke(stypy.reporting.localization.Localization(__file__, 355, 16), array_112650, *[list_112651], **kwargs_112658)
        
        # Assigning a type to the variable 'invh2' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'invh2', array_call_result_112659)
        
        # Call to assert_array_equal(...): (line 357)
        # Processing the call arguments (line 357)
        
        # Call to invhilbert(...): (line 357)
        # Processing the call arguments (line 357)
        int_112662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 38), 'int')
        # Processing the call keyword arguments (line 357)
        # Getting the type of 'True' (line 357)
        True_112663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 47), 'True', False)
        keyword_112664 = True_112663
        kwargs_112665 = {'exact': keyword_112664}
        # Getting the type of 'invhilbert' (line 357)
        invhilbert_112661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 27), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 357)
        invhilbert_call_result_112666 = invoke(stypy.reporting.localization.Localization(__file__, 357, 27), invhilbert_112661, *[int_112662], **kwargs_112665)
        
        # Getting the type of 'invh2' (line 357)
        invh2_112667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 54), 'invh2', False)
        # Processing the call keyword arguments (line 357)
        kwargs_112668 = {}
        # Getting the type of 'assert_array_equal' (line 357)
        assert_array_equal_112660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 357)
        assert_array_equal_call_result_112669 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), assert_array_equal_112660, *[invhilbert_call_result_112666, invh2_112667], **kwargs_112668)
        
        
        # Call to assert_array_almost_equal(...): (line 358)
        # Processing the call arguments (line 358)
        
        # Call to invhilbert(...): (line 358)
        # Processing the call arguments (line 358)
        int_112672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 45), 'int')
        # Processing the call keyword arguments (line 358)
        kwargs_112673 = {}
        # Getting the type of 'invhilbert' (line 358)
        invhilbert_112671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 34), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 358)
        invhilbert_call_result_112674 = invoke(stypy.reporting.localization.Localization(__file__, 358, 34), invhilbert_112671, *[int_112672], **kwargs_112673)
        
        # Getting the type of 'invh2' (line 358)
        invh2_112675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 49), 'invh2', False)
        # Processing the call keyword arguments (line 358)
        kwargs_112676 = {}
        # Getting the type of 'assert_array_almost_equal' (line 358)
        assert_array_almost_equal_112670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 358)
        assert_array_almost_equal_call_result_112677 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), assert_array_almost_equal_112670, *[invhilbert_call_result_112674, invh2_112675], **kwargs_112676)
        
        
        # Assigning a Call to a Name (line 360):
        
        # Call to array(...): (line 360)
        # Processing the call arguments (line 360)
        
        # Obtaining an instance of the builtin type 'list' (line 360)
        list_112679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 360)
        # Adding element type (line 360)
        
        # Obtaining an instance of the builtin type 'list' (line 360)
        list_112680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 360)
        # Adding element type (line 360)
        int_112681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 23), list_112680, int_112681)
        # Adding element type (line 360)
        int_112682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 23), list_112680, int_112682)
        # Adding element type (line 360)
        int_112683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 23), list_112680, int_112683)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 22), list_112679, list_112680)
        # Adding element type (line 360)
        
        # Obtaining an instance of the builtin type 'list' (line 361)
        list_112684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 361)
        # Adding element type (line 361)
        int_112685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 23), list_112684, int_112685)
        # Adding element type (line 361)
        int_112686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 23), list_112684, int_112686)
        # Adding element type (line 361)
        int_112687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 23), list_112684, int_112687)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 22), list_112679, list_112684)
        # Adding element type (line 360)
        
        # Obtaining an instance of the builtin type 'list' (line 362)
        list_112688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 362)
        # Adding element type (line 362)
        int_112689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 24), list_112688, int_112689)
        # Adding element type (line 362)
        int_112690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 24), list_112688, int_112690)
        # Adding element type (line 362)
        int_112691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 24), list_112688, int_112691)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 22), list_112679, list_112688)
        
        # Processing the call keyword arguments (line 360)
        kwargs_112692 = {}
        # Getting the type of 'array' (line 360)
        array_112678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'array', False)
        # Calling array(args, kwargs) (line 360)
        array_call_result_112693 = invoke(stypy.reporting.localization.Localization(__file__, 360, 16), array_112678, *[list_112679], **kwargs_112692)
        
        # Assigning a type to the variable 'invh3' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'invh3', array_call_result_112693)
        
        # Call to assert_array_equal(...): (line 363)
        # Processing the call arguments (line 363)
        
        # Call to invhilbert(...): (line 363)
        # Processing the call arguments (line 363)
        int_112696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 38), 'int')
        # Processing the call keyword arguments (line 363)
        # Getting the type of 'True' (line 363)
        True_112697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 47), 'True', False)
        keyword_112698 = True_112697
        kwargs_112699 = {'exact': keyword_112698}
        # Getting the type of 'invhilbert' (line 363)
        invhilbert_112695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 363)
        invhilbert_call_result_112700 = invoke(stypy.reporting.localization.Localization(__file__, 363, 27), invhilbert_112695, *[int_112696], **kwargs_112699)
        
        # Getting the type of 'invh3' (line 363)
        invh3_112701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 54), 'invh3', False)
        # Processing the call keyword arguments (line 363)
        kwargs_112702 = {}
        # Getting the type of 'assert_array_equal' (line 363)
        assert_array_equal_112694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 363)
        assert_array_equal_call_result_112703 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), assert_array_equal_112694, *[invhilbert_call_result_112700, invh3_112701], **kwargs_112702)
        
        
        # Call to assert_array_almost_equal(...): (line 364)
        # Processing the call arguments (line 364)
        
        # Call to invhilbert(...): (line 364)
        # Processing the call arguments (line 364)
        int_112706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 45), 'int')
        # Processing the call keyword arguments (line 364)
        kwargs_112707 = {}
        # Getting the type of 'invhilbert' (line 364)
        invhilbert_112705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 34), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 364)
        invhilbert_call_result_112708 = invoke(stypy.reporting.localization.Localization(__file__, 364, 34), invhilbert_112705, *[int_112706], **kwargs_112707)
        
        # Getting the type of 'invh3' (line 364)
        invh3_112709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 49), 'invh3', False)
        # Processing the call keyword arguments (line 364)
        kwargs_112710 = {}
        # Getting the type of 'assert_array_almost_equal' (line 364)
        assert_array_almost_equal_112704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 364)
        assert_array_almost_equal_call_result_112711 = invoke(stypy.reporting.localization.Localization(__file__, 364, 8), assert_array_almost_equal_112704, *[invhilbert_call_result_112708, invh3_112709], **kwargs_112710)
        
        
        # Assigning a Call to a Name (line 366):
        
        # Call to array(...): (line 366)
        # Processing the call arguments (line 366)
        
        # Obtaining an instance of the builtin type 'list' (line 366)
        list_112713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 366)
        # Adding element type (line 366)
        
        # Obtaining an instance of the builtin type 'list' (line 366)
        list_112714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 366)
        # Adding element type (line 366)
        int_112715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 23), list_112714, int_112715)
        # Adding element type (line 366)
        int_112716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 23), list_112714, int_112716)
        # Adding element type (line 366)
        int_112717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 23), list_112714, int_112717)
        # Adding element type (line 366)
        int_112718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 23), list_112714, int_112718)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 22), list_112713, list_112714)
        # Adding element type (line 366)
        
        # Obtaining an instance of the builtin type 'list' (line 367)
        list_112719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 367)
        # Adding element type (line 367)
        int_112720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 23), list_112719, int_112720)
        # Adding element type (line 367)
        int_112721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 23), list_112719, int_112721)
        # Adding element type (line 367)
        int_112722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 23), list_112719, int_112722)
        # Adding element type (line 367)
        int_112723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 23), list_112719, int_112723)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 22), list_112713, list_112719)
        # Adding element type (line 366)
        
        # Obtaining an instance of the builtin type 'list' (line 368)
        list_112724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 368)
        # Adding element type (line 368)
        int_112725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 23), list_112724, int_112725)
        # Adding element type (line 368)
        int_112726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 23), list_112724, int_112726)
        # Adding element type (line 368)
        int_112727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 23), list_112724, int_112727)
        # Adding element type (line 368)
        int_112728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 23), list_112724, int_112728)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 22), list_112713, list_112724)
        # Adding element type (line 366)
        
        # Obtaining an instance of the builtin type 'list' (line 369)
        list_112729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 369)
        # Adding element type (line 369)
        int_112730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 23), list_112729, int_112730)
        # Adding element type (line 369)
        int_112731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 23), list_112729, int_112731)
        # Adding element type (line 369)
        int_112732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 23), list_112729, int_112732)
        # Adding element type (line 369)
        int_112733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 23), list_112729, int_112733)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 22), list_112713, list_112729)
        
        # Processing the call keyword arguments (line 366)
        kwargs_112734 = {}
        # Getting the type of 'array' (line 366)
        array_112712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'array', False)
        # Calling array(args, kwargs) (line 366)
        array_call_result_112735 = invoke(stypy.reporting.localization.Localization(__file__, 366, 16), array_112712, *[list_112713], **kwargs_112734)
        
        # Assigning a type to the variable 'invh4' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'invh4', array_call_result_112735)
        
        # Call to assert_array_equal(...): (line 370)
        # Processing the call arguments (line 370)
        
        # Call to invhilbert(...): (line 370)
        # Processing the call arguments (line 370)
        int_112738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 38), 'int')
        # Processing the call keyword arguments (line 370)
        # Getting the type of 'True' (line 370)
        True_112739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 47), 'True', False)
        keyword_112740 = True_112739
        kwargs_112741 = {'exact': keyword_112740}
        # Getting the type of 'invhilbert' (line 370)
        invhilbert_112737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 27), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 370)
        invhilbert_call_result_112742 = invoke(stypy.reporting.localization.Localization(__file__, 370, 27), invhilbert_112737, *[int_112738], **kwargs_112741)
        
        # Getting the type of 'invh4' (line 370)
        invh4_112743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 54), 'invh4', False)
        # Processing the call keyword arguments (line 370)
        kwargs_112744 = {}
        # Getting the type of 'assert_array_equal' (line 370)
        assert_array_equal_112736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 370)
        assert_array_equal_call_result_112745 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), assert_array_equal_112736, *[invhilbert_call_result_112742, invh4_112743], **kwargs_112744)
        
        
        # Call to assert_array_almost_equal(...): (line 371)
        # Processing the call arguments (line 371)
        
        # Call to invhilbert(...): (line 371)
        # Processing the call arguments (line 371)
        int_112748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 45), 'int')
        # Processing the call keyword arguments (line 371)
        kwargs_112749 = {}
        # Getting the type of 'invhilbert' (line 371)
        invhilbert_112747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 34), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 371)
        invhilbert_call_result_112750 = invoke(stypy.reporting.localization.Localization(__file__, 371, 34), invhilbert_112747, *[int_112748], **kwargs_112749)
        
        # Getting the type of 'invh4' (line 371)
        invh4_112751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 49), 'invh4', False)
        # Processing the call keyword arguments (line 371)
        kwargs_112752 = {}
        # Getting the type of 'assert_array_almost_equal' (line 371)
        assert_array_almost_equal_112746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 371)
        assert_array_almost_equal_call_result_112753 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), assert_array_almost_equal_112746, *[invhilbert_call_result_112750, invh4_112751], **kwargs_112752)
        
        
        # Assigning a Call to a Name (line 373):
        
        # Call to array(...): (line 373)
        # Processing the call arguments (line 373)
        
        # Obtaining an instance of the builtin type 'list' (line 373)
        list_112755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 373)
        # Adding element type (line 373)
        
        # Obtaining an instance of the builtin type 'list' (line 373)
        list_112756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 373)
        # Adding element type (line 373)
        int_112757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 23), list_112756, int_112757)
        # Adding element type (line 373)
        int_112758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 23), list_112756, int_112758)
        # Adding element type (line 373)
        int_112759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 23), list_112756, int_112759)
        # Adding element type (line 373)
        int_112760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 23), list_112756, int_112760)
        # Adding element type (line 373)
        int_112761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 23), list_112756, int_112761)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), list_112755, list_112756)
        # Adding element type (line 373)
        
        # Obtaining an instance of the builtin type 'list' (line 374)
        list_112762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 374)
        # Adding element type (line 374)
        int_112763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 23), list_112762, int_112763)
        # Adding element type (line 374)
        int_112764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 23), list_112762, int_112764)
        # Adding element type (line 374)
        int_112765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 23), list_112762, int_112765)
        # Adding element type (line 374)
        int_112766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 23), list_112762, int_112766)
        # Adding element type (line 374)
        int_112767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 23), list_112762, int_112767)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), list_112755, list_112762)
        # Adding element type (line 373)
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_112768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        # Adding element type (line 375)
        int_112769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 23), list_112768, int_112769)
        # Adding element type (line 375)
        int_112770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 23), list_112768, int_112770)
        # Adding element type (line 375)
        int_112771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 23), list_112768, int_112771)
        # Adding element type (line 375)
        int_112772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 23), list_112768, int_112772)
        # Adding element type (line 375)
        int_112773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 23), list_112768, int_112773)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), list_112755, list_112768)
        # Adding element type (line 373)
        
        # Obtaining an instance of the builtin type 'list' (line 376)
        list_112774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 376)
        # Adding element type (line 376)
        int_112775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 23), list_112774, int_112775)
        # Adding element type (line 376)
        int_112776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 23), list_112774, int_112776)
        # Adding element type (line 376)
        int_112777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 23), list_112774, int_112777)
        # Adding element type (line 376)
        int_112778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 23), list_112774, int_112778)
        # Adding element type (line 376)
        int_112779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 23), list_112774, int_112779)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), list_112755, list_112774)
        # Adding element type (line 373)
        
        # Obtaining an instance of the builtin type 'list' (line 377)
        list_112780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 377)
        # Adding element type (line 377)
        int_112781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 23), list_112780, int_112781)
        # Adding element type (line 377)
        int_112782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 23), list_112780, int_112782)
        # Adding element type (line 377)
        int_112783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 23), list_112780, int_112783)
        # Adding element type (line 377)
        int_112784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 23), list_112780, int_112784)
        # Adding element type (line 377)
        int_112785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 23), list_112780, int_112785)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), list_112755, list_112780)
        
        # Processing the call keyword arguments (line 373)
        kwargs_112786 = {}
        # Getting the type of 'array' (line 373)
        array_112754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'array', False)
        # Calling array(args, kwargs) (line 373)
        array_call_result_112787 = invoke(stypy.reporting.localization.Localization(__file__, 373, 16), array_112754, *[list_112755], **kwargs_112786)
        
        # Assigning a type to the variable 'invh5' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'invh5', array_call_result_112787)
        
        # Call to assert_array_equal(...): (line 378)
        # Processing the call arguments (line 378)
        
        # Call to invhilbert(...): (line 378)
        # Processing the call arguments (line 378)
        int_112790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 38), 'int')
        # Processing the call keyword arguments (line 378)
        # Getting the type of 'True' (line 378)
        True_112791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 47), 'True', False)
        keyword_112792 = True_112791
        kwargs_112793 = {'exact': keyword_112792}
        # Getting the type of 'invhilbert' (line 378)
        invhilbert_112789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 27), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 378)
        invhilbert_call_result_112794 = invoke(stypy.reporting.localization.Localization(__file__, 378, 27), invhilbert_112789, *[int_112790], **kwargs_112793)
        
        # Getting the type of 'invh5' (line 378)
        invh5_112795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 54), 'invh5', False)
        # Processing the call keyword arguments (line 378)
        kwargs_112796 = {}
        # Getting the type of 'assert_array_equal' (line 378)
        assert_array_equal_112788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 378)
        assert_array_equal_call_result_112797 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), assert_array_equal_112788, *[invhilbert_call_result_112794, invh5_112795], **kwargs_112796)
        
        
        # Call to assert_array_almost_equal(...): (line 379)
        # Processing the call arguments (line 379)
        
        # Call to invhilbert(...): (line 379)
        # Processing the call arguments (line 379)
        int_112800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 45), 'int')
        # Processing the call keyword arguments (line 379)
        kwargs_112801 = {}
        # Getting the type of 'invhilbert' (line 379)
        invhilbert_112799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 34), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 379)
        invhilbert_call_result_112802 = invoke(stypy.reporting.localization.Localization(__file__, 379, 34), invhilbert_112799, *[int_112800], **kwargs_112801)
        
        # Getting the type of 'invh5' (line 379)
        invh5_112803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 49), 'invh5', False)
        # Processing the call keyword arguments (line 379)
        kwargs_112804 = {}
        # Getting the type of 'assert_array_almost_equal' (line 379)
        assert_array_almost_equal_112798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 379)
        assert_array_almost_equal_call_result_112805 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), assert_array_almost_equal_112798, *[invhilbert_call_result_112802, invh5_112803], **kwargs_112804)
        
        
        # Assigning a Call to a Name (line 381):
        
        # Call to array(...): (line 381)
        # Processing the call arguments (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 381)
        list_112807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 381)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 382)
        list_112808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 382)
        # Adding element type (line 382)
        int_112809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, int_112809)
        # Adding element type (line 382)
        int_112810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, int_112810)
        # Adding element type (line 382)
        int_112811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, int_112811)
        # Adding element type (line 382)
        int_112812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, int_112812)
        # Adding element type (line 382)
        int_112813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, int_112813)
        # Adding element type (line 382)
        long_112814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 57), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112814)
        # Adding element type (line 382)
        long_112815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112815)
        # Adding element type (line 382)
        long_112816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 26), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112816)
        # Adding element type (line 382)
        long_112817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 41), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112817)
        # Adding element type (line 382)
        long_112818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 55), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112818)
        # Adding element type (line 382)
        long_112819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112819)
        # Adding element type (line 382)
        long_112820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 28), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112820)
        # Adding element type (line 382)
        long_112821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 44), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112821)
        # Adding element type (line 382)
        long_112822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 59), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112822)
        # Adding element type (line 382)
        long_112823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112823)
        # Adding element type (line 382)
        long_112824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 27), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112824)
        # Adding element type (line 382)
        long_112825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 42), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 12), list_112808, long_112825)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_112808)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 386)
        list_112826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 386)
        # Adding element type (line 386)
        int_112827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, int_112827)
        # Adding element type (line 386)
        int_112828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, int_112828)
        # Adding element type (line 386)
        int_112829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, int_112829)
        # Adding element type (line 386)
        long_112830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 42), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112830)
        # Adding element type (line 386)
        long_112831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 55), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112831)
        # Adding element type (line 386)
        long_112832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 70), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112832)
        # Adding element type (line 386)
        long_112833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112833)
        # Adding element type (line 386)
        long_112834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 29), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112834)
        # Adding element type (line 386)
        long_112835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 45), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112835)
        # Adding element type (line 386)
        long_112836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 63), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112836)
        # Adding element type (line 386)
        long_112837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112837)
        # Adding element type (line 386)
        long_112838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 31), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112838)
        # Adding element type (line 386)
        long_112839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 48), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112839)
        # Adding element type (line 386)
        long_112840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 66), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112840)
        # Adding element type (line 386)
        long_112841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112841)
        # Adding element type (line 386)
        long_112842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 31), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112842)
        # Adding element type (line 386)
        long_112843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 47), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 12), list_112826, long_112843)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_112826)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 390)
        list_112844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 390)
        # Adding element type (line 390)
        int_112845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, int_112845)
        # Adding element type (line 390)
        int_112846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, int_112846)
        # Adding element type (line 390)
        long_112847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 34), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112847)
        # Adding element type (line 390)
        long_112848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 47), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112848)
        # Adding element type (line 390)
        long_112849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 62), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112849)
        # Adding element type (line 390)
        long_112850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112850)
        # Adding element type (line 390)
        long_112851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 30), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112851)
        # Adding element type (line 390)
        long_112852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 47), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112852)
        # Adding element type (line 390)
        long_112853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 66), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112853)
        # Adding element type (line 390)
        long_112854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112854)
        # Adding element type (line 390)
        long_112855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 33), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112855)
        # Adding element type (line 390)
        long_112856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 52), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112856)
        # Adding element type (line 390)
        long_112857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112857)
        # Adding element type (line 390)
        long_112858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 32), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112858)
        # Adding element type (line 390)
        long_112859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 52), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112859)
        # Adding element type (line 390)
        long_112860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112860)
        # Adding element type (line 390)
        long_112861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 32), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), list_112844, long_112861)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_112844)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 395)
        list_112862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 395)
        # Adding element type (line 395)
        int_112863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, int_112863)
        # Adding element type (line 395)
        long_112864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 24), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112864)
        # Adding element type (line 395)
        long_112865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 37), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112865)
        # Adding element type (line 395)
        long_112866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 52), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112866)
        # Adding element type (line 395)
        long_112867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112867)
        # Adding element type (line 395)
        long_112868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 31), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112868)
        # Adding element type (line 395)
        long_112869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 49), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112869)
        # Adding element type (line 395)
        long_112870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112870)
        # Adding element type (line 395)
        long_112871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 32), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112871)
        # Adding element type (line 395)
        long_112872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 53), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112872)
        # Adding element type (line 395)
        long_112873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112873)
        # Adding element type (line 395)
        long_112874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 34), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112874)
        # Adding element type (line 395)
        long_112875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 55), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112875)
        # Adding element type (line 395)
        long_112876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112876)
        # Adding element type (line 395)
        long_112877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 33), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112877)
        # Adding element type (line 395)
        long_112878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 54), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112878)
        # Adding element type (line 395)
        long_112879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 12), list_112862, long_112879)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_112862)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 401)
        list_112880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 401)
        # Adding element type (line 401)
        int_112881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, int_112881)
        # Adding element type (line 401)
        long_112882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 24), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112882)
        # Adding element type (line 401)
        long_112883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112883)
        # Adding element type (line 401)
        long_112884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112884)
        # Adding element type (line 401)
        long_112885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 31), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112885)
        # Adding element type (line 401)
        long_112886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 49), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112886)
        # Adding element type (line 401)
        long_112887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112887)
        # Adding element type (line 401)
        long_112888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 33), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112888)
        # Adding element type (line 401)
        long_112889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 55), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112889)
        # Adding element type (line 401)
        long_112890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112890)
        # Adding element type (line 401)
        long_112891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 35), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112891)
        # Adding element type (line 401)
        long_112892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 57), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112892)
        # Adding element type (line 401)
        long_112893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112893)
        # Adding element type (line 401)
        long_112894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 35), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112894)
        # Adding element type (line 401)
        long_112895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 58), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112895)
        # Adding element type (line 401)
        long_112896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112896)
        # Adding element type (line 401)
        long_112897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 35), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 12), list_112880, long_112897)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_112880)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 407)
        list_112898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 407)
        # Adding element type (line 407)
        long_112899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112899)
        # Adding element type (line 407)
        long_112900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 26), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112900)
        # Adding element type (line 407)
        long_112901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 41), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112901)
        # Adding element type (line 407)
        long_112902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 58), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112902)
        # Adding element type (line 407)
        long_112903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112903)
        # Adding element type (line 407)
        long_112904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 33), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112904)
        # Adding element type (line 407)
        long_112905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 53), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112905)
        # Adding element type (line 407)
        long_112906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112906)
        # Adding element type (line 407)
        long_112907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 35), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112907)
        # Adding element type (line 407)
        long_112908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 58), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112908)
        # Adding element type (line 407)
        long_112909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112909)
        # Adding element type (line 407)
        long_112910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 37), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112910)
        # Adding element type (line 407)
        long_112911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 60), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112911)
        # Adding element type (line 407)
        long_112912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112912)
        # Adding element type (line 407)
        long_112913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 36), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112913)
        # Adding element type (line 407)
        long_112914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 59), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112914)
        # Adding element type (line 407)
        long_112915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 12), list_112898, long_112915)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_112898)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 413)
        list_112916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 413)
        # Adding element type (line 413)
        long_112917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112917)
        # Adding element type (line 413)
        long_112918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 26), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112918)
        # Adding element type (line 413)
        long_112919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 42), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112919)
        # Adding element type (line 413)
        long_112920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112920)
        # Adding element type (line 413)
        long_112921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 33), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112921)
        # Adding element type (line 413)
        long_112922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 53), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112922)
        # Adding element type (line 413)
        long_112923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112923)
        # Adding element type (line 413)
        long_112924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 35), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112924)
        # Adding element type (line 413)
        long_112925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 58), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112925)
        # Adding element type (line 413)
        long_112926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112926)
        # Adding element type (line 413)
        long_112927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 37), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112927)
        # Adding element type (line 413)
        long_112928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 60), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112928)
        # Adding element type (line 413)
        long_112929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112929)
        # Adding element type (line 413)
        long_112930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 37), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112930)
        # Adding element type (line 413)
        long_112931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 61), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112931)
        # Adding element type (line 413)
        long_112932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112932)
        # Adding element type (line 413)
        long_112933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 36), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_112916, long_112933)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_112916)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 419)
        list_112934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 419)
        # Adding element type (line 419)
        long_112935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112935)
        # Adding element type (line 419)
        long_112936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 28), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112936)
        # Adding element type (line 419)
        long_112937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 44), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112937)
        # Adding element type (line 419)
        long_112938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 63), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112938)
        # Adding element type (line 419)
        long_112939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112939)
        # Adding element type (line 419)
        long_112940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 35), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112940)
        # Adding element type (line 419)
        long_112941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 57), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112941)
        # Adding element type (line 419)
        long_112942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112942)
        # Adding element type (line 419)
        long_112943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 36), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112943)
        # Adding element type (line 419)
        long_112944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 60), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112944)
        # Adding element type (line 419)
        long_112945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112945)
        # Adding element type (line 419)
        long_112946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112946)
        # Adding element type (line 419)
        long_112947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112947)
        # Adding element type (line 419)
        long_112948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112948)
        # Adding element type (line 419)
        long_112949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112949)
        # Adding element type (line 419)
        long_112950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112950)
        # Adding element type (line 419)
        long_112951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 61), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 12), list_112934, long_112951)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_112934)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 425)
        list_112952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 425)
        # Adding element type (line 425)
        long_112953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112953)
        # Adding element type (line 425)
        long_112954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 27), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112954)
        # Adding element type (line 425)
        long_112955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 45), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112955)
        # Adding element type (line 425)
        long_112956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 63), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112956)
        # Adding element type (line 425)
        long_112957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112957)
        # Adding element type (line 425)
        long_112958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 34), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112958)
        # Adding element type (line 425)
        long_112959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 57), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112959)
        # Adding element type (line 425)
        long_112960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112960)
        # Adding element type (line 425)
        long_112961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 37), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112961)
        # Adding element type (line 425)
        long_112962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112962)
        # Adding element type (line 425)
        long_112963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112963)
        # Adding element type (line 425)
        long_112964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112964)
        # Adding element type (line 425)
        long_112965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112965)
        # Adding element type (line 425)
        long_112966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112966)
        # Adding element type (line 425)
        long_112967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112967)
        # Adding element type (line 425)
        long_112968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112968)
        # Adding element type (line 425)
        long_112969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 12), list_112952, long_112969)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_112952)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 432)
        list_112970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 432)
        # Adding element type (line 432)
        long_112971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112971)
        # Adding element type (line 432)
        long_112972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 29), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112972)
        # Adding element type (line 432)
        long_112973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 46), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112973)
        # Adding element type (line 432)
        long_112974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112974)
        # Adding element type (line 432)
        long_112975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 33), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112975)
        # Adding element type (line 432)
        long_112976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 55), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112976)
        # Adding element type (line 432)
        long_112977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112977)
        # Adding element type (line 432)
        long_112978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 37), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112978)
        # Adding element type (line 432)
        long_112979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112979)
        # Adding element type (line 432)
        long_112980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112980)
        # Adding element type (line 432)
        long_112981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112981)
        # Adding element type (line 432)
        long_112982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112982)
        # Adding element type (line 432)
        long_112983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112983)
        # Adding element type (line 432)
        long_112984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112984)
        # Adding element type (line 432)
        long_112985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112985)
        # Adding element type (line 432)
        long_112986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112986)
        # Adding element type (line 432)
        long_112987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 12), list_112970, long_112987)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_112970)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 440)
        list_112988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 440)
        # Adding element type (line 440)
        long_112989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_112989)
        # Adding element type (line 440)
        long_112990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 28), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_112990)
        # Adding element type (line 440)
        long_112991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_112991)
        # Adding element type (line 440)
        long_112992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 32), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_112992)
        # Adding element type (line 440)
        long_112993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 53), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_112993)
        # Adding element type (line 440)
        long_112994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_112994)
        # Adding element type (line 440)
        long_112995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 37), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_112995)
        # Adding element type (line 440)
        long_112996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 60), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_112996)
        # Adding element type (line 440)
        long_112997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_112997)
        # Adding element type (line 440)
        long_112998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_112998)
        # Adding element type (line 440)
        long_112999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_112999)
        # Adding element type (line 440)
        long_113000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_113000)
        # Adding element type (line 440)
        long_113001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_113001)
        # Adding element type (line 440)
        long_113002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_113002)
        # Adding element type (line 440)
        long_113003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_113003)
        # Adding element type (line 440)
        long_113004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_113004)
        # Adding element type (line 440)
        long_113005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 63), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 12), list_112988, long_113005)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_112988)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 447)
        list_113006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 447)
        # Adding element type (line 447)
        long_113007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113007)
        # Adding element type (line 447)
        long_113008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 29), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113008)
        # Adding element type (line 447)
        long_113009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 46), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113009)
        # Adding element type (line 447)
        long_113010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113010)
        # Adding element type (line 447)
        long_113011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 34), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113011)
        # Adding element type (line 447)
        long_113012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 57), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113012)
        # Adding element type (line 447)
        long_113013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113013)
        # Adding element type (line 447)
        long_113014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113014)
        # Adding element type (line 447)
        long_113015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113015)
        # Adding element type (line 447)
        long_113016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113016)
        # Adding element type (line 447)
        long_113017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113017)
        # Adding element type (line 447)
        long_113018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113018)
        # Adding element type (line 447)
        long_113019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113019)
        # Adding element type (line 447)
        long_113020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 40), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113020)
        # Adding element type (line 447)
        long_113021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113021)
        # Adding element type (line 447)
        long_113022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113022)
        # Adding element type (line 447)
        long_113023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 12), list_113006, long_113023)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_113006)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 455)
        list_113024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 455)
        # Adding element type (line 455)
        long_113025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113025)
        # Adding element type (line 455)
        long_113026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 28), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113026)
        # Adding element type (line 455)
        long_113027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 46), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113027)
        # Adding element type (line 455)
        long_113028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113028)
        # Adding element type (line 455)
        long_113029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 35), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113029)
        # Adding element type (line 455)
        long_113030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 57), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113030)
        # Adding element type (line 455)
        long_113031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113031)
        # Adding element type (line 455)
        long_113032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 37), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113032)
        # Adding element type (line 455)
        long_113033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113033)
        # Adding element type (line 455)
        long_113034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113034)
        # Adding element type (line 455)
        long_113035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113035)
        # Adding element type (line 455)
        long_113036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113036)
        # Adding element type (line 455)
        long_113037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113037)
        # Adding element type (line 455)
        long_113038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113038)
        # Adding element type (line 455)
        long_113039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113039)
        # Adding element type (line 455)
        long_113040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113040)
        # Adding element type (line 455)
        long_113041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 12), list_113024, long_113041)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_113024)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 463)
        list_113042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 463)
        # Adding element type (line 463)
        long_113043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113043)
        # Adding element type (line 463)
        long_113044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 29), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113044)
        # Adding element type (line 463)
        long_113045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113045)
        # Adding element type (line 463)
        long_113046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 33), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113046)
        # Adding element type (line 463)
        long_113047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 53), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113047)
        # Adding element type (line 463)
        long_113048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113048)
        # Adding element type (line 463)
        long_113049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 36), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113049)
        # Adding element type (line 463)
        long_113050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 60), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113050)
        # Adding element type (line 463)
        long_113051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113051)
        # Adding element type (line 463)
        long_113052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113052)
        # Adding element type (line 463)
        long_113053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113053)
        # Adding element type (line 463)
        long_113054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113054)
        # Adding element type (line 463)
        long_113055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113055)
        # Adding element type (line 463)
        long_113056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113056)
        # Adding element type (line 463)
        long_113057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113057)
        # Adding element type (line 463)
        long_113058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 39), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113058)
        # Adding element type (line 463)
        long_113059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 12), list_113042, long_113059)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_113042)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 471)
        list_113060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 471)
        # Adding element type (line 471)
        long_113061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113061)
        # Adding element type (line 471)
        long_113062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 27), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113062)
        # Adding element type (line 471)
        long_113063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113063)
        # Adding element type (line 471)
        long_113064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 32), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113064)
        # Adding element type (line 471)
        long_113065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 53), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113065)
        # Adding element type (line 471)
        long_113066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113066)
        # Adding element type (line 471)
        long_113067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 36), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113067)
        # Adding element type (line 471)
        long_113068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 59), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113068)
        # Adding element type (line 471)
        long_113069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113069)
        # Adding element type (line 471)
        long_113070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 37), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113070)
        # Adding element type (line 471)
        long_113071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113071)
        # Adding element type (line 471)
        long_113072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113072)
        # Adding element type (line 471)
        long_113073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113073)
        # Adding element type (line 471)
        long_113074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113074)
        # Adding element type (line 471)
        long_113075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113075)
        # Adding element type (line 471)
        long_113076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113076)
        # Adding element type (line 471)
        long_113077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 63), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 12), list_113060, long_113077)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_113060)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 478)
        list_113078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 478)
        # Adding element type (line 478)
        long_113079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113079)
        # Adding element type (line 478)
        long_113080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 28), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113080)
        # Adding element type (line 478)
        long_113081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 44), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113081)
        # Adding element type (line 478)
        long_113082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 63), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113082)
        # Adding element type (line 478)
        long_113083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113083)
        # Adding element type (line 478)
        long_113084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 35), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113084)
        # Adding element type (line 478)
        long_113085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 57), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113085)
        # Adding element type (line 478)
        long_113086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113086)
        # Adding element type (line 478)
        long_113087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 36), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113087)
        # Adding element type (line 478)
        long_113088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 61), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113088)
        # Adding element type (line 478)
        long_113089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113089)
        # Adding element type (line 478)
        long_113090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113090)
        # Adding element type (line 478)
        long_113091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113091)
        # Adding element type (line 478)
        long_113092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113092)
        # Adding element type (line 478)
        long_113093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113093)
        # Adding element type (line 478)
        long_113094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 38), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113094)
        # Adding element type (line 478)
        long_113095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 61), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 12), list_113078, long_113095)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_113078)
        # Adding element type (line 381)
        
        # Obtaining an instance of the builtin type 'list' (line 484)
        list_113096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 484)
        # Adding element type (line 484)
        long_113097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113097)
        # Adding element type (line 484)
        long_113098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 26), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113098)
        # Adding element type (line 484)
        long_113099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 42), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113099)
        # Adding element type (line 484)
        long_113100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 59), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113100)
        # Adding element type (line 484)
        long_113101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113101)
        # Adding element type (line 484)
        long_113102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 33), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113102)
        # Adding element type (line 484)
        long_113103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 55), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113103)
        # Adding element type (line 484)
        long_113104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113104)
        # Adding element type (line 484)
        long_113105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 36), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113105)
        # Adding element type (line 484)
        long_113106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 59), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113106)
        # Adding element type (line 484)
        long_113107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113107)
        # Adding element type (line 484)
        long_113108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 36), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113108)
        # Adding element type (line 484)
        long_113109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 61), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113109)
        # Adding element type (line 484)
        long_113110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113110)
        # Adding element type (line 484)
        long_113111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 37), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113111)
        # Adding element type (line 484)
        long_113112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 60), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113112)
        # Adding element type (line 484)
        long_113113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 13), 'long')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_113096, long_113113)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), list_112807, list_113096)
        
        # Processing the call keyword arguments (line 381)
        kwargs_113114 = {}
        # Getting the type of 'array' (line 381)
        array_112806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 17), 'array', False)
        # Calling array(args, kwargs) (line 381)
        array_call_result_113115 = invoke(stypy.reporting.localization.Localization(__file__, 381, 17), array_112806, *[list_112807], **kwargs_113114)
        
        # Assigning a type to the variable 'invh17' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'invh17', array_call_result_113115)
        
        # Call to assert_array_equal(...): (line 491)
        # Processing the call arguments (line 491)
        
        # Call to invhilbert(...): (line 491)
        # Processing the call arguments (line 491)
        int_113118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 38), 'int')
        # Processing the call keyword arguments (line 491)
        # Getting the type of 'True' (line 491)
        True_113119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 48), 'True', False)
        keyword_113120 = True_113119
        kwargs_113121 = {'exact': keyword_113120}
        # Getting the type of 'invhilbert' (line 491)
        invhilbert_113117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 27), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 491)
        invhilbert_call_result_113122 = invoke(stypy.reporting.localization.Localization(__file__, 491, 27), invhilbert_113117, *[int_113118], **kwargs_113121)
        
        # Getting the type of 'invh17' (line 491)
        invh17_113123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 55), 'invh17', False)
        # Processing the call keyword arguments (line 491)
        kwargs_113124 = {}
        # Getting the type of 'assert_array_equal' (line 491)
        assert_array_equal_113116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 491)
        assert_array_equal_call_result_113125 = invoke(stypy.reporting.localization.Localization(__file__, 491, 8), assert_array_equal_113116, *[invhilbert_call_result_113122, invh17_113123], **kwargs_113124)
        
        
        # Call to assert_allclose(...): (line 492)
        # Processing the call arguments (line 492)
        
        # Call to invhilbert(...): (line 492)
        # Processing the call arguments (line 492)
        int_113128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 35), 'int')
        # Processing the call keyword arguments (line 492)
        kwargs_113129 = {}
        # Getting the type of 'invhilbert' (line 492)
        invhilbert_113127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 24), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 492)
        invhilbert_call_result_113130 = invoke(stypy.reporting.localization.Localization(__file__, 492, 24), invhilbert_113127, *[int_113128], **kwargs_113129)
        
        
        # Call to astype(...): (line 492)
        # Processing the call arguments (line 492)
        # Getting the type of 'float' (line 492)
        float_113133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 54), 'float', False)
        # Processing the call keyword arguments (line 492)
        kwargs_113134 = {}
        # Getting the type of 'invh17' (line 492)
        invh17_113131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 40), 'invh17', False)
        # Obtaining the member 'astype' of a type (line 492)
        astype_113132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 40), invh17_113131, 'astype')
        # Calling astype(args, kwargs) (line 492)
        astype_call_result_113135 = invoke(stypy.reporting.localization.Localization(__file__, 492, 40), astype_113132, *[float_113133], **kwargs_113134)
        
        # Processing the call keyword arguments (line 492)
        float_113136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 67), 'float')
        keyword_113137 = float_113136
        kwargs_113138 = {'rtol': keyword_113137}
        # Getting the type of 'assert_allclose' (line 492)
        assert_allclose_113126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 492)
        assert_allclose_call_result_113139 = invoke(stypy.reporting.localization.Localization(__file__, 492, 8), assert_allclose_113126, *[invhilbert_call_result_113130, astype_call_result_113135], **kwargs_113138)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 350)
        stypy_return_type_113140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113140)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_113140


    @norecursion
    def test_inverse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_inverse'
        module_type_store = module_type_store.open_function_context('test_inverse', 494, 4, False)
        # Assigning a type to the variable 'self' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInvHilbert.test_inverse.__dict__.__setitem__('stypy_localization', localization)
        TestInvHilbert.test_inverse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInvHilbert.test_inverse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInvHilbert.test_inverse.__dict__.__setitem__('stypy_function_name', 'TestInvHilbert.test_inverse')
        TestInvHilbert.test_inverse.__dict__.__setitem__('stypy_param_names_list', [])
        TestInvHilbert.test_inverse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInvHilbert.test_inverse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInvHilbert.test_inverse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInvHilbert.test_inverse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInvHilbert.test_inverse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInvHilbert.test_inverse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInvHilbert.test_inverse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_inverse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_inverse(...)' code ##################

        
        
        # Call to xrange(...): (line 495)
        # Processing the call arguments (line 495)
        int_113142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 24), 'int')
        int_113143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 27), 'int')
        # Processing the call keyword arguments (line 495)
        kwargs_113144 = {}
        # Getting the type of 'xrange' (line 495)
        xrange_113141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 495)
        xrange_call_result_113145 = invoke(stypy.reporting.localization.Localization(__file__, 495, 17), xrange_113141, *[int_113142, int_113143], **kwargs_113144)
        
        # Testing the type of a for loop iterable (line 495)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 495, 8), xrange_call_result_113145)
        # Getting the type of the for loop variable (line 495)
        for_loop_var_113146 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 495, 8), xrange_call_result_113145)
        # Assigning a type to the variable 'n' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'n', for_loop_var_113146)
        # SSA begins for a for statement (line 495)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 496):
        
        # Call to hilbert(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'n' (line 496)
        n_113148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 24), 'n', False)
        # Processing the call keyword arguments (line 496)
        kwargs_113149 = {}
        # Getting the type of 'hilbert' (line 496)
        hilbert_113147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 16), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 496)
        hilbert_call_result_113150 = invoke(stypy.reporting.localization.Localization(__file__, 496, 16), hilbert_113147, *[n_113148], **kwargs_113149)
        
        # Assigning a type to the variable 'a' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'a', hilbert_call_result_113150)
        
        # Assigning a Call to a Name (line 497):
        
        # Call to invhilbert(...): (line 497)
        # Processing the call arguments (line 497)
        # Getting the type of 'n' (line 497)
        n_113152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 27), 'n', False)
        # Processing the call keyword arguments (line 497)
        kwargs_113153 = {}
        # Getting the type of 'invhilbert' (line 497)
        invhilbert_113151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'invhilbert', False)
        # Calling invhilbert(args, kwargs) (line 497)
        invhilbert_call_result_113154 = invoke(stypy.reporting.localization.Localization(__file__, 497, 16), invhilbert_113151, *[n_113152], **kwargs_113153)
        
        # Assigning a type to the variable 'b' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'b', invhilbert_call_result_113154)
        
        # Assigning a Call to a Name (line 500):
        
        # Call to cond(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'a' (line 500)
        a_113156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 21), 'a', False)
        # Processing the call keyword arguments (line 500)
        kwargs_113157 = {}
        # Getting the type of 'cond' (line 500)
        cond_113155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 16), 'cond', False)
        # Calling cond(args, kwargs) (line 500)
        cond_call_result_113158 = invoke(stypy.reporting.localization.Localization(__file__, 500, 16), cond_113155, *[a_113156], **kwargs_113157)
        
        # Assigning a type to the variable 'c' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'c', cond_call_result_113158)
        
        # Call to assert_allclose(...): (line 501)
        # Processing the call arguments (line 501)
        
        # Call to dot(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'b' (line 501)
        b_113162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 34), 'b', False)
        # Processing the call keyword arguments (line 501)
        kwargs_113163 = {}
        # Getting the type of 'a' (line 501)
        a_113160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 28), 'a', False)
        # Obtaining the member 'dot' of a type (line 501)
        dot_113161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 28), a_113160, 'dot')
        # Calling dot(args, kwargs) (line 501)
        dot_call_result_113164 = invoke(stypy.reporting.localization.Localization(__file__, 501, 28), dot_113161, *[b_113162], **kwargs_113163)
        
        
        # Call to eye(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'n' (line 501)
        n_113166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 42), 'n', False)
        # Processing the call keyword arguments (line 501)
        kwargs_113167 = {}
        # Getting the type of 'eye' (line 501)
        eye_113165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 38), 'eye', False)
        # Calling eye(args, kwargs) (line 501)
        eye_call_result_113168 = invoke(stypy.reporting.localization.Localization(__file__, 501, 38), eye_113165, *[n_113166], **kwargs_113167)
        
        # Processing the call keyword arguments (line 501)
        float_113169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 51), 'float')
        # Getting the type of 'c' (line 501)
        c_113170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 57), 'c', False)
        # Applying the binary operator '*' (line 501)
        result_mul_113171 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 51), '*', float_113169, c_113170)
        
        keyword_113172 = result_mul_113171
        float_113173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 65), 'float')
        # Getting the type of 'c' (line 501)
        c_113174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 71), 'c', False)
        # Applying the binary operator '*' (line 501)
        result_mul_113175 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 65), '*', float_113173, c_113174)
        
        keyword_113176 = result_mul_113175
        kwargs_113177 = {'rtol': keyword_113176, 'atol': keyword_113172}
        # Getting the type of 'assert_allclose' (line 501)
        assert_allclose_113159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 501)
        assert_allclose_call_result_113178 = invoke(stypy.reporting.localization.Localization(__file__, 501, 12), assert_allclose_113159, *[dot_call_result_113164, eye_call_result_113168], **kwargs_113177)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_inverse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_inverse' in the type store
        # Getting the type of 'stypy_return_type' (line 494)
        stypy_return_type_113179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113179)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_inverse'
        return stypy_return_type_113179


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 348, 0, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInvHilbert.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestInvHilbert' (line 348)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 0), 'TestInvHilbert', TestInvHilbert)
# Declaration of the 'TestPascal' class

class TestPascal(object, ):

    @norecursion
    def check_case(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_case'
        module_type_store = module_type_store.open_function_context('check_case', 528, 4, False)
        # Assigning a type to the variable 'self' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPascal.check_case.__dict__.__setitem__('stypy_localization', localization)
        TestPascal.check_case.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPascal.check_case.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPascal.check_case.__dict__.__setitem__('stypy_function_name', 'TestPascal.check_case')
        TestPascal.check_case.__dict__.__setitem__('stypy_param_names_list', ['n', 'sym', 'low'])
        TestPascal.check_case.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPascal.check_case.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPascal.check_case.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPascal.check_case.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPascal.check_case.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPascal.check_case.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPascal.check_case', ['n', 'sym', 'low'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_case', localization, ['n', 'sym', 'low'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_case(...)' code ##################

        
        # Call to assert_array_equal(...): (line 529)
        # Processing the call arguments (line 529)
        
        # Call to pascal(...): (line 529)
        # Processing the call arguments (line 529)
        # Getting the type of 'n' (line 529)
        n_113182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 34), 'n', False)
        # Processing the call keyword arguments (line 529)
        kwargs_113183 = {}
        # Getting the type of 'pascal' (line 529)
        pascal_113181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 27), 'pascal', False)
        # Calling pascal(args, kwargs) (line 529)
        pascal_call_result_113184 = invoke(stypy.reporting.localization.Localization(__file__, 529, 27), pascal_113181, *[n_113182], **kwargs_113183)
        
        # Getting the type of 'sym' (line 529)
        sym_113185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 38), 'sym', False)
        # Processing the call keyword arguments (line 529)
        kwargs_113186 = {}
        # Getting the type of 'assert_array_equal' (line 529)
        assert_array_equal_113180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 529)
        assert_array_equal_call_result_113187 = invoke(stypy.reporting.localization.Localization(__file__, 529, 8), assert_array_equal_113180, *[pascal_call_result_113184, sym_113185], **kwargs_113186)
        
        
        # Call to assert_array_equal(...): (line 530)
        # Processing the call arguments (line 530)
        
        # Call to pascal(...): (line 530)
        # Processing the call arguments (line 530)
        # Getting the type of 'n' (line 530)
        n_113190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 34), 'n', False)
        # Processing the call keyword arguments (line 530)
        str_113191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 42), 'str', 'lower')
        keyword_113192 = str_113191
        kwargs_113193 = {'kind': keyword_113192}
        # Getting the type of 'pascal' (line 530)
        pascal_113189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 27), 'pascal', False)
        # Calling pascal(args, kwargs) (line 530)
        pascal_call_result_113194 = invoke(stypy.reporting.localization.Localization(__file__, 530, 27), pascal_113189, *[n_113190], **kwargs_113193)
        
        # Getting the type of 'low' (line 530)
        low_113195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 52), 'low', False)
        # Processing the call keyword arguments (line 530)
        kwargs_113196 = {}
        # Getting the type of 'assert_array_equal' (line 530)
        assert_array_equal_113188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 530)
        assert_array_equal_call_result_113197 = invoke(stypy.reporting.localization.Localization(__file__, 530, 8), assert_array_equal_113188, *[pascal_call_result_113194, low_113195], **kwargs_113196)
        
        
        # Call to assert_array_equal(...): (line 531)
        # Processing the call arguments (line 531)
        
        # Call to pascal(...): (line 531)
        # Processing the call arguments (line 531)
        # Getting the type of 'n' (line 531)
        n_113200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 34), 'n', False)
        # Processing the call keyword arguments (line 531)
        str_113201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 42), 'str', 'upper')
        keyword_113202 = str_113201
        kwargs_113203 = {'kind': keyword_113202}
        # Getting the type of 'pascal' (line 531)
        pascal_113199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 27), 'pascal', False)
        # Calling pascal(args, kwargs) (line 531)
        pascal_call_result_113204 = invoke(stypy.reporting.localization.Localization(__file__, 531, 27), pascal_113199, *[n_113200], **kwargs_113203)
        
        # Getting the type of 'low' (line 531)
        low_113205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 52), 'low', False)
        # Obtaining the member 'T' of a type (line 531)
        T_113206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 52), low_113205, 'T')
        # Processing the call keyword arguments (line 531)
        kwargs_113207 = {}
        # Getting the type of 'assert_array_equal' (line 531)
        assert_array_equal_113198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 531)
        assert_array_equal_call_result_113208 = invoke(stypy.reporting.localization.Localization(__file__, 531, 8), assert_array_equal_113198, *[pascal_call_result_113204, T_113206], **kwargs_113207)
        
        
        # Call to assert_array_almost_equal(...): (line 532)
        # Processing the call arguments (line 532)
        
        # Call to pascal(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 'n' (line 532)
        n_113211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 41), 'n', False)
        # Processing the call keyword arguments (line 532)
        # Getting the type of 'False' (line 532)
        False_113212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 50), 'False', False)
        keyword_113213 = False_113212
        kwargs_113214 = {'exact': keyword_113213}
        # Getting the type of 'pascal' (line 532)
        pascal_113210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 34), 'pascal', False)
        # Calling pascal(args, kwargs) (line 532)
        pascal_call_result_113215 = invoke(stypy.reporting.localization.Localization(__file__, 532, 34), pascal_113210, *[n_113211], **kwargs_113214)
        
        # Getting the type of 'sym' (line 532)
        sym_113216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 58), 'sym', False)
        # Processing the call keyword arguments (line 532)
        kwargs_113217 = {}
        # Getting the type of 'assert_array_almost_equal' (line 532)
        assert_array_almost_equal_113209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 532)
        assert_array_almost_equal_call_result_113218 = invoke(stypy.reporting.localization.Localization(__file__, 532, 8), assert_array_almost_equal_113209, *[pascal_call_result_113215, sym_113216], **kwargs_113217)
        
        
        # Call to assert_array_almost_equal(...): (line 533)
        # Processing the call arguments (line 533)
        
        # Call to pascal(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'n' (line 533)
        n_113221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 41), 'n', False)
        # Processing the call keyword arguments (line 533)
        # Getting the type of 'False' (line 533)
        False_113222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 50), 'False', False)
        keyword_113223 = False_113222
        str_113224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 62), 'str', 'lower')
        keyword_113225 = str_113224
        kwargs_113226 = {'kind': keyword_113225, 'exact': keyword_113223}
        # Getting the type of 'pascal' (line 533)
        pascal_113220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 34), 'pascal', False)
        # Calling pascal(args, kwargs) (line 533)
        pascal_call_result_113227 = invoke(stypy.reporting.localization.Localization(__file__, 533, 34), pascal_113220, *[n_113221], **kwargs_113226)
        
        # Getting the type of 'low' (line 533)
        low_113228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 72), 'low', False)
        # Processing the call keyword arguments (line 533)
        kwargs_113229 = {}
        # Getting the type of 'assert_array_almost_equal' (line 533)
        assert_array_almost_equal_113219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 533)
        assert_array_almost_equal_call_result_113230 = invoke(stypy.reporting.localization.Localization(__file__, 533, 8), assert_array_almost_equal_113219, *[pascal_call_result_113227, low_113228], **kwargs_113229)
        
        
        # Call to assert_array_almost_equal(...): (line 534)
        # Processing the call arguments (line 534)
        
        # Call to pascal(...): (line 534)
        # Processing the call arguments (line 534)
        # Getting the type of 'n' (line 534)
        n_113233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 41), 'n', False)
        # Processing the call keyword arguments (line 534)
        # Getting the type of 'False' (line 534)
        False_113234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 50), 'False', False)
        keyword_113235 = False_113234
        str_113236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 62), 'str', 'upper')
        keyword_113237 = str_113236
        kwargs_113238 = {'kind': keyword_113237, 'exact': keyword_113235}
        # Getting the type of 'pascal' (line 534)
        pascal_113232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 34), 'pascal', False)
        # Calling pascal(args, kwargs) (line 534)
        pascal_call_result_113239 = invoke(stypy.reporting.localization.Localization(__file__, 534, 34), pascal_113232, *[n_113233], **kwargs_113238)
        
        # Getting the type of 'low' (line 534)
        low_113240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 72), 'low', False)
        # Obtaining the member 'T' of a type (line 534)
        T_113241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 72), low_113240, 'T')
        # Processing the call keyword arguments (line 534)
        kwargs_113242 = {}
        # Getting the type of 'assert_array_almost_equal' (line 534)
        assert_array_almost_equal_113231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 534)
        assert_array_almost_equal_call_result_113243 = invoke(stypy.reporting.localization.Localization(__file__, 534, 8), assert_array_almost_equal_113231, *[pascal_call_result_113239, T_113241], **kwargs_113242)
        
        
        # ################# End of 'check_case(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_case' in the type store
        # Getting the type of 'stypy_return_type' (line 528)
        stypy_return_type_113244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113244)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_case'
        return stypy_return_type_113244


    @norecursion
    def test_cases(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cases'
        module_type_store = module_type_store.open_function_context('test_cases', 536, 4, False)
        # Assigning a type to the variable 'self' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPascal.test_cases.__dict__.__setitem__('stypy_localization', localization)
        TestPascal.test_cases.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPascal.test_cases.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPascal.test_cases.__dict__.__setitem__('stypy_function_name', 'TestPascal.test_cases')
        TestPascal.test_cases.__dict__.__setitem__('stypy_param_names_list', [])
        TestPascal.test_cases.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPascal.test_cases.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPascal.test_cases.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPascal.test_cases.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPascal.test_cases.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPascal.test_cases.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPascal.test_cases', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cases', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cases(...)' code ##################

        
        # Getting the type of 'self' (line 537)
        self_113245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 27), 'self')
        # Obtaining the member 'cases' of a type (line 537)
        cases_113246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 27), self_113245, 'cases')
        # Testing the type of a for loop iterable (line 537)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 537, 8), cases_113246)
        # Getting the type of the for loop variable (line 537)
        for_loop_var_113247 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 537, 8), cases_113246)
        # Assigning a type to the variable 'n' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 8), for_loop_var_113247))
        # Assigning a type to the variable 'sym' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'sym', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 8), for_loop_var_113247))
        # Assigning a type to the variable 'low' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'low', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 8), for_loop_var_113247))
        # SSA begins for a for statement (line 537)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to check_case(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'n' (line 538)
        n_113250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 28), 'n', False)
        # Getting the type of 'sym' (line 538)
        sym_113251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 31), 'sym', False)
        # Getting the type of 'low' (line 538)
        low_113252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 36), 'low', False)
        # Processing the call keyword arguments (line 538)
        kwargs_113253 = {}
        # Getting the type of 'self' (line 538)
        self_113248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'self', False)
        # Obtaining the member 'check_case' of a type (line 538)
        check_case_113249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 12), self_113248, 'check_case')
        # Calling check_case(args, kwargs) (line 538)
        check_case_call_result_113254 = invoke(stypy.reporting.localization.Localization(__file__, 538, 12), check_case_113249, *[n_113250, sym_113251, low_113252], **kwargs_113253)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_cases(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cases' in the type store
        # Getting the type of 'stypy_return_type' (line 536)
        stypy_return_type_113255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113255)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cases'
        return stypy_return_type_113255


    @norecursion
    def test_big(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_big'
        module_type_store = module_type_store.open_function_context('test_big', 540, 4, False)
        # Assigning a type to the variable 'self' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPascal.test_big.__dict__.__setitem__('stypy_localization', localization)
        TestPascal.test_big.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPascal.test_big.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPascal.test_big.__dict__.__setitem__('stypy_function_name', 'TestPascal.test_big')
        TestPascal.test_big.__dict__.__setitem__('stypy_param_names_list', [])
        TestPascal.test_big.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPascal.test_big.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPascal.test_big.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPascal.test_big.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPascal.test_big.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPascal.test_big.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPascal.test_big', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_big', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_big(...)' code ##################

        
        # Assigning a Call to a Name (line 541):
        
        # Call to pascal(...): (line 541)
        # Processing the call arguments (line 541)
        int_113257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 19), 'int')
        # Processing the call keyword arguments (line 541)
        kwargs_113258 = {}
        # Getting the type of 'pascal' (line 541)
        pascal_113256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'pascal', False)
        # Calling pascal(args, kwargs) (line 541)
        pascal_call_result_113259 = invoke(stypy.reporting.localization.Localization(__file__, 541, 12), pascal_113256, *[int_113257], **kwargs_113258)
        
        # Assigning a type to the variable 'p' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'p', pascal_call_result_113259)
        
        # Call to assert_equal(...): (line 542)
        # Processing the call arguments (line 542)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 542)
        tuple_113261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 542)
        # Adding element type (line 542)
        int_113262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 23), tuple_113261, int_113262)
        # Adding element type (line 542)
        int_113263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 23), tuple_113261, int_113263)
        
        # Getting the type of 'p' (line 542)
        p_113264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 21), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 542)
        getitem___113265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 21), p_113264, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 542)
        subscript_call_result_113266 = invoke(stypy.reporting.localization.Localization(__file__, 542, 21), getitem___113265, tuple_113261)
        
        
        # Call to comb(...): (line 542)
        # Processing the call arguments (line 542)
        int_113268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 37), 'int')
        int_113269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 41), 'int')
        # Processing the call keyword arguments (line 542)
        # Getting the type of 'True' (line 542)
        True_113270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 51), 'True', False)
        keyword_113271 = True_113270
        kwargs_113272 = {'exact': keyword_113271}
        # Getting the type of 'comb' (line 542)
        comb_113267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 32), 'comb', False)
        # Calling comb(args, kwargs) (line 542)
        comb_call_result_113273 = invoke(stypy.reporting.localization.Localization(__file__, 542, 32), comb_113267, *[int_113268, int_113269], **kwargs_113272)
        
        # Processing the call keyword arguments (line 542)
        kwargs_113274 = {}
        # Getting the type of 'assert_equal' (line 542)
        assert_equal_113260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 542)
        assert_equal_call_result_113275 = invoke(stypy.reporting.localization.Localization(__file__, 542, 8), assert_equal_113260, *[subscript_call_result_113266, comb_call_result_113273], **kwargs_113274)
        
        
        # ################# End of 'test_big(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_big' in the type store
        # Getting the type of 'stypy_return_type' (line 540)
        stypy_return_type_113276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_big'
        return stypy_return_type_113276


    @norecursion
    def test_threshold(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_threshold'
        module_type_store = module_type_store.open_function_context('test_threshold', 544, 4, False)
        # Assigning a type to the variable 'self' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPascal.test_threshold.__dict__.__setitem__('stypy_localization', localization)
        TestPascal.test_threshold.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPascal.test_threshold.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPascal.test_threshold.__dict__.__setitem__('stypy_function_name', 'TestPascal.test_threshold')
        TestPascal.test_threshold.__dict__.__setitem__('stypy_param_names_list', [])
        TestPascal.test_threshold.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPascal.test_threshold.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPascal.test_threshold.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPascal.test_threshold.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPascal.test_threshold.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPascal.test_threshold.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPascal.test_threshold', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_threshold', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_threshold(...)' code ##################

        
        # Assigning a Call to a Name (line 549):
        
        # Call to pascal(...): (line 549)
        # Processing the call arguments (line 549)
        int_113278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 19), 'int')
        # Processing the call keyword arguments (line 549)
        kwargs_113279 = {}
        # Getting the type of 'pascal' (line 549)
        pascal_113277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'pascal', False)
        # Calling pascal(args, kwargs) (line 549)
        pascal_call_result_113280 = invoke(stypy.reporting.localization.Localization(__file__, 549, 12), pascal_113277, *[int_113278], **kwargs_113279)
        
        # Assigning a type to the variable 'p' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'p', pascal_call_result_113280)
        
        # Call to assert_equal(...): (line 550)
        # Processing the call arguments (line 550)
        int_113282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 21), 'int')
        
        # Call to item(...): (line 550)
        # Processing the call arguments (line 550)
        int_113285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 30), 'int')
        int_113286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 34), 'int')
        # Processing the call keyword arguments (line 550)
        kwargs_113287 = {}
        # Getting the type of 'p' (line 550)
        p_113283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 23), 'p', False)
        # Obtaining the member 'item' of a type (line 550)
        item_113284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 23), p_113283, 'item')
        # Calling item(args, kwargs) (line 550)
        item_call_result_113288 = invoke(stypy.reporting.localization.Localization(__file__, 550, 23), item_113284, *[int_113285, int_113286], **kwargs_113287)
        
        # Applying the binary operator '*' (line 550)
        result_mul_113289 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 21), '*', int_113282, item_call_result_113288)
        
        
        # Call to item(...): (line 550)
        # Processing the call arguments (line 550)
        int_113292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 46), 'int')
        int_113293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 50), 'int')
        # Processing the call keyword arguments (line 550)
        kwargs_113294 = {}
        # Getting the type of 'p' (line 550)
        p_113290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 39), 'p', False)
        # Obtaining the member 'item' of a type (line 550)
        item_113291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 39), p_113290, 'item')
        # Calling item(args, kwargs) (line 550)
        item_call_result_113295 = invoke(stypy.reporting.localization.Localization(__file__, 550, 39), item_113291, *[int_113292, int_113293], **kwargs_113294)
        
        # Processing the call keyword arguments (line 550)
        str_113296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 63), 'str', 'n = 34')
        keyword_113297 = str_113296
        kwargs_113298 = {'err_msg': keyword_113297}
        # Getting the type of 'assert_equal' (line 550)
        assert_equal_113281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 550)
        assert_equal_call_result_113299 = invoke(stypy.reporting.localization.Localization(__file__, 550, 8), assert_equal_113281, *[result_mul_113289, item_call_result_113295], **kwargs_113298)
        
        
        # Assigning a Call to a Name (line 551):
        
        # Call to pascal(...): (line 551)
        # Processing the call arguments (line 551)
        int_113301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 19), 'int')
        # Processing the call keyword arguments (line 551)
        kwargs_113302 = {}
        # Getting the type of 'pascal' (line 551)
        pascal_113300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'pascal', False)
        # Calling pascal(args, kwargs) (line 551)
        pascal_call_result_113303 = invoke(stypy.reporting.localization.Localization(__file__, 551, 12), pascal_113300, *[int_113301], **kwargs_113302)
        
        # Assigning a type to the variable 'p' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'p', pascal_call_result_113303)
        
        # Call to assert_equal(...): (line 552)
        # Processing the call arguments (line 552)
        int_113305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 21), 'int')
        
        # Call to item(...): (line 552)
        # Processing the call arguments (line 552)
        int_113308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 30), 'int')
        int_113309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 34), 'int')
        # Processing the call keyword arguments (line 552)
        kwargs_113310 = {}
        # Getting the type of 'p' (line 552)
        p_113306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 23), 'p', False)
        # Obtaining the member 'item' of a type (line 552)
        item_113307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 23), p_113306, 'item')
        # Calling item(args, kwargs) (line 552)
        item_call_result_113311 = invoke(stypy.reporting.localization.Localization(__file__, 552, 23), item_113307, *[int_113308, int_113309], **kwargs_113310)
        
        # Applying the binary operator '*' (line 552)
        result_mul_113312 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 21), '*', int_113305, item_call_result_113311)
        
        
        # Call to item(...): (line 552)
        # Processing the call arguments (line 552)
        int_113315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 46), 'int')
        int_113316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 50), 'int')
        # Processing the call keyword arguments (line 552)
        kwargs_113317 = {}
        # Getting the type of 'p' (line 552)
        p_113313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 39), 'p', False)
        # Obtaining the member 'item' of a type (line 552)
        item_113314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 39), p_113313, 'item')
        # Calling item(args, kwargs) (line 552)
        item_call_result_113318 = invoke(stypy.reporting.localization.Localization(__file__, 552, 39), item_113314, *[int_113315, int_113316], **kwargs_113317)
        
        # Processing the call keyword arguments (line 552)
        str_113319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 63), 'str', 'n = 35')
        keyword_113320 = str_113319
        kwargs_113321 = {'err_msg': keyword_113320}
        # Getting the type of 'assert_equal' (line 552)
        assert_equal_113304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 552)
        assert_equal_call_result_113322 = invoke(stypy.reporting.localization.Localization(__file__, 552, 8), assert_equal_113304, *[result_mul_113312, item_call_result_113318], **kwargs_113321)
        
        
        # ################# End of 'test_threshold(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_threshold' in the type store
        # Getting the type of 'stypy_return_type' (line 544)
        stypy_return_type_113323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113323)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_threshold'
        return stypy_return_type_113323


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 504, 0, False)
        # Assigning a type to the variable 'self' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPascal.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestPascal' (line 504)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 0), 'TestPascal', TestPascal)

# Assigning a List to a Name (line 506):

# Obtaining an instance of the builtin type 'list' (line 506)
list_113324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 506)
# Adding element type (line 506)

# Obtaining an instance of the builtin type 'tuple' (line 507)
tuple_113325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 507)
# Adding element type (line 507)
int_113326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 9), tuple_113325, int_113326)
# Adding element type (line 507)

# Call to array(...): (line 507)
# Processing the call arguments (line 507)

# Obtaining an instance of the builtin type 'list' (line 507)
list_113328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 507)
# Adding element type (line 507)

# Obtaining an instance of the builtin type 'list' (line 507)
list_113329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 507)
# Adding element type (line 507)
int_113330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 19), list_113329, int_113330)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 18), list_113328, list_113329)

# Processing the call keyword arguments (line 507)
kwargs_113331 = {}
# Getting the type of 'array' (line 507)
array_113327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'array', False)
# Calling array(args, kwargs) (line 507)
array_call_result_113332 = invoke(stypy.reporting.localization.Localization(__file__, 507, 12), array_113327, *[list_113328], **kwargs_113331)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 9), tuple_113325, array_call_result_113332)
# Adding element type (line 507)

# Call to array(...): (line 507)
# Processing the call arguments (line 507)

# Obtaining an instance of the builtin type 'list' (line 507)
list_113334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 507)
# Adding element type (line 507)

# Obtaining an instance of the builtin type 'list' (line 507)
list_113335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 507)
# Adding element type (line 507)
int_113336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 33), list_113335, int_113336)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 32), list_113334, list_113335)

# Processing the call keyword arguments (line 507)
kwargs_113337 = {}
# Getting the type of 'array' (line 507)
array_113333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 26), 'array', False)
# Calling array(args, kwargs) (line 507)
array_call_result_113338 = invoke(stypy.reporting.localization.Localization(__file__, 507, 26), array_113333, *[list_113334], **kwargs_113337)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 9), tuple_113325, array_call_result_113338)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 12), list_113324, tuple_113325)
# Adding element type (line 506)

# Obtaining an instance of the builtin type 'tuple' (line 508)
tuple_113339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 508)
# Adding element type (line 508)
int_113340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 9), tuple_113339, int_113340)
# Adding element type (line 508)

# Call to array(...): (line 508)
# Processing the call arguments (line 508)

# Obtaining an instance of the builtin type 'list' (line 508)
list_113342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 508)
# Adding element type (line 508)

# Obtaining an instance of the builtin type 'list' (line 508)
list_113343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 508)
# Adding element type (line 508)
int_113344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 19), list_113343, int_113344)
# Adding element type (line 508)
int_113345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 19), list_113343, int_113345)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 18), list_113342, list_113343)
# Adding element type (line 508)

# Obtaining an instance of the builtin type 'list' (line 509)
list_113346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 509)
# Adding element type (line 509)
int_113347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 19), list_113346, int_113347)
# Adding element type (line 509)
int_113348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 19), list_113346, int_113348)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 18), list_113342, list_113346)

# Processing the call keyword arguments (line 508)
kwargs_113349 = {}
# Getting the type of 'array' (line 508)
array_113341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'array', False)
# Calling array(args, kwargs) (line 508)
array_call_result_113350 = invoke(stypy.reporting.localization.Localization(__file__, 508, 12), array_113341, *[list_113342], **kwargs_113349)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 9), tuple_113339, array_call_result_113350)
# Adding element type (line 508)

# Call to array(...): (line 510)
# Processing the call arguments (line 510)

# Obtaining an instance of the builtin type 'list' (line 510)
list_113352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 510)
# Adding element type (line 510)

# Obtaining an instance of the builtin type 'list' (line 510)
list_113353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 510)
# Adding element type (line 510)
int_113354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 19), list_113353, int_113354)
# Adding element type (line 510)
int_113355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 19), list_113353, int_113355)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 18), list_113352, list_113353)
# Adding element type (line 510)

# Obtaining an instance of the builtin type 'list' (line 511)
list_113356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 511)
# Adding element type (line 511)
int_113357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 19), list_113356, int_113357)
# Adding element type (line 511)
int_113358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 19), list_113356, int_113358)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 18), list_113352, list_113356)

# Processing the call keyword arguments (line 510)
kwargs_113359 = {}
# Getting the type of 'array' (line 510)
array_113351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'array', False)
# Calling array(args, kwargs) (line 510)
array_call_result_113360 = invoke(stypy.reporting.localization.Localization(__file__, 510, 12), array_113351, *[list_113352], **kwargs_113359)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 9), tuple_113339, array_call_result_113360)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 12), list_113324, tuple_113339)
# Adding element type (line 506)

# Obtaining an instance of the builtin type 'tuple' (line 512)
tuple_113361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 512)
# Adding element type (line 512)
int_113362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 9), tuple_113361, int_113362)
# Adding element type (line 512)

# Call to array(...): (line 512)
# Processing the call arguments (line 512)

# Obtaining an instance of the builtin type 'list' (line 512)
list_113364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 512)
# Adding element type (line 512)

# Obtaining an instance of the builtin type 'list' (line 512)
list_113365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 512)
# Adding element type (line 512)
int_113366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 19), list_113365, int_113366)
# Adding element type (line 512)
int_113367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 19), list_113365, int_113367)
# Adding element type (line 512)
int_113368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 19), list_113365, int_113368)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 18), list_113364, list_113365)
# Adding element type (line 512)

# Obtaining an instance of the builtin type 'list' (line 513)
list_113369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 513)
# Adding element type (line 513)
int_113370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 19), list_113369, int_113370)
# Adding element type (line 513)
int_113371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 19), list_113369, int_113371)
# Adding element type (line 513)
int_113372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 19), list_113369, int_113372)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 18), list_113364, list_113369)
# Adding element type (line 512)

# Obtaining an instance of the builtin type 'list' (line 514)
list_113373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 514)
# Adding element type (line 514)
int_113374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 19), list_113373, int_113374)
# Adding element type (line 514)
int_113375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 19), list_113373, int_113375)
# Adding element type (line 514)
int_113376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 19), list_113373, int_113376)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 18), list_113364, list_113373)

# Processing the call keyword arguments (line 512)
kwargs_113377 = {}
# Getting the type of 'array' (line 512)
array_113363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'array', False)
# Calling array(args, kwargs) (line 512)
array_call_result_113378 = invoke(stypy.reporting.localization.Localization(__file__, 512, 12), array_113363, *[list_113364], **kwargs_113377)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 9), tuple_113361, array_call_result_113378)
# Adding element type (line 512)

# Call to array(...): (line 515)
# Processing the call arguments (line 515)

# Obtaining an instance of the builtin type 'list' (line 515)
list_113380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 515)
# Adding element type (line 515)

# Obtaining an instance of the builtin type 'list' (line 515)
list_113381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 515)
# Adding element type (line 515)
int_113382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 19), list_113381, int_113382)
# Adding element type (line 515)
int_113383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 19), list_113381, int_113383)
# Adding element type (line 515)
int_113384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 19), list_113381, int_113384)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 18), list_113380, list_113381)
# Adding element type (line 515)

# Obtaining an instance of the builtin type 'list' (line 516)
list_113385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 516)
# Adding element type (line 516)
int_113386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 19), list_113385, int_113386)
# Adding element type (line 516)
int_113387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 19), list_113385, int_113387)
# Adding element type (line 516)
int_113388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 19), list_113385, int_113388)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 18), list_113380, list_113385)
# Adding element type (line 515)

# Obtaining an instance of the builtin type 'list' (line 517)
list_113389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 517)
# Adding element type (line 517)
int_113390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 19), list_113389, int_113390)
# Adding element type (line 517)
int_113391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 19), list_113389, int_113391)
# Adding element type (line 517)
int_113392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 19), list_113389, int_113392)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 18), list_113380, list_113389)

# Processing the call keyword arguments (line 515)
kwargs_113393 = {}
# Getting the type of 'array' (line 515)
array_113379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'array', False)
# Calling array(args, kwargs) (line 515)
array_call_result_113394 = invoke(stypy.reporting.localization.Localization(__file__, 515, 12), array_113379, *[list_113380], **kwargs_113393)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 9), tuple_113361, array_call_result_113394)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 12), list_113324, tuple_113361)
# Adding element type (line 506)

# Obtaining an instance of the builtin type 'tuple' (line 518)
tuple_113395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 518)
# Adding element type (line 518)
int_113396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 9), tuple_113395, int_113396)
# Adding element type (line 518)

# Call to array(...): (line 518)
# Processing the call arguments (line 518)

# Obtaining an instance of the builtin type 'list' (line 518)
list_113398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 518)
# Adding element type (line 518)

# Obtaining an instance of the builtin type 'list' (line 518)
list_113399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 518)
# Adding element type (line 518)
int_113400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 19), list_113399, int_113400)
# Adding element type (line 518)
int_113401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 19), list_113399, int_113401)
# Adding element type (line 518)
int_113402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 19), list_113399, int_113402)
# Adding element type (line 518)
int_113403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 19), list_113399, int_113403)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 18), list_113398, list_113399)
# Adding element type (line 518)

# Obtaining an instance of the builtin type 'list' (line 519)
list_113404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 519)
# Adding element type (line 519)
int_113405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 19), list_113404, int_113405)
# Adding element type (line 519)
int_113406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 19), list_113404, int_113406)
# Adding element type (line 519)
int_113407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 19), list_113404, int_113407)
# Adding element type (line 519)
int_113408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 19), list_113404, int_113408)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 18), list_113398, list_113404)
# Adding element type (line 518)

# Obtaining an instance of the builtin type 'list' (line 520)
list_113409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 520)
# Adding element type (line 520)
int_113410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 19), list_113409, int_113410)
# Adding element type (line 520)
int_113411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 19), list_113409, int_113411)
# Adding element type (line 520)
int_113412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 19), list_113409, int_113412)
# Adding element type (line 520)
int_113413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 19), list_113409, int_113413)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 18), list_113398, list_113409)
# Adding element type (line 518)

# Obtaining an instance of the builtin type 'list' (line 521)
list_113414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 521)
# Adding element type (line 521)
int_113415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 19), list_113414, int_113415)
# Adding element type (line 521)
int_113416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 19), list_113414, int_113416)
# Adding element type (line 521)
int_113417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 19), list_113414, int_113417)
# Adding element type (line 521)
int_113418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 19), list_113414, int_113418)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 18), list_113398, list_113414)

# Processing the call keyword arguments (line 518)
kwargs_113419 = {}
# Getting the type of 'array' (line 518)
array_113397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'array', False)
# Calling array(args, kwargs) (line 518)
array_call_result_113420 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), array_113397, *[list_113398], **kwargs_113419)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 9), tuple_113395, array_call_result_113420)
# Adding element type (line 518)

# Call to array(...): (line 522)
# Processing the call arguments (line 522)

# Obtaining an instance of the builtin type 'list' (line 522)
list_113422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 522)
# Adding element type (line 522)

# Obtaining an instance of the builtin type 'list' (line 522)
list_113423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 522)
# Adding element type (line 522)
int_113424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 19), list_113423, int_113424)
# Adding element type (line 522)
int_113425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 19), list_113423, int_113425)
# Adding element type (line 522)
int_113426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 19), list_113423, int_113426)
# Adding element type (line 522)
int_113427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 19), list_113423, int_113427)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 18), list_113422, list_113423)
# Adding element type (line 522)

# Obtaining an instance of the builtin type 'list' (line 523)
list_113428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 523)
# Adding element type (line 523)
int_113429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 19), list_113428, int_113429)
# Adding element type (line 523)
int_113430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 19), list_113428, int_113430)
# Adding element type (line 523)
int_113431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 19), list_113428, int_113431)
# Adding element type (line 523)
int_113432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 19), list_113428, int_113432)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 18), list_113422, list_113428)
# Adding element type (line 522)

# Obtaining an instance of the builtin type 'list' (line 524)
list_113433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 524)
# Adding element type (line 524)
int_113434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 19), list_113433, int_113434)
# Adding element type (line 524)
int_113435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 19), list_113433, int_113435)
# Adding element type (line 524)
int_113436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 19), list_113433, int_113436)
# Adding element type (line 524)
int_113437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 19), list_113433, int_113437)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 18), list_113422, list_113433)
# Adding element type (line 522)

# Obtaining an instance of the builtin type 'list' (line 525)
list_113438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 525)
# Adding element type (line 525)
int_113439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 19), list_113438, int_113439)
# Adding element type (line 525)
int_113440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 19), list_113438, int_113440)
# Adding element type (line 525)
int_113441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 19), list_113438, int_113441)
# Adding element type (line 525)
int_113442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 19), list_113438, int_113442)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 18), list_113422, list_113438)

# Processing the call keyword arguments (line 522)
kwargs_113443 = {}
# Getting the type of 'array' (line 522)
array_113421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'array', False)
# Calling array(args, kwargs) (line 522)
array_call_result_113444 = invoke(stypy.reporting.localization.Localization(__file__, 522, 12), array_113421, *[list_113422], **kwargs_113443)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 9), tuple_113395, array_call_result_113444)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 12), list_113324, tuple_113395)

# Getting the type of 'TestPascal'
TestPascal_113445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestPascal')
# Setting the type of the member 'cases' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestPascal_113445, 'cases', list_113324)

@norecursion
def test_invpascal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_invpascal'
    module_type_store = module_type_store.open_function_context('test_invpascal', 555, 0, False)
    
    # Passed parameters checking function
    test_invpascal.stypy_localization = localization
    test_invpascal.stypy_type_of_self = None
    test_invpascal.stypy_type_store = module_type_store
    test_invpascal.stypy_function_name = 'test_invpascal'
    test_invpascal.stypy_param_names_list = []
    test_invpascal.stypy_varargs_param_name = None
    test_invpascal.stypy_kwargs_param_name = None
    test_invpascal.stypy_call_defaults = defaults
    test_invpascal.stypy_call_varargs = varargs
    test_invpascal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_invpascal', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_invpascal', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_invpascal(...)' code ##################


    @norecursion
    def check_invpascal(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_invpascal'
        module_type_store = module_type_store.open_function_context('check_invpascal', 557, 4, False)
        
        # Passed parameters checking function
        check_invpascal.stypy_localization = localization
        check_invpascal.stypy_type_of_self = None
        check_invpascal.stypy_type_store = module_type_store
        check_invpascal.stypy_function_name = 'check_invpascal'
        check_invpascal.stypy_param_names_list = ['n', 'kind', 'exact']
        check_invpascal.stypy_varargs_param_name = None
        check_invpascal.stypy_kwargs_param_name = None
        check_invpascal.stypy_call_defaults = defaults
        check_invpascal.stypy_call_varargs = varargs
        check_invpascal.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check_invpascal', ['n', 'kind', 'exact'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_invpascal', localization, ['n', 'kind', 'exact'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_invpascal(...)' code ##################

        
        # Assigning a Call to a Name (line 558):
        
        # Call to invpascal(...): (line 558)
        # Processing the call arguments (line 558)
        # Getting the type of 'n' (line 558)
        n_113447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 23), 'n', False)
        # Processing the call keyword arguments (line 558)
        # Getting the type of 'kind' (line 558)
        kind_113448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 31), 'kind', False)
        keyword_113449 = kind_113448
        # Getting the type of 'exact' (line 558)
        exact_113450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 43), 'exact', False)
        keyword_113451 = exact_113450
        kwargs_113452 = {'kind': keyword_113449, 'exact': keyword_113451}
        # Getting the type of 'invpascal' (line 558)
        invpascal_113446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 13), 'invpascal', False)
        # Calling invpascal(args, kwargs) (line 558)
        invpascal_call_result_113453 = invoke(stypy.reporting.localization.Localization(__file__, 558, 13), invpascal_113446, *[n_113447], **kwargs_113452)
        
        # Assigning a type to the variable 'ip' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'ip', invpascal_call_result_113453)
        
        # Assigning a Call to a Name (line 559):
        
        # Call to pascal(...): (line 559)
        # Processing the call arguments (line 559)
        # Getting the type of 'n' (line 559)
        n_113455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 19), 'n', False)
        # Processing the call keyword arguments (line 559)
        # Getting the type of 'kind' (line 559)
        kind_113456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 27), 'kind', False)
        keyword_113457 = kind_113456
        # Getting the type of 'exact' (line 559)
        exact_113458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 39), 'exact', False)
        keyword_113459 = exact_113458
        kwargs_113460 = {'kind': keyword_113457, 'exact': keyword_113459}
        # Getting the type of 'pascal' (line 559)
        pascal_113454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'pascal', False)
        # Calling pascal(args, kwargs) (line 559)
        pascal_call_result_113461 = invoke(stypy.reporting.localization.Localization(__file__, 559, 12), pascal_113454, *[n_113455], **kwargs_113460)
        
        # Assigning a type to the variable 'p' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'p', pascal_call_result_113461)
        
        # Assigning a Call to a Name (line 566):
        
        # Call to dot(...): (line 566)
        # Processing the call arguments (line 566)
        
        # Call to astype(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'object' (line 566)
        object_113470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 43), 'object', False)
        # Processing the call keyword arguments (line 566)
        kwargs_113471 = {}
        # Getting the type of 'p' (line 566)
        p_113468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 34), 'p', False)
        # Obtaining the member 'astype' of a type (line 566)
        astype_113469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 34), p_113468, 'astype')
        # Calling astype(args, kwargs) (line 566)
        astype_call_result_113472 = invoke(stypy.reporting.localization.Localization(__file__, 566, 34), astype_113469, *[object_113470], **kwargs_113471)
        
        # Processing the call keyword arguments (line 566)
        kwargs_113473 = {}
        
        # Call to astype(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'object' (line 566)
        object_113464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 22), 'object', False)
        # Processing the call keyword arguments (line 566)
        kwargs_113465 = {}
        # Getting the type of 'ip' (line 566)
        ip_113462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'ip', False)
        # Obtaining the member 'astype' of a type (line 566)
        astype_113463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 12), ip_113462, 'astype')
        # Calling astype(args, kwargs) (line 566)
        astype_call_result_113466 = invoke(stypy.reporting.localization.Localization(__file__, 566, 12), astype_113463, *[object_113464], **kwargs_113465)
        
        # Obtaining the member 'dot' of a type (line 566)
        dot_113467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 12), astype_call_result_113466, 'dot')
        # Calling dot(args, kwargs) (line 566)
        dot_call_result_113474 = invoke(stypy.reporting.localization.Localization(__file__, 566, 12), dot_113467, *[astype_call_result_113472], **kwargs_113473)
        
        # Assigning a type to the variable 'e' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'e', dot_call_result_113474)
        
        # Call to assert_array_equal(...): (line 567)
        # Processing the call arguments (line 567)
        # Getting the type of 'e' (line 567)
        e_113476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 27), 'e', False)
        
        # Call to eye(...): (line 567)
        # Processing the call arguments (line 567)
        # Getting the type of 'n' (line 567)
        n_113478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 34), 'n', False)
        # Processing the call keyword arguments (line 567)
        kwargs_113479 = {}
        # Getting the type of 'eye' (line 567)
        eye_113477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 30), 'eye', False)
        # Calling eye(args, kwargs) (line 567)
        eye_call_result_113480 = invoke(stypy.reporting.localization.Localization(__file__, 567, 30), eye_113477, *[n_113478], **kwargs_113479)
        
        # Processing the call keyword arguments (line 567)
        str_113481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 46), 'str', 'n=%d  kind=%r exact=%r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 568)
        tuple_113482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 568)
        # Adding element type (line 568)
        # Getting the type of 'n' (line 568)
        n_113483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 47), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 47), tuple_113482, n_113483)
        # Adding element type (line 568)
        # Getting the type of 'kind' (line 568)
        kind_113484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 50), 'kind', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 47), tuple_113482, kind_113484)
        # Adding element type (line 568)
        # Getting the type of 'exact' (line 568)
        exact_113485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 56), 'exact', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 47), tuple_113482, exact_113485)
        
        # Applying the binary operator '%' (line 567)
        result_mod_113486 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 46), '%', str_113481, tuple_113482)
        
        keyword_113487 = result_mod_113486
        kwargs_113488 = {'err_msg': keyword_113487}
        # Getting the type of 'assert_array_equal' (line 567)
        assert_array_equal_113475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 567)
        assert_array_equal_call_result_113489 = invoke(stypy.reporting.localization.Localization(__file__, 567, 8), assert_array_equal_113475, *[e_113476, eye_call_result_113480], **kwargs_113488)
        
        
        # ################# End of 'check_invpascal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_invpascal' in the type store
        # Getting the type of 'stypy_return_type' (line 557)
        stypy_return_type_113490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113490)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_invpascal'
        return stypy_return_type_113490

    # Assigning a type to the variable 'check_invpascal' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'check_invpascal', check_invpascal)
    
    # Assigning a List to a Name (line 570):
    
    # Obtaining an instance of the builtin type 'list' (line 570)
    list_113491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 570)
    # Adding element type (line 570)
    str_113492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 13), 'str', 'symmetric')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 12), list_113491, str_113492)
    # Adding element type (line 570)
    str_113493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 26), 'str', 'lower')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 12), list_113491, str_113493)
    # Adding element type (line 570)
    str_113494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 35), 'str', 'upper')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 12), list_113491, str_113494)
    
    # Assigning a type to the variable 'kinds' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'kinds', list_113491)
    
    # Assigning a List to a Name (line 572):
    
    # Obtaining an instance of the builtin type 'list' (line 572)
    list_113495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 572)
    # Adding element type (line 572)
    int_113496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 9), list_113495, int_113496)
    # Adding element type (line 572)
    int_113497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 9), list_113495, int_113497)
    # Adding element type (line 572)
    int_113498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 9), list_113495, int_113498)
    # Adding element type (line 572)
    int_113499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 572, 9), list_113495, int_113499)
    
    # Assigning a type to the variable 'ns' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'ns', list_113495)
    
    # Getting the type of 'ns' (line 573)
    ns_113500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 13), 'ns')
    # Testing the type of a for loop iterable (line 573)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 573, 4), ns_113500)
    # Getting the type of the for loop variable (line 573)
    for_loop_var_113501 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 573, 4), ns_113500)
    # Assigning a type to the variable 'n' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'n', for_loop_var_113501)
    # SSA begins for a for statement (line 573)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'kinds' (line 574)
    kinds_113502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 20), 'kinds')
    # Testing the type of a for loop iterable (line 574)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 574, 8), kinds_113502)
    # Getting the type of the for loop variable (line 574)
    for_loop_var_113503 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 574, 8), kinds_113502)
    # Assigning a type to the variable 'kind' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'kind', for_loop_var_113503)
    # SSA begins for a for statement (line 574)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 575)
    list_113504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 575)
    # Adding element type (line 575)
    # Getting the type of 'True' (line 575)
    True_113505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 26), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 25), list_113504, True_113505)
    # Adding element type (line 575)
    # Getting the type of 'False' (line 575)
    False_113506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 32), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 25), list_113504, False_113506)
    
    # Testing the type of a for loop iterable (line 575)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 575, 12), list_113504)
    # Getting the type of the for loop variable (line 575)
    for_loop_var_113507 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 575, 12), list_113504)
    # Assigning a type to the variable 'exact' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'exact', for_loop_var_113507)
    # SSA begins for a for statement (line 575)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check_invpascal(...): (line 576)
    # Processing the call arguments (line 576)
    # Getting the type of 'n' (line 576)
    n_113509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 32), 'n', False)
    # Getting the type of 'kind' (line 576)
    kind_113510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 35), 'kind', False)
    # Getting the type of 'exact' (line 576)
    exact_113511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 41), 'exact', False)
    # Processing the call keyword arguments (line 576)
    kwargs_113512 = {}
    # Getting the type of 'check_invpascal' (line 576)
    check_invpascal_113508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'check_invpascal', False)
    # Calling check_invpascal(args, kwargs) (line 576)
    check_invpascal_call_result_113513 = invoke(stypy.reporting.localization.Localization(__file__, 576, 16), check_invpascal_113508, *[n_113509, kind_113510, exact_113511], **kwargs_113512)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 578):
    
    # Obtaining an instance of the builtin type 'list' (line 578)
    list_113514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 578)
    # Adding element type (line 578)
    int_113515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 9), list_113514, int_113515)
    # Adding element type (line 578)
    int_113516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 9), list_113514, int_113516)
    # Adding element type (line 578)
    int_113517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 9), list_113514, int_113517)
    # Adding element type (line 578)
    int_113518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 9), list_113514, int_113518)
    
    # Assigning a type to the variable 'ns' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'ns', list_113514)
    
    # Getting the type of 'ns' (line 579)
    ns_113519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 13), 'ns')
    # Testing the type of a for loop iterable (line 579)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 579, 4), ns_113519)
    # Getting the type of the for loop variable (line 579)
    for_loop_var_113520 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 579, 4), ns_113519)
    # Assigning a type to the variable 'n' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'n', for_loop_var_113520)
    # SSA begins for a for statement (line 579)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'kinds' (line 580)
    kinds_113521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 20), 'kinds')
    # Testing the type of a for loop iterable (line 580)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 580, 8), kinds_113521)
    # Getting the type of the for loop variable (line 580)
    for_loop_var_113522 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 580, 8), kinds_113521)
    # Assigning a type to the variable 'kind' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'kind', for_loop_var_113522)
    # SSA begins for a for statement (line 580)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check_invpascal(...): (line 581)
    # Processing the call arguments (line 581)
    # Getting the type of 'n' (line 581)
    n_113524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 28), 'n', False)
    # Getting the type of 'kind' (line 581)
    kind_113525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 31), 'kind', False)
    # Getting the type of 'True' (line 581)
    True_113526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 37), 'True', False)
    # Processing the call keyword arguments (line 581)
    kwargs_113527 = {}
    # Getting the type of 'check_invpascal' (line 581)
    check_invpascal_113523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 12), 'check_invpascal', False)
    # Calling check_invpascal(args, kwargs) (line 581)
    check_invpascal_call_result_113528 = invoke(stypy.reporting.localization.Localization(__file__, 581, 12), check_invpascal_113523, *[n_113524, kind_113525, True_113526], **kwargs_113527)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_invpascal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_invpascal' in the type store
    # Getting the type of 'stypy_return_type' (line 555)
    stypy_return_type_113529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_113529)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_invpascal'
    return stypy_return_type_113529

# Assigning a type to the variable 'test_invpascal' (line 555)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 0), 'test_invpascal', test_invpascal)

@norecursion
def test_dft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dft'
    module_type_store = module_type_store.open_function_context('test_dft', 584, 0, False)
    
    # Passed parameters checking function
    test_dft.stypy_localization = localization
    test_dft.stypy_type_of_self = None
    test_dft.stypy_type_store = module_type_store
    test_dft.stypy_function_name = 'test_dft'
    test_dft.stypy_param_names_list = []
    test_dft.stypy_varargs_param_name = None
    test_dft.stypy_kwargs_param_name = None
    test_dft.stypy_call_defaults = defaults
    test_dft.stypy_call_varargs = varargs
    test_dft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dft', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dft', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dft(...)' code ##################

    
    # Assigning a Call to a Name (line 585):
    
    # Call to dft(...): (line 585)
    # Processing the call arguments (line 585)
    int_113531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 12), 'int')
    # Processing the call keyword arguments (line 585)
    kwargs_113532 = {}
    # Getting the type of 'dft' (line 585)
    dft_113530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'dft', False)
    # Calling dft(args, kwargs) (line 585)
    dft_call_result_113533 = invoke(stypy.reporting.localization.Localization(__file__, 585, 8), dft_113530, *[int_113531], **kwargs_113532)
    
    # Assigning a type to the variable 'm' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'm', dft_call_result_113533)
    
    # Assigning a Call to a Name (line 586):
    
    # Call to array(...): (line 586)
    # Processing the call arguments (line 586)
    
    # Obtaining an instance of the builtin type 'list' (line 586)
    list_113535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 586)
    # Adding element type (line 586)
    
    # Obtaining an instance of the builtin type 'list' (line 586)
    list_113536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 586)
    # Adding element type (line 586)
    float_113537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 22), list_113536, float_113537)
    # Adding element type (line 586)
    float_113538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 22), list_113536, float_113538)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 21), list_113535, list_113536)
    # Adding element type (line 586)
    
    # Obtaining an instance of the builtin type 'list' (line 586)
    list_113539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 586)
    # Adding element type (line 586)
    float_113540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 34), list_113539, float_113540)
    # Adding element type (line 586)
    float_113541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 34), list_113539, float_113541)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 21), list_113535, list_113539)
    
    # Processing the call keyword arguments (line 586)
    kwargs_113542 = {}
    # Getting the type of 'array' (line 586)
    array_113534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 15), 'array', False)
    # Calling array(args, kwargs) (line 586)
    array_call_result_113543 = invoke(stypy.reporting.localization.Localization(__file__, 586, 15), array_113534, *[list_113535], **kwargs_113542)
    
    # Assigning a type to the variable 'expected' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'expected', array_call_result_113543)
    
    # Call to assert_array_almost_equal(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'm' (line 587)
    m_113545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'm', False)
    # Getting the type of 'expected' (line 587)
    expected_113546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 33), 'expected', False)
    # Processing the call keyword arguments (line 587)
    kwargs_113547 = {}
    # Getting the type of 'assert_array_almost_equal' (line 587)
    assert_array_almost_equal_113544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 587)
    assert_array_almost_equal_call_result_113548 = invoke(stypy.reporting.localization.Localization(__file__, 587, 4), assert_array_almost_equal_113544, *[m_113545, expected_113546], **kwargs_113547)
    
    
    # Assigning a Call to a Name (line 588):
    
    # Call to dft(...): (line 588)
    # Processing the call arguments (line 588)
    int_113550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 12), 'int')
    # Processing the call keyword arguments (line 588)
    str_113551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 21), 'str', 'n')
    keyword_113552 = str_113551
    kwargs_113553 = {'scale': keyword_113552}
    # Getting the type of 'dft' (line 588)
    dft_113549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'dft', False)
    # Calling dft(args, kwargs) (line 588)
    dft_call_result_113554 = invoke(stypy.reporting.localization.Localization(__file__, 588, 8), dft_113549, *[int_113550], **kwargs_113553)
    
    # Assigning a type to the variable 'm' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'm', dft_call_result_113554)
    
    # Call to assert_array_almost_equal(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'm' (line 589)
    m_113556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 30), 'm', False)
    # Getting the type of 'expected' (line 589)
    expected_113557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 33), 'expected', False)
    float_113558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 42), 'float')
    # Applying the binary operator 'div' (line 589)
    result_div_113559 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 33), 'div', expected_113557, float_113558)
    
    # Processing the call keyword arguments (line 589)
    kwargs_113560 = {}
    # Getting the type of 'assert_array_almost_equal' (line 589)
    assert_array_almost_equal_113555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 589)
    assert_array_almost_equal_call_result_113561 = invoke(stypy.reporting.localization.Localization(__file__, 589, 4), assert_array_almost_equal_113555, *[m_113556, result_div_113559], **kwargs_113560)
    
    
    # Assigning a Call to a Name (line 590):
    
    # Call to dft(...): (line 590)
    # Processing the call arguments (line 590)
    int_113563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 12), 'int')
    # Processing the call keyword arguments (line 590)
    str_113564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 21), 'str', 'sqrtn')
    keyword_113565 = str_113564
    kwargs_113566 = {'scale': keyword_113565}
    # Getting the type of 'dft' (line 590)
    dft_113562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'dft', False)
    # Calling dft(args, kwargs) (line 590)
    dft_call_result_113567 = invoke(stypy.reporting.localization.Localization(__file__, 590, 8), dft_113562, *[int_113563], **kwargs_113566)
    
    # Assigning a type to the variable 'm' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'm', dft_call_result_113567)
    
    # Call to assert_array_almost_equal(...): (line 591)
    # Processing the call arguments (line 591)
    # Getting the type of 'm' (line 591)
    m_113569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 30), 'm', False)
    # Getting the type of 'expected' (line 591)
    expected_113570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 33), 'expected', False)
    
    # Call to sqrt(...): (line 591)
    # Processing the call arguments (line 591)
    float_113572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 47), 'float')
    # Processing the call keyword arguments (line 591)
    kwargs_113573 = {}
    # Getting the type of 'sqrt' (line 591)
    sqrt_113571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 42), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 591)
    sqrt_call_result_113574 = invoke(stypy.reporting.localization.Localization(__file__, 591, 42), sqrt_113571, *[float_113572], **kwargs_113573)
    
    # Applying the binary operator 'div' (line 591)
    result_div_113575 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 33), 'div', expected_113570, sqrt_call_result_113574)
    
    # Processing the call keyword arguments (line 591)
    kwargs_113576 = {}
    # Getting the type of 'assert_array_almost_equal' (line 591)
    assert_array_almost_equal_113568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 591)
    assert_array_almost_equal_call_result_113577 = invoke(stypy.reporting.localization.Localization(__file__, 591, 4), assert_array_almost_equal_113568, *[m_113569, result_div_113575], **kwargs_113576)
    
    
    # Assigning a Call to a Name (line 593):
    
    # Call to array(...): (line 593)
    # Processing the call arguments (line 593)
    
    # Obtaining an instance of the builtin type 'list' (line 593)
    list_113579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 593)
    # Adding element type (line 593)
    int_113580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 14), list_113579, int_113580)
    # Adding element type (line 593)
    int_113581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 14), list_113579, int_113581)
    # Adding element type (line 593)
    int_113582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 14), list_113579, int_113582)
    # Adding element type (line 593)
    int_113583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 14), list_113579, int_113583)
    # Adding element type (line 593)
    int_113584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 14), list_113579, int_113584)
    # Adding element type (line 593)
    int_113585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 14), list_113579, int_113585)
    # Adding element type (line 593)
    int_113586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 14), list_113579, int_113586)
    # Adding element type (line 593)
    int_113587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 14), list_113579, int_113587)
    
    # Processing the call keyword arguments (line 593)
    kwargs_113588 = {}
    # Getting the type of 'array' (line 593)
    array_113578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'array', False)
    # Calling array(args, kwargs) (line 593)
    array_call_result_113589 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), array_113578, *[list_113579], **kwargs_113588)
    
    # Assigning a type to the variable 'x' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'x', array_call_result_113589)
    
    # Assigning a Call to a Name (line 594):
    
    # Call to dft(...): (line 594)
    # Processing the call arguments (line 594)
    int_113591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 12), 'int')
    # Processing the call keyword arguments (line 594)
    kwargs_113592 = {}
    # Getting the type of 'dft' (line 594)
    dft_113590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'dft', False)
    # Calling dft(args, kwargs) (line 594)
    dft_call_result_113593 = invoke(stypy.reporting.localization.Localization(__file__, 594, 8), dft_113590, *[int_113591], **kwargs_113592)
    
    # Assigning a type to the variable 'm' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'm', dft_call_result_113593)
    
    # Assigning a Call to a Name (line 595):
    
    # Call to dot(...): (line 595)
    # Processing the call arguments (line 595)
    # Getting the type of 'x' (line 595)
    x_113596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 15), 'x', False)
    # Processing the call keyword arguments (line 595)
    kwargs_113597 = {}
    # Getting the type of 'm' (line 595)
    m_113594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 9), 'm', False)
    # Obtaining the member 'dot' of a type (line 595)
    dot_113595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 9), m_113594, 'dot')
    # Calling dot(args, kwargs) (line 595)
    dot_call_result_113598 = invoke(stypy.reporting.localization.Localization(__file__, 595, 9), dot_113595, *[x_113596], **kwargs_113597)
    
    # Assigning a type to the variable 'mx' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'mx', dot_call_result_113598)
    
    # Assigning a Call to a Name (line 596):
    
    # Call to fft(...): (line 596)
    # Processing the call arguments (line 596)
    # Getting the type of 'x' (line 596)
    x_113601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 21), 'x', False)
    # Processing the call keyword arguments (line 596)
    kwargs_113602 = {}
    # Getting the type of 'fftpack' (line 596)
    fftpack_113599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 9), 'fftpack', False)
    # Obtaining the member 'fft' of a type (line 596)
    fft_113600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 9), fftpack_113599, 'fft')
    # Calling fft(args, kwargs) (line 596)
    fft_call_result_113603 = invoke(stypy.reporting.localization.Localization(__file__, 596, 9), fft_113600, *[x_113601], **kwargs_113602)
    
    # Assigning a type to the variable 'fx' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'fx', fft_call_result_113603)
    
    # Call to assert_array_almost_equal(...): (line 597)
    # Processing the call arguments (line 597)
    # Getting the type of 'mx' (line 597)
    mx_113605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 30), 'mx', False)
    # Getting the type of 'fx' (line 597)
    fx_113606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 34), 'fx', False)
    # Processing the call keyword arguments (line 597)
    kwargs_113607 = {}
    # Getting the type of 'assert_array_almost_equal' (line 597)
    assert_array_almost_equal_113604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 597)
    assert_array_almost_equal_call_result_113608 = invoke(stypy.reporting.localization.Localization(__file__, 597, 4), assert_array_almost_equal_113604, *[mx_113605, fx_113606], **kwargs_113607)
    
    
    # ################# End of 'test_dft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dft' in the type store
    # Getting the type of 'stypy_return_type' (line 584)
    stypy_return_type_113609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_113609)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dft'
    return stypy_return_type_113609

# Assigning a type to the variable 'test_dft' (line 584)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 0), 'test_dft', test_dft)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
